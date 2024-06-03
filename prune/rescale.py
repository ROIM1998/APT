import torch
from tqdm import tqdm
from typing import List, Tuple
from utils.fisher_utils.linalg import lsmr_cupy_solver
from trainer.model_arch import (
    get_layers,
    get_mha_proj,
    get_ffn2,
    hijack_input,
    get_encoder
)

@torch.no_grad()
def remove_padding(hidden_states, attention_mask):
    attention_mask = attention_mask.view(-1)
    nonzero = torch.nonzero(attention_mask).squeeze()
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[2])
    hidden_states = torch.index_select(hidden_states, dim=0, index=nonzero).contiguous()
    return hidden_states

def collect_module_inputs(model, dataloader, module_func):
    got_inputs = []
    handle = hijack_input(module_func(model), got_inputs)
    for inputs in dataloader:
        inputs = {k: torch.stack(v, dim=1).to(model.device) if isinstance(v, list) else v.to(model.device) for k, v in inputs.items()}
        model(**inputs)
    handle.remove()
    return got_inputs   

@torch.no_grad()
def collect_layer_inputs(
    model,
    head_mask,
    neuron_mask,
    layer_idx,
    prev_inputs,
):
    layers = get_layers(model)
    target_layer = layers[layer_idx]
    inputs = []
    if layer_idx == 0:
        model.head_mask = head_mask
        model.intermediate_mask = neuron_mask
        encoder = get_encoder(model)
        layers = encoder.layer
        encoder.layers = layers[:1]

        handle = hijack_input(target_layer, inputs)
        for batch in prev_inputs:
            for k, v in batch.items():
                batch[k] = v.to(model.device, non_blocking=True) if isinstance(v, torch.Tensor) else torch.stack(v, dim=1).to(model.device) if isinstance(v, list) else v
            model(**batch)

        handle.remove()
        encoder.layers = layers
        inputs = [list(x) for x in inputs]
    else:
        prev_layer = layers[layer_idx - 1]

        for batch in prev_inputs:
            prev_output = prev_layer(
                *batch,
                head_z=head_mask[layer_idx - 1].view(1, -1, 1, 1),
                intermediate_z=neuron_mask[layer_idx-1],
            )

            batch[0] = prev_output[0]
            inputs.append(batch)

    return inputs


@torch.no_grad()
def get_mha_lstsq(
    model,
    config,
    teacher_inputs,
    teacher_neuron_mask,
    student_inputs,
    student_head_mask,
    student_neuron_mask,
    layer_idx,
):
    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    attention_head_size = int(hidden_size / num_attention_heads)

    nonzero_heads = student_head_mask[layer_idx].nonzero().flatten()
    num_nonzero_heads = nonzero_heads.shape[0]

    layer = get_layers(model)[layer_idx]
    mha_proj = get_mha_proj(model, layer_idx, return_parent=True)
    weights_per_head = mha_proj.dense.weight.t().view(num_attention_heads, -1, hidden_size)
    weights_per_head = weights_per_head.index_select(dim=0, index=nonzero_heads)

    inputs = []
    handle = hijack_input(mha_proj, inputs)

    ATA = torch.zeros(num_nonzero_heads + 1, num_nonzero_heads + 1).cuda()
    ATB = torch.zeros(num_nonzero_heads + 1).cuda()

    model.eval()
    for teacher_batch, student_batch in zip(teacher_inputs, student_inputs):
        attention_mask = (teacher_batch[1] == 0)

        # Get the outputs of the teacher model
        layer(
            *teacher_batch,
            head_z=None,
            intermediate_z=teacher_neuron_mask[layer_idx]
        )
        # print("Length of collected inputs", len(inputs[0]))
        hidden_states, input_tensor = inputs.pop(0)
        teacher_output = mha_proj.dense(hidden_states) + input_tensor
        teacher_output = remove_padding(teacher_output, attention_mask)

        # Get the outputs of the student model
        layer(
            *student_batch,
            head_z=student_head_mask[layer_idx].view(1, -1, 1, 1),
            intermediate_z=student_neuron_mask[layer_idx],
        )
        hidden_states, input_tensor = inputs.pop(0)
        hidden_states = remove_padding(hidden_states, attention_mask)
        input_tensor = remove_padding(input_tensor, attention_mask)

        hidden_states = hidden_states.view(-1, num_attention_heads, attention_head_size)
        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_states = hidden_states.index_select(dim=0, index=nonzero_heads)

        outputs_per_head = hidden_states @ weights_per_head
        outputs_per_head = outputs_per_head.view(num_nonzero_heads, -1)

        A = outputs_per_head.t()
        A = torch.cat([A, torch.ones(A.shape[0], 1).cuda()], dim=1)
        B = teacher_output - mha_proj.dense.bias - input_tensor
        B = B.flatten()

        ATA += A.t() @ A
        ATB += A.t() @ B

    handle.remove()
    return ATA, ATB


@torch.no_grad()
def get_ffn_lstsq(
    model,
    config,
    teacher_inputs,
    teacher_neuron_mask,
    student_inputs,
    student_head_mask,
    student_neuron_mask,
    layer_idx,
    cls_only=False,
):
    layer = get_layers(model)[layer_idx]
    ffn2 = get_ffn2(model, layer_idx)
    weights_per_neuron = ffn2.dense.weight.t()

    nonzero_neurons = student_neuron_mask[layer_idx].nonzero().flatten()
    num_neurons = nonzero_neurons.shape[0]
    weights_per_neuron = weights_per_neuron.index_select(dim=0, index=nonzero_neurons)
    W = weights_per_neuron @ weights_per_neuron.t()

    inputs = []
    handle = hijack_input(ffn2, inputs)

    ATA = torch.zeros(num_neurons, num_neurons).cuda()
    ATB = torch.zeros(num_neurons).cuda()

    model.eval()
    for teacher_batch, student_batch in zip(teacher_inputs, student_inputs):
        attention_mask = (teacher_batch[1] == 0)

        # Get the outputs of the teacher model
        layer(
            *teacher_batch,
            head_z=None,
            intermediate_z=teacher_neuron_mask[layer_idx]
        )
        hidden_states, input_tensor = inputs.pop(0)
        teacher_output = ffn2.dense(hidden_states) + input_tensor
        if cls_only:
            teacher_output = teacher_output[:, 0, :]
        else:
            teacher_output = remove_padding(teacher_output, attention_mask)

        # Get the outputs of the student model
        layer(
            *student_batch,
            head_z=student_head_mask[layer_idx].view(1, -1, 1, 1),
            intermediate_z=student_neuron_mask[layer_idx],
        )
        hidden_states, input_tensor = inputs.pop(0)
        if cls_only:
            hidden_states = hidden_states[:, 0, :]
            input_tensor = input_tensor[:, 0, :]
        else:
            hidden_states = remove_padding(hidden_states, attention_mask)
            input_tensor = remove_padding(input_tensor, attention_mask)

        hidden_states = hidden_states.t()
        hidden_states = hidden_states.index_select(dim=0, index=nonzero_neurons)

        ATA += W * (hidden_states @ hidden_states.t())

        B = teacher_output - ffn2.dense.bias - input_tensor
        ATB += (hidden_states.unsqueeze(1) @ (weights_per_neuron @ B.t()).unsqueeze(2)).squeeze()

    handle.remove()
    return ATA, ATB

@torch.no_grad()
def rescale_mask(
    model,
    config,
    teacher_head_mask,
    teacher_neuron_mask,
    student_head_mask,
    student_neuron_mask,
    dataloader,
    classification_task=False,
):
    num_hidden_layers = config.num_hidden_layers
    rescaled_head_mask = student_head_mask.clone()
    rescaled_neuron_mask = student_neuron_mask.clone()

    for layer_idx in tqdm(range(num_hidden_layers)):
        teacher_inputs = collect_layer_inputs(
            model,
            teacher_head_mask,
            teacher_neuron_mask,
            layer_idx,
            prev_inputs=dataloader if layer_idx == 0 else teacher_inputs,
        )
        student_inputs = collect_layer_inputs(
            model,
            rescaled_head_mask,
            rescaled_neuron_mask,
            layer_idx,
            prev_inputs=dataloader if layer_idx == 0 else student_inputs,
        )

        if torch.count_nonzero(student_head_mask[layer_idx]) != 0 and layer_idx != 0:
            ATA, ATB = get_mha_lstsq(
                model,
                config,
                teacher_inputs,
                teacher_neuron_mask,
                student_inputs,
                rescaled_head_mask,
                rescaled_neuron_mask,
                layer_idx,
            )
            scale_factor, success = lsmr_cupy_solver(ATA, ATB)
            if not success:
                break
            scale_factor = scale_factor[:-1]
            if scale_factor.max() > 10 or scale_factor.min() < -10:
                break
            nonzero_heads = rescaled_head_mask[layer_idx].nonzero().flatten()
            for index, scale in zip(nonzero_heads, scale_factor):
                rescaled_head_mask[layer_idx][index] *= scale

        if torch.count_nonzero(student_neuron_mask[layer_idx]) != 0:
            cls_only = classification_task and (layer_idx == num_hidden_layers - 1)
            ATA, ATB = get_ffn_lstsq(
                model,
                config,
                teacher_inputs,
                teacher_neuron_mask,
                student_inputs,
                rescaled_head_mask,
                rescaled_neuron_mask,
                layer_idx,
                cls_only=cls_only,
            )
            scale_factor, success = lsmr_cupy_solver(ATA, ATB)
            if not success:
                break
            if scale_factor.max() > 10 or scale_factor.min() < -10:
                break
            nonzero_neurons = rescaled_neuron_mask[layer_idx].nonzero().flatten()
            for index, scale in zip(nonzero_neurons, scale_factor):
                rescaled_neuron_mask[layer_idx][index] *= scale

    return rescaled_head_mask, rescaled_neuron_mask



def get_l2_diff(x: torch.Tensor, y: torch.Tensor):
    return (x - y).pow(2).sum()


def get_output_l2_diff(xs: List[Tuple[torch.Tensor]], ys: List[Tuple[torch.Tensor]]):
    diff = 0
    for x, y in zip(xs, ys):
        x, y = x[0], y[0]
        diff += get_l2_diff(x, y)
    return diff

# trainer.set_prune_mask(model)
# trainer.evaluate()
# head_mask, intermediate_mask = model.head_mask.clone(), model.intermediate_mask.clone()
# dataloader = trainer.pruninig_dataloader
# rescaled_head_mask, rescaled_intermediate_mask = rescale_mask(
#     model,
#     model.config,
#     torch.ones(head_mask.shape).cuda(),
#     torch.ones(intermediate_mask.shape).cuda(),
#     head_mask,
#     intermediate_mask,
#     dataloader,
# )