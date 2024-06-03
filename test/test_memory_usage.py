import torch
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import loralib as lora


from transformers import HfArgumentParser
from args import DataTrainingArguments
from models import build_model
from models.model_args import ModelArguments
from trainer.trainer_minus import ParamController
from utils import build_dataloader, build_trainer
from prune.fisher import collect_additive_mask_grads, collect_mask_grads
from utils.utils import *
from args import MinusTrainingArguments
from torch.utils.data import DataLoader, Subset
from utils.minus_utils import lora_to_prunelora, lora_to_linear
from transformers.trainer import nested_detach

MB = 1024 * 1024

def theoretical_calculation(model, batch_size, seq_len):
    vocab_len = model.config.vocab_size
    d_model = model.config.hidden_size
    d_ff = model.config.intermediate_size
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_param_per_mha = (d_model * d_model + d_model) * 4
    num_param_per_ff = (d_model * d_ff + d_ff) * 2
    num_param_per_layer = num_param_per_mha + num_param_per_ff
    num_param_encoder = num_param_per_layer * num_layers
    num_emb_param = vocab_len * d_model
    num_param = num_param_encoder + num_emb_param
    forward_activation_size = batch_size * seq_len * d_model

def main():
    sys.argv = ['neuron_importance.py',
            '--output_dir',
            './output/neuron_importance/',
            '--model_name_or_path',
            'output/roberta-base_lora_minus_mnli_once_global_free_inout_nodistill/mac0.6/epoch30/bz128/numprune5/paramq:0-11,v:0-11,i:0-11/lora_r8/lora_alpha16/pre_pruning_model',
            '--task_name',
            'mnli',
            '--do_train',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '32',
            '--per_device_eval_batch_size',
            '32',
            '--lora_r',
            '64',
            '--apply_lora',
            '--do_distill',
            '--report_to',
            'none',
            ]
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MinusTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    t_name, raw_datasets = get_raw_datasets(model_args, data_args, training_args)
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    head_mask, intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_head_mask.pt')), torch.load(os.path.join(model_args.model_name_or_path, '../final_intermediate_mask.pt'))
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset)
    
    # print(trainer.evaluate())
    pruning_batch_size = 32
    num_pruning_batches = 64
    dataloader = build_dataloader(Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]), pruning_batch_size, data_args, training_args, tokenizer)
    # Original model memory usage test
    model = model.to(training_args.device)
    model.train()
    inputs = next(iter(dataloader))
    inputs = {k: v.to(training_args.device) for k, v in inputs.items()}
    
    # Converted model memory usage test
    teacher_config  = ParamController.parse_tuning_param_str(training_args.teacher_param_tuning_config)
    student_config = ParamController.parse_tuning_param_str(training_args.student_param_tuning_config)
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        student_config=student_config,
        lora_with_bias=False,
    )
    model.head_mask, model.intermediate_mask = torch.ones(model.config.num_hidden_layers, model.config.num_attention_heads).to(training_args.device), torch.ones(model.config.num_hidden_layers, model.config.intermediate_size).to(training_args.device)
    # param_controller.convert_to_distill(head_mask, intermediate_mask)
    
    # Test mask grads only
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start_mem = torch.cuda.max_memory_allocated() / MB
    for p in model.parameters():
        p.requires_grad = False
    model.head_mask.requires_grad = True
    model.intermediate_mask.requires_grad = True
    outputs = model(**inputs)
    post_input_mem = torch.cuda.max_memory_allocated() / MB
    loss = outputs.loss
    loss.backward()
    end_mem = torch.cuda.max_memory_allocated() / MB
    print('Pre-mask grads only: ', start_mem)
    print('Post-mask grads only: ', post_input_mem)
    print('Backward grads only: ', end_mem)

    # Test LoRA grads only
    for n, p in model.named_parameters():
        if 'lora' in n and 'intermediate' not in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
    model.head_mask.requires_grad = False
    model.intermediate_mask.requires_grad = False
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start_mem = torch.cuda.max_memory_allocated() / MB
    outputs = model(**inputs)
    post_input_mem = torch.cuda.max_memory_allocated() / MB
    loss = outputs.loss
    loss.backward()
    end_mem = torch.cuda.max_memory_allocated() / MB
    print('Pre-mask grads only: ', start_mem)
    print('Post-mask grads only: ', post_input_mem)
    print('Backward grads only: ', end_mem)
    
    # Test traditional distillation memory usage
    trainer.distill_mapping_strategy = 'dynamic_block_teacher_dynamic_student'
    config, tokenizer, teacher_model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    teacher_model = teacher_model.to(training_args.device)
    teacher_model.head_mask = None
    teacher_model.intermediate_mask = None
    teacher_model.eval()
    model.head_mask, model.intermediate_mask = head_mask.to(training_args.device),  intermediate_mask.to(training_args.device)
    model.prune_model_with_masks()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start_mem = torch.cuda.max_memory_allocated() / MB
    with torch.no_grad():
        teacher_outputs = teacher_model(
            **inputs,
            output_hidden_states=True,
            return_dict=False,
        )
    student_outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=False,
    )
    post_input_mem = torch.cuda.max_memory_allocated() / MB
    zs = {
        'intermediate_z': model.intermediate_mask,
        'head_z': model.head_mask,
    } #! extract the zs
    distill_loss, distill_ce_loss, loss = trainer.calculate_distillation_loss(
        teacher_outputs,
        student_outputs,
        zs,
    )
    loss.backward()
    end_mem = torch.cuda.max_memory_allocated() / MB
    
    # Test self-distillation with using mask and without using mask
    trainer.distill_mapping_strategy = 'dynamic_block_teacher_dynamic_student'
    model.head_mask, model.intermediate_mask = head_mask.to(training_args.device),  intermediate_mask.to(training_args.device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start_mem = torch.cuda.max_memory_allocated() / MB
    model.eval()
    with torch.no_grad():
        teacher_outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=False,
            pass_mask=False
        )
    model.train()
    student_outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=False,
    )
    post_input_mem = torch.cuda.max_memory_allocated() / MB
    zs = {
        'intermediate_z': model.intermediate_mask,
        'head_z': model.head_mask,
    } #! extract the zs
    distill_loss, distill_ce_loss, loss = trainer.calculate_distillation_loss(
        teacher_outputs,
        student_outputs,
        zs,
    )
    loss.backward()
    end_mem = torch.cuda.max_memory_allocated() / MB

    # Test the memory usage during fisher information calculation
    model.eval()
    for n, p in model.named_parameters():
        p.requires_grad = False
    named_modules = dict(model.named_modules())
    for n, p in model.named_modules():
        if isinstance(p, lora.Linear) and '.' in n and 'intermediate' not in n:
            parent_layer_attr, attr = n.rsplit('.', 1)
            parent_layer = named_modules[parent_layer_attr]
            if 'intermediate' in n:
                setattr(parent_layer, attr, lora_to_linear(p))
            else:
                new_layer = lora_to_prunelora(p, r=8, lora_alpha=16)
                new_layer.set_grafting_mask()
                setattr(parent_layer, attr, new_layer)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.max_memory_allocated() / MB
    all_masks = collect_additive_mask_grads(model, dataloader)
    head_grads, intermediate_grads = collect_mask_grads(model, dataloader)
    end_mem = torch.cuda.max_memory_allocated() / MB
    print('Pre-mask grads only: ', start_mem)
    print('Backward grads only: ', end_mem)
    
    # Test in-layer distillation memory usage
    trainer.distill_mapping_strategy = 'dynamic_block_teacher_dynamic_student'
    model.head_mask, model.intermediate_mask = head_mask.to(training_args.device), intermediate_mask.to(training_args.device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start_mem = torch.cuda.max_memory_allocated() / MB
    combined_outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=False,
        output_masked_states=True,
    )
    post_input_mem = torch.cuda.max_memory_allocated() / MB
    teacher_outputs = combined_outputs[:3]
    student_outputs = (None, None, combined_outputs[3])
    teacher_loss = teacher_outputs[0]
    teacher_outputs = nested_detach(teacher_outputs)
    zs = {
        'intermediate_z': model.intermediate_mask,
        'head_z': model.head_mask,
    } #! extract the zs
    distill_loss, distill_ce_loss, loss = trainer.calculate_distillation_loss(
        teacher_outputs,
        student_outputs,
        zs,
    )
    loss = loss * 0.5 + teacher_loss * 0.5
    loss.backward()
    end_mem = torch.cuda.max_memory_allocated() / MB
    
    # Test co-learning memory usage
    trainer.distill_mapping_strategy = 'dynamic_block_teacher_dynamic_student'
    model.head_mask, model.intermediate_mask = head_mask.to(training_args.device), intermediate_mask.to(training_args.device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start_mem = torch.cuda.max_memory_allocated() / MB
    combined_outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=False,
        output_masked_states=True,
        use_cross_masked_states=True,
    )
    post_input_mem = torch.cuda.max_memory_allocated() / MB
    teacher_outputs = (combined_outputs[0], combined_outputs[2], combined_outputs[4])
    student_outputs = (combined_outputs[1], combined_outputs[3], combined_outputs[5])
    teacher_loss = teacher_outputs[0]
    teacher_outputs = nested_detach(teacher_outputs)
    zs = {
        'intermediate_z': model.intermediate_mask,
        'head_z': model.head_mask,
    } #! extract the zs
    distill_loss, distill_ce_loss, loss = trainer.calculate_distillation_loss(
        teacher_outputs,
        student_outputs,
        zs,
    )
    loss = loss * 0.5 + teacher_loss * 0.5
    loss.backward()
    end_mem = torch.cuda.max_memory_allocated() / MB
    
    # Test co-learning memory usage
    # Test the correlation between tunable teacher layers and memory usage
    results = []
    for i in range(12):
        param_controller.clear_states()
        param_controller.model_teacher_with_student()
        model.head_mask, model.intermediate_mask = head_mask, intermediate_mask
        if isinstance(model.head_mask, torch.Tensor):
            model.head_mask.requires_grad = False
            model.intermediate_mask.requires_grad = False
        for j in range(i):
            for n, p in model.roberta.encoder.layer[j].named_parameters():
                if 'teacher' in n or 'intermediate' in n:
                    p.requires_grad = False
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        start_mem = torch.cuda.max_memory_allocated() / MB
        combined_outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=False,
            output_masked_states=True,
            use_cross_masked_states=True,
        )
        post_input_mem = torch.cuda.max_memory_allocated() / MB
        teacher_outputs = (combined_outputs[0], combined_outputs[2], combined_outputs[4])
        student_outputs = (combined_outputs[1], combined_outputs[3], combined_outputs[5])
        teacher_loss = teacher_outputs[0]
        teacher_outputs = nested_detach(teacher_outputs)
        zs = {
            'intermediate_z': model.intermediate_mask,
            'head_z': model.head_mask,
        } #! extract the zs
        distill_loss, distill_ce_loss, loss = trainer.calculate_distillation_loss(
            teacher_outputs,
            student_outputs,
            zs,
        )
        loss = loss * 0.5 + teacher_loss * 0.5
        loss.backward()
        end_mem = torch.cuda.max_memory_allocated() / MB
        # print('Pre-mask grads only: ', start_mem)
        # print('Post-mask grads only: ', post_input_mem)
        # print('Backward grads only: ', end_mem)
        results.append({
            'start_mem': start_mem,
            'post_input_mem': post_input_mem,
            'end_mem': end_mem,
        })