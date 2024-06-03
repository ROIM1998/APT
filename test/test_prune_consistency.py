from post_analysis import compare_module_inputs_equality
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
from transformers import HfArgumentParser
from args import DataTrainingArguments
from models import build_model
from models.model_args import ModelArguments
from utils.utils import *
from utils import build_trainer, build_dataloader
from args import MinusTrainingArguments
from trainer.model_arch import get_layers

def main():
    sys.argv = ['neuron_importance.py',
            '--output_dir',
            './output/neuron_importance/',
            '--model_name_or_path',
            'output/debug_output/pruning_5_model',
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
            '--apply_lora',
            '--do_distill',
            '--lora_r',
            '8',
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
    os.makedirs(training_args.output_dir, exist_ok=True)
    # training_args.disable_tqdm = False
    t_name, raw_datasets = get_raw_datasets(model_args, data_args, training_args)
    config, tokenizer, pruned_model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, pruned_model, tokenizer, config, raw_datasets)
    mask_schedule = torch.load(os.path.join('output/debug_output/masks_schedule.pt'))
    
    for i, masks in enumerate(mask_schedule):
        print("%d-th pruning:" % i, "Head density:", sum([m.sum().item() for m in masks['head_mask']]) / sum([m.numel() for m in masks['head_mask']]), "Intermediate density:", sum([m.sum().item() for m in masks['intermediate_mask']]) / sum([m.numel() for m in masks['intermediate_mask']]))
        
    head_mask, intermediate_mask = mask_schedule[-1]['head_mask'], mask_schedule[-1]['intermediate_mask']

    pruned_model.head_mask, pruned_model.intermediate_mask = None, None
    pruned_trainer = build_trainer(data_args, training_args, pruned_model, tokenizer, train_dataset, eval_dataset)
    unpruned_result = pruned_trainer.evaluate()
    
    pruned_model.head_mask, pruned_model.intermediate_mask = [v.to(training_args.device) for v in head_mask], [v.to(training_args.device) for v in intermediate_mask]
    masked_result = pruned_trainer.evaluate()

    pruned_model.prune_model_with_masks()
    pruned_result = pruned_trainer.evaluate()
    
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    if isinstance(head_mask, torch.Tensor) and isinstance(intermediate_mask, torch.Tensor):
        model.head_mask, model.intermediate_mask = head_mask.to(training_args.device), intermediate_mask.to(training_args.device)
    else:
        model.head_mask, model.intermediate_mask = [v.to(training_args.device) for v in head_mask], [v.to(training_args.device) for v in intermediate_mask]
    
    pruned_result = pruned_trainer.evaluate()

    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset)
    result = trainer.evaluate()

    dataloader = build_dataloader(train_dataset, training_args.per_device_train_batch_size, data_args, training_args, tokenizer)
    inputs = next(iter(dataloader))
    
    for i in range(model.config.num_hidden_layers):
        layer_func = lambda x: get_layers(x)[i]
        result = compare_module_inputs_equality([model, pruned_model], inputs, layer_func)
        print("Layer %d:" % i, "the biggest difference is", (result[0] - result[1]).abs().max().item())

    for i in range(model.config.num_hidden_layers):
        layer_func = lambda x: get_layers(x)[i].output
        result = compare_module_inputs_equality([model, pruned_model], inputs, layer_func)
        print("Layer intermediate outputs %d:" % i, "the biggest difference is", (result[0][:, :, (intermediate_mask[i]==1).nonzero().squeeze()] - result[1]).abs().max().item())

    for i in range(model.config.num_hidden_layers):
        layer_dense_func = lambda x: get_layers(x)[i].attention
        layer_dropout_func = lambda x: get_layers(x)[i].attention.output
        layer_attndropout_func = lambda x: get_layers(x)[i].attention.self.dropout
        layer_func = lambda x: get_layers(x)[i].intermediate
        dense_result = compare_module_inputs_equality([model, pruned_model], inputs, layer_dense_func)
        print("Attention inputs %d:" % i, "the biggest difference is", (dense_result[0] - dense_result[1]).abs().max().item())
        try:
            attndropout_result = compare_module_inputs_equality([model, pruned_model], inputs, layer_attndropout_func)
            print("Attention query-key outputs %d:" % i, "the biggest difference is", (attndropout_result[0][:, head_mask[i].nonzero().squeeze(), :, :] - attndropout_result[1]).abs().max().item())
        except:
            pass
        try:
            dropout_result = compare_module_inputs_equality([model, pruned_model], inputs, layer_dropout_func)
            print("Attention outputs %d:" % i, "the biggest difference is", (dropout_result[0][:, :, head_mask[i].repeat_interleave(64).nonzero().squeeze()] - dropout_result[1]).abs().max().item())
        except:
            pass
        result = compare_module_inputs_equality([model, pruned_model], inputs, layer_func)
        print("Intermediate inputs %d:" % i, "the biggest difference is", (result[0] - result[1]).abs().max().item())


    for i in range(model.config.num_hidden_layers):
        layer_dense_func = lambda x: get_layers(x)[i].output.dense
        layer_dropout_func = lambda x: get_layers(x)[i].output.dropout
        layer_func = lambda x: get_layers(x)[i].output.LayerNorm
        dense_result = compare_module_inputs_equality([model, pruned_model], inputs, layer_dense_func)
        dropout_result = compare_module_inputs_equality([model, pruned_model], inputs, layer_dropout_func)
        result = compare_module_inputs_equality([model, pruned_model], inputs, layer_func)
        print("Output dense inputs %d:" % i, "the biggest difference is", (dense_result[0][:, :, (intermediate_mask[i]==1).nonzero().squeeze()] - dense_result[1]).abs().max().item())
        print("Output dense outputs %d:" % i, "the biggest difference is", (dropout_result[0] - dropout_result[1]).abs().max().item())
        print("Output layernorm inputs %d:" % i, "the biggest difference is", (result[0] - result[1]).abs().max().item())



    # Testing intermediate matrix consistency
    for i in range(model.config.num_hidden_layers):
        lora_a = lambda x: get_layers(x)[i].intermediate.dense.lora_A
        lora_b = lambda x: get_layers(x)[i].intermediate.dense.lora_B
        weight = lambda x: get_layers(x)[i].intermediate.dense.weight
        bias = lambda x: get_layers(x)[i].intermediate.dense.bias
        print("Layer %d:" % i, "the biggest difference is", (lora_a(model) - lora_a(pruned_model)).abs().max().item())
        print("Layer %d:" % i, "the biggest difference is", (lora_b(model)[(intermediate_mask[i] == 1).nonzero().squeeze()] - lora_b(pruned_model)).abs().max().item())
        print("Layer %d:" % i, "the biggest difference is", (weight(model)[(intermediate_mask[i] == 1).nonzero().squeeze()] - weight(pruned_model)).abs().max().item())
        print("Layer %d:" % i, "the biggest difference is", (bias(model)[(intermediate_mask[i] == 1).nonzero().squeeze()] - bias(pruned_model)).abs().max().item())
    
    module_func = lambda x: get_layers(x)[0].output.dense
    result = compare_module_inputs_equality([model, pruned_model], inputs, module_func)
    hidden_states = module_func(model)(result[0])
    new_hidden_states = module_func(pruned_model)(result[1])
    print("Pre dense-layer hidden_states equality:",
        (result[1][:,:, pruned_model.intermediate_mask[0].nonzero().T.squeeze()] == result[0]).all().item()
    )
    print("Post dense-layer hidden states equality:", (hidden_states == new_hidden_states).all().item())
    print("Post dense-layer hidden states equality (with float64):", (module_func(model).double()(result[0].double()) == module_func(pruned_model).double()(result[1].double())).all().item())
    
    original_weights = get_layers(model)[0].output.dense.weight
    selected_weights = get_layers(pruned_model)[0].output.dense.weight[:, pruned_model.intermediate_mask[0].nonzero().T.squeeze()]
    post_dense_states = result[0].matmul(original_weights.T) + get_layers(model)[0].output.dense.bias
    other_post_dense_states = result[0].matmul(selected_weights.T) + get_layers(model)[0].output.dense.bias
    print("Layer forward equal to pruned-weight-matmul:", ((post_dense_states == hidden_states).all().item()))
    print("Layer forward equal to selected-weight-matmul:", ((other_post_dense_states == hidden_states).all().item()))
    
    original_post_dense_states = result[1].matmul(get_layers(pruned_model)[0].output.dense.weight.T) + get_layers(pruned_model)[0].output.dense.bias
    select_states_original_layer = result[1][:,:, pruned_model.intermediate_mask[0].nonzero().T.squeeze()].matmul(get_layers(pruned_model)[0].output.dense.weight[:, pruned_model.intermediate_mask[0].nonzero().T.squeeze()].T) + get_layers(pruned_model)[0].output.dense.bias
    zeroed_layer_weight = get_layers(pruned_model)[0].output.dense.weight.clone()
    zeroed_layer_weight[:,(pruned_model.intermediate_mask[0] == 0).nonzero().T.squeeze()] = 0
    new_output_states = result[1].matmul(zeroed_layer_weight.T) + get_layers(pruned_model)[0].output.dense.bias
    

    
if __name__ == '__main__':
    main()