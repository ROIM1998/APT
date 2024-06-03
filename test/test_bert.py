# Test AdaP BERT model with pruning consistency and tuning consistency

from post_analysis import compare_module_inputs_equality
import os
import sys
import torch

from transformers import (HfArgumentParser)
from args import DataTrainingArguments
from models.model_args import ModelArguments
from args import MinusTrainingArguments
from utils.utils import *
from utils import build_dataloader, build_trainer
from models import build_model
from prune.pruner import AdapterPruner
from torch.utils.data import Subset
from trainer.param_control import ParamController

def main():
    sys.argv = ['test_bert.py',
            '--output_dir',
            './output/test_bert/',
            '--model_name_or_path',
            'output/bert-base-uncased_lora_minus_mnli_once_global_none_nodistill_smalllr/mac0.4/epoch30/bz32/numprune5/paramq:0-11,v:0-11,i:0-11/lora_r8/lora_alpha16/pre_pruning_model',
            '--task_name',
            'mnli',
            '--do_train',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '32',
            '--per_device_eval_batch_size',
            '128',
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
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_head_mask.pt')).to(training_args.device)
    model.head_mask[-2] = 0
    model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_intermediate_mask.pt')).to(training_args.device)
    model.hidden_mask[:20] = 0
    model.hidden_mask = model.hidden_mask.to(training_args.device)
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)
    model = model.to(training_args.device)

    config, tokenizer, original_model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    original_model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_head_mask.pt')).to(training_args.device)
    original_model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_intermediate_mask.pt')).to(training_args.device)
    original_model.head_mask[-2] = 0
    
    original_model = original_model.to(training_args.device)
    model.prune_model_with_masks()
    
    # Also add ffn input layers to teacher config
    teacher_keys = ['query', 'value', 'intermediate']
    teacher_config  = {
        k: [i for i in range(model.config.num_hidden_layers)] for k in teacher_keys
    }

    pruning_batch_size = 4
    num_pruning_batches = 64
    dataloader = build_dataloader(Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]), pruning_batch_size, data_args, training_args, tokenizer)
    adapter_pruner = AdapterPruner(model, dataloader)
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        student_config=None,
        lora_with_bias=False,
        adapter_pruner=adapter_pruner,
        param_allocation_strategy=training_args.param_allocation_strategy,
    )
    original_param_controller = ParamController(
        original_model,
        teacher_config=teacher_config,
        student_config=None,
        lora_with_bias=False,
        adapter_pruner=adapter_pruner,
        param_allocation_strategy=training_args.param_allocation_strategy,
    )
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    original_trainer = build_trainer(data_args, training_args, original_model, tokenizer, train_dataset, eval_dataset, param_controller=original_param_controller)
    
    # First, compare the trainer's eval result
    result = trainer.evaluate()
    original_result = original_trainer.evaluate()
    
    # Then, compare the co-learning outputs with no mask outputs and pruned outputs
    inputs = next(iter(dataloader))
    inputs = trainer._prepare_inputs(inputs)
    model.eval()
    model = model.double()
    original_model.eval()
    original_model = original_model.double()
    with torch.no_grad():
        head_mask, intermediate_mask = original_model.head_mask, original_model.intermediate_mask
        original_model.head_mask, original_model.intermediate_mask = None, None
        nomask_outputs = original_model(
            **inputs,
            output_hidden_states=True,
            return_dict=False,
        )
        original_model.head_mask, original_model.intermediate_mask = head_mask, intermediate_mask
        masked_outputs = original_model(
            **inputs,
            output_hidden_states=True,
            return_dict=False,
        )
        pruned_outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=False,
        )
        combined_outputs = original_model(
            **inputs,
            output_hidden_states=True,
            return_dict=False,
            output_masked_states=True,
            use_cross_masked_states=True,
        )
        teacher_outputs = (combined_outputs[0], combined_outputs[2], combined_outputs[4])
        student_outputs = (combined_outputs[1], combined_outputs[3], combined_outputs[5])
    
    # Compare the masked outputs with pruned outputs
    masked_hidden_states = masked_outputs[-1]
    pruned_hidden_states = pruned_outputs[-1]
    for i, (masked_hidden_state, pruned_hidden_state) in enumerate(zip(masked_hidden_states, pruned_hidden_states)):
        print(i, torch.allclose(masked_hidden_state, pruned_hidden_state),(masked_hidden_state - pruned_hidden_state).abs().max())
        
    # Compare the combined teacher's outputs with the original model's outputs
    teacher_hidden_states = teacher_outputs[-1]
    original_hidden_states = nomask_outputs[-1]
    for i, (teacher_hidden_state, original_hidden_state) in enumerate(zip(teacher_hidden_states, original_hidden_states)):
        print(i, torch.allclose(teacher_hidden_state, original_hidden_state),(teacher_hidden_state - original_hidden_state).abs().max())
        
    # Compare the combined student's outputs with the pruned model's outputs
    student_hidden_states = student_outputs[-1]
    for i, (student_hidden_state, pruned_hidden_state) in enumerate(zip(student_hidden_states, pruned_hidden_states[1:])):
        print(i, torch.allclose(student_hidden_state, pruned_hidden_state),(student_hidden_state - pruned_hidden_state).abs().max())
    
if __name__ == '__main__':
    main()