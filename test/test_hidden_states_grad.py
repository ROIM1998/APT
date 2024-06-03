import os
import sys
import torch
import torch.nn as nn

from typing import Tuple
from transformers import (HfArgumentParser)
from torch.utils.data import Subset
from args import DataTrainingArguments
from models.model_args import ModelArguments
from args import MinusTrainingArguments
from utils.utils import *
from utils import build_trainer, build_dataloader
from models import build_model
from trainer.param_control import ParamController
from prune import build_scorer, build_pruner

MB = 1024 * 1024

def main():
    sys.argv = ['test_bert.py',
            '--output_dir',
            './output/test_magnitude_scorer/',
            '--model_name_or_path',
            'bert-base-uncased',
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
    model.head_mask = model.head_mask.to(training_args.device).view(-1)
    model.intermediate_mask = model.intermediate_mask.to(training_args.device).view(-1)
    model.hidden_mask = model.hidden_mask.to(training_args.device).view(-1)
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)
    model = model.to(training_args.device)
    pruning_batch_size = 32
    num_pruning_batches = 64
    dataloader = build_dataloader(Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]), pruning_batch_size, data_args, training_args, tokenizer)
    dataloader = build_dataloader(data_args, training_args, model, tokenizer, raw_datasets)

    # Also add ffn input layers to teacher config
    teacher_keys = ['query', 'value']
    teacher_config  = {
        k: [i for i in range(model.config.num_hidden_layers)] for k in teacher_keys
    }
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        student_config=None,
        lora_with_bias=False,
        param_allocation_strategy=training_args.param_allocation_strategy,
    )
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    param_controller.convert_to_pre_pruning_lora_teacher()
    param_controller.model_as_teacher()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    inputs = next(iter(dataloader))
    inputs = trainer._prepare_inputs(inputs)
    
    scorer = build_scorer('running_hidden_states_salience', model, None, param_controller = param_controller, state=trainer.state, gather_freq=1, beta_1=0.85, beta_2=0.85, use_uncertainty=False, block_normalize_dict=None)
    
    
    def calculate_mha_score(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]):
        print(torch.cuda.memory_allocated() / MB)
        mha_hidden_states = outputs[0]
        if mha_hidden_states is not None and mha_hidden_states.requires_grad:
            mha_hidden_states.retain_grad()
            module.output_states = mha_hidden_states.data
        input_states = layer_inputs[0]
        if input_states is not None and input_states.requires_grad:
            input_states.retain_grad()
            module.input_states = input_states.data
    
    def calculate_ffn_score(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: torch.Tensor):
        print(torch.cuda.memory_allocated() / MB)
        ffn_hidden_states = outputs
        if ffn_hidden_states is not None and ffn_hidden_states.requires_grad:
            ffn_hidden_states.retain_grad()
            module.output_states = ffn_hidden_states.data
        input_states = layer_inputs[0]
        if input_states is not None and input_states.requires_grad:
            input_states.retain_grad()
            module.input_states = input_states.data
        
    def model_output_hook(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: torch.Tensor):
        print(torch.cuda.memory_allocated() / MB)
        output_hidden_states = outputs[0]
        if output_hidden_states is not None and output_hidden_states.requires_grad:
            output_hidden_states.retain_grad()
            module.output_states = output_hidden_states.data
            
            
    for i in range(12):
        model.bert.encoder.layer[i].attention.self.register_forward_hook(calculate_mha_score)
        model.bert.encoder.layer[i].intermediate.register_forward_hook(calculate_ffn_score)
        model.bert.encoder.register_forward_hook(model_output_hook)
        
    # scorer = build_scorer('running_hidden_states_magnitude', model, None, param_controller = param_controller, state=trainer.state, gather_freq=1, beta_1=0.85, beta_2=0.85, use_uncertainty=False, block_normalize_dict=None)

    outputs = model(**inputs, return_dict=False)
    print(len(trainer.scorer.current_mha_states), len(trainer.scorer.current_ffn_states), len(trainer.scorer.current_hidden_states))
    print(len(trainer.scorer.all_head_scores), len(trainer.scorer.all_ffn_scores))
    outputs[0].backward()
    print(len(trainer.scorer.current_mha_states), len(trainer.scorer.current_ffn_states), len(trainer.scorer.current_hidden_states))
    print(len(trainer.scorer.all_head_scores), len(trainer.scorer.all_ffn_scores))
    trainer.scorer.step()
    print(len(trainer.scorer.current_mha_states), len(trainer.scorer.current_ffn_states), len(trainer.scorer.current_hidden_states))
    print(len(trainer.scorer.all_head_scores), len(trainer.scorer.all_ffn_scores))
    print(torch.cuda.memory_allocated() / MB)
    print(torch.cuda.max_memory_allocated() / MB)

    model.head_mask[[0,4,5,12,45]] = 0
    model.head_mask[-12:] = 0

if __name__ == '__main__':
    main()