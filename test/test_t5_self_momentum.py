import os
import sys
import torch
import torch.nn as nn
import loralib as lora

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
MB = 1024 * 1024

def main():
    sys.argv = ['test_bert.py',
            '--output_dir',
            './output/test_magnitude_scorer/',
            '--model_name_or_path',
            'output/google/t5-xl-lm-adapt_lora_minus_rte_cubic_gradual_running_fisher_alloc_running_fisher_momentum_mapping_static_teacher_dynamic_cofi_student_distill_tophalf_limited/mac0.4/epoch120/bz16/numprune5/parameq:0-23,ev:0-23,dq:0-23,dv:0-23,cq:0-23,cv:0-23,ei0:0-23,di0:0-23/lora_r8/pruning_start-1/distill_epoch96/best_model',
            '--task_name',
            'rte',
            '--do_train',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '16',
            '--per_device_eval_batch_size',
            '16',
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
    # training_args.disable_tqdm = False
    t_name, raw_datasets = get_raw_datasets(model_args, data_args, training_args)
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets, True)
    
    model.head_mask = model.head_mask.to(training_args.device).view(-1)
    model.intermediate_mask = model.intermediate_mask.to(training_args.device).view(-1)
    model.hidden_mask = model.hidden_mask.to(training_args.device).view(-1)
    model = model.to(training_args.device)
    
    # pretrain_path = 'output/google/t5-xl-lm-adapt_lora_minus_rte_cubic_gradual_running_fisher_alloc_running_fisher_momentum_mapping_static_teacher_dynamic_cofi_student_distill_tophalf_limited/mac0.4/epoch120/bz16/numprune5/parameq:0-23,ev:0-23,dq:0-23,dv:0-23,cq:0-23,cv:0-23,ei0:0-23,di0:0-23/lora_r8/pruning_start-1/distill_epoch96/best_model'
    # model.head_mask = torch.load(os.path.join(pretrain_path, '../final_head_mask.pt'))
    # model.intermediate_mask = torch.load(os.path.join(pretrain_path, '../final_intermediate_mask.pt'))
    # model.hidden_mask = torch.load(os.path.join(pretrain_path, '../final_hidden_mask.pt'))
    # model.prune_model_with_masks()
    # model.load_state_dict(torch.load(os.path.join(pretrain_path, 'pytorch_model.bin')))
    
    pruning_batch_size = 16
    num_pruning_batches = 64
    dataloader = build_dataloader(Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]), pruning_batch_size, data_args, training_args, tokenizer)

    # Also add ffn input layers to teacher config
    warmup_keys = ['enc_self_query', 'enc_self_value', 'dec_self_query', 'dec_self_value', 'cross_query', 'cross_value']
    warmup_config  = {
        k: [i for i in range(model.config.num_hidden_layers)] for k in warmup_keys
    }
    teacher_keys = ['enc_self_query', 'enc_self_value', 'dec_self_query', 'dec_self_value', 'cross_query', 'cross_value', 'encoder_i0', 'decoder_i0']
    teacher_config  = {
        k: [i for i in range(model.config.num_hidden_layers)] for k in teacher_keys
    }
    param_controller = ParamController(
        model,
        warmup_config=warmup_config,
        teacher_config=teacher_config,
        student_config=None,
        lora_with_bias=False,
        param_allocation_strategy=training_args.param_allocation_strategy,
    )
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    
    param_controller.convert_to_self_momentum_distill()
    for m in model.modules():
        if isinstance(m, lora.Linear):
            m.scaling = 2.0
    trainer.evaluate()
    
    inputs = next(iter(dataloader))
    inputs = trainer._prepare_inputs(inputs)
    # labels = inputs.pop("labels")
    outputs = model(**inputs)
    output_tokens = outputs[1].argmax(dim=-1)

if __name__ == '__main__':
    main()