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
            'roberta-base',
            '--task_name',
            'sst2',
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
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets, force_model_shape_deduction=True)
    model.head_mask = model.head_mask.to(training_args.device).view(-1)
    model.intermediate_mask = model.intermediate_mask.to(training_args.device).view(-1)
    model.hidden_mask = model.hidden_mask.to(training_args.device).view(-1)
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)
    model = model.to(training_args.device)
    pruning_batch_size = 32
    num_pruning_batches = 64
    dataloader = build_dataloader(Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]), pruning_batch_size, data_args, training_args, tokenizer)

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
    for m in model.modules():
        if isinstance(m, lora.Linear):
            m.scaling = 2
    
    
    inputs = next(iter(dataloader))
    inputs = trainer._prepare_inputs(inputs)

    
    outputs = model(**inputs, return_dict=True)
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
    
    
    
    trainer.scorer._remove_hooks()


if __name__ == '__main__':
    main()