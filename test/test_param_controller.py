import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
from transformers import HfArgumentParser, default_data_collator, DataCollatorWithPadding
from args import DataTrainingArguments
from models import build_model
from models.model_args import ModelArguments
from utils.utils import *
from args import MinusTrainingArguments
from torch.utils.data import DataLoader, Subset
from trainer.param_control import ParamController
from utils.minus_utils import count_params

def main():
    sys.argv = ['neuron_importance.py',
            '--output_dir',
            './output/neuron_importance/',
            '--model_name_or_path',
            'output/roberta-base_lora_minus_mnli_once_fisher_distill_full/step1.0/batchuse64/mac0.6',
            '--task_name',
            'mnli',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '32',
            '--per_device_eval_batch_size',
            '32',
            '--apply_lora',
            '--do_distill'
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
    _, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)
    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    dataloader = DataLoader(
        Subset(eval_dataset, torch.randperm(len(eval_dataset)).tolist()[:training_args.per_device_eval_batch_size * 64]),
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    inputs = next(iter(dataloader))
    
    teacher_config = {
        'key': [9,10,11],
        'query': [9, 10, 11],
        'value': [9, 10, 11],
    }
    student_config = {
        'intermediate': [9,10,11],
    }
    controller = ParamController(model, teacher_config, student_config)
    results = {}
    results['original'] = count_params(model, mode='tuned')
    controller.freeze()
    results['freeze'] = count_params(model, mode='tuned')
    controller.model_as_teacher()
    results['teacher'] = count_params(model, mode='tuned')
    controller.model_as_student()
    results['student'] = count_params(model, mode='tuned')
    
if __name__ == '__main__':
    main()