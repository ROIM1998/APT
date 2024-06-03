import torch
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import time

from transformers import HfArgumentParser
from args import DataTrainingArguments
from models import build_model
from models.model_args import ModelArguments
from utils import build_trainer
from utils.utils import *
from args import MinusTrainingArguments
from utils.cofi_utils import prune_model_with_z

def main():
    sys.argv = ['test_pruned_teacher_training.py',
            '--output_dir',
            './output/test_pruned_teacher_training/',
            '--model_name_or_path',
            'output/roberta-base_lora_minus_mnli_once_global_co_learning_loratransform_distill/mac0.4/epoch25/bz128/numprune5/lora_r64/lora_alpha16/pre_pruning_model',
            '--task_name',
            'mnli',
            '--evaluation_strategy',
            'steps',
            '--save_strategy',
            'no',
            '--do_train',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '32',
            '--per_device_eval_batch_size',
            '32',
            '--apply_lora',
            '--lora_r',
            '8',
            '--lora_alpha',
            '16',
            '--num_train_epochs',
            '30',
            '--learning_rate',
            '5e-4',
            '--warmup_ratio',
            '0.06',
            '--weight_decay',
            '0.1',
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
    t_name, raw_datasets = get_raw_datasets(model_args, data_args, training_args)
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)
    training_args.disable_tqdm = True
    head_mask, intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt')), torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt'))
    head_mask[-4:, :] = 1
    intermediate_mask[-4:, :] = 1
    zs = {
        'head_z': [v.to('cpu') for v in head_mask],
        'intermediate_z': [v.to('cpu') for v in intermediate_mask],
    }
    prune_model_with_z(zs, model)

    model.head_mask, model.intermediate_mask = None, None
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset)