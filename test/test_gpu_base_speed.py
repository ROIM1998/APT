import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
from models import build_model
from transformers import HfArgumentParser
from args import DataTrainingArguments
from models.model_args import ModelArguments
from args import MinusTrainingArguments
from utils.utils import *
from utils.minus_utils import bench_latency

NUM_GPUS=8

def main():
    sys.argv = ['neuron_importance.py',
            '--output_dir',
            './output/neuron_importance/',
            '--model_name_or_path',
            'roberta-base',
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
            '64'
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
    
    results = {}
    for i in range(NUM_GPUS):
        model.cuda(i)
        results[i] = bench_latency(model, 128, 128, tokenizer)['t_mean'] * 1000
        
    

if __name__ == '__main__':
    main()