import pickle
import json
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import logging
import transformers
import torch
transformers.logging.set_verbosity_error()


from transformers import HfArgumentParser
from args import DataTrainingArguments
from models.model_args import ModelArguments
from utils.utils import *
from args import MinusTrainingArguments
from models import build_model
from torch.utils.data import DataLoader
from post_analysis import get_torch_saved_dirs
from utils.minus_utils import bench_latency
from utils.cofi_utils import update_params, prune_model_with_z

logger = logging.getLogger(__name__)

def test(eval_datasets, tasks, trainer, data_args):
    for eval_dataset, task in zip(eval_datasets, tasks):
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_val_samples = data_args.max_val_samples if hasattr(data_args, 'max_val_samples') and data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
    return metrics


def main():
    sys.argv = ['get_model_efficiency.py',
            '--output_dir',
            './output/roberta_lora_minus_mnli_once_test/freq0.1/batchuse64/mac0.6/',
            '--model_name_or_path',
            'roberta-base',
            '--task_name',
            'mnli',
            # '--do_train',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '32',
            '--per_device_eval_batch_size',
            '32',
            '--apply_lora',
            '--pruning_frequency',
            '0.1',
            '--pruning_batches',
            '64',
            '--mac_constraint',
            '0.6',
            '--pruning_scheduler',
            'once',
            '--pruning_steppoint',
            '0.5',
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
    IS_SQUAD = 'squad' in data_args.task_name
    t_name, raw_datasets = get_raw_datasets(model_args, data_args, training_args)
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    train_dataset, eval_dataset, predict_dataset, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)
    dataloader = DataLoader(
        eval_dataset,
        batch_size=training_args.per_device_eval_batch_size,
    )
    # inputs = next(iter(dataloader))
    # inputs = {k: torch.stack(v, dim=1).to(training_args.device) if isinstance(v, list) else v.to(training_args.device) for k, v in inputs.items()}

    model = model.to(training_args.device)
    if model.head_mask is not None:
        model.head_mask = model.head_mask.to(training_args.device)
    if model.intermediate_mask is not None:
        model.intermediate_mask = model.intermediate_mask.to(training_args.device)
    
    results = {}
    results['baseline'] = bench_latency(model, dataloader, warm_steps=3, num_reps=5)
    masks = pickle.load(open('masks.pt', 'rb'))
    dirs = list(set([
        '/'.join(path.split('/')[:-1]) for path in masks.keys()
    ]))
    for path in dirs:
        config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
        model.head_mask = masks[os.path.join(path, 'final_head_mask.pt')]
        model.intermediate_mask = masks[os.path.join(path, 'final_intermediate_mask.pt')]
        for n, p in model.named_parameters():
            p.requires_grad_(False)
        model = model.to(training_args.device)
        model.head_mask = model.head_mask.to(training_args.device)
        model.intermediate_mask = model.intermediate_mask.to(training_args.device)
        zs = {
            'head_z': model.head_mask,
            'intermediate_z': model.intermediate_mask,
        }
        update_params(model, zs)
        prune_model_with_z(zs, model)
        model.head_mask = [model.head_mask[i].index_select(dim=0, index=model.head_mask[i].nonzero().squeeze()) for i in range(model.head_mask.shape[0])]
        model.intermediate_mask = [model.intermediate_mask[i].index_select(dim=0, index=model.intermediate_mask[i].nonzero().squeeze()) for i in range(model.intermediate_mask.shape[0])]
        model.eval()
        results[path] = bench_latency(model, dataloader, warm_steps=3, num_reps=5)
    json.dump(results, open('new_efficiency.json', 'w'))

if __name__ == '__main__':
    main()