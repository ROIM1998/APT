import torch
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import time

from transformers import HfArgumentParser
from args import DataTrainingArguments
from models import build_model
from models.model_args import ModelArguments
from trainer.trainer_minus import ParamController
from utils import build_dataloader, build_trainer
from utils.utils import *
from args import MinusTrainingArguments
from loralib.layers import LoRALayer, DistillLinear, PruningLinear
from utils.minus_utils import efficiency_testing, compare_module_inputs_equality, collect_module_inputs
from tqdm import tqdm
from torch.utils.data import Subset
from prune import build_pruner, build_scorer, build_pruning_scheduler, BetterFisherPruner
from prune.search import search_encoder_decoder_mac

def collect_lora_info(model):
    lora_vars = [n for n, p in model.named_parameters() if 'lora' in n]
    lora_param_num = sum([p.numel() for n, p in model.named_parameters() if 'lora' in n])
    lora_layers = [n for n, p in model.named_modules() if isinstance(p, LoRALayer)]
    prune_layers = [n for n, p in model.named_modules() if isinstance(p, PruningLinear)]
    distill_layers = [n for n, p in model.named_modules() if isinstance(p, DistillLinear)]
    return {
        'lora_vars': lora_vars,
        'lora_param_num': lora_param_num,
        'lora_layers': lora_layers,
        'prune_layers': prune_layers,
        'distill_layers': distill_layers,
    }
    

def main():
    sys.argv = ['test_t5.py',
            '--output_dir',
            './output/test_t5_grafting/',
            '--model_name_or_path',
            'old_output/t5-3b_lora_minus_once_search_nodistill/epochs20/batch32/mac0.4/lora_alpha16/lora_r8/pre_pruning_model',
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
            '--eval_accumulation_steps',
            '1',
            '--lora_r',
            '8',
            '--lora_alpha',
            '16',
            '--apply_lora',
            '--pruner_type',
            'global',
            '--head_scorer_type',
            'gradient_l2',
            '--intermediate_scorer_type',
            'gradient_l2',
            '--pruning_batch_size',
            '4',
            '--pruning_batches',
            '64',
            '--pruning_scheduler',
            'once',
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
    MODEL_GENERATIVE = True
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets, MODEL_GENERATIVE)

    # Building the trainer
    pruning_batch_size, num_pruning_batches = 4, 64
    # print(trainer.evaluate())
    dataloader = build_dataloader(Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]), training_args.per_device_train_batch_size, data_args, training_args, tokenizer)
    model.to(training_args.device)

    teacher_keys = ['enc_self_query', 'enc_self_value', 'dec_self_query', 'dec_self_value', 'cross_query', 'cross_value']
    teacher_config = {
        k: [i for i in range(config.num_layers)] for k in teacher_keys
    }
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        lora_with_bias=False,
    )
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)