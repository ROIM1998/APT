import seaborn as sns
sns.set_theme(style="darkgrid")
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
import loralib as lora

from transformers import (HfArgumentParser)
from args import DataTrainingArguments
from models.model_args import ModelArguments
from utils.utils import *
from utils import build_trainer, build_dataloader
from args import MinusTrainingArguments
from models import build_model
from prune.pruner import AdapterPruner
from torch.utils.data import Subset
from trainer.param_control import ParamController
from prune.fisher import collect_grads_by_suffix

def main():
    sys.argv = ['test_distill_with_reallocation.py',
            '--output_dir',
            './exp_output/test_distill_with_reallocation/',
            '--model_name_or_path',
            'exp_output/roberta-base_lora_minus_mnli_once_global_alloc_free_inout_self_interleave_mapping_none_distill/mac0.4/epoch30/bz128/numprune5/paramq:0-11,v:0-11,i:0-11/lora_r32/lora_alpha16/best_model',
            '--task_name',
            'mnli',
            '--do_train',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '128',
            '--per_device_eval_batch_size',
            '128',
            '--apply_lora',
            '--lora_r',
            '32',
            '--lora_alpha',
            '16',
            '--save_strategy',
            'no',
            '--evaluation_strategy',
            'steps',
            '--num_train_epochs',
            '0.1',
            '--learning_rate',
            '5e-4',
            '--weight_decay',
            '0.1',
            '--warmup_ratio',
            '0.06',
            '--report_to',
            'none',
            '--do_distill',
            '--continuous_allocation',
            '--continuous_alloc_interval',
            '1',
            '--distillation_type',
            'self_interleave',
            '--distill_mapping_strategy',
            'none',
            '--param_allocation_strategy',
            'free_inout'
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
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)

    pruning_batch_size = 4
    num_pruning_batches = 64
    dataloader = build_dataloader(Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]), pruning_batch_size, data_args, training_args, tokenizer)

    teacher_keys = ['query', 'value', 'intermediate']
    teacher_config  = {
        k: [i for i in range(model.config.num_hidden_layers)] for k in teacher_keys
    }
    student_keys = ['query', 'value', 'intermediate']
    student_config = {
        k: [i for i in range(model.config.num_hidden_layers)] for k in student_keys
    }
    adapter_pruner = AdapterPruner(model, dataloader)
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        student_config=student_config,
        lora_with_bias=False,
        adapter_pruner=adapter_pruner,
        param_allocation_strategy=training_args.param_allocation_strategy,
    )
        
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    trainer.auto_layer_conversion=False
    model.head_mask, model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt'), map_location='cuda'), torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt'), map_location='cuda')
    trainer.distilling = True
    trainer.evaluate()
    
    param_controller.set_grafting_mask(mode=True, target='student')
    all_grads = collect_grads_by_suffix(model, adapter_pruner.dataloader, '.bottleneck_mask')
        
if __name__ == '__main__':
    main()