import seaborn as sns
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
from prune.fisher import collect_grads_by_suffix, collect_mask_grads, collect_param_salience
from matplotlib import pyplot as plt

if __name__ == 'main':
    sys.argv = ['compare_tuning_freeze_fisher.py',
            '--output_dir',
            './output/test_model_prune_recover/',
            '--model_name_or_path',
            'output/bert-base-uncased_lora_minus_mnli_once_global_free_inout_nodistill/mac0.4/epoch30/bz128/numprune5/paramq:0-11,v:0-11,i:0-11/lora_r8/lora_alpha16/pre_pruning_model',
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
            '--report_to',
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
    
    model = model.to(training_args.device)
    model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_head_mask.pt')).cuda()
    model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_intermediate_mask.pt')).cuda()
    
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
    param_controller.convert_to_pre_pruning_lora_teacher()
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    
    param_controller.set_grafting_mask(mode=True, target='teacher')
    model.reset_masks()
    
    head_grads, intermediate_grads = collect_mask_grads(model, dataloader)
    output_mask_grads = collect_grads_by_suffix(model, dataloader, '.output_mask')
    
    query_template = 'bert.encoder.layer.%d.attention.self.query.output_mask'
    query_output_grads = [
        torch.stack([output_mask_grads[query_template % i][:, j:j+64].sum(dim=1) for j in range(0, 768, 64)], dim=1)
        for i in range(12)
    ]
    query_output_grads = torch.stack(query_output_grads, dim=1)
    
    value_template = 'bert.encoder.layer.%d.attention.self.value.output_mask'
    value_output_grads = [
        torch.stack([output_mask_grads[value_template % i][:, j:j+64].sum(dim=1) for j in range(0, 768, 64)], dim=1)
        for i in range(12)
    ]
    value_output_grads = torch.stack(value_output_grads, dim=1)
    
    intermediate_template = 'bert.encoder.layer.%d.intermediate.dense.output_mask'
    intermediate_output_grads = torch.stack([
        output_mask_grads[intermediate_template % i]
        for i in range(12)
    ], dim=1)
    
    param_to_collect = set([n for n, p in model.named_parameters() if 'lora' in n])
    params_salience = collect_param_salience(model, dataloader, param_to_collect)
    aggregated_mask_salience = {k: v.sum(dim=2) for k, v in params_salience.items()}
    
    
    head_fisher = head_grads.pow(2).sum(dim=0)
    query_output_fisher = query_output_grads.pow(2).sum(dim=0)
    value_output_fisher = value_output_grads.pow(2).sum(dim=0)
    intermediate_fisher = intermediate_grads.pow(2).sum(dim=0)
    intermediate_output_fisher = intermediate_output_grads.pow(2).sum(dim=0)
    
    total_params_per_head = (768 * 768 // 12 + 768 // 12) * 4
    tuning_params_per_head = (8 * 768 // 12) * 2
    total_params_per_neuron = (768 * 2 + 1)
    tuning_params_per_neuron = 8
    
    avg_head_fisher_per_param = head_fisher.mean() / total_params_per_head
    avg_neuron_fisher_per_param = intermediate_fisher.mean() / total_params_per_neuron
    
    sns.scatterplot(x=head_fisher.cpu().log().view(-1).numpy(), y=query_output_fisher.cpu().log().view(-1).numpy())
    plt.savefig('head_query_output.png')
    plt.clf()
    sns.scatterplot(x=head_fisher.cpu().log().view(-1).numpy(), y=value_output_fisher.cpu().log().view(-1).numpy())
    plt.savefig('head_value_output.png')
    plt.clf()
    sns.scatterplot(x=intermediate_fisher.cpu().log().view(-1).numpy(), y=intermediate_output_fisher.cpu().log().view(-1).numpy())
    plt.savefig('intermediate_output.png')
    plt.clf()