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
from prune.fisher import collect_mask_grads
from matplotlib import pyplot as plt
from prune import BetterFisherPruner, GradientScorer
from utils.fisher_utils.efficiency.mac import mac_per_head, mac_per_neuron

if __name__ == 'main':
    sys.argv = ['compare_tuning_freeze_fisher.py',
            '--output_dir',
            './output/test_model_prune_recover/',
            '--model_name_or_path',
            'output/bert-base-uncased_lora_minus_mnli_once_fixed_none_nodistill/mac0.4/epoch30/bz32/numprune5/paramq:0-11,v:0-11,i:0-11/lora_r8/lora_alpha16/pre_pruning_model',
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
    model.reset_masks()
    
    
    total_params_per_head = (768 * 768 // 12 + 768 // 12) * 4
    tuning_params_per_head = (8 * 768 // 12) * 2
    total_params_per_neuron = (768 * 2 + 1)
    tuning_params_per_neuron = 8
    
    scorer = GradientScorer(model, dataloader, normalize=True)
    scorer_dict = {
        'head_mask': scorer,
        'intermediate_mask': scorer
    }
    pruner = BetterFisherPruner(model, ['head_mask', 'intermediate_mask'], scorer_dict, 128, True, ['search', 'better_rearrange', 'global'])
    masks = pruner.generate_mask(0.4)
    head_mask, intermediate_mask = masks['head_mask'], masks['intermediate_mask']
    head_density = head_mask.sum() / head_mask.numel()
    intermediate_density = intermediate_mask.sum() / intermediate_mask.numel()
    torch.save(head_mask, 'output/adap_normalized_mask/head_mask.pt')
    torch.save(intermediate_mask, 'output/adap_normalized_mask/intermediate_mask.pt')