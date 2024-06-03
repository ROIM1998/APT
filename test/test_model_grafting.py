import seaborn as sns
sns.set_theme(style="darkgrid")
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
import time
import pandas as pd
import loralib as lora

from transformers import (HfArgumentParser)
from args import DataTrainingArguments
from models.model_args import ModelArguments
from utils.utils import *
from utils.minus_utils import lora_to_prunelora, shrink_pruning_lora, shrink_pruning_lora_outdim, shrink_pruning_lora_bottleneckdim, expand_pruning_lora_bottleneckdim, lora_to_linear
from utils import build_trainer, build_dataloader
from args import MinusTrainingArguments
from models import build_model
from prune.fisher import collect_additive_mask_grads
from prune.pruner import AdapterPruner
from torch.utils.data import DataLoader, Subset
from trainer.param_control import ParamController

def test_expand_or_shrink(prune_ratio: float = 0.5, init_dim: int = 8, init_num: int = 24):
    dims = [torch.ones(init_dim) for i in range(init_num)]
    print(f'init_dims: {[len(v) for v in dims]}')
    for _ in range(10):
        summed = int(sum([v.sum().item() for v in dims]))
        num_pruned = int(summed * prune_ratio)
        mask = torch.ones(summed)
        mask[torch.randperm(summed)[:num_pruned]] = 0
        new_dims = []
        i = 0
        for dim in dims:
            status = mask[i:i+len(dim)]
            if status.all():
                new_dims.append(torch.ones(len(dim) * 2))
            elif status.any():
                new_dims.append(torch.ones(int(len(dim) - (1 - status).sum().item())))
            i += len(dim)
        print(f'new_dims: {[len(v) for v in new_dims]}')
        dims = new_dims


def main():
    sys.argv = ['test_pre_tuning_prune.py',
            '--output_dir',
            './output/test_model_grafting_dynamic/',
            '--model_name_or_path',
            'output/roberta-base_lora_minus_mnli_once_global_co_learning_loratransform_distill/mac0.4/epoch25/bz128/numprune5/lora_r64/lora_alpha16/pre_pruning_model',
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
            '8',
            '--lora_alpha',
            '16',
            '--save_strategy',
            'no',
            '--evaluation_strategy',
            'steps',
            '--num_train_epochs',
            '30',
            '--learning_rate',
            '5e-4',
            '--weight_decay',
            '0.1',
            '--warmup_ratio',
            '0.06',
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
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)
    
    named_modules = dict(model.named_modules())
    for n, p in model.named_modules():
        if isinstance(p, lora.Linear):
            parent_layer_attr, attr = n.rsplit('.', 1)
            parent_layer = named_modules[parent_layer_attr]
            if 'intermediate' in n:
                setattr(parent_layer, attr, lora_to_linear(p))
            else:
                new_layer = lora_to_prunelora(p, r=8, lora_alpha=16)
                new_layer.set_grafting_mask()
                setattr(parent_layer, attr, new_layer)

    model.head_mask, model.intermediate_mask = None, None

    pruning_batch_size = 4
    num_pruning_batches = 64
    dataloader = build_dataloader(Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]), pruning_batch_size, data_args, training_args, tokenizer)

    teacher_keys = ['query', 'value']
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
    )
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    original_results = trainer.evaluate()
    output_names, bottleneck_names, all_output_mask, all_bottleneck_mask = param_controller.expand_or_shrink_teacher_layers(0.5, 0.5)
    converted_results = trainer.evaluate()

    for p in model.modules():
        if isinstance(p, lora.PruningLinear):
            p.set_grafting_mask()
    # Maybe need to switch to head-level granularity
    all_grads = collect_additive_mask_grads(model, dataloader)
    print({k: v.shape for k, v in all_grads.items()})
    all_output_scores, all_bottleneck_scores = {k: v.pow(2).sum(dim=0) for k, v in all_grads.items() if 'intermediate' not in k and 'output_mask' in k}, {k: v.pow(2).sum(dim=0) for k, v in all_grads.items() if 'intermediate' not in k and 'bottleneck_mask' in k}
    output_names, concat_output_scores = list(all_output_scores), torch.cat(list(all_output_scores.values()), dim=0)
    bottleneck_names, concat_bottleneck_scores = list(all_bottleneck_scores), torch.cat(list(all_bottleneck_scores.values()), dim=0)
    output_score_lens, bottleneck_score_lens = [len(all_output_scores[k]) for k in output_names], [len(all_bottleneck_scores[k]) for k in bottleneck_names]

    select_ratio = 0.5
    sorted_output_scores, sorted_output_indices = concat_output_scores.sort(descending=True)
    sorted_bottleneck_scores, sorted_bottleneck_indices = concat_bottleneck_scores.sort(descending=True)
    all_output_mask, all_bottleneck_mask = torch.zeros_like(concat_output_scores), torch.zeros_like(concat_bottleneck_scores)
    all_output_mask[sorted_output_indices[:int(len(sorted_output_indices) * select_ratio)]] = 1
    all_bottleneck_mask[sorted_bottleneck_indices[:int(len(sorted_bottleneck_indices) * select_ratio)]] = 1
    all_output_mask, all_bottleneck_mask = all_output_mask.split(output_score_lens), all_bottleneck_mask.split(bottleneck_score_lens)
    
    # Test PruningLinear shrinking
    layer = model.roberta.encoder.layer[0].attention.self.query
    pruned_bottleneck = (all_bottleneck_mask[1] == 0).nonzero().squeeze().tolist()
    pruned_out_dim = (all_output_mask[1] == 0).nonzero().squeeze().tolist()
    shrinked_layer = shrink_pruning_lora_outdim(layer, pruned_out_dim)
    shrinked_layer = shrink_pruning_lora_bottleneckdim(shrinked_layer, pruned_bottleneck)
    layer.eval()
    shrinked_layer.eval()
    print((layer.weight - shrinked_layer.weight).abs().max())
    layer.train()
    direct_shrinked_layer = shrink_pruning_lora(layer, pruned_out_dim, pruned_bottleneck)
    layer.eval()
    direct_shrinked_layer.eval()
    print((layer.weight - direct_shrinked_layer.weight).abs().max())
    layer.train()
    expanded_layer = expand_pruning_lora_bottleneckdim(layer, layer.r * 2)
    layer.eval()
    expanded_layer.eval()
    print((layer.weight - expanded_layer.weight).abs().max())
    
    expanding_ratio = 2
    extra_bottleneck_dims = int(model_args.lora_r * all_bottleneck_mask.all(dim=1).sum().item() * (expanding_ratio - 1))
    new_all_bottleneck_mask = torch.zeros_like(all_bottleneck_mask)
    new_all_bottleneck_mask.view(-1)[sorted_bottleneck_indices[:int(len(sorted_bottleneck_indices) * select_ratio) - extra_bottleneck_dims]] = 1
    named_modules = dict(model.named_modules())
    for output_name, bottleneck_name, output_mask, old_bottleneck_mask, bottleneck_mask in zip(output_names, bottleneck_names, all_output_mask, all_bottleneck_mask, new_all_bottleneck_mask):
        layer_name = output_name.rsplit('.', 1)[0]
        layer = named_modules[layer_name]
        parent_layer_name, layer_attr = layer_name.rsplit('.', 1)
        parent_layer = named_modules[parent_layer_name]
        if (not output_mask.any() or not bottleneck_mask.any()) and not old_bottleneck_mask.all():
            # mask all equals to 0
            shrinked_layer = lora_to_linear(layer)
        else:
            pruned_bottleneck_dim = (bottleneck_mask == 0).nonzero().squeeze()
            pruned_out_dim = (output_mask == 0).nonzero().squeeze()
            pruned_bottleneck_dim = pruned_bottleneck_dim.tolist() if pruned_bottleneck_dim.dim() else [pruned_bottleneck_dim.item()]
            pruned_out_dim = pruned_out_dim.tolist() if pruned_out_dim.dim() else [pruned_out_dim.item()]
            if old_bottleneck_mask.all():
                if not output_mask.all():
                    shrinked_layer = layer
                else:
                    shrinked_layer = shrink_pruning_lora_outdim(layer, pruned_out_dim)
                target_r  = layer.r * expanding_ratio
                if bottleneck_mask.any():
                    if not bottleneck_mask.all():
                        shrinked_layer = shrink_pruning_lora_bottleneckdim(shrinked_layer, pruned_bottleneck_dim)
                        target_r -= len(pruned_bottleneck_dim)
                    shrinked_layer = expand_pruning_lora_bottleneckdim(shrinked_layer, target_r)
            else:
                if output_mask.all():
                    shrinked_layer = shrink_pruning_lora_bottleneckdim(layer, pruned_bottleneck_dim)
                else:
                    shrinked_layer = shrink_pruning_lora(layer, pruned_out_dim, pruned_bottleneck_dim)        
        setattr(parent_layer, layer_attr, shrinked_layer)
    print("Sum of lora r dimensions:", sum([m.r for m in model.modules() if isinstance(m, lora.LoRALayer)]))

    trainer.auto_layer_conversion = False
    trainer.dynamic_grafting = True
    train_result = trainer.train()

if __name__ == '__main__':
    main()