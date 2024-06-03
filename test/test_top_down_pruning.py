import seaborn as sns
from matplotlib import pyplot as plt
import os
import sys
import torch
import re
import pandas as pd

from transformers import (HfArgumentParser)
from args import DataTrainingArguments
from models.model_args import ModelArguments
from args import MinusTrainingArguments
from utils.utils import *
from utils import build_dataloader, build_trainer
from models import build_model
from prune.pruner import AdapterPruner
from prune import build_pruner, build_scorer
from prune.search import search_mac
from torch.utils.data import Subset
from trainer.param_control import ParamController
from prune.fisher import collect_grads_by_suffix, collect_mask_grads, collect_hidden_mask_grads
from utils.fisher_utils.efficiency.mac import mac_per_head, mac_per_hidden_dim, mac_per_neuron, compute_mac

def violin_plot_neuron(grads, name='neuron'):
    fisher = grads.pow(2).sum(dim=0).cpu().log10()
    scores = [
        {
            'layer_id': i,
            'score': val.item(),
        }
        for i in range(fisher.shape[0]) for val in fisher[i]
    ]
    score_df = pd.DataFrame(scores)
    sns.violinplot(data=score_df, x='layer_id', y='score')
    plt.savefig('roberta_%s_scores_violin.png' % name)
    plt.clf()
    return score_df

def violin_plot_by_suffix(model, dataloader, suffix, status):
    # Create the violin plot grouped with #layer and attr (query or value)
    roberta_regex = r'^roberta\.encoder\.layer\.(\d+?)\.attention\.self\.(.+?)\..+$'
    grads = collect_grads_by_suffix(model, dataloader, suffix)
    scores = {k: v.pow(2).sum(dim=0) for k, v in grads.items() if suffix in k}
    scores = [
        {
            'layer_id': int(re.match(roberta_regex, k).group(1)),
            'attr': re.match(roberta_regex, k).group(2),
            'score': np.log10(val.item()),
        }
        for k, v in scores.items() for val in v
    ]
    score_df = pd.DataFrame(scores)
    sns.violinplot(data=score_df, x='layer_id', y='score', hue='attr', split=True)
    plt.savefig('%s_%s_scores_violin.png' % (status, suffix[1:].replace('_mask', '')))
    plt.clf()
    return score_df

def distribute_integer(m, ratios, n):
    # Calculate the total sum of the ratios
    total_ratio = sum(ratios)

    # Calculate the initial distribution based on ratios
    initial_distribution = [min(n, int(round(m * ratio / total_ratio))) for ratio in ratios]

    # Calculate the remaining amount after the initial distribution
    remaining = m - sum(initial_distribution)

    # Sort the ratios in descending order
    sorted_ratios = sorted(enumerate(ratios), key=lambda x: x[1], reverse=True)

    # Distribute the remaining amount to the chunks with the highest ratios
    for i in range(remaining):
        index, ratio = sorted_ratios[i % len(sorted_ratios)]
        while True:
            if initial_distribution[index] < n:
                initial_distribution[index] += 1
                break
            else:
                index = (index + 1) % len(sorted_ratios)

    return initial_distribution

@torch.no_grad()
def search_mac_topdown(
    config,
    head_importance,
    neuron_importance,
    hidden_importance,
    seq_len,
    mac_constraint,
):
    assert mac_constraint < 1

    num_hidden_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    hidden_size = config.hidden_size
    attention_head_size = int(hidden_size / num_attention_heads)

    original_mac = compute_mac(
        [num_attention_heads] * num_hidden_layers,
        [intermediate_size] * num_hidden_layers,
        seq_len,
        hidden_size,
        attention_head_size,
    )
    max_mac = mac_constraint * original_mac
    is_mask_matched = isinstance(head_importance, torch.Tensor) and isinstance(neuron_importance, torch.Tensor)
    device = head_importance.device if is_mask_matched else head_importance[0].device

    # Globally rank heads and neurons
    if is_mask_matched:
        concatenated_head_importance = head_importance.view(-1).clone()
        concatenated_neuron_importance = neuron_importance.view(-1).clone()
    else:
        assert isinstance(head_importance, list) and isinstance(neuron_importance, list)
        assert isinstance(head_importance[0], torch.Tensor) and isinstance(neuron_importance[0], torch.Tensor)
        concatenated_head_importance = torch.cat(head_importance, dim=0).clone()
        concatenated_neuron_importance = torch.cat(neuron_importance, dim=0).clone()

    sorted_head_importance, sorted_head_indicies = concatenated_head_importance.sort(descending=False)
    sorted_neuron_importance, sorted_neuron_indicies = concatenated_neuron_importance.sort(descending=False)
    sorted_hidden_importance, sorted_hidden_indicies = hidden_importance.sort(descending=False)
    total_head_num = head_importance.numel() if is_mask_matched else sum([h.numel() for h in head_importance])
    total_neuron_num = neuron_importance.numel() if is_mask_matched else sum([n.numel() for n in neuron_importance])
    total_hidden_num = hidden_importance.numel()

    # Calculate layer-level importance for top-down pruning
    head_layer_importance, neuron_layer_importance = head_importance.sum(dim=1), neuron_importance.sum(dim=1)
    smoothed_head_importance = head_layer_importance + head_layer_importance.mean()
    smoothed_neuron_importance = neuron_layer_importance + neuron_layer_importance.mean()
    smoothed_relative_head_importance = smoothed_head_importance / smoothed_head_importance.sum()
    smoothed_relative_neuron_importance = smoothed_neuron_importance / smoothed_neuron_importance.sum()
    sorted_head_importance_perlayer, sorted_head_indices_perlayer = head_importance.sort(dim=1, descending=True)
    sorted_neuron_importance_perlayer, sorted_neuron_indices_perlayer = neuron_importance.sort(dim=1, descending=True)
    
    # Firstly, set the hidden-head&neuron searching results
    min_importance = float('inf')
    for num_pruned_hidden in range(total_hidden_num + 1):
        rest_hidden_dim = total_hidden_num - num_pruned_hidden
        total_mac_adjusted = original_mac * rest_hidden_dim / total_hidden_num
        head_neuron_pruning_ratio = (total_mac_adjusted - max_mac) / total_mac_adjusted
        if head_neuron_pruning_ratio < 0:
            break
        if head_neuron_pruning_ratio > 1:
            continue
        hidden_importance = sorted_hidden_importance[:num_pruned_hidden]
        num_heads, num_neurons = int(head_neuron_pruning_ratio * total_head_num), int(head_neuron_pruning_ratio * total_neuron_num)
        pruned_head_importance, pruned_neuron_importance = sorted_head_importance[:num_heads], sorted_neuron_importance[:num_neurons]
        total_imoprtance = hidden_importance.sum() + pruned_head_importance.sum() + pruned_neuron_importance.sum()
        if total_imoprtance < min_importance:
            min_importance = total_imoprtance
            hidden_indicies = sorted_hidden_indicies[num_pruned_hidden:]
    
    rest_hidden_dim = len(hidden_indicies)
    hidden_size = rest_hidden_dim
    max_importance = -float('inf')
    
    for num_pruned_heads in range(total_head_num + 1):
        heads_mac = mac_per_head(seq_len, hidden_size, attention_head_size) * (total_head_num - num_pruned_heads)
        neurons_mac = max_mac - heads_mac
        num_pruned_neurons = total_neuron_num - int(neurons_mac / mac_per_neuron(seq_len, hidden_size))
        num_pruned_neurons = max(num_pruned_neurons, 0)
        num_heads_remained, num_neurons_remained = total_head_num - num_pruned_heads, total_neuron_num - num_pruned_neurons
        heads_per_layer, neuron_per_layer = distribute_integer(num_heads_remained, smoothed_relative_head_importance.tolist(), num_attention_heads), distribute_integer(num_neurons_remained, smoothed_relative_neuron_importance.tolist(), intermediate_size)
        total_importance = sum(v[:h].sum().item() for v, h in zip(sorted_head_importance_perlayer, heads_per_layer)) + sum(v[:n].sum().item() for v, n in zip(sorted_neuron_importance_perlayer, neuron_per_layer))
        if total_importance > max_importance:
            max_importance = total_importance
            head_indicies = [sorted_head_indices_perlayer[i][:h] for i, h in enumerate(heads_per_layer)]
            neuron_indicies = [sorted_neuron_indices_perlayer[i][:n] for i, n in enumerate(neuron_per_layer)]
    
    head_mask = torch.zeros(num_hidden_layers, num_attention_heads).to(device)
    neuron_mask = torch.zeros(num_hidden_layers, intermediate_size).to(device)
    hidden_mask = torch.zeros(total_hidden_num).to(device)
    for i, (h, n) in enumerate(zip(head_indicies, neuron_indicies)):
        head_mask[i][h] = 1.0
        neuron_mask[i][n] = 1.0
    hidden_mask[hidden_indicies] = 1.0

    if is_mask_matched:
        head_mask = head_mask.view(num_hidden_layers, num_attention_heads)
        neuron_mask = neuron_mask.view(num_hidden_layers, intermediate_size)
    else:
        head_mask = list(torch.split(head_mask, [h.numel() for h in head_importance]))
        neuron_mask = list(torch.split(neuron_mask, [n.numel() for n in neuron_importance]))

    return head_mask, neuron_mask, hidden_mask

def main():
    sys.argv = ['test_bert.py',
            '--output_dir',
            './output/test_bert/',
            '--model_name_or_path',
            'output/roberta-base_lora_minus_mnli_once_global_alloc_none_self_student_mapping_dynamic_block_teacher_dynamic_cofi_student_distill_fixedteacher/mac0.4/epoch40/bz32/numprune5/paramq:0-11,v:0-11,i:0-11/lora_r8/pruning_start1/distill_epoch19/pre_pruning_model',
            '--task_name',
            'mnli',
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
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)
    model = model.to(training_args.device)
    model.reset_masks()
    
    # Also add ffn input layers to teacher config
    teacher_keys = ['query', 'value', 'intermediate']
    teacher_config  = {
        k: [i for i in range(model.config.num_hidden_layers)] for k in teacher_keys
    }

    pruning_batch_size = 4
    num_pruning_batches = 64
    dataloader = build_dataloader(Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]), pruning_batch_size, data_args, training_args, tokenizer)
    adapter_pruner = AdapterPruner(model, dataloader)
    scorer = build_scorer('gradient_l2', model, dataloader)
    pruner = build_pruner(training_args.pruner_type, training_args, model, scorer_dict={
        'head_mask': scorer,
        'intermediate_mask': scorer,
    })
    
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        student_config=None,
        lora_with_bias=False,
        adapter_pruner=adapter_pruner,
        param_allocation_strategy=training_args.param_allocation_strategy,
    )
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    
    # Also setting hidden mask and get its grads
    model.hidden_mask = torch.ones(model.config.hidden_size, device=training_args.device)
    head_grads, intermediate_grads = collect_mask_grads(model, dataloader)
    head_scores, intermediate_scores = head_grads.pow(2).sum(dim=0), intermediate_grads.pow(2).sum(dim=0)
    hidden_grads = collect_hidden_mask_grads(model, dataloader)
    hidden_scores = hidden_grads.pow(2).sum(dim=0)
    sorted_score, sorted_idx = torch.sort(hidden_scores, descending=True)
    model.hidden_mask[sorted_idx[-5:]] = 0
    roberta_mac_per_head = mac_per_head(128, 768, 768 // 12)
    roberta_mac_per_neuron = mac_per_neuron(128, 768)
    roberta_mac_per_hidden = mac_per_hidden_dim(128, [768 for i in range(12)], [3072 for i in range(12)]) 
    
    score_per_mac_head = head_scores.mean() / roberta_mac_per_head
    score_per_mac_neuron = intermediate_scores.mean() / roberta_mac_per_neuron
    score_per_mac_hidden = hidden_scores.mean() / roberta_mac_per_hidden
    
    roberta_total_mac = compute_mac([12] * 12, [3072] * 12, 128, 768, 64)

    head_mask, intermediate_mask, hidden_mask = search_mac_topdown(model.config, head_scores, intermediate_scores, hidden_scores / 8, 128, 0.4)
    
    pruned_mac = compute_mac(head_mask.sum(dim=1).tolist(), intermediate_mask.sum(dim=1).tolist(), 128, hidden_mask.sum().item(), 64)
    model.prune_model_with_masks()

    head_layer_information = head_grads.pow(2).sum(dim=0).sum(dim=-1)
    head_layer_information_add_avg = head_layer_information + head_layer_information.mean()
    neuron_layer_information = intermediate_grads.pow(2).sum(dim=0).sum(dim=-1)
    neuron_layer_information_add_avg = neuron_layer_information + head_layer_information.mean()
    
    # Find the head-neuron allocation
    searched_head_mask, searched_intermediate_mask = search_mac(model.config, head_grads.pow(2).sum(dim=0), intermediate_grads.pow(2).sum(dim=0), 128, 0.4)
    l1_searched_head_mask, l1_searched_intermediate_mask = search_mac(model.config, head_grads.abs().sum(dim=0), intermediate_grads.abs().sum(dim=0), 128, 0.4)
    
    violin_plot_neuron(head_grads, 'head')
    violin_plot_neuron(intermediate_grads, 'neuron')
    head_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_head_mask.pt')).to(training_args.device)
    intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_intermediate_mask.pt')).to(training_args.device)
    
if __name__ == '__main__':
    main()