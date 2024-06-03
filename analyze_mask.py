import seaborn as sns
import os
import sys
import torch

from typing import Tuple
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
from prune.fisher import collect_hidden_mask_grads, collect_mask_grads
from matplotlib import pyplot as plt
from utils.fisher_utils.efficiency.mac import *
from post_analysis import load_gen_mask_with_restoration
from utils.cofi_utils import parse_cofi_zs

def mask_similarity(mask_a: torch.Tensor, mask_b: torch.Tensor) -> Tuple[float]:
    assert mask_a.shape == mask_b.shape
    similarity = (mask_a == mask_b).sum() / mask_a.numel()
    pruned_in_a = mask_a.view(-1).nonzero().squeeze().tolist()
    pruned_in_b = mask_b.view(-1).nonzero().squeeze().tolist()
    # Assert b is the correct answer. Calculating the recall and precision of pruned blocks in a
    a_recall = len(set(pruned_in_a) & set(pruned_in_b)) / len(pruned_in_b)
    a_precision = len(set(pruned_in_a) & set(pruned_in_b)) / len(pruned_in_a)
    return (mask_a == mask_b).sum() / mask_a.numel(), a_recall, a_precision

if __name__ == 'main':
    sys.argv = ['analyze_mask.py',
            '--output_dir',
            './output/analyze_mask/',
            '--model_name_or_path',
            'output/bert-base-uncased_ft_minus_mnli_once_fixed_alloc_none_self_student_mapping_dynamic_block_teacher_dynamic_student_distill_fixedteacher_smalllr/mac0.4/epoch40/bz32/numprune5/pruning_start1/distill_epoch19/pre_pruning_model',
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
            '--do_distill',
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


    
    # Also add ffn input layers to teacher config
    teacher_keys = ['query', 'value', 'intermediate']
    teacher_config  = {
        k: [i for i in range(model.config.num_hidden_layers)] for k in teacher_keys
    }

    pruning_batch_size = 32
    num_pruning_batches = 2048
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
    
    model.head_mask, model.intermediate_mask = model.head_mask.cuda(), model.intermediate_mask.cuda()
    model.hidden_mask = torch.ones(768).cuda()
    trainer.evaluate()
    
    head_grads, intermediate_grads = collect_mask_grads(model, dataloader)
    hidden_grads = collect_hidden_mask_grads(model, dataloader)
    print(torch.cuda.max_memory_allocated() / (1024 ** 2))
    hidden_score = hidden_grads.pow(2).sum(dim=0)
    sorted_score, sorted_idx = torch.sort(hidden_score, descending=True)
    model.hidden_mask[sorted_idx[-5:]] = 0
    model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_head_mask.pt')).cuda()
    model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_intermediate_mask.pt')).cuda()
    pre_prune_masked_metrics = trainer.evaluate()
    
    model = model.cpu()
    model.head_mask, model.intermediate_mask, model.hidden_mask = model.head_mask.cpu(), model.intermediate_mask.cpu(), model.hidden_mask.cpu()
    model.prune_model_with_masks()
    model = model.cuda()
    model.hidden_mask = None
    post_prune_masked_metrics = trainer.evaluate()
    
    head_l1_score = head_grads.abs().sum(dim=0)
    intermediate_l1_score = intermediate_grads.abs().sum(dim=0)
    head_l2_score = head_grads.pow(2).sum(dim=0)
    intermediate_l2_score = intermediate_grads.pow(2).sum(dim=0)    

    cofi_head_mask, cofi_intermediate_mask, cofi_hidden_mask = torch.load('output/cofi_mask/head_mask.pt'), torch.load('output/cofi_mask/intermediate_mask.pt'), torch.load('output/cofi_mask/hidden_mask.pt')
    minus_head_mask, minus_intermediate_mask = torch.load('output/bert-base-uncased_lora_minus_mnli_once_global_free_inout_nodistill/mac0.4/epoch30/bz128/numprune5/paramq:0-11,v:0-11,i:0-11/lora_r8/lora_alpha16/final_head_mask.pt', map_location='cpu'), torch.load('output/bert-base-uncased_lora_minus_mnli_once_global_free_inout_nodistill/mac0.4/epoch30/bz128/numprune5/paramq:0-11,v:0-11,i:0-11/lora_r8/lora_alpha16/final_intermediate_mask.pt', map_location='cpu')
    
    cofi_retained_head_l1_scores = head_l1_score.view(-1)[cofi_head_mask.view(-1) == 1]
    cofi_pruned_head_l1_scores = head_l1_score.view(-1)[cofi_head_mask.view(-1) == 0]
    cofi_retained_head_l2_scores = head_l2_score.view(-1)[cofi_head_mask.view(-1) == 1]
    cofi_pruned_head_l2_scores = head_l2_score.view(-1)[cofi_head_mask.view(-1) == 0]
    sns.distplot(cofi_retained_head_l1_scores.cpu().detach().numpy(), label='Retained')
    sns.distplot(cofi_pruned_head_l1_scores.cpu().detach().numpy(), label='Pruned')
    plt.legend()
    plt.savefig('cofi_head_l1.png')
    plt.clf()
    sns.distplot(cofi_retained_head_l2_scores.cpu().log().detach().numpy(), label='Retained')
    sns.distplot(cofi_pruned_head_l2_scores.cpu().log().detach().numpy(), label='Pruned')
    plt.legend()
    plt.savefig('cofi_head_l2.png')
    plt.clf()
    
    cofi_retained_neuron_l1_scores = intermediate_l1_score.view(-1)[cofi_intermediate_mask.view(-1) == 1]
    cofi_pruned_neuron_l1_scores = intermediate_l1_score.view(-1)[cofi_intermediate_mask.view(-1) == 0]
    cofi_retained_neuron_l2_scores = intermediate_l2_score.view(-1)[cofi_intermediate_mask.view(-1) == 1]
    cofi_pruned_neuron_l2_scores = intermediate_l2_score.view(-1)[cofi_intermediate_mask.view(-1) == 0]
    sns.distplot(cofi_retained_neuron_l1_scores.cpu().log().detach().numpy(), label='Retained')
    sns.distplot(cofi_pruned_neuron_l1_scores.cpu().log().detach().numpy(), label='Pruned')
    plt.legend()
    plt.savefig('cofi_neuron_l1.png')
    plt.clf()
    sns.distplot(cofi_retained_neuron_l2_scores.cpu().log().detach().numpy(), label='Retained')
    sns.distplot(cofi_pruned_neuron_l2_scores.cpu().log().detach().numpy(), label='Pruned')
    plt.legend()
    plt.savefig('cofi_neuron_l2.png')
    plt.clf()
    head_grad_mean = head_grads.mean(dim=0)
    neuron_grad_mean = intermediate_grads.mean(dim=0)
    cofi_retained_head_grad_mean = head_grad_mean.view(-1)[cofi_head_mask.view(-1) == 1]
    cofi_pruned_head_grad_mean = head_grad_mean.view(-1)[cofi_head_mask.view(-1) == 0]
    sns.distplot(cofi_retained_head_grad_mean.cpu().log().detach().numpy(), label='Retained')
    sns.distplot(cofi_pruned_head_grad_mean.cpu().log().detach().numpy(), label='Pruned')
    plt.legend()
    plt.savefig('cofi_head_grad_mean.png')
    plt.clf()
    cofi_retained_neuron_grad_mean = neuron_grad_mean.view(-1)[cofi_intermediate_mask.view(-1) == 1]
    cofi_pruned_neuron_grad_mean = neuron_grad_mean.view(-1)[cofi_intermediate_mask.view(-1) == 0]
    sns.distplot(cofi_retained_neuron_grad_mean.cpu().log().detach().numpy(), label='Retained')
    sns.distplot(cofi_pruned_neuron_grad_mean.cpu().log().detach().numpy(), label='Pruned')
    plt.legend()
    plt.savefig('cofi_neuron_grad_mean.png')
    plt.clf()
    
    head_sum_abs = head_grads.sum(dim=0).abs()
    intermediate_sum_abs = intermediate_grads.sum(dim=0).abs()
    cofi_retained_head_grad_abs_sum = head_sum_abs.view(-1)[cofi_head_mask.view(-1) == 1]
    cofi_pruned_head_grad_abs_sum = head_sum_abs.view(-1)[cofi_head_mask.view(-1) == 0]
    sns.distplot(cofi_retained_head_grad_abs_sum.cpu().log().detach().numpy(), label='Retained')
    sns.distplot(cofi_pruned_head_grad_abs_sum.cpu().log().detach().numpy(), label='Pruned')
    plt.legend()
    plt.savefig('cofi_head_grad_sum_abs.png')
    plt.clf()
    cofi_retained_neuron_grad_abs_sum = intermediate_sum_abs.view(-1)[cofi_intermediate_mask.view(-1) == 1]
    cofi_pruned_neuron_grad_abs_sum = intermediate_sum_abs.view(-1)[cofi_intermediate_mask.view(-1) == 0]
    sns.distplot(cofi_retained_neuron_grad_abs_sum.cpu().log().detach().numpy(), label='Retained')
    sns.distplot(cofi_pruned_neuron_grad_abs_sum.cpu().log().detach().numpy(), label='Pruned')
    plt.legend()
    plt.savefig('cofi_neuron_grad_sum_abs.png')
    plt.clf()
    
    
    
    cofi_head_density = cofi_head_mask.sum() / cofi_head_mask.numel()
    cofi_intermediate_density = cofi_intermediate_mask.sum() / cofi_intermediate_mask.numel()
    minus_head_density = minus_head_mask.sum() / minus_head_mask.numel()
    minus_intermediate_density = minus_intermediate_mask.sum() / minus_intermediate_mask.numel()
    
    num_attention_heads, num_hidden_layers = 12, 12
    intermediate_size = 3072
    seq_len = 128
    hidden_size = 768
    attention_head_size = hidden_size // num_attention_heads
    
    mac_constraint = 0.4
    total_mac = compute_mac(
        [num_attention_heads] * num_hidden_layers,
        [intermediate_size] * num_hidden_layers,
        seq_len,
        hidden_size,
        attention_head_size,
    )
    target_mac = total_mac * mac_constraint
    
    ratio_mac = []
    sorted_head_importance = head_l2_score.view(-1).sort(descending=True)[0]
    sorted_neuron_importance = intermediate_l2_score.view(-1).sort(descending=True)[0]
    
    cofi_mac = cofi_head_mask.sum().item() * mac_per_head(seq_len, hidden_size, attention_head_size) + cofi_intermediate_mask.sum().item() * mac_per_neuron(seq_len, hidden_size)
    cofi_importance = torch.dot(cofi_head_mask.view(-1), head_l2_score.view(-1).cpu()) + torch.dot(cofi_intermediate_mask.view(-1), intermediate_l2_score.view(-1).cpu())
        
    max_importance = -float('inf')
    
    for num_heads in range(0, 144 + 1):
        heads_mac = mac_per_head(seq_len, hidden_size, attention_head_size) * num_heads
        neurons_mac = cofi_mac - heads_mac
        num_neurons = int(neurons_mac / mac_per_neuron(seq_len, hidden_size))
        num_neurons = max(num_neurons, 0)

        total_importance = sorted_head_importance[:num_heads].sum() + sorted_neuron_importance[:num_neurons].sum()
        if total_importance > max_importance:
            max_importance = total_importance
        ratio_mac.append((num_heads, num_neurons, total_importance, heads_mac, neurons_mac, (heads_mac + num_neurons * mac_per_neuron(seq_len, hidden_size)) / cofi_mac))
        
    sns.lineplot(x=[v[0] for v in ratio_mac], y=[v[2].item() for v in ratio_mac])
    # Add cofi result as the point on the figure
    sns.scatterplot(x=[cofi_head_mask.sum().item()], y=[cofi_importance.item()], color='red')
    
    plt.xlabel('num heads')
    plt.ylabel('total importance')
    plt.savefig('corr_head_total_importance.png')
    plt.clf()
    
    cofi_head_per_layer = cofi_head_mask.sum(dim=1)
    cofi_neuron_per_layer = cofi_intermediate_mask.sum(dim=1)
    
    # Manually generate minus mask according to CoFi's coarse-grained allocation
    cofi_head_pruned_num = (1 - cofi_head_mask).sum(dim=1)
    cofi_intermediate_pruned_num = (1 - cofi_intermediate_mask).sum(dim=1)
    head_mask = torch.ones_like(cofi_head_mask)
    intermediate_mask = torch.ones_like(cofi_intermediate_mask)
    for layer in range(12):
        sorted_head_importance, sorted_head_indices = head_l2_score[layer].sort(descending=False)
        sorted_neuron_importance, sorted_neuron_indices = intermediate_l2_score[layer].sort(descending=False)
        head_mask[layer, sorted_head_indices[:cofi_head_pruned_num[layer].int().item()]] = 0
        intermediate_mask[layer, sorted_neuron_indices[:cofi_intermediate_pruned_num[layer].int().item()]] = 0
    
    cofi_hidden_pruned_num = (1 - cofi_hidden_mask).sum().int().item()
    hidden_l2_score = hidden_grads.pow(2).sum(dim=0)
    hidden_mask = torch.ones_like(cofi_hidden_mask)
    sorted_hidden_importance, sorted_hidden_indices = hidden_l2_score.sort(descending=False)
    hidden_mask[sorted_hidden_indices[:cofi_hidden_pruned_num]] = 0
        
    torch.save(head_mask, 'output/adap_mask/head_mask.pt')
    torch.save(intermediate_mask, 'output/adap_mask/intermediate_mask.pt')
    torch.save(hidden_mask, 'output/adap_mask/hidden_mask.pt')
    
    # Load prune-and-restore masks and compare to CoFi's final masks
    cofi_zs = torch.load('/home/zbw/projects/CoFiPruning/out/bert-base/MNLI/CoFi/MNLI_sparsity0.60_lora/zs.pt')
    head_mask, intermediate_mask, hidden_mask = parse_cofi_zs(cofi_zs)
    mask_history = load_gen_mask_with_restoration('output/bert-base-uncased_lora_minus_mnli_cubic_gradual_global_none_nodistill_restore/mac0.4/epoch20/bz32/warmup5/paramq:0-11,v:0-11,i:0-11/lora_r8/lora_alpha16')
    head_similarity_with_cofi = [mask_similarity(head_mask, v['head_mask'])[0] for v in mask_history]
    intermediate_similarity_with_cofi = [mask_similarity(intermediate_mask, v['intermediate_mask'])[0] for v in mask_history]