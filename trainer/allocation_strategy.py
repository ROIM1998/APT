import torch

from typing import Dict, List
from prune.fisher import collect_grads_by_suffix
from ortools.algorithms import pywrapknapsack_solver

def binary_knapsack_search(values_tensor: torch.Tensor, weights_tensor: torch.Tensor, capacities: List[int]) -> torch.Tensor:
    sorted_values, sorted_indices = torch.sort(values_tensor, descending=True)
    # Binary search for the best solution
    max_i, min_i = len(sorted_values), 0
    next_i = len(sorted_values) // 2
    while max_i - min_i > 1:
        selected_indices = sorted_indices[:next_i]
        weights_sum = torch.index_select(weights_tensor, 0, selected_indices).sum()
        if weights_sum < capacities[0]:
            min_i = next_i
        else:
            max_i = next_i
        next_i = (max_i + min_i) // 2
    if weights_sum > capacities[0]:
        next_i -= 1
    print(f"Total capacity: {weights_tensor.sum().item()}. Target capacity: {capacities[0]}, selected capacity: {weights_sum}")
    dim_masks = torch.zeros(len(sorted_values))
    dim_masks[sorted_indices[:next_i]] = 1
    dim_masks = dim_masks.int()
    return dim_masks

#  TODO: current strategy issue: bottleneck dimension explosion
def deprecated_adjust_r_then_shrink_inout(model, adapter_pruner, current_tuning_param_num, target_tuning_param_num, bottleneck_prune_ratio, dependent: bool = True, *args, **kwargs):
    model.eval()
    # Firstly, calculate the current tuning parameter status compared with the target tuning parameter number
    current_tuning_param_num = 0
    named_modules = dict(model.named_modules())
    for name, module in named_modules.items():
        if hasattr(module, 'bottleneck_mask'):
            # tuning layer
            current_tuning_param_num += module.bottleneck_mask.sum().item() * (module.in_features + module.out_features)
    
    bottleneck_names, all_bottleneck_mask, all_bottleneck_grads = adapter_pruner.prune_by_suffix(bottleneck_prune_ratio, '.bottleneck_mask')
    pruning_bottleneck_dim_num = sum([(1 - v).sum() for v in all_bottleneck_mask]).item()
    extra_bottleneck_dims = sum([v.all().int() * v.numel() for v in all_bottleneck_mask]).item()
    if extra_bottleneck_dims == 0:
        expanding_scale = 1
    else:
        expanding_scale = int(pruning_bottleneck_dim_num // extra_bottleneck_dims + 2)
    print(f'expanding_scale: {expanding_scale}')
    target_lora_rs = {
        k.rsplit('.', 1)[0]: int(expanding_scale * m.numel()) if m.all() else int(m.sum().item())
        for k, m in zip(bottleneck_names, all_bottleneck_mask)
    }

    # Not set the masks back to the layers as an independent bottleneck-output pruning setting
    if dependent:
        for name, mask in zip(bottleneck_names, all_bottleneck_mask):
            layer_name = name.rsplit('.', 1)[0]
            named_modules[layer_name].bottleneck_mask = torch.nn.Parameter(mask, requires_grad=False)
    output_grads = collect_grads_by_suffix(model, adapter_pruner.dataloader, '.output_mask')
    output_scores = {k: v.pow(2).sum(dim=0) for k, v in output_grads.items() if 'output_mask' in k}
    input_grads = collect_grads_by_suffix(model, adapter_pruner.dataloader, '.input_mask')
    input_scores = {k: v.pow(2).sum(dim=0) for k, v in input_grads.items() if 'input_mask' in k}

    weights = [[int(target_lora_rs[k.rsplit('.', 1)[0]]) for k in output_scores for _ in range(output_scores[k].numel())] + [int(target_lora_rs[k.rsplit('.', 1)[0]]) for k in input_scores for _ in range(input_scores[k].numel())]]
    lens = [len(output_scores[k]) for k in output_scores] + [len(input_scores[k]) for k in input_scores]
    capacities = [target_tuning_param_num]
    # Using our customized search function (solve it quicker instead of better)
    values_tensor = torch.cat([v for v in output_scores.values()] + [v for v in input_scores.values()]).cpu()
    weights_tensor = torch.tensor(weights[0])
    sorted_values, sorted_indices = torch.sort(values_tensor, descending=True)
    # Binary search for the best solution
    max_i, min_i = len(sorted_values), 0
    next_i = len(sorted_values) // 2
    while max_i - min_i > 1:
        selected_indices = sorted_indices[:next_i]
        weights_sum = torch.index_select(weights_tensor, 0, selected_indices).sum()
        if weights_sum < capacities[0]:
            min_i = next_i
        else:
            max_i = next_i
        next_i = (max_i + min_i) // 2
    if weights_sum > capacities[0]:
        next_i -= 1
    dim_masks = torch.zeros(len(sorted_values))
    dim_masks[sorted_indices[:next_i]] = 1
    dim_masks = dim_masks.int().tolist()
    splitted_dim_masks = torch.split(torch.tensor(dim_masks), lens)
    output_dim_masks = splitted_dim_masks[:len(output_scores)]
    input_dim_masks = splitted_dim_masks[len(output_scores):]
    return bottleneck_names, all_bottleneck_mask, output_dim_masks, input_dim_masks, target_lora_rs

def adjust_r_then_shrink_inout(model, adapter_pruner, current_tuning_param_num, target_tuning_param_num, bottleneck_prune_ratio, dependent: bool = True, *args, **kwargs):
    model.eval()
    all_grads = collect_grads_by_suffix(model, adapter_pruner.dataloader, '.bottleneck_mask')
    bottleneck_names, all_bottleneck_grads = list(all_grads.keys()), list(all_grads.values())
    # Try adding pseudo bottleneck dimensions to expand some of the layers
    relative_ratio = current_tuning_param_num / target_tuning_param_num
    if relative_ratio > 1:
        decay_factor = (1 / relative_ratio) ** 0.5
    else:
        decay_factor = (1 / relative_ratio) ** 1.5
    # Add pseudo bottleneck dimensions to the layers, with their scores set as the average of the corresponding layers' scores
    # Only adding pseudo bottleneck dimensions if all of the current dimensions are relatively high
    all_bottleneck_scores = {k: v.pow(2).sum(dim=0) for k, v in all_grads.items()}
    # We penalize the pseudo bottleneck dimensions by setting their scores as the average of the corresponding layers' scores minus the standard deviation
    # Edit: get the log score first, get the mean minus std, then get the exp score
    all_bottleneck_scores = {
        k: torch.cat([v, torch.tensor([(v.log().mean() - v.log().std()).exp()] * int(v.numel() * decay_factor), dtype=v.dtype, device=v.device)])
        for k, v in all_bottleneck_scores.items()
    }
    # Using a single binary knapsack search to find the optimal bottleneck allocation
    named_modules = dict(model.named_modules())
    weights_tensor = torch.tensor([int(named_modules[k.rsplit('.', 1)[0]].in_features + named_modules[k.rsplit('.', 1)[0]].out_features) for k in all_bottleneck_scores for _ in range(all_bottleneck_scores[k].numel())])
    lens = [len(all_bottleneck_scores[k]) for k in all_bottleneck_scores]
    capacities = [int(current_tuning_param_num * decay_factor)]
    # Using our customized search function (solve it quicker instead of better)
    values_tensor = torch.cat([v for v in all_bottleneck_scores.values()]).cpu()
    all_pseudo_bottleneck_mask = binary_knapsack_search(values_tensor, weights_tensor, capacities)
    all_pseudo_bottleneck_mask = torch.split(all_pseudo_bottleneck_mask, lens)
    target_lora_rs = {
        k.rsplit('.', 1)[0]: int(m.sum().item())
        for k, m in zip(bottleneck_names, all_pseudo_bottleneck_mask)
    }
    # Then fix the bottleneck dimensions and search for the input-output dimension masks
    # Set the masks back to get the dependent results
    new_bottleneck_masks = []
    for original_grads, pseudo_mask in zip(all_bottleneck_grads, all_pseudo_bottleneck_mask):
        original_length = original_grads.shape[1]
        mask = torch.zeros(original_length).to(original_grads.device)
        mask[pseudo_mask[:original_length].nonzero().squeeze()] = 1
        new_bottleneck_masks.append(mask)
    dependent=True
    if dependent:
        for name, mask in zip(bottleneck_names, new_bottleneck_masks):
            layer_name = name.rsplit('.', 1)[0]
            named_modules[layer_name].bottleneck_mask = torch.nn.Parameter(mask, requires_grad=False)
    output_grads = collect_grads_by_suffix(model, adapter_pruner.dataloader, '.output_mask')
    output_scores = {k: v.pow(2).sum(dim=0) for k, v in output_grads.items() if 'output_mask' in k}
    input_grads = collect_grads_by_suffix(model, adapter_pruner.dataloader, '.input_mask')
    input_scores = {k: v.pow(2).sum(dim=0) for k, v in input_grads.items() if 'input_mask' in k}
    weights = torch.tensor([int(target_lora_rs[k.rsplit('.', 1)[0]]) for k in output_scores for _ in range(output_scores[k].numel())] + [int(target_lora_rs[k.rsplit('.', 1)[0]]) for k in input_scores for _ in range(input_scores[k].numel())])
    values_tensor = torch.cat([v for v in output_scores.values()] + [v for v in input_scores.values()]).cpu()
    lens = [len(output_scores[k]) for k in output_scores] + [len(input_scores[k]) for k in input_scores]
    capacities = [target_tuning_param_num]
    dim_masks = binary_knapsack_search(values_tensor, weights, capacities)
    dim_masks = dim_masks.int().tolist()
    splitted_dim_masks = torch.split(torch.tensor(dim_masks), lens)
    output_dim_masks = splitted_dim_masks[:len(output_scores)]
    input_dim_masks = splitted_dim_masks[len(output_scores):]
    all_scores = {
        'bottleneck': all_bottleneck_scores,
        'output': output_scores,
        'input': input_scores
    }
    return bottleneck_names, new_bottleneck_masks, output_dim_masks, input_dim_masks, target_lora_rs, all_scores
    

def adjust_r_with_static_inout(model, adapter_pruner, current_tuning_param_num, target_tuning_param_num, bottleneck_prune_ratio, *args, **kwargs):
    model.eval()
    named_modules = dict(model.named_modules())
    bottleneck_names, all_bottleneck_mask, all_bottleneck_grads = adapter_pruner.prune_by_suffix(bottleneck_prune_ratio, '.bottleneck_mask')
    inout_info = {
        k: {
            'in': named_modules[k.rsplit('.', 1)[0]].in_features,
            'out': named_modules[k.rsplit('.', 1)[0]].out_features,
            'all': named_modules[k.rsplit('.', 1)[0]].in_features + named_modules[k.rsplit('.', 1)[0]].out_features,
            'score': all_bottleneck_grads[k].sum().item(),
        }
        for k in bottleneck_names
    }
    
    tuning_param_num_after_r_adjust = sum([v.sum().item() * inout_info[k]['all'] for k, v in zip(bottleneck_names, all_bottleneck_mask)])
    valid_expanding_param_num = target_tuning_param_num - tuning_param_num_after_r_adjust
    bottleneck_scores = {k: v.pow(2).sum(dim=0) for k, v in all_bottleneck_grads.items() if 'bottleneck_mask' in k}
    if valid_expanding_param_num < 0:
        extra_param_num = -valid_expanding_param_num
        print(f"{extra_param_num} parameters extra after dimension restoration. Even the in-out- channels and several bottleneck dimensions are pruned. Further decaying the bottleneck dimensions to meet the constraint.")
        # Using a single binary knapsack search to find the optimal bottleneck allocation
        weights_tensor = torch.tensor([int(inout_info['all']) for k in bottleneck_scores for _ in range(bottleneck_scores[k].numel())])
        lens = [len(bottleneck_scores[k]) for k in bottleneck_scores]
        capacities = [target_tuning_param_num]
        # Using our customized search function (solve it quicker instead of better)
        values_tensor = torch.cat([v for v in bottleneck_scores.values()]).cpu()
        all_bottleneck_mask = binary_knapsack_search(values_tensor, weights_tensor, capacities)
        all_bottleneck_mask = torch.split(all_bottleneck_mask, lens)
        target_lora_rs = {
            k.rsplit('.', 1)[0]: int(m.sum().item())
            for k, m in zip(bottleneck_names, all_bottleneck_mask)
        }
    else:
        added_r = valid_expanding_param_num // sum([inout_info[k]['all'] for k, mask in zip(bottleneck_names, all_bottleneck_mask) if mask.all() and inout_info[k]['all'] > 0])

        # Now calculating the expanded r targets that follows the constraint of valid_expanding_param_num
        target_lora_rs = {
            k.rsplit('.', 1)[0]: int(m.numel() + added_r) if m.all() and inout_info[k]['all'] > 0 else int(m.sum().item())
            for k, m in zip(bottleneck_names, all_bottleneck_mask)
        }

    # Directly set input and output masks as ones
    output_dim_masks = [torch.ones(inout_info[k]['out']) for k in bottleneck_names]
    input_dim_masks = [torch.ones(inout_info[k]['in']) for k in bottleneck_names]
    all_scores = {
        'bottleneck': bottleneck_scores,
    }
    return bottleneck_names, all_bottleneck_mask, output_dim_masks, input_dim_masks, target_lora_rs, all_scores


def adjust_r_with_preserved_inout(model, adapter_pruner, current_tuning_param_num, target_tuning_param_num, bottleneck_prune_ratio, final_head_mask, final_intermediate_mask, dependent=True, *args, **kwargs):
    model.eval()
    named_modules = dict(model.named_modules())
    named_masks: Dict[str, torch.Tensor] = model.named_masks(final_head_mask, final_intermediate_mask)
    hidden_per_head = model.config.hidden_size // model.config.num_attention_heads

    # Get the dependent bottleneck score if needed
    if dependent:
        model.head_mask, model.intermediate_mask = final_head_mask, final_intermediate_mask
    bottleneck_names, all_bottleneck_mask, all_bottleneck_grads = adapter_pruner.prune_by_suffix(bottleneck_prune_ratio, '.bottleneck_mask')
    model.head_mask, model.intermediate_mask = None, None
    inout_info = {
        k: {
            'in': named_modules[k.rsplit('.', 1)[0]].in_features,
            'out': named_modules[k.rsplit('.', 1)[0]].out_features,
            'all': named_modules[k.rsplit('.', 1)[0]].in_features + named_modules[k.rsplit('.', 1)[0]].out_features,
            'score': all_bottleneck_grads[k].sum().item(),
        }
        for k in bottleneck_names
    }

    # Set the input and output dimensions according to the final head and intermediate masks
    # TODO: support the case with output_dynamic=False, meaning that the input dimension can be pruned throughout the training process
    output_dim_masks = [named_masks[k.rsplit('.', 1)[0]] for k in bottleneck_names]
    input_dim_masks = [torch.ones(inout_info[k]['in']) for k in bottleneck_names]
    
    tuning_param_num_after_adjust = 0
    for k, v, out_mask, in_mask in zip(bottleneck_names, all_bottleneck_mask, output_dim_masks, input_dim_masks):
        if out_mask is not None and in_mask is not None and out_mask.any() and in_mask.any():
            tuning_param_num_after_adjust += v.sum().item() * (out_mask.sum().item() + in_mask.sum().item()) 
            inout_info[k]['in'] = in_mask.sum().item()
            inout_info[k]['out'] = out_mask.sum().item()
            inout_info[k]['all'] = inout_info[k]['in'] + inout_info[k]['out']
        else:
            inout_info[k]['in'] = 0
            inout_info[k]['out'] = 0
            inout_info[k]['all'] = 0
    
    valid_expanding_param_num = target_tuning_param_num - tuning_param_num_after_adjust
    bottleneck_scores = {k: v.pow(2).sum(dim=0) for k, v in all_bottleneck_grads.items() if 'bottleneck_mask' in k}
    if valid_expanding_param_num < 0:
        extra_param_num = -valid_expanding_param_num
        print(f"{extra_param_num} parameters extra after dimension restoration. Even the in-out- channels and several bottleneck dimensions are pruned. Further decaying the bottleneck dimensions to meet the constraint.")
        # Using a single binary knapsack search to find the optimal bottleneck allocation
        weights_tensor = torch.tensor([int(inout_info['all']) for k in bottleneck_scores for _ in range(bottleneck_scores[k].numel())])
        lens = [len(bottleneck_scores[k]) for k in bottleneck_scores]
        capacities = [target_tuning_param_num]
        # Using our customized search function (solve it quicker instead of better)
        values_tensor = torch.cat([v for v in bottleneck_scores.values()]).cpu()
        all_bottleneck_mask = binary_knapsack_search(values_tensor, weights_tensor, capacities)
        all_bottleneck_mask = torch.split(all_bottleneck_mask, lens)
        target_lora_rs = {
            k.rsplit('.', 1)[0]: int(m.sum().item())
            for k, m in zip(bottleneck_names, all_bottleneck_mask)
        }
    else:
        added_r = valid_expanding_param_num // sum([inout_info[k]['all'] for k, mask in zip(bottleneck_names, all_bottleneck_mask) if mask.all() and inout_info[k]['all'] > 0])

        # Now calculating the expanded r targets that follows the constraint of valid_expanding_param_num
        target_lora_rs = {
            k.rsplit('.', 1)[0]: int(m.numel() + added_r) if m.all() and inout_info[k]['all'] > 0 else int(m.sum().item())
            for k, m in zip(bottleneck_names, all_bottleneck_mask)
        }
    all_scores = {
        'bottleneck': bottleneck_scores,
    }
    return bottleneck_names, all_bottleneck_mask, output_dim_masks, input_dim_masks, target_lora_rs, all_scores