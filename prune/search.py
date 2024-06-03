import torch

from typing import Optional, Union, List
from utils.fisher_utils.efficiency.mac import compute_mac, compute_encoder_decoder_mac, mac_per_head, mac_per_neuron, mac_per_cross_head, mac_per_hidden_dim
from utils.fisher_utils.efficiency.latency import estimate_latency, fit_latency_fn

def distribute_integer(m, ratios, n):
    # Calculate the total sum of the ratios
    total_ratio = sum(ratios)
    initial_distribution = [min(n, int(round(m * ratio / total_ratio))) for ratio in ratios]
    remaining = m - sum(initial_distribution)

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

# TODO: Using DP-like binary knapsack solver to find the optimal head and neuron pruning, instead of exhaustive search
@torch.no_grad()
def search_mac(
    config,
    head_importance,
    neuron_importance,
    seq_len,
    mac_constraint,
    gated=False,
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
        gated=gated,
    )
    max_mac = mac_constraint * original_mac
    is_mask_matched = isinstance(head_importance, torch.Tensor) and isinstance(neuron_importance, torch.Tensor)
    device = head_importance.device if is_mask_matched else head_importance[0].device
    # Globally rank heads and neurons
    if is_mask_matched:
        sorted_head_importance, sorted_head_indicies = head_importance.view(-1).sort(descending=True)
        sorted_neuron_importance, sorted_neuron_indicies = neuron_importance.view(-1).sort(descending=True)
    else:
        assert isinstance(head_importance, list) and isinstance(neuron_importance, list)
        assert isinstance(head_importance[0], torch.Tensor) and isinstance(neuron_importance[0], torch.Tensor)
        concatenated_head_importance = torch.cat(head_importance, dim=0)
        concatenated_neuron_importance = torch.cat(neuron_importance, dim=0)
        sorted_head_importance, sorted_head_indicies = concatenated_head_importance.sort(descending=True)
        sorted_neuron_importance, sorted_neuron_indicies = concatenated_neuron_importance.sort(descending=True)

    total_head_num = head_importance.numel() if is_mask_matched else sum([h.numel() for h in head_importance])
    total_neuron_num = neuron_importance.numel() if is_mask_matched else sum([n.numel() for n in neuron_importance])

    max_importance = - float('inf')
    for num_heads in range(1, total_head_num + 1):
        heads_mac = mac_per_head(seq_len, hidden_size, attention_head_size) * num_heads
        neurons_mac = max_mac - heads_mac
        num_neurons = int(neurons_mac / mac_per_neuron(seq_len, hidden_size, gated=gated))
        num_neurons = max(num_neurons, 0)

        total_importance = sorted_head_importance[:num_heads].sum() + sorted_neuron_importance[:num_neurons].sum()
        if total_importance > max_importance:
            max_importance = total_importance
            head_indicies = sorted_head_indicies[:num_heads]
            neuron_indicies = sorted_neuron_indicies[:num_neurons]
    
    head_mask = torch.zeros(total_head_num).to(device)
    neuron_mask = torch.zeros(total_neuron_num).to(device)
    head_mask[head_indicies] = 1.0
    neuron_mask[neuron_indicies] = 1.0
    if is_mask_matched:
        head_mask = head_mask.view(num_hidden_layers, num_attention_heads)
        neuron_mask = neuron_mask.view(num_hidden_layers, intermediate_size)
    else:
        head_mask = list(torch.split(head_mask, [h.numel() for h in head_importance]))
        neuron_mask = list(torch.split(neuron_mask, [n.numel() for n in neuron_importance]))

    return head_mask, neuron_mask

@torch.no_grad()
def search_mac_reverse(
    config,
    head_importance,
    neuron_importance,
    seq_len,
    mac_constraint,
    head_mask_condition: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    neuron_mask_condition: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
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

    if isinstance(head_mask_condition, torch.Tensor):
        invalid_pruning_head_indices = head_mask_condition.view(-1).nonzero(as_tuple=False).squeeze()
    else:
        invalid_pruning_head_indices = torch.cat(head_mask_condition).nonzero(as_tuple=False).squeeze()
    if isinstance(neuron_mask_condition, torch.Tensor):
        invalid_pruning_neuron_indices = neuron_mask_condition.view(-1).nonzero(as_tuple=False).squeeze()
    else:
        invalid_pruning_neuron_indices = torch.cat(neuron_mask_condition).nonzero(as_tuple=False).squeeze()

    if head_mask_condition is not None and neuron_mask_condition is not None:
        concatenated_head_importance[invalid_pruning_head_indices] = float('inf')
        concatenated_neuron_importance[invalid_pruning_neuron_indices] = float('inf')    

    sorted_head_importance, sorted_head_indicies = concatenated_head_importance.sort(descending=False)
    sorted_neuron_importance, sorted_neuron_indicies = concatenated_neuron_importance.sort(descending=False)
    total_head_num = head_importance.numel() if is_mask_matched else sum([h.numel() for h in head_importance])
    total_neuron_num = neuron_importance.numel() if is_mask_matched else sum([n.numel() for n in neuron_importance])

    min_importance = float('inf')
    for num_pruned_heads in range(total_head_num + 1):
        heads_mac = mac_per_head(seq_len, hidden_size, attention_head_size) * (total_head_num - num_pruned_heads)
        neurons_mac = max_mac - heads_mac
        num_pruned_neurons = total_neuron_num - int(neurons_mac / mac_per_neuron(seq_len, hidden_size))
        num_pruned_neurons = max(num_pruned_neurons, 0)

        total_importance = sorted_head_importance[:num_pruned_heads].sum() + sorted_neuron_importance[:num_pruned_neurons].sum()
        if total_importance < min_importance:
            min_importance = total_importance
            head_indicies = sorted_head_indicies[:num_pruned_heads]
            neuron_indicies = sorted_neuron_indicies[:num_pruned_neurons]
    
    head_mask = torch.ones(total_head_num).to(device)
    neuron_mask = torch.ones(total_neuron_num).to(device)
    head_mask[head_indicies] = 0.0
    neuron_mask[neuron_indicies] = 0.0
    if is_mask_matched:
        head_mask = head_mask.view(num_hidden_layers, num_attention_heads)
        neuron_mask = neuron_mask.view(num_hidden_layers, intermediate_size)
    else:
        head_mask = list(torch.split(head_mask, [h.numel() for h in head_importance]))
        neuron_mask = list(torch.split(neuron_mask, [n.numel() for n in neuron_importance]))

    return head_mask, neuron_mask

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
    total_head_num = head_importance.numel() if is_mask_matched else sum([h.numel() for h in head_importance])
    total_neuron_num = neuron_importance.numel() if is_mask_matched else sum([n.numel() for n in neuron_importance])
    
    # Calculate layer-level importance for top-down pruning
    head_layer_importance, neuron_layer_importance = head_importance.sum(dim=1), neuron_importance.sum(dim=1)
    smoothed_head_importance = head_layer_importance + head_layer_importance.mean()
    smoothed_neuron_importance = neuron_layer_importance + neuron_layer_importance.mean()
    smoothed_relative_head_importance = smoothed_head_importance / smoothed_head_importance.sum()
    smoothed_relative_neuron_importance = smoothed_neuron_importance / smoothed_neuron_importance.sum()
    sorted_head_importance_perlayer, sorted_head_indices_perlayer = head_importance.sort(dim=1, descending=True)
    sorted_neuron_importance_perlayer, sorted_neuron_indices_perlayer = neuron_importance.sort(dim=1, descending=True)
    
    if hidden_importance is not None:
        sorted_hidden_importance, sorted_hidden_indicies = hidden_importance.sort(descending=False)
        total_hidden_num = hidden_importance.numel()
        
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


@torch.no_grad()
def search_encoder_decoder_mac(
    config,
    head_importance,
    intermediate_importance,
    input_seq_len,
    output_seq_len,
    mac_constraint,
    gated=True,
):
    assert mac_constraint < 1

    num_encoder_layers, num_decoder_layers = config.num_layers, config.num_decoder_layers
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.d_ff
    hidden_size = config.hidden_size
    attention_head_size = int(hidden_size / num_attention_heads)

    original_mac = compute_encoder_decoder_mac(
        [[num_attention_heads] * num_encoder_layers, [num_attention_heads] * num_decoder_layers, [num_attention_heads] * num_decoder_layers],
        [[intermediate_size] * num_encoder_layers, [intermediate_size] * num_decoder_layers],
        input_seq_len,
        output_seq_len,
        hidden_size,
        attention_head_size,
        gated=gated,
    )
    max_mac = mac_constraint * original_mac
    mac_per_encoder_self_head = mac_per_head(input_seq_len, hidden_size, attention_head_size)
    mac_per_decoder_self_head = mac_per_head(output_seq_len, hidden_size, attention_head_size)
    mac_per_decoder_cross_head = mac_per_cross_head(input_seq_len, output_seq_len, hidden_size, attention_head_size)
    mac_per_encoder_neuron = mac_per_neuron(input_seq_len, hidden_size, gated=gated)
    mac_per_decoder_neuron = mac_per_neuron(output_seq_len, hidden_size, gated=gated)

    # Globally rank heads and neurons
    encoder_self_head_scores, decoder_self_head_scores, decoder_cross_head_scores = head_importance
    encoder_intermediate_scores, decoder_intermediate_scores = intermediate_importance
    encoder_self_head_importance, sorted_encoder_self_head_indicies = encoder_self_head_scores.view(-1).sort(descending=True)
    decoder_self_head_importance, sorted_decoder_self_head_indicies = decoder_self_head_scores.view(-1).sort(descending=True)
    decoder_cross_head_importance, sorted_decoder_cross_head_indicies = decoder_cross_head_scores.view(-1).sort(descending=True)
    encoder_intermediate_importance, sorted_encoder_intermediate_indicies = encoder_intermediate_scores.view(-1).sort(descending=True)
    decoder_intermediate_importance, sorted_decoder_intermediate_indicies = decoder_intermediate_scores.view(-1).sort(descending=True)

    max_importance = -float('inf')
    for num_heads in range(1, num_encoder_layers * num_attention_heads + 1):
        heads_mac = (mac_per_encoder_self_head + mac_per_decoder_self_head + mac_per_decoder_cross_head) * num_heads
        neurons_mac = max_mac - heads_mac
        if neurons_mac < 0:
            break
        num_neurons = int(neurons_mac / (mac_per_encoder_neuron + mac_per_decoder_neuron))
        num_neurons = max(num_neurons, 0)

        total_importance = encoder_self_head_importance[:num_heads].sum() +  decoder_self_head_importance[:num_heads].sum() +  decoder_cross_head_importance[:num_heads].sum() + encoder_intermediate_importance[:num_neurons].sum() + decoder_intermediate_importance[:num_neurons].sum()
        if total_importance > max_importance:
            max_importance = total_importance
            encoder_self_head_indices = sorted_encoder_self_head_indicies[:num_heads]
            decoder_self_head_indices = sorted_decoder_self_head_indicies[:num_heads]
            decoder_cross_head_indices = sorted_decoder_cross_head_indicies[:num_heads]
            encoder_intermediate_indices = sorted_encoder_intermediate_indicies[:num_neurons]
            decoder_intermediate_indices = sorted_decoder_intermediate_indicies[:num_neurons]

    head_mask = torch.zeros_like(head_importance)
    head_mask[0].view(-1)[encoder_self_head_indices] = 1.0
    head_mask[1].view(-1)[decoder_self_head_indices] = 1.0
    head_mask[2].view(-1)[decoder_cross_head_indices] = 1.0

    neuron_mask = torch.zeros_like(intermediate_importance)
    neuron_mask[0].view(-1)[encoder_intermediate_indices] = 1.0
    neuron_mask[1].view(-1)[decoder_intermediate_indices] = 1.0

    return head_mask, neuron_mask


@torch.no_grad()
def search_latency(
    config,
    head_importance,
    neuron_importance,
    latency_constraint,
    mha_lut,
    ffn_lut,
):
    assert latency_constraint < 1

    num_hidden_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size

    original_latency = estimate_latency(
        mha_lut,
        ffn_lut,
        torch.ones(num_hidden_layers, num_attention_heads),
        torch.ones(num_hidden_layers, intermediate_size),
    )
    max_latency = latency_constraint * original_latency

    mha_latency_fn = fit_latency_fn(mha_lut)
    ffn_latency_fn = fit_latency_fn(ffn_lut)

    # Locally rank heads and neurons
    local_head_importance, local_head_indicies = head_importance.sort(dim=1, descending=True)
    local_neuron_importance, local_neuron_indicies = neuron_importance.sort(dim=1, descending=True)

    base_head_indicies = local_head_indicies[:, :mha_latency_fn.threshold]
    base_neuron_indicies = local_neuron_indicies[:, :ffn_latency_fn.threshold]

    base_head_importance = local_head_importance[:, :mha_latency_fn.threshold].sum(dim=1)
    base_neuron_importance = local_neuron_importance[:, :ffn_latency_fn.threshold].sum(dim=1)

    _, mha_sorted_indicies = base_head_importance.sort(descending=False)
    _, ffn_sorted_indicies = base_neuron_importance.sort(descending=False)

    mha_offset = torch.arange(0, end=num_hidden_layers).unsqueeze(1).cuda() * num_attention_heads
    ffn_offset = torch.arange(0, end=num_hidden_layers).unsqueeze(1).cuda() * intermediate_size

    base_head_indicies = base_head_indicies + mha_offset
    base_neuron_indicies = base_neuron_indicies + ffn_offset

    orig_neuron_importance = neuron_importance.clone()
    max_importance = -float('inf')
    for num_mha_drops in range(num_hidden_layers):
        head_importance[mha_sorted_indicies[:num_mha_drops]] = 0
        remaining_base_head_indicies = base_head_indicies[mha_sorted_indicies[num_mha_drops:]].flatten()
        num_mha_layers = num_hidden_layers - num_mha_drops

        neuron_importance = orig_neuron_importance.clone()
        for num_ffn_drops in range(num_hidden_layers):
            neuron_importance[ffn_sorted_indicies[:num_ffn_drops]] = 0
            remaining_base_neuron_indicies = base_neuron_indicies[ffn_sorted_indicies[num_ffn_drops:]].flatten()
            num_ffn_layers = num_hidden_layers - num_ffn_drops

            remaining_head_indicies = local_head_indicies[:, mha_latency_fn.threshold:]
            remaining_neuron_indicies = local_neuron_indicies[:, ffn_latency_fn.threshold:]

            remaining_head_importance = head_importance.gather(dim=1, index=remaining_head_indicies)
            remaining_neuron_importance = neuron_importance.gather(dim=1, index=remaining_neuron_indicies)

            # Globally rank the remaining heads and neurons
            sorted_head_importance, sorted_head_indicies = remaining_head_importance.view(-1).sort(descending=True)
            sorted_neuron_importance, sorted_neuron_indicies = remaining_neuron_importance.view(-1).sort(descending=True)

            max_latency = max_latency - (num_mha_layers * mha_latency_fn.c + num_ffn_layers * ffn_latency_fn.c)
            if max_latency < 0:
                continue

            num_remaining_heads = num_mha_layers * (num_attention_heads - mha_latency_fn.threshold)
            for num_heads in range(1, num_remaining_heads + 1):
                heads_latency = mha_latency_fn.slope * num_heads
                neurons_latency = max_latency - heads_latency
                num_neurons = int(neurons_latency / ffn_latency_fn.slope)
                if num_neurons < 0:
                    break

                total_importance = sorted_head_importance[:num_heads].sum() + sorted_neuron_importance[:num_neurons].sum()
                if total_importance > max_importance:
                    max_importance = total_importance

                    head_indicies = sorted_head_indicies[:num_heads]
                    head_indicies = (remaining_head_indicies + mha_offset).flatten()[head_indicies]
                    head_indicies = torch.cat([remaining_base_head_indicies, head_indicies], dim=0)

                    neuron_indicies = sorted_neuron_indicies[:num_neurons]
                    neuron_indicies = (remaining_neuron_indicies + ffn_offset).flatten()[neuron_indicies]
                    neuron_indicies = torch.cat([remaining_base_neuron_indicies, neuron_indicies], dim=0)

    head_mask = torch.zeros(num_hidden_layers, num_attention_heads).flatten()
    head_mask[head_indicies] = 1.0
    head_mask = head_mask.view(num_hidden_layers, num_attention_heads)

    neuron_mask = torch.zeros(num_hidden_layers, intermediate_size).flatten()
    neuron_mask[neuron_indicies] = 1.0
    neuron_mask = neuron_mask.view(num_hidden_layers, intermediate_size)

    head_mask = head_mask.cuda()
    neuron_mask = neuron_mask.cuda()
    return head_mask, neuron_mask
