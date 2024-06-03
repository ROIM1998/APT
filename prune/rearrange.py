import torch
from typing import Tuple, Dict, Union, List
from utils.minus_utils import sum_fisher_score

@torch.no_grad()
def greedy_rearrange(mask, scores, grads):
    # Score dimension: [num_masks]
    # Grads dimension: [num_masks, num_batch]
    num_unpruned = int(mask.sum())
    num_pruned = mask.shape[0] - num_unpruned
    if num_unpruned == 0 or num_pruned == 0:
        return mask

    _, indicies = scores.sort(descending=False)
    indicies = indicies.tolist()

    # Greedy search
    masked_indicies = indicies[:num_pruned]
    for index in indicies[num_pruned:]:
        masked_indicies.append(index)
        grad_vectors = grads[masked_indicies]
        grad_sum = grad_vectors.sum(dim=0)
        # grad_sum dimension: [num_batches]
        complement = grad_sum - grad_vectors
        # complement dimension: [num_pruned + 1, num_batches]
        grad_sum_length = complement.pow(2).sum(dim=1)

        removed = grad_sum_length.argmin()
        del masked_indicies[removed]

    new_mask = torch.ones_like(mask)
    new_mask[masked_indicies] = 0
    return new_mask

@torch.no_grad()
def better_greedy_rearrange(mask: torch.Tensor, grads: torch.Tensor, debug: bool = False) -> torch.Tensor:
    # TODO: change it to beam search for better outcome
    # Mask dimension: [num_masks]
    # Grads dimension: [num_masks, num_batch]
    num_unpruned = int(mask.sum())
    num_pruned = mask.shape[0] - num_unpruned
    if num_unpruned == 0 or num_pruned == 0:
        return mask
    
    masked = ((mask == 0).nonzero().squeeze().tolist())
    retained = (mask == 1).nonzero().squeeze().tolist()
    if isinstance(masked, int):
        masked = [masked]
    if isinstance(retained, int):
        retained = [retained]
    self_score = grads.pow(2).sum(dim=1)
    masked_chunk_sum = grads[masked].sum(dim=0)
    searching = True
    if debug:
        print("Initial local fisher score calculated: {}, using chunk calculated: {}".format(grads[masked].sum(dim=0).pow(2).sum(), masked_chunk_sum.pow(2).sum()))
    while searching:
        retained_scores = 2 * torch.matmul(grads[retained], masked_chunk_sum) + self_score[retained]
        add_pos = retained_scores.argmin()
        masked_score = retained_scores[add_pos]
        assert masked_score == retained_scores.min()
        masked_index = retained[add_pos]
        masked.append(masked_index)
        del retained[add_pos]
        masked_chunk_sum += grads[masked_index]
        masked_scores = 2 * torch.matmul(grads[masked], masked_chunk_sum) - self_score[masked]
        remove_pos = masked_scores.argmax()
        unmasked_score = masked_scores[remove_pos]
        assert unmasked_score == masked_scores.max()
        unmasked_index = masked[remove_pos]
        if masked_index == unmasked_index:
            searching = False
        retained.append(unmasked_index)
        del masked[remove_pos]
        masked_chunk_sum -= grads[unmasked_index]
        if debug:
            print("Switching {} to be masked and {} to be retained, local fisher score calculated: {}, using chunk calculated: {}, with masked score {} and unmasked score {}".format(unmasked_index, masked_index, grads[masked].sum(dim=0).pow(2).sum(), masked_chunk_sum.pow(2).sum(), masked_score, unmasked_score))
    new_mask = torch.ones_like(mask)
    new_mask[masked] = 0
    return new_mask

def rearrange_mask(mask, grads, original_grads):
    # NOTE: temporarily convert to CPU tensors as the arithmetic intensity is very low
    device = mask.device
    mask = mask.cpu()
    grads = grads.cpu()
    original_grads = original_grads.cpu()

    if mask.ndim == 2:
        num_hidden_layers = mask.shape[0]
        for i in range(num_hidden_layers):
            mask[i] = greedy_rearrange(mask[i], grads[i, :], original_grads[:, i, :].permute(1, 0))
    elif mask.ndim == 3:
        num_mask_categories, num_hidden_layers = mask.shape[0], mask.shape[1]
        for i in range(num_mask_categories):
            for j in range(num_hidden_layers):
                mask[i, j] = greedy_rearrange(mask[i, j], grads[i, j, :], original_grads[:, i, j, :].permute(1, 0))

    mask = mask.to(device)
    return mask

def better_rearrange_mask(mask, grads):
    # NOTE: temporarily convert to CPU tensors as the arithmetic intensity is very low
    is_mask_matched = isinstance(mask, torch.Tensor)
    ndim = mask.ndim if is_mask_matched else 2 if isinstance(mask[0], torch.Tensor) else 3
    if is_mask_matched:
        device = mask.device
        mask = mask.cpu()
        grads = grads.cpu()
    elif ndim == 2:
        device = mask[0].device
        mask = [m.cpu() for m in mask]
        grads = [g.cpu() for g in grads]
    elif ndim == 3:
        device = mask[0][0].device
        mask = [[m.cpu() for m in mc] for mc in mask]
        grads = [[g.cpu() for g in gc] for gc in grads]
        

    if ndim == 2:
        num_hidden_layers = mask.shape[0] if is_mask_matched else len(mask)
        for i in range(num_hidden_layers):
            mask[i] = better_greedy_rearrange(mask[i], grads[:, i, :].permute(1, 0) if is_mask_matched else grads[i].permute(1, 0))
    elif ndim == 3:
        num_mask_categories, num_hidden_layers = (mask.shape[0], mask.shape[1]) if is_mask_matched else (len(mask), len(mask[0]))
        for i in range(num_mask_categories):
            for j in range(num_hidden_layers):
                mask[i][j] = better_greedy_rearrange(mask[i, j], grads[:, i, j, :].permute(1, 0) if is_mask_matched else grads[i][j].permute(1, 0))

    mask = mask.to(device) if is_mask_matched else [m.to(device) for m in mask] if ndim == 2 else [[m.to(device) for m in mc] for mc in mask]
    return mask


def layer_wise_rearrange_mask(head_mask, intermediate_mask, head_grads, intermediate_grads):
    # NOTE: temporarily convert to CPU tensors as the arithmetic intensity is very low
    is_mask_matched = isinstance(head_mask, torch.Tensor) and isinstance(intermediate_mask, torch.Tensor)
    if is_mask_matched:
        device = head_mask.device
        head_mask, intermediate_mask = head_mask.cpu(), intermediate_mask.cpu()
        head_grads, intermediate_grads = head_grads.cpu(), intermediate_grads.cpu()
        # head_grads *= 12 ** 0.5
        # intermediate_grads *= 3072 ** 0.5
        new_head_mask, new_intermediate_mask = torch.ones_like(head_mask), torch.ones_like(intermediate_mask)
        num_hidden_layers = head_mask.shape[0]
    else:
        device = head_mask[0].device
        head_mask, intermediate_mask = [v.cpu() for v in head_mask], [v.cpu() for v in intermediate_mask]
        # head_grads, intermediate_grads = [v.cpu() * (12 ** 0.5) for v in head_grads], [v.cpu() * (3072 ** 0.5) for v in intermediate_grads]
        new_head_mask, new_intermediate_mask = [torch.ones_like(v) for v in head_mask], [torch.ones_like(v) for v in intermediate_mask]
    
    new_head_mask[0] = better_greedy_rearrange(head_mask[0], head_grads[:, 0, :].permute(1, 0) if is_mask_matched else head_grads[0].permute(1, 0))
    for i in range(2 * num_hidden_layers - 1):
        if i % 2 == 0:
            layer_idx = i // 2
            new_intermediate_mask[layer_idx] = layer_wise_rearrange(
                head_mask[layer_idx],
                intermediate_mask[layer_idx],
                torch.cat([
                    head_grads[:, layer_idx, :] if is_mask_matched else head_grads[layer_idx],
                    intermediate_grads[:, layer_idx, :] if is_mask_matched else intermediate_grads[layer_idx]
                    ], dim=1).permute(1, 0)
                )
        else:
            intermediate_layer_idx = i // 2
            head_layer_idx = i // 2 + 1
            new_head_mask[head_layer_idx] = layer_wise_rearrange(
                intermediate_mask[intermediate_layer_idx],
                head_mask[head_layer_idx],
                torch.cat([
                    intermediate_grads[:, intermediate_layer_idx, :] if is_mask_matched else intermediate_grads[intermediate_layer_idx],
                    head_grads[:, head_layer_idx, :] if is_mask_matched else head_grads[head_layer_idx]
                    ] ,dim=1).permute(1, 0)
                )
    new_head_mask, new_intermediate_mask = new_head_mask.to(device), new_intermediate_mask.to(device)
    return new_head_mask, new_intermediate_mask

def layer_wise_rearrange(fixed_mask: torch.Tensor, tuning_mask: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
    if (tuning_mask == 0).all() or (tuning_mask == 1).all():
        return tuning_mask
    composed_mask = 1 - torch.cat([fixed_mask, tuning_mask], dim=0)
    masked_indices = composed_mask.nonzero().squeeze().tolist()
    retained_indices = (composed_mask==0).nonzero().squeeze().tolist()
    retained_indices = [i for i in retained_indices if i >= fixed_mask.shape[0]]
    possible_masked_indices = [i for i in masked_indices if i >= fixed_mask.shape[0]]
    searching = True
    masked_chunk_sum = grads[masked_indices].sum(dim=0)
    self_score = grads.pow(2).sum(dim=1)
    
    # Greedy search
    # TODO: change it to beam search for better outcome
    while searching:
        masked_score = 2 * torch.matmul(grads[possible_masked_indices], masked_chunk_sum) - self_score[possible_masked_indices]
        unpruned_index = possible_masked_indices[(masked_score).argmax().item()]
        masked_indices.remove(unpruned_index)
        possible_masked_indices.remove(unpruned_index)
        retained_indices.append(unpruned_index)
        masked_chunk_sum -= grads[unpruned_index]
        
        retained_score = 2 * torch.matmul(grads[retained_indices], masked_chunk_sum) + self_score[retained_indices]
        pruned_index = retained_indices[retained_score.argmin().item()]
        if pruned_index == unpruned_index:
            searching = False
        retained_indices.remove(pruned_index)
        masked_indices.append(pruned_index)
        possible_masked_indices.append(pruned_index)
    
    new_mask = torch.zeros_like(tuning_mask)
    new_mask[torch.Tensor(retained_indices).long() - fixed_mask.shape[0]] = 1
    return new_mask


def form_layerwise_chunk(head_score_chunk: torch.Tensor, intermediate_score_chunk: torch.Tensor) -> Dict[str, torch.Tensor]:
    accumulated = (head_score_chunk[:, 0] + intermediate_score_chunk[:, 0]).clone()
    head_layerwise_chunk, intermediate_layerwise_chunk = [], []
    for i in range(head_score_chunk.shape[1] * 2):
        if i % 2 == 0:
            head_layerwise_chunk.append(accumulated.clone())
            if i < head_score_chunk.shape[1] * 2 - 2:
                accumulated += head_score_chunk[:, i // 2 + 1]
            if i > 0:
                accumulated -= intermediate_score_chunk[:, i // 2 - 1]
        else:
            intermediate_layerwise_chunk.append(accumulated.clone())
            accumulated -= head_score_chunk[:, i // 2]
            if i < head_score_chunk.shape[1] * 2 - 1:
                accumulated += intermediate_score_chunk[:, i // 2 + 1]
    return {'head': torch.stack(head_layerwise_chunk, dim=1), 'intermediate': torch.stack(intermediate_layerwise_chunk, dim=1)}


def global_rearrange(mask: Union[List[torch.Tensor], torch.Tensor], grads: Union[List[torch.Tensor], torch.Tensor], debug: bool = False) -> Tuple[Union[List[torch.Tensor], torch.Tensor]]:
    # masks shape: [num_hidden_layers, num_attention_heads or num_intermediate_size]
    # grads shape: [batch_size, num_hidden_layers, num_attention_heads or num_intermediate_size]
    is_mask_matched = isinstance(mask, torch.Tensor)
    if is_mask_matched:
        num_hidden_layers = mask.shape[0] if is_mask_matched else len(mask)
        device = mask.device if is_mask_matched else mask[0].device
        grads = grads.cpu()
        mask = mask.cpu()
        self_score = grads.pow(2).sum(dim=0)
        chunk_sum_by_layer = [(grads[:, i, :] * (1-mask[i].unsqueeze(0))).sum(dim=1) for i in range(num_hidden_layers)]
        masked, retained = (mask == 0).cpu().nonzero().tolist(), (mask == 1).cpu().nonzero().tolist()
    else:
        num_hidden_layers = len(mask)
        device = mask[0].device
        grads = [grad.cpu() for grad in grads]
        mask = [m.cpu() for m in mask]
        self_score = [grad.pow(2).sum(dim=0) for grad in grads]
        chunk_sum_by_layer = [(grads[i] * (1-mask[i].unsqueeze(0))).sum(dim=1) for i in range(len(mask))]
        masked, retained = [[i, j] for i, m in enumerate(mask) for j in range(m.shape[0]) if m[j] == 0], [[i, j] for i, m in enumerate(mask) for j in range(m.shape[0]) if m[j] == 1]
    
    # composed_mask's values are actually reversed, 0 means retained, 1 means masked
    masked_by_layer, retained_by_layer = {i: [] for i in range(num_hidden_layers)}, {i: [] for i in range(num_hidden_layers)}
    for item_masked in masked:
        masked_by_layer[item_masked[0]].append(item_masked[1])
    for item_retained in retained:
        retained_by_layer[item_retained[0]].append(item_retained[1])
    
    # TODO: change it to beam search for better outcome
    searching = True
    if debug:
        print("Initial local fisher score calculated: {}, using chunk calculated: {}".format((grads * (1 - mask)).sum(dim=0).pow(2).sum(), sum([chunk_sum.pow(2).sum() for chunk_sum in chunk_sum_by_layer]).item()))
    
    def get_grad(grads_to_get: Union[torch.Tensor, List[torch.Tensor]], layer_idx: int, block_idx: int) -> torch.Tensor:
        if is_mask_matched:
            return grads_to_get[:, layer_idx, block_idx]
        else:
            assert isinstance(grads_to_get, list) and isinstance(grads_to_get[0], torch.Tensor)
            return grads_to_get[layer_idx][:, block_idx]
        
    while searching:
        retained_scores = [
            2 * torch.matmul(
                get_grad(grads, i, retained_by_layer[i]).T,
                chunk_sum_by_layer[i]
                ) + self_score[i][retained_by_layer[i]] 
            for i in range(num_hidden_layers)
        ]
        add_pos_each, min_score_each = [score.argmin().item() if score.numel() else -1 for score in retained_scores], torch.Tensor([score.min() if score.numel() else float('inf') for score in retained_scores])
        add_pos_layer = min_score_each.argmin().item()
        masked_score = min_score_each.min()
        if add_pos_each[add_pos_layer] == -1:
            break
        masked_index = retained_by_layer[add_pos_layer][add_pos_each[add_pos_layer]]
        
        masked_by_layer[add_pos_layer].append(masked_index)
        del retained_by_layer[add_pos_layer][add_pos_each[add_pos_layer]]
        chunk_sum_by_layer[add_pos_layer] += get_grad(grads, add_pos_layer, masked_index)
        
        masked_scores = [
            2 * torch.matmul(
                get_grad(grads, i, masked_by_layer[i]).T,
                chunk_sum_by_layer[i]) - self_score[i][masked_by_layer[i]]
            for i in range(num_hidden_layers)
        ]
        remove_pos_each, max_score_each = [score.argmax().item() if score.numel() else -1 for score in masked_scores], torch.Tensor([score.max() if score.numel() else float('-inf') for score in masked_scores])
        remove_pos_layer = max_score_each.argmax().item()
        unmasked_score = max_score_each.max()
        unmasked_index = masked_by_layer[remove_pos_layer][remove_pos_each[remove_pos_layer]]
        
        if masked_index == unmasked_index and add_pos_layer == remove_pos_layer:
            searching = False
        retained_by_layer[remove_pos_layer].append(unmasked_index)
        del masked_by_layer[remove_pos_layer][remove_pos_each[remove_pos_layer]]
        chunk_sum_by_layer[remove_pos_layer] -= get_grad(grads, remove_pos_layer, unmasked_index)
        if debug:
            print("Switching {} to be masked and {} to be retained, using chunk calculated: {}, with masked score {} and unmasked score {}".format((remove_pos_layer, unmasked_index), (add_pos_layer, masked_index), sum([chunk_sum.pow(2).sum() for chunk_sum in chunk_sum_by_layer]).item(), masked_score, unmasked_score))
    masked_indices = [[i for i in range(num_hidden_layers) for _ in masked_by_layer[i]], [v for k in masked_by_layer for v in masked_by_layer[k]]]
    
    if is_mask_matched:
        new_mask = torch.ones_like(mask)
        new_mask[masked_indices] = 0
        new_mask = new_mask.to(device)
    else:
        new_mask = [torch.ones_like(m) for m in mask]
        for i, j in zip(*masked_indices):
            new_mask[i][j] = 0
        new_mask = [m.to(device) for m in new_mask]
    return new_mask

def global_layerwise_rearrange(head_mask: torch.Tensor, intermediate_mask: torch.Tensor, head_grads: torch.Tensor, intermediate_grads: torch.Tensor, debug: bool = False) -> Tuple[torch.Tensor]:
    # masks shape: [num_hidden_layers, num_attention_heads or num_intermediate_size]
    # grads shape: [batch_size, num_hidden_layers, num_attention_heads or num_intermediate_size]
    num_hidden_layer = head_mask.shape[0]
    num_batches = head_grads.shape[0]
    num_heads, num_intermediates = head_mask.shape[1], intermediate_mask.shape[1]
    head_grads, intermediate_grads = head_grads.cpu() * (num_heads ** 0.5), intermediate_grads.cpu() * (num_intermediates ** 0.5)
    composed_mask = 1 - torch.cat([head_mask.view(-1), intermediate_mask.view(-1)])
    composed_grads = torch.cat([head_grads.view(num_batches, -1), intermediate_grads.view(num_batches, -1)], dim=1).cpu()
    self_score = composed_grads.pow(2).sum(dim=0)
    self_score[:num_heads] *= 2
    self_score[num_heads:-num_intermediates] *= 3
    self_score[-num_intermediates:] *= 2
    head_score_chunk = (head_grads * (1-head_mask.unsqueeze(0))).sum(dim=2)
    intermediate_score_chunk = (intermediate_grads * (1-intermediate_mask.unsqueeze(0))).sum(dim=2)
    # composed_mask's values are actually reversed, 0 means retained, 1 means masked
    masked, retained = (composed_mask == 1).cpu().nonzero().squeeze().tolist(), ((composed_mask == 0).cpu().nonzero().squeeze()).tolist()
    def identify_unit(i: int):
        if i < num_hidden_layer * num_heads:
            return 'head', i // num_heads, i % num_heads
        else:
            i -= num_hidden_layer * num_heads
            return 'intermediate', i // num_intermediates, i % num_intermediates
        
    layerwise_chunk = form_layerwise_chunk(head_score_chunk, intermediate_score_chunk)
    masked_units = [identify_unit(i) for i in masked]
    retained_units = [identify_unit(i) for i in retained]
    def update_chunk_score(unit, diff):
        layerwise_chunk['head'][:, unit[1]] += diff
        layerwise_chunk['intermediate'][:, unit[1]] += diff
        if unit[0] == 'head' and unit[1] > 0:
            layerwise_chunk['intermediate'][:, unit[1] - 1] += diff
        if unit[0] == 'intermediate' and unit[1] < num_hidden_layer - 1:
            layerwise_chunk['head'][:, unit[1] + 1] += diff
    
    def calculate_fisher_score():
        return layerwise_chunk['head'].sum(dim=1).pow(2).sum().item() + layerwise_chunk['intermediate'].sum(dim=1).pow(2).sum().item()
    searching = True
    while searching:
        # Calculating all masked heads and intermediates' importance, and unmask the max one
        masked_layerwise_scores = torch.Tensor([2 * torch.dot(composed_grads[:, i], layerwise_chunk[unit[0]][:, unit[1]]) - self_score[i] for i, unit in zip(masked, masked_units)])
        max_pos = masked_layerwise_scores.argmax().item()
        select_unmask_index = masked[max_pos]
        select_unmask_unit = masked_units[max_pos]
        del masked[max_pos]
        del masked_units[max_pos]
        retained.append(select_unmask_index)
        retained_units.append(identify_unit(select_unmask_index))
        if select_unmask_unit[0] == 'head':
            # Remove a head, then find a head to replace
            diff = -head_grads[:, select_unmask_unit[1], select_unmask_unit[2]]
            diff_score = 2 * torch.dot(diff, layerwise_chunk['head'][:, select_unmask_unit[1]]) + self_score[select_unmask_index]
            update_chunk_score(select_unmask_unit, diff)
            usable_retained_indices = [i for i, index in enumerate(retained) if index < num_hidden_layer * num_heads]
            usable_retained, usable_retained_units = [retained[i] for i in usable_retained_indices], [retained_units[i] for i in usable_retained_indices]
            retained_layerwise_scores = torch.Tensor([2 * torch.dot(composed_grads[:, i], layerwise_chunk[unit[0]][:, unit[1]]) + self_score[i] for i, unit in zip(usable_retained, usable_retained_units)])
            min_pos = retained_layerwise_scores.argmin().item()
            select_mask_index = usable_retained[min_pos]
            select_mask_unit = usable_retained_units[min_pos]
            retained.remove(select_mask_index)
            retained_units.remove(select_mask_unit)
            masked.append(select_mask_index)
            masked_units.append(select_mask_unit)
            if select_mask_index == select_unmask_index:
                # This means that the unmasking operation is not useful, so we stop the searching
                searching = False
                break
            add_diff = head_grads[:, select_mask_unit[1], select_mask_unit[2]]
            add_diff_score = 2 * torch.dot(add_diff, layerwise_chunk['head'][:, select_mask_unit[1]]) + self_score[select_mask_index]
            update_chunk_score(select_mask_unit, add_diff)
        else:
            # Remove an intermediate neuron, then find an intermediate neuron to replace
            diff = -intermediate_grads[:, select_unmask_unit[1], select_unmask_unit[2]]
            diff_score = 2 * torch.dot(diff, layerwise_chunk['intermediate'][:, select_unmask_unit[1]]) + self_score[select_unmask_index]
            update_chunk_score(select_unmask_unit, diff)
            usable_retained_indices = [i for i, index in enumerate(retained) if index >= num_hidden_layer * num_heads]
            usable_retained, usable_retained_units = [retained[i] for i in usable_retained_indices], [retained_units[i] for i in usable_retained_indices]
            retained_layerwise_scores = torch.Tensor([2 * torch.dot(composed_grads[:, i], layerwise_chunk[unit[0]][:, unit[1]]) + self_score[i] for i, unit in zip(usable_retained, usable_retained_units)])
            min_pos = retained_layerwise_scores.argmin().item()
            select_mask_index = usable_retained[min_pos]
            select_mask_unit = usable_retained_units[min_pos]
            retained.remove(select_mask_index)
            retained_units.remove(select_mask_unit)
            masked.append(select_mask_index)
            masked_units.append(select_mask_unit)
            if select_mask_index == select_unmask_index:
                # This means that the unmasking operation is not useful, so we stop the searching
                searching = False
                break
            add_diff = intermediate_grads[:, select_mask_unit[1], select_mask_unit[2]]
            add_diff_score = 2 * torch.dot(add_diff, layerwise_chunk['intermediate'][:, select_mask_unit[1]]) + self_score[select_mask_index]
            update_chunk_score(select_mask_unit, add_diff)
        if debug:
            new_head_mask, new_intermediate_mask = torch.zeros(num_hidden_layer * num_heads), torch.zeros(num_hidden_layer * num_intermediates)
            new_head_mask[[v for v in retained if v < num_hidden_layer * num_heads]] = 1
            new_intermediate_mask[[v - num_hidden_layer * num_heads for v in retained if v >= num_hidden_layer * num_heads]] = 1
            new_head_mask, new_intermediate_mask = new_head_mask.view(num_hidden_layer, num_heads), new_intermediate_mask.view(num_hidden_layer, num_intermediates)
            print("Switching {} to be masked and {} to be retained, fisher score {}, score reduced {}, local fisher score calculated: {}".format(select_unmask_index, select_mask_index, sum_fisher_score(head_grads / (num_heads ** 0.5), intermediate_grads / (num_intermediates ** 0.5), new_head_mask, new_intermediate_mask, deduplication=True), diff_score + add_diff_score, calculate_fisher_score()))
    new_head_mask, new_intermediate_mask = torch.zeros(num_hidden_layer * num_heads), torch.zeros(num_hidden_layer * num_intermediates)
    new_head_mask[[v for v in retained if v < num_hidden_layer * num_heads]] = 1
    new_intermediate_mask[[v - num_hidden_layer * num_heads for v in retained if v >= num_hidden_layer * num_heads]] = 1
    new_head_mask, new_intermediate_mask = new_head_mask.view(num_hidden_layer, num_heads), new_intermediate_mask.view(num_hidden_layer, num_intermediates)
    return new_head_mask, new_intermediate_mask