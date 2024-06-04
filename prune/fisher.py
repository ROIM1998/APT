# This file contains code derived from [Mask-Tuning](https://github.com/WoosukKwon/retraining-free-pruning),
# originally developed by Woosuk Kwon.
# Modifications were made to adapt to the current project's context.

import torch
from tqdm import tqdm

def collect_mask_grads(model, dataloader):
    model.eval()
    is_mask_matched = isinstance(model.head_mask, torch.Tensor) and isinstance(model.intermediate_mask, torch.Tensor)
    pre_collect_tuning_status = {}
    for n, p in model.named_parameters():
        pre_collect_tuning_status[n] = p.requires_grad
        p.requires_grad_(False)
    if is_mask_matched:
        model.head_mask.requires_grad_(True)
        model.head_mask.retain_grad()
        model.intermediate_mask.requires_grad_(True)
        model.intermediate_mask.retain_grad()
    else:
        for m in model.head_mask:
            if m is not None:
                m.requires_grad_(True)
                m.retain_grad()
        for m in model.intermediate_mask:
            if m is not None:
                m.requires_grad_(True)
                m.retain_grad()

    head_grads = []
    intermediate_grads = []
    for batch in tqdm(dataloader):
        for k, v in batch.items():
            batch[k] = v.to(model.device, non_blocking=True) if isinstance(v, torch.Tensor) else torch.stack(v, dim=1).to(model.device) if isinstance(v, list) else v
        if 't5' in model.config.model_type or 'llama' in model.config.model_type:
            batch['use_cache'] = False
        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        if is_mask_matched:
            head_grads.append(model.head_mask.grad.detach())
            model.head_mask.grad = None
            intermediate_grads.append(model.intermediate_mask.grad.detach())
            model.intermediate_mask.grad = None
        else:
            head_grads.append([m.grad.detach() if m is not None else torch.tensor([]).to(model.device) for m in model.head_mask])
            for m in model.head_mask:
                if m is not None:
                    m.grad = None
            intermediate_grads.append([m.grad.detach() if m is not None else torch.tensor([]).to(model.device) for m in model.intermediate_mask])
            for m in model.intermediate_mask:
                if m is not None:
                    m.grad = None

    if is_mask_matched:
        head_grads = torch.stack(head_grads, dim=0)
        model.head_mask = model.head_mask.detach()
        intermediate_grads = torch.stack(intermediate_grads, dim=0)
        model.intermediate_mask = model.intermediate_mask.detach()
    else:
        head_grads = [torch.stack(g, dim=0) for g in zip(*head_grads)]
        model.head_mask = [m.detach() if m is not None else None for m in model.head_mask]
        intermediate_grads = [torch.stack(g, dim=0) for g in zip(*intermediate_grads)]
        model.intermediate_mask = [m.detach() if m is not None else None for m in model.intermediate_mask]
    # restore tuning status
    for n, p in model.named_parameters():
        p.requires_grad_(pre_collect_tuning_status[n])
    return head_grads, intermediate_grads

def collect_hidden_mask_grads(model, dataloader):
    if not hasattr(model, 'hidden_mask'):
        raise ValueError('The model does not have hidden_mask')
    model.eval()
    # With the size of (768,)
    pre_collect_tuning_status = {}
    for n, p in model.named_parameters():
        pre_collect_tuning_status[n] = p.requires_grad
        p.requires_grad_(False)
    model.hidden_mask.requires_grad_(True)
    model.hidden_mask.retain_grad()

    hidden_grads = []
    for batch in tqdm(dataloader):
        for k, v in batch.items():
            batch[k] = v.to(model.device, non_blocking=True) if isinstance(v, torch.Tensor) else torch.stack(v, dim=1).to(model.device) if isinstance(v, list) else v
        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        hidden_grads.append(model.hidden_mask.grad.detach())
        model.hidden_mask.grad = None

    hidden_grads = torch.stack(hidden_grads, dim=0)
    model.hidden_mask = model.hidden_mask.detach()
    # restore tuning status
    for n, p in model.named_parameters():
        p.requires_grad_(pre_collect_tuning_status[n])
    return hidden_grads

def collect_additive_mask_grads(model, dataloader):
    model.eval()
    all_grads = {}
    pre_collect_tuning_status = {}
    for n, p in model.named_parameters():
        if n.endswith('.output_mask') or n.endswith('.bottleneck_mask'):
            p.requires_grad_(True)
            p.retain_grad()
            all_grads[n] = []
        else:
            pre_collect_tuning_status[n] = p.requires_grad
            p.requires_grad_(False)

    for batch in tqdm(dataloader):
        for k, v in batch.items():
            batch[k] = v.to(model.device, non_blocking=True) if isinstance(v, torch.Tensor) else torch.stack(v, dim=1).to(model.device) if isinstance(v, list) else v
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        for n, p in model.named_parameters():
            if n.endswith('.output_mask') or n.endswith('.bottleneck_mask'):
                all_grads[n].append(p.grad.detach())
                p.grad = None
    for n, p in model.named_parameters():
        if n.endswith('.output_mask') or n.endswith('.bottleneck_mask'):
            p.requires_grad_(False)
            all_grads[n] = torch.stack(all_grads[n], dim=0)
        else:
            p.requires_grad_(pre_collect_tuning_status[n])
    return all_grads

def collect_grads_by_suffix(model, dataloader, suffix):
    assert suffix.startswith('.') and suffix.endswith('_mask')
    model.eval()
    all_grads = {}
    pre_collect_tuning_status = {}
    for n, p in model.named_parameters():
        if n.endswith(suffix):
            p.requires_grad_(True)
            p.retain_grad()
            all_grads[n] = []
        else:
            pre_collect_tuning_status[n] = p.requires_grad
            p.requires_grad_(False)

    for batch in tqdm(dataloader):
        for k, v in batch.items():
            batch[k] = v.to(model.device, non_blocking=True) if isinstance(v, torch.Tensor) else torch.stack(v, dim=1).to(model.device) if isinstance(v, list) else v
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        for n, p in model.named_parameters():
            if n.endswith(suffix):
                if p.grad is not None:
                    all_grads[n].append(p.grad.detach())
                else:
                    # Set to zero if the gradient is None
                    all_grads[n].append(torch.zeros_like(p).detach())
                p.grad = None
    for n, p in model.named_parameters():
        if n.endswith(suffix):
            p.requires_grad_(False)
            all_grads[n] = torch.stack(all_grads[n], dim=0)
        else:
            p.requires_grad_(pre_collect_tuning_status[n])
    return all_grads

def collect_weight_saliency(model, dataloader, layer_names):
    all_grads = {}
    for n, p in model.named_parameters():
        if n.rsplit('.', 1)[0] in layer_names and 'weight' in n:
            p.requires_grad = True
            all_grads[n] = []
        else:
            p.requires_grad = False
    model.zero_grad()
    for batch in tqdm(dataloader):
        for k, v in batch.items():
            batch[k] = v.to(model.device, non_blocking=True) if isinstance(v, torch.Tensor) else torch.stack(v, dim=1).to(model.device) if isinstance(v, list) else v
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        for n, p in model.named_parameters():
            if n in all_grads:
                # Collect weight-gradient product
                all_grads[n].append(p.grad.detach() * p.detach())
                p.grad = None
    for n, p in model.named_parameters():
        if n in all_grads:
            all_grads[n] = torch.stack(all_grads[n], dim=0)
    return all_grads

def collect_param_salience(model, dataloader, param_names):
    all_grads = {}
    pre_collect_tuning_status = {}
    for n, p in model.named_parameters():
        if n in param_names:
            p.requires_grad = True
            all_grads[n] = []
        else:
            pre_collect_tuning_status[n] = p.requires_grad
            p.requires_grad = False
    model.zero_grad()
    for batch in tqdm(dataloader):
        for k, v in batch.items():
            batch[k] = v.to(model.device, non_blocking=True) if isinstance(v, torch.Tensor) else torch.stack(v, dim=1).to(model.device) if isinstance(v, list) else v
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        for n, p in model.named_parameters():
            if n in all_grads:
                # Collect weight-gradient product
                all_grads[n].append(p.grad.detach() * p.detach())
                p.grad = None
    for n, p in model.named_parameters():
        if n in param_names:
            p.requires_grad_(False)
            all_grads[n] = torch.stack(all_grads[n], dim=0)
        else:
            p.requires_grad_(pre_collect_tuning_status[n])
    return all_grads

@torch.no_grad()
def compute_fisher_info(grads, dim=0):
    if isinstance(grads, torch.Tensor):
        fisher_info = grads.pow(2).sum(dim=dim)
    elif isinstance(grads, list):
        fisher_info = [g.pow(2).sum(dim=dim) for g in grads]
    return fisher_info

@torch.no_grad()
def compute_l1_fisher_info(grads, dim=0):
    if isinstance(grads, torch.Tensor):
        fisher_info = grads.sum(dim=dim)
    elif isinstance(grads, list):
        fisher_info = [g.sum(dim=dim) for g in grads]
    return fisher_info