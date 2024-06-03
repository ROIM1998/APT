import math
import time
import torch
import random
import numpy as np
import pandas as pd
import loralib as lora
import torch.nn as nn
from typing import Dict, Tuple, Union, List, Optional
from torch.utils.data import DataLoader
from .fisher_utils.arch import get_layers, hijack_input, hijack_output
from transformers import PreTrainedModel
from trainer.model_arch import get_encoder, get_layers, get_decoder

ADJUST_SCALING = False

def to_cpu_recursive(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {key: to_cpu_recursive(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [to_cpu_recursive(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(to_cpu_recursive(item) for item in obj)
    else:
        return obj

def flatten_states(states, mask):
    batch_size = states.shape[0]
    if len(states.shape) == 4:
        seq_length = states.shape[2]
        return None
    else:
        seq_length = states.shape[1]
        states_dim_size = states.shape[2]
        valid_indices = mask.sum(dim=1)
        states_flattened = [
            torch.cat([
                states[j,:valid_indices[j],i]
                for j in range(batch_size)
            ])
            for i in range(states_dim_size)
        ]
        return states_flattened

def _mask_fine_to_coarse(model, mask):
    if isinstance(mask, torch.Tensor):
        if mask.ndim == 1:
            return mask.detach().any().float()
        else:
            return mask.detach().any(dim=1).float()
    elif isinstance(mask, list):
        return [v.detach().any().float() if v is not None else None for v in mask]
    
def del_parameter(layer: nn.Linear, param_name_list: Union[str, List[str]] = ['weight', 'bias', 'lora_A', 'lora_B', 'in_transformation', 'out_transformation', 'teacher_lora_A', 'teacher_lora_B', 'teacher_in_transformation', 'teacher_out_transformation', 'input_mask', 'output_mask', 'bottleneck_mask', 'in_retained_indices', 'out_retained_indices', 'teacher_in_retained_indices', 'teacher_out_retained_indices']):
    if isinstance(param_name_list, str):
        param_name_list = [param_name_list]
    for param_name in param_name_list:
        if hasattr(layer, param_name):
            delattr(layer, param_name)

def prune_layer_norm(layernorm, index):
    layernorm.weight = torch.nn.parameter.Parameter(
        layernorm.weight.index_select(0, index)).contiguous()
    layernorm.bias = torch.nn.parameter.Parameter(
        layernorm.bias.index_select(0, index)).contiguous()
    layernorm.normalized_shape = (len(index),)
    
def prune_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    if isinstance(layer, lora.layers.Linear) and layer.r > 0:
        new_layer = prune_lora_layer(layer, index, dim)
    else:
        new_layer = prune_linear_layer(layer, index, dim)
    del_parameter(layer)
    return new_layer
    
def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).detach().clone()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.detach().clone()
        else:
            b = layer.bias[index].detach().clone()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    layer_cls = lora.SelectLinear if isinstance(layer, lora.SelectLinear) else nn.Linear
    new_layer = layer_cls(new_size[1], new_size[0], bias=layer.bias is not None, dtype=layer.weight.dtype).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = layer.weight.requires_grad
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = layer.bias.requires_grad
    return new_layer

def _1d_tolist(t: torch.Tensor) -> List:
    if isinstance(t, torch.Tensor):
        if t.ndim > 1:
            print("Received tensor %s with shape %s" % (t, t.shape))
            raise ValueError("Only 0D or 1D tensors are supported.")
        v = t.tolist()
        return v if isinstance(v, list) else [v] # Support 0d tensors
    else:
        raise NotImplementedError("Only torch.Tensor is supported.")
    
def detect_no_zero(t: Union[None, torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]) -> bool:
    if t is None:
        return True
    elif isinstance(t, torch.Tensor):
        return (t != 0).all().item()
    elif isinstance(t, list) or isinstance(t, tuple):
        return all(detect_no_zero(x) for x in t)
    else:
        raise NotImplementedError("Only torch.Tensor is supported.")
    
def prune_lora_layer(layer: lora.layers.Linear, index: torch.LongTensor, dim: int = 0) -> lora.layers.Linear:
    """
    Prune a LoRA linear layer to keep only entries in index.

    Used to remove heads. (especially query & value)

    Args:
        layer (:obj:`lora.layers.Linear`): The LoRA layer to prune.
        index (:obj:`torch.LongTensor`): The indices to keep in the layer.
        dim (:obj:`int`, `optional`, defaults to 0): The dimension on which to keep the indices.

    Returns:
        :obj:`lora.layers.Linear`: The pruned LoRA layer as a new layer with :obj:`requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).detach().clone()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.detach().clone()
        else:
            b = layer.bias[index].detach().clone()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    dtype = layer.weight.dtype
    # TODO: add support when output_dynamic is False
    target_class = lora.layers.DistillLinear if isinstance(layer, lora.layers.DistillLinear) else lora.layers.PruningLinear if isinstance(layer, lora.layers.PruningLinear) else lora.layers.Linear
    if target_class is lora.layers.Linear:
        new_layer = target_class(new_size[1], new_size[0], r = layer.r, lora_alpha = layer.lora_alpha, lora_dropout = layer.lora_dropout.p if isinstance(layer.lora_dropout, nn.Dropout) else 0, dtype=dtype, bias=layer.bias is not None).to(layer.weight.device)
        if dim == 0:
            new_layer.lora_A.data = layer.lora_A.detach().clone().contiguous()
            new_layer.lora_B.data = layer.lora_B.index_select(0, index).detach().clone().contiguous()
        else:
            new_layer.lora_B.data = layer.lora_B.detach().clone().contiguous()
            new_layer.lora_A.data = layer.lora_A.index_select(1, index).detach().clone().contiguous()
        new_layer.lora_A.requires_grad = layer.lora_A.requires_grad
        new_layer.lora_B.requires_grad = layer.lora_B.requires_grad
    else:
        assert isinstance(layer, lora.layers.PruningLinear)
        if target_class is lora.layers.PruningLinear:
            new_layer = lora.layers.PruningLinear(new_size[1], new_size[0], r=layer.r, lora_alpha=layer.lora_alpha, retained_indices=None, out_retained_indices=None, in_retained_indices=None, lora_dropout=layer.lora_dropout.p if isinstance(layer.lora_dropout, nn.Dropout) else 0, fan_in_fan_out=layer.fan_in_fan_out, merge_weights=layer.merge_weights, act_fn=layer.act_fn_type, dtype=dtype, bias=layer.bias is not None).to(layer.weight.device) # Set all retained_indices to None, and then set them later
            if dim == 0:
                new_layer.lora_A.data = layer.lora_A.detach().clone().contiguous()
                new_layer.in_retained_indices = layer.in_retained_indices
                if layer.out_retained_indices is None:
                    new_layer.lora_B.data = layer.lora_B.index_select(0, index).detach().clone().contiguous()
                    if layer.output_mask is not None:
                        new_layer.output_mask = layer.output_mask[index].detach().clone().contiguous()
                else:
                    valid_indices, new_retained_indices = torch.where(torch.tensor(layer.out_retained_indices, device=index.device)[:, None] == index)
                    new_layer.lora_B.data = layer.lora_B.index_select(0, valid_indices).detach().clone().contiguous()
                    new_layer.out_retained_indices = _1d_tolist(new_retained_indices)
                    if layer.output_mask is not None:
                        new_layer.output_mask = layer.output_mask[valid_indices].detach().clone().contiguous()
                if layer.input_mask is not None:
                    new_layer.input_mask = layer.input_mask.detach().clone().contiguous()
            else:
                new_layer.lora_B.data = layer.lora_B.detach().clone().contiguous()
                new_layer.out_retained_indices = layer.out_retained_indices
                if layer.in_retained_indices is None:
                    new_layer.lora_A.data = layer.lora_A.index_select(1, index).detach().clone().contiguous()
                    if layer.input_mask is not None:
                        new_layer.input_mask = layer.input_mask[index].detach().clone().contiguous()
                else:
                    valid_indices, new_retained_indices = torch.where(torch.tensor(layer.in_retained_indices, device=index.device)[:, None] == index)
                    new_layer.lora_A.data = layer.lora_A.index_select(1, valid_indices).detach().clone().contiguous()
                    new_layer.in_retained_indices = _1d_tolist(new_retained_indices)
                    if layer.input_mask is not None:
                        new_layer.input_mask = layer.input_mask[valid_indices].detach().clone().contiguous()
                if layer.output_mask is not None:
                    new_layer.output_mask = layer.output_mask.detach().clone().contiguous()
            if layer.bottleneck_mask is not None:
                new_layer.bottleneck_mask = layer.bottleneck_mask.detach().clone().contiguous()
            new_layer.lora_A.requires_grad = layer.lora_A.requires_grad
            new_layer.lora_B.requires_grad = layer.lora_B.requires_grad
            new_layer._init_transformations()
        elif target_class is lora.layers.DistillLinear:
            assert isinstance(layer, lora.layers.DistillLinear)
            new_layer = lora.layers.DistillLinear(new_size[1], new_size[0], r=layer.r, teacher_r=layer.teacher_r, lora_alpha=layer.lora_alpha, teacher_lora_alpha=layer.teacher_lora_alpha,lora_dropout=layer.lora_dropout.p if isinstance(layer.lora_dropout, nn.Dropout) else 0, fan_in_fan_out=layer.fan_in_fan_out, merge_weights=layer.merge_weights, act_fn=layer.act_fn_type, dtype=dtype, bias=layer.bias is not None).to(layer.weight.device) # Set all retained_indices to None, and then set them later
            if dim == 0:
                new_layer.lora_A.data = layer.lora_A.detach().clone().contiguous()
                new_layer.in_retained_indices = layer.in_retained_indices
                if layer.out_retained_indices is None:
                    new_layer.lora_B.data = layer.lora_B.index_select(0, index).detach().clone().contiguous()
                    if layer.output_mask is not None:
                        new_layer.output_mask = layer.output_mask[index].detach().clone().contiguous()
                else:
                    valid_indices, new_retained_indices = torch.where(torch.tensor(layer.out_retained_indices, device=index.device)[:, None] == index)
                    new_layer.lora_B.data = layer.lora_B.index_select(0, valid_indices).detach().clone().contiguous()
                    new_layer.out_retained_indices = _1d_tolist(new_retained_indices)
                    if layer.output_mask is not None:
                        new_layer.output_mask = layer.output_mask[valid_indices].detach().clone().contiguous()
                if layer.input_mask is not None:
                    new_layer.input_mask = layer.input_mask.detach().clone().contiguous()
                if layer.teacher_r > 0:
                    new_layer.teacher_lora_A.data = layer.teacher_lora_A.detach().clone().contiguous()
                    new_layer.teacher_in_retained_indices = layer.teacher_in_retained_indices
                    if layer.teacher_out_retained_indices is None:
                        new_layer.teacher_lora_B.data = layer.teacher_lora_B.index_select(0, index).detach().clone().contiguous()
                    else:
                        valid_teacher_indices, new_teacher_retained_indices = torch.where(torch.tensor(layer.teacher_out_retained_indices, device=index.device)[:, None] == index)
                        new_layer.teacher_lora_B.data = layer.teacher_lora_B.index_select(0, valid_teacher_indices).detach().clone().contiguous()
                        new_layer.teacher_out_retained_indices = _1d_tolist(new_teacher_retained_indices)
            else:
                new_layer.lora_B.data = layer.lora_B.detach().clone().contiguous()
                new_layer.out_retained_indices = layer.out_retained_indices
                if layer.in_retained_indices is None:
                    new_layer.lora_A.data = layer.lora_A.index_select(1, index).detach().clone().contiguous()
                    if layer.input_mask is not None:
                        new_layer.input_mask = layer.input_mask[index].detach().clone().contiguous()
                else:
                    valid_indices, new_retained_indices = torch.where(torch.tensor(layer.in_retained_indices, device=index.device)[:, None] == index)
                    new_layer.lora_A.data = layer.lora_A.index_select(1, valid_indices).detach().clone().contiguous()
                    new_layer.in_retained_indices = _1d_tolist(new_retained_indices)
                    if layer.input_mask is not None:
                        new_layer.input_mask = layer.input_mask[valid_indices].detach().clone().contiguous()
                if layer.output_mask is not None:
                    new_layer.output_mask = layer.output_mask.detach().clone().contiguous()
                if layer.teacher_r > 0:
                    new_layer.teacher_lora_B.data = layer.teacher_lora_B.detach().clone().contiguous()
                    new_layer.teacher_out_retained_indices = layer.teacher_out_retained_indices
                    if layer.teacher_in_retained_indices is None:
                        new_layer.teacher_lora_A.data = layer.teacher_lora_A.index_select(1, index).detach().clone().contiguous()
                    else:
                        valid_teacher_indices, new_teacher_retained_indices = torch.where(torch.tensor(layer.teacher_in_retained_indices, device=index.device)[:, None] == index)
                        new_layer.teacher_lora_A.data = layer.teacher_lora_A.index_select(1, valid_teacher_indices).detach().clone().contiguous()
                        new_layer.teacher_in_retained_indices = _1d_tolist(new_teacher_retained_indices)
            if new_layer.input_mask is not None:
                new_layer.input_mask.requires_grad = layer.input_mask.requires_grad
            if new_layer.output_mask is not None:
                new_layer.output_mask.requires_grad = layer.output_mask.requires_grad
            if layer.bottleneck_mask is not None:
                new_layer.bottleneck_mask = layer.bottleneck_mask.detach().clone().contiguous()
                new_layer.bottleneck_mask.requires_grad = layer.bottleneck_mask.requires_grad
            if layer.r > 0:
                new_layer.lora_A.requires_grad = layer.lora_A.requires_grad
                new_layer.lora_B.requires_grad = layer.lora_B.requires_grad
            if layer.teacher_r > 0:
                new_layer.teacher_lora_A.requires_grad = layer.teacher_lora_A.requires_grad
                new_layer.teacher_lora_B.requires_grad = layer.teacher_lora_B.requires_grad
            new_layer._init_transformations()
        else:
            raise NotImplementedError("Unsupported target class.")
        new_layer.history = layer.history
    
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = layer.weight.requires_grad
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = layer.bias.requires_grad
    return new_layer

def shrink_pruning_lora_outdim(layer: Union[lora.PruningLinear, lora.DistillLinear], pruned_out_dim: List[int], teacher_pruned_out_dim: Optional[List[int]] = None) -> Union[lora.PruningLinear, lora.DistillLinear]:
    if layer.out_transformation is None:
        if pruned_out_dim:
            new_retained_indices = list(set(range(layer.out_features)) - set(pruned_out_dim))
        else:
            new_retained_indices = None
    else:
        inner_outdim_mask = torch.zeros(layer.out_transformation.shape[0], dtype=layer.weight.dtype).to(layer.out_transformation.device)
        inner_outdim_mask[pruned_out_dim] = 1
        outside_pruned_dims = _1d_tolist((inner_outdim_mask @ layer.out_transformation).nonzero().squeeze())
        new_retained_indices = list(set(layer.out_retained_indices) - set(outside_pruned_dims))
    new_retained_indices = sorted(new_retained_indices)
    # Creating the new layer based on the pruning shape
    if isinstance(layer, lora.DistillLinear):
        if layer.teacher_out_transformation is None:
            if teacher_pruned_out_dim:
                new_teacher_retained_indices = list(set(range(layer.out_features)) - set(teacher_pruned_out_dim))
            else:
                new_teacher_retained_indices = None
        else:
            inner_teacher_outdim_mask = torch.zeros(layer.teacher_out_transformation.shape[0], dtype=layer.weight.dtype).to(layer.teacher_out_transformation.device)
            inner_teacher_outdim_mask[teacher_pruned_out_dim] = 1
            outside_teacher_pruned_dims = _1d_tolist((inner_teacher_outdim_mask @ layer.teacher_out_transformation).nonzero().squeeze())
            new_teacher_retained_indices = list(set(layer.teacher_retained_indices) - set(outside_teacher_pruned_dims))
        new_layer = lora.DistillLinear(layer.in_features, layer.out_features, r=layer.r, lora_alpha=layer.lora_alpha, lora_dropout=layer.lora_dropout.p if isinstance(layer.lora_dropout, nn.Dropout) else 0, fan_in_fan_out=layer.fan_in_fan_out, merge_weights=layer.merge_weights, out_retained_indices=new_retained_indices, in_retained_indices=layer.in_retained_indices, teacher_out_retained_indices=new_teacher_retained_indices, teacher_in_retained_indices=layer.teacher_in_retained_indices, output_dynamic=layer.output_dynamic, act_fn=layer.act_fn_type, bias=layer.bias is not None).to(layer.weight.device)
    else:
        new_layer = lora.PruningLinear(layer.in_features, layer.out_features, r=layer.r, lora_alpha=layer.lora_alpha, lora_dropout=layer.lora_dropout.p if isinstance(layer.lora_dropout, nn.Dropout) else 0, fan_in_fan_out=layer.fan_in_fan_out, merge_weights=layer.merge_weights, out_retained_indices=new_retained_indices, in_retained_indices=layer.in_retained_indices, output_dynamic=layer.output_dynamic, act_fn=layer.act_fn_type, bias=layer.bias is not None).to(layer.weight.device)
    new_layer.to(layer.weight.device, dtype=layer.weight.dtype)
    # Copying the parameters from the old layer to the new layer
    new_layer.weight.data = layer.weight.data.detach().clone().contiguous()
    if layer.bias is not None:
        new_layer.bias.data = layer.bias.data.detach().clone().contiguous()
    # Copy the shrinked bottleneck parameters' values
    # TODO: add DistillLinear support
    # Copy the shrinked out_dimension parameters' values
    if pruned_out_dim:
        weight_edited = layer.calculate_pruned_outdim_weights(pruned_out_dim)
        if weight_edited is not None:
            new_layer.weight.data += weight_edited
        new_retained_indices_to_select = list(set(range(layer.lora_B.shape[0])) - set(pruned_out_dim))
        new_retained_indices_to_select = sorted(new_retained_indices_to_select)
        new_layer.lora_B.data = layer.lora_B[new_retained_indices_to_select, :].detach().clone().contiguous()
        new_layer.lora_A.data = layer.lora_A.detach().clone().contiguous()
        new_layer.output_mask = layer.output_mask[new_retained_indices_to_select].detach().clone().contiguous() if layer.output_mask is not None else None
    else:
        new_layer.lora_A.data = layer.lora_A.detach().clone().contiguous()
        new_layer.lora_B.data = layer.lora_B.detach().clone().contiguous()
        new_layer.output_mask = layer.output_mask.detach().clone().contiguous()
    new_layer.input_mask = layer.input_mask.detach().clone().contiguous() if layer.input_mask is not None else None
    new_layer.bottleneck_mask = layer.bottleneck_mask.detach().clone().contiguous() if layer.bottleneck_mask is not None else None
    if teacher_pruned_out_dim:
        assert isinstance(layer, lora.DistillLinear)
        new_teacher_retained_indices_to_select = list(set(range(layer.teacher_lora_B.shape[0])) - set(teacher_pruned_out_dim))
        new_teacher_retained_indices_to_select = sorted(new_teacher_retained_indices_to_select)
        new_layer.teacher_lora_B.data = layer.teacher_lora_B[new_teacher_retained_indices_to_select, :].detach().clone().contiguous()
        new_layer.teacher_lora_A.data = layer.teacher_lora_A.detach().clone().contiguous()
    elif isinstance(layer, lora.DistillLinear):
        new_layer.teacher_lora_A.data = layer.teacher_lora_A.detach().clone().contiguous()
        new_layer.teacher_lora_B.data = layer.teacher_lora_B.detach().clone().contiguous()
    return new_layer
    
def shrink_pruning_lora_indim(layer: Union[lora.PruningLinear, lora.DistillLinear], pruned_in_dim: List[int], teacher_pruned_in_dim: Optional[List[int]] = None) -> Union[lora.PruningLinear, lora.DistillLinear]:
    # TODO: add DistillLinear support
    if layer.in_transformation is None:
        if pruned_in_dim:
            new_retained_indices = list(set(range(layer.in_features)) - set(pruned_in_dim))
        else:
            new_retained_indices = None
    else:
        inner_indim_mask = torch.zeros(layer.in_transformation.shape[1], dtype=layer.weight.dtype).to(layer.in_transformation.device)
        inner_indim_mask[pruned_in_dim] = 1
        outside_pruned_dims = _1d_tolist((layer.in_transformation @ inner_indim_mask).nonzero().squeeze())
        new_retained_indices = list(set(layer.in_retained_indices) - set(outside_pruned_dims))
    new_retained_indices = sorted(new_retained_indices)
    # Creating the new layer based on the pruning shap
    if isinstance(layer, lora.DistillLinear):
        if layer.teacher_in_transformation is None:
            if teacher_pruned_in_dim:
                new_teacher_retained_indices = list(set(range(layer.in_features)) - set(teacher_pruned_in_dim))
            else:
                new_teacher_retained_indices = None
        else:
            inner_teacher_indim_mask = torch.zeros(layer.teacher_in_transformation.shape[1], dtype=layer.weight.dtype).to(layer.teacher_in_transformation.device)
            inner_teacher_indim_mask[teacher_pruned_in_dim] = 1
            outside_teacher_pruned_dims = _1d_tolist((layer.teacher_in_transformation @ inner_teacher_indim_mask).nonzero().squeeze())
            new_teacher_retained_indices = list(set(layer.teacher_retained_indices) - set(outside_teacher_pruned_dims))
        new_layer = lora.DistillLinear(layer.in_features, layer.out_features, r=layer.r, lora_alpha=layer.lora_alpha, lora_dropout=layer.lora_dropout.p if isinstance(layer.lora_dropout, nn.Dropout) else 0, fan_in_fan_out=layer.fan_in_fan_out, merge_weights=layer.merge_weights, out_retained_indices=layer.out_retained_indices, in_retained_indices=new_retained_indices, teacher_out_retained_indices=layer.teacher_out_retained_indices, teacher_in_retained_indices=new_teacher_retained_indices, output_dynamic=layer.output_dynamic, act_fn=layer.act_fn_type, bias=layer.bias is not None).to(layer.weight.device)
    else:
        new_layer = lora.PruningLinear(layer.in_features, layer.out_features, r=layer.r, lora_alpha=layer.lora_alpha, lora_dropout=layer.lora_dropout.p if isinstance(layer.lora_dropout, nn.Dropout) else 0, fan_in_fan_out=layer.fan_in_fan_out, merge_weights=layer.merge_weights, in_retained_indices=new_retained_indices, out_retained_indices=layer.out_retained_indices, output_dynamic=layer.output_dynamic, act_fn=layer.act_fn_type, bias=layer.bias is not None).to(layer.weight.device)
    new_layer.to(layer.weight.device, dtype=layer.weight.dtype)
    # Copying the parameters from the old layer to the new layer
    new_layer.weight.data = layer.weight.data.detach().clone().contiguous()
    if layer.bias is not None:
        new_layer.bias.data = layer.bias.data.detach().clone().contiguous()
    # Copy the shrinked bottleneck parameters' values
    # TODO: add DistillLinear support
    # Copy the shrinked out_dimension parameters' values
    if pruned_in_dim:
        weight_edited = layer.calculate_pruned_outdim_weights(pruned_in_dim=pruned_in_dim)
        if weight_edited is not None:
            new_layer.weight.data += weight_edited
        new_retained_indices_to_select = list(set(range(layer.lora_A.shape[1])) - set(pruned_in_dim))
        new_retained_indices_to_select = sorted(new_retained_indices_to_select)
        new_layer.lora_A.data = layer.lora_A[:, new_retained_indices_to_select].detach().clone().contiguous()
        new_layer.lora_B.data = layer.lora_B.detach().clone().contiguous()
        new_layer.input_mask = layer.input_mask[new_retained_indices_to_select].detach().clone().contiguous() if layer.input_mask is not None else None
    else:
        new_layer.lora_A.data = layer.lora_A.detach().clone().contiguous()
        new_layer.lora_B.data = layer.lora_B.detach().clone().contiguous()
        new_layer.input_mask = layer.input_mask.detach().clone().contiguous() if layer.input_mask is not None else None
    new_layer.output_mask = layer.output_mask.detach().clone().contiguous() if layer.output_mask is not None else None
    new_layer.bottleneck_mask = layer.bottleneck_mask.detach().clone().contiguous() if layer.bottleneck_mask is not None else None
    if teacher_pruned_in_dim:
        assert isinstance(layer, lora.DistillLinear)
        new_teacher_retained_indices_to_select = list(set(range(layer.teacher_lora_A.shape[1])) - set(teacher_pruned_in_dim))
        new_teacher_retained_indices_to_select = sorted(new_teacher_retained_indices_to_select)
        new_layer.teacher_lora_A.data = layer.teacher_lora_A[:, new_teacher_retained_indices_to_select].detach().clone().contiguous()
        new_layer.teacher_lora_B.data = layer.teacher_lora_B.detach().clone().contiguous()
    elif isinstance(layer, lora.DistillLinear):
        new_layer.teacher_lora_A.data = layer.teacher_lora_A.detach().clone().contiguous()
        new_layer.teacher_lora_B.data = layer.teacher_lora_B.detach().clone().contiguous()
    return new_layer
    

def shrink_pruning_lora_bottleneckdim(layer: Union[lora.PruningLinear, lora.DistillLinear], pruned_bottleneck_dim: List[int], teacher_pruned_bottleneck_dim: Optional[List[int]] = None) -> Union[lora.PruningLinear, lora.DistillLinear]:
    new_r = layer.r - len(pruned_bottleneck_dim)
    # Creating the new layer based on the pruning shape
    new_alpha = layer.lora_alpha * new_r / layer.r if not ADJUST_SCALING else layer.lora_alpha
    if isinstance(layer, lora.DistillLinear):
        new_teacher_r = layer.teacher_r - len(teacher_pruned_bottleneck_dim) if teacher_pruned_bottleneck_dim is not None else layer.teacher_r
        new_teacher_alpha = layer.teacher_lora_alpha * new_teacher_r / layer.teacher_r if not ADJUST_SCALING else layer.teacher_lora_alpha
        new_layer = lora.DistillLinear(layer.in_features, layer.out_features, r=new_r, teacher_r=new_teacher_r, lora_alpha=new_alpha, teacher_lora_alpha=new_teacher_alpha, lora_dropout=layer.lora_dropout.p if isinstance(layer.lora_dropout, nn.Dropout) else 0, fan_in_fan_out=layer.fan_in_fan_out, merge_weights=layer.merge_weights, out_retained_indices=layer.out_retained_indices, in_retained_indices=layer.in_retained_indices, teacher_out_retained_indices=layer.teacher_out_retained_indices, teacher_in_retained_indices=layer.teacher_in_retained_indices, output_dynamic=layer.output_dynamic, act_fn=layer.act_fn_type, bias=layer.bias is not None).to(layer.weight.device)
    else:
        new_layer = lora.PruningLinear(layer.in_features, layer.out_features, r=new_r, lora_alpha=new_alpha, lora_dropout=layer.lora_dropout.p if isinstance(layer.lora_dropout, nn.Dropout) else 0, fan_in_fan_out=layer.fan_in_fan_out, merge_weights=layer.merge_weights, out_retained_indices=layer.out_retained_indices, in_retained_indices=layer.in_retained_indices, output_dynamic=layer.output_dynamic, act_fn=layer.act_fn_type, bias=layer.bias is not None).to(layer.weight.device)
    new_layer.to(layer.weight.device, dtype=layer.weight.dtype)
    # Copying the parameters from the old layer to the new layer
    new_layer.weight.data = layer.weight.data.detach().clone().contiguous()
    if layer.bias is not None:
        new_layer.bias.data = layer.bias.data.detach().clone().contiguous()
    # Copy the shrinked bottleneck parameters' values
    # TODO: a potential improvement: compress the high-rank decomposition values into the lower-rank decomposition, and add the difference to the weight matrix
    if pruned_bottleneck_dim:
        retained_bottle_neck_indices = list(set(range(layer.r)) - set(pruned_bottleneck_dim))
        retained_bottle_neck_indices = sorted(retained_bottle_neck_indices)
        new_layer.weight.data += layer.calculate_pruned_bottleneck_weights(pruned_bottleneck_dim)
        if ADJUST_SCALING:
            relative_scaling = (new_r / layer.r) ** 0.5
            new_layer.lora_A.data = layer.lora_A.data[retained_bottle_neck_indices, :].detach().clone().contiguous() * relative_scaling
            new_layer.lora_B.data = layer.lora_B.data[:, retained_bottle_neck_indices].detach().clone().contiguous() * relative_scaling
        else:
            new_layer.lora_A.data = layer.lora_A.data[retained_bottle_neck_indices, :].detach().clone().contiguous()
            new_layer.lora_B.data = layer.lora_B.data[:, retained_bottle_neck_indices].detach().clone().contiguous()
        new_layer.bottleneck_mask = layer.bottleneck_mask[retained_bottle_neck_indices].detach().clone().contiguous() if layer.bottleneck_mask is not None else None
    else:
        new_layer.lora_A.data = layer.lora_A.detach().clone().contiguous()
        new_layer.lora_B.data = layer.lora_B.detach().clone().contiguous()
        new_layer.bottleneck_mask = layer.bottleneck_mask.detach().clone().contiguous()
    new_layer.input_mask = layer.input_mask.detach().clone().contiguous() if layer.input_mask is not None else None
    new_layer.output_mask = layer.output_mask.detach().clone().contiguous() if layer.output_mask is not None else None
    if teacher_pruned_bottleneck_dim:
        teacher_retained_bottle_neck_indices = list(set(range(layer.teacher_r)) - set(teacher_pruned_bottleneck_dim))
        teacher_retained_bottle_neck_indices = sorted(teacher_retained_bottle_neck_indices)
        # new_layer.weight.data += layer.calculate_pruned_bottleneck_weights(pruned_bottleneck_dim)
        if ADJUST_SCALING:
            teacher_relative_scaling = (new_teacher_r / layer.teacher_r) ** 0.5
            new_layer.teacher_lora_A.data = layer.teacher_lora_A.data[teacher_retained_bottle_neck_indices, :].detach().clone().contiguous() * teacher_relative_scaling
            new_layer.teacher_lora_B.data = layer.teacher_lora_B.data[:, teacher_retained_bottle_neck_indices].detach().clone().contiguous() * teacher_relative_scaling
        else:
            new_layer.teacher_lora_A.data = layer.teacher_lora_A.data[teacher_retained_bottle_neck_indices, :].detach().clone().contiguous()
            new_layer.teacher_lora_B.data = layer.teacher_lora_B.data[:, teacher_retained_bottle_neck_indices].detach().clone().contiguous()
    elif isinstance(layer, lora.DistillLinear):
        new_layer.teacher_lora_A.data = layer.teacher_lora_A.detach().clone().contiguous()
        new_layer.teacher_lora_B.data = layer.teacher_lora_B.detach().clone().contiguous()
    return new_layer


def expand_pruning_lora_bottleneckdim(layer: Union[lora.PruningLinear, lora.DistillLinear], target_r: int, teacher_target_r: Optional[int] = None) -> Union[lora.PruningLinear, lora.DistillLinear]:
    assert target_r >= layer.r
    if teacher_target_r is not None:
        assert isinstance(layer, lora.DistillLinear) and teacher_target_r > layer.teacher_r
    new_alpha = layer.lora_alpha * target_r / layer.r if not ADJUST_SCALING else layer.lora_alpha
    dtype=layer.weight.dtype
    # Creating the new layer based on the pruning shape
    if isinstance(layer, lora.DistillLinear):
        teacher_target_r = teacher_target_r if teacher_target_r is not None else layer.teacher_r
        new_teacher_alpha = layer.teacher_lora_alpha * teacher_target_r / layer.teacher_r if not ADJUST_SCALING else layer.teacher_lora_alpha
        new_layer = lora.DistillLinear(layer.in_features, layer.out_features, r=target_r, teacher_r=teacher_target_r, lora_alpha=new_alpha, teacher_lora_alpha=new_teacher_alpha, lora_dropout=layer.lora_dropout.p if isinstance(layer.lora_dropout, nn.Dropout) else 0, fan_in_fan_out=layer.fan_in_fan_out, merge_weights=layer.merge_weights, out_retained_indices=layer.out_retained_indices, in_retained_indices=layer.in_retained_indices, teacher_out_retained_indices=layer.teacher_out_retained_indices, teacher_in_retained_indices=layer.teacher_in_retained_indices, output_dynamic=layer.output_dynamic, act_fn=layer.act_fn_type, bias=layer.bias is not None, dtype=dtype).to(layer.weight.device)
    else:
        new_layer = lora.PruningLinear(layer.in_features, layer.out_features, r=target_r, lora_alpha=new_alpha, lora_dropout=layer.lora_dropout.p if isinstance(layer.lora_dropout, nn.Dropout) else 0, fan_in_fan_out=layer.fan_in_fan_out, merge_weights=layer.merge_weights, out_retained_indices=layer.out_retained_indices, in_retained_indices=layer.in_retained_indices, output_dynamic=layer.output_dynamic, act_fn=layer.act_fn_type, bias=layer.bias is not None, dtype=dtype).to(layer.weight.device)
    # Copying the parameters from the old layer to the new layer
    new_layer.weight.data = layer.weight.data.detach().clone().contiguous()
    if layer.bias is not None:
        new_layer.bias.data = layer.bias.data.detach().clone().contiguous()
    # Copy the shrinked bottleneck parameters' values
    # TODO: a potential improvement: compress the high-rank decomposition values into the lower-rank decomposition, and add the difference to the weight matrix
    extra_r = target_r - layer.r
    expanded_lora_A, expanded_lora_B = torch.randn(extra_r, new_layer.lora_A.shape[1], dtype=dtype).to(layer.weight.device), torch.zeros(new_layer.lora_B.shape[0], extra_r, dtype=dtype).to(layer.weight.device)
    nn.init.kaiming_uniform_(expanded_lora_A, a=math.sqrt(5))
    if ADJUST_SCALING:
        relative_scaling = (target_r / layer.r) ** 0.5
        new_layer.lora_A.data = torch.cat([layer.lora_A.data, expanded_lora_A], dim=0).detach().clone().contiguous() * relative_scaling
        new_layer.lora_B.data = torch.cat([layer.lora_B.data, expanded_lora_B], dim=1).detach().clone().contiguous() * relative_scaling
    else:
        new_layer.lora_A.data = torch.cat([layer.lora_A.data, expanded_lora_A], dim=0).detach().clone().contiguous()
        new_layer.lora_B.data = torch.cat([layer.lora_B.data, expanded_lora_B], dim=1).detach().clone().contiguous()
    new_layer.input_mask = layer.input_mask.detach().clone().contiguous() if layer.input_mask is not None else None
    new_layer.output_mask = layer.output_mask.detach().clone().contiguous() if layer.output_mask is not None else None
    new_layer.bottleneck_mask = torch.cat([layer.bottleneck_mask, torch.ones(extra_r).to(layer.bottleneck_mask.device)], dim=0).detach().clone().contiguous() if layer.bottleneck_mask is not None else None
    if isinstance(layer, lora.DistillLinear):
        teacher_extra_r = teacher_target_r - layer.teacher_r
        if teacher_extra_r > 0:
            teacher_expanded_lora_A, teacher_expanded_lora_B = torch.randn(teacher_extra_r, new_layer.teacher_lora_A.shape[1], dtype=dtype).to(layer.weight.device), torch.zeros(new_layer.teacher_lora_B.shape[0], teacher_extra_r, dtype=dtype).to(layer.weight.device)
            nn.init.kaiming_uniform_(teacher_expanded_lora_A, a=math.sqrt(5))
            if ADJUST_SCALING:
                teacher_relative_scaling = (teacher_target_r / layer.teacher_r) ** 0.5
                new_layer.teacher_lora_A.data = torch.cat([layer.teacher_lora_A.data, teacher_expanded_lora_A], dim=0).detach().clone().contiguous() * teacher_relative_scaling
                new_layer.teacher_lora_B.data = torch.cat([layer.teacher_lora_B.data, teacher_expanded_lora_B], dim=1).detach().clone().contiguous() * teacher_relative_scaling
            else:
                new_layer.teacher_lora_A.data = torch.cat([layer.teacher_lora_A.data, teacher_expanded_lora_A], dim=0).detach().clone().contiguous()
                new_layer.teacher_lora_B.data = torch.cat([layer.teacher_lora_B.data, teacher_expanded_lora_B], dim=1).detach().clone().contiguous()
        else:
            new_layer.teacher_lora_A.data = layer.teacher_lora_A.data.detach().clone().contiguous()
            new_layer.teacher_lora_B.data = layer.teacher_lora_B.data.detach().clone().contiguous()
    return new_layer


def shrink_pruning_lora(layer: Union[lora.PruningLinear, lora.DistillLinear], pruned_out_dim: Optional[List[int]] = None, pruned_bottleneck_dim: Optional[List[int]] = None, pruned_in_dim: Optional[List[int]] = None, teacher_pruned_out_dim: Optional[List[int]] = None, teacher_pruned_bottleneck_dim: Optional[List[int]] = None, teacher_pruned_in_dim: Optional[List[int]] = None) -> Union[lora.PruningLinear, lora.DistillLinear]:
    if pruned_out_dim or teacher_pruned_out_dim:
        new_layer = shrink_pruning_lora_outdim(layer, pruned_out_dim, teacher_pruned_out_dim)
    else:
        new_layer = layer
    if pruned_in_dim or teacher_pruned_in_dim:
        new_layer = shrink_pruning_lora_indim(new_layer, pruned_in_dim, teacher_pruned_in_dim)
    if pruned_bottleneck_dim or teacher_pruned_bottleneck_dim:
        new_layer = shrink_pruning_lora_bottleneckdim(new_layer, pruned_bottleneck_dim, teacher_pruned_bottleneck_dim)
    return new_layer

def shrink_and_expand_pruning_lora(layer: Union[lora.PruningLinear, lora.DistillLinear], target_r: int, pruned_out_dim: Optional[List[int]] = None, pruned_bottleneck_dim: Optional[List[int]] = None, pruned_in_dim: Optional[List[int]] = None, teacher_pruned_out_dim: Optional[List[int]] = None, teacher_pruned_bottleneck_dim: Optional[List[int]] = None, teacher_pruned_in_dim: Optional[List[int]] = None) -> Union[lora.PruningLinear, lora.DistillLinear]:
    new_layer = shrink_pruning_lora(layer, pruned_out_dim, pruned_bottleneck_dim, pruned_in_dim, teacher_pruned_out_dim, teacher_pruned_bottleneck_dim, teacher_pruned_in_dim)
    if target_r > new_layer.r:
        new_layer = expand_pruning_lora_bottleneckdim(new_layer, target_r)
    if id(new_layer) != id(layer):
        del_parameter(layer) # delete the old layer if the layer is changed
    return new_layer

def input_constructor(batch_size, seq_len, tokenizer, add_labels=False, output_seq_len=None):
    fake_seq = ""
    for _ in range(seq_len - 2):  # ignore the two special tokens [CLS] and [SEP]
      fake_seq += tokenizer.pad_token
    inputs = tokenizer([fake_seq] * batch_size,
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    inputs = dict(inputs)
    if add_labels:
        labels = torch.tensor([1] * batch_size)
        inputs.update({"labels": labels})
    if output_seq_len is not None:
        decoder_input_ids = torch.tensor([tokenizer.pad_token_id] * batch_size).unsqueeze(1)
        decoder_input_ids = decoder_input_ids.repeat(1, output_seq_len)
        inputs.update({"decoder_input_ids": decoder_input_ids})
    return inputs


@torch.no_grad()
def bench_latency(model, batch_size: int = 32, seq_len: int = 128, tokenizer = None, dataloader: Optional[DataLoader] = None, warm_steps=3, num_reps=10, model_generative=False):
    model.eval()
    if dataloader is not None:
        inputs = next(iter(dataloader))
        inputs = {k: torch.stack(v, dim=1).to(model.device) if isinstance(v, list) else v.to(model.device) for k, v in inputs.items()}
    else:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided if dataloader is not provided")
        if model_generative:
            inputs = input_constructor(batch_size, seq_len, tokenizer, output_seq_len=2)
        else:
            inputs = input_constructor(batch_size, seq_len, tokenizer)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    for _ in range(warm_steps):
        model(**inputs, return_dict=False)

    timings = []
    start_mems = []
    end_mems = []
    diff_mems = []
    MB = 1024.0 * 1024.0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    for _ in range(num_reps):
        if dataloader is None:
            start_mem = torch.cuda.max_memory_allocated() / MB
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(**inputs, return_dict=False)
            torch.cuda.synchronize()
            end = time.perf_counter()
            inference_time = end - start
            end_mem = torch.cuda.max_memory_allocated() / MB
            diff_mem = end_mem - start_mem
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        else:
            inference_time = 0
            for i, batch in enumerate(dataloader):
                batch = {k: torch.stack(v, dim=1).to(model.device) if isinstance(v, list) else v.to(model.device) for k, v in batch.items()}
                torch.cuda.synchronize()
                if i == 0:
                    start_mem = torch.cuda.max_memory_allocated() / MB
                start = time.perf_counter()
                model(**inputs, return_dict=False)
                torch.cuda.synchronize()
                end = time.perf_counter()
                inference_time += end - start
                if i == 0:
                    end_mem = torch.cuda.max_memory_allocated() / MB
                    diff_mem = end_mem - start_mem
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
        timings.append(inference_time)
        start_mems.append(start_mem)
        end_mems.append(end_mem)
        diff_mems.append(diff_mem)

    timings = torch.as_tensor(timings, dtype=torch.float32)
    start_mems = torch.as_tensor(start_mems, dtype=torch.float32)
    end_mems = torch.as_tensor(end_mems, dtype=torch.float32)
    diff_mems = torch.as_tensor(diff_mems, dtype=torch.float32)
    t_mean = timings.mean().item()
    t_std = timings.std().item()
    sm_mean = start_mems.mean().item()
    em_mean = end_mems.mean().item()
    dm_mean = diff_mems.mean().item()
    sm_std = start_mems.std().item()
    em_std = end_mems.std().item()
    dm_std = diff_mems.std().item()
    result = {
        't_mean': t_mean,
        't_std': t_std,
        'sm_mean': sm_mean,
        'sm_std': sm_std,
        'em_mean': em_mean,
        'em_std': em_std,
        'dm_mean': dm_mean,
        'dm_std': dm_std,
    }
    # logger.info(t_mean, t_std, sm_mean, sm_std, em_mean, em_std, dm_mean, dm_std)
    return result


def compute_throughput(model, tokenizer, batch_size: int = 32, seq_len: int = 128, t0: float = 10.0, t1: float = 60.0, model_generative=False):
    inputs = input_constructor(batch_size, seq_len, tokenizer, output_seq_len=2 if model_generative else None)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    with torch.cuda.amp.autocast():
        while time.time() - start < t0:
            model(**inputs)
    timing = []
    torch.cuda.synchronize()
    with torch.cuda.amp.autocast():
        while sum(timing) < t1:
            start = time.time()
            model(**inputs)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    return batch_size / timing.mean().item()


def efficiency_testing(model, tokenizer, device, batch_sizes=[32, 128], seq_len=128, model_generative=False):
    model = model.to(device)
    model.eval()
    for module in model.modules():
        if isinstance(module, lora.layers.Linear):
            module.eval()
    for p in model.parameters():
        p.requires_grad = False

    overall_results = {}
    for bz in batch_sizes:
        bench_result = bench_latency(model, batch_size=bz, seq_len=seq_len, tokenizer=tokenizer, model_generative=model_generative)
        throughput_result = compute_throughput(model, tokenizer, batch_size=bz, seq_len=seq_len, model_generative=model_generative)
        overall_results.update({
            **{'bz{}_'.format(bz) + k: v for k, v in bench_result.items()},
            'bz{}_throughput'.format(bz): throughput_result,
        })

    model_encoder, model_decoder = get_encoder(model), get_decoder(model)
    overall_results.update({
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'encoder_num_parameters': sum(p.numel() for p in model_encoder.parameters()) if model_encoder is not None else 0,
        'decoder_num_parameters': sum(p.numel() for p in model_decoder.parameters()) if model_decoder is not None else 0,
    })
    for module in model.modules():
        if isinstance(module, lora.layers.Linear):
            module.train()
    return overall_results

@torch.no_grad()
def collect_module_inputs(model, inputs, module_func, **kwargs):
    got_inputs = []
    handle = hijack_input(module_func(model), got_inputs)
    inputs = {k: torch.stack(v, dim=1).to(model.device) if isinstance(v, list) else v.to(model.device) for k, v in inputs.items()}
    model(**inputs, **kwargs)
    handle.remove()
    return got_inputs  

@torch.no_grad()
def collect_module_outputs(model, inputs, module_func, **kwargs):
    got_inputs = []
    handle = hijack_output(module_func(model), got_inputs)
    inputs = {k: torch.stack(v, dim=1).to(model.device) if isinstance(v, list) else v.to(model.device) for k, v in inputs.items()}
    model(**inputs, **kwargs)
    handle.remove()
    return got_inputs  
    

def collect_direct_layer_inputs(model, inputs, layer_idx, **kwargs):
    return collect_module_inputs(model, inputs, lambda x: get_layers(x)[layer_idx], **kwargs)
    
    
def compare_layer_inputs_equality(models, inputs):
    equality = []
    for i in range(models[0].config.num_hidden_layers):
        hijacked_inputs = [
            collect_direct_layer_inputs(model, inputs, i)
            for model in models
        ]
        equality.append((hijacked_inputs[0][0][0] == hijacked_inputs[1][0][0]).all().item())
    return equality


def compare_module_inputs_equality(models, inputs, module_func, **kwargs):
    hijacked_inputs = [
        collect_module_inputs(model, inputs, module_func, **kwargs)
        for model in models
    ]
    return (hijacked_inputs[0][0][0], hijacked_inputs[1][0][0], (hijacked_inputs[0][0][0].shape == hijacked_inputs[1][0][0].shape) and (hijacked_inputs[0][0][0] == hijacked_inputs[1][0][0]).all().item())

def compare_module_outputs_equality(models, inputs, module_func, **kwargs):
    hijacked_outputs = [
        collect_module_outputs(model, inputs, module_func, **kwargs)
        for model in models
    ]
    return (hijacked_outputs[0][0][0], hijacked_outputs[1][0][0], (hijacked_outputs[0][0][0].shape == hijacked_outputs[1][0][0].shape) and (hijacked_outputs[0][0][0] == hijacked_outputs[1][0][0]).all().item())

def count_params(model: PreTrainedModel, mode: str='all', return_names: bool = False) -> Union[Tuple[int, int], Tuple[int, int, Dict[str,int]]]:
    if mode == 'tuned':
        params_to_count = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    elif mode == 'untuned':
        params_to_count = {n: p.numel() for n, p in model.named_parameters() if not p.requires_grad}
    else: # all or main
        params_to_count = {n: p.numel() for n, p in model.named_parameters()}
    
    # Exclude embeddings
    params_to_count = {n: p for n, p in params_to_count.items() if 'embeddings' not in n}
    params_to_count = {n: p for n, p in params_to_count.items() if 'shared' not in n}
    params_to_count = {n: p for n, p in params_to_count.items() if 'embed_tokens' not in n}
    if mode == 'main':
        # Exclude adapter parameters
        params_to_count = {n: p for n, p in params_to_count.items() if 'lora' not in n}
        params_to_count = {n: p for n, p in params_to_count.items() if 'transformation' not in n}
        params_to_count = {n: p for n, p in params_to_count.items() if 'mask' not in n}
        # Exclude cls, qa, and lm heads
        params_to_count = {n: p for n, p in params_to_count.items() if 'classifier' not in n}
        params_to_count = {n: p for n, p in params_to_count.items() if 'qa_output' not in n}
        params_to_count = {n: p for n, p in params_to_count.items() if 'lm_head' not in n}
    if return_names:
        return sum(params_to_count.values()), len(params_to_count), params_to_count
    else:
        return sum(params_to_count.values()), len(params_to_count)
    

# We use equation and constants in the model's config to directly count the number of parameters in the model
def count_pruned_params(model: PreTrainedModel, zs: Dict[str, torch.Tensor], mode: str = 'all') -> int:
    config = model.config
    dim_per_head = config.hidden_size // config.num_attention_heads
    if mode == 'all':
        return (2 * config.lora_r + 4 * config.hidden_size + 3) * dim_per_head * (zs['head_z'] == 0).sum().item() + (2 * config.hidden_size + 1) * (zs['intermediate_z'] == 0).sum().item()
    elif mode == 'tuned':
        return 2 * config.lora_r * dim_per_head * (zs['head_z'] == 0).sum().item()
    elif mode == 'untuned':
        return ((4 * config.hidden_size + 3) * dim_per_head) * (zs['head_z'] == 0).sum().item() + (2 * config.hidden_size + 1) * (zs['intermediate_z'] == 0).sum().item()
    
def fisher_mask_score(m: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Compute the fisher mask score for a given mask and fisher matrix.
    
    returned score: m * f * m^T
    """
    assert m.ndim == 1 and f.ndim == 2
    return torch.matmul(torch.matmul(m, f), m.T).item()
    
    
def random_switch(t: torch.Tensor) -> torch._tensor_str:
    ones = t.nonzero().squeeze()
    zeros = (1 - t).nonzero().squeeze()
    select_one = torch.randint(0, ones.shape[0], (1,))
    select_zero = torch.randint(0, zeros.shape[0], (1,))
    returned_t = t.clone()
    returned_t[ones[select_one]] = 0
    returned_t[zeros[select_zero]] = 1
    return returned_t

def run_fisher_mask_score_test(trainer, scorer, intermediate_mask: torch.Tensor, head_mask: torch.Tensor, rearranged_intermediate_mask: torch.Tensor, rearranged_head_mask: torch.Tensor, layer_idx: int = 8):
    num_heads = trainer.model.config.num_attention_heads
    original = torch.cat([intermediate_mask[layer_idx], head_mask[layer_idx]])
    rearranged = torch.cat([rearranged_intermediate_mask[layer_idx], rearranged_head_mask[layer_idx]])
    combined_grads = torch.cat([scorer.intermediate_grads[:, layer_idx, :], scorer.head_grads[:, layer_idx, :]], dim=1)
    combined_fisher = torch.matmul(combined_grads.T, combined_grads)
    original_score = fisher_mask_score(original, combined_fisher)
    rearranged_score = fisher_mask_score(rearranged, combined_fisher)
    for i in range(100):
        new_m = torch.cat([random_switch(rearranged[:3072]), random_switch(rearranged[-num_heads:])])
        res[tuple(new_m.cpu().tolist())] = fisher_mask_score(new_m, combined_fisher)
    best_mask = sorted(res.items(), key=lambda x: x[1], reverse=True)[0][0]
    best_mask = torch.Tensor(best_mask).to(rearranged.device)
    res[tuple(original.cpu().tolist())] = original_score
    res[tuple(rearranged.cpu().tolist())] = rearranged_score
    res = {
        k: {
            'fisher_score': v,
        }
        for k, v in res.items()
    }
    for k in res:
        t = torch.Tensor(k).to(rearranged.device)
        trainer.model.head_mask[layer_idx], trainer.model.intermediate_mask[layer_idx] = t[-num_heads:], t[:-num_heads]
        metrics = trainer.evaluate()
        res[k]['acc'] = metrics['eval_accuracy']
        res[k]['loss'] = metrics['eval_loss']
    return res

def calculate_fisher(head_grad: torch.Tensor, intermediate_grad: torch.Tensor, head_mask: torch.Tensor, intermediate_mask: torch.Tensor) -> float:
    composed_grad = torch.cat([intermediate_grad, head_grad], dim=1)
    fisher = torch.matmul(composed_grad.T, composed_grad)
    composed_masks = torch.cat([intermediate_mask, head_mask], dim=0)
    return (1 - composed_masks) @ fisher @ (1 - composed_masks).T


def sum_fisher_score(head_grads: torch.Tensor, intermediate_grads: torch.Tensor, head_mask: torch.Tensor, intermediate_mask: torch.Tensor, deduplication: bool = False):
    summation = 0
    num_hidden_layers = head_grads.shape[1]
    for i in range(2 * num_hidden_layers - 1):
        if i % 2 == 0:
            layer_idx = i // 2
            summation += calculate_fisher(head_grads[:, layer_idx, :], intermediate_grads[:, layer_idx, :], head_mask[layer_idx], intermediate_mask[layer_idx])
        else:
            intermediate_layer_idx = i // 2
            head_layer_idx = i // 2 + 1
            summation += calculate_fisher(head_grads[:, head_layer_idx, ], intermediate_grads[:, intermediate_layer_idx, :],intermediate_mask[intermediate_layer_idx], head_mask[head_layer_idx])
    if deduplication:
        for i in range(num_hidden_layers):
            head_fisher = torch.matmul((head_grads[:, i, :]).T, head_grads[:, i, :])
            intermediate_fisher = torch.matmul((intermediate_grads[:, i, :]).T, intermediate_grads[:, i, :])
            if i > 0:
                summation -= (1 - head_mask[i]) @ head_fisher @ (1 - head_mask[i]).T
            if i < num_hidden_layers - 1:
                summation -= (1 - intermediate_mask[i]) @ intermediate_fisher @ (1 - intermediate_mask[i]).T
    return summation


def lora_to_linear(layer: lora.layers.LoRALayer) -> torch.nn.Linear:
    """Convert a LoRALayer to a Linear layer."""
    layer.eval()
    linear_cls = lora.SelectLinear if isinstance(layer, lora.SelectLinear) else torch.nn.Linear
    new_layer = linear_cls(layer.in_features, layer.out_features, bias=layer.bias is not None, dtype=layer.weight.dtype)
    new_layer.weight.data = layer.weight.data.detach().clone().contiguous()
    if layer.bias is not None:
        new_layer.bias.data = layer.bias.data.detach().clone().contiguous()
    device = layer.weight.device
    del_parameter(layer)
    return new_layer.to(device)


def linear_to_lora(layer: torch.nn.Linear, r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0, fan_in_fan_out: bool = False, merge_weights: bool = True) -> lora.layers.Linear:
    """Convert a Linear layer to a LoRALayer."""
    new_layer = lora.layers.Linear(layer.in_features, layer.out_features, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, fan_in_fan_out=fan_in_fan_out, merge_weights=merge_weights, bias=layer.bias is not None, dtype=layer.weight.dtype)
    new_layer.weight.data = layer.weight.data.detach().clone().contiguous()
    if hasattr(layer, 'bias') and layer.bias is not None:
        new_layer.bias.data = layer.bias.data.detach().clone().contiguous()
    device = layer.weight.device
    del_parameter(layer, ['weight', 'bias'])
    return new_layer.to(device)

def lora_to_prunelora(layer: lora.layers.Linear, r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0, fan_in_fan_out: bool = False, merge_weights: bool = True, retained_indices: Optional[List[int]] = None, out_retained_indices: Optional[List[int]] = None, in_retained_indices: Optional[List[int]] = None, output_dynamic: bool = True, copy_to_lora: bool = True, act_fn: Optional[str] = None) -> lora.layers.PruningLinear:
    """Convert a LoRA Linear layer to a PruningLinear layer."""
    new_layer = lora.layers.PruningLinear(layer.in_features, layer.out_features, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, retained_indices=retained_indices, out_retained_indices=out_retained_indices, in_retained_indices=in_retained_indices, output_dynamic=output_dynamic, fan_in_fan_out=fan_in_fan_out, merge_weights=merge_weights, bias=layer.bias is not None, act_fn=act_fn, dtype=layer.weight.dtype)
    if copy_to_lora:
        layer.train()
        if retained_indices is not None:
            if output_dynamic:
                out_retained_indices = retained_indices
            else:
                in_retained_indices = retained_indices
        new_layer.lora_A.data = layer.lora_A.data[:, torch.tensor(in_retained_indices).to(new_layer.weight.device)].detach().clone().contiguous() if in_retained_indices is not None else layer.lora_A.data
        new_layer.lora_B.data = layer.lora_B.data[torch.tensor(out_retained_indices).to(new_layer.weight.device), :].detach().clone().contiguous() if out_retained_indices is not None else layer.lora_B.data
    else:
        layer.eval()
    new_layer.weight.data = layer.weight.data.detach().clone().contiguous()
    if hasattr(layer, 'bias') and layer.bias is not None:
        new_layer.bias.data = layer.bias.data.detach().clone().contiguous()
    device = layer.weight.device
    del_parameter(layer, ['weight', 'bias'])
    return new_layer.to(device)

def teacher_svd(equivalent: torch.Tensor, teacher_scaling: float, teacher_r: int):
    u, s, v = torch.svd(equivalent / teacher_scaling)
    teacher_lora_B = u[:, :teacher_r]
    teacher_lora_A = s[:teacher_r].unsqueeze(1) * v[:teacher_r, :]
    error = torch.norm(equivalent - teacher_lora_B @ teacher_lora_A * teacher_scaling)
    return teacher_lora_B, teacher_lora_A, error

def lora_to_distill(layer: lora.layers.Linear, r: int = 64, teacher_r: int = 8, lora_alpha: int = 16, teacher_lora_alpha: int = 16, retained_indices: Optional[List[int]] = None, out_retained_indices: Optional[List[int]] = None, in_retained_indices: Optional[List[int]] = None, teacher_retained_indices: Optional[List[int]] = None, teacher_out_retained_indices: Optional[List[int]] = None, teacher_in_retained_indices: Optional[List[int]] = None, lora_dropout: float = 0, fan_in_fan_out: bool = False, merge_weights: bool = True, copy_to_teacher: bool = False, teacher_svd_init: bool = False) -> lora.layers.DistillLinear:
    """Convert a LoRA Linear layer to a DistillLinear layer."""
    new_layer = lora.layers.DistillLinear(layer.in_features, layer.out_features, r=r, teacher_r=teacher_r, lora_alpha=lora_alpha, teacher_lora_alpha=teacher_lora_alpha, retained_indices=retained_indices, in_retained_indices=in_retained_indices, out_retained_indices=out_retained_indices, teacher_retained_indices=teacher_retained_indices, teacher_in_retained_indices=teacher_in_retained_indices, teacher_out_retained_indices=teacher_out_retained_indices, lora_dropout=lora_dropout, fan_in_fan_out=fan_in_fan_out, merge_weights=merge_weights, bias=layer.bias is not None, dtype=layer.weight.dtype)
    if copy_to_teacher:
        layer.train()
        new_layer.weight.data = layer.weight.data.detach().clone().contiguous()
        # Teacher init the same as the stuent LoRA layers
        new_layer.lora_A.data = layer.lora_A.data.detach().clone().contiguous()
        new_layer.lora_B.data = layer.lora_B.data.detach().clone().contiguous()
        new_layer.teacher_lora_A.data = layer.lora_A.data.detach().clone().contiguous()
        new_layer.teacher_lora_B.data = layer.lora_B.data.detach().clone().contiguous()
    elif teacher_svd_init:
        # Using the svd function to get the decomposed teacher lora_A and lora_B, while keeping the original lora_A and lora_B
        new_layer.weight.data = layer.weight.data.detach().clone().contiguous()
        new_layer.lora_A.data = layer.lora_A.data.detach().clone().contiguous()
        new_layer.lora_B.data = layer.lora_B.data.detach().clone().contiguous()
        if teacher_r > 0:
            equivalent = layer.scaling * (layer.lora_B.data @ layer.lora_A.data)
            teacher_lora_B, teacher_lora_A, error = teacher_svd(equivalent, new_layer.teacher_scaling, new_layer.teacher_r)
            new_layer.teacher_lora_A.data = teacher_lora_A.detach().clone().contiguous()
            new_layer.teacher_lora_B.data = teacher_lora_B.detach().clone().contiguous()
    else:
        layer.eval()
        new_layer.weight.data = layer.weight.data.detach().clone().contiguous()
    if hasattr(layer, 'bias') and layer.bias is not None:
        new_layer.bias.data = layer.bias.data.detach().clone().contiguous()
    device = layer.weight.device
    for mask_name in ['input_mask', 'output_mask', 'bottleneck_mask']:
        mask = getattr(layer, mask_name, None)
        if isinstance(mask, torch.Tensor):
            setting_mask = mask.detach().clone().contiguous().to(device)
            setting_mask.requires_grad = True
            setattr(new_layer, mask_name, setting_mask)
    del_parameter(layer)
    return new_layer.to(device)

def distill_to_lora(layer: lora.layers.DistillLinear, r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0, fan_in_fan_out: bool = False, merge_weights: bool = False) -> lora.layers.Linear:
    """Convert a DistillLinear layer to a lora.Linear layer."""
    layer.train()
    new_layer = lora.layers.PruningLinear(layer.in_features, layer.out_features, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, fan_in_fan_out=fan_in_fan_out, merge_weights=merge_weights, retained_indices=layer.retained_indices, output_dynamic=layer.output_dynamic, bias=layer.bias is not None, act_fn=layer.act_fn_type, dtype=layer.weight.dtype)
    new_layer.weight.data = layer.weight.data.detach().clone().contiguous()
    if hasattr(layer, 'bias') and layer.bias is not None:
        new_layer.bias.data = layer.bias.data.detach().clone().contiguous()
    new_layer.lora_A.data = layer.lora_A.data.detach().clone().contiguous()
    new_layer.lora_B.data = layer.lora_B.data.detach().clone().contiguous()
    device = layer.weight.device
    del_parameter(layer, ['weight', 'bias'])
    return new_layer.to(device)

def get_history_param_allocation(path: str):
    history = torch.load(path, map_location='cpu')
    history_by_step = {}
    attrs = list(history.keys())
    for attr, histories in history.items():
        for his in histories:
            his['tuning_in_dim'] = list(set(range(his['in_features'])) - set(his['pruned_in_dim']))
            his['tuning_out_dim'] = list(set(range(his['out_features'])) - set(his['pruned_out_dim']))
            his['num_tuning_params'] = his['target_r'] * (len(his['tuning_in_dim']) + len(his['tuning_out_dim']))
            if his['step'] not in history_by_step:
                history_by_step[his['step']] = {}
            history_by_step[his['step']][attr] = his
    steps = sorted(list(history_by_step.keys()))
    alloc_shapes = []
    for step in steps:
        alloc_shapes.append(history_by_step[step])
        for attr in attrs:
            if attr not in history_by_step[step]:
                alloc_shapes[-1][attr] = None
    return alloc_shapes
    

def model_layer_switch(model, type_from: str, type_to: str):
    if type_from == 'lora' and type_to == 'linear':
        func_use = lora_to_linear
        src_class = lora.layers.Linear
    elif type_from == 'linear' and type_to == 'lora':
        func_use = linear_to_lora
        src_class = torch.nn.Linear
    elif type_from == 'lora' and type_to == 'distill_lora':
        func_use = lora_to_distill
        src_class = lora.layers.Linear
    else:
        raise NotImplementedError("Only support lora to linear and linear to lora")
    # Changing modules out-of-sequential
    #   Currently not available and not useful, because all LoRA layers exist in transformer layers
    # for name, module in model.named_modules():
    #     if isinstance(module, src_class):
    #         new_module = func_use(module)
    #         setattr(model, name, new_module)
    
    considered_layers = ['query', 'key', 'value']
    # Changing modules for each transformer layer (in sequential)
    #   Currently, only considering q, k, v, and intermediate.dense
    for i in range(model.config.num_hidden_layers):
        attention_module = get_layers(model)[i].attention.self
        for module_name in considered_layers:
            module = getattr(attention_module, module_name)
            if isinstance(module, src_class):
                new_module = func_use(module)
                setattr(attention_module, module_name, new_module)
        intermediate_module = get_layers(model)[i].intermediate
        if isinstance(intermediate_module.dense, src_class):
            new_module = func_use(intermediate_module.dense)
            intermediate_module.dense = new_module
    return model


def random_positive_with_sum(n: int, total: int, nonzero: bool = False) -> List[int]:
    if nonzero:
        return [v + 1 for v in random_positive_with_sum(n, total - n, False)]
    result = []
    i = 0
    while i < n and total > 0:
        if i == n - 1:
            res = total
        else:
            res = np.random.randint(0, total + 1)
        total -= res
        i += 1
        result.append(res)
    if i < n:
        result += [0] * (n - i)
    random.shuffle(result)
    return result

def generate_mask(num_retained: int, num_layer_with_one: Union[List[int], int] = -1, element_per_layer: int = 12, num_layers: int = 12) -> torch.Tensor:
    if isinstance(num_layer_with_one, int):
        if num_layer_with_one == -1:
            dist = random_positive_with_sum(num_layers, num_retained)
        else:
            dist = random_positive_with_sum(num_layer_with_one, num_retained, nonzero=True) + [0] * (num_layers - num_layer_with_one)
        random.shuffle(dist)
    else:
        gen_dist = random_positive_with_sum(len(num_layer_with_one), num_retained, nonzero=True)
        dist = [0] * num_layers
        for i, num in enumerate(num_layer_with_one):
            dist[num] = gen_dist[i]
    position_per_layer = [random.sample(range(element_per_layer), d) for d in dist]
    mask = torch.zeros(num_layers, element_per_layer)
    for i, poses in enumerate(position_per_layer):
        for pos in poses:
            mask[i, pos] = 1
    return mask


def init_masks(model):
    head_mask_shapes, intermediate_mask_shapes = [], []
    for layer in get_layers(model):
        head_mask_shapes.append(layer.attention.self.num_attention_heads)
        intermediate_mask_shapes.append(layer.intermediate.dense.out_features)
    if all([v == head_mask_shapes[0] for v in head_mask_shapes]):
        head_mask = torch.ones(model.config.num_hidden_layers, head_mask_shapes[0]).to(model.device)
    else:
        head_mask = [torch.ones(v) for v in head_mask_shapes]
    if all([v == intermediate_mask_shapes[0] for v in intermediate_mask_shapes]):
        intermediate_mask = torch.ones(model.config.num_hidden_layers, intermediate_mask_shapes[0]).to(model.device)
    else:
        intermediate_mask = [torch.ones(v) for v in intermediate_mask_shapes]
    return head_mask, intermediate_mask
# head_mask: ii
# intermediate_mask: 1344


def sequential_neuron_mask_generation(mask: torch.Tensor) -> List[torch.Tensor]:
    masks = []
    now_masks = [v for v in mask.clone()]
    for i in range(mask.max().long().item()):
        gen_mask = [(v != i).float() for v in now_masks]
        masks.append(gen_mask)
        if mask.ndim == 2:
            now_masks = [v[gen.nonzero().squeeze()] if v.size() else torch.Tensor([]) for gen, v in zip(gen_mask, now_masks)]
        elif mask.ndim == 3:
            now_masks = [
                [v[gen.nonzero().squeeze()] if v.size() else torch.Tensor([]) for gen, v in zip(current_gen_mask, current_now_mask)]
                for current_gen_mask, current_now_mask in zip(gen_mask, now_masks)
            ]
        else:
            raise NotImplementedError
    return masks

def sequential_head_mask_generation(mask: torch.Tensor) -> List[torch.Tensor]:
    masks = []
    for i in range(mask.max().long().item()):
        gen_mask = [(v != i).float() for v in mask]
        masks.append(gen_mask)
    return masks

def compare_parameters(model_one, model_two):
    model_one_params, model_two_params = dict(model_one.named_parameters()), dict(model_two.named_parameters())
    equal_params = sum((model_one_params[k] == model_two_params[k]).sum().item() for k in model_one_params.keys() if k in model_two_params)
    equal_vars = [k for k in model_one_params.keys() if k in model_two_params and (model_one_params[k] == model_two_params[k]).all().item()]
    return equal_params, equal_vars

def parse_collected_salience(path: str):
    salience_collected = torch.load(path, map_location='cpu')
    param_names = list(salience_collected[0]['grads'])
    stacked_salience = {}
    for n in param_names:
        stacked_salience[n] = torch.stack([s['grads'][n] for s in salience_collected], dim=0)
    query_template = 'bert.encoder.layer.%d.attention.self.query.lora_B'
    value_template = 'bert.encoder.layer.%d.attention.self.value.lora_B'
    neuron_template = 'bert.encoder.layer.%d.intermediate.dense.lora_B'
    query_output_salience = torch.stack([
        torch.stack([stacked_salience[query_template % i].sum(dim=2)[:, j:j+64].sum(dim=1) for j in range(0, 768, 64)], dim=1) for i in range(12)
    ], dim=1)
    query_output_fisher = query_output_salience.pow(2)
    value_output_salience = torch.stack([
        torch.stack([stacked_salience[value_template % i].sum(dim=2)[:, j:j+64].sum(dim=1) for j in range(0, 768, 64)], dim=1) for i in range(12)
    ], dim=1)
    value_output_fisher = value_output_salience.pow(2)
    neuron_output_salience = torch.stack([
        torch.stack([stacked_salience[neuron_template % i].sum(dim=2)[:, j:j+64].sum(dim=1) for j in range(0, 768, 64)], dim=1) for i in range(12)
    ], dim=1)
    neuron_output_fisher = neuron_output_salience.pow(2)
    head_fisher = torch.stack([v['head_mask_grad'] for v in salience_collected], dim=0).pow(2)
    intermediate_fisher = torch.stack([v['intermediate_mask_grad'] for v in salience_collected], dim=0).pow(2)
    return {
        'raw': salience_collected,
        'query_fisher': query_output_fisher,
        'value_fisher': value_output_fisher,
        'neuron_fisher': neuron_output_fisher,
        'head_fisher': head_fisher,
        'intermediate_fisher': intermediate_fisher,
    }
    
def load_grafting_masks(model: PreTrainedModel, masks: Dict[str, torch.Tensor]):
    named_modules = dict(model.named_modules())
    for layer_name, ms in masks.items():
        if layer_name in named_modules:
            layer = named_modules[layer_name]
            if isinstance(layer, lora.PruningLinear):
                for m, value in ms.items():
                    setattr(layer, m, value)
            else:
                raise ValueError("Layer %s is not a PruningLinear layer" % layer_name)
        else:
            raise ValueError("Layer %s not found in model" % layer_name)
        
@torch.no_grad()
def kurtosis(a: torch.Tensor, axis: int = 0, fisher: bool = True, bias: bool = True):
    """Compute the kurtosis (Fisher or Pearson) of a distribution.

    Kurtosis is the fourth central moment divided by the square of the
    variance. If Fisher's definition is used, then 3.0 is subtracted from
    the result to give 0.0 for a normal distribution.

    If bias is False then the kurtosis is calculated using k statistics to
    eliminate bias coming from biased moment estimators

    Use `kurtosistest` to see if result is close enough to normal.

    Parameters
    ----------
    a : torch.Tensor
        Data for which the kurtosis is calculated.
    axis : int or None, optional
        Axis along which the kurtosis is calculated. Default is 0.
        If None, compute over the whole array `a`.

    Returns
    -------
    kurtosis : torch.Tensor
        The kurtosis of values along an axis, returning NaN where all values
        are equal.

    """
    # Compute the mean of the tensor
    std = torch.std(a, axis, keepdim=True)
    mu = torch.mean(a, axis, keepdim=True)

    # Compute the centered values
    centered = a - mu

    # Compute the zscore
    # Set zscores to 0 where std is 0
    zscores = centered / std
    zscores = torch.where(torch.isnan(zscores), torch.zeros_like(zscores), zscores)

    # Compute kurtosis
    kurt = torch.mean(zscores.pow(4), axis, keepdim=True).squeeze()

    return kurt - 3 if fisher else kurt