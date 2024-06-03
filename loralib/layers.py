#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
            self.merged = False
    
    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(
                    x, self.lora_A.T, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse
                )
                result += (after_A @ self.lora_B.T) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)
          

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged and mode:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
                
                
def select_wandb(weight: nn.Parameter, bias: nn.Parameter, in_retained_indices: Optional[torch.Tensor] = None, out_retained_indices: Optional[torch.Tensor] = None):
    # Handling retained-indices situation
    if in_retained_indices is not None:
        selected_weight = weight.index_select(1, in_retained_indices)
    else:
        selected_weight = weight
    if out_retained_indices is not None:
        selected_weight = selected_weight.index_select(0, out_retained_indices)
        selected_bias = bias.index_select(0, out_retained_indices) if bias is not None else None
    else:
        selected_bias = bias
    return selected_weight, selected_bias

def _do_reconstruct_outputs(outputs: torch.Tensor, out_retained_indices: Optional[torch.Tensor], out_features: int) -> torch.Tensor:
        if out_retained_indices is not None:
            padded_outputs = torch.zeros(outputs.shape[:-1] + (out_features,), device=outputs.device)
            padded_outputs[..., out_retained_indices] = outputs
            return padded_outputs
        else:
            return outputs
        
class SelectLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        Linear.__init__(self, *args, **kwargs)

    def forward(self, x: torch.Tensor, use_teacher: bool = False, in_retained_indices: Optional[torch.Tensor] = None, out_retained_indices: Optional[torch.Tensor] = None, reconstruct_output: bool = False):
        selected_weight, selected_bias = select_wandb(self.weight, self.bias, in_retained_indices, out_retained_indices) if not use_teacher else (self.weight, self.bias)
        result = F.linear(x, selected_weight, bias=selected_bias)
        if reconstruct_output:
            return _do_reconstruct_outputs(result, out_retained_indices, self.out_features)
        else:
            return result
    

class PruningLinear(SelectLinear, Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        retained_indices: Optional[List[int]] = None,
        out_retained_indices: Optional[List[int]] = None,
        in_retained_indices: Optional[List[int]] = None,
        output_dynamic: bool = True,
        act_fn: Optional[str] = None,
        **kwargs
    ):
        SelectLinear.__init__(self, in_features, out_features, **kwargs)
        self.act_fn_type = act_fn
        if act_fn is not None:
            merge_weights = False
            if act_fn == 'relu':
                self.act_fn = nn.ReLU()
            else:
                raise ValueError("Unsupported activation function")
        else:
            self.act_fn = None
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        if retained_indices is not None and (out_retained_indices is not None or in_retained_indices is not None):
            raise ValueError("You can only specify retained_indices or out_retained_indices and in_retained_indices")
        if retained_indices is not None:
            if output_dynamic:
                out_retained_indices = retained_indices
                in_retained_indices = None
            else:
                in_retained_indices = retained_indices
                out_retained_indices = None
        self.retained_indices = retained_indices
        self.out_retained_indices = out_retained_indices
        self.in_retained_indices = in_retained_indices
        self._init_transformations()
        
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        self.output_dynamic = output_dynamic
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, len(self.in_retained_indices) if self.in_retained_indices is not None else in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((len(self.out_retained_indices) if self.out_retained_indices is not None else out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        self.input_mask = None
        self.output_mask = None
        self.bottleneck_mask = None
        # Snapshot lists to store the history of the lora weights
        self.history = []
        
    def _init_transformations(self):
        if self.out_retained_indices is None:
            self.out_transformation = None
        else:
            self.out_transformation = nn.Parameter(self.weight.new_zeros(len(self.out_retained_indices), self.out_features))
            for i, idx in enumerate(self.out_retained_indices):
                self.out_transformation.data[i, idx] = 1
            self.out_transformation.requires_grad = False
        if self.in_retained_indices is None:
            self.in_transformation = None
        else:
            self.in_transformation = nn.Parameter(self.weight.new_zeros(self.in_features, len(self.in_retained_indices)))
            for i, idx in enumerate(self.in_retained_indices):
                self.in_transformation.data[idx, i] = 1
            self.in_transformation.requires_grad = False
        
    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged and mode:
            # Make sure that the weights are not merged
            if self.r > 0:
                lora_B_use = self.lora_B * self.output_mask[:, None] if self.output_mask is not None else self.lora_B
                lora_B_use = lora_B_use * self.bottleneck_mask if self.bottleneck_mask is not None else lora_B_use
                lora_A_use = self.lora_A * self.input_mask if self.input_mask is not None else self.lora_A
                transformed_lora = lora_B_use @ lora_A_use
                if self.out_transformation is not None:
                    transformed_lora = self.out_transformation.T @ transformed_lora
                if self.in_transformation is not None:
                    transformed_lora = transformed_lora @ self.in_transformation.T
                self.weight.data -= T(transformed_lora) * self.scaling
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                lora_B_use = self.lora_B * self.output_mask[:, None] if self.output_mask is not None else self.lora_B
                lora_B_use = lora_B_use * self.bottleneck_mask if self.bottleneck_mask is not None else lora_B_use
                lora_A_use = self.lora_A * self.input_mask if self.input_mask is not None else self.lora_A
                transformed_lora = lora_B_use @ lora_A_use
                if self.out_transformation is not None:
                    transformed_lora = self.out_transformation.T @ transformed_lora
                if self.in_transformation is not None:
                    transformed_lora = transformed_lora @ self.in_transformation.T
                self.weight.data += T(transformed_lora) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor, use_teacher: bool = False, in_retained_indices: Optional[torch.Tensor] = None, out_retained_indices: Optional[torch.Tensor] = None, reconstruct_output: bool = False):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            selected_weight, selected_bias = select_wandb(self.weight, self.bias, in_retained_indices, out_retained_indices) if not use_teacher else (self.weight, self.bias)
            result = F.linear(x, T(selected_weight), bias=selected_bias)
            # lora_A: (r, in_features); lora_B: (out_features, r); weight: (out_features, in_features)
            lora_A_use = self.lora_A * self.input_mask if self.input_mask is not None else self.lora_A # (r, in_features)
            lora_A_use = lora_A_use.T if self.bottleneck_mask is None else lora_A_use.T * self.bottleneck_mask # (in_features, r)
            lora_B_use = self.lora_B.T * self.output_mask if self.output_mask is not None else self.lora_B.T # (r, out_features)
            
            # Handling retained-indices situation
            if not use_teacher:
                if in_retained_indices is not None:
                    lora_A_use = lora_A_use.index_select(0, in_retained_indices)
                if out_retained_indices is not None:
                    lora_B_use = lora_B_use.index_select(1, out_retained_indices)

            if self.r > 0 and self.act_fn is not None:
                additive_result = (self.lora_dropout(x) @ lora_A_use)
                additive_result = self.act_fn(additive_result)
                additive_result = (additive_result @ lora_B_use) * self.scaling
                result += additive_result
            elif self.r > 0:
                additive_result = (self.lora_dropout(x) @ lora_A_use @ lora_B_use) * self.scaling
                result += additive_result
            if reconstruct_output:
                result = _do_reconstruct_outputs(result, out_retained_indices, self.out_features)
            return result
        else:
            return super().forward(x, use_teacher=use_teacher, in_retained_indices=in_retained_indices, out_retained_indices=out_retained_indices, reconstruct_output=reconstruct_output)

        
    def set_grafting_mask(self, requires_grad: bool = False):
        if self.r == 0:
            return
        # Only set masks when they are None
        if self.input_mask is None:
            self.input_mask = torch.ones(self.lora_A.shape[1], dtype=self.lora_A.dtype, device=self.weight.device)
        self.input_mask.hidden_size = self.r
        if self.output_mask is None:
            self.output_mask = torch.ones(self.lora_B.shape[0], dtype=self.lora_A.dtype, device=self.weight.device)
        self.output_mask.hidden_size = self.r
        if self.bottleneck_mask is None:
            self.bottleneck_mask = torch.ones(self.r, dtype=self.lora_A.dtype, device=self.weight.device)
        self.bottleneck_mask.hidden_size = self.in_features + self.out_features
        if requires_grad:
            self.input_mask.requires_grad = True
            self.output_mask.requires_grad = True
            self.bottleneck_mask.requires_grad = True
        else:
            self.input_mask.requires_grad = False
            self.output_mask.requires_grad = False
            self.bottleneck_mask.requires_grad = False

    def remove_grafting_mask(self):
        self.input_mask = None
        self.output_mask = None
        self.bottleneck_mask = None

    def calculate_pruned_bottleneck_weights(self, pruned_bottleneck_dim: List[int]):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        lora_B_use = self.lora_B * self.output_mask[:, None] if self.output_mask is not None else self.lora_B
        lora_B_use = lora_B_use * self.bottleneck_mask if self.bottleneck_mask is not None else lora_B_use
        lora_A_use = self.lora_A * self.input_mask if self.input_mask is not None else self.lora_A
        pruned_lora_A = lora_A_use.index_select(0, torch.tensor(pruned_bottleneck_dim).int().to(lora_A_use.device))
        pruned_lora_B = lora_B_use.index_select(1, torch.tensor(pruned_bottleneck_dim).int().to(lora_B_use.device))
        if self.r > 0:
            transformed_lora_B = self.out_transformation.T @ pruned_lora_B if self.out_transformation is not None else pruned_lora_B
            transformed_lora_A = pruned_lora_A @ self.in_transformation.T if self.in_transformation is not None else pruned_lora_A
            weight = T(transformed_lora_B @ transformed_lora_A) * self.scaling
            return weight
        else:
            return None

    def calculate_pruned_outdim_weights(self, pruned_out_dim: Optional[List[int]] = None, pruned_bottleneck_dim: Optional[List[int]] = None, pruned_in_dim: Optional[List[int]] = None):
        if self.act_fn is not None:
            return None
        else:
            lora_B_use = self.lora_B * self.output_mask[:, None] if self.output_mask is not None else self.lora_B
            lora_B_use = lora_B_use * self.bottleneck_mask if self.bottleneck_mask is not None else lora_B_use
            lora_A_use = self.lora_A * self.input_mask if self.input_mask is not None else self.lora_A
            return calculate_pruned_equivalent_weights(self, lora_A_use, lora_B_use, self.r, self.scaling, self.in_transformation, self.out_transformation, pruned_out_dim, pruned_bottleneck_dim, pruned_in_dim)
        
    def create_transformation(self, retained_indices: List[int], size: int, dtype: torch.dtype, device: torch.device, construct_pruned: bool = False, transpose: bool = False):
        if construct_pruned:
            retained_indices = sorted(list(set(range(size)) - set(retained_indices)))
        transformation = torch.zeros(len(retained_indices), size, dtype=dtype, device=device)
        for i, index in enumerate(retained_indices):
            transformation[i, index] = 1
        if transpose:
            transformation = transformation.T
        return transformation
    
    def get_r_sensitivity(self, abs=True):
        a, a_grad = self.lora_A, getattr(self.lora_A, 'grad', None)
        if a_grad is None:
            # print("Warning: lora_A's gradient is None", flush=True)
            return None
        else:
            with torch.no_grad():
                if abs:
                    return (a * a_grad).sum(dim=1).abs()
                else:
                    return (a * a_grad).sum(dim=1)
        
        
    def log_history(self, step: int, pruned_out_dim: Optional[List[int]] = None, pruned_bottleneck_dim: Optional[List[int]] = None, pruned_in_dim: Optional[List[int]] = None, target_r: Optional[int] = None):
        self.history.append({
            'step': step,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'r': self.r,
            'target_r': target_r if target_r is not None else self.r,
            'weight': self.weight.detach().clone().cpu(),
            'lora_A': self.lora_A.detach().clone().cpu(),
            'lora_B': self.lora_B.detach().clone().cpu(),
            'scaling': self.scaling,
            'in_transformation': self.in_transformation.detach().clone().cpu() if self.in_transformation is not None else None,
            'out_transformation': self.out_transformation.detach().clone().cpu() if self.out_transformation is not None else None,
            'pruned_out_dim': pruned_out_dim,
            'pruned_bottleneck_dim': pruned_bottleneck_dim,
            'pruned_in_dim': pruned_in_dim,
        })
        
    def restore_dims(self):
        if len(self.history) == 0:
            return
        history_use = self.history[-1]
        if self.lora_A is None and self.lora_B is None:
            self.eval()
            self.lora_A, self.lora_B = nn.Parameter(history_use['lora_A'].detach().clone().to(self.weight.device)), nn.Parameter(history_use['lora_B'].detach().clone().to(self.weight.device))
            self.scaling = history_use['scaling']
            self.r = self.lora_A.shape[0]
            self.train()
            self.lora_A.requires_grad, self.lora_B.requires_grad = True, True
            return
        A_requires_grad, B_requires_grad = self.lora_A.requires_grad, self.lora_B.requires_grad
        tuned_lora_A_projected = self.lora_A @ self.in_transformation.T if self.in_transformation is not None else self.lora_A
        tuned_lora_B_projected = self.out_transformation.T @ self.lora_B if self.out_transformation is not None else self.lora_B
        pruned_lora_A = history_use['lora_A'].index_select(1, torch.tensor(history_use['pruned_in_dim']).int()).to(self.weight.device)
        pruned_lora_B = history_use['lora_B'].index_select(0, torch.tensor(history_use['pruned_out_dim']).int()).to(self.weight.device)
        if history_use['pruned_bottleneck_dim']:
            bottleneck_transformation = self.create_transformation(history_use['pruned_bottleneck_dim'], history_use['lora_A'].shape[0], self.lora_A.dtype, self.lora_A.device, construct_pruned=True)
            tuned_lora_A_projected = bottleneck_transformation.T @ tuned_lora_A_projected
            tuned_lora_B_projected = tuned_lora_B_projected @ bottleneck_transformation
        else:
            extra_r = self.lora_A.shape[0] - history_use['lora_A'].shape[0]
            pruned_lora_A = torch.cat([pruned_lora_A, torch.zeros(extra_r, pruned_lora_A.shape[1], dtype=pruned_lora_A.dtype, device=pruned_lora_A.device)])
            pruned_lora_B = torch.cat([pruned_lora_B, torch.zeros(pruned_lora_B.shape[0], extra_r, dtype=pruned_lora_B.dtype, device=pruned_lora_B.device)], dim=1)
        if history_use['pruned_bottleneck_dim']:
            tuned_lora_A_projected *= (self.scaling / history_use['scaling']) ** 0.5
            tuned_lora_B_projected *= (self.scaling / history_use['scaling']) ** 0.5
        if history_use['pruned_out_dim']:
            pruned_outdim_transformation = self.create_transformation(history_use['pruned_out_dim'], self.out_features, self.lora_B.dtype, self.lora_B.device)
            pruned_lora_B_projected = pruned_outdim_transformation.T @ pruned_lora_B
            merged_lora_B = tuned_lora_B_projected + pruned_lora_B_projected
        else:
            merged_lora_B = tuned_lora_B_projected
        if history_use['pruned_in_dim']:
            pruned_indim_transformation = self.create_transformation(history_use['pruned_in_dim'], self.in_features, self.lora_A.dtype, self.lora_A.device)
            pruned_lora_A_projected = pruned_lora_A @ pruned_indim_transformation
            merged_lora_A = tuned_lora_A_projected + pruned_lora_A_projected
        else:
            merged_lora_A = tuned_lora_A_projected
        self.eval()
        self.lora_A.data = merged_lora_A
        self.lora_B.data = merged_lora_B
        self.in_transformation, self.out_transformation = None, None
        self.in_retained_indices, self.out_retained_indices = None, None
        self.r = merged_lora_A.shape[0]
        self.scaling = self.lora_alpha / self.r
        self.train()
        self.lora_A.requires_grad, self.lora_B.requires_grad = A_requires_grad, B_requires_grad

    def refill_input(self):
        raise NotImplementedError
    
    def refill_output(self):
        raise NotImplementedError

    def refill_bottleneck(self):
        raise NotImplementedError
        
    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        if self.input_mask is not None:
            self.input_mask = self.input_mask.to(*args, **kwargs)
        if self.output_mask is not None:
            self.output_mask = self.output_mask.to(*args, **kwargs)
        if self.bottleneck_mask is not None:
            self.bottleneck_mask = self.bottleneck_mask.to(*args, **kwargs)
        return result
        
    def __repr__(self):
        s = super().__repr__()
        for attr in ['in_transformation', 'input_mask', 'lora_A', 'bottleneck_mask', 'lora_B', 'output_mask', 'out_transformation']:
            val = getattr(self, attr, None)
            if val is None:
                continue
            if val.ndim == 1:
                s += "\n  %s(%d)" % (attr, val.shape[0])
            elif 'transformation' in attr:
                s += "\n  %s(in_features=%d, out_features=%d)" % (attr, val.shape[0], val.shape[1])
            else:
                s += "\n  %s(in_features=%d, out_features=%d)" % (attr, val.shape[1], val.shape[0])
        s += "\n)"
        return s
        

class DistillLinear(PruningLinear):
    # LoRA implemented in a dense layer, but for distillation usage, the train() wouldn't change the weights merged/unmerged
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        teacher_r: int = 8,
        lora_alpha: int = 1,
        teacher_lora_alpha: int = 16, 
        retained_indices: Optional[List[int]] = None,
        out_retained_indices: Optional[List[int]] = None,
        in_retained_indices: Optional[List[int]] = None,
        teacher_retained_indices: Optional[List[int]] = None,
        teacher_out_retained_indices: Optional[List[int]] = None,
        teacher_in_retained_indices: Optional[List[int]] = None,
        output_dynamic: bool = True,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        PruningLinear.__init__(self, in_features, out_features, r, lora_alpha, lora_dropout, fan_in_fan_out, merge_weights, retained_indices, out_retained_indices, in_retained_indices, output_dynamic, **kwargs)
        self.teacher_r = teacher_r
        self.teacher_lora_alpha = teacher_lora_alpha
        if teacher_retained_indices is not None and (teacher_out_retained_indices is not None or teacher_in_retained_indices is not None):
            raise ValueError("You can only specify teacher_retained_indices or teacher_out_retained_indices and teacher_in_retained_indices")
        if teacher_retained_indices is not None:
            if output_dynamic:
                teacher_out_retained_indices = teacher_retained_indices
                teacher_in_retained_indices = None
            else:
                teacher_in_retained_indices = teacher_retained_indices
                teacher_out_retained_indices = None

        self.teacher_out_retained_indices = teacher_out_retained_indices
        self.teacher_in_retained_indices = teacher_in_retained_indices
        self.teacher_retained_indices = teacher_retained_indices

        if teacher_r > 0:
            if output_dynamic:
                self.teacher_lora_A = nn.Parameter(self.weight.new_zeros((teacher_r, in_features)))
                self.teacher_lora_B = nn.Parameter(self.weight.new_zeros((len(self.teacher_retained_indices) if self.teacher_retained_indices is not None else out_features, teacher_r)))
            else:
                self.teacher_lora_A = nn.Parameter(self.weight.new_zeros((teacher_r, len(self.teacher_retained_indices) if self.teacher_retained_indices is not None else in_features)))
                self.teacher_lora_B = nn.Parameter(self.weight.new_zeros((out_features, teacher_r)))
            self.teacher_scaling = teacher_lora_alpha / teacher_r
            nn.init.kaiming_uniform_(self.teacher_lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.teacher_lora_B)

        if teacher_out_retained_indices is None:
            self.teacher_out_transformation = None
        else:
            self.teacher_out_transformation = nn.Parameter(self.weight.new_zeros(len(self.teacher_out_retained_indices), self.out_features))
            for i, idx in enumerate(self.teacher_out_retained_indices):
                self.teacher_out_transformation.data[i, idx] = 1
            self.teacher_out_transformation.requires_grad = False
        if teacher_in_retained_indices is None:
            self.teacher_in_transformation = None
        else:
            self.teacher_in_transformation = nn.Parameter(self.weight.new_zeros(self.in_features, len(self.teacher_in_retained_indices)))
            for i, idx in enumerate(self.teacher_in_retained_indices):
                self.teacher_in_transformation.data[idx, i] = 1
            self.teacher_in_transformation.requires_grad = False
            
        
    def forward(self, x: torch.Tensor, use_teacher: bool = False, in_retained_indices: Optional[torch.Tensor] = None, out_retained_indices: Optional[torch.Tensor] = None, reconstruct_output: bool = False):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if ((self.r > 0 and not use_teacher) or (self.teacher_r > 0 and use_teacher)) and not self.merged:
            selected_weight, selected_bias = select_wandb(self.weight, self.bias, in_retained_indices, out_retained_indices)
            if self.teacher_r > 0 and use_teacher:
                result = F.linear(x, T(self.weight), bias=self.bias)
                additive_result = (self.lora_dropout(x) @ self.teacher_lora_A.T @ self.teacher_lora_B.T) * self.teacher_scaling
                result += additive_result
            elif self.r > 0 and not use_teacher:
                # lora_A: (r, in_features); lora_B: (out_features, r); weight: (out_features, in_features)
                lora_A_use = self.lora_A * self.input_mask if self.input_mask is not None else self.lora_A # (r, in_features)
                lora_A_use = lora_A_use.T if self.bottleneck_mask is None else lora_A_use.T * self.bottleneck_mask # (in_features, r)
                lora_B_use = self.lora_B.T * self.output_mask if self.output_mask is not None else self.lora_B.T # (r, out_features)
                
                # Handling retained-indices situation
                if in_retained_indices is not None:
                    lora_A_use = lora_A_use.index_select(0, in_retained_indices)
                if out_retained_indices is not None:
                    lora_B_use = lora_B_use.index_select(1, out_retained_indices)
                
                result = F.linear(x, T(selected_weight), bias=selected_bias)
                additive_result = (self.lora_dropout(x) @ lora_A_use @ lora_B_use) * self.scaling
                result += additive_result
                if reconstruct_output:
                    result = _do_reconstruct_outputs(result, out_retained_indices, self.out_features)
            return result
        else:
            return super().forward(x, use_teacher=use_teacher, in_retained_indices=in_retained_indices, out_retained_indices=out_retained_indices, reconstruct_output=reconstruct_output)
        
    def calculate_teacher_pruned_outdim_weights(self, pruned_out_dim: Optional[List[int]] = None, pruned_bottleneck_dim: Optional[List[int]] = None, pruned_in_dim: Optional[List[int]] = None):
        return calculate_pruned_equivalent_weights(self, self.teacher_lora_A, self.teacher_lora_B, self.teacher_r, self.teacher_scaling, self.teacher_in_transformation, self.teacher_out_transformation, pruned_out_dim, pruned_bottleneck_dim, pruned_in_dim)

    # def train(self, mode: bool = True):
    #     def T(w):
    #         return w.T if self.fan_in_fan_out else w
    #     nn.Linear.train(self, mode)
    #     if self.merge_weights and self.merged and mode:
    #         # Make sure that the weights are not merged
    #         if self.r > 0:
    #             if self.out_transformation is None:
    #                 self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
    #             elif self.output_dynamic:
    #                self.weight.data -= T(self.out_transformation.T @ self.lora_B @ self.lora_A) * self.scaling
    #             else:
    #                 self.weight.data -= T(self.lora_B @ self.lora_A @ self.out_transformation) * self.scaling
    #         if self.teacher_r > 0:
    #             if self.teacher_out_transformation is None:
    #                 self.weight.data -= T(self.teacher_lora_B @ self.teacher_lora_A) * self.teacher_scaling
    #             elif self.output_dynamic:
    #                self.weight.data -= T(self.teacher_out_transformation.T @ self.teacher_lora_B @ self.teacher_lora_A) * self.teacher_scaling
    #             else:
    #                 self.weight.data -= T(self.teacher_lora_B @ self.teacher_lora_A @ self.teacher_out_transformation) * self.teacher_scaling
    #         self.merged = False
    
    # def eval(self):
    #     def T(w):
    #         return w.T if self.fan_in_fan_out else w
    #     nn.Linear.eval(self)
    #     if self.merge_weights and not self.merged:
    #         # Merge the weights and mark it
    #         if self.r > 0:
    #             if self.out_transformation is None:
    #                 self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
    #             elif self.output_dynamic:
    #                 self.weight.data += T(self.out_transformation.T @ self.lora_B @ self.lora_A) * self.scaling
    #             else:
    #                 self.weight.data += T(self.lora_B @ self.lora_A @ self.out_transformation) * self.scaling
    #         if self.teacher_r > 0:
    #             if self.teacher_out_transformation is None:
    #                 self.weight.data += T(self.teacher_lora_B @ self.teacher_lora_A) * self.teacher_scaling
    #             elif self.output_dynamic:
    #                 self.weight.data += T(self.teacher_out_transformation.T @ self.teacher_lora_B @ self.teacher_lora_A) * self.teacher_scaling
    #             else:
    #                 self.weight.data += T(self.teacher_lora_B @ self.teacher_lora_A @ self.teacher_out_transformation) * self.teacher_scaling
    #         self.merged = True
    
def calculate_pruned_equivalent_weights(layer: PruningLinear, lora_A: nn.Parameter, lora_B: nn.Parameter, r, scaling, in_transformation: nn.Parameter, out_transformation: nn.Parameter, pruned_out_dim: Optional[List[int]] = None, pruned_bottleneck_dim: Optional[List[int]] = None, pruned_in_dim: Optional[List[int]] = None):
    def T(w):
        return w.T if layer.fan_in_fan_out else w
    if pruned_out_dim is not None:
        pruned_out_transformation = lora_B.new_zeros((len(pruned_out_dim), layer.out_features))
        for i, index in enumerate(pruned_out_dim):
            pruned_out_transformation[i, index] = 1
    else:
        pruned_out_transformation = None
    if pruned_in_dim is not None:
        pruned_in_transformation = lora_A.new_zeros((layer.in_features, len(pruned_in_dim)))
        for i, index in enumerate(pruned_in_dim):
            pruned_in_transformation[index, i] = 1
    else:
        pruned_in_transformation = None

    if pruned_bottleneck_dim:
        pruned_lora_A = lora_A.index_select(0, torch.tensor(pruned_bottleneck_dim).int().to(lora_A.device))
        pruned_lora_B = lora_B.index_select(1, torch.tensor(pruned_bottleneck_dim).int().to(lora_B.device))
    else:
        pruned_lora_A = lora_A
        pruned_lora_B = lora_B


    if r > 0:
        if pruned_out_dim is not None:
            pruned_lora_B = pruned_lora_B.index_select(0, torch.tensor(pruned_out_dim).int().to(pruned_lora_B.device))
        if pruned_in_dim is not None:
            pruned_lora_A = pruned_lora_A.index_select(1, torch.tensor(pruned_in_dim).int().to(pruned_lora_A.device))
        transformed_lora_B = pruned_out_transformation.T @ pruned_lora_B if pruned_out_transformation is not None else out_transformation.T @ pruned_lora_B if out_transformation is not None else pruned_lora_B
        transformed_lora_A = pruned_lora_A @ pruned_in_transformation.T if pruned_in_transformation is not None else pruned_lora_A @ in_transformation.T if in_transformation is not None else pruned_lora_A
        weight = T(transformed_lora_B @ transformed_lora_A) * scaling
        return weight
    else:
        return None  
    
                   
class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), 
                    self.lora_B.data.unsqueeze(-1), 
                    groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), 
                    self.lora_B.data.unsqueeze(-1), 
                    groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data += self.zero_pad(T(delta_w * self.scaling))
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(
                    after_A.transpose(-2, -1), 
                    self.lora_B.unsqueeze(-1), 
                    groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result
            

class Conv2d(nn.Conv2d, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert type(kernel_size) is int
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = False
    
    def eval(self):
        nn.Conv2d.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.conv2d(
                x, 
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        return nn.Conv2d.forward(self, x)


if __name__ == '__main__':
    # Test lora-weight decouple
    masked = torch.randint(0, 32, [10]).sort()[0]
    masked = list(set(masked.tolist()))
    non_masked = list(set(range(32)) - set(masked))
    x = torch.randn(2,8,32)
    a = torch.randn(32,8)
    b = torch.randn(8,32)
    b[:, masked] = 0
    original_output = x @ a @ b
    trans = torch.zeros(32 - len(masked), 32)
    smaller_b = b[:,b.any(dim=0)].clone().contiguous()
    for i, m in enumerate(non_masked):
        trans[i, m] = 1
    smaller_output = x @ a @ (smaller_b @ trans)
    print("Checking: ", torch.allclose(original_output, smaller_output))
    
    x = torch.randn(2,8,64)
    lora = Linear(64, 32, r=8, lora_alpha=1)
    lora.lora_B.data = torch.randn(32, 8)
    lora.lora_B.data[masked, :] = 0
    small_lora = PruningLinear(64, 32, r=8, lora_alpha=1, retained_indices=non_masked)
    small_lora.lora_A.data = lora.lora_A.data
    small_lora.lora_B.data = lora.lora_B.data[lora.lora_B.data.any(dim=1),:]
    small_lora.weight.data = lora.weight.data
    small_lora.bias.data = lora.bias.data
    original_outputs = lora(x)
    small_outputs = small_lora(x)
    print("Checking: ", torch.allclose(original_outputs, small_outputs))
    small_lora.eval()
    small_outputs = small_lora(x)
    print("Checking: ", torch.allclose(original_outputs, small_outputs, 0.00001, 1e-7))
    small_lora.train()
    small_outputs = small_lora(x)
    print("Checking: ", torch.allclose(original_outputs, small_outputs))
    
    distill_linear = DistillLinear(64, 32, r=64, teacher_r=8, lora_alpha=16, teacher_lora_alpha=16, retained_indices=non_masked, teacher_retained_indices=masked)
    student_output = distill_linear(x)
    teacher_output = distill_linear(x, use_teacher=True)