# This file contains code derived from [CoFiPruning](https://github.com/princeton-nlp/CoFiPruning),
# originally developed by Princeton-NLP and released under the MIT-License.
# Modifications were made to adapt to the current project's context.

# allows removing layers of heads and mlps
import transformers

__version__ = transformers.__version__

from typing import Optional


from transformers.models.roberta.modeling_roberta import (RobertaForSequenceClassification,
    RobertaForQuestionAnswering,
    RobertaForMaskedLM,
    RobertaModel,
    RobertaEncoder,
    RobertaLayer,
    RobertaAttention,
    RobertaSelfAttention,
    RobertaSelfOutput,
    RobertaOutput,
    MaskedLMOutput,
    RobertaIntermediate,
    RobertaClassificationHead,
    create_position_ids_from_input_ids,
    RobertaEmbeddings)


import torch
import torch.nn.functional as F
import loralib as lora
from loralib import Linear as LoRALinear
from loralib.layers import _do_reconstruct_outputs, select_wandb
from torch import nn
from transformers.modeling_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices
from torch.nn import CrossEntropyLoss, MSELoss
import math
import logging
from typing import Dict, List, Tuple
import numpy as np
from models.modeling_bert import CoFiLayerNorm 
from models.modeling_outputs import NewBaseModelOutputWithPooling, NewBaseModelOutput, NewSequenceClassifierOutput, NewQuestionAnsweringModelOutput
from utils.minus_utils import prune_layer_norm, prune_layer, detect_no_zero, _mask_fine_to_coarse

logger = logging.getLogger(__name__)

BertLayerNorm = CoFiLayerNorm 

def clear_masks(model):
    model.head_mask = None
    model.intermediate_mask = None
    model.hidden_mask = None
    
def reset_masks(model):
    attn_head_size = model.config.hidden_size // model.config.num_attention_heads
    model.head_mask = [
        torch.ones(model.roberta.encoder.layer[i].attention.self.query.weight.shape[0] // attn_head_size, device=model.device)
        if model.roberta.encoder.layer[i].attention.self.query is not None else None
        for i in range(model.config.num_hidden_layers)
    ]
    model.intermediate_mask = [
        torch.ones(model.roberta.encoder.layer[i].intermediate.dense.weight.shape[0], device=model.device)
        if model.roberta.encoder.layer[i].intermediate.dense is not None else None
        for i in range(model.config.num_hidden_layers)
    ]
    model.hidden_mask = torch.ones(model.config.hidden_size, device=model.device)
    if all(model.head_mask[i].shape == model.head_mask[0].shape for i in range(len(model.head_mask))):
        model.head_mask = torch.stack(model.head_mask, dim=0)
    if all(model.intermediate_mask[i].shape == model.intermediate_mask[0].shape for i in range(len(model.intermediate_mask))):
        model.intermediate_mask = torch.stack(model.intermediate_mask, dim=0)
    
def mask_fine_to_coarse(model, mask):
    if isinstance(mask, torch.Tensor):
        return mask.detach().any(dim=1).float()
    elif isinstance(mask, list):
        return [v.detach().any().float() if v is not None else None for v in mask]

def resize_intermediate(model, kept_intermediate_dims: Dict[int, List[int]]):
    roberta = model.roberta
    device = model.device
    for layer in kept_intermediate_dims:
        if len(kept_intermediate_dims[layer]) == 0:
            roberta.encoder.layer[layer].intermediate.dense = None
            roberta.encoder.layer[layer].output.dense = None
        else:
            if isinstance(roberta.encoder.layer[layer].intermediate.dense, lora.Linear) and roberta.encoder.layer[layer].intermediate.dense.r > 0:
                roberta.encoder.layer[layer].intermediate.dense = prune_layer(roberta.encoder.layer[layer].intermediate.dense, index=torch.LongTensor(kept_intermediate_dims[layer]).to(device), dim=0)
            else:
                roberta.encoder.layer[layer].intermediate.dense = prune_layer(roberta.encoder.layer[layer].intermediate.dense, index=torch.LongTensor(kept_intermediate_dims[layer]).to(device), dim=0)
            roberta.encoder.layer[layer].output.dense = prune_layer(roberta.encoder.layer[layer].output.dense, index=torch.LongTensor(kept_intermediate_dims[layer]).to(device), dim=1)
            
def named_masks(model, head_mask=None, intermediate_mask=None):
    hidden_per_head = model.config.hidden_size // model.config.num_attention_heads
    head_mask = model.head_mask if head_mask is None else head_mask
    intermediate_mask = model.intermediate_mask if intermediate_mask is None else intermediate_mask
    mask_dict = {}
    for i in range(model.config.num_hidden_layers):
        query_name = "roberta.encoder.layer.%d.attention.self.query" % i
        key_name = "roberta.encoder.layer.%d.attention.self.key" % i
        value_name = "roberta.encoder.layer.%d.attention.self.value" % i
        attn_output_name = "roberta.encoder.layer.%d.attention.output.dense" % i
        up_name = "roberta.encoder.layer.%d.intermediate.dense" % i
        down_name = "roberta.encoder.layer.%d.output.dense" % i
        for name in [query_name, key_name, value_name, attn_output_name]:
            mask_dict[name] = torch.repeat_interleave(head_mask[i], hidden_per_head) if head_mask is not None else None
        for name in [up_name, down_name]:
            mask_dict[name] = intermediate_mask[i] if intermediate_mask is not None else None
    return mask_dict

def print_model_shape(model):
    roberta = model.roberta
    for layer in range(model.config.num_hidden_layers):
        print("Layer:", layer)
        if roberta.encoder.layer[layer].attention.self.query is not None:
            print("query:", roberta.encoder.layer[layer].attention.self.query.weight.shape)
            print("key:", roberta.encoder.layer[layer].attention.self.key.weight.shape)
        else:
            print("query:", None)
            print("key:", None)
        if roberta.encoder.layer[layer].attention.self.value is not None:
            print("value:", roberta.encoder.layer[layer].attention.self.value.weight.shape)
            print("output:", roberta.encoder.layer[layer].attention.output.dense.weight.shape)
        else:
            print("value:", None)
            print("output:", None)
        if roberta.encoder.layer[layer].intermediate.dense is not None:
            print("up:", roberta.encoder.layer[layer].intermediate.dense.weight.shape)
            print("down:", roberta.encoder.layer[layer].output.dense.weight.shape)
        else:
            print("up", None)
            print("down", None)
            
def print_lora_info_by_layer(model):
    def print_lora_info(l, layername):
        if isinstance(l, LoRALinear) and hasattr(l, 'lora_A') and hasattr(l, 'lora_B') and l.lora_A is not None and l.lora_B is not None:
            print("%s: r: " % layername, l.r if hasattr(l, 'r') else 0, ', input dim: ', l.lora_A.shape[1] if hasattr(l, 'lora_A') and l.lora_A is not None else 0, ', output dim: ', l.lora_B.shape[0] if hasattr(l, 'lora_B') and l.lora_B is not None else 0)
        elif isinstance(l, LoRALinear):
            print("%s: frozen LoRA layer" % layername)
        else:
            print("%s: frozen Linear layer" % layername)
    for i in range(model.config.num_hidden_layers):
        print("Layer:", i)
        layer: NewRobertaLayer = model.roberta.encoder.layer[i]
        query, key, value, output = layer.attention.self.query, layer.attention.self.key, layer.attention.self.value, layer.attention.output.dense
        up, down = layer.intermediate.dense, layer.output.dense
        print_lora_info(query, "query")
        print_lora_info(key, "key")
        print_lora_info(value, "value")
        print_lora_info(output, "output")
        print_lora_info(up, "up")
        print_lora_info(down, "down")
        
def split_mask_or_score(model, head_mask: torch.Tensor, intermediate_mask: torch.Tensor) -> Tuple[List[torch.Tensor]]:
    backbone: NewRobertaModel = model.roberta if hasattr(model, 'roberta') else model
    if isinstance(head_mask, torch.Tensor) and head_mask.ndim == 1:
        with torch.no_grad():
            if getattr(model, 'virtual_pruned', False):
                head_mask = torch.split(head_mask, [backbone.encoder.layer[i].attention.self.num_teacher_attention_heads for i in range(backbone.config.num_hidden_layers)])
            else:
                head_mask = torch.split(head_mask, [backbone.encoder.layer[i].attention.self.num_attention_heads for i in range(backbone.config.num_hidden_layers)])
    if isinstance(intermediate_mask, torch.Tensor) and intermediate_mask.ndim == 1:
        with torch.no_grad():
            intermediate_mask = torch.split(intermediate_mask, [0 if backbone.encoder.layer[i].intermediate.dense is None else backbone.encoder.layer[i].intermediate.dense.out_features for i in range(backbone.config.num_hidden_layers)])
    return head_mask, intermediate_mask

def prune_model_with_masks(model, continual_pruning=True):
    head_mask, intermediate_mask, hidden_mask = model.head_mask, model.intermediate_mask, model.hidden_mask
    if detect_no_zero(head_mask) and detect_no_zero(intermediate_mask) and detect_no_zero(hidden_mask):
        print("No pruning is performed. Skipping pruning.")
        return
    pruned_history = {}
    # pruned_history['params'] = dict(model.named_parameters())
    # pruned_history['modules'] = dict(model.named_modules())
    head_mask, intermediate_mask = split_mask_or_score(model, head_mask, intermediate_mask)
    pruned_history['head_mask'] = head_mask
    pruned_history['intermediate_mask'] = intermediate_mask
    pruned_history['hidden_mask'] = model.hidden_mask

    if head_mask is not None:
        encoder_pruned_heads = {}
        pruned_heads = model.config.pruned_heads if continual_pruning else {}
        # Encoder-only architecture
        for layer in range(head_mask.shape[0] if isinstance(head_mask, torch.Tensor) else len(head_mask)):
            if head_mask[layer] is None:
                continue
            head_to_prune = head_mask[layer] == 0
            if head_to_prune.all():
                model.head_layer_z[layer] = 0
            now_pruning_heads = torch.where(head_to_prune)[0].tolist()
            # Shift the now_pruning_heads based on the pruned heads (re-index them as if all heads are kept)
            if layer in pruned_heads and pruned_heads[layer]:
                retained_head_indices = [x for x in range(model.config.num_attention_heads) if x not in pruned_heads[layer]]
                encoder_pruned_heads[layer] = [retained_head_indices[x] for x in now_pruning_heads]
            else:
                encoder_pruned_heads[layer] = now_pruning_heads
            print(f"Encoder layer {layer} prune heads: {encoder_pruned_heads[layer]}.")
        model.prune_heads(encoder_pruned_heads)
        
    # Pruning hidden dimensions
    hidden_size = model.config.hidden_size
    # If init from pruend config, hidden_size should be smaller than hidden mask's length
    if isinstance(hidden_mask, torch.Tensor) and hidden_mask.numel() != hidden_size:
        print("hidden mask's length is not equal to hidden_size, hidden mask's length is", hidden_mask.numel(), "hidden_size is", hidden_size)
        print("Skipping hidden dimension pruning")
    elif hidden_mask is not None and (hidden_mask == 0).any():
        index = torch.LongTensor(hidden_mask.squeeze().nonzero().squeeze().tolist())
        index = index.to(model.device)

        model.roberta.embeddings.word_embeddings.weight = torch.nn.parameter.Parameter(
            model.roberta.embeddings.word_embeddings.weight.index_select(1, index).detach().clone())
        model.roberta.embeddings.word_embeddings.embedding_dim = index.shape[0]
        model.roberta.embeddings.position_embeddings.weight = torch.nn.parameter.Parameter(
            model.roberta.embeddings.position_embeddings.weight.index_select(1, index).detach().clone())
        model.roberta.embeddings.position_embeddings.embedding_dim = index.shape[0]
        model.roberta.embeddings.token_type_embeddings.weight = torch.nn.parameter.Parameter(
            model.roberta.embeddings.token_type_embeddings.weight.index_select(1, index).detach().clone())
        model.roberta.embeddings.token_type_embeddings.embedding_dim = index.shape[0]
        prune_layer_norm(model.roberta.embeddings.LayerNorm, index)

        for layer in range(0, 12):
            print("Pruning layer:", layer)
            if model.roberta.encoder.layer[layer].attention.self.query is not None:
                model.roberta.encoder.layer[layer].attention.self.query = \
                    prune_layer(model.roberta.encoder.layer[layer].attention.self.query , index, dim=1)
                model.roberta.encoder.layer[layer].attention.self.key = \
                    prune_layer(model.roberta.encoder.layer[layer].attention.self.key , index, dim=1)
            if model.roberta.encoder.layer[layer].attention.self.value is not None:
                model.roberta.encoder.layer[layer].attention.self.value = \
                    prune_layer(model.roberta.encoder.layer[layer].attention.self.value , index, dim=1)
                model.roberta.encoder.layer[layer].attention.output.dense = \
                    prune_layer(model.roberta.encoder.layer[layer].attention.output.dense , index, dim=0)
                prune_layer_norm(model.roberta.encoder.layer[layer].attention.output.LayerNorm, index)
            if model.roberta.encoder.layer[layer].intermediate.dense is not None:
                model.roberta.encoder.layer[layer].intermediate.dense = \
                    prune_layer( model.roberta.encoder.layer[layer].intermediate.dense, index, dim=1)
                model.roberta.encoder.layer[layer].output.dense = \
                    prune_layer( model.roberta.encoder.layer[layer].output.dense, index, dim=0)
                prune_layer_norm(model.roberta.encoder.layer[layer].output.LayerNorm, index)

        # accommodate for different models
        if hasattr(model, "classifier"):
            if hasattr(model.classifier, "dense"):
                model.classifier.dense = prune_layer(model.classifier.dense, index, dim=1)
        if hasattr(model, "cls"):
            if hasattr(model.cls, "dense"):
                model.cls.dense = prune_layer(model.classifier.dense, index, dim=1)
        if hasattr(model.roberta.pooler, "dense"):
            model.roberta.pooler.dense = prune_layer(model.roberta.pooler.dense, index, dim=1)
        if hasattr(model, "qa_outputs"):
            model.qa_outputs = prune_layer(model.qa_outputs, index, dim=1)
        if getattr(model, "layer_transformation", None) is not None:
            model.layer_transformation = prune_layer(model.layer_transformation, index, dim=1)
            print("layer transformation", model.layer_transformation.weight.shape)
        if getattr(model, "mha_layer_transformation", None) is not None:
            model.mha_layer_transformation = prune_layer(model.mha_layer_transformation, index, dim=1)
            print("layer mha_layer_transformation", model.mha_layer_transformation.weight.shape)
        # Reduce model's hidden size
        # model.config.hidden_size = index.shape[0]
        
    encoder_kept_intermediate_dims = {}
    if intermediate_mask is not None:
        for layer in range(intermediate_mask.shape[0] if isinstance(intermediate_mask, torch.Tensor) else len(intermediate_mask)):
            if intermediate_mask[layer] is None:
                continue
            intermediate_to_retain = intermediate_mask[layer] != 0
            if not intermediate_to_retain.any():
                model.mlp_z[layer] = 0
            encoder_kept_intermediate_dims[layer] = torch.where(intermediate_to_retain)[0].tolist()
        model.resize_intermediate(encoder_kept_intermediate_dims)
    model.print_model_shape()
    model.head_mask = torch.cat([v[v.nonzero().squeeze()].detach().contiguous().flatten() for v in head_mask])
    model.intermediate_mask = torch.cat([v[v.nonzero().squeeze()].detach().contiguous().flatten() for v in intermediate_mask])
    model.hidden_mask = hidden_mask[hidden_mask.nonzero().squeeze()].detach().contiguous().flatten() if hidden_mask is not None else None
    model.pruned_history = pruned_history

def virtual_prune(model):
    if model.virtual_pruned:
        print("Model is already virtual pruned. Skipping virtual pruning.", flush=True)
        return
    print("Virtual pruning model", flush=True)
    head_mask, intermediate_mask = model.split_mask_or_score()
    hidden_mask = model.hidden_mask
    hidden_retained_indices = hidden_mask.nonzero().squeeze()
    num_dim_per_head = model.config.hidden_size // model.config.num_attention_heads
    model.roberta.embeddings.retained_indices = hidden_retained_indices
    model.roberta.embeddings.LayerNorm.retained_indices = hidden_retained_indices
    for layer in range(model.config.num_hidden_layers):
        mask = head_mask[layer]
        num_retained_heads = mask.sum().int().item()
        block_retained_indices = torch.repeat_interleave(mask, num_dim_per_head).nonzero().squeeze()
        model.roberta.encoder.layer[layer].attention.self.num_teacher_attention_heads = model.roberta.encoder.layer[layer].attention.self.num_attention_heads
        model.roberta.encoder.layer[layer].attention.self.num_attention_heads = num_retained_heads 
        model.roberta.encoder.layer[layer].attention.self.block_retained_indices = block_retained_indices
        model.roberta.encoder.layer[layer].attention.self.hidden_retained_indices = hidden_retained_indices
        model.roberta.encoder.layer[layer].attention.output.block_retained_indices = block_retained_indices
        model.roberta.encoder.layer[layer].attention.output.hidden_retained_indices = hidden_retained_indices
        model.roberta.encoder.layer[layer].attention.output.LayerNorm.retained_indices = hidden_retained_indices
        
        block_retained_indices = intermediate_mask[layer].nonzero().squeeze()
        model.roberta.encoder.layer[layer].intermediate.block_retained_indices = block_retained_indices
        model.roberta.encoder.layer[layer].intermediate.hidden_retained_indices = hidden_retained_indices
        model.roberta.encoder.layer[layer].output.block_retained_indices = block_retained_indices
        model.roberta.encoder.layer[layer].output.hidden_retained_indices = hidden_retained_indices
        model.roberta.encoder.layer[layer].output.LayerNorm.retained_indices = hidden_retained_indices
    
    if hasattr(model, "classifier"):
        model.classifier.retained_indices = hidden_retained_indices
    model.retained_indices = hidden_retained_indices
    model.backup_head_mask = model.head_mask
    model.backup_intermediate_mask = model.intermediate_mask
    model.backup_hidden_mask = model.hidden_mask
    model.head_mask = None
    model.intermediate_mask = None
    model.hidden_mask = None
    model.virtual_pruned = True
    
    
def virtual_prune_restore(model):
    if not model.virtual_pruned:
        print("Model is not virtual pruned. Skipping virtual pruning restoration.", flush=True)
        return
    print("Restoring model from virtual pruning", flush=True)
    model.head_mask, model.intermediate_mask, model.hidden_mask = model.backup_head_mask, model.backup_intermediate_mask, model.backup_hidden_mask
    model.backup_head_mask, model.backup_intermediate_mask, model.backup_hidden_mask = None, None, None
    for layer in range(model.config.num_hidden_layers):
        model.roberta.encoder.layer[layer].attention.self.block_retained_indices = None
        model.roberta.encoder.layer[layer].attention.self.hidden_retained_indices = None
        model.roberta.encoder.layer[layer].attention.self.num_attention_heads = model.roberta.encoder.layer[layer].attention.self.num_teacher_attention_heads
        model.roberta.encoder.layer[layer].attention.output.block_retained_indices = None
        model.roberta.encoder.layer[layer].attention.output.hidden_retained_indices = None
        model.roberta.encoder.layer[layer].attention.output.LayerNorm.retained_indices = None
        model.roberta.encoder.layer[layer].intermediate.block_retained_indices = None
        model.roberta.encoder.layer[layer].intermediate.hidden_retained_indices = None
        model.roberta.encoder.layer[layer].output.block_retained_indices = None
        model.roberta.encoder.layer[layer].output.hidden_retained_indices = None
        model.roberta.encoder.layer[layer].output.LayerNorm.retained_indices = None
        
    model.roberta.embeddings.retained_indices = None
    model.roberta.embeddings.LayerNorm.retained_indices = None
    if hasattr(model, "classifier"):
        model.classifier.retained_indices = None
    model.retained_indices = None
    model.virtual_pruned = False

def restore_model_from_history(model):
    dim_per_head = model.config.hidden_size // model.config.num_attention_heads
    current_params = dict(model.named_parameters())
    modules = dict(model.named_modules())
    history_params = model.pruned_history['params']
    history_modules = model.pruned_history['modules']
    q_template = 'roberta.encoder.layer.%d.attention.self.query'
    k_template = 'roberta.encoder.layer.%d.attention.self.key'
    v_template = 'roberta.encoder.layer.%d.attention.self.value'
    o_template = 'roberta.encoder.layer.%d.attention.output.dense'
    up_template = 'roberta.encoder.layer.%d.intermediate.dense'
    down_template = 'roberta.encoder.layer.%d.output.dense'
    possible_param_names = ['weight', 'bias', 'lora_A', 'lora_B']
    
    for layer in range(model.config.num_hidden_layers):
        head_mask = model.pruned_history['head_mask'][layer].repeat_interleave(dim_per_head) if model.pruned_history['head_mask'][layer] is not None else None
        intermediate_mask = model.pruned_history['intermediate_mask'][layer]
        q, k, v, o, up, down = q_template % layer, k_template % layer, v_template % layer, o_template % layer, up_template % layer, down_template % layer
        
        for module_name in [q, k, v, o, up, down]:
            if module_name not in modules:
                assert module_name in history_modules
                parent_layer_name, module_subname = module_name.rsplit('.', 1)
                setattr(modules[parent_layer_name], module_subname, history_modules[module_name])
                continue
            for param_name in possible_param_names:
                name = "%s.%s" % (module_name, param_name)
                if name in history_params:
                    old_param, current_param = history_params[name], current_params[name]
                    if old_param.shape != current_param.shape:
                        mask_use = head_mask if 'attention' in module_name else intermediate_mask
                        combined_param = old_param.data.clone()
                        mask_use = mask_use.bool()
                        if param_name == 'bias':
                            # For example, the original shape is (768,) and the current shape is (640), copy the current param to the the old param based on the mask
                            combined_param[mask_use] = current_param
                        else:
                            assert old_param.ndim == 2 and current_param.ndim == 2
                            # For example, the original shape is (768, 768) and the current shape is (640, 768), copy the current param to the the old param based on the mask
                            if old_param.shape[1] == current_param.shape[1]:
                                # The pruning happened on the 0 dimension
                                if param_name == 'lora_A' and hasattr(modules[module_name], 'in_transformation') and modules[module_name].in_transformation is not None:
                                    combined_param[mask_use] = current_param @ modules[module_name].in_transformation.T
                                elif param_name == 'lora_B' and hasattr(modules[module_name], 'out_transformation') and modules[module_name].out_transformation is not None:
                                    combined_param[mask_use] = modules[module_name].out_transformation.T @ current_param
                                else:
                                    combined_param[mask_use] = current_param
                            else:
                                # The pruning happened on the 1 dimension
                                if param_name == 'lora_A' and hasattr(modules[module_name], 'in_transformation') and modules[module_name].in_transformation is not None:
                                    combined_param[:, mask_use] = current_param @ modules[module_name].in_transformation.T
                                elif param_name == 'lora_B' and hasattr(modules[module_name], 'out_transformation') and modules[module_name].out_transformation is not None:
                                    combined_param[:, mask_use] = modules[module_name].out_transformation.T @ current_param
                                else:
                                    combined_param[:, mask_use] = current_param
                        combined_param = nn.Parameter(combined_param)
                        setattr(modules[module_name], param_name, combined_param)
        # Reset num hidden layers and in- out- features
        model.roberta.encoder.layer[layer].attention.self.num_attention_heads = model.config.num_attention_heads
        model.roberta.encoder.layer[layer].attention.self.num_teacher_attention_heads = model.config.num_attention_heads
        model.roberta.encoder.layer[layer].attention.pruned_heads = set()
        model.roberta.encoder.layer[layer].attention.self.query.in_features = model.config.hidden_size
        model.roberta.encoder.layer[layer].attention.self.query.out_features = model.config.hidden_size
        model.roberta.encoder.layer[layer].attention.self.key.in_features = model.config.hidden_size
        model.roberta.encoder.layer[layer].attention.self.key.out_features = model.config.hidden_size
        model.roberta.encoder.layer[layer].attention.self.value.in_features = model.config.hidden_size
        model.roberta.encoder.layer[layer].attention.self.value.out_features = model.config.hidden_size
        model.roberta.encoder.layer[layer].attention.output.dense.in_features = model.config.hidden_size
        model.roberta.encoder.layer[layer].attention.output.dense.out_features = model.config.hidden_size
        model.roberta.encoder.layer[layer].intermediate.dense.in_features = model.config.hidden_size
        model.roberta.encoder.layer[layer].intermediate.dense.out_features = model.config.intermediate_size
        model.roberta.encoder.layer[layer].output.dense.in_features = model.config.intermediate_size
        model.roberta.encoder.layer[layer].output.dense.out_features = model.config.hidden_size
    model.pruned_history = None
    model.config.pruned_heads = {}


class CoFiRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = NewRobertaModel(config, add_pooling_layer=False)
        self.classifier = NewRobertaClassificationHead(config)
        self.head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads).view(-1)
        self.intermediate_mask = torch.ones([config.num_hidden_layers, config.intermediate_size]).view(-1)
        self.hidden_mask = torch.ones(config.hidden_size)
        self.backup_head_mask, self.backup_intermediate_mask, self.backup_hidden_mask = None, None, None
        self.virtual_pruned = False
        self.retained_indices = None
        self.is_teacher = False
        self.is_student = True
        self.is_distilling=False
        self.is_colearning = False
        self.pruned_history = None
        if config.do_distill:
            if config.apply_lora:
                self.layer_transformation = lora.PruningLinear(config.hidden_size, config.hidden_size, r=8, lora_alpha=16, bias=False)
            else:
                self.layer_transformation = nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=self.dtype)
            # self.layer_transformation.weight.data = torch.eye(config.hidden_size)
        else:
            self.layer_transformation = None
        self.head_layer_z = torch.ones(config.num_hidden_layers)
        self.mlp_z = torch.ones(config.num_hidden_layers)
            
    def clear_masks(self):
        clear_masks(self)
        
    def reset_masks(self):
        reset_masks(self)
        
    
    def _mask_fine_to_coarse(self, mask):
        return mask_fine_to_coarse(self, mask)

    def resize_intermediate(self, kept_intermediate_dims: Dict[int, List[int]]):
        return resize_intermediate(self, kept_intermediate_dims)
                
    def named_masks(self, head_mask=None, intermediate_mask=None):
        return named_masks(self, head_mask, intermediate_mask)

    def print_model_shape(self):
        print_model_shape(self)
                
    def print_lora_info_by_layer(self):
        print_lora_info_by_layer(self)
            
    def split_mask_or_score(self, head_mask = None, intermediate_mask = None) -> Tuple[List[torch.Tensor]]:
        head_mask = self.head_mask if head_mask is None else head_mask
        intermediate_mask = self.intermediate_mask if intermediate_mask is None else intermediate_mask
        return split_mask_or_score(self, head_mask, intermediate_mask)
    
    def update_layer_z(self):
        head_mask, intermediate_mask = self.split_mask_or_score() if not self.virtual_pruned else self.split_mask_or_score(self.backup_head_mask, self.backup_intermediate_mask)
        self.head_layer_z = torch.cat([v.any().float().unsqueeze(0) for v in head_mask]).to(self.device)
        self.mlp_z = torch.cat([v.any().float().unsqueeze(0) for v in intermediate_mask]).to(self.device)
    
    def prune_model_with_masks(self, continual_pruning=True):
        prune_model_with_masks(self, continual_pruning)
    
    def restore_model_from_history(self):
        restore_model_from_history(self)
    
    def virtual_prune(self):
        virtual_prune(self)
        
        
    def virtual_prune_restore(self):
        virtual_prune_restore(self)
                    

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            output_masked_states=False,
            use_unmasked_states=True,
            use_cross_masked_states=False,
            use_teacher=False,
            return_dict=None,
            head_z=None,
            intermediate_z=None,
            head_layer_z=None,
            mlp_z=None,
            inference=False,
            hidden_z=None,
            pass_mask=True,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        head_z = head_z if head_z is not None else self.head_mask if pass_mask else None
        intermediate_z = intermediate_z if intermediate_z is not None else self.intermediate_mask if pass_mask else None
        hidden_z = hidden_z if hidden_z is not None else self.hidden_mask if pass_mask else None
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_masked_states=output_masked_states,
            use_unmasked_states=use_unmasked_states,
            use_cross_masked_states=use_cross_masked_states,
            use_teacher=use_teacher,
            return_dict=return_dict,
            intermediate_z=intermediate_z,
            head_z=head_z,
            mlp_z=mlp_z,
            head_layer_z=head_layer_z,
            inference=inference,
            hidden_z=hidden_z
        )

        sequence_output = outputs[0]
        if use_cross_masked_states:
            masked_sequence_output = outputs[-1][-1]

        logits = self.classifier(sequence_output, use_teacher=use_teacher or use_cross_masked_states)
        masked_logits = self.classifier(masked_sequence_output, use_teacher=False) if use_cross_masked_states else None

        loss = None
        masked_loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
                if use_cross_masked_states:
                    masked_loss = loss_fct(masked_logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                if use_cross_masked_states:
                    masked_loss = loss_fct(masked_logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            if use_cross_masked_states:
                output = (masked_loss, logits, masked_logits) + outputs[2:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return NewSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            masked_loss=masked_loss,
            masked_logits=masked_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            masked_states=outputs.masked_states,
        )

class NewRobertaForQuestionAnswering(RobertaForQuestionAnswering):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = NewRobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()
        self.head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads).view(-1)
        self.intermediate_mask = torch.ones([config.num_hidden_layers, config.intermediate_size]).view(-1)
        self.hidden_mask = torch.ones(config.hidden_size)
        self.backup_head_mask, self.backup_intermediate_mask, self.backup_hidden_mask = None, None, None
        self.virtual_pruned = False
        self.retained_indices = None
        self.is_teacher = False
        self.is_student = True
        self.is_distilling=False
        self.is_colearning = False
        if hasattr(config, 'do_distill') and config.do_distill:
            if config.apply_lora:
                self.layer_transformation = lora.Linear(config.hidden_size, config.hidden_size, r=8, lora_alpha=16, bias=False)
            else:
                self.layer_transformation = nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=self.dtype)
            # self.layer_transformation.weight.data = torch.eye(config.hidden_size)
        else:
            self.layer_transformation = None
        self.head_layer_z = torch.ones(config.num_hidden_layers)
        self.mlp_z = torch.ones(config.num_hidden_layers)
            
    def clear_masks(self):
        clear_masks(self)
        
    def reset_masks(self):
        reset_masks(self)
        
    
    def _mask_fine_to_coarse(self, mask):
        return mask_fine_to_coarse(self, mask)

    def resize_intermediate(self, kept_intermediate_dims: Dict[int, List[int]]):
        return resize_intermediate(self, kept_intermediate_dims)
                
    def named_masks(self, head_mask=None, intermediate_mask=None):
        return named_masks(self, head_mask, intermediate_mask)

    def print_model_shape(self):
        print_model_shape(self)
                
    def print_lora_info_by_layer(self):
        print_lora_info_by_layer(self)
            
    def split_mask_or_score(self, head_mask = None, intermediate_mask = None) -> Tuple[List[torch.Tensor]]:
        head_mask = self.head_mask if head_mask is None else head_mask
        intermediate_mask = self.intermediate_mask if intermediate_mask is None else intermediate_mask
        return split_mask_or_score(self, head_mask, intermediate_mask)
    
    def update_layer_z(self):
        head_mask, intermediate_mask = self.split_mask_or_score() if not self.virtual_pruned else self.split_mask_or_score(self.backup_head_mask, self.backup_intermediate_mask)
        self.head_layer_z = torch.cat([v.any().float().unsqueeze(0) for v in head_mask]).to(self.device)
        self.mlp_z = torch.cat([v.any().float().unsqueeze(0) for v in intermediate_mask]).to(self.device)
    
    def prune_model_with_masks(self, continual_pruning=True):
        prune_model_with_masks(self, continual_pruning)
        
    def virtual_prune(self):
        virtual_prune(self)
        
    def virtual_prune_restore(self):
        virtual_prune_restore(self)
        
    def restore_model_from_history(self):
        restore_model_from_history(self)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions: Optional[torch.LongTensor] = None,
            end_positions: Optional[torch.LongTensor] = None,
            output_attentions=None,
            output_hidden_states=None,
            output_masked_states=False,
            use_unmasked_states=True,
            use_cross_masked_states=False,
            use_teacher=False,
            return_dict=None,
            head_z=None,
            intermediate_z=None,
            head_layer_z=None,
            mlp_z=None,
            inference=False,
            hidden_z=None,
            pass_mask=True,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        head_z = head_z if head_z is not None else self.head_mask if pass_mask else None
        intermediate_z = intermediate_z if intermediate_z is not None else self.intermediate_mask if pass_mask else None
        hidden_z = hidden_z if hidden_z is not None else self.hidden_mask if pass_mask else None
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_masked_states=output_masked_states,
            use_unmasked_states=use_unmasked_states,
            use_cross_masked_states=use_cross_masked_states,
            use_teacher=use_teacher,
            return_dict=return_dict,
            intermediate_z=intermediate_z,
            head_z=head_z,
            mlp_z=mlp_z,
            head_layer_z=head_layer_z,
            inference=inference,
            hidden_z=hidden_z
        )

        sequence_output = outputs[0]
        if use_cross_masked_states:
            masked_sequence_output = outputs[-1][-1]

        if isinstance(self.qa_outputs, lora.PruningLinear):
            logits = self.qa_outputs(sequence_output, use_teacher=use_teacher or use_cross_masked_states, in_retained_indices=self.retained_indices)
            masked_logits = self.qa_outputs(masked_sequence_output, use_teacher=False) if use_cross_masked_states else None
        elif use_teacher:
            logits = self.qa_outputs(sequence_output)
        else:
            selected_weight, selected_bias = select_wandb(self.qa_outputs.weight, self.qa_outputs.bias, self.retained_indices)
            logits = F.linear(sequence_output, selected_weight, selected_bias)
            masked_logits = self.qa_outputs(masked_sequence_output) if use_cross_masked_states else None

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        if use_cross_masked_states:
            masked_start_logits, masked_end_logits = masked_logits.split(1, dim=-1)
            masked_start_logits = masked_start_logits.squeeze(-1).contiguous()
            masked_end_logits = masked_end_logits.squeeze(-1).contiguous()
        else:
            masked_start_logits, masked_end_logits = None, None

        total_loss = None
        masked_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            if use_cross_masked_states:
                masked_start_loss = loss_fct(masked_start_logits, start_positions)
                masked_end_loss = loss_fct(masked_end_logits, end_positions)
                masked_loss = (masked_start_loss + masked_end_loss) / 2
            else:
                masked_loss = None

        if not return_dict:
            if use_cross_masked_states:
                output = (masked_loss, start_logits, masked_start_logits, end_logits, masked_end_logits) + outputs[2:]
            else:
                output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return NewQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            masked_loss=masked_loss,
            masked_start_logits=masked_start_logits,
            masked_end_logits=masked_end_logits,
            masked_states=outputs.masked_states,
        )


class NewRobertaClassificationHead(RobertaClassificationHead):
    def __init__(self, config):
        super().__init__(config)
        self.retained_indices = None

    def forward(self, features, use_teacher=False, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        if isinstance(self.dense, lora.PruningLinear):
            x = self.dense(x, use_teacher=use_teacher, in_retained_indices=self.retained_indices)
        elif use_teacher:
            x = self.dense(x)
        else:
            selected_weight, selected_bias = select_wandb(self.dense.weight, self.dense.bias, self.retained_indices)
            x = F.linear(x, selected_weight, selected_bias)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    

class NewRobertaBertForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = NewRobertaModel(config)


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            output_masked_states=False,
            use_unmasked_states=True,
            use_cross_masked_states=False,
            return_dict=None,
            intermediate_z=None,
            head_z=None,
            head_layer_z=None,
            mlp_z=None,
            inference=False,
            hidden_z=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_masked_states=output_masked_states,
            use_unmasked_states=use_unmasked_states,
            use_cross_masked_states=use_cross_masked_states,
            return_dict=return_dict,
            intermediate_z=intermediate_z,
            head_z=head_z,
            mlp_z=mlp_z,
            head_layer_z=head_layer_z,
            inference=inference,
            hidden_z=hidden_z
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        prediction_scores = prediction_scores[labels != -100]
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prune_indices(self, indices_to_prune: Dict[int, List[int]], vo_indices_to_prune: Dict[int, List[int]] = None,
                      q_input_index=None, k_input_index=None,
                      v_input_index=None, o_output_index=None):
        """
        Prunes heads of the base model.

        Arguments:
            heads_to_prune (:obj:`Dict[int, List[int]]`):
                Dictionary with keys being selected layer indices (:obj:`int`) and associated values being the list
                of heads to prune in said layer (list of :obj:`int`). For instance {1: [0, 2], 2: [2, 3]} will
                prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        """
        self.base_model._prune_indices(indices_to_prune, vo_indices_to_prune, q_input_index, k_input_index,
                                       v_input_index, o_output_index)


class NewRobertaEmbeddings(RobertaEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__(config)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.retained_indices = None

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, hidden_z=None, use_teacher=False):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # Copied from transformers.modeling_bert.BertEmbeddings.forward
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        if hidden_z is not None:
            embeddings = embeddings.mul(hidden_z)
        if self.retained_indices is not None and not use_teacher:
            embeddings = embeddings.index_select(-1, self.retained_indices)

        embeddings = self.LayerNorm(embeddings, hidden_z, use_teacher=use_teacher)
        embeddings = self.dropout(embeddings)

        if hidden_z is not None:
            embeddings = embeddings.mul(hidden_z)
            
        return embeddings


class NewRobertaModel(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.encoder = NewRobertaEncoder(config)
        self.embeddings = NewRobertaEmbeddings(config)
        self.embedding_transformer = None
        if getattr(config, "transform_embedding", False):
            self.embedding_transformer = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            output_masked_states=None,
            use_unmasked_states=True,
            use_cross_masked_states=False,
            use_teacher=False,
            return_dict=None,
            intermediate_z=None,
            head_z=None,
            mlp_z=None,
            head_layer_z=None,
            inference=False,
            hidden_z=None
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_z, intermediate_z = split_mask_or_score(self, head_z, intermediate_z)
        if isinstance(head_z, torch.Tensor) and head_z.ndim == 2:
                head_z = self.get_head_mask(head_z, self.config.num_hidden_layers)
        elif isinstance(head_z, list) or isinstance(head_z, tuple):
            head_z = [
                mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) if mask is not None else None
                for mask in head_z
            ]
        

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds,
            hidden_z=hidden_z, use_teacher=use_teacher
        )

        if self.embedding_transformer is not None:
            embedding_output = self.embedding_transformer(embedding_output)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_masked_states=output_masked_states,
            use_unmasked_states=use_unmasked_states,
            use_cross_masked_states=use_cross_masked_states,
            use_teacher=use_teacher,
            return_dict=return_dict,
            intermediate_z=intermediate_z,
            head_z=head_z,
            mlp_z=mlp_z,
            head_layer_z=head_layer_z,
            inference=inference,
            hidden_z=hidden_z
        )
        # self.encoder_outputs = encoder_outputs
        sequence_output = encoder_outputs[0]
        if self.pooler is not None:
            if isinstance(self.pooler, lora.PruningLinear):
                pooled_output = self.pooler(sequence_output, use_teacher=use_teacher)
            else:
                pooled_output = self.pooler(sequence_output)
        else:
            pooled_output = None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return NewBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            attention_layers=encoder_outputs.attention_layers,
            masked_states=encoder_outputs.masked_states,
        )

    def _prune_indices(self, qk_indice_to_prune, vo_indice_to_prune=None, q_input_index=None, k_input_index=None,
                       v_input_index=None, o_output_index=None):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, indices in qk_indice_to_prune.items():
            qk_indices = indices
            if vo_indice_to_prune is not None:
                vo_indices = vo_indice_to_prune[layer]
            else:
                vo_indices = None
            self.encoder.layer[layer].attention.prune_indices(qk_indices, vo_indices)


class NewRobertaEncoder(RobertaEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([NewRobertaLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            output_masked_states=False,
            use_unmasked_states=True,
            use_cross_masked_states=False,
            use_teacher=False,
            return_dict=False,
            intermediate_z=None,
            head_z=None,
            mlp_z=None,
            head_layer_z=None,
            inference=False,
            hidden_z=None
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_attention_outputs = () if output_attentions else None
        all_masked_states = () if output_masked_states else None
        masked_hidden_states = None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    masked_hidden_states,
                    output_attentions,
                    output_masked_states,
                    use_teacher=use_teacher,
                    intermediate_z=intermediate_z[i] if intermediate_z is not None else None,
                    head_z=head_z[i] if head_z is not None else None,
                    mlp_z=mlp_z[i] if mlp_z is not None else None,
                    head_layer_z=head_layer_z[i] if head_layer_z is not None else None,
                    inference=inference,
                    hidden_z=hidden_z
                )
                if output_masked_states and (not use_unmasked_states) and (not use_cross_masked_states):
                    layer_outputs = (layer_outputs[-1],) + layer_outputs[1:-1] + (layer_outputs[0],)
                if output_masked_states and use_cross_masked_states:
                    masked_hidden_states = layer_outputs[-1]
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                all_attention_outputs = all_attention_outputs + (layer_outputs[2],)
            if output_masked_states:
                all_masked_states = all_masked_states + (layer_outputs[-1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_attention_outputs, all_masked_states] if v is not None)
        return NewBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions,
            attention_layers=all_attention_outputs, masked_states=all_masked_states
        )


class NewRobertaLayer(RobertaLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = NewRobertaAttention(config)
        self.intermediate = NewRobertaIntermediate(config)
        self.output = NewRobertaOutput(config)
        self.config = config

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            masked_hidden_states=None,
            output_attentions=False,
            output_masked_states=False,
            use_teacher=False,
            intermediate_z=None,
            head_z=None,
            mlp_z=None,
            head_layer_z=None,
            inference=False,
            hidden_z=None
    ):
        mlp_z = _mask_fine_to_coarse(None, intermediate_z) if mlp_z is None else mlp_z
        if masked_hidden_states is None:
            self_attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_masked_states=output_masked_states,
                use_teacher=use_teacher,
                head_z=head_z,
                head_layer_z=head_layer_z,
                inference=inference,
                hidden_z=hidden_z
            )
        else:
            self_attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                head_mask=None,
                output_attentions=output_attentions,
                output_masked_states=False,
                use_teacher=True,
                head_z=None,
                head_layer_z=None,
                inference=inference,
                hidden_z=hidden_z
            )
            masked_self_attention_outputs = self.attention(
                masked_hidden_states,
                attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_masked_states=False,
                use_teacher=False,
                head_z=head_z,
                head_layer_z=head_layer_z,
                inference=inference,
                hidden_z=hidden_z
            )


        attention_output = self_attention_outputs[0]
        masked_attention_output = self_attention_outputs[-1] if output_masked_states and masked_hidden_states is None else masked_self_attention_outputs[0] if output_masked_states else None
        outputs = self_attention_outputs[1:] if output_masked_states and masked_hidden_states is None else self_attention_outputs[1:-1]  # add self attentions if we output attention weights

        # self.attention_output = attention_output

        if self.intermediate.dense is None or (self.intermediate.block_retained_indices is not None and self.intermediate.block_retained_indices.numel() == 0 and not use_teacher):
            layer_output = attention_output
        else:
            self.inference = inference
            if not output_masked_states:
                self.intermediate_z = intermediate_z
                self.mlp_z = mlp_z
                self.hidden_z = hidden_z
                layer_output = apply_chunking_to_forward(
                    self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output, use_teacher
                )
            else:
                self.intermediate_z = None
                self.mlp_z = None
                self.hidden_z = None
                layer_output = apply_chunking_to_forward(
                    self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output, True
                ) # calculating the masked layer outputs, but might not be the side one
                if intermediate_z.all():
                    masked_layer_output = layer_output
                else:
                    self.intermediate_z = intermediate_z
                    self.mlp_z = mlp_z
                    self.hidden_z = hidden_z
                    masked_layer_output = apply_chunking_to_forward(
                        self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, masked_attention_output, False
                    )
        # self.layer_output = layer_output
        if output_masked_states:
            outputs = (layer_output,) + outputs + (attention_output,) + (masked_layer_output,)
        else:
            outputs = (layer_output,) + outputs + (attention_output,)
        self.intermediate_z = None
        self.mlp_z = None
        self.hidden_z = None
        return outputs

    def feed_forward_chunk(self, attention_output, use_teacher):
        intermediate_output = self.intermediate(attention_output, use_teacher=use_teacher)
        if self.intermediate_z is not None:
            intermediate_output = intermediate_output.mul(self.intermediate_z)
        layer_output = self.output(intermediate_output, attention_output, mlp_z=self.mlp_z, hidden_z=self.hidden_z, inference=self.inference, use_teacher=use_teacher)
        return layer_output
    
class NewRobertaIntermediate(RobertaIntermediate):
    def __init__(self, config):
        super().__init__(config)
        self.block_retained_indices = None
        self.hidden_retained_indices = None

    def forward(self, hidden_states, use_teacher=False):
        if isinstance(self.dense, lora.PruningLinear):
            hidden_states = self.dense(hidden_states, use_teacher=use_teacher, in_retained_indices=self.hidden_retained_indices, out_retained_indices=self.block_retained_indices)
        elif use_teacher:
            hidden_states = self.dense(hidden_states)
        else:
            selected_weight, selected_bias = select_wandb(self.dense.weight, self.dense.bias, self.hidden_retained_indices, self.block_retained_indices)
            hidden_states = F.linear(hidden_states, selected_weight, selected_bias)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

    # def train(self, mode):
    #     t = super().train(mode)
    #     if isinstance(self.dense, lora.LoRALayer):
    #         if not mode:
    #             if self.dense is not None:
    #                 self.dense.eval()
    #         else:
    #             if self.dense is not None:
    #                 self.dense.train()
    #     return t

class NewRobertaAttention(RobertaAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = NewRobertaSelfAttention(config)
        self.output = NewRobertaSelfOutput(config)

        self.config = config

    def prune_indices(self, qk_index, vo_index=None):

        if type(qk_index) == list or isinstance(qk_index, np.ndarray):
            qk_index = torch.LongTensor(qk_index)

        if type(vo_index) == list or isinstance(vo_index, np.ndarray):
            vo_index = torch.LongTensor(vo_index)

        if vo_index is None:
            vo_index = qk_index

        # Prune linear layers
        if len(qk_index) == 0:
            self.self.query = None
            self.self.key = None
        else:
            self.self.query = prune_layer(self.self.query, qk_index)
            self.self.key = prune_layer(self.self.key, qk_index)

        if len(vo_index) == 0:
            self.self.value = None
            self.output.dense = None
        else:
            self.self.value = prune_layer(self.self.value, vo_index)
            self.output.dense = prune_layer(self.output.dense, vo_index, dim=1)

        # print(f"query: {self.self.query.weight.shape if self.self.query is not None else [0, 0]}")
        # print(f"key: {self.self.key.weight.shape if self.self.key is not None else [0, 0]}")
        # print(f"value: {self.self.value.weight.shape if self.self.value is not None else [0, 0]}")
        # print(f"dense: {self.output.dense.weight.shape if self.output.dense is not None else [0, 0]}")

    def prune_heads(self, heads):
        len_heads = len(heads)
        if len_heads == 0:
            return

        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        if len(index) == 0:
            self.self.query = None
            self.self.key = None
            self.self.value = None
            self.output.dense = None
        else:
            self.self.key = prune_layer(self.self.key, index)
            if isinstance(self.self.query, lora.LoRALayer) and self.self.query.r > 0:
                self.self.query = prune_layer(self.self.query, index)
            else:
                self.self.query = prune_layer(self.self.query, index)
            if isinstance(self.self.value, lora.LoRALayer) and self.self.value.r > 0:
                self.self.value = prune_layer(self.self.value, index)
            else:
                self.self.value = prune_layer(self.self.value, index)
            self.output.dense = prune_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.num_teacher_attention_heads = self.self.num_attention_heads
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            output_masked_states=False,
            use_teacher=False,
            head_z=None,
            head_layer_z=None,
            mlp_z=None,
            inference=False,
            hidden_z=None
    ):
        head_layer_z = _mask_fine_to_coarse(None, head_z) if head_layer_z is None else head_layer_z
        if output_masked_states and isinstance(self.self.query, lora.DistillLinear):
            teacher_outputs = self.self(
                hidden_states,
                attention_mask,
                output_attentions,
                output_masked_states=False,
                use_teacher=True,
                head_mask=head_mask,
                head_z=None,
                head_layer_z=None
            )
            student_outputs = self.self(
                hidden_states,
                attention_mask,
                output_attentions,
                output_masked_states=False,
                use_teacher=use_teacher,
                head_mask=head_mask,
                head_z=head_z,
                head_layer_z=head_layer_z
            )
            teacher_attention_outputs = self.output(teacher_outputs[0], hidden_states, masked_hidden_states=None, head_layer_z=None, hidden_z=None,
                                        inference=inference, use_teacher=True)
            student_attention_outputs = self.output(student_outputs[0], hidden_states, masked_hidden_states=None, head_layer_z=head_layer_z, hidden_z=hidden_z,
                                        inference=inference, use_teacher=False)
        else:
            self_outputs = self.self(
                hidden_states,
                attention_mask,
                output_attentions,
                output_masked_states=output_masked_states,
                use_teacher=use_teacher,
                head_mask=head_mask,
                head_z=head_z,
                head_layer_z=head_layer_z
            )
            attention_output = self.output(self_outputs[0], hidden_states, masked_hidden_states=self_outputs[-1] if output_masked_states else None, head_layer_z=head_layer_z, hidden_z=hidden_z,
                                        inference=inference, use_teacher=use_teacher)
        if output_masked_states:
            if isinstance(self.self.query, lora.DistillLinear):
                self_outputs = teacher_outputs + (student_outputs[-1], )
                attention_output = (teacher_attention_outputs, student_attention_outputs)
            if self_outputs[0] is None:
                outputs = (attention_output,) + self_outputs[1:-1] + (attention_output.clone(),)
            else:
                outputs = (attention_output[0],) + self_outputs[1:-1] + (attention_output[1],)  # add attentions if we output them, while also adding the masked hidden states output
        else:
            outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class NewRobertaSelfAttention(RobertaSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.config = config

        self.num_attention_heads = config.num_attention_heads
        self.num_teacher_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.block_retained_indices = None
        self.hidden_retained_indices = None
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, use_teacher=False):
        x_shape = x.size()
        last_dim = x_shape[-1]
        num_attention_head_use = self.num_teacher_attention_heads if use_teacher else self.num_attention_heads
        size_per_head = last_dim // num_attention_head_use
        new_x_shape = x_shape[:-1] + (num_attention_head_use, size_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def project(self, hidden_states: torch.Tensor, proj_layer: nn.Linear, use_teacher: bool = False):
        if isinstance(proj_layer, lora.PruningLinear):
            hidden_states = proj_layer(hidden_states, use_teacher=use_teacher, in_retained_indices=self.hidden_retained_indices, out_retained_indices=self.block_retained_indices)
        elif use_teacher:
            hidden_states = proj_layer(hidden_states)
        else:
            selected_weight, selected_bias = select_wandb(proj_layer.weight, proj_layer.bias, self.hidden_retained_indices, self.block_retained_indices)
            hidden_states = F.linear(hidden_states, selected_weight, selected_bias)
        return hidden_states

    def forward(self,
                hidden_states,
                attention_mask=None,
                output_attentions=False,
                output_masked_states=False,
                use_teacher=False,
                head_mask=None,
                head_z=None,
                head_layer_z=None,
                ):
        num_attention_head_use = self.num_teacher_attention_heads if use_teacher else self.num_attention_heads
        if self.value is None or num_attention_head_use == 0 :
            outputs = (None, None) if output_attentions else (None,)
            if output_masked_states:
                outputs = outputs + (None,)
            return outputs
        # if self.value.weight.sum() == 0: # only for updated full model
        #     return (None, None) if output_attentions else (None, )

        if self.query is None:
            mixed_query_layer = None
        else:
            mixed_query_layer = self.project(hidden_states, self.query, use_teacher=use_teacher)
            mixed_key_layer = self.project(hidden_states, self.key, use_teacher=use_teacher)
            mixed_value_layer = self.project(hidden_states, self.value, use_teacher=use_teacher)


        # batch * sequence_length * dim => batch * sequence_length
        batch_size, seq_length, _ = hidden_states.shape

        if not hasattr(self, "ones"):
            self.ones = torch.ones(batch_size, seq_length, seq_length).float().to(hidden_states.device)

        if mixed_query_layer is not None:
            query_layer = self.transpose_for_scores(mixed_query_layer, use_teacher=use_teacher)
            key_layer = self.transpose_for_scores(mixed_key_layer, use_teacher=use_teacher)
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        else:
            attention_scores = self.ones[:batch_size]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        value_layer = self.transpose_for_scores(mixed_value_layer, use_teacher=use_teacher)
        context_layer = torch.matmul(attention_probs, value_layer)
        masked_context_layer = None
        if head_z is not None or head_mask is not None:
            if head_mask is not None and head_z is not None:
                print("Only one of head_mask and head_z can be used!")
                raise AttributeError
            head_mask_use = head_mask if head_mask is not None else head_z
            if output_masked_states:
                if head_mask_use.all():
                    masked_context_layer = context_layer
                else:
                    masked_context_layer = context_layer * head_mask_use
            else:
                context_layer = context_layer * head_mask_use
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (context_layer.shape[-1] * context_layer.shape[-2],)
        context_layer = context_layer.view(*new_context_layer_shape)
        if output_masked_states:
            masked_context_layer = masked_context_layer.permute(0, 2, 1, 3).contiguous().view(*new_context_layer_shape)

        # from https://github.com/pmichel31415/pytorch-pretrained-BERT/blob/paul/pytorch_pretrained_bert/modeling.py line 306
        if getattr(self, "vo_s", None) is not None:
            context_layer = context_layer.mul(self.vo_s)
        # self.context_layer = context_layer

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if output_masked_states:
            outputs = outputs + (masked_context_layer,)
        return outputs
    
    # def train(self, mode):
    #     t = super().train(mode)
    #     if self.config.apply_lora:
    #         if not mode:
    #             if self.query is not None:
    #                 self.query.eval()
    #             if self.value is not None:
    #                 self.value.eval()
    #         else:
    #             if self.query is not None:
    #                 self.query.train()
    #             if self.value is not None:
    #                 self.value.train()
    #     return t


class NewRobertaSelfOutput(RobertaSelfOutput):
    def __init__(self, config):
        super().__init__(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        self.block_retained_indices = None
        self.hidden_retained_indices = None

    def forward(self, hidden_states, input_tensor, masked_hidden_states=None, head_layer_z=None, hidden_z=None, inference=False, use_teacher=False):
        if masked_hidden_states is None and (hidden_states is None or (head_layer_z is not None and not head_layer_z.item())):
            return input_tensor

        if isinstance(self.dense, lora.PruningLinear):
            hidden_states = self.dense(hidden_states, use_teacher=False, in_retained_indices=self.block_retained_indices, out_retained_indices=self.hidden_retained_indices)
        elif use_teacher:
            hidden_states = self.dense(hidden_states)
        else:           
            selected_weight, selected_bias = select_wandb(self.dense.weight, self.dense.bias, self.block_retained_indices, self.hidden_retained_indices)
            hidden_states = F.linear(hidden_states, selected_weight, selected_bias)
            
        if hidden_z is not None:
            hidden_states = hidden_states.mul(hidden_z)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor, hidden_z, use_teacher=use_teacher)
        if hidden_z is not None:
            hidden_states = hidden_states.mul(hidden_z)

        if masked_hidden_states is not None:
            if head_layer_z is not None and not head_layer_z.item():
                masked_hidden_states = input_tensor
            else:
                masked_hidden_states = self.dense(masked_hidden_states)
                if hidden_z is not None:
                    masked_hidden_states = masked_hidden_states.mul(hidden_z)
                masked_hidden_states = self.dropout(masked_hidden_states)
                masked_hidden_states = self.LayerNorm(masked_hidden_states + input_tensor, hidden_z, use_teacher=False)
            if hidden_z is not None:
                masked_hidden_states = masked_hidden_states.mul(hidden_z)
            return hidden_states, masked_hidden_states
        return hidden_states


class NewRobertaOutput(RobertaOutput):
    def __init__(self, config):
        super().__init__(config)
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        self.block_retained_indices = None
        self.hidden_retained_indices = None

    def forward(self, hidden_states, input_tensor, mlp_z, hidden_z=None, inference=False, use_teacher=False):
        if isinstance(self.dense, lora.PruningLinear):
            hidden_states = self.dense(hidden_states, use_teacher=False, in_retained_indices=self.block_retained_indices, out_retained_indices=self.hidden_retained_indices)
        elif use_teacher:
            hidden_states = self.dense(hidden_states)
        else:
            selected_weight, selected_bias = select_wandb(self.dense.weight, self.dense.bias, self.block_retained_indices, self.hidden_retained_indices)
            hidden_states = F.linear(hidden_states, selected_weight, selected_bias)
            
        if mlp_z is not None:
            hidden_states *= mlp_z
        if not inference and hidden_states.sum().eq(0).item():
            return hidden_states + input_tensor
        else:
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor, hidden_z, use_teacher=use_teacher)
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
        return hidden_states