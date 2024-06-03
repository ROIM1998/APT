# copied from ./modeling_t5.py
# allows removing layers of heads and mlps
import transformers

__version__ = transformers.__version__

from dataclasses import dataclass
from typing import Union

from transformers.file_utils import ModelOutput

from transformers.models.mt5.modeling_mt5 import (
    MT5ForConditionalGeneration,
    MT5Model,
    MT5Stack,
    MT5EncoderModel,
    MT5Block,
    MT5LayerSelfAttention,
    MT5LayerCrossAttention,
    MT5LayerFF,
    MT5Attention,
    MT5LayerNorm,
    MT5DenseActDense,
    MT5DenseGatedActDense,
    MT5Config,
    BaseModelOutputWithPastAndCrossAttentions,
    __HEAD_MASK_WARNING_MSG,
)
from torch.utils.checkpoint import checkpoint

import warnings
import copy
import torch
import torch.nn.functional as F
import loralib as lora

from loralib import Linear as LoRALinear
from loralib.layers import select_wandb, _do_reconstruct_outputs
from torch import nn
from transformers.modeling_utils import find_pruneable_heads_and_indices
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
import logging
from utils.minus_utils import prune_layer, _mask_fine_to_coarse, detect_no_zero
from .modeling_outputs import AdaPBaseModelOutputWithPastAndCrossAttentions

logger = logging.getLogger(__name__)

class AdaPMT5LayerNorm(MT5LayerNorm):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)
        self.retained_indices = None

    def forward(self, hidden_states, hidden_z=None, use_teacher=False):
        if self.retained_indices is not None and not use_teacher:
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            return self.weight.index_select(0, self.retained_indices) * hidden_states
        elif hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            variance = hidden_states[:, :, remaining_index].to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            return self.weight * hidden_states
        else:
            return super().forward(hidden_states)
        
def split_mask_or_score(model, head_mask: torch.Tensor, intermediate_mask: torch.Tensor):
    if isinstance(head_mask, torch.Tensor) and head_mask.ndim == 1:
        head_mask = torch.split(head_mask, [sum(model.enc_selfattn_headnum), sum(model.dec_selfattn_headnum), sum(model.dec_crossattn_headnum)])
        head_mask = (head_mask[0].split(model.enc_selfattn_headnum), head_mask[1].split(model.dec_selfattn_headnum), head_mask[2].split(model.dec_crossattn_headnum))
    if isinstance(intermediate_mask, torch.Tensor) and intermediate_mask.ndim == 1:
        intermediate_mask = torch.split(intermediate_mask, [sum(model.enc_neuron_nums), sum(model.dec_neuron_nums)])
        intermediate_mask = (intermediate_mask[0].split(model.enc_neuron_nums), intermediate_mask[1].split(model.dec_neuron_nums))
    return head_mask, intermediate_mask

class AdaPMT5ForConditionalGeneration(MT5ForConditionalGeneration):
    def __init__(self, config):
        # Manually skip the head pruning of the parent class
        pruned_heads = config.pruned_heads
        config.pruned_heads = {}
        
        super().__init__(config)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = AdaPMT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = AdaPMT5Stack(decoder_config, self.shared)

        self.post_init()
        config.pruned_heads = pruned_heads
        self.config.pruned_heads = pruned_heads
        if not hasattr(self.config, 'adap_pruned_heads'):
            self.config.adap_pruned_heads = {}
        
        self.head_mask = torch.ones(3, config.num_layers, config.num_heads).view(-1)
        self.intermediate_mask = torch.ones(2, config.num_layers, config.d_ff).view(-1)
        self.hidden_mask = torch.ones(config.d_model)
        self.backup_head_mask, self.backup_intermediate_mask, self.backup_hidden_mask = None, None, None
        self.retained_indices = None
        self.virtual_pruned = False
        self.is_teacher = False
        self.is_student = True
        self.is_distilling=False
        self.is_colearning = False
        if hasattr(config, 'do_distill') and config.do_distill:
            if config.apply_lora:
                self.layer_transformation = lora.PruningLinear(config.hidden_size, config.hidden_size, r=8, lora_alpha=16, bias=False)
            else:
                self.layer_transformation = nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=self.dtype)
            # self.layer_transformation.weight.data = torch.eye(config.hidden_size)
        else:
            self.layer_transformation = None
        self.head_layer_z = torch.ones(2 * config.num_layers)
        self.self_head_layer_z = torch.ones(2 * config.num_layers)
        self.cross_head_layer_z = torch.ones(config.num_layers)
        self.mlp_z = torch.ones(2 * config.num_layers)
        
        self.enc_selfattn_headnum = [block.layer[0].SelfAttention.n_heads for block in self.encoder.block]
        self.dec_selfattn_headnum = [block.layer[0].SelfAttention.n_heads for block in self.decoder.block]
        self.dec_crossattn_headnum = [block.layer[1].EncDecAttention.n_heads for block in self.decoder.block]
        self.enc_neuron_nums = [self.config.d_ff for i in range(self.config.num_layers)]
        self.dec_neuron_nums = [self.config.d_ff for i in range(self.config.num_decoder_layers)]
            
    def clear_masks(self):
        self.head_mask = None
        self.intermediate_mask = None
        self.hidden_mask = None
        
    def reset_masks(self):
        encoder_self_attn_mask = [
            torch.ones(self.encoder.block[i].layer[0].SelfAttention.q.weight.shape[0] // self.config.d_kv, device=self.device)
            for i in range(self.config.num_layers)
        ]
        decoder_self_attn_mask = [
            torch.ones(self.decoder.block[i].layer[0].SelfAttention.q.weight.shape[0] // self.config.d_kv, device=self.device)
            for i in range(self.config.num_decoder_layers)
        ]
        decoder_cross_attn_mask = [
            torch.ones(self.decoder.block[i].layer[1].EncDecAttention.q.weight.shape[0] // self.config.d_kv, device=self.device)
            for i in range(self.config.num_decoder_layers)
        ]
        if all([v.shape == encoder_self_attn_mask[0].shape for v in encoder_self_attn_mask]) and all([v.shape == decoder_self_attn_mask[0].shape for v in decoder_self_attn_mask]) and all([v.shape == decoder_cross_attn_mask[0].shape for v in decoder_cross_attn_mask]):
            encoder_self_attn_mask, decoder_self_attn_mask, decoder_cross_attn_mask = torch.stack(encoder_self_attn_mask), torch.stack(decoder_self_attn_mask), torch.stack(decoder_cross_attn_mask)
            self.head_mask = torch.stack([encoder_self_attn_mask, decoder_self_attn_mask, decoder_cross_attn_mask]).view(-1)
        else:
            self.head_mask = torch.cat([encoder_self_attn_mask + decoder_self_attn_mask + decoder_cross_attn_mask])
        encoder_intermediate_mask = [
            torch.ones(self.encoder.block[i].layer[1].DenseReluDense.wo.weight.shape[1]).to(self.device)
            for i in range(self.config.num_layers)
        ]
        decoder_intermediate_mask = [
            torch.ones(self.decoder.block[i].layer[2].DenseReluDense.wo.weight.shape[1]).to(self.device)
            for i in range(self.config.num_decoder_layers)
        ]
        if all([v.shape == encoder_intermediate_mask[0].shape for v in encoder_intermediate_mask]) and all([v.shape == decoder_intermediate_mask[0].shape for v in decoder_intermediate_mask]):
            encoder_intermediate_mask, decoder_intermediate_mask = torch.stack(encoder_intermediate_mask), torch.stack(decoder_intermediate_mask)
            self.intermediate_mask = torch.stack([encoder_intermediate_mask, decoder_intermediate_mask]).view(-1)
        else:
            self.intermediate_mask = torch.cat([encoder_intermediate_mask + decoder_intermediate_mask])
        self.hidden_mask = torch.ones(self.config.d_model).to(self.device)

    def _mask_fine_to_coarse(self, mask):
        if isinstance(mask, torch.Tensor):
            return mask.detach().any(dim=-1).float()
        elif isinstance(mask, list):
            return [v.detach().any().float() if v is not None else None for v in mask]
        
    def split_mask_or_score(self, head_mask = None, intermediate_mask = None):
        head_mask = self.head_mask if head_mask is None else head_mask
        intermediate_mask = self.intermediate_mask if intermediate_mask is None else intermediate_mask
        return split_mask_or_score(self, head_mask, intermediate_mask)
    
    def update_layer_z(self):
        head_mask, intermediate_mask = self.split_mask_or_score(self.backup_head_mask, self.backup_intermediate_mask) if self.virtual_pruned else self.split_mask_or_score()
        self.head_layer_z = torch.cat([v.any().float().unsqueeze(0) for v in head_mask[0]] + [(self_v.any() | cross_v.any()).float().unsqueeze(0) for self_v, cross_v in zip(head_mask[1], head_mask[2])]).to(self.device)
        self.mlp_z = torch.cat([v.any().float().unsqueeze(0) for v in intermediate_mask[0]] + [v.any().float().unsqueeze(0) for v in intermediate_mask[1]]).to(self.device)

    def prune_model_with_masks(self, continual_pruning=True):
        head_mask, intermediate_mask, hidden_mask = self.head_mask, self.intermediate_mask, self.hidden_mask
        if detect_no_zero(head_mask) and detect_no_zero(intermediate_mask) and detect_no_zero(hidden_mask):
            print("No pruning is performed. Skipping pruning.")
            return
        head_mask, intermediate_mask = split_mask_or_score(self, head_mask, intermediate_mask)

        if head_mask is not None:
            encoder_pruned_heads, decoder_pruned_heads = {}, {}
            cross_attn_pruned_heads = {}
            for layer in range(self.config.num_layers):
                pruned_encoder_self_heads, pruned_decoder_self_heads, pruned_cross_attn_heads = head_mask[0][layer] == 0, head_mask[1][layer] == 0, head_mask[2][layer] == 0
                if pruned_encoder_self_heads.all():
                    self.self_head_layer_z[layer] = 0
                if pruned_decoder_self_heads.all():
                    self.self_head_layer_z[layer + self.config.num_layers] = 0
                if pruned_cross_attn_heads.all():
                    self.cross_head_layer_z[layer] = 0
                encoder_self_pruned_heads = self.config.adap_pruned_heads.get('encoder', {}).get(layer, {})  if continual_pruning else {}
                decoder_self_pruned_heads = self.config.adap_pruned_heads.get('decoder', {}).get(layer, {})  if continual_pruning else {}
                cross_pruned_heads = self.config.adap_pruned_heads.get('cross', {}).get(layer, {})  if continual_pruning else {}
                enc_self_retained_head_indices = [x for x in range(self.config.num_heads) if x not in encoder_self_pruned_heads]
                dec_self_retained_head_indices = [x for x in range(self.config.num_heads) if x not in decoder_self_pruned_heads]
                cross_retained_head_indices = [x for x in range(self.config.num_heads) if x not in cross_pruned_heads]
                encoder_pruned_heads[layer] = [enc_self_retained_head_indices[v] for v in torch.where(pruned_encoder_self_heads)[0].tolist()]
                decoder_pruned_heads[layer] = [dec_self_retained_head_indices[v] for v in torch.where(pruned_decoder_self_heads)[0].tolist()]
                cross_attn_pruned_heads[layer] = [cross_retained_head_indices[v] for v in torch.where(pruned_cross_attn_heads)[0].tolist()]
                self.enc_selfattn_headnum[layer] -= len(encoder_pruned_heads[layer])
                self.dec_selfattn_headnum[layer] -= len(decoder_pruned_heads[layer])
                self.dec_crossattn_headnum[layer] -= len(cross_attn_pruned_heads[layer])
                print(f"Encoder layer {layer} prune heads: {encoder_pruned_heads[layer]}. Decoder layer {layer} prune heads: {decoder_pruned_heads[layer]}. Cross-attention layer {layer} prune heads: {cross_attn_pruned_heads[layer]}.")
            self.prune_heads(encoder_pruned_heads, decoder_pruned_heads, cross_attn_pruned_heads)
            self.head_layer_z = self.self_head_layer_z.bool() | torch.cat([torch.zeros(self.config.num_layers), self.cross_head_layer_z]).bool()
            
        # Pruning hidden dimensions
        d_model = self.config.d_model
        # If init from pruend config, d_model should be smaller than hidden mask's length
        if isinstance(hidden_mask, torch.Tensor) and hidden_mask.numel() != d_model:
            print("hidden mask's length is not equal to d_model, hidden mask's length is", hidden_mask.numel(), "d_model is", d_model)
            print("Skipping hidden dimension pruning")
        elif hidden_mask is not None and (hidden_mask == 0).any():
            index = torch.LongTensor(hidden_mask.squeeze().nonzero().squeeze().tolist())
            index = index.to(self.device)

            self.shared.weight = nn.parameter.Parameter(
                self.shared.weight.index_select(1, index).detach().clone())
            self.shared.embedding_dim = index.shape[0]

            for layer in range(0, self.config.num_layers):
                print("Pruning encoder layer:", layer)
                if self.encoder.block[layer].layer[0].SelfAttention.q is not None:
                    self.encoder.block[layer].layer[0].SelfAttention.q = \
                        prune_layer(self.encoder.block[layer].layer[0].SelfAttention.q , index, dim=1)
                    self.encoder.block[layer].layer[0].SelfAttention.k = \
                        prune_layer(self.encoder.block[layer].layer[0].SelfAttention.k , index, dim=1)
                if self.encoder.block[layer].layer[0].SelfAttention.v is not None:
                    self.encoder.block[layer].layer[0].SelfAttention.v = \
                        prune_layer(self.encoder.block[layer].layer[0].SelfAttention.v , index, dim=1)
                    self.encoder.block[layer].layer[0].SelfAttention.o = \
                        prune_layer(self.encoder.block[layer].layer[0].SelfAttention.o , index, dim=0)
                self.encoder.block[layer].layer[0].layer_norm.weight = nn.parameter.Parameter(self.encoder.block[layer].layer[0].layer_norm.weight.index_select(0, index).detach().clone())
                if self.encoder.block[layer].layer[1].DenseReluDense.wo is not None:
                    if hasattr(self.encoder.block[layer].layer[1].DenseReluDense, 'wi'):
                        self.encoder.block[layer].layer[1].DenseReluDense.wi = \
                            prune_layer( self.encoder.block[layer].layer[1].DenseReluDense.wi, index, dim=1)
                    if hasattr(self.encoder.block[layer].layer[1].DenseReluDense, 'wi_0'):
                        self.encoder.block[layer].layer[1].DenseReluDense.wi_0 = \
                            prune_layer( self.encoder.block[layer].layer[1].DenseReluDense.wi_0, index, dim=1)
                    if hasattr(self.encoder.block[layer].layer[1].DenseReluDense, 'wi_1'):
                        self.encoder.block[layer].layer[1].DenseReluDense.wi_1 = \
                            prune_layer( self.encoder.block[layer].layer[1].DenseReluDense.wi_1, index, dim=1)
                    self.encoder.block[layer].layer[1].DenseReluDense.wo = \
                        prune_layer( self.encoder.block[layer].layer[1].DenseReluDense.wo, index, dim=0)
                self.encoder.block[layer].layer[1].layer_norm.weight = nn.parameter.Parameter(self.encoder.block[layer].layer[1].layer_norm.weight.index_select(0, index).detach().clone())
            self.encoder.final_layer_norm.weight = nn.parameter.Parameter(self.encoder.final_layer_norm.weight.index_select(0, index).detach().clone())
            
            for layer in range(0, self.config.num_decoder_layers):
                print("Pruning decoder layer:", layer)
                if self.decoder.block[layer].layer[0].SelfAttention.q is not None:
                    self.decoder.block[layer].layer[0].SelfAttention.q = \
                        prune_layer(self.decoder.block[layer].layer[0].SelfAttention.q , index, dim=1)
                    self.decoder.block[layer].layer[0].SelfAttention.k = \
                        prune_layer(self.decoder.block[layer].layer[0].SelfAttention.k , index, dim=1)
                if self.decoder.block[layer].layer[0].SelfAttention.v is not None:
                    self.decoder.block[layer].layer[0].SelfAttention.v = \
                        prune_layer(self.decoder.block[layer].layer[0].SelfAttention.v , index, dim=1)
                    self.decoder.block[layer].layer[0].SelfAttention.o = \
                        prune_layer(self.decoder.block[layer].layer[0].SelfAttention.o , index, dim=0)
                self.decoder.block[layer].layer[0].layer_norm.weight = nn.parameter.Parameter(self.decoder.block[layer].layer[0].layer_norm.weight.index_select(0, index).detach().clone())
                if self.decoder.block[layer].layer[1].EncDecAttention.q is not None:
                    self.decoder.block[layer].layer[1].EncDecAttention.q = \
                        prune_layer(self.decoder.block[layer].layer[1].EncDecAttention.q , index, dim=1)
                    self.decoder.block[layer].layer[1].EncDecAttention.k = \
                        prune_layer(self.decoder.block[layer].layer[1].EncDecAttention.k , index, dim=1)
                if self.decoder.block[layer].layer[1].EncDecAttention.v is not None:
                    self.decoder.block[layer].layer[1].EncDecAttention.v = \
                        prune_layer(self.decoder.block[layer].layer[1].EncDecAttention.v , index, dim=1)
                    self.decoder.block[layer].layer[1].EncDecAttention.o = \
                        prune_layer(self.decoder.block[layer].layer[1].EncDecAttention.o , index, dim=0)
                self.decoder.block[layer].layer[1].layer_norm.weight = nn.parameter.Parameter(self.decoder.block[layer].layer[1].layer_norm.weight.index_select(0, index).detach().clone())
                
                if self.decoder.block[layer].layer[2].DenseReluDense.wo is not None:
                    if hasattr(self.decoder.block[layer].layer[2].DenseReluDense, 'wi'):
                        self.decoder.block[layer].layer[2].DenseReluDense.wi = \
                            prune_layer( self.decoder.block[layer].layer[2].DenseReluDense.wi, index, dim=1)
                    if hasattr(self.decoder.block[layer].layer[2].DenseReluDense, 'wi_0'):
                        self.decoder.block[layer].layer[2].DenseReluDense.wi_0 = \
                            prune_layer( self.decoder.block[layer].layer[2].DenseReluDense.wi_0, index, dim=1)
                    if hasattr(self.decoder.block[layer].layer[2].DenseReluDense, 'wi_1'):
                        self.decoder.block[layer].layer[2].DenseReluDense.wi_1 = \
                            prune_layer( self.decoder.block[layer].layer[2].DenseReluDense.wi_1, index, dim=1)
                    self.decoder.block[layer].layer[2].DenseReluDense.wo = \
                        prune_layer( self.decoder.block[layer].layer[2].DenseReluDense.wo, index, dim=0)
                self.decoder.block[layer].layer[2].layer_norm.weight = nn.parameter.Parameter(self.decoder.block[layer].layer[2].layer_norm.weight.index_select(0, index).detach().clone())
            self.decoder.final_layer_norm.weight = nn.parameter.Parameter(self.decoder.final_layer_norm.weight.index_select(0, index).detach().clone())

            if hasattr(self, "lm_head"):
                self.lm_head = prune_layer(self.lm_head, index, dim=1)
            if getattr(self, "layer_transformation", None) is not None:
                self.layer_transformation = prune_layer(self.layer_transformation, index, dim=1)
                print("layer transformation", self.layer_transformation.weight.shape)
            if getattr(self, "mha_layer_transformation", None) is not None:
                self.mha_layer_transformation = prune_layer(self.mha_layer_transformation, index, dim=1)
                print("layer mha_layer_transformation", self.mha_layer_transformation.weight.shape)
            # Reduce self's hidden size
            # self.config.d_model = index.shape[0]
            # self.config.hidden_size = index.shape[0]
            
        encoder_kept_intermediate_dims, decoder_kept_intermediate_dims = {}, {}
        if intermediate_mask is not None:
            for layer in range(self.config.num_layers):
                encoder_retained_intermediates = intermediate_mask[0][layer] != 0
                decoder_retained_intermediates = intermediate_mask[1][layer] != 0
                if not encoder_retained_intermediates.any():
                    self.mlp_z[layer] = 0
                if not decoder_retained_intermediates.any():
                    self.mlp_z[layer + self.config.num_layers] = 0
                encoder_kept_intermediate_dims[layer] = torch.where(encoder_retained_intermediates)[0].tolist()
                self.enc_neuron_nums[layer] -= len(torch.where(intermediate_mask[0][layer] == 0)[0].tolist())
                decoder_kept_intermediate_dims[layer] = torch.where(decoder_retained_intermediates)[0].tolist()
                self.dec_neuron_nums[layer] -= len(torch.where(intermediate_mask[1][layer] == 0)[0].tolist())
            self.resize_intermediate(encoder_kept_intermediate_dims, decoder_kept_intermediate_dims)

        self.print_model_shape()
        if isinstance(self.head_mask, torch.Tensor) and self.head_mask.ndim == 1:
            self.head_mask = self.head_mask[self.head_mask.nonzero().squeeze()].detach().clone().contiguous().flatten()
        else:
            self.head_mask = None
        if isinstance(self.intermediate_mask, torch.Tensor) and self.intermediate_mask.ndim == 1:
            self.intermediate_mask = self.intermediate_mask[self.intermediate_mask.nonzero().squeeze()].detach().clone().contiguous().flatten()
        else:
            self.intermediate_mask = None
        if isinstance(self.hidden_mask, torch.Tensor):
            self.hidden_mask = self.hidden_mask[self.hidden_mask.nonzero().squeeze()].detach().clone().contiguous().flatten()

    def virtual_prune(self):
        if self.virtual_pruned:
            print("Model is already virtual pruned. Skipping virtual pruning.", flush=True)
            return
        print("Virtual pruning model", flush=True)
        head_mask, intermediate_mask = self.split_mask_or_score()
        enc_head_masks, dec_head_masks, cross_head_masks = head_mask
        enc_intermediate_masks, dec_intermediate_masks = intermediate_mask
        hidden_mask = self.hidden_mask
        hidden_retained_indices = hidden_mask.nonzero().squeeze()
        num_dim_per_head = self.config.d_model // self.config.num_heads
        self.retained_indices = hidden_retained_indices
        self.encoder.retained_indices = hidden_retained_indices
        self.decoder.retained_indices = hidden_retained_indices
        self.encoder.final_layer_norm.retained_indices = hidden_retained_indices
        self.decoder.final_layer_norm.retained_indices = hidden_retained_indices
        for layer in range(self.config.num_layers):
            enc_mha_mask, dec_mha_mask, cross_mha_mask = enc_head_masks[layer], dec_head_masks[layer], cross_head_masks[layer]
            enc_ffn_mask, dec_ffn_mask = enc_intermediate_masks[layer], dec_intermediate_masks[layer]
            
            # Identify retained heads for continual pruning
            encoder_self_pruned_heads = self.config.adap_pruned_heads.get('encoder', {}).get(layer, {})
            decoder_self_pruned_heads = self.config.adap_pruned_heads.get('decoder', {}).get(layer, {})
            cross_pruned_heads = self.config.adap_pruned_heads.get('cross', {}).get(layer, {})
            enc_self_retained_head_indices = [x for x in range(self.config.num_heads) if x not in encoder_self_pruned_heads]
            dec_self_retained_head_indices = [x for x in range(self.config.num_heads) if x not in decoder_self_pruned_heads]
            cross_retained_head_indices = [x for x in range(self.config.num_heads) if x not in cross_pruned_heads]
            
            # encoder self-mha
            num_retained_enc_heads = enc_mha_mask.sum().int().item()
            pruned_enc_heads = (enc_mha_mask == 0).nonzero().squeeze()
            pruned_enc_heads = pruned_enc_heads.tolist() if pruned_enc_heads.ndim > 0 else [pruned_enc_heads.item()]
            pruned_enc_heads = [enc_self_retained_head_indices[v] for v in pruned_enc_heads]
            enc_mha_retained_indices = torch.repeat_interleave(enc_mha_mask, num_dim_per_head).nonzero().squeeze()
            self.encoder.block[layer].layer[0].SelfAttention.n_teacher_heads = self.encoder.block[layer].layer[0].SelfAttention.n_heads
            self.encoder.block[layer].layer[0].SelfAttention.teacher_pruned_heads = self.encoder.block[layer].layer[0].SelfAttention.pruned_heads
            self.encoder.block[layer].layer[0].SelfAttention.teacher_inner_dim = self.encoder.block[layer].layer[0].SelfAttention.inner_dim
            self.encoder.block[layer].layer[0].SelfAttention.pruned_heads = self.encoder.block[layer].layer[0].SelfAttention.pruned_heads.union(set(pruned_enc_heads))
            self.encoder.block[layer].layer[0].SelfAttention.n_heads = num_retained_enc_heads
            self.encoder.block[layer].layer[0].SelfAttention.inner_dim = self.config.d_kv * num_retained_enc_heads
            self.encoder.block[layer].layer[0].SelfAttention.block_retained_indices = enc_mha_retained_indices
            self.encoder.block[layer].layer[0].SelfAttention.hidden_retained_indices = hidden_retained_indices
            self.encoder.block[layer].layer[0].layer_norm.retained_indices = hidden_retained_indices
            
            # encoder ffn
            enc_ffn_retained_indices = enc_ffn_mask.nonzero().squeeze()
            self.encoder.block[layer].layer[1].DenseReluDense.block_retained_indices = enc_ffn_retained_indices
            self.encoder.block[layer].layer[1].DenseReluDense.hidden_retained_indices = hidden_retained_indices
            self.encoder.block[layer].layer[1].layer_norm.retained_indices = hidden_retained_indices

            # decoder self-mha
            num_retained_dec_self_heads = dec_mha_mask.sum().int().item()
            pruned_dec_self_heads = (dec_mha_mask == 0).nonzero().squeeze()
            pruned_dec_self_heads = pruned_dec_self_heads.tolist() if pruned_dec_self_heads.ndim > 0 else [pruned_dec_self_heads.item()]
            pruned_dec_self_heads = [dec_self_retained_head_indices[v] for v in pruned_dec_self_heads]
            dec_self_mha_retained_indices = torch.repeat_interleave(dec_mha_mask, num_dim_per_head).nonzero().squeeze()
            self.decoder.block[layer].layer[0].SelfAttention.n_teacher_heads = self.decoder.block[layer].layer[0].SelfAttention.n_heads
            self.decoder.block[layer].layer[0].SelfAttention.teacher_pruned_heads = self.decoder.block[layer].layer[0].SelfAttention.pruned_heads
            self.decoder.block[layer].layer[0].SelfAttention.teacher_inner_dim = self.decoder.block[layer].layer[0].SelfAttention.inner_dim
            self.decoder.block[layer].layer[0].SelfAttention.pruned_heads = self.decoder.block[layer].layer[0].SelfAttention.pruned_heads.union(set(pruned_dec_self_heads))
            self.decoder.block[layer].layer[0].SelfAttention.n_heads = num_retained_dec_self_heads
            self.decoder.block[layer].layer[0].SelfAttention.inner_dim = self.config.d_kv * num_retained_dec_self_heads
            self.decoder.block[layer].layer[0].SelfAttention.block_retained_indices = dec_self_mha_retained_indices
            self.decoder.block[layer].layer[0].SelfAttention.hidden_retained_indices = hidden_retained_indices
            self.decoder.block[layer].layer[0].layer_norm.retained_indices = hidden_retained_indices

            # decoder cross-mha
            num_retained_dec_cross_heads = cross_mha_mask.sum().int().item()
            pruned_cross_heads = (cross_mha_mask == 0).nonzero().squeeze()
            pruned_cross_heads = pruned_cross_heads.tolist() if pruned_cross_heads.ndim > 0 else [pruned_cross_heads.item()]
            pruned_cross_heads = [cross_retained_head_indices[v] for v in pruned_cross_heads]
            dec_cross_mha_retained_indices = torch.repeat_interleave(cross_mha_mask, num_dim_per_head).nonzero().squeeze()
            self.decoder.block[layer].layer[1].EncDecAttention.n_teacher_heads = self.decoder.block[layer].layer[1].EncDecAttention.n_heads
            self.decoder.block[layer].layer[1].EncDecAttention.teacher_pruned_heads = self.decoder.block[layer].layer[1].EncDecAttention.pruned_heads
            self.decoder.block[layer].layer[1].EncDecAttention.teacher_inner_dim = self.decoder.block[layer].layer[1].EncDecAttention.inner_dim
            self.decoder.block[layer].layer[1].EncDecAttention.pruned_heads = self.decoder.block[layer].layer[1].EncDecAttention.pruned_heads.union(set(pruned_cross_heads))
            self.decoder.block[layer].layer[1].EncDecAttention.n_heads = num_retained_dec_cross_heads
            self.decoder.block[layer].layer[1].EncDecAttention.inner_dim = self.config.d_kv * num_retained_dec_cross_heads
            self.decoder.block[layer].layer[1].EncDecAttention.block_retained_indices = dec_cross_mha_retained_indices
            self.decoder.block[layer].layer[1].EncDecAttention.hidden_retained_indices = hidden_retained_indices
            self.decoder.block[layer].layer[1].layer_norm.retained_indices = hidden_retained_indices

            # decoder ffn
            dec_ffn_retained_indices = dec_ffn_mask.nonzero().squeeze()
            self.decoder.block[layer].layer[2].DenseReluDense.block_retained_indices = dec_ffn_retained_indices
            self.decoder.block[layer].layer[2].DenseReluDense.hidden_retained_indices = hidden_retained_indices
            self.decoder.block[layer].layer[2].layer_norm.retained_indices = hidden_retained_indices


        self.backup_head_mask = self.head_mask
        self.backup_intermediate_mask = self.intermediate_mask
        self.backup_hidden_mask = self.hidden_mask
        self.head_mask = None
        self.intermediate_mask = None
        self.hidden_mask = None
        self.virtual_pruned = True

    def virtual_prune_restore(self):
        if not self.virtual_pruned:
            print("Model is not virtual pruned. Skipping virtual pruning restoration.", flush=True)
            return
        print("Restoring model from virtual pruning", flush=True)
        self.head_mask, self.intermediate_mask, self.hidden_mask = self.backup_head_mask, self.backup_intermediate_mask, self.backup_hidden_mask
        self.backup_head_mask, self.backup_intermediate_mask, self.backup_hidden_mask = None, None, None
        self.retained_indices = None
        self.encoder.retained_indices = None
        self.decoder.retained_indices = None
        self.encoder.final_layer_norm.retained_indices = None
        self.decoder.final_layer_norm.retained_indices = None
        for layer in range(self.config.num_layers):
            # encoder self-mha
            self.encoder.block[layer].layer[0].SelfAttention.n_heads = self.encoder.block[layer].layer[0].SelfAttention.n_teacher_heads
            self.encoder.block[layer].layer[0].SelfAttention.pruned_heads = self.encoder.block[layer].layer[0].SelfAttention.teacher_pruned_heads
            self.encoder.block[layer].layer[0].SelfAttention.inner_dim = self.encoder.block[layer].layer[0].SelfAttention.teacher_inner_dim
            self.encoder.block[layer].layer[0].SelfAttention.block_retained_indices = None
            self.encoder.block[layer].layer[0].SelfAttention.hidden_retained_indices = None
            self.encoder.block[layer].layer[0].layer_norm.retained_indices = None
            
            # encoder ffn
            self.encoder.block[layer].layer[1].DenseReluDense.block_retained_indices = None
            self.encoder.block[layer].layer[1].DenseReluDense.hidden_retained_indices = None
            self.encoder.block[layer].layer[1].layer_norm.retained_indices = None

            # decoder self-mha
            self.decoder.block[layer].layer[0].SelfAttention.n_heads = self.decoder.block[layer].layer[0].SelfAttention.n_teacher_heads
            self.decoder.block[layer].layer[0].SelfAttention.pruned_heads = self.decoder.block[layer].layer[0].SelfAttention.teacher_pruned_heads
            self.decoder.block[layer].layer[0].SelfAttention.inner_dim = self.decoder.block[layer].layer[0].SelfAttention.teacher_inner_dim
            self.decoder.block[layer].layer[0].SelfAttention.block_retained_indices = None
            self.decoder.block[layer].layer[0].SelfAttention.hidden_retained_indices = None
            self.decoder.block[layer].layer[0].layer_norm.retained_indices = None

            # decoder cross-mha
            self.decoder.block[layer].layer[1].EncDecAttention.n_heads = self.decoder.block[layer].layer[1].EncDecAttention.n_teacher_heads
            self.decoder.block[layer].layer[1].EncDecAttention.pruned_heads = self.decoder.block[layer].layer[1].EncDecAttention.teacher_pruned_heads
            self.decoder.block[layer].layer[1].EncDecAttention.inner_dim = self.decoder.block[layer].layer[1].EncDecAttention.teacher_inner_dim
            self.decoder.block[layer].layer[1].EncDecAttention.block_retained_indices = None
            self.decoder.block[layer].layer[1].EncDecAttention.hidden_retained_indices = None
            self.decoder.block[layer].layer[1].layer_norm.retained_indices = None

            # decoder ffn
            self.decoder.block[layer].layer[2].DenseReluDense.block_retained_indices = None
            self.decoder.block[layer].layer[2].DenseReluDense.hidden_retained_indices = None
            self.decoder.block[layer].layer[2].layer_norm.retained_indices = None
        self.virtual_pruned = False

    def prune_heads(self, encoder_pruned_heads, decoder_pruned_heads, cross_attn_pruned_heads):
        """
        Prunes heads of the base model Moreover, also pruning decoder self-attention heads and cross-attention heads.

        Arguments:
            encoder_pruned_heads (:obj:`Dict[int, List[int]]`):
                Dictionary with keys being selected encoder layer indices (:obj:`int`) and associated values being the list of
                heads to prune in said layer (list of :obj:`int`). For instance {1: [0, 2], 2: [2, 3]} will prune heads
                0 and 2 on layer 1 and heads 2 and 3 on layer 2.
            decoder_pruned_heads (:obj:`Dict[int, List[int]]`):
                Dictionary with keys being selected decoder layer indices (:obj:`int`) and associated self-attention values being the list of
            cross_attn_pruned_heads (:obj:`Dict[int, List[int]]`):
                Dictionary with keys being selected decoder layer indices (:obj:`int`) and associated cross-attention values being the list of
        """
        # save new sets of pruned heads as union of previously stored pruned heads and newly pruned heads
        pruned_heads = self.config.adap_pruned_heads
        # 0 for encoder, 1 for decoder, 2 for cross
        pruned_encoder_heads, pruned_decoder_self_heads, pruned_decoder_cross_heads = pruned_heads.get('encoder', {}), pruned_heads.get('decoder', {}), pruned_heads.get('cross', {})
        for k in 'encoder', 'decoder', 'cross':
            if k not in pruned_heads:
                pruned_heads[k] = {}
        for layer, heads in encoder_pruned_heads.items():
            union_heads = set(pruned_encoder_heads.get(layer, [])) | set(heads)
            self.config.adap_pruned_heads['encoder'][layer] = list(union_heads)  # Unfortunately we have to store it as list for JSON
        for layer, heads in decoder_pruned_heads.items():
            union_heads = set(pruned_decoder_self_heads.get(layer, [])) | set(heads)
            self.config.adap_pruned_heads['decoder'][layer] = list(union_heads)
        for layer, heads in cross_attn_pruned_heads.items():
            union_heads = set(pruned_decoder_cross_heads.get(layer, [])) | set(heads)
            self.config.adap_pruned_heads['cross'][layer] = list(union_heads)
        
        # Pruning heads, as _prune_heads functions
        self.encoder._prune_heads(encoder_pruned_heads)
        self.decoder._prune_heads(decoder_pruned_heads, cross_attn_pruned_heads)



    def resize_intermediate(self, encoder_kept_intermediate_dims, decoder_kept_intermediate_dims):
        encoder, decoder = self.encoder, self.decoder
        device = self.device
        for layer in encoder_kept_intermediate_dims:
            if len(encoder_kept_intermediate_dims[layer]) == 0:
                for layer_name in ['wi', 'wi_0', 'wi_1']:
                    if hasattr(encoder.block[layer].layer[1].DenseReluDense, layer_name):
                        setattr(encoder.block[layer].layer[1].DenseReluDense, layer_name, None)
                encoder.block[layer].layer[1].DenseReluDense.wo = None
            else:
                for layer_name in ['wi', 'wi_0', 'wi_1']:
                    if not hasattr(encoder.block[layer].layer[1].DenseReluDense, layer_name):
                        continue
                    current_layer = getattr(encoder.block[layer].layer[1].DenseReluDense, layer_name)
                    if isinstance(current_layer, lora.Linear) and current_layer.r > 0:
                        setattr(encoder.block[layer].layer[1].DenseReluDense, layer_name, prune_layer(current_layer, index=torch.LongTensor(encoder_kept_intermediate_dims[layer]).to(device), dim=0))
                    else:
                        setattr(encoder.block[layer].layer[1].DenseReluDense, layer_name, prune_layer(current_layer, index=torch.LongTensor(encoder_kept_intermediate_dims[layer]).to(device), dim=0))
                encoder.block[layer].layer[1].DenseReluDense.wo = prune_layer(encoder.block[layer].layer[1].DenseReluDense.wo, index=torch.LongTensor(encoder_kept_intermediate_dims[layer]).to(device), dim=1)

        for layer in decoder_kept_intermediate_dims:
            if len(decoder_kept_intermediate_dims[layer]) == 0:
                for layer_name in ['wi', 'wi_0', 'wi_1']:
                    if hasattr(decoder.block[layer].layer[2].DenseReluDense, layer_name):
                        setattr(decoder.block[layer].layer[2].DenseReluDense, layer_name, None)
                decoder.block[layer].layer[2].DenseReluDense.wo = None
            else:
                for layer_name in ['wi', 'wi_0', 'wi_1']:
                    if not hasattr(decoder.block[layer].layer[2].DenseReluDense, layer_name):
                        continue
                    current_layer = getattr(decoder.block[layer].layer[2].DenseReluDense, layer_name)
                    if isinstance(current_layer, lora.Linear) and current_layer.r > 0:
                        setattr(decoder.block[layer].layer[2].DenseReluDense, layer_name, prune_layer(current_layer, index=torch.LongTensor(decoder_kept_intermediate_dims[layer]).to(device), dim=0))
                    else:
                        setattr(decoder.block[layer].layer[2].DenseReluDense, layer_name, prune_layer(current_layer, index=torch.LongTensor(decoder_kept_intermediate_dims[layer]).to(device), dim=0))
                decoder.block[layer].layer[2].DenseReluDense.wo = prune_layer(decoder.block[layer].layer[2].DenseReluDense.wo, index=torch.LongTensor(decoder_kept_intermediate_dims[layer]).to(device), dim=1)


    def print_model_shape(self):
        print("Encoder:")
        for layer in range(self.config.num_layers):
            print("Layer:", layer)
            if self.encoder.block[layer].layer[0].SelfAttention.q is not None:
                print("query:", self.encoder.block[layer].layer[0].SelfAttention.q.weight.shape)
                print("key:", self.encoder.block[layer].layer[0].SelfAttention.k.weight.shape)
            else:
                print("query:", None)
                print("key:", None)
            if self.encoder.block[layer].layer[0].SelfAttention.v is not None:
                print("value:", self.encoder.block[layer].layer[0].SelfAttention.v.weight.shape)
                print("output:", self.encoder.block[layer].layer[0].SelfAttention.o.weight.shape)
            else:
                print("value:", None)
                print("output:", None)
            wi = self.encoder.block[layer].layer[1].DenseReluDense.wi if hasattr(self.encoder.block[layer].layer[1].DenseReluDense, 'wi') else self.encoder.block[layer].layer[1].DenseReluDense.wi_0
            if wi is not None:
                print("up:", wi.weight.shape)
                print("down:", self.encoder.block[layer].layer[1].DenseReluDense.wo.weight.shape)
            else:
                print("up", None)
                print("down", None)

        print("Decoder:")
        for layer in range(self.config.num_decoder_layers):
            print("Layer:", layer)
            if self.decoder.block[layer].layer[0].SelfAttention.q is not None:
                print("self-attention query:", self.decoder.block[layer].layer[0].SelfAttention.q.weight.shape)
                print("self-attention key:", self.decoder.block[layer].layer[0].SelfAttention.k.weight.shape)
            else:
                print("self-attention query:", None)
                print("self-attention key:", None)
            if self.decoder.block[layer].layer[0].SelfAttention.v is not None:
                print("self-attention value:", self.decoder.block[layer].layer[0].SelfAttention.v.weight.shape)
                print("self-attention output:", self.decoder.block[layer].layer[0].SelfAttention.o.weight.shape)
            else:
                print("self-attention value:", None)
                print("self-attention output:", None)
            if self.decoder.block[layer].layer[1].EncDecAttention.q is not None:
                print("cross-attention query:", self.decoder.block[layer].layer[1].EncDecAttention.q.weight.shape)
                print("cross-attention key:", self.decoder.block[layer].layer[1].EncDecAttention.k.weight.shape)
            else:
                print("cross-attention query:", None)
                print("cross-attention key:", None)
            if self.decoder.block[layer].layer[1].EncDecAttention.v is not None:
                print("cross-attention value:", self.decoder.block[layer].layer[1].EncDecAttention.v.weight.shape)
                print("cross-attention output:", self.decoder.block[layer].layer[1].EncDecAttention.o.weight.shape)
            else:
                print("cross-attention value:", None)
                print("cross-attention output:", None)
            wi = self.decoder.block[layer].layer[2].DenseReluDense.wi if hasattr(self.decoder.block[layer].layer[2].DenseReluDense, 'wi') else self.decoder.block[layer].layer[2].DenseReluDense.wi_0
            if wi is not None:
                print("up:", wi.weight.shape)
                print("down:", self.decoder.block[layer].layer[2].DenseReluDense.wo.weight.shape)
            else:
                print("up", None)
                print("down", None)
    
    def print_lora_info_by_layer(self):
        def print_lora_info(l, layername):
            if isinstance(l, LoRALinear) and hasattr(l, 'lora_A') and hasattr(l, 'lora_B') and l.lora_A is not None and l.lora_B is not None:
                print("%s: r: " % layername, l.r if hasattr(l, 'r') else 0, ', input dim: ', l.lora_A.shape[1] if hasattr(l, 'lora_A') and l.lora_A is not None else 0, ', output dim: ', l.lora_B.shape[0] if hasattr(l, 'lora_B') and l.lora_B is not None else 0)
            elif isinstance(l, LoRALinear):
                print("%s: frozen LoRA layer" % layername)
            else:
                print("%s: frozen Linear layer" % layername)
        for i in range(self.config.num_layers):
            print("Layer:", i)
            layer: AdaPMT5Block = self.encoder.block[i]
            query, key, value, output = layer.layer[0].SelfAttention.q, layer.layer[0].SelfAttention.k, layer.layer[0].SelfAttention.v, layer.layer[0].SelfAttention.o
            up, down = layer.layer[1].DenseReluDense.wi if hasattr(layer.layer[1].DenseReluDense, 'wi') else layer.layer[1].DenseReluDense.wi_0, layer.layer[1].DenseReluDense.wo
            print_lora_info(query, "query")
            print_lora_info(key, "key")
            print_lora_info(value, "value")
            print_lora_info(output, "output")
            print_lora_info(up, "up")
            print_lora_info(down, "down")

        for i in range(self.config.num_decoder_layers):
            print("Layer:", i)
            layer: AdaPMT5Block = self.decoder.block[i]
            self_query, self_key, self_value, self_output = layer.layer[0].SelfAttention.q, layer.layer[0].SelfAttention.k, layer.layer[0].SelfAttention.v, layer.layer[0].SelfAttention.o
            cross_query, cross_key, cross_value, cross_output = layer.layer[1].EncDecAttention.q, layer.layer[1].EncDecAttention.k, layer.layer[1].EncDecAttention.v, layer.layer[1].EncDecAttention.o
            up, down = layer.layer[2].DenseReluDense.wi if hasattr(layer.layer[2].DenseReluDense, 'wi') else layer.layer[2].DenseReluDense.wi_0, layer.layer[2].DenseReluDense.wo
            print_lora_info(self_query, "self-query")
            print_lora_info(self_key, "self-key")
            print_lora_info(self_value, "self-value")
            print_lora_info(self_output, "self-output")
            print_lora_info(cross_query, "cross-query")
            print_lora_info(cross_key, "cross-key")
            print_lora_info(cross_value, "cross-value")
            print_lora_info(cross_output, "cross-output")
            print_lora_info(up, "up")
            print_lora_info(down, "down")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        use_teacher=False,
        head_z=None,
        head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
        pass_mask=True,
    ):
        head_z = head_z if head_z is not None else self.head_mask if pass_mask else None
        intermediate_z = intermediate_z if intermediate_z is not None else self.intermediate_mask if pass_mask else None
        hidden_z = hidden_z if hidden_z is not None else self.hidden_mask if pass_mask else None
        # Using bottom-up pruning, disable layer-level zs
        head_layer_z = None
        mlp_z = None
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask
        
        # Unpack customized head layer masks and intermediate layer masks
        head_z, intermediate_z = split_mask_or_score(self, head_z, intermediate_z)
        if head_z is not None:
            encoder_head_z, decoder_head_z, cross_head_z = head_z
        else:
            encoder_head_z, decoder_head_z, cross_head_z = None, None, None
        
        if head_layer_z is not None:
            encoder_head_layer_z, decoder_head_layer_z, cross_head_layer_z = head_layer_z
        else:
            encoder_head_layer_z, decoder_head_layer_z, cross_head_layer_z = None, None, None
            
        if intermediate_z is not None:
            encoder_intermediate_z, decoder_intermediate_z = intermediate_z
        else:
            encoder_intermediate_z, decoder_intermediate_z = None, None
        
        if mlp_z is not None:
            encoder_mlp_z, decoder_mlp_z = mlp_z
        else:
            encoder_mlp_z, decoder_mlp_z = None, None

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                use_teacher=use_teacher,
                head_z=encoder_head_z,
                head_layer_z=encoder_head_layer_z,
                intermediate_z=encoder_intermediate_z,
                mlp_z=encoder_mlp_z,
                hidden_z=hidden_z,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_teacher=use_teacher,
            head_z=decoder_head_z,
            head_layer_z=decoder_head_layer_z,
            cross_head_z=cross_head_z,
            cross_head_layer_z=cross_head_layer_z,
            intermediate_z=decoder_intermediate_z,
            mlp_z=decoder_mlp_z,
            hidden_z=hidden_z,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        if isinstance(self.lm_head, lora.PruningLinear):
            lm_logits = self.lm_head(sequence_output, use_teacher=use_teacher, in_retained_indices=self.retained_indices)
        elif use_teacher:
            lm_logits = self.lm_head(sequence_output)
        else:
            selected_weight, selected_bias = select_wandb(self.lm_head.weight, self.lm_head.bias, in_retained_indices=self.retained_indices)
            lm_logits = F.linear(sequence_output, selected_weight, selected_bias)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.contiguous().view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

        
class AdaPMT5DenseActDense(MT5DenseActDense):
    def __init__(self, config):
        super().__init__(config)
        self.block_retained_indices = None
        self.hidden_retained_indices = None
        self.wo = lora.SelectLinear(config.d_ff, config.d_model, bias=False)

    def forward(self,
                hidden_states,
                use_teacher=False,
                intermediate_z=None,
        ):
        if isinstance(self.wi, lora.PruningLinear):
            hidden_states = self.wi(hidden_states, use_teacher=use_teacher, in_retained_indices=self.hidden_retained_indices, out_retained_indices=self.block_retained_indices)
        elif use_teacher:
            hidden_states = self.wi(hidden_states)
        else:
            selected_weight, selected_bias = select_wandb(self.wi.weight, self.wi.bias, in_retained_indices=self.hidden_retained_indices, out_retained_indices=self.block_retained_indices)
            hidden_states = F.linear(hidden_states, selected_weight, selected_bias)

        if intermediate_z is not None:
            hidden_states = hidden_states.mul(intermediate_z)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        if isinstance(self.wo, lora.SelectLinear):
            hidden_states = self.wo(hidden_states, use_teacher=use_teacher, in_retained_indices=self.block_retained_indices, out_retained_indices=self.hidden_retained_indices)
        else:
            raise ValueError("wo should be a SelectLinear layer")
        return hidden_states


class AdaPMT5DenseGatedActDense(MT5DenseGatedActDense):
    def __init__(self, config):
        super().__init__(config)
        self.block_retained_indices = None
        self.hidden_retained_indices = None
        self.wo = lora.SelectLinear(config.d_ff, config.d_model, bias=False)

    def forward(self,
                hidden_states,
                use_teacher=False,
                intermediate_z=None,
        ):
        if isinstance(self.wi_0, lora.PruningLinear):
            hidden_gelu = self.wi_0(hidden_states, use_teacher=use_teacher, in_retained_indices=self.hidden_retained_indices, out_retained_indices=self.block_retained_indices)
        elif use_teacher:
            hidden_gelu = self.wi_0(hidden_states)
        else:
            selected_weight, selected_bias = select_wandb(self.wi_0.weight, self.wi_0.bias, in_retained_indices=self.hidden_retained_indices, out_retained_indices=self.block_retained_indices)
            hidden_gelu = F.linear(hidden_states, selected_weight, selected_bias)
        hidden_gelu = self.act(hidden_gelu)
        
        if isinstance(self.wi_1, lora.PruningLinear):
            hidden_linear = self.wi_1(hidden_states, use_teacher=use_teacher, in_retained_indices=self.hidden_retained_indices, out_retained_indices=self.block_retained_indices)
        elif use_teacher:
            hidden_linear = self.wi_1(hidden_states)
        else:
            selected_weight, selected_bias = select_wandb(self.wi_1.weight, self.wi_1.bias, in_retained_indices=self.hidden_retained_indices, out_retained_indices=self.block_retained_indices)
            hidden_linear = F.linear(hidden_states, selected_weight, selected_bias)
        hidden_states = hidden_gelu * hidden_linear
        
        if intermediate_z is not None:
            hidden_states = hidden_states.mul(intermediate_z)
        hidden_states = self.dropout(hidden_states)
        
        if isinstance(self.wo, lora.SelectLinear):
            hidden_states = self.wo(hidden_states, use_teacher=use_teacher, in_retained_indices=self.block_retained_indices, out_retained_indices=self.hidden_retained_indices)
        else:
            raise ValueError("wo should be a SelectLinear layer")
        return hidden_states


class AdaPMT5LayerFF(MT5LayerFF):
    def __init__(self, config):
        super().__init__(config)
        self.layer_norm = AdaPMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        if config.feed_forward_proj == "relu":
            self.DenseReluDense = AdaPMT5DenseActDense(config)
        elif config.feed_forward_proj == "gated-gelu":
            self.DenseReluDense = AdaPMT5DenseGatedActDense(config)
        else:
            raise ValueError(
                f"{self.config.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`"
            )
        # LayerNorm and Dropout layers keep the same as the original MT5 model

    def forward(self,
                hidden_states,
                output_masked_states=False,
                use_teacher=False,
                intermediate_z=None,
                mlp_z=None,
                hidden_z=None,
                ):
        mlp_z = _mask_fine_to_coarse(None, intermediate_z) if mlp_z is None else mlp_z
        is_none = (hasattr(self.DenseReluDense, 'wi') and self.DenseReluDense.wi is None) or (hasattr(self.DenseReluDense, 'wi_0') and self.DenseReluDense.wi_0 is None)
        if is_none or (self.DenseReluDense.block_retained_indices is not None and self.DenseReluDense.block_retained_indices.numel() == 0 and not use_teacher):
            if output_masked_states:
                return hidden_states, hidden_states
            else:
                return hidden_states
        forwarded_states = self.layer_norm(hidden_states, hidden_z=hidden_z, use_teacher=use_teacher)
        if output_masked_states:
            masked_forwarded_states = self.DenseReluDense(
                forwarded_states,
                use_teacher=True,
                intermediate_z=None,
            )
            if hidden_z is not None:
                masked_forwarded_states = (hidden_states + self.dropout(masked_forwarded_states)).mul(hidden_z)
            else:
                masked_forwarded_states = hidden_states + self.dropout(masked_forwarded_states)
            forwarded_states = self.DenseReluDense(
                forwarded_states,
                use_teacher=False,
                intermediate_z=intermediate_z,
            )
        else:
            forwarded_states = self.DenseReluDense(
                forwarded_states,
                use_teacher=use_teacher,
                intermediate_z=intermediate_z,
            )
        if output_masked_states:
            return hidden_states + self.dropout(forwarded_states), masked_forwarded_states
        else:
            hidden_states = hidden_states + self.dropout(forwarded_states)
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
            return hidden_states
        

class AdaPMT5Attention(MT5Attention):
    def __init__(self, config: MT5Config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.block_retained_indices = None
        self.hidden_retained_indices = None
        self.n_teacher_heads = self.n_heads
        self.teacher_pruned_heads = set()
        self.teacher_inner_dim = self.inner_dim
        self.o = lora.SelectLinear(self.inner_dim, self.d_model, bias=False)
        
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # retained_heads = torch.tensor([x for x in range(self.n_heads) if x not in heads], device=self.q.weight.device, dtype=torch.long)

        if len(index) == 0:
            self.q = None
            self.k = None
            self.v = None
            self.o = None
        else:
            # Prune linear layers
            self.q = prune_layer(self.q, index)
            self.k = prune_layer(self.k, index)
            self.v = prune_layer(self.v, index)
            self.o = prune_layer(self.o, index, dim=1)

        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)
        self.n_teacher_heads = self.n_heads
        self.teacher_inner_dim = self.inner_dim
        self.teacher_pruned_heads = self.pruned_heads
        
    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        output_masked_states=False,
        use_teacher=False,
        head_z=None,
        head_layer_z=None,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        n_heads_use = self.n_teacher_heads if use_teacher else self.n_heads
        if self.v is None or n_heads_use == 0:
            outputs = (None, (None, None) if (self.is_decoder and use_cache) else None, position_bias)
            if output_attentions:
                outputs = outputs + (None,)
            if output_masked_states:
                outputs = outputs + (None,)
            return outputs
        
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, (self.n_teacher_heads if use_teacher else self.n_heads), self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.teacher_inner_dim if use_teacher else self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value, use_teacher):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                if isinstance(proj_layer, lora.PruningLinear):
                    hidden_states = shape(proj_layer(hidden_states, use_teacher=use_teacher, in_retained_indices=self.hidden_retained_indices, out_retained_indices=self.block_retained_indices))
                elif use_teacher:
                    hidden_states = shape(proj_layer(hidden_states))
                else:
                    selected_weight, selected_bias = select_wandb(proj_layer.weight, proj_layer.bias, in_retained_indices=self.hidden_retained_indices, out_retained_indices=self.block_retained_indices)
                    hidden_states = shape(F.linear(hidden_states, selected_weight, selected_bias))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                if isinstance(proj_layer, lora.PruningLinear):
                    hidden_states = shape(proj_layer(key_value_states, use_teacher=use_teacher, in_retained_indices=self.hidden_retained_indices, out_retained_indices=self.block_retained_indices))
                elif use_teacher:
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    selected_weight, selected_bias = select_wandb(proj_layer.weight, proj_layer.bias, in_retained_indices=self.hidden_retained_indices, out_retained_indices=self.block_retained_indices)
                    hidden_states = shape(F.linear(key_value_states, selected_weight, selected_bias))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states
        
        if not hasattr(self, "ones"):
            self.ones = torch.ones(batch_size, 1, seq_length, seq_length).float().to(hidden_states.device)
            
        if self.q is None:
            scores = self.ones[:batch_size, :, :real_seq_length, :real_seq_length]
        else:
            # get query states
            if isinstance(self.q, lora.PruningLinear):
                query_states = shape(self.q(hidden_states, use_teacher=use_teacher, in_retained_indices=self.hidden_retained_indices, out_retained_indices=self.block_retained_indices))
            elif use_teacher:
                query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
            else:
                selected_weight, selected_bias = select_wandb(self.q.weight, self.q.bias, in_retained_indices=self.hidden_retained_indices, out_retained_indices=self.block_retained_indices)
                query_states = shape(F.linear(hidden_states, selected_weight, selected_bias))

            # get key/value states
            key_states = project(
                hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None, use_teacher
            )
            value_states = project(
                hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None, use_teacher
            )

            # compute scores
            scores = torch.matmul(
                query_states, key_states.transpose(3, 2)
            )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, (self.n_teacher_heads if use_teacher else self.n_heads) + len(self.teacher_pruned_heads if use_teacher else self.pruned_heads), real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        # print("Position bias is None", position_bias is None, "has relative attention bias", self.has_relative_attention_bias, "past key value", past_key_value is not None, "hidden states", hidden_states.size(1), "key length", key_length)
        # Merged from the fix in the issue here https://github.com/huggingface/transformers/issues/17886
        pruned_heads = self.teacher_pruned_heads if use_teacher else self.pruned_heads
        if not pruned_heads:
            position_bias_masked = position_bias
        else:
            # print("Pruned heads for relative bias")
            mask = torch.ones(position_bias.shape[1])
            mask[list(pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]

        self.tracked_position_bias = position_bias_masked
        
        if position_bias_masked.shape[1] > 0:
            scores += position_bias_masked

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # if layer_head_mask is not None:
        #     attn_weights = attn_weights * layer_head_mask

        
        # Mask heads if we want to
        masked_attn_output = None
        if head_z is not None or layer_head_mask is not None:
            if layer_head_mask is not None and head_z is not None:
                raise AttributeError("Only one of layer_head_mask and head_z can be used!")
            head_mask_use = layer_head_mask if layer_head_mask is not None else head_z
            if output_masked_states:
                if not head_mask_use.all():
                    masked_attn_weights = attn_weights * head_mask_use
                    masked_attn_output = unshape(torch.matmul(masked_attn_weights, value_states))  # (batch_size, seq_length, dim)
                    masked_attn_output = self.o(masked_attn_output)
            else:
                attn_weights = attn_weights * head_mask_use
        
        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        if isinstance(self.o, lora.SelectLinear):
            attn_output = self.o(attn_output, use_teacher=use_teacher, in_retained_indices=self.block_retained_indices, out_retained_indices=self.hidden_retained_indices)
        else:
            raise TypeError("Attention outpu layer 'o' should be a SelectLinear layer")
    
        if output_masked_states and head_mask_use.all():
            masked_attn_output = attn_output

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        if output_masked_states:
            outputs = outputs + (masked_attn_output,)
        return outputs
    
class AdaPMT5LayerSelfAttention(MT5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.SelfAttention = AdaPMT5Attention(config, has_relative_attention_bias)
        self.layer_norm = AdaPMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # Dropout layers keep the same as the original MT5 model
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        output_masked_states=False,
        use_teacher=False,
        head_z=None,
        head_layer_z=None,
        hidden_z=None,
    ):
        head_layer_z = _mask_fine_to_coarse(None, head_z) if head_layer_z is None else head_layer_z
        normed_hidden_states = self.layer_norm(hidden_states, hidden_z=hidden_z, use_teacher=use_teacher)
        if output_masked_states and isinstance(self.SelfAttention.q, lora.DistillLinear):
            teacher_attention_output = self.SelfAttention(
                normed_hidden_states,
                mask=attention_mask,
                position_bias=position_bias,
                layer_head_mask=layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_masked_states=False,
                use_teacher=True,
                head_z=None,
                head_layer_z=None,
            )
            student_attention_output = self.SelfAttention(
                normed_hidden_states,
                mask=attention_mask,
                position_bias=position_bias,
                layer_head_mask=layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_masked_states=False,
                use_teacher=False,
                head_z=head_z,
                head_layer_z=head_layer_z,
            )
            if teacher_attention_output[0] is not None:
                teacher_hidden_states = hidden_states + self.dropout(teacher_attention_output[0])
            else:
                teacher_hidden_states = hidden_states
            if student_attention_output[0] is not None:
                student_hidden_states = hidden_states + self.dropout(student_attention_output[0])
            else:
                student_hidden_states = hidden_states
            return (teacher_hidden_states,) + teacher_attention_output[1:] + (student_hidden_states,)
        else:
            attention_output = self.SelfAttention(
                normed_hidden_states,
                mask=attention_mask,
                position_bias=position_bias,
                layer_head_mask=layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_masked_states=output_masked_states,
                use_teacher=use_teacher,
                head_z=head_z,
                head_layer_z=head_layer_z,
            )
            if attention_output[0] is not None:
                hidden_states = hidden_states + self.dropout(attention_output[0])
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
            outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
            if output_masked_states:
                outputs = outputs + (attention_output[-1],)
            return outputs
    
class AdaPMT5LayerCrossAttention(MT5LayerCrossAttention):
    def __init__(self, config):
        super().__init__(config)
        self.EncDecAttention = AdaPMT5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = AdaPMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        #Dropout layers keep the same as the original MT5 model

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
        output_masked_states=False,
        use_teacher=False,
        head_z=None,
        head_layer_z=None,
        hidden_z=None,
    ):
        head_layer_z = _mask_fine_to_coarse(None, head_z) if head_layer_z is None else head_layer_z
        normed_hidden_states = self.layer_norm(hidden_states, hidden_z=hidden_z, use_teacher=use_teacher)
        if output_masked_states and isinstance(self.EncDecAttention.q, lora.DistillLinear):
            teacher_attention_output = self.EncDecAttention(
                normed_hidden_states,
                mask=attention_mask,
                key_value_states=key_value_states,
                position_bias=position_bias,
                layer_head_mask=layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                query_length=query_length,
                output_attentions=output_attentions,
                output_masked_states=output_masked_states,
                use_teacher=True,
                head_z=None,
                head_layer_z=None,
            )
            student_attention_output = self.EncDecAttention(
                normed_hidden_states,
                mask=attention_mask,
                key_value_states=key_value_states,
                position_bias=position_bias,
                layer_head_mask=layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                query_length=query_length,
                output_attentions=output_attentions,
                output_masked_states=output_masked_states,
                use_teacher=False,
                head_z=head_z,
                head_layer_z=head_layer_z,
            )
            if teacher_attention_output[0] is not None:
                teacher_hidden_states = hidden_states + self.dropout(teacher_attention_output[0])
            else:
                teacher_hidden_states = hidden_states
            if student_attention_output[0] is not None:
                student_hidden_states = hidden_states + self.dropout(student_attention_output[0])
            else:
                student_hidden_states = hidden_states
            return (teacher_hidden_states,) + teacher_attention_output[1:] + (student_hidden_states,)
        else:
            attention_output = self.EncDecAttention(
                normed_hidden_states,
                mask=attention_mask,
                key_value_states=key_value_states,
                position_bias=position_bias,
                layer_head_mask=layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                query_length=query_length,
                output_attentions=output_attentions,
                output_masked_states=output_masked_states,
                use_teacher=use_teacher,
                head_z=head_z,
                head_layer_z=head_layer_z,
            )
            if attention_output[0] is not None:
                hidden_states = hidden_states + self.dropout(attention_output[0])
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
            outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
            if output_masked_states:
                outputs = outputs + (attention_output[-1],)
            return outputs
    
    
class AdaPMT5Block(MT5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.layer = nn.ModuleList([
            AdaPMT5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias),
        ])            
        if self.is_decoder:
            self.layer.append(AdaPMT5LayerCrossAttention(config))
        self.layer.append(AdaPMT5LayerFF(config))
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        output_masked_states=False,
        use_teacher=False,
        head_z=None,
        head_layer_z=None,
        cross_head_z=None,
        cross_head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
    ):

        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            use_teacher=use_teacher,
            head_z=head_z,
            head_layer_z=head_layer_z,
            hidden_z=hidden_z,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None and present_key_value_state[0] is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
                use_teacher=use_teacher,
                head_z=cross_head_z,
                head_layer_z=cross_head_layer_z,
                hidden_z=hidden_z,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](
                hidden_states,
                use_teacher=use_teacher,
                intermediate_z=intermediate_z,
                mlp_z=mlp_z,
                hidden_z=hidden_z,
            )

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        if output_masked_states:
            return (outputs,) + () # adding in-layer masked hidden states output
            # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights), (masked hidden states)
        else:
            return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)

class AdaPMT5Stack(MT5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)
        self.block = nn.ModuleList(
            [AdaPMT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = AdaPMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.retained_indices = None
        self.init_weights()

    def _prune_heads(self, self_heads_to_prune, cross_heads_to_prune=None):
        for layer, heads in self_heads_to_prune.items():
            self.block[layer].layer[0].SelfAttention.prune_heads(heads)
        if cross_heads_to_prune is not None:
            for layer, heads in cross_heads_to_prune.items():
                self.block[layer].layer[1].EncDecAttention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        output_masked_states=False,
        use_teacher=False,
        head_z=None,
        head_layer_z=None,
        cross_head_z=None,
        cross_head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None and past_key_values[0][0] is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f":obj:`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        # Also, prepare head_z if needed
        if head_z is not None:
            if isinstance(head_z, torch.Tensor):
                head_z = self.get_head_mask(head_z, self.config.num_layers)
            else:
                head_z = [mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) if mask is not None else None for mask in head_z]
        if cross_head_z is not None:
            if isinstance(cross_head_z, torch.Tensor):
                cross_head_z = self.get_head_mask(cross_head_z, self.config.num_layers)
            else:
                cross_head_z = [mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) if mask is not None else None for mask in cross_head_z]
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        if self.retained_indices is not None and not use_teacher:
            inputs_embeds = inputs_embeds.index_select(-1, self.retained_indices)
        hidden_states = self.dropout(inputs_embeds)
        if hidden_z is not None:
            hidden_states = hidden_states.mul(hidden_z)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                # print("Decoder" if self.config.is_decoder else "Encoder", "Layer", i)
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    use_teacher=use_teacher,
                    head_z=head_z[i] if head_z is not None else None,
                    head_layer_z=head_layer_z[i] if head_layer_z is not None else None,
                    cross_head_z=cross_head_z[i] if cross_head_z is not None else None,
                    cross_head_layer_z=cross_head_layer_z[i] if cross_head_layer_z is not None else None,
                    intermediate_z=intermediate_z[i] if intermediate_z is not None else None,
                    mlp_z=mlp_z[i] if mlp_z is not None else None,
                    hidden_z=hidden_z,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states, hidden_z=hidden_z, use_teacher=use_teacher)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return AdaPBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )