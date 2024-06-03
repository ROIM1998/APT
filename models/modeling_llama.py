import transformers

from transformers.utils import logging
from transformers.models.llama.modeling_llama import (
    LlamaForSequenceClassification,
    LlamaForCausalLM,
    LlamaModel,
    LlamaDecoderLayer,
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaConfig,
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
    apply_rotary_pos_emb,
    repeat_kv,
)

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import loralib as lora

from dataclasses import dataclass
from typing import Union, Optional, Tuple, List, Dict
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from utils.minus_utils import prune_layer, _mask_fine_to_coarse, detect_no_zero
from loralib.layers import select_wandb, _do_reconstruct_outputs
from transformers.modeling_utils import find_pruneable_heads_and_indices

logger = logging.get_logger(__name__)

class ElasticLlamaRMSNorm(LlamaRMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__(hidden_size, eps)
        self.retained_indices = None

    def forward(self, hidden_states, hidden_z=None, use_teacher=False):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if self.retained_indices is not None and not use_teacher:
            if hidden_states.shape[-1] != self.retained_indices.shape[-1]:
                variance = hidden_states[..., self.retained_indices].pow(2).mean(-1, keepdim=True)
            else:
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
            weight_use = self.weight.index_select(0, self.retained_indices)
        elif hidden_z is not None:
            remaining_indices = torch.where(~hidden_z.eq(0))[0]
            variance = hidden_states[..., remaining_indices].pow(2).mean(-1, keepdim=True)
            weight_use = self.weight
        else:
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            weight_use = self.weight
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return weight_use * hidden_states.to(input_dtype)
    

def split_mask_or_score(model, head_mask: torch.Tensor, intermediate_mask: torch.Tensor):
    if isinstance(head_mask, torch.Tensor) and head_mask.ndim == 1:
        head_mask = torch.split(head_mask, [model.model.layers[i].self_attn.num_heads for i in range(model.config.num_hidden_layers)])
    if isinstance(intermediate_mask, torch.Tensor) and intermediate_mask.ndim == 1:
        intermediate_mask = torch.split(intermediate_mask, [model.model.layers[i].mlp.gate_proj.weight.shape[0] if model.model.layers[i].mlp.gate_proj is not None else 0 for i in range(model.config.num_hidden_layers)])
    return head_mask, intermediate_mask

class ElasticLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = ElasticLlamaModel(config)
        
        self.head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads).view(-1)
        self.intermediate_mask = torch.ones(config.num_hidden_layers, config.intermediate_size).view(-1)
        self.hidden_mask = torch.ones(config.hidden_size)
        self.backup_head_mask, self.backup_intermediate_mask, self.backup_hidden_mask = None, None, None
        self.retained_indices = None
        self.virtual_pruned = False
        self.is_teacher = False
        self.is_student = True
        self.is_distilling=False
        self.is_colearning = False
        if hasattr(config, 'do_distill') and config.do_distill:
            if config.apply_lora:
                self.layer_transformation = lora.PruningLinear(config.hidden_size, config.hidden_size, r=8, lora_alpha=16, bias=False, dtype=self.dtype)
            else:
                self.layer_transformation = nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=self.dtype)
            # self.layer_transformation.weight.data = torch.eye(config.hidden_size)
        else:
            self.layer_transformation = None
        
        self.head_layer_z = torch.ones(config.num_hidden_layers)
        self.mlp_z = torch.ones(config.num_hidden_layers)
            
    def clear_masks(self):
        self.head_mask = None
        self.intermediate_mask = None
        self.hidden_mask = None
        
    def reset_masks(self):
        head_nums = [self.model.layers[i].self_attn.num_heads for i in range(self.config.num_hidden_layers)]
        intermediate_sizes = [self.model.layers[i].mlp.gate_proj.weight.shape[0] for i in range(self.config.num_hidden_layers)]
        self.head_mask = torch.ones(sum(head_nums)).view(-1).to(self.device)
        self.intermediate_mask = torch.ones(sum(intermediate_sizes)).view(-1).to(self.device)
        
    def _mask_fine_to_coarse(self, mask):
        if isinstance(mask, torch.Tensor):
            return mask.detach().any(dim=-1).float()
        elif isinstance(mask, list) or isinstance(mask, tuple):
            return [v.detach().any().float().unsqueeze(0) if v is not None else None for v in mask]
        
    def split_mask_or_score(self, head_mask = None, intermediate_mask = None):
        head_mask = self.head_mask if head_mask is None else head_mask
        intermediate_mask = self.intermediate_mask if intermediate_mask is None else intermediate_mask
        return split_mask_or_score(self, head_mask, intermediate_mask)
    
    def update_layer_z(self):
        head_mask, intermediate_mask = self.split_mask_or_score(self.backup_head_mask, self.backup_intermediate_mask) if self.virtual_pruned else self.split_mask_or_score(self.head_mask, self.intermediate_mask)
        self.head_layer_z = torch.cat(self._mask_fine_to_coarse(head_mask)).float()
        self.mlp_z = torch.cat(self._mask_fine_to_coarse(intermediate_mask)).float()
        
    # TODO: define prune model with masks function here
    def prune_model_with_masks(self, continual_pruning=True):
        assert isinstance(self.config, LlamaConfig)
        head_mask, intermediate_mask, hidden_mask = self.head_mask, self.intermediate_mask, self.hidden_mask
        if detect_no_zero(head_mask) and detect_no_zero(intermediate_mask) and detect_no_zero(hidden_mask):
            print("No pruning is performed. Skipping pruning.")
            return
        pruned_history = {}
        # pruned_history['params'] = dict(self.named_parameters())
        # pruned_history['modules'] = dict(self.named_modules())
        head_mask, intermediate_mask = self.split_mask_or_score(head_mask, intermediate_mask)
        pruned_history['head_mask'] = head_mask
        pruned_history['intermediate_mask'] = intermediate_mask
        pruned_history['hidden_mask'] = self.hidden_mask

        if head_mask is not None:
            decoder_pruned_heads = {}
            pruned_heads = self.config.pruned_heads if continual_pruning else {}
            # Encoder-only architecture
            for layer in range(head_mask.shape[0] if isinstance(head_mask, torch.Tensor) else len(head_mask)):
                if head_mask[layer] is None:
                    continue
                head_to_prune = head_mask[layer] == 0
                if head_to_prune.all():
                    self.head_layer_z[layer] = 0
                now_pruning_heads = torch.where(head_to_prune)[0].tolist()
                # Shift the now_pruning_heads based on the pruned heads (re-index them as if all heads are kept)
                if layer in pruned_heads and pruned_heads[layer]:
                    retained_head_indices = [x for x in range(self.config.num_attention_heads) if x not in pruned_heads[layer]]
                    decoder_pruned_heads[layer] = [retained_head_indices[x] for x in now_pruning_heads]
                else:
                    decoder_pruned_heads[layer] = now_pruning_heads
            self.prune_heads(decoder_pruned_heads)
            
        # Pruning hidden dimensions
        hidden_size = self.lm_head.weight.shape[1]
        # If init from pruend config, hidden_size should be smaller than hidden mask's length
        if isinstance(hidden_mask, torch.Tensor) and hidden_mask.numel() != hidden_size:
            print("hidden mask's length is not equal to hidden_size, hidden mask's length is", hidden_mask.numel(), "hidden_size is", hidden_size)
            print("Skipping hidden dimension pruning")
        elif hidden_mask is not None and (hidden_mask == 0).any():
            index = torch.LongTensor(hidden_mask.squeeze().nonzero().squeeze().tolist())
            index = index.to(self.device)

            self.model.embed_tokens.weight = torch.nn.parameter.Parameter(
                self.model.embed_tokens.weight.index_select(1, index).detach().clone())
            self.model.embed_tokens.embedding_dim = index.shape[0]
            self.model.norm.weight = torch.nn.parameter.Parameter(
                self.model.norm.weight.index_select(0, index).detach().clone()
            )

            for layer in range(0, self.config.num_hidden_layers):
                print("Pruning layer:", layer)
                if self.model.layers[layer].self_attn.q_proj is not None:
                    self.model.layers[layer].self_attn.q_proj = \
                        prune_layer(self.model.layers[layer].self_attn.q_proj , index, dim=1)
                    self.model.layers[layer].self_attn.k_proj = \
                        prune_layer(self.model.layers[layer].self_attn.k_proj , index, dim=1)
                if self.model.layers[layer].self_attn.v_proj is not None:
                    self.model.layers[layer].self_attn.v_proj = \
                        prune_layer(self.model.layers[layer].self_attn.v_proj , index, dim=1)
                    self.model.layers[layer].self_attn.o_proj = \
                        prune_layer( self.model.layers[layer].self_attn.o_proj , index, dim=0)
                self.model.layers[layer].input_layernorm.weight = nn.Parameter(self.model.layers[layer].input_layernorm.weight.index_select(0, index).detach().clone())
                self.model.layers[layer].post_attention_layernorm.weight = nn.Parameter(self.model.layers[layer].post_attention_layernorm.weight.index_select(0, index).detach().clone())
                    
                if self.model.layers[layer].mlp.up_proj is not None:
                    self.model.layers[layer].mlp.up_proj = \
                        prune_layer( self.model.layers[layer].mlp.up_proj, index, dim=1)
                    self.model.layers[layer].mlp.gate_proj = \
                        prune_layer( self.model.layers[layer].mlp.gate_proj, index, dim=1)
                    self.model.layers[layer].mlp.down_proj = \
                        prune_layer( self.model.layers[layer].mlp.down_proj, index, dim=0)

            # accommodate for different models
            self.lm_head = prune_layer(self.lm_head, index, dim=1)

            if getattr(self, "layer_transformation", None) is not None:
                self.layer_transformation = prune_layer(self.layer_transformation, index, dim=1)
                print("layer transformation", self.layer_transformation.weight.shape)
            if getattr(self, "mha_layer_transformation", None) is not None:
                self.mha_layer_transformation = prune_layer(self.mha_layer_transformation, index, dim=1)
                print("layer mha_layer_transformation", self.mha_layer_transformation.weight.shape)
            # Reduce model's hidden size
            # model.config.hidden_size = index.shape[0]
            
        encoder_kept_intermediate_dims = {}
        if intermediate_mask is not None:
            for layer in range(intermediate_mask.shape[0] if isinstance(intermediate_mask, torch.Tensor) else len(intermediate_mask)):
                if intermediate_mask[layer] is None:
                    continue
                intermediate_to_retain = intermediate_mask[layer] != 0
                if not intermediate_to_retain.any():
                    self.mlp_z[layer] = 0
                encoder_kept_intermediate_dims[layer] = torch.where(intermediate_to_retain)[0].tolist()
            self.resize_intermediate(encoder_kept_intermediate_dims)
        self.print_model_shape()
        self.head_mask = torch.cat([v[v.nonzero().squeeze()].detach().contiguous().flatten() for v in head_mask])
        self.intermediate_mask = torch.cat([v[v.nonzero().squeeze()].detach().contiguous().flatten() for v in intermediate_mask])
        self.hidden_mask = hidden_mask[hidden_mask.nonzero().squeeze()].detach().contiguous().flatten() if hidden_mask is not None else None
        self.pruned_history = pruned_history
        
    def virtual_prune(self):
        if self.virtual_pruned:
            print("Model is already virtual pruned. Skipping virtual pruning.", flush=True)
            return
        print("Virtual pruning model.", flush=True)
        head_mask, intermediate_mask = self.split_mask_or_score()
        hidden_mask = self.hidden_mask
        hidden_retained_indices = hidden_mask.nonzero().squeeze()
        assert isinstance(self.config, LlamaConfig)
        num_dim_per_head = self.config.hidden_size // self.config.num_attention_heads
        self.retained_indices = hidden_retained_indices
        self.model.retained_indices = hidden_retained_indices
        self.model.norm.retained_indices = hidden_retained_indices
        for layer in range(self.config.num_hidden_layers):
            layer_head_mask, layer_intermediate_mask = head_mask[layer], intermediate_mask[layer]
            
            # Identify retained heads for continual pruning
            decoder_self_pruned_heads = self.config.pruned_heads.get(layer, {})
            retained_head_indices = [x for x in range(self.config.num_attention_heads) if x not in decoder_self_pruned_heads]
            
            # decoder self-mha
            num_retained_heads = layer_head_mask.sum().int().item()
            pruned_heads = (layer_head_mask == 0).nonzero().squeeze()
            pruned_heads = pruned_heads.tolist() if pruned_heads.ndim > 0 else [pruned_heads.item()]
            pruned_heads = [retained_head_indices[v] for v in pruned_heads]
            enc_mha_retained_indices = torch.repeat_interleave(layer_head_mask, num_dim_per_head).nonzero().squeeze()
            
            decoder_layer: ElasticLlamaDecoderLayer = self.model.layers[layer]
            attn_layer: ElasticLlamaAttention = decoder_layer.self_attn
            attn_layer.num_teacher_heads = attn_layer.num_heads
            attn_layer.teacher_pruned_heads = attn_layer.pruned_heads
            attn_layer.num_teacher_key_value_heads = attn_layer.num_key_value_heads
            attn_layer.teacher_hidden_size = attn_layer.hidden_size
            
            attn_layer.pruned_heads = attn_layer.pruned_heads.union(set(pruned_heads))
            attn_layer.num_heads = num_retained_heads
            attn_layer.num_key_value_heads = num_retained_heads
            attn_layer.hidden_size = attn_layer.num_heads * num_dim_per_head
            
            attn_layer.block_retained_indices = enc_mha_retained_indices
            attn_layer.hidden_retained_indices = hidden_retained_indices
            decoder_layer.input_layernorm.retained_indices = hidden_retained_indices
            decoder_layer.post_attention_layernorm.retained_indices = hidden_retained_indices
            
            # decoder ffn
            ffn_retained_indices = layer_intermediate_mask.nonzero().squeeze()
            ffn_layer: ElasticLlamaMLP = decoder_layer.mlp
            ffn_layer.hidden_retained_indices = hidden_retained_indices
            ffn_layer.block_retained_indices = ffn_retained_indices

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
        self.model.retained_indices = None
        self.model.norm.retained_indices = None

        for layer in range(self.config.num_hidden_layers):
            decoder_layer: ElasticLlamaDecoderLayer = self.model.layers[layer]
            attn_layer: ElasticLlamaAttention = decoder_layer.self_attn
            # decoder self-mha
            attn_layer.num_heads = attn_layer.num_teacher_heads
            attn_layer.num_key_value_heads = attn_layer.num_teacher_key_value_heads
            attn_layer.hidden_size = attn_layer.teacher_hidden_size
            attn_layer.pruned_heads = attn_layer.teacher_pruned_heads
            attn_layer.block_retained_indices = None
            attn_layer.hidden_retained_indices = None
            decoder_layer.input_layernorm.retained_indices = None
            decoder_layer.post_attention_layernorm.retained_indices = None
            
            # decoder ffn
            ffn_layer: ElasticLlamaMLP = decoder_layer.mlp
            ffn_layer.hidden_retained_indices = None
            ffn_layer.block_retained_indices = None

        self.virtual_pruned = False
        
    def resize_intermediate(self, kept_intermediate_dims: Dict[int, List[int]]):
        model = self.model
        device = self.device
        for layer in kept_intermediate_dims:
            if len(kept_intermediate_dims[layer]) == 0:
                model.layers[layer].mlp.up_proj = None
                model.layers[layer].mlp.gate_proj = None
                model.layers[layer].mlp.down_proj = None
            else:
                model.layers[layer].mlp.gate_proj = prune_layer(model.layers[layer].mlp.gate_proj, index=torch.LongTensor(kept_intermediate_dims[layer]).to(device), dim=0)
                model.layers[layer].mlp.up_proj = prune_layer(model.layers[layer].mlp.up_proj, index=torch.LongTensor(kept_intermediate_dims[layer]).to(device), dim=0)
                model.layers[layer].mlp.down_proj = prune_layer(model.layers[layer].mlp.down_proj, index=torch.LongTensor(kept_intermediate_dims[layer]).to(device), dim=1)
            
    def print_model_shape(self):
        for layer in range(self.config.num_hidden_layers):
            print("Layer:", layer)
            if self.model.layers[layer].self_attn.q_proj is not None:
                print("self-attention query:", self.model.layers[layer].self_attn.q_proj.weight.shape)
                print("self-attention key:", self.model.layers[layer].self_attn.k_proj.weight.shape)
            else:
                print("self-attention query:", None)
                print("self-attention key:", None)
            if self.model.layers[layer].self_attn.v_proj is not None:
                print("self-attention value:", self.model.layers[layer].self_attn.v_proj.weight.shape)
                print("self-attention output:", self.model.layers[layer].self_attn.o_proj.weight.shape)
            else:
                print("self-attention value:", None)
                print("self-attention output:", None)
            wi = self.model.layers[layer].mlp.gate_proj
            if wi is not None:
                print("up & gated:", wi.weight.shape)
                print("down:", self.model.layers[layer].mlp.down_proj.weight.shape)
            else:
                print("up & gated", None)
                print("down", None)
    
    def print_lora_info_by_layer(self):
        def print_lora_info(l, layername):
            if isinstance(l, lora.Linear) and hasattr(l, 'lora_A') and hasattr(l, 'lora_B') and l.lora_A is not None and l.lora_B is not None:
                print("%s: r: " % layername, l.r if hasattr(l, 'r') else 0, ', input dim: ', l.lora_A.shape[1] if hasattr(l, 'lora_A') and l.lora_A is not None else 0, ', output dim: ', l.lora_B.shape[0] if hasattr(l, 'lora_B') and l.lora_B is not None else 0)
            elif isinstance(l, lora.Linear):
                print("%s: frozen LoRA layer" % layername)
            else:
                print("%s: frozen Linear layer" % layername)
        for i in range(self.config.num_hidden_layers):
            print("Layer:", i)
            layer: ElasticLlamaDecoderLayer = self.model.layers[i]
            query, key, value, output = layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj, layer.self_attn.o_proj
            up, gate, down = layer.mlp.up_proj, layer.mlp.gate_proj, layer.mlp.down_proj
            print_lora_info(query, "query")
            print_lora_info(key, "key")
            print_lora_info(value, "value")
            print_lora_info(output, "output")
            print_lora_info(up, "up")
            print_lora_info(gate, "gate")
            print_lora_info(down, "down")
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_teacher: bool =False,
        head_z: Optional[torch.Tensor]=None,
        head_layer_z: Optional[torch.Tensor] = None,
        intermediate_z: Optional[torch.Tensor] = None,
        mlp_z: Optional[torch.Tensor] = None,
        hidden_z: Optional[torch.Tensor] = None,
        pass_mask: bool = True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        head_z = head_z if head_z is not None else self.head_mask if pass_mask else None
        intermediate_z = intermediate_z if intermediate_z is not None else self.intermediate_mask if pass_mask else None
        hidden_z = hidden_z if hidden_z is not None else self.hidden_mask if pass_mask else None
        # Using bottom-up pruning, disable layer-level zs
        head_layer_z = None
        mlp_z = None
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        head_z, intermediate_z = split_mask_or_score(self, head_z, intermediate_z)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_teacher=use_teacher,
            head_z=head_z,
            head_layer_z=head_layer_z,
            intermediate_z=intermediate_z,
            mlp_z=mlp_z,
            hidden_z=hidden_z,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            if isinstance(self.lm_head, lora.PruningLinear):
                logits = self.lm_head(hidden_states, use_teacher=use_teacher, in_retained_indices=self.retained_indices)
            elif use_teacher:
                logits = self.lm_head(hidden_states)
            else:
                selected_weight, selected_bias = select_wandb(self.lm_head.weight, self.lm_head.bias, in_retained_indices=self.retained_indices)
                logits = F.linear(hidden_states, selected_weight, selected_bias)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
        
class ElasticLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([ElasticLlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = ElasticLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Re-initialize weights and apply final processing
        self.post_init()
        self.retained_indices = None
        
    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            print(f"Encoder layer {layer} prune heads: {heads}.")
            self.layers[layer].self_attn.prune_heads(heads)
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_teacher: bool = False,
        head_z: Optional[torch.Tensor] = None,
        head_layer_z: Optional[torch.Tensor] = None,
        intermediate_z: Optional[torch.Tensor] = None,
        mlp_z: Optional[torch.Tensor] = None,
        hidden_z: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        # TODO: support mask shape conversion, and retained_indices use
        
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds
        
        if self.retained_indices is not None and not use_teacher:
            hidden_states = hidden_states.index_select(-1, self.retained_indices)

        if hidden_z is not None:
            hidden_states = hidden_states.mul(hidden_z)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    use_teacher=use_teacher,
                    head_z=head_z[idx] if head_z is not None else None,
                    head_layer_z=head_layer_z[idx] if head_layer_z is not None else None,
                    intermediate_z=intermediate_z[idx] if intermediate_z is not None else None,
                    mlp_z=mlp_z[idx] if mlp_z is not None else None,
                    hidden_z=hidden_z,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states, hidden_z, use_teacher=use_teacher)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

        
class ElasticLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.self_attn = ElasticLlamaAttention(config)
        self.mlp = ElasticLlamaMLP(config)
        self.input_layernorm = ElasticLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ElasticLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        use_teacher: bool = False,
        head_z: Optional[torch.Tensor] = None,
        head_layer_z: Optional[torch.Tensor] = None,
        intermediate_z: Optional[torch.Tensor] = None,
        mlp_z: Optional[torch.Tensor] = None,
        hidden_z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states, hidden_z, use_teacher=use_teacher)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            use_teacher=use_teacher,
            head_z=head_z,
            head_layer_z=head_layer_z,
        )
        if hidden_states is None:
            hidden_states = residual
        else:
            hidden_states = residual + hidden_states
        if hidden_z is not None:
            hidden_states = hidden_states.mul(hidden_z)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states, hidden_z, use_teacher=use_teacher)
        hidden_states = self.mlp(
            hidden_states,
            use_teacher=use_teacher,
            intermediate_z=intermediate_z,
            mlp_z=mlp_z,
        )
        if hidden_states is None:
            hidden_states = residual
        else:
            hidden_states = residual + hidden_states
        if hidden_z is not None:
            hidden_states = hidden_states.mul(hidden_z)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
        
        
class ElasticLlamaAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        
        self.block_retained_indices = None
        self.hidden_retained_indices = None
        self.num_teacher_heads = self.num_heads
        self.num_teacher_key_value_heads = self.num_key_value_heads
        self.teacher_hidden_size = self.hidden_size
        self.pruned_heads = set()
        self.teacher_pruned_heads = set()
        self.head_size = config.hidden_size // config.num_attention_heads
        self.o_proj = lora.SelectLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
    def prune_heads(self, heads):
        len_heads = len(heads)
        if len_heads == 0:
            return

        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_heads, self.head_size, self.pruned_heads
        )

        # Prune linear layers
        if len(index) == 0:
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.o_proj = None
        else:
            self.q_proj = prune_layer(self.q_proj, index)
            self.k_proj = prune_layer(self.k_proj, index)
            self.v_proj = prune_layer(self.v_proj, index)
            self.o_proj = prune_layer(self.o_proj, index, dim=1)

        # Update hyper params and store pruned heads
        self.num_heads = self.num_heads - len(heads)
        self.num_teacher_heads = self.num_heads
        self.num_key_value_heads = self.num_heads
        self.num_teacher_key_value_heads = self.num_key_value_heads
        self.hidden_size = self.num_heads * self.head_size
        self.teacher_hidden_size = self.hidden_size
        self.pruned_heads = self.pruned_heads.union(heads)
        self.teacher_pruned_heads = self.pruned_heads
        
    def project(self, hidden_states, proj_layer: nn.Linear, use_teacher=False):
        if isinstance(proj_layer, lora.PruningLinear):
            hidden_states = proj_layer(hidden_states, use_teacher=use_teacher, in_retained_indices=self.hidden_retained_indices, out_retained_indices=self.block_retained_indices)
        elif use_teacher:
            hidden_states = proj_layer(hidden_states)
        else:
            selected_weight, selected_bias = select_wandb(proj_layer.weight, proj_layer.bias, in_retained_indices=self.hidden_retained_indices, out_retained_indices=self.block_retained_indices)
            hidden_states = F.linear(hidden_states, selected_weight, selected_bias)
        return hidden_states
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        use_teacher:bool = None,
        head_z: Optional[torch.Tensor] = None,
        head_layer_z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        num_heads_use = self.num_teacher_heads if use_teacher else self.num_heads
        num_key_value_heads_use = self.num_teacher_key_value_heads if use_teacher else self.num_key_value_heads
        hidden_size_use = self.teacher_hidden_size if use_teacher else self.hidden_size

        if self.v_proj is None or num_heads_use == 0:
            return None, None, None

        if self.config.pretraining_tp > 1:
            key_value_slicing = (num_key_value_heads_use * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (num_heads_use * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.project(hidden_states, self.q_proj, use_teacher=use_teacher)
            key_states = self.project(hidden_states, self.k_proj, use_teacher=use_teacher)
            value_states = self.project(hidden_states, self.v_proj, use_teacher=use_teacher)

        query_states = query_states.view(bsz, q_len, num_heads_use, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads_use, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads_use, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, num_heads_use, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, num_heads_use, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states) # shape (bsz, num_heads, q_len, head_dim)
        
        if head_z is not None:
            head_z = head_z.view(-1, self.num_heads).unsqueeze(-1).unsqueeze(-1) # (1, num_heads, 1, 1)
            attn_output = attn_output * head_z

        if attn_output.size() != (bsz, num_heads_use, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, num_heads_use, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, hidden_size_use)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(hidden_size_use // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(hidden_size_use // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            if isinstance(self.o_proj, lora.SelectLinear):
                attn_output = self.o_proj(attn_output, use_teacher=use_teacher, in_retained_indices=self.block_retained_indices, out_retained_indices=self.hidden_retained_indices)
            else:
                raise ValueError("Output projection layer must be a SelectLinear layer")

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class ElasticLlamaMLP(LlamaMLP):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.block_retained_indices = None
        self.hidden_retained_indices = None
        self.down_proj = lora.SelectLinear(config.intermediate_size, config.hidden_size, bias=False)
        
    def forward(self, x, use_teacher=False, intermediate_z=None, mlp_z=None):
        if self.up_proj is None or (self.block_retained_indices is not None and self.block_retained_indices.numel() == 0 and not use_teacher):
            return None
        
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj)
            if intermediate_z is not None:
                intermediate_states = intermediate_states.mul(intermediate_z)
            intermediate_states = intermediate_states.split(slice, dim=2)
            
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            if isinstance(self.gate_proj, lora.PruningLinear):
                gated_x = self.gate_proj(x, use_teacher=use_teacher, in_retained_indices=self.hidden_retained_indices, out_retained_indices=self.block_retained_indices)
            elif use_teacher:
                gated_x = self.gate_proj(x)
            else:
                selected_weight, selected_bias = select_wandb(self.gate_proj.weight, self.gate_proj.bias, in_retained_indices=self.hidden_retained_indices, out_retained_indices=self.block_retained_indices)
                gated_x = F.linear(x, selected_weight, selected_bias)
            gated_x = self.act_fn(gated_x)
            
            if isinstance(self.up_proj, lora.PruningLinear):
                upped_x = self.up_proj(x, use_teacher=use_teacher, in_retained_indices=self.hidden_retained_indices, out_retained_indices=self.block_retained_indices)
            elif use_teacher:
                upped_x = self.up_proj(x)
            else:
                selected_weight, selected_bias = select_wandb(self.up_proj.weight, self.up_proj.bias, in_retained_indices=self.hidden_retained_indices, out_retained_indices=self.block_retained_indices)
                upped_x = F.linear(x, selected_weight, selected_bias)
            
            upped_x = gated_x * upped_x
            if intermediate_z is not None:
                upped_x = upped_x.mul(intermediate_z)
                
            if isinstance(self.down_proj, lora.SelectLinear):
                down_proj = self.down_proj(upped_x, use_teacher=use_teacher, in_retained_indices=self.block_retained_indices, out_retained_indices=self.hidden_retained_indices)
            else:
                raise ValueError("Down projection layer must be a SelectLinear layer")

        return down_proj
        
@dataclass
class ElasticCausalLMOutputWithPast(CausalLMOutputWithPast):
    pass