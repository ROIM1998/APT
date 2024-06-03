from transformers import PreTrainedModel
import torch
import logging
import sys
import loralib as lora
import torch.nn as nn

from typing import List, Dict, Union, Optional
from trainer.param_control import NAME2ATTR
from trainer.model_arch import get_encoder, get_decoder
from utils.fisher_utils.efficiency.param import *
from utils.minus_utils import count_params
from .fisher import collect_additive_mask_grads, collect_grads_by_suffix
from .scorer import GradientScorer, PredictivityScorer, RunningMaskSalienceScorer
from .search import search_mac, search_mac_topdown, search_encoder_decoder_mac
from .rearrange import rearrange_mask, layer_wise_rearrange_mask, better_rearrange_mask, global_rearrange
from .rescale import rescale_mask

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

MASK_TO_SHAPE = {
    'intermediate_mask': lambda x: [x.config.num_hidden_layers, x.config.intermediate_size],
    'head_mask': lambda x: [x.config.num_hidden_layers, x.config.num_attention_heads],
    'hidden_mask': lambda x: [x.config.hidden_size],
}

T5_MASK_TO_SHAPE = {
    'intermediate_mask': lambda x: [2, x.config.num_layers, x.config.d_ff],
    'head_mask': lambda x: [3, x.config.num_layers, x.config.num_heads],
    'hidden_mask': lambda x: [x.config.d_model],
}

def prune_model_with_masks(model: PreTrainedModel, masks: Dict[str, torch.Tensor]):
    head_mask, intermediate_mask = masks.get('head_mask', None), masks.get('intermediate_mask', None)
    # Head mask shapes: [num_layers, num_heads] (encoder or decoder-only architecture) or [3, num_layers, num_heads] (encoder-decoder architecture)
    # Intermediate mask shapes: [num_layers, intermediate_size] (encoder or decoder-only architecture) or [2, num_layers, intermediate_size] (encoder-decoder architecture)
    
    encoder, decoder = get_encoder(model), get_decoder(model)
    if head_mask is not None:
        encoder_pruned_heads, decoder_pruned_heads = {}, {}
        cross_attn_pruned_heads = {}
        if head_mask.shape[0] == 3:
            for layer in range(head_mask.shape[1]):
                encoder_pruned_heads[layer] = torch.where(head_mask[0, layer] == 0)[0].tolist()
                decoder_pruned_heads[layer] = torch.where(head_mask[1, layer] == 0)[0].tolist()
                cross_attn_pruned_heads[layer] = torch.where(head_mask[2, layer] == 0)[0].tolist()
                print(f"Encoder layer {layer} prune heads: {encoder_pruned_heads[layer]}. Decoder layer {layer} prune heads: {decoder_pruned_heads[layer]}. Cross-attention layer {layer} prune heads: {cross_attn_pruned_heads[layer]}.")
            model.prune_heads(encoder_pruned_heads, decoder_pruned_heads, cross_attn_pruned_heads)
        else:
            # Either encoder-only or decoder-only architecture
            for layer in range(head_mask.shape[0]):
                if encoder is not None and decoder is not None:
                    raise ValueError("Head mask shape is not compatible with encoder-decoder architecture.")
                if encoder is None and decoder is None:
                    raise ValueError("Either encoder or decoder should be provided.")
                if encoder is not None:
                    encoder_pruned_heads[layer] = torch.where(head_mask[layer] == 0)[0].tolist()
                    print(f"Encoder layer {layer} prune heads: {encoder_pruned_heads[layer]}.")
                else:
                    decoder_pruned_heads[layer] = torch.where(head_mask[layer] == 0)[0].tolist()
                    print(f"Decoder layer {layer} prune heads: {decoder_pruned_heads[layer]}.")
            model.prune_heads(encoder_pruned_heads if encoder is not None else decoder_pruned_heads)
        
    encoder_kept_intermediate_dims, decoder_kept_intermediate_dims = {}, {}
    if 'intermediate_mask' in masks:
        if intermediate_mask.ndim == 3:
            assert intermediate_mask.shape[0] == 2
            for layer in range(intermediate_mask.shape[1]):
                encoder_kept_intermediate_dims[layer] = torch.where(intermediate_mask[0, layer] == 1)[0].tolist()
                decoder_kept_intermediate_dims[layer] = torch.where(intermediate_mask[1, layer] == 1)[0].tolist()
        else:
            assert intermediate_mask.ndim == 2
            for layer in range(intermediate_mask.shape[0]):
                if encoder is not None and decoder is not None:
                    raise ValueError("Intermediate mask shape is not compatible with encoder-decoder architecture.")
                if encoder is None and decoder is None:
                    raise ValueError("Either encoder or decoder should be provided.")
                if encoder is not None:
                    encoder_kept_intermediate_dims[layer] = torch.where(intermediate_mask[layer] == 1)[0].tolist()
                else:
                    decoder_kept_intermediate_dims[layer] = torch.where(intermediate_mask[layer] == 1)[0].tolist()
        if encoder_kept_intermediate_dims and decoder_kept_intermediate_dims:
            model.resize_intermediate(encoder_kept_intermediate_dims, decoder_kept_intermediate_dims)
        else:
            model.resize_intermediate(encoder_kept_intermediate_dims if encoder is not None else decoder_kept_intermediate_dims)

    model.print_model_shape()

class BasePruner:
    def __init__(self, model: PreTrainedModel, mask_required: List[str]):
        self.model = model
        self.mask_required = mask_required
        self.mask_to_shape = T5_MASK_TO_SHAPE if model.base_model_prefix == 'transformer' else MASK_TO_SHAPE
        self.mask_shapes = {
            key: self.mask_to_shape[key](model)
            for key in mask_required
        }
    
    def generate_mask(self):
        raise NotImplementedError
    
class RandomPruner(BasePruner):
    def __init__(self, model: PreTrainedModel, mask_required: List[str]):
        super().__init__(self, model, mask_required)

        
    def generate_mask(self):
        return self.random_mask()
    
    def random_mask(self, mask_possibility: Union[Dict[str, float], None]=None, mask_ratio: Union[Dict[str, float], None]=None) -> Union[None, Dict[str, torch.Tensor]]:
        mask_dict = {}
        if mask_ratio is None:
            if mask_possibility is None:
                print("At least one of mask_possibility and mask_ratio should be provided.")
                return None
            else:
                construct_func = lambda x: torch.bernoulli(
                    torch.ones(self.mask_shapes[x]) * mask_possibility[x]
                )
        else:
            if mask_possibility is not None:
                print("Both mask_possibility and mask_ratio are provided. Please only provide one of them.")
            else:
                lengths = {
                    key: [int(torch.prod(torch.Tensor(self.mask_shapes[key])).item()), None, None]
                    for key in self.mask_required
                }
                for key in lengths:
                    lengths[key][1] = int(lengths[key][0] * mask_ratio[key])
                    lengths[key][2] = lengths[key][0] - lengths[key][1]
                construct_func = lambda x: torch.cat(
                    [
                        torch.zeros(lengths[x][1]),
                        torch.ones(lengths[x][2]),
                    ], 
                )[torch.randperm(lengths[x][0])].view(self.mask_shapes[x])
        for mask in self.mask_required:
            mask_dict[mask] = construct_func(mask)
        return mask_dict
        
class GreedyPruner(BasePruner):
    def __init__(self, model: PreTrainedModel, mask_required: List[str], scorer_dict: Dict[str, Union[GradientScorer, PredictivityScorer]]):
        super().__init__(model, mask_required)
        self.scorer_dict = scorer_dict
        
    def greedy_mask(self, score: torch.Tensor, flop_constraint: float):
        # TODO: figure out why smaller divergence is better
        _, indices = score.view(-1).sort(descending=True)
        mask = torch.ones_like(indices)
        mask[indices[:int((1 - flop_constraint) * len(indices))]] = 0
        mask = mask.view(score.shape)
        return mask
        
        
    def generate_mask(self, flop_constraint: float=0.6):
        masks = {}
        for k in self.mask_required:
            if k == 'head_mask':
                score = self.scorer_dict['head_mask'].head_score() if self.scorer_dict['head_mask'] is not None else None
            elif k == 'intermediate_mask':
                score = self.scorer_dict['intermediate_mask'].intermediate_score() if self.scorer_dict['intermediate_mask'] is not None else None
            if score is not None:
                masks[k] = self.greedy_mask(score, flop_constraint)
            else:
                masks[k] = torch.ones(self.mask_shapes[k])
        return masks
    
class FisherPruner(BasePruner):
    def __init__(self, model: PreTrainedModel, mask_required: List[str], scorer_dict: Dict[str, Union[GradientScorer, PredictivityScorer]], seq_len: int, cls_task: bool, do_rescale: bool = False): 
        super().__init__(model, mask_required)
        self.scorer_dict = scorer_dict
        self.seq_len = seq_len
        self.cls_task = cls_task
        self.do_rescale = do_rescale
        self.dataloader = None
        for v in scorer_dict.values():
            if v.dataloader is not None:
                self.dataloader = v.dataloader
                break
        
    def generate_mask(self, flop_constraint: float=0.6):
        head_grads, intermediate_grads = self.scorer_dict['head_mask'].head_score() if self.scorer_dict['head_mask'] is not None else None, self.scorer_dict['intermediate_mask'].intermediate_score() if self.scorer_dict['intermediate_mask'] is not None else None
        self.model.head_mask, self.model.intermediate_mask = search_mac(
            self.model.config,
            head_grads,
            intermediate_grads,
            self.seq_len,
            flop_constraint,
        )
        self.model.head_mask = rearrange_mask(self.model.head_mask, head_grads, self.scorer_dict['head_mask'].head_grads)
        self.model.intermediate_mask = rearrange_mask(self.model.intermediate_mask, intermediate_grads, self.scorer_dict['intermediate_mask'].intermediate_grads)
        if self.do_rescale:
            rescaled_head_mask, rescaled_intermediate_mask = rescale_mask(
                self.model,
                self.model.config,
                torch.ones(self.model.head_mask.shape).to(self.model.device),
                torch.ones(self.model.intermediate_mask.shape).to(self.model.device),
                self.model.head_mask.clone(),
                self.model.intermediate_mask.clone(),
                self.dataloader,
                self.cls_task,
            )
            return {
                'head_mask': rescaled_head_mask,
                'intermediate_mask': rescaled_intermediate_mask,
            }
        else:
            return {
                'head_mask': self.model.head_mask,
                'intermediate_mask': self.model.intermediate_mask,
            }
            
class BetterFisherPruner(BasePruner):
    def __init__(self, model: PreTrainedModel, mask_required: List[str], scorer_dict: Dict[str, Union[GradientScorer, PredictivityScorer]], seq_len: int, cls_task: bool, applied_procedure: List[str], output_seq_len=None, gated=False): 
        super().__init__(model, mask_required)
        self.scorer_dict = scorer_dict
        self.seq_len = seq_len
        self.output_seq_len = output_seq_len
        self.cls_task = cls_task
        self.dataloader = None
        for v in scorer_dict.values():
            if v.dataloader is not None:
                self.dataloader = v.dataloader
                break
        assert applied_procedure[0] == 'search' or applied_procedure[0] == 'topdown_search'
        applied_procedure.reverse()
        self.applied_procedure = applied_procedure
        self.gated = gated
        
    def generate_mask(self, flop_constraint: float=0.6):
        head_self_scores, intermediate_self_scores = self.scorer_dict['head_mask'].head_score() if self.scorer_dict['head_mask'] is not None else None, self.scorer_dict['intermediate_mask'].intermediate_score() if self.scorer_dict['intermediate_mask'] is not None else None
        if 'hidden_mask' in self.scorer_dict:
            hidden_self_scores = self.scorer_dict['hidden_mask'].hidden_score()
        else:
            hidden_self_scores = None
        mask_is_tensor = isinstance(self.model.head_mask, torch.Tensor)
        while self.applied_procedure:
            action = self.applied_procedure.pop()
            if action == 'search':
                if self.model.base_model_prefix == 'transformer':
                    self.model.head_mask, self.model.intermediate_mask = search_encoder_decoder_mac(
                        self.model.config,
                        head_self_scores,
                        intermediate_self_scores,
                        self.seq_len,
                        2 if self.cls_task else self.seq_len if self.output_seq_len is None else self.output_seq_len,
                        flop_constraint,
                        gated=self.gated,
                    )
                else:
                    self.model.head_mask, self.model.intermediate_mask = search_mac(
                        self.model.config,
                        head_self_scores,
                        intermediate_self_scores,
                        self.seq_len,
                        flop_constraint,
                        gated=self.gated,
                    )
            elif action == 'topdown_search':
                if self.model.base_model_prefix == 'transformer':
                    raise NotImplementedError("Topdown search is not implemented for encoder-decoder models")
                else:
                    self.model.head_mask, self.model.intermediate_mask, self.model.hidden_mask = search_mac_topdown(
                        self.model.config,
                        head_self_scores,
                        intermediate_self_scores,
                        hidden_self_scores / 8 if hidden_self_scores is not None else None,
                        self.seq_len,
                        flop_constraint,
                    )
            elif action == 'rearrange':
                self.model.head_mask = rearrange_mask(self.model.head_mask, head_self_scores, self.scorer_dict['head_mask'].head_grads)
                self.model.intermediate_mask = rearrange_mask(self.model.intermediate_mask, intermediate_self_scores, self.scorer_dict['intermediate_mask'].intermediate_grads)
            elif action == 'better_rearrange':
                self.model.head_mask = better_rearrange_mask(self.model.head_mask, self.scorer_dict['head_mask'].head_grads)
                self.model.intermediate_mask = better_rearrange_mask(self.model.intermediate_mask, self.scorer_dict['intermediate_mask'].intermediate_grads)
            elif action == 'layerwise_rearrange':
                self.model.head_mask, self.model.intermediate_mask = layer_wise_rearrange_mask(
                    self.model.head_mask,
                    self.model.intermediate_mask,
                    self.scorer_dict['head_mask'].head_grads,
                    self.scorer_dict['intermediate_mask'].intermediate_grads,
                )
            elif action == 'global':
                head_ndim = self.model.head_mask.ndim if mask_is_tensor else 2 if isinstance(self.model.head_mask[0], torch.Tensor) else 3
                intermediate_ndim = self.model.intermediate_mask.ndim if mask_is_tensor else 2 if isinstance(self.model.intermediate_mask[0], torch.Tensor) else 3
                if head_ndim == 2:
                    self.model.head_mask = global_rearrange(
                        self.model.head_mask,
                        self.scorer_dict['head_mask'].head_grads,
                    )
                elif head_ndim == 3:
                    for category in range(self.model.head_mask.shape[0]):
                        self.model.head_mask[category] = global_rearrange(
                            self.model.head_mask[category],
                            self.scorer_dict['head_mask'].head_grads[:, category, :, :],
                        )
                if intermediate_ndim == 2:
                    self.model.intermediate_mask = global_rearrange(
                        self.model.intermediate_mask,
                        self.scorer_dict['intermediate_mask'].intermediate_grads,
                    )
                elif intermediate_ndim == 3:
                    for category in range(self.model.intermediate_mask.shape[0]):
                        self.model.intermediate_mask[category] = global_rearrange(
                            self.model.intermediate_mask[category],
                            self.scorer_dict['intermediate_mask'].intermediate_grads[:, category, :, :],
                        )
            elif action == 'rescale':
                self.model.head_mask, self.model.intermediate_mask = rescale_mask(
                    self.model,
                    self.model.config,
                    torch.ones(self.model.head_mask.shape).to(self.model.device),
                    torch.ones(self.model.intermediate_mask.shape).to(self.model.device),
                    self.model.head_mask.clone(),
                    self.model.intermediate_mask.clone(),
                    self.dataloader,
                    self.cls_task,
                )
            else:
                print("Command %s not implemented" % action)
                raise NotImplementedError
        return {
            'head_mask': self.model.head_mask,
            'intermediate_mask': self.model.intermediate_mask,
            'hidden_mask': self.model.hidden_mask if hasattr(self.model, 'hidden_mask') else None,
        }
            
            
class LayerWisePruner(BasePruner):
    def __init__(self, model: PreTrainedModel, mask_required: List[str], scorer_dict: Dict[str, Union[GradientScorer, PredictivityScorer]], seq_len: int, cls_task: bool, do_rescale: bool = False): 
        super().__init__(model, mask_required)
        self.scorer_dict = scorer_dict
        self.seq_len = seq_len
        self.cls_task = cls_task
        self.do_rescale = do_rescale
        self.dataloader = None
        for v in scorer_dict.values():
            if v.dataloader is not None:
                self.dataloader = v.dataloader
                break
        
    def generate_mask(self, flop_constraint: float=0.6):
        head_grads, intermediate_grads = self.scorer_dict['head_mask'].head_score() if self.scorer_dict['head_mask'] is not None else None, self.scorer_dict['intermediate_mask'].intermediate_score() if self.scorer_dict['intermediate_mask'] is not None else None
        self.model.head_mask, self.model.intermediate_mask = search_mac(
            self.model.config,
            head_grads,
            intermediate_grads,
            self.seq_len,
            flop_constraint,
        )
        self.model.head_mask = rearrange_mask(self.model.head_mask, head_grads, self.scorer_dict['head_mask'].head_grads)
        self.model.intermediate_mask = rearrange_mask(self.model.intermediate_mask, intermediate_grads, self.scorer_dict['intermediate_mask'].intermediate_grads)
        self.model.head_mask, self.model.intermediate_mask = layer_wise_rearrange_mask(
            self.model.head_mask,
            self.model.intermediate_mask,
            self.scorer_dict['head_mask'].head_grads,
            self.scorer_dict['intermediate_mask'].intermediate_grads,
        )
        return {
            'head_mask': self.model.head_mask,
            'intermediate_mask': self.model.intermediate_mask,
        }

        
class FixedPruner(BasePruner):
    def __init__(self, model: PreTrainedModel, mask_required: List[str], head_mask_path: Union[None, str] = None, intermediate_mask_path: Union[None, str] = None, hidden_mask_path: Union[None, str] = None):
        super().__init__(model, mask_required)
        self.head_mask_path = head_mask_path
        self.intermediate_mask_path = intermediate_mask_path
        self.hidden_mask_path = hidden_mask_path
        
    def generate_mask(self, mac_constraint=None):
        return {
            'head_mask': torch.load(self.head_mask_path) if self.head_mask_path is not None else torch.ones(self.mask_shapes['head_mask']),
            'intermediate_mask': torch.load(self.intermediate_mask_path) if self.intermediate_mask_path is not None else torch.ones(self.mask_shapes['intermediate_mask']),
            'hidden_mask': torch.load(self.hidden_mask_path) if self.hidden_mask_path is not None else torch.ones(self.mask_shapes['hidden_mask']),
        }
        
class DensityPruner:
    def __init__(self, model, args, scorer: RunningMaskSalienceScorer):
        self.model = model
        self.args = args
        if args is not None:
            fileHandler = logging.FileHandler("{0}/{1}.log".format(args.output_dir, 'trainer'))
            fileHandler.setFormatter(logFormatter)
            logger.addHandler(fileHandler)
        self.scorer = scorer
        if 't5' in model.config.model_type:
            self.attention_head_size = model.config.d_kv
            self.t5_backbone = True
        elif 'bert' in model.config.model_type or 'llama' in model.config.model_type:
            self.attention_head_size = model.config.hidden_size // model.config.num_attention_heads
            self.t5_backbone = False
        else:
            raise NotImplementedError("Model type %s not supported" % model.config.model_type)
        self.intermediate_size = model.config.intermediate_size if hasattr(model.config, 'intermediate_size') else model.config.d_ff
        self.model_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else model.config.num_layers
        self.gated_ffn = model.config.model_type == 'llama' or 'gated' in getattr(model.config, 'feed_forward_proj', '')
        if 'bert' in model.config.model_type: # BERT and RoBERTa
            self.param_per_block = {
                'head_mask': param_per_head(self.model.hidden_mask.shape[0] if getattr(self.model, "hidden_mask", None) is not None else model.config.hidden_size, self.attention_head_size),
                'intermediate_mask': param_per_neuron(self.model.hidden_mask.shape[0] if getattr(self.model, "hidden_mask", None) is not None else model.config.hidden_size),
                'hidden_mask': param_per_hidden_dim(model.head_mask.numel() * self.attention_head_size, model.intermediate_mask.numel(), self.model_layers)
            }
        elif model.config.model_type == 'llama': # LLaMA 1 and 2
            self.param_per_block = {
                'head_mask': param_per_head(self.model.hidden_mask.shape[0] if getattr(self.model, "hidden_mask", None) is not None else model.config.hidden_size, self.attention_head_size),
                'intermediate_mask': param_per_neuron(self.model.hidden_mask.shape[0] if getattr(self.model, "hidden_mask", None) is not None else model.config.hidden_size, gated=True),
                'hidden_mask': param_per_hidden_dim(model.head_mask.numel() * self.attention_head_size, model.intermediate_mask.numel(), self.model_layers, ffn_gated=True)
            }
        else: # T5
            num_heads = self.model.head_mask.numel()
            num_neurons = self.model.intermediate_mask.numel()
            num_hidden_dim = self.model.hidden_mask.numel()
            self.param_per_block = {
                'head_mask': param_per_t5_head(num_hidden_dim if getattr(self.model, "hidden_mask", None) is not None else model.config.hidden_size, self.attention_head_size),
                'intermediate_mask': param_per_t5_neuron(num_hidden_dim if getattr(self.model, "hidden_mask", None) is not None else model.config.hidden_size, gated=self.gated_ffn),
                'hidden_mask': param_per_t5_hidden_dim(num_heads * self.attention_head_size, num_neurons, ffn_gated=self.gated_ffn, num_hidden_layers=self.model_layers)
            }
        logger.info("Param per block: " + str(self.param_per_block))
        self.n_params, self.n_param_vars = count_params(self.model, mode='main')
        
    def set_param_per_block(self):
        if self.model.virtual_pruned:
            head_mask, intermediate_mask, hidden_mask = self.model.backup_head_mask, self.model.backup_intermediate_mask, self.model.backup_hidden_mask
        else:
            head_mask, intermediate_mask, hidden_mask = self.model.head_mask, self.model.intermediate_mask, self.model.hidden_mask
        hidden_size = (hidden_mask > 0).sum().item() if isinstance(hidden_mask, torch.Tensor) else self.model.config.hidden_size if hasattr(self.model.config, 'hidden_size') else self.model.config.d_model
        num_heads = (head_mask > 0).sum().item() if isinstance(head_mask, torch.Tensor) else sum([(v > 0).sum().item() for v in head_mask])
        num_neurons = (intermediate_mask > 0).sum().item() if isinstance(intermediate_mask, torch.Tensor) else sum([(v > 0).sum().item() for v in intermediate_mask])
        if self.t5_backbone:
            self.param_per_block = {
                'head_mask': param_per_t5_head(hidden_size, self.attention_head_size),
                'intermediate_mask': param_per_t5_neuron(hidden_size, gated=self.gated_ffn),
                'hidden_mask': param_per_t5_hidden_dim(num_heads * self.attention_head_size, num_neurons, ffn_gated=self.gated_ffn, num_hidden_layers=self.model_layers)
            }
        else:
            self.param_per_block = {
                'head_mask': param_per_head(hidden_size, self.attention_head_size),
                'intermediate_mask': param_per_neuron(hidden_size, gated=self.gated_ffn),
                'hidden_mask': param_per_hidden_dim(num_heads * self.attention_head_size, num_neurons, self.model_layers, ffn_gated=self.gated_ffn)
            }
        
    def get_mask_score_density(self, k: str):
        return self.scorer.get_score(k) / self.param_per_block[k]
    
    def get_tuning_mask_and_score_density(self, src_layer, i: int, k: str, m: str):
        current_m: nn.Parameter = getattr(src_layer, m, None)
        score = self.scorer.salience_dict['modules'][i][k][m]['s']
        score /= current_m.hidden_size
        return current_m, score

class DensityBSMaskPruner(DensityPruner):
    def __init__(self, model, args, scorer: RunningMaskSalienceScorer):
        super().__init__(model, args, scorer)
        self.mask_updated_once = False
            
    def update_mask(self, top_k: float=0.4, mask_lr: float = 0.01, is_last: bool = False, unstable: bool = False):
        if top_k >= 1:
            return
        if unstable:
            if self.mask_updated_once:
                return
            else:
                is_last = True # when using unstable pruning, only update mask once
                self.mask_updated_once = True
        with torch.no_grad():
            all_masks = []
            all_scores = []
            # sort and update mask based on information density -> salience / num_parameters for each block
            for k in ['head_mask', 'intermediate_mask', 'hidden_mask']:
                score = self.get_mask_score_density(k)
                mask = getattr(self.model, k).detach()
                all_scores.append(score)
                all_masks.append(mask)
            if any([not isinstance(v, torch.Tensor) for v in all_scores]):
                print("Warning: some scores are not tensors. Skipping mask update.", flush=True)
                return
            num_heads, num_neurons = all_masks[0].shape[0], all_masks[1].shape[0]            
            _, sorted_idx = torch.sort(torch.cat(all_scores), descending=True)
            all_block_type_reshaped = ((num_heads <= sorted_idx) & (sorted_idx < (num_heads + num_neurons))).float() + ((num_heads + num_neurons) <= sorted_idx).float() * 2
            
            target_param_size = top_k * self.n_params
            # Using dynamic binary search method to accumulate pruned parameter size
            i = all_block_type_reshaped.shape[0] // 2
            low_i, high_i = 0, all_block_type_reshaped.shape[0]
            while True:
                current_head_num = all_block_type_reshaped[:i].eq(0).sum().item()
                current_intermediate_num = all_block_type_reshaped[:i].eq(1).sum().item()
                current_hidden_dim = all_block_type_reshaped[:i].eq(2).sum().item()
                current_param_size = compute_param(current_head_num, current_intermediate_num, current_hidden_dim, self.attention_head_size, self.model_layers, is_t5=self.t5_backbone, ffn_gated=self.gated_ffn)
                if current_param_size < target_param_size:
                    low_i = i
                    i = (i + high_i) // 2
                else:
                    high_i = i
                    i = (i + low_i) // 2
                if high_i - low_i <= 1:
                    break
            
            if current_param_size >= target_param_size:
                i -= 1
                current_head_num = all_block_type_reshaped[:i].eq(0).sum().item()
                current_intermediate_num = all_block_type_reshaped[:i].eq(1).sum().item()
                current_hidden_dim = all_block_type_reshaped[:i].eq(2).sum().item()
                current_param_size = compute_param(current_head_num, current_intermediate_num, current_hidden_dim, self.attention_head_size, self.model_layers, is_t5=self.t5_backbone, ffn_gated=self.gated_ffn)
            
            top_k_idx = sorted_idx[:i]
            rest_idx = sorted_idx[i:]
            
            top_k_idx_head, rest_idx_head = top_k_idx[top_k_idx < num_heads], rest_idx[rest_idx < num_heads]
            top_k_idx_intermediate, rest_idx_intermediate = top_k_idx[(num_heads <= top_k_idx) & (top_k_idx < (num_heads + num_neurons))] - num_heads, rest_idx[(num_heads <= rest_idx) & (rest_idx < (num_heads + num_neurons))] - num_heads
            top_k_idx_hidden, rest_idx_hidden = top_k_idx[(num_heads + num_neurons) <= top_k_idx] - num_heads - num_neurons, rest_idx[(num_heads + num_neurons) <= rest_idx] - num_heads - num_neurons
            for indices, name in zip([(top_k_idx_head, rest_idx_head), (top_k_idx_intermediate, rest_idx_intermediate), (top_k_idx_hidden, rest_idx_hidden)], ['head_mask', 'intermediate_mask', 'hidden_mask']):
                mask = getattr(self.model, name)
                if is_last:
                    # Guarantee that after the last pruning step, the mask for to-be-pruned blocks is set to 0
                    logger.info("The target parameter size is %d, while the current parameter size is %d." % (target_param_size, current_param_size))
                    logger.info("Last pruning step. Setting the mask for to-be-pruned blocks to 0.")
                    mask[indices[1]] = 0
                    mask[indices[0]] = 1
                else:
                    mask[indices[1]] = torch.maximum(mask[indices[1]] - mask_lr, torch.tensor(0))
                    mask[indices[0]] = torch.minimum(mask[indices[0]] + mask_lr, torch.tensor(1))
            if is_last:
                print("All scores: %s" % {
                'head_mask': all_scores[0],
                'intermediate_mask': all_scores[1],
                'hidden_mask': all_scores[2],
                }, flush=True)
                print("Number of nans: %d" % torch.isnan(torch.cat(all_scores)).sum().item(), flush=True)
            if self.scorer is not None and self.scorer.param_controller.teacher_config is not None and hasattr(self.scorer, 'salience_dict'):
                self._update_mask_salience(top_k=self.args.grafting_top_k, mask_lr=self.args.grafting_mask_lr, is_last=is_last)

    def _update_mask_salience(self, top_k: float = 0.5, target='teacher', mask_lr=0.01, beta_1=0.85, beta_2=0.85, is_last: bool = False, hidden_normalize: bool = True):
        if top_k <= 0:
            return
        assert target in ['teacher', 'student']
        all_scores = []
        all_masks = []
        all_types = []
        MASK2TYPE = {
            'input_mask': 0,
            'output_mask': 1,
            'bottleneck_mask': 2
        }
        N = 0
        with torch.no_grad():
            for i in self.scorer.salience_dict['modules']:
                for k in self.scorer.salience_dict['modules'][i]:
                    k_attr = NAME2ATTR[k][self.scorer.param_controller.model_arch.model_category]
                    parent_layer = self.scorer.param_controller.get_parent_layer(i, k)
                    src_layer = getattr(parent_layer, k_attr)
                    if not isinstance(src_layer, lora.PruningLinear) or src_layer.r == 0:
                        continue
                    N += 1
                    for m in ['input_mask', 'output_mask', 'bottleneck_mask']:
                        current_m, score = self.get_tuning_mask_and_score_density(src_layer, i, k, m)
                        all_scores.append(score)
                        all_masks.append(current_m)
                        all_types.append(torch.ones_like(score) * MASK2TYPE[m])
            
            if not len(all_types):
                print("No PruningLinear layer found. Skipping mask update.")
                return
            all_types = torch.cat(all_types)
            # Adjust mask based on top_k
            _, sorted_idx = torch.sort(torch.cat(all_scores), descending=True)
            all_types_sorted = all_types[sorted_idx]
            i = int(len(sorted_idx) * top_k)
            target_param_num = self.scorer.param_controller.next_tuning_param_num * top_k if self.scorer.param_controller.next_tuning_param_num is not None else self.scorer.param_controller.tuning_param_number * top_k
            high_i, low_i = len(sorted_idx), 0
            while high_i - low_i > 1:
                ins, outs, rs = (all_types_sorted[:i] == 0).sum(), (all_types_sorted[:i] == 1).sum(), (all_types_sorted[:i] == 2).sum()
                predict_num = (ins + outs) * rs / N
                if predict_num > target_param_num:
                    high_i = i
                    i = (i + low_i) // 2
                else:
                    low_i = i
                    i = (i + high_i) // 2

            in_top_k = torch.zeros_like(sorted_idx)
            top_k_idx = sorted_idx[:i]
            in_top_k[top_k_idx] = 1
            in_top_k = torch.split(in_top_k, [v.numel() for v in all_masks])
            
            for mask, ink in zip(all_masks, in_top_k):
                ink_i = ink.nonzero().squeeze()
                outk_i = (ink == 0).nonzero().squeeze()
                if is_last:
                    logger.info("Last pruning step. Setting the grafting mask for to-be-pruned blocks to 0.")
                    mask[outk_i] = 0
                    mask[ink_i] = 1
                else:
                    mask[ink_i] = torch.minimum(mask[ink_i] + mask_lr, torch.tensor(1))
                    mask[outk_i] = torch.maximum(mask[outk_i] - mask_lr, torch.tensor(0))


class RandomBSMaskPruner(DensityBSMaskPruner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # Override pruning mask score density function
    def get_mask_score_density(self, k: str):
        mask = getattr(self.model, k)
        return torch.rand_like(mask)
    
    # Override tuning mask score density function
    def get_tuning_mask_and_score_density(self, src_layer, i: int, k: str, m: str):
        current_m: nn.Parameter = getattr(src_layer, m, None)
        score = torch.rand_like(current_m)
        return current_m, score

class DensityUniformPruner(DensityPruner):
    def __init__(self, model, args, scorer: RunningMaskSalienceScorer):
        super().__init__(model, args, scorer)
    
    def _update_1dim_mask_by_score(self, score: torch.Tensor, mask: torch.Tensor, top_k: float, mask_lr: float, is_last: bool):
        assert isinstance(score, torch.Tensor) and score.ndim == 1
        sorted_idx = torch.sort(score, descending=True)[1]
        if is_last:
            mask[sorted_idx[int(len(sorted_idx) * top_k):]] = 0
            mask[sorted_idx[:int(len(sorted_idx) * top_k)]] = 1
        else:
            mask[sorted_idx[int(len(sorted_idx) * top_k):]] = torch.maximum(torch.tensor(0), mask[sorted_idx[int(len(sorted_idx) * top_k):]] - mask_lr)
            mask[sorted_idx[:int(len(sorted_idx) * top_k)]] = torch.minimum(torch.tensor(1), mask[sorted_idx[:int(len(sorted_idx) * top_k)]] + mask_lr)
        
    def update_mask(self, top_k: float=0.4, mask_lr: float = 0.01, is_last: bool = False):
        sqrt_top_k = top_k ** 0.5
        # Get the density score (actually no need, cuz it's uniform pruning within structures)
        head_score, intermediate_score, hidden_score = self.get_mask_score_density('head_mask'), self.get_mask_score_density('intermediate_mask'), self.get_mask_score_density('hidden_mask')
        head_mask, intermediate_mask = self.model.split_mask_or_score(self.model.head_mask, self.model.intermediate_mask)
        hidden_mask = self.model.hidden_mask # Directly get masks for editing them, instead of getting cloned masks
        
        # Pruning the MHA head and FFN neuron with the squared-root of the top_k * n_params
        head_score, intermediate_score = self.model.split_mask_or_score(head_score, intermediate_score)
        # Uniformly distributing the pruning rate among the MHA head and FFN neuron
        for current_score, current_mask in zip([head_score, intermediate_score], [head_mask, intermediate_mask]):
            for scores, masks in zip(current_score, current_mask):
                if isinstance(scores, list) or isinstance(scores, tuple): # T5-like encoder-decoder models
                    for score, mask in zip(scores, masks):
                        self._update_1dim_mask_by_score(score, mask, sqrt_top_k, mask_lr, is_last)
                else:
                    self._update_1dim_mask_by_score(scores, masks, sqrt_top_k, mask_lr, is_last)
        
        # Pruning the model hidden size also with the squared-root of the top_k * n_params
        self._update_1dim_mask_by_score(hidden_score, hidden_mask, sqrt_top_k, mask_lr, is_last)

class RuleMixSaliencePruner(DensityPruner):
    def __init__(self, model, args, scorer: RunningMaskSalienceScorer, mha_pruning_layer_start: int = 4, mha_pruning_layer_end: int = 30, ffn_pruning_layer_start: int = 4, ffn_pruning_layer_end: int = 30, mha_pruning_ratio: Optional[float] = None, d_model_pruning_ratio: float = 0.):
        super().__init__(model, args, scorer)
        target_mha_layers = list(range(mha_pruning_layer_start, mha_pruning_layer_end))
        target_ffn_layers = list(range(ffn_pruning_layer_start, ffn_pruning_layer_end))
        self.non_pruning_mha_layers = list(set(range(self.model_layers)) - set(target_mha_layers))
        self.non_pruning_ffn_layers = list(set(range(self.model_layers)) - set(target_ffn_layers))
        
        assert 0 <= d_model_pruning_ratio <= 1
        assert mha_pruning_ratio is None or 0 <= mha_pruning_ratio <= 1
        # Specifying the mha to be pruned, then deducing the ffn to be pruned
        self.mha_pruning_ratio = mha_pruning_ratio
        self.d_model_pruning_ratio = d_model_pruning_ratio
        if 'bert' in model.config.model_type: # BERT and RoBERTa
            self.mha_total_ratio = 4 * self.model.config.hidden_size * self.model.config.hidden_size / (4 * self.model.config.hidden_size * self.model.config.hidden_size + 2 * self.model.config.hidden_size * self.model.config.intermediate_size)
            self.ffn_total_ratio = 1 - self.mha_total_ratio
        elif model.config.model_type == 't5' and 'gate' not in model.config.feed_forward_proj: # T5, but not gated
            self.mha_total_ratio = 12 * self.config.d_model * self.config.d_model / (12 * self.config.d_model * self.config.d_model + 4 * self.config.d_model * self.config.d_ff)
            self.ffn_total_ratio = 1 - self.mha_total_ratio
        elif 't5' in model.config.model_type and 'gate' in model.config.feed_forward_proj: # T5, gated
            self.mha_total_ratio = 12 * self.config.d_model * self.config.d_model / (12 * self.config.d_model * self.config.d_model + 6 * self.config.d_model * self.config.d_ff)
            self.ffn_total_ratio = 1 - self.mha_total_ratio
        elif model.config.model_type == 'llama': # LLaMA 1 and 2
            self.mha_total_ratio = 4 * self.model.config.hidden_size * self.model.config.hidden_size / (4 * self.model.config.hidden_size * self.model.config.hidden_size + 3 * self.model.config.hidden_size * self.model.config.intermediate_size)
            self.ffn_total_ratio = 1 - self.mha_total_ratio
            
    def _update_1dim_mask_by_score(self, score: torch.Tensor, mask: torch.Tensor, top_k: float, mask_lr: float, is_last: bool):
        assert isinstance(score, torch.Tensor) and score.ndim == 1
        sorted_idx = torch.sort(score, descending=True)[1]
        if is_last:
            mask[sorted_idx[int(len(sorted_idx) * top_k):]] = 0
            mask[sorted_idx[:int(len(sorted_idx) * top_k)]] = 1
        else:
            mask[sorted_idx[int(len(sorted_idx) * top_k):]] = torch.maximum(torch.tensor(0), mask[sorted_idx[int(len(sorted_idx) * top_k):]] - mask_lr)
            mask[sorted_idx[:int(len(sorted_idx) * top_k)]] = torch.minimum(torch.tensor(1), mask[sorted_idx[:int(len(sorted_idx) * top_k)]] + mask_lr)
        
    def update_mask(self, top_k: float=0.4, mask_lr: float = 0.01, is_last: bool = False):
        d_model_top_k = (1 - self.d_model_pruning_ratio) # By default it is 1
        block_top_k = top_k / d_model_top_k
        if self.mha_pruning_ratio is None:
            mha_top_k = block_top_k
            ffn_top_k = block_top_k
        else:
            if self.mha_pruning_ratio > (1 - block_top_k):
                raise ValueError("The MHA pruning ratio %f is too large and not necessary. With top-k %f, d_model top-k set as %f, block top-k calculated as %f" % (self.mha_pruning_ratio, top_k, d_model_top_k, block_top_k))
            mha_top_k = self.mha_pruning_ratio
            ffn_top_k = 1 - (1 - block_top_k - (1 - mha_top_k) * self.mha_total_ratio) / self.ffn_total_ratio
        # Get the density score (actually no need, cuz it's uniform pruning within structures)
        head_score, intermediate_score, hidden_score = self.get_mask_score_density('head_mask'), self.get_mask_score_density('intermediate_mask'), self.get_mask_score_density('hidden_mask')
        head_mask, intermediate_mask = self.model.split_mask_or_score(self.model.head_mask, self.model.intermediate_mask)
        hidden_mask = self.model.hidden_mask # Directly get masks for editing them, instead of getting cloned masks
        
        # Pruning the MHA head and FFN neuron with the squared-root of the top_k * n_params
        splitted_head_score, splitted_intermediate_score = self.model.split_mask_or_score(head_score, intermediate_score)
        # Setting those non-pruning layers' scores to 1e10
        if isinstance(splitted_head_score[0], torch.Tensor): # Encoder-only or decoder-only models
            for layer_i, score in enumerate(splitted_head_score):
                if layer_i in self.non_pruning_mha_layers:
                    assert isinstance(score, torch.Tensor)
                    score[:] = 1e10
            for layer_i, score in enumerate(splitted_intermediate_score):
                if layer_i in self.non_pruning_ffn_layers:
                    assert isinstance(score, torch.Tensor)
                    score[:] = 1e10
        else: # T5-like encoder-decoder models
            for layer_i, score in enumerate(splitted_head_score[0]):
                if layer_i in self.non_pruning_mha_layers:
                    assert isinstance(score, torch.Tensor)
                    score[:] = 1e10
            for layer_i, score in enumerate(splitted_head_score[1]):
                if layer_i in self.non_pruning_mha_layers:
                    assert isinstance(score, torch.Tensor)
                    score[:] = 1e10
            for layer_i, score in enumerate(splitted_head_score[2]):
                if layer_i in self.non_pruning_mha_layers:
                    assert isinstance(score, torch.Tensor)
                    score[:] = 1e10
            for layer_i, score in enumerate(splitted_intermediate_score[0]):
                if layer_i in self.non_pruning_ffn_layers:
                    assert isinstance(score, torch.Tensor)
                    score[:] = 1e10
            for layer_i, score in enumerate(splitted_intermediate_score[1]):
                if layer_i in self.non_pruning_ffn_layers:
                    assert isinstance(score, torch.Tensor)
                    score[:] = 1e10
        
        # MHA pruning
        self._update_1dim_mask_by_score(head_score, self.model.head_mask, mha_top_k, mask_lr, is_last)
        
        # FFN pruning
        self._update_1dim_mask_by_score(intermediate_score, self.model.intermediate_mask, ffn_top_k, mask_lr, is_last)
        
        # Pruning the model hidden size also with the squared-root of the top_k * n_params
        if self.d_model_pruning_ratio < 1.:
            self._update_1dim_mask_by_score(hidden_score, hidden_mask, d_model_top_k, mask_lr, is_last)


class AdapterPruner:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        
    def prune_by_suffix(self, constraint, suffix, avoid_no_expand=True):
        assert 0 < constraint < 1
        all_grads = collect_grads_by_suffix(self.model, self.dataloader, suffix)
        all_scores = {k: v.pow(2).sum(dim=0) for k, v in all_grads.items() if k.endswith(suffix)}
        names, concat_scores = list(all_scores), torch.cat(list(all_scores.values()), dim=0)
        score_lens = [len(all_scores[k]) for k in names]
        sorted_scores, sorted_indices = concat_scores.sort(descending=True)
        expanded_num = 0
        while True:
            all_mask = torch.zeros_like(concat_scores)
            if constraint is None:
                all_mask = torch.ones_like(all_mask)
            else:
                all_mask[sorted_indices[:int(len(sorted_indices) * constraint)]] = 1
            all_mask = all_mask.split(score_lens)
            expanded_num = len([v for v in all_mask if v.all()])
            if not avoid_no_expand or expanded_num > 0:
                break
            else:
                constraint = min(1, constraint * 1.05)
                print('No expanded layer, try to expand more. Current constraint: %f' % constraint)
        return names, all_mask, all_grads

    def prune(self, out_dim_constraint, bottleneck_constraint):
        assert (out_dim_constraint is None or 0 < out_dim_constraint < 1) and (bottleneck_constraint is None or 0 < bottleneck_constraint < 1)
        all_grads = collect_additive_mask_grads(self.model, self.dataloader)
        all_output_scores, all_bottleneck_scores = {k: v.pow(2).sum(dim=0) for k, v in all_grads.items() if 'output_mask' in k}, {k: v.pow(2).sum(dim=0) for k, v in all_grads.items() if 'bottleneck_mask' in k}
        output_names, concat_output_scores = list(all_output_scores), torch.cat(list(all_output_scores.values()), dim=0)
        bottleneck_names, concat_bottleneck_scores = list(all_bottleneck_scores), torch.cat(list(all_bottleneck_scores.values()), dim=0)
        output_score_lens, bottleneck_score_lens = [len(all_output_scores[k]) for k in output_names], [len(all_bottleneck_scores[k]) for k in bottleneck_names]

        sorted_output_scores, sorted_output_indices = concat_output_scores.sort(descending=True)
        sorted_bottleneck_scores, sorted_bottleneck_indices = concat_bottleneck_scores.sort(descending=True)
        all_output_mask, all_bottleneck_mask = torch.zeros_like(concat_output_scores), torch.zeros_like(concat_bottleneck_scores)
        if out_dim_constraint is None:
            all_output_mask = torch.ones_like(all_output_mask)
        else:
            all_output_mask[sorted_output_indices[:int(len(sorted_output_indices) * out_dim_constraint)]] = 1
        if bottleneck_constraint is None:
            all_bottleneck_mask = torch.ones_like(all_bottleneck_mask)
        else:
            all_bottleneck_mask[sorted_bottleneck_indices[:int(len(sorted_bottleneck_indices) * bottleneck_constraint)]] = 1
        all_output_mask, all_bottleneck_mask = all_output_mask.split(output_score_lens), all_bottleneck_mask.split(bottleneck_score_lens)
        return output_names, bottleneck_names, all_output_mask, all_bottleneck_mask