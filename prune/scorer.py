import torch
import loralib as lora
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from transformers import PreTrainedModel
from transformers.trainer import TrainerState
from typing import Union, Optional, Dict, List, Tuple
from trainer.param_control import ParamController, NAME2ATTR
from trainer.model_arch import hijack_input, get_ffn2
from tqdm import tqdm
from scipy.stats import entropy
from utils.minus_utils import kurtosis
from .fisher import compute_fisher_info, compute_l1_fisher_info, collect_mask_grads, collect_hidden_mask_grads

ATTENTION_LAYER_KEYS = {
    'bert': [['query', 'key', 'value', 'attention.output']],
    't5': [['enc_self_query', 'enc_self_key', 'enc_self_value', 'enc_self_output'], ['dec_self_query', 'dec_self_key', 'dec_self_value', 'dec_self_output'], ['cross_query', 'cross_key', 'cross_value', 'cross_output']],
    't5_lm_adapt': [['enc_self_query', 'enc_self_key', 'enc_self_value', 'enc_self_output'], ['dec_self_query', 'dec_self_key', 'dec_self_value', 'dec_self_output'], ['cross_query', 'cross_key', 'cross_value', 'cross_output']],
}

NEURON_LAYER_KEYS = {
    'bert': [['intermediate', 'intermediate.output']],
    't5': [['encoder_i', 'encoder_io'], ['decoder_i', 'decoder_io']],
    't5_lm_adapt': [['encoder_i0', 'encoder_i1', 'encoder_io'], ['decoder_i0', 'decoder_i1', 'decoder_io']],
}

def shortens_inputs(inputs):
    max_length = inputs["attention_mask"].sum(-1).max().item()
    inputs["input_ids"] = inputs["input_ids"][:, :max_length]
    inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]
    if "token_type_ids" in inputs:
        inputs["token_type_ids"] = inputs["token_type_ids"][:, :max_length]

def to_same_shape(p: torch.Tensor, q: torch.Tensor, mode: str='trim'):
    p_shape, q_shape = p.shape[0], q.shape[0]
    if p_shape != q_shape:
        if mode == 'trim':
            min_shape = min(p_shape, q_shape)
            if p_shape < q_shape:
                q = q[torch.randperm(q_shape)[:min_shape]]
            else:
                p = p[torch.randperm(p_shape)[:min_shape]]
            return p, q
    else:
        return p, q
    

def js_divergence(p: torch.Tensor, q: torch.Tensor):
    p, q = to_same_shape(p, q)
    m = 0.5 * (p + q)
    return torch.Tensor(0.5 * (entropy(p, m) + entropy(q, m)))

def kl_divergence(p: torch.Tensor, q: torch.Tensor):
    p, q = to_same_shape(p, q)
    return torch.Tensor(entropy(p, q))

class BaseScorer:
    def __init__(self, model: PreTrainedModel, dataloader: Optional[DataLoader]):
        self.model = model
        self.dataloader = dataloader
    
    def head_score(self) -> Union[torch.Tensor, None]:
        return None
    
    def intermediate_score(self) -> Union[torch.Tensor, None]:
        return None
    
class GradientScorer(BaseScorer):
    def __init__(self, model: PreTrainedModel, dataloader: DataLoader, norm: str='l2', normalize: bool = False):
        super().__init__(model, dataloader)
        self.grads_collected = False
        self.head_grads = None
        self.intermediate_grads = None
        self.hidden_grads = None
        if norm not in set(['l1', 'l2']):
            raise NotImplementedError(f'Norm {norm} not implemented')
        self.norm = norm
        self.normalize = normalize
        self.total_params_per_head = (768 * 768 // 12 + 768 // 12) * 4
        self.total_params_per_neuron = (768 * 2 + 1)
        self.scaling = None
        
    def head_score(self):
        if not self.grads_collected:
            self._collect_grads()
        return compute_l1_fisher_info(self.head_grads) if self.norm == 'l1' else compute_fisher_info(self.head_grads) if self.norm == 'l2' else None
    
    def intermediate_score(self):
        if not self.grads_collected:
            self._collect_grads()
        return compute_l1_fisher_info(self.intermediate_grads) if self.norm == 'l1' else compute_fisher_info(self.intermediate_grads) if self.norm == 'l2' else None

    def hidden_score(self):
        if getattr(self.model, 'hidden_mask', None) is None:
            return None
        if not self.grads_collected:
            self._collect_grads()
        return compute_l1_fisher_info(self.hidden_grads) if self.norm == 'l1' else compute_fisher_info(self.hidden_grads) if self.norm == 'l2' else None
    
    def _collect_grads(self):
        self.head_grads, self.intermediate_grads = collect_mask_grads(
            self.model,
            self.dataloader,
        )
        if getattr(self.model, 'hidden_mask', None) is not None:
            self.hidden_grads = collect_hidden_mask_grads(self.model, self.dataloader)
        else:
            self.hidden_grads = None
        if self.normalize:
            avg_head_fisher_per_param = compute_fisher_info(self.head_grads).mean() / self.total_params_per_head
            avg_neuron_fisher_per_param = compute_fisher_info(self.intermediate_grads).mean() / self.total_params_per_neuron
            self.scaling = (avg_head_fisher_per_param.item() / avg_neuron_fisher_per_param.item())
            print("Head fisher per param: ", avg_head_fisher_per_param, "Neuron fisher per param: ", avg_neuron_fisher_per_param, "Scaling: ", self.scaling)
            self.intermediate_grads *= self.scaling
        self.grads_collected = True


class PredictivityScorer(BaseScorer):
    # TODO: Lowering the GPU memory usage
    def head_score(self):
        return None
    
    @torch.no_grad()
    def gather_activations(self) -> torch.Tensor:
        self.model.eval()
        gathered_activations = []
        handles = []
        labels = []
        for i in range(self.model.config.num_hidden_layers):
            activations = []
            handle = hijack_input(get_ffn2(self.model, i), activations, input_index=0, device='cpu')
            gathered_activations.append(activations)
            handles.append(handle)
        for inputs in tqdm(self.dataloader):
            inputs = {k: torch.stack(v, dim=1).to(self.model.device) if isinstance(v, list) else v.to(self.model.device) for k, v in inputs.items()}
            self.model(**inputs)
            labels.append(inputs['labels'])
        for handle in handles:
            handle.remove()
        gathered_activations = [
            torch.concat(v, dim=0)[:, 0, :].to('cpu') # Only retaining the activation on token [CLS] (or <s> for Roberta)
            for v in gathered_activations
        ]
        gathered_activations = torch.stack(gathered_activations, dim=1)
        labels = torch.concat(labels, dim=0).cpu()
        return gathered_activations, labels
    
    def get_activation_divergence(self, aggregation: str='max') -> torch.Tensor:
        gathered_activations, labels = self.gather_activations()
        activations = softmax(gathered_activations, dim=0)
        num_labels = len(self.model.config.label2id)
        activations_by_label = [
            activations.index_select(dim=0, index=(labels == i).nonzero().squeeze())
            for i in range(num_labels)
        ]
        mutual_divergence = torch.stack([
            js_divergence(activations_by_label[i], activations_by_label[j])
            for i in range(num_labels) for j in range(i + 1, num_labels)
        ], dim=0)
        if aggregation == 'max':
            aggregated_divergence = mutual_divergence.max(dim=0)[0]
        elif aggregation == 'mean':
            aggregated_divergence = mutual_divergence.mean(dim=0)
        return aggregated_divergence, mutual_divergence, gathered_activations, labels

    def intermediate_score(self, aggregation: str='max') -> Union[torch.Tensor, None]:
        return self.get_activation_divergence(aggregation=aggregation)[0]
    
class RunningScorer(BaseScorer):
    def __init__(self, model: PreTrainedModel, param_controller: ParamController, state: TrainerState, dataloader: Optional[DataLoader] = None, gather_freq: int = 1):
        super().__init__(model, dataloader)
        self.param_controller = param_controller
        self.score = {}
        self.state = state
        self.gather_freq = gather_freq
        self.model_category = param_controller.model_arch.model_category
        self.training_step_func = None
        print(f"Model category: {self.model_category}, is_gated_act: {getattr(model.config, 'is_gated_act', False)}", flush=True)
        if 't5' in self.model_category and getattr(model.config, 'is_gated_act', False):
            self.model_category = 't5_lm_adapt'
        
    def _gather_score(self):
        raise NotImplementedError("Please implement this method in the subclass")
        
    def step(self):
        if self.state.global_step % self.gather_freq:
            # Omit the gathering step if not needed
            return
        if self.dataloader is not None:
            # Should be called when model's grad is zeroed or set to None
            # Checking model grad status
            for p in self.model.parameters():
                if p.grad is not None and (p.grad != 0).any().item():
                    raise RuntimeError("Model's grad is not zeroed. Please call scorer.step() after zeroing the grad.")
            for inputs in tqdm(self.dataloader):
                if self.training_step_func is not None:
                    self.training_step_func(self.model, inputs)
                else:
                    shortens_inputs(inputs)
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    if self.model_category != 'bert':
                        inputs['use_cache'] = False
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    loss.backward()
                
                self._gather_score()
                self.model.zero_grad()
        else:
                self._gather_score()
            
    def get_score(self, m):
        return self.score.get(m, None)
            
    def save_state(self, save_path: str):
        print(f"Saving state to {save_path}")
        saved_dict = {
            'score': self.score,
            'step': self.state.global_step,
        }
        torch.save(saved_dict, save_path)
        
    def end(self):
        self.score = {}
        
class RunningSalienceScorer(RunningScorer):
    def __init__(self, model: PreTrainedModel, param_controller: ParamController, state: TrainerState, dataloader: Optional[DataLoader] = None, gather_freq: int = 1, beta_1: float = 0.85, beta_2: float = 0.85, use_uncertainty: bool = False, block_normalize_dict: Optional[Dict[str, float]] = None, **kwargs):
        super().__init__(model, param_controller, state, dataloader, gather_freq)
        self.salience_dict = {
            'head_mask':{
                's': 0,
                'u': 0,
            },
            'intermediate_mask':{
                's': 0,
                'u': 0,
            },
            'hidden_mask': {
                's': 0,
                'u': 0,
            },
            'modules': {}
        }
        self.reset_module_scores()
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.use_uncertainty = use_uncertainty
        self.block_normalize_dict = block_normalize_dict
        self.accumulation_started = False
        if 'bert' in model.config.model_type or 'llama' in model.config.model_type:
            self.attention_head_size = self.model.config.hidden_size // self.model.config.num_attention_heads
            self.num_layers = self.model.config.num_hidden_layers
        elif 't5' in model.config.model_type:
            self.attention_head_size = self.model.config.d_model // self.model.config.num_heads
            self.num_layers = self.model.config.num_layers
        else:
            raise NotImplementedError(f"Model type {model.config.model_type} not implemented")
        self.retained_indices = {}
        self.removed_indices = {}
        
    def reset_module_scores(self):
        for i, v in self.param_controller.param_config.items():
            self.salience_dict['modules'][i] = {}
            for k, val in v.items():
                if val != 'none':
                    self.salience_dict['modules'][i][k] = {
                        'input_mask': {
                            's': 0,
                            'u': 0,
                        },
                        'output_mask': {
                            's': 0,
                            'u': 0,
                        },
                        'bottleneck_mask': {
                            's': 0,
                            'u': 0,
                        }
                    }
        self.accumulation_started = False
        
    def _gather_module_score(self):
        for i in self.salience_dict['modules']:
            for k in self.salience_dict['modules'][i]:
                    k_attr = NAME2ATTR[k][self.param_controller.model_arch.model_category]
                    parent_layer = self.param_controller.get_parent_layer(i, k)
                    src_layer = getattr(parent_layer, k_attr)
                    if not isinstance(src_layer, lora.PruningLinear) or src_layer.r == 0:
                        continue
                    # for m in ['input_mask', 'output_mask', 'bottleneck_mask']:
                    #     current_m: nn.Parameter = getattr(src_layer, m, None)
                    #     if current_m is None:
                    #         continue
                    #     if current_m.grad is None:
                    #         current_s = torch.zeros_like(current_m.data)
                    #     else:
                    #         current_s = (current_m.detach().clone() * current_m.grad.detach().clone()).abs()
                    #     self.salience_dict['modules'][i][k][m]['s'] = self.salience_dict['modules'][i][k][m]['s'] * self.beta_1 + (1 - self.beta_1) * current_s
                    #     if self.use_uncertainty:
                    #         self.salience_dict['modules'][i][k][m]['u'] = self.salience_dict['modules'][i][k][m]['u'] * self.beta_2 + (1 - self.beta_2) * ((current_s - self.salience_dict['modules'][i][k][m]['s']) / self.salience_dict['modules'][i][k][m]['s']).abs()
                    # Get bottleneck sensitivity only
                    m = 'bottleneck_mask'
                    calculated_sensitivity = src_layer.get_r_sensitivity()
                    if calculated_sensitivity is None:
                        # print("Warning: sensitivity is None for layer {}, attr {}".format(i, k), flush=True)
                        # When the corresponding attn/ffn mask 0, the grad is None
                        calculated_sensitivity = 0
                    self.salience_dict['modules'][i][k][m]['s'] = self.salience_dict['modules'][i][k][m]['s'] * self.beta_1 + (1 - self.beta_1) * calculated_sensitivity
                    if self.use_uncertainty:
                        self.salience_dict['modules'][i][k][m]['u'] = self.salience_dict['modules'][i][k][m]['u'] * self.beta_2 + (1 - self.beta_2) * ((calculated_sensitivity - self.salience_dict['modules'][i][k][m]['s']) / self.salience_dict['modules'][i][k][m]['s']).abs()

    def get_score(self, m):
        return self.salience_dict[m]['s'] if not self.use_uncertainty else self.salience_dict[m]['s'] * self.salience_dict[m]['u']
    
    def get_module_score(self):
        return self.salience_dict['modules']
    
    def get_salience_dict(self):
        return {
            'score': {k: self.get_score(k) for k in self.salience_dict.keys() if k.endswith('mask')},
            'salience': self.salience_dict,
            'step': self.state.global_step,
        }
    
    def prune_salience(self, m, index, clean_to_zero: bool = False):
        if not isinstance(self.salience_dict[m]['s'], torch.Tensor):
            return
        if index is not None:
            self.salience_dict[m]['s'] = self.salience_dict[m]['s'][index].clone().contiguous()
            if isinstance(self.salience_dict[m]['u'], torch.Tensor):
                self.salience_dict[m]['u'] = self.salience_dict[m]['u'][index].clone().contiguous()
        if clean_to_zero:
            self.salience_dict[m]['s'] = torch.zeros_like(self.salience_dict[m]['s'])
            if isinstance(self.salience_dict[m]['u'], torch.Tensor):
                self.salience_dict[m]['u'] = torch.zeros_like(self.salience_dict[m]['u'])
        if m in self.retained_indices:
            self.retained_indices.pop(m)
            
    def set_retained_indices(self, m, index, clean_to_zero: bool = False):
        print(f"Setting retained indices for {m} to {index} with shape {index.shape}, salience shape: {self.salience_dict[m]['s'].shape}")
        mask = torch.zeros_like(self.salience_dict[m]['s']).cpu()
        mask[index.cpu()] = 1
        self.retained_indices[m] = index
        self.removed_indices[m] = (~mask.bool()).nonzero().squeeze().to(self.model.device)
        if clean_to_zero:
            self.salience_dict[m]['s'] = torch.zeros_like(self.salience_dict[m]['s'])
            if isinstance(self.salience_dict[m]['u'], torch.Tensor):
                self.salience_dict[m]['u'] = torch.zeros_like(self.salience_dict[m]['u'])
    
    def save_state(self, save_path: str):
        print(f"Saving state to {save_path}")
        torch.save(self.get_salience_dict(), save_path)
    
    def end(self):
        super().end()
        self.reset_module_scores()
        self.salience_dict = {
            'head_mask':{
                's': 0,
                'u': 0,
            },
            'intermediate_mask':{
                's': 0,
                'u': 0,
            },
            'hidden_mask': {
                's': 0,
                'u': 0,
            },
            'modules': {}
        }
        
class MagnitudeScorer(RunningScorer):
    def __init__(self, model: PreTrainedModel, param_controller: ParamController, state: TrainerState):
        super().__init__(model, param_controller, state)
        self.score = {}
        self.attention_head_size = self.model.config.hidden_size // self.model.config.num_attention_heads
        self.num_layers = self.model.config.num_hidden_layers if self.param_controller.model_arch.model_category == 'bert' else self.model.config.num_layers
    
    def step(self):
        # Nothing to do here with magnitude scorer
        return
        
    def get_score(self, m, pow=1):
        # Getting magnitudes based on the mask type
        if m == 'head_mask':
            scores = []
            for layer_attr_group in ATTENTION_LAYER_KEYS[self.param_controller.model_arch.model_category]:
                for i in range(self.num_layers):
                    # Init a layer's head scores (will be added up with)
                    current_head_score = 0
                    for layer_attr in layer_attr_group:
                        layer = self.param_controller.get_layer(i, layer_attr)
                        # Get weights
                        weight: nn.Parameter = layer.weight
                        bias: Optional[nn.Parameter] = getattr(layer, 'bias', None)
                        weight_score = weight.detach().clone().abs() if pow == 1 else weight.detach().clone().pow(pow)
                        bias_score = (bias.detach().clone().abs() if pow == 1 else bias.detach().clone().pow(pow)) if bias is not None else None
                        # Compute magnitude sum
                        if 'output' in layer_attr:
                            # Output weights, the input dimension corresponding to the head, while the output dimension corresponding to the hidden size
                            aggregated_score = weight_score.sum(dim=0) # Summing up the output dimension
                            # Ignoring the bias because it is corresponding to the hidden size
                        else:
                            # qkv, the input dimension corresponding to the hidden size, while the output dimension corresponding to the head
                            aggregated_score = weight_score.sum(dim=1) # Summing up the input dimension
                            if bias_score is not None:
                                aggregated_score += bias_score
                        # Aggregating the dimension score to the head score
                        head_score = aggregated_score.view(-1, self.attention_head_size).sum(dim=1)
                        current_head_score += head_score
                    scores.append(current_head_score)
            scores = torch.cat(scores)
        elif m == 'intermediate_mask':
            scores = []
            for layer_attr_group in NEURON_LAYER_KEYS[self.model_category]:
                for i in range(self.num_layers):
                    # Init a layer's neuron scores (will be added up with)
                    current_neuron_score = 0
                    for layer_attr in layer_attr_group:
                        layer = self.param_controller.get_layer(i, layer_attr)
                        # Get weights
                        weight: nn.Parameter = layer.weight
                        bias: Optional[nn.Parameter] = getattr(layer, 'bias', None)
                        weight_score = weight.detach().clone().abs() if pow == 1 else weight.detach().clone().pow(pow)
                        bias_score = (bias.detach().clone().abs() if pow == 1 else bias.detach().clone().pow(pow)) if bias is not None else None
                        # Compute magnitude sum
                        if 'output' in layer_attr or 'io' in layer_attr:
                            aggregated_score = weight_score.sum(dim=0) # Summing up the output dimension
                            # Ignoring the bias because it is corresponding to the hidden size
                        else:
                            aggregated_score = weight_score.sum(dim=1) # Summing up the input dimension
                            if bias_score is not None:
                                aggregated_score += bias_score
                        # Aggregating the dimension score to the head score
                        current_neuron_score += aggregated_score
                    scores.append(current_neuron_score)
            scores = torch.cat(scores)
        elif m == 'hidden_mask':
            # We only consider MHA and FFN layers, not layer norm parameters
            scores = 0
            # Aggregate MHA scores first
            for layer_attr_group in ATTENTION_LAYER_KEYS[self.param_controller.model_arch.model_category]:
                for i in range(self.num_layers):
                    # Init a layer's head scores (will be added up with)
                    for layer_attr in layer_attr_group:
                        layer = self.param_controller.get_layer(i, layer_attr)
                        # Get weights
                        weight: nn.Parameter = layer.weight
                        bias: Optional[nn.Parameter] = getattr(layer, 'bias', None)
                        weight_score = weight.detach().clone().abs() if pow == 1 else weight.detach().clone().pow(pow)
                        bias_score = (bias.detach().clone().abs() if pow == 1 else bias.detach().clone().pow(pow)) if bias is not None else None
                        # Compute magnitude sum
                        if 'output' in layer_attr:
                            # Output weights, the input dimension corresponding to the head, while the output dimension corresponding to the hidden size
                            aggregated_score = weight_score.sum(dim=1) # Summing up the output dimension
                            # Ignoring the bias because it is corresponding to the hidden size
                            if bias_score is not None:
                                aggregated_score += bias_score
                        else:
                            # qkv, the input dimension corresponding to the hidden size, while the output dimension corresponding to the head
                            aggregated_score = weight_score.sum(dim=0) # Summing up the input dimension
                        # Aggregating the dimension score to the head score
                        scores += aggregated_score
            # Then aggregating FFN scores
            for layer_attr_group in NEURON_LAYER_KEYS[self.model_category]:
                for i in range(self.num_layers):
                    # Init a layer's neuron scores (will be added up with)
                    current_neuron_score = 0
                    for layer_attr in layer_attr_group:
                        layer = self.param_controller.get_layer(i, layer_attr)
                        # Get weights
                        weight: nn.Parameter = layer.weight
                        bias: Optional[nn.Parameter] = getattr(layer, 'bias', None)
                        weight_score = weight.detach().clone().abs() if pow == 1 else weight.detach().clone().pow(pow)
                        bias_score = (bias.detach().clone().abs() if pow == 1 else bias.detach().clone().pow(pow)) if bias is not None else None
                        # Compute magnitude sum
                        if 'output' in layer_attr or 'io' in layer_attr:
                            aggregated_score = weight_score.sum(dim=1) # Summing up the output dimension
                            # Ignoring the bias because it is corresponding to the hidden size
                            if bias_score is not None:
                                aggregated_score += bias_score
                        else:
                            aggregated_score = weight_score.sum(dim=0) # Summing up the input dimension
                        scores += aggregated_score
        else:
            raise ValueError(f"Mask type {m} not supported.")
        return scores
    
    def save_state(self, save_path: str):
        scores = {
            'head_mask': self.get_score('head_mask'),
            'intermediate_mask': self.get_score('intermediate_mask'),
            'hidden_mask': self.get_score('hidden_mask'),
            'step': self.state.global_step,
        }
        torch.save(scores, save_path)
    
class RunningWandaScorer(RunningScorer):
    def __init__(self, model: PreTrainedModel, param_controller: ParamController, state: TrainerState, dataloader: Optional[DataLoader] = None, gather_freq: int = 1, beta_1: float = 0.85, beta_2: float = 0.85, use_uncertainty: bool = False, block_normalize_dict: Optional[Dict[str, float]] = None, **kwargs):
        super().__init__(model, param_controller, state, dataloader, gather_freq)
        self.score = {}
        self.handlers = []
        for m in ['head_mask', 'intermediate_mask', 'hidden_mask']:
            self.score[m] = {
                's': 0,
                'u': 0,
            }
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.use_uncertainty = use_uncertainty
        self.block_normalize_dict = block_normalize_dict
        self.accumulation_started = False
        self.attention_head_size = self.model.config.hidden_size // self.model.config.num_attention_heads
        self.num_layers = self.model.config.num_hidden_layers if self.param_controller.model_arch.model_category == 'bert' else self.model.config.num_layers
        
    def clear_wanda(self):
        for n in self.model.modules():
            if hasattr(n, 'wanda'):
                n.wanda = 0
    
    def step(self, inputs: Optional[Dict[str, torch.Tensor]] = None):
        if self.state.global_step % self.gather_freq:
            # Omit the gathering step if not needed
            return
        for layer_attr_group in ATTENTION_LAYER_KEYS[self.param_controller.model_arch.model_category] + NEURON_LAYER_KEYS[self.model_category]:
            for i in range(self.num_layers):
                # Init a layer's head scores (will be added up with)
                for layer_attr in layer_attr_group:
                    layer: nn.Linear = self.param_controller.get_layer(i, layer_attr)
                    def calculate_wanda(module: nn.Linear, layer_inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]):
                        with torch.no_grad():
                            hidden_states = layer_inputs[0]
                            input_norm = hidden_states.pow(2).view(-1, hidden_states.shape[-1]).sum(dim=0) # input_norm shape: (hidden_size), module.weight: (proj_size, hidden_size), if qkv or io, otherwise (hidden_size, proj_size)
                            wanda = input_norm * module.weight.abs() # wanda shape: (proj_size, hidden_size), if qkv or io, otherwise (hidden_size, proj_size)
                            # Store the wanda first, then aggregate based on layer type
                            if hasattr(module, 'wanda'):
                                module.wanda += wanda
                            else:
                                module.wanda = wanda
                    if layer is not None:
                        handle = layer.register_forward_hook(calculate_wanda)
                    self.handlers.append(handle)
        if self.dataloader is not None:
            # Should be called when model's grad is zeroed or set to None
            # Checking model grad status
            for p in self.model.parameters():
                if p.grad is not None and (p.grad != 0).any().item():
                    raise RuntimeError("Model's grad is not zeroed. Please call scorer.step() after zeroing the grad.")
            for current_inputs in self.dataloader:
                self._gather_score(current_inputs)
        else:
            if inputs is None:
                raise ValueError("Please provide inputs when dataloader is None")
            self._gather_score(inputs)
        for handle in self.handlers:
            handle.remove()
        self.handlers = []
        
    def _gather_score(self, inputs: Dict[str, torch.Tensor] = None):
        # Get each block's activation-weight norm production scores
            shortens_inputs(inputs)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                self.model(**inputs)
            
    def get_score(self, m):
        # Aggregate the wanda scores
        if m == 'head_mask':
            scores = []
            for layer_attr_group in ATTENTION_LAYER_KEYS[self.param_controller.model_arch.model_category]:
                for i in range(self.num_layers):
                    # Init a layer's head scores (will be added up with)
                    current_score = 0
                    for layer_attr in layer_attr_group:
                        layer = self.param_controller.get_layer(i, layer_attr)
                        with torch.no_grad():
                            wanda = layer.wanda
                            if 'output' in layer_attr:
                                current_score += wanda.sum(dim=0).view(-1, self.attention_head_size).sum(dim=1)
                            else:
                                current_score += wanda.sum(dim=1).view(-1, self.attention_head_size).sum(dim=1)
                    scores.append(current_score)
            scores = torch.cat(scores)
        elif m == 'intermediate_mask':
            scores = []
            for layer_attr_group in NEURON_LAYER_KEYS[self.model_category]:
                for i in range(self.num_layers):
                    # Init a layer's intermediate scores (will be added up with)
                    current_score = 0
                    for layer_attr in layer_attr_group:
                        layer: nn.Linear = self.param_controller.get_layer(i, layer_attr)
                        with torch.no_grad():
                            wanda = layer.wanda
                            if ('output' in layer_attr or 'io' in layer_attr):
                                current_score += wanda.sum(dim=0)
                            else:
                                current_score += wanda.sum(dim=1)
                    scores.append(current_score)
            scores = torch.cat(scores)
        elif m == 'hidden_mask':
            scores = 0
            for layer_attr_group in ATTENTION_LAYER_KEYS[self.param_controller.model_arch.model_category] + NEURON_LAYER_KEYS[self.model_category]:
                for i in range(self.num_layers):
                    # Init a layer's head scores (will be added up with)
                    for layer_attr in layer_attr_group:
                        layer = self.param_controller.get_layer(i, layer_attr)
                        with torch.no_grad():
                            wanda = layer.wanda
                            if 'output' in layer_attr or 'io' in layer_attr:
                                scores += wanda.sum(dim=1)
                            else:
                                scores += wanda.sum(dim=0)
        else:
            raise ValueError(f"Mask type {m} not supported.")
        return scores

    def delete_wanda(self):
        for n in self.model.modules():
            if hasattr(n, 'wanda'):
                del n.wanda
        self.handlers = []

    def end(self):
        super().end()
        self.delete_wanda()
            
class RunningMaskSalienceScorer(RunningSalienceScorer):
    def __init__(self, model: PreTrainedModel, param_controller: ParamController, state: TrainerState, dataloader: Optional[DataLoader] = None, gather_freq: int = 1, beta_1: float = 0.85, beta_2: float = 0.85, use_uncertainty: bool = False, block_normalize_dict: Optional[Dict[str, float]] = None, **kwargs):
        super().__init__(model, param_controller, state, dataloader, gather_freq, beta_1, beta_2, use_uncertainty, block_normalize_dict, **kwargs)
    
    def _gather_score(self):
        """
            Overriding the _gather_score method
        """
        self.accumulation_started = True
        for m in ['head_mask', 'intermediate_mask', 'hidden_mask']:
            mask: torch.Tensor = getattr(self.model, m, None)
            if mask is not None and mask.grad is not None:
                current_salience = (mask.detach().clone() * mask.grad.detach().clone()).pow(2)
                if self.block_normalize_dict is not None:
                    current_salience *= self.block_normalize_dict[m]
                retained_indices = self.retained_indices.get(m, None)
                if retained_indices is not None and self.model.virtual_pruned:
                    removed_indices = self.removed_indices[m]
                    if self.use_uncertainty:
                        current_uncertainty = (current_salience - self.salience_dict[m]['s'][retained_indices]).abs()
                        self.salience_dict[m]['u'][retained_indices] = self.salience_dict[m]['u'][retained_indices] * self.beta_2 + (1 - self.beta_2) * current_uncertainty
                        self.salience_dict[m]['u'][removed_indices] = self.salience_dict[m]['u'][removed_indices] * self.beta_2
                    self.salience_dict[m]['s'][retained_indices] = self.salience_dict[m]['s'][retained_indices] * self.beta_1 + (1 - self.beta_1) * current_salience
                    self.salience_dict[m]['s'][removed_indices] = self.salience_dict[m]['s'][removed_indices] * self.beta_1
                else:
                    if self.use_uncertainty:
                        current_uncertainty = (current_salience - self.salience_dict[m]['s']).abs()
                        self.salience_dict[m]['u'] = self.salience_dict[m]['u'] * self.beta_2 + (1 - self.beta_2) * current_uncertainty
                    self.salience_dict[m]['s'] = self.salience_dict[m]['s'] * self.beta_1 + (1 - self.beta_1) * current_salience
        self._gather_module_score()

        
class JointRunningMaskSalienceScorer(RunningMaskSalienceScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _get_combined_grafting_score(self, m: str):
        if m.startswith('hidden'):
            hiddens = [
                val['input_mask']['s']
                for v in self.salience_dict['modules'].values() for val in v.values() if isinstance(val['input_mask']['s'], torch.Tensor)
            ]
            if len(hiddens):
                return torch.stack(hiddens).sum(dim=0)
            else:
                return torch.tensor(0).to(self.model.device)
        elif m.startswith('head'):
            queries = [self.salience_dict['modules'][layer]['query']['output_mask']['s'].view(-1, self.attention_head_size).sum(dim=1) for layer in range(self.num_layers) if isinstance(self.salience_dict['modules'][layer]['query']['output_mask']['s'], torch.Tensor)]
            values = [self.salience_dict['modules'][layer]['value']['output_mask']['s'].view(-1, self.attention_head_size).sum(dim=1) for layer in range(self.num_layers) if isinstance(self.salience_dict['modules'][layer]['value']['output_mask']['s'], torch.Tensor)]
            if len(queries) and len(values):
                query_tuning_scores = torch.cat(queries)
                value_tuning_scores = torch.cat(values)
                return query_tuning_scores + value_tuning_scores
            else:
                return torch.tensor(0).to(self.model.device)
        elif m.startswith('intermediate'):
            intermediates = [self.salience_dict['modules'][layer]['intermediate']['output_mask']['s'] for layer in range(self.num_layers) if 'intermediate' in self.salience_dict['modules'][layer] and isinstance(self.salience_dict['modules'][layer]['intermediate']['output_mask']['s'], torch.Tensor)]
            if len(intermediates):
                return torch.cat(intermediates)
            else:
                return torch.tensor(0).to(self.model.device)
        else:
            raise NotImplementedError(f"Mask type {m} not implemented.")
        
    def get_score(self, m):
        super_score = super().get_score(m)
        if isinstance(super_score, torch.Tensor):
            if self.accumulation_started:
                return (super_score * self._get_combined_grafting_score(m)).abs()
            else:
                return super_score
        else:
            return self._get_combined_grafting_score(m)
        

class T5JointRunningMaskSalienceScorer(RunningMaskSalienceScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.param_controller.teacher_config is None:
            return
        self.attention_head_size = self.model.config.d_model // self.model.config.num_heads
        self.num_layers = self.model.config.num_layers
        if 'encoder_i' in self.param_controller.teacher_config:
            self.encoder_i_key = 'encoder_i'
        elif 'encoder_i0' in self.param_controller.teacher_config:
            self.encoder_i_key = 'encoder_i0'
        elif 'encoder_i1' in self.param_controller.teacher_config:
            self.encoder_i_key = 'encoder_i1'
        else:
            self.encoder_i_key = None
        if 'decoder_i' in self.param_controller.teacher_config:
            self.decoder_i_key = 'decoder_i'
        elif 'decoder_i0' in self.param_controller.teacher_config:
            self.decoder_i_key = 'decoder_i0'
        elif 'decoder_i1' in self.param_controller.teacher_config:
            self.decoder_i_key = 'decoder_i1'
        else:
            self.decoder_i_key = None
        
    def _get_combined_grafting_score(self, m: str):
        if m.startswith('hidden'):
            hiddens = [
                val['input_mask']['s']
                for v in self.salience_dict['modules'].values() for val in v.values() if isinstance(val['input_mask']['s'], torch.Tensor)
            ]
            if len(hiddens):
                return torch.stack(hiddens).sum(dim=0)
            else:
                return torch.tensor(0).to(self.model.device)
        elif m.startswith('head'):
            enc_queries = [self.salience_dict['modules'][layer]['enc_self_query']['output_mask']['s'].view(-1, self.attention_head_size).mean(dim=1) for layer in range(self.num_layers) if isinstance(self.salience_dict['modules'][layer]['enc_self_query']['output_mask']['s'], torch.Tensor)]
            enc_values = [self.salience_dict['modules'][layer]['enc_self_value']['output_mask']['s'].view(-1, self.attention_head_size).mean(dim=1) for layer in range(self.num_layers) if isinstance(self.salience_dict['modules'][layer]['enc_self_value']['output_mask']['s'], torch.Tensor)]
            if len(enc_queries) and len(enc_values):
                enc_query_tuning_scores = torch.cat(enc_queries)
                enc_value_tuning_scores = torch.cat(enc_values)
                enc_head_tuning_scores = (enc_query_tuning_scores + enc_value_tuning_scores) / 2
            else:
                enc_head_tuning_scores = torch.tensor([]).to(self.model.device)
                
            dec_queries = [self.salience_dict['modules'][layer]['dec_self_query']['output_mask']['s'].view(-1, self.attention_head_size).mean(dim=1) for layer in range(self.num_layers) if isinstance(self.salience_dict['modules'][layer]['dec_self_query']['output_mask']['s'], torch.Tensor)]
            dec_values = [self.salience_dict['modules'][layer]['dec_self_value']['output_mask']['s'].view(-1, self.attention_head_size).mean(dim=1) for layer in range(self.num_layers) if isinstance(self.salience_dict['modules'][layer]['dec_self_value']['output_mask']['s'], torch.Tensor)]
            if len(dec_queries) and len(dec_values):
                dec_query_tuning_scores = torch.cat(dec_queries)
                dec_value_tuning_scores = torch.cat(dec_values)
                dec_head_tuning_scores = (dec_query_tuning_scores + dec_value_tuning_scores) / 2
            else:
                dec_head_tuning_scores = torch.tensor([]).to(self.model.device)
                
            cross_queries = [self.salience_dict['modules'][layer]['cross_query']['output_mask']['s'].view(-1, self.attention_head_size).mean(dim=1) for layer in range(self.num_layers) if isinstance(self.salience_dict['modules'][layer]['cross_query']['output_mask']['s'], torch.Tensor)]
            cross_values = [self.salience_dict['modules'][layer]['cross_value']['output_mask']['s'].view(-1, self.attention_head_size).mean(dim=1) for layer in range(self.num_layers) if isinstance(self.salience_dict['modules'][layer]['cross_value']['output_mask']['s'], torch.Tensor)]
            if len(cross_queries) and len(cross_values):
                cross_query_tuning_scores = torch.cat(cross_queries)
                cross_value_tuning_scores = torch.cat(cross_values)
                cross_head_tuning_scores = (cross_query_tuning_scores + cross_value_tuning_scores) / 2
            else:
                cross_head_tuning_scores = torch.tensor([]).to(self.model.device)

            heads = [enc_head_tuning_scores, dec_head_tuning_scores, cross_head_tuning_scores]
            if len(heads):
                return torch.cat(heads)
            else:
                return torch.tensor(0).to(self.model.device)
        elif m.startswith('intermediate'):
            enc_intermediates = [self.salience_dict['modules'][layer][self.encoder_i_key]['output_mask']['s'] for layer in range(self.num_layers) if self.encoder_i_key in self.salience_dict['modules'][layer] and isinstance(self.salience_dict['modules'][layer][self.encoder_i_key]['output_mask']['s'], torch.Tensor)]
            dec_intermediates = [self.salience_dict['modules'][layer][self.decoder_i_key]['output_mask']['s'] for layer in range(self.num_layers) if self.decoder_i_key in self.salience_dict['modules'][layer] and isinstance(self.salience_dict['modules'][layer][self.decoder_i_key]['output_mask']['s'], torch.Tensor)]
            intermediates = enc_intermediates + dec_intermediates
            if len(intermediates):
                return torch.cat(intermediates)
            else:
                return torch.tensor(0).to(self.model.device)
        else:
            raise NotImplementedError(f"Mask type {m} not implemented.")

    def get_score(self, m):
        super_score = super().get_score(m)
        if isinstance(super_score, torch.Tensor):
            if self.accumulation_started:
                return (super_score * self._get_combined_grafting_score(m)).abs()
            else:
                return super_score
        else:
            return self._get_combined_grafting_score(m)
        
class RunningHiddenStatesSalienceScorer(RunningSalienceScorer):
    def __init__(self, model: PreTrainedModel, param_controller: ParamController, state: TrainerState, dataloader: Optional[DataLoader] = None, gather_freq: int = 1, beta_1: float = 0.85, beta_2: float = 0.85, use_uncertainty: bool = False, block_normalize_dict: Optional[Dict[str, float]] = None, **kwargs):
        # Add hooks to the model (MHA, FFN, LayerNorm)
        # Add MHA hooks
        self.mha_handlers = []
        self.current_mha_states = []
        self.ffn_handlers = []
        self.current_ffn_states = []
        self.hidden_handlers = []
        self.current_hidden_states = []
        self.hooks_registered = False
        super().__init__(model, param_controller, state, dataloader, gather_freq, beta_1, beta_2, use_uncertainty, block_normalize_dict)

    def _register_hooks(self):
        print("Registering hooks in the model with hidden states salience scorer...", flush=True)
        def calculate_mha_score(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]):
            mha_hidden_states = outputs[0]
            if mha_hidden_states is not None and mha_hidden_states.requires_grad:
                mha_hidden_states.retain_grad()
                self.current_mha_states.append(mha_hidden_states)
            input_states = layer_inputs[0]
            if input_states is not None and input_states.requires_grad:
                input_states.retain_grad()
                self.current_hidden_states.append(input_states)
        
        def calculate_ffn_score(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: torch.Tensor):
            ffn_hidden_states = outputs
            if ffn_hidden_states is not None and ffn_hidden_states.requires_grad:
                ffn_hidden_states.retain_grad()
                self.current_ffn_states.append(ffn_hidden_states)
            input_states = layer_inputs[0]
            if input_states is not None and input_states.requires_grad:
                input_states.retain_grad()
                self.current_hidden_states.append(input_states)
            
        def model_output_hook(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: torch.Tensor):
            output_hidden_states = outputs[0]
            if output_hidden_states is not None and output_hidden_states.requires_grad:
                output_hidden_states.retain_grad()
                self.current_hidden_states.append(output_hidden_states)
        
        for layer in range(self.num_layers):
            mha_layer = self.param_controller.get_parent_layer(layer, 'query')
            if mha_layer is not None:
                mha_handler = mha_layer.register_forward_hook(calculate_mha_score)
                self.mha_handlers.append(mha_handler)
            ffn_layer = self.param_controller.get_parent_layer(layer, 'intermediate')
            if ffn_layer is not None:
                ffn_handler = self.param_controller.get_parent_layer(layer, 'intermediate').register_forward_hook(calculate_ffn_score)
                self.ffn_handlers.append(ffn_handler)

        self.hidden_handlers.append(getattr(self.model,self.model.base_model_prefix).encoder.register_forward_hook(model_output_hook))
        self.hooks_registered = True
            
            
    def _gather_score(self):
        if not self.hooks_registered:
            self._register_hooks()
            return # Skip the first gathering step
        with torch.no_grad():
            all_head_scores = []
            for mha in self.current_mha_states:
                # shape (batch_size, seq_len, hidden_size)
                salience = (mha.detach().clone() * mha.grad.detach().clone()).abs() if mha.grad is not None else torch.zeros_like(mha) # sometimes the grad is None because the mask is set to zero
                head_salience = salience.sum(dim=0).sum(dim=0).view(-1, self.attention_head_size).sum(dim=1)
                all_head_scores.append(head_salience)
                
            all_neuron_scores = []
            for ffn in self.current_ffn_states:
                # shape (batch_size, seq_len, intermediate_size)
                salience = (ffn.detach().clone() * ffn.grad.detach().clone()).abs() if ffn.grad is not None else torch.zeros_like(ffn)
                neuron_salience = salience.sum(dim=0).sum(dim=0)
                all_neuron_scores.append(neuron_salience)
                
            all_hidden_scores = 0
            for hidden in self.current_hidden_states:
                # shape (batch_size, seq_len, hidden_size)
                salience = (hidden.detach().clone() * hidden.grad.detach().clone()).abs() if hidden.grad is not None else torch.zeros_like(hidden)
                salience = salience.sum(dim=0).sum(dim=0)
                all_hidden_scores += salience
                
            current_salience_dict = {
                'head_mask': torch.cat(all_head_scores),
                'intermediate_mask': torch.cat(all_neuron_scores),
                'hidden_mask': all_hidden_scores,
            }
            self.current_mha_states = []
            self.current_ffn_states = []
            self.current_hidden_states = []
            for m in ['head_mask', 'intermediate_mask', 'hidden_mask']:
                current_salience = current_salience_dict[m]
                if self.block_normalize_dict is not None:
                    current_salience *= self.block_normalize_dict[m]
                if self.use_uncertainty:
                    current_uncertainty = (current_salience - self.salience_dict[m]['s']).abs()
                    self.salience_dict[m]['u'] = self.salience_dict[m]['u'] * self.beta_2 + (1 - self.beta_2) * current_uncertainty
                self.salience_dict[m]['s'] = self.salience_dict[m]['s'] * self.beta_1 + (1 - self.beta_1) * current_salience
            self._gather_module_score()
            
    def _remove_hooks(self):
        print("Removing hooks in the model with hidden states salience scorer...", flush=True)
        for handle in self.mha_handlers:
            handle.remove()
        for handle in self.ffn_handlers:
            handle.remove()
        for handle in self.hidden_handlers:
            handle.remove()
        self.mha_handlers = []
        self.ffn_handlers = []
        self.hidden_handlers = []
        self.hooks_registered = False
        
    def _reset_hooks(self):
        self._remove_hooks()
        self._register_hooks()
        
    def end(self):
        self._remove_hooks()
        
    def reset_module_scores(self):
        super().reset_module_scores()
        if len(self.mha_handlers) or len(self.ffn_handlers) or len(self.hidden_handlers):
            self._reset_hooks() # When salience is pruned, it means the model layers are changed, so we need to re-register the hooks


class BackwardRunningHiddenStatesSalienceScorer(RunningSalienceScorer):
    def __init__(self, model: PreTrainedModel, param_controller: ParamController, state: TrainerState, dataloader: Optional[DataLoader] = None, gather_freq: int = 1, beta_1: float = 0.85, beta_2: float = 0.85, use_uncertainty: bool = False, block_normalize_dict: Optional[Dict[str, float]] = None, static: bool = False, use_kurtosis: bool = False, log_kurtosis: bool = False, **kwargs):
        # Add hooks to the model (MHA, FFN, LayerNorm)
        # Add MHA hooks
        self.mha_handlers = []
        self.current_mha_states = []
        self.ffn_handlers = []
        self.current_ffn_states = []
        self.hidden_handlers = []
        self.current_hidden_states = []
        self.all_head_scores = []
        self.all_ffn_scores = []
        self.all_mha_kurtosis = []
        self.all_ffn_kurtosis = []
        self.all_hidden_kurtosis_log = []
        self.all_hidden_kurtosis = 0
        self.all_hidden_scores = 0
        self.hidden_cnt = 0
        self.hooks_registered = False
        self.log_kurtosis = log_kurtosis
        self.training_step_func = lambda model, inputs: model(**inputs, return_dict=False)
        super().__init__(model, param_controller, state, dataloader, gather_freq, beta_1, beta_2, use_uncertainty, block_normalize_dict)
        self.expected_hidden_cnt = 2 * self.num_layers + 1 # each layer's mha and ffn outputs, plus embedding output
        self.static = static
        self.use_kurtosis = use_kurtosis
        print("Scorer's static mode:", self.static, flush=True)

    def _register_hooks(self):
        print("Registering hooks in the model with hidden states salience scorer...", flush=True)
        def cache_mha_states(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]):
            # print("mha forward hook fired")
            mha_hidden_states = outputs[0]
            input_states = layer_inputs[0]
            if mha_hidden_states is not None and mha_hidden_states.requires_grad:
                if mha_hidden_states.any():
                    # print("mha state is not None and not zero")
                    with torch.no_grad():
                        mha_state = mha_hidden_states.abs().sum(dim=0).sum(dim=0)
                        self.current_mha_states.append(mha_state)
                        self.current_hidden_states.append(input_states.abs().sum(dim=0).sum(dim=0))
                else:
                    # print("mha state is not None yet zero")
                    self.current_mha_states.append((None, torch.zeros((mha_hidden_states.shape[-1] // self.attention_head_size, ), device=mha_hidden_states.device)))
            elif mha_hidden_states is None and input_states is not None and input_states.requires_grad:
                # print("mha state is None, is input state None?", input_states is None)
                with torch.no_grad():
                    self.current_hidden_states.append(None)
        
        def calculate_mha_score(module: nn.Module, grad_layer_inputs, grad_outputs):
            # print("mha backward hook fired")
            mha_states_grad = grad_outputs[0]
            mha_states = self.current_mha_states.pop()
            while not isinstance(mha_states, torch.Tensor):
                self.all_head_scores.append(mha_states[1])
                mha_states = self.current_mha_states.pop()
            hidden_states_grad = grad_layer_inputs[0]
            hidden_states = self.current_hidden_states.pop()
            while hidden_states is None:
                hidden_states = self.current_hidden_states.pop()
            with torch.no_grad():
                salience = (mha_states * mha_states_grad.abs().sum(dim=0).sum(dim=0)) if mha_states_grad is not None else torch.zeros_like(mha_states)
                head_salience = salience.view(-1, self.attention_head_size).sum(dim=1)
                self.all_head_scores.append(head_salience)
                if hidden_states_grad is not None:
                    salience = (hidden_states * hidden_states_grad.abs().sum(dim=0).sum(dim=0))
                    self.hidden_cnt += 1
                else:
                    salience = torch.zeros_like(hidden_states)
                self.all_hidden_scores += salience
        
        def cache_ffn_states(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: torch.Tensor):
            # print("ffn forward hook fired")
            ffn_hidden_states = outputs
            input_states = layer_inputs[0]
            if ffn_hidden_states is not None and ffn_hidden_states.requires_grad:
                if ffn_hidden_states.any():
                    # print("ffn state is not None and not zero")
                    with torch.no_grad():
                        ffn_state = ffn_hidden_states.abs().sum(dim=0).sum(dim=0)
                        self.current_ffn_states.append(ffn_state)
                        self.current_hidden_states.append(input_states.abs().sum(dim=0).sum(dim=0))
                else:
                    # print("ffn state is not None yet zero")
                    self.current_ffn_states.append((None, torch.zeros((ffn_hidden_states.shape[-1], ), device=ffn_hidden_states.device)))
            elif ffn_hidden_states is None and input_states is not None and input_states.requires_grad:
                # print("ffn state is None, is input state None?", input_states is None)
                with torch.no_grad():
                    self.current_hidden_states.append(None)

        def calculate_ffn_score(module: nn.Module, grad_layer_inputs, grad_outputs):
            # print("ffn backward hook fired")
            ffn_states_grad = grad_outputs[0]
            ffn_states = self.current_ffn_states.pop()
            while not isinstance(ffn_states, torch.Tensor):
                self.all_ffn_scores.append(ffn_states[1])
                ffn_states = self.current_ffn_states.pop()
            hidden_states_grad = grad_layer_inputs[0]
            hidden_states = self.current_hidden_states.pop()
            while hidden_states is None:
                hidden_states = self.current_hidden_states.pop()
            with torch.no_grad():
                neuron_salience = (ffn_states * ffn_states_grad.abs().sum(dim=0).sum(dim=0)) if ffn_states_grad is not None else torch.zeros_like(ffn_states)
                self.all_ffn_scores.append(neuron_salience)
                if hidden_states_grad is not None:
                    salience = (hidden_states * hidden_states_grad.abs().sum(dim=0).sum(dim=0))
                    self.hidden_cnt += 1
                else:
                    salience = torch.zeros_like(hidden_states)
                self.all_hidden_scores += salience
            
        def model_output_hook(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: torch.Tensor):
            # print("Model output hook fired")
            output_hidden_states = outputs[0]
            if output_hidden_states is not None and output_hidden_states.requires_grad:
                # print("Model output is not None and requires grad")
                with torch.no_grad():
                    self.current_hidden_states.append(output_hidden_states.abs().sum(dim=0).sum(dim=0))

        def calculate_output_score(module: nn.Module, grad_layer_inputs, grad_outputs):
            # print("Model output backward hook fired")
            hidden_states_grad = grad_outputs[0]
            hidden_states = self.current_hidden_states.pop()
            with torch.no_grad():
                if hidden_states_grad is not None:
                    salience = (hidden_states * hidden_states_grad.abs().sum(dim=0).sum(dim=0))
                    self.hidden_cnt += 1
                else:
                    salience = torch.zeros_like(hidden_states)
                self.all_hidden_scores += salience
        
        for layer in range(self.num_layers):
            mha_layer = self.param_controller.get_parent_layer(layer, 'query')
            if mha_layer is not None:
                mha_handler = mha_layer.register_forward_hook(cache_mha_states)
                self.mha_handlers.append(mha_handler)
                mha_handler = mha_layer.register_full_backward_hook(calculate_mha_score)
                self.mha_handlers.append(mha_handler)
            ffn_layer = self.param_controller.get_parent_layer(layer, 'intermediate')
            if ffn_layer is not None:
                ffn_handler = ffn_layer.register_forward_hook(cache_ffn_states)
                self.ffn_handlers.append(ffn_handler)
                ffn_handler = ffn_layer.register_full_backward_hook(calculate_ffn_score)
                self.ffn_handlers.append(ffn_handler)

        output_layer = getattr(self.model,self.model.base_model_prefix).encoder
        self.hidden_handlers.append(output_layer.register_forward_hook(model_output_hook))
        self.hidden_handlers.append(output_layer.register_full_backward_hook(calculate_output_score))
        self.hooks_registered = True
            
            
    def _gather_score(self):
        if not self.hooks_registered:
            self._register_hooks()
            return # Skip the first gathering step
        with torch.no_grad():    
            self.all_head_scores.reverse()
            self.all_ffn_scores.reverse() # reverse scores because in backward propagation, the last layer is calculated first
            if self.use_kurtosis and len(self.all_mha_kurtosis) and len(self.all_ffn_kurtosis):
                self.all_head_scores = [
                    salience + kurtosis.clamp(min=0).sqrt()
                    for salience, kurtosis in zip(self.all_head_scores, self.all_mha_kurtosis)
                ]
                self.all_ffn_scores = [
                    salience + kurtosis.clamp(min=0).sqrt()
                    for salience, kurtosis in zip(self.all_ffn_scores, self.all_ffn_kurtosis)
                ]
                self.all_hidden_scores += self.all_hidden_kurtosis
                # self.all_hidden_scores[:] = 1e30
            if not len(self.all_head_scores) and not len(self.all_ffn_scores):
                print("No salience is calculated in this step. Skip score gathering", flush=True)
                return
            current_salience_dict = {
                'head_mask': torch.cat(self.all_head_scores),
                'intermediate_mask': torch.cat(self.all_ffn_scores),
                # 'hidden_mask': self.all_hidden_scores / self.hidden_cnt * self.expected_hidden_cnt,
                'hidden_mask': self.all_hidden_scores,
            }
            if not ( len(self.current_mha_states) == 0 and len(self.current_ffn_states) == 0 and len(self.current_hidden_states) == 0):
                # make sure all states are popped so there's no mismatch
                raise RuntimeError("Not all states are popped. Length of current states: {}, {}, {}".format(len(self.current_mha_states), len(self.current_ffn_states), len(self.current_hidden_states)))
            self.all_mha_kurtosis = []
            self.all_ffn_kurtosis = []
            self.all_hidden_kurtosis_log = []
            self.all_head_scores = []
            self.all_ffn_scores = []
            self.all_hidden_scores = 0
            self.all_hidden_kurtosis = 0
            self.hidden_cnt = 0
            for m in ['head_mask', 'intermediate_mask', 'hidden_mask']:
                current_salience = current_salience_dict[m]
                retained_indices = self.retained_indices.get(m, None)
                if retained_indices is not None and self.model.virtual_pruned:
                    removed_indices = self.removed_indices[m]
                    if self.block_normalize_dict is not None:
                        current_salience *= self.block_normalize_dict[m]
                    if self.use_uncertainty and not self.static:
                        current_uncertainty = (current_salience - self.salience_dict[m]['s'][retained_indices]).abs()
                        self.salience_dict[m]['u'][retained_indices] = self.salience_dict[m]['u'][retained_indices] * self.beta_2 + (1 - self.beta_2) * current_uncertainty
                        self.salience_dict[m]['u'][removed_indices] = self.salience_dict[m]['u'][removed_indices] * self.beta_2
                    if self.static:
                        self.salience_dict[m]['s'][retained_indices] += current_salience
                    else:
                        self.salience_dict[m]['s'][retained_indices] = self.salience_dict[m]['s'][retained_indices] * self.beta_1 + (1 - self.beta_1) * current_salience
                        self.salience_dict[m]['s'][removed_indices] = self.salience_dict[m]['s'][removed_indices] * self.beta_1
                else:
                    if self.block_normalize_dict is not None:
                        current_salience *= self.block_normalize_dict[m]
                    if self.use_uncertainty and not self.static:
                        current_uncertainty = (current_salience - self.salience_dict[m]['s']).abs()
                        self.salience_dict[m]['u'] = self.salience_dict[m]['u'] * self.beta_2 + (1 - self.beta_2) * current_uncertainty
                    if self.static:
                        self.salience_dict[m]['s'] += current_salience
                    else:
                        self.salience_dict[m]['s'] = self.salience_dict[m]['s'] * self.beta_1 + (1 - self.beta_1) * current_salience
            self._gather_module_score()
            
    def _remove_hooks(self):
        print("Removing hooks in the model with hidden states salience scorer...", flush=True)
        for handle in self.mha_handlers:
            handle.remove()
        for handle in self.ffn_handlers:
            handle.remove()
        for handle in self.hidden_handlers:
            handle.remove()
        self.mha_handlers = []
        self.ffn_handlers = []
        self.hidden_handlers = []
        self.hooks_registered = False
        
    def _reset_hooks(self):
        self._remove_hooks()
        self._register_hooks()
        
    def end(self):
        self._remove_hooks()
        
    def reset_module_scores(self):
        super().reset_module_scores()
        if len(self.mha_handlers) or len(self.ffn_handlers) or len(self.hidden_handlers):
            self._reset_hooks() # When salience is pruned, it means the model layers are changed, so we need to re-register the hooks
            
class RunningT5HiddenStatesSalienceScorer(RunningHiddenStatesSalienceScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_mha_states = [[], [], []] # encoder, decoder, cross
        self.current_ffn_states = [[], []] # encoder, decoder
        
    def _register_hooks(self):
        def calculate_enc_mha_score(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]):
            input_states = layer_inputs[0]
            if input_states is not None and input_states.requires_grad:
                input_states.retain_grad()
                self.current_mha_states[0].append(input_states)
            output_states = outputs
            if output_states is not None and output_states.requires_grad:
                output_states.retain_grad()
                self.current_hidden_states.append(output_states)

        def calculate_dec_mha_score(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]):
            input_states = layer_inputs[0]
            if input_states is not None and input_states.requires_grad:
                input_states.retain_grad()
                self.current_mha_states[1].append(input_states)
            output_states = outputs
            if output_states is not None and output_states.requires_grad:
                output_states.retain_grad()
                self.current_hidden_states.append(output_states)

        def calculate_cross_mha_score(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]):
            input_states = layer_inputs[0]
            if input_states is not None and input_states.requires_grad:
                input_states.retain_grad()
                self.current_mha_states[2].append(input_states)
            output_states = outputs
            if output_states is not None and output_states.requires_grad:
                output_states.retain_grad()
                self.current_hidden_states.append(output_states)
        
        def calculate_enc_ffn_score(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: torch.Tensor):
            input_states = layer_inputs[0]
            if input_states is not None and input_states.requires_grad:
                input_states.retain_grad()
                self.current_ffn_states[0].append(input_states)
            output_states = outputs
            if output_states is not None and output_states.requires_grad:
                output_states.retain_grad()
                self.current_hidden_states.append(output_states)

        def calculate_dec_ffn_score(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: torch.Tensor):
            input_states = layer_inputs[0]
            if input_states is not None and input_states.requires_grad:
                input_states.retain_grad()
                self.current_ffn_states[1].append(input_states)
            output_states = outputs
            if output_states is not None and output_states.requires_grad:
                output_states.retain_grad()
                self.current_hidden_states.append(output_states)
        
        for layer in range(self.num_layers):
            mha_layer = self.param_controller.get_layer(layer, 'enc_self_output')
            if mha_layer is not None:
                mha_handler = mha_layer.register_forward_hook(calculate_enc_mha_score)
                self.mha_handlers.append(mha_handler)
            mha_layer = self.param_controller.get_layer(layer, 'dec_self_output')
            if mha_layer is not None:
                mha_handler = mha_layer.register_forward_hook(calculate_dec_mha_score)
                self.mha_handlers.append(mha_handler)
            mha_layer = self.param_controller.get_layer(layer, 'cross_output')
            if mha_layer is not None:
                mha_handler = mha_layer.register_forward_hook(calculate_cross_mha_score)
                self.mha_handlers.append(mha_handler)
            ffn_layer = self.param_controller.get_layer(layer, 'encoder_io')
            if ffn_layer is not None:
                ffn_handler = ffn_layer.register_forward_hook(calculate_enc_ffn_score)
                self.ffn_handlers.append(ffn_handler)
            ffn_layer = self.param_controller.get_layer(layer, 'decoder_io')
            if ffn_layer is not None:
                ffn_handler = ffn_layer.register_forward_hook(calculate_dec_ffn_score)
                self.ffn_handlers.append(ffn_handler)

        self.hooks_registered = True
        
    def _gather_score(self):
        enc_mha_list, dec_mha_list, cross_mha_list = self.current_mha_states
        enc_ffn_list, dec_ffn_list = self.current_ffn_states
        self.current_mha_states = self.current_mha_states[0] + self.current_mha_states[1] + self.current_mha_states[2]
        self.current_ffn_states = self.current_ffn_states[0] + self.current_ffn_states[1]
        super()._gather_score()
        self.current_mha_states = [enc_mha_list, dec_mha_list, cross_mha_list]
        self.current_ffn_states = [enc_ffn_list, dec_ffn_list]
        for mha_list in self.current_mha_states:
            mha_list.clear()
        for ffn_list in self.current_ffn_states:
            ffn_list.clear()

class BackwardT5RunningHiddenStatesSalienceScorer(BackwardRunningHiddenStatesSalienceScorer):
    def __init__(self, *args, **kwargs):
        # Add hooks to the model (MHA, FFN, LayerNorm)
        # Add MHA hooks
        super().__init__(*args, **kwargs)
        self.current_mha_states = [[], [], []] # encoder, decoder, cross
        self.current_ffn_states = [[], []] # encoder, decoder
        self.all_head_scores = [[], [], []] # encoder, decoder, cross
        self.all_ffn_scores = [[], []] # encoder, decoder
        self.expected_hidden_cnt = 5 * self.num_layers + 1 # each layer's selfmha, (decdoer cross mha), and ffn outputs, plus embedding output

    def _register_hooks(self):
        print("Registering hooks in T5-like model with hidden states salience scorer...", flush=True)
        def cache_enc_mha_states(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]):
            # print("mha forward hook fired")
            hidden_states = outputs
            mha_hidden_states = layer_inputs[0]
            if mha_hidden_states is not None and mha_hidden_states.requires_grad:
                # print("mha state is not None and not zero")
                with torch.no_grad():
                    mha_state = mha_hidden_states.abs().sum(dim=0).mean(dim=0)
                    self.current_mha_states[0].append(mha_state)
                    self.current_hidden_states.append(hidden_states.abs().sum(dim=0).mean(dim=0))
            elif mha_hidden_states is None and hidden_states is not None and hidden_states.requires_grad:
                # print("mha state is None, is input state None?", hidden_states is None)
                with torch.no_grad():
                    self.current_hidden_states.append(None)

        def calculate_enc_mha_score(module: nn.Module, grad_layer_inputs, grad_outputs):
            # print("mha backward hook fired")
            hidden_states_grad = grad_outputs[0]
            mha_states = self.current_mha_states[0].pop()
            while not isinstance(mha_states, torch.Tensor):
                self.all_head_scores[0].append(mha_states[1])
                mha_states = self.current_mha_states[0].pop()
            mha_states_grad = grad_layer_inputs[0]
            hidden_states = self.current_hidden_states.pop()
            while hidden_states is None:
                hidden_states = self.current_hidden_states.pop()
            with torch.no_grad():
                seq_len = mha_states_grad.shape[1] # shape (batch_size, seq_len, hidden_size)
                salience = (mha_states * mha_states_grad.abs().sum(dim=0).mean(dim=0) * seq_len) if mha_states_grad is not None else torch.zeros_like(mha_states)
                head_salience = salience.view(-1, self.attention_head_size).sum(dim=1)
                self.all_head_scores[0].append(head_salience)
                if hidden_states_grad is not None:
                    salience = (hidden_states * hidden_states_grad.abs().sum(dim=0).mean(dim=0) * seq_len)
                    self.hidden_cnt += 1
                else:
                    salience = torch.zeros_like(hidden_states)
                self.all_hidden_scores += salience

        def cache_dec_mha_states(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]):
            # print("mha forward hook fired")
            hidden_states = outputs
            mha_hidden_states = layer_inputs[0]
            if mha_hidden_states is not None and mha_hidden_states.requires_grad:
                # print("mha state is not None and not zero")
                with torch.no_grad():
                    mha_state = mha_hidden_states.abs().sum(dim=0).mean(dim=0)
                    self.current_mha_states[1].append(mha_state)
                    self.current_hidden_states.append(hidden_states.abs().sum(dim=0).mean(dim=0))
            elif mha_hidden_states is None and hidden_states is not None and hidden_states.requires_grad:
                # print("mha state is None, is input state None?", input_states is None)
                with torch.no_grad():
                    self.current_hidden_states.append(None)
                    
        def calculate_dec_mha_score(module: nn.Module, grad_layer_inputs, grad_outputs):
            # print("mha backward hook fired")
            hidden_states_grad = grad_outputs[0]
            mha_states = self.current_mha_states[1].pop()
            while not isinstance(mha_states, torch.Tensor):
                self.all_head_scores[1].append(mha_states[1])
                mha_states = self.current_mha_states[1].pop()
            mha_states_grad = grad_layer_inputs[0]
            hidden_states = self.current_hidden_states.pop()
            while hidden_states is None:
                hidden_states = self.current_hidden_states.pop()
            with torch.no_grad():
                seq_len = mha_states_grad.shape[1]
                salience = (mha_states * mha_states_grad.abs().sum(dim=0).mean(dim=0) * seq_len) if mha_states_grad is not None else torch.zeros_like(mha_states)
                head_salience = salience.view(-1, self.attention_head_size).sum(dim=1)
                self.all_head_scores[1].append(head_salience)
                if hidden_states_grad is not None:
                    salience = (hidden_states * hidden_states_grad.abs().sum(dim=0).mean(dim=0) * seq_len)
                    self.hidden_cnt += 1
                else:
                    salience = torch.zeros_like(hidden_states)
                self.all_hidden_scores += salience

        def cache_cross_mha_states(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]):
            # print("mha forward hook fired")
            hidden_states = outputs
            mha_hidden_states = layer_inputs[0]
            if mha_hidden_states is not None and mha_hidden_states.requires_grad:
                # print("mha state is not None and not zero")
                with torch.no_grad():
                    mha_state = mha_hidden_states.abs().sum(dim=0).mean(dim=0)
                    self.current_mha_states[2].append(mha_state)
                    self.current_hidden_states.append(hidden_states.abs().sum(dim=0).mean(dim=0))
            elif mha_hidden_states is None and hidden_states is not None and hidden_states.requires_grad:
                # print("mha state is None, is input state None?", input_states is None)
                with torch.no_grad():
                    self.current_hidden_states.append(None)
        
        def calculate_cross_mha_score(module: nn.Module, grad_layer_inputs, grad_outputs):
            # print("mha backward hook fired")
            hidden_states_grad = grad_outputs[0]
            mha_states = self.current_mha_states[2].pop()
            while not isinstance(mha_states, torch.Tensor):
                self.all_head_scores[2].append(mha_states[1])
                mha_states = self.current_mha_states[2].pop()
            mha_states_grad = grad_layer_inputs[0]
            hidden_states = self.current_hidden_states.pop()
            while hidden_states is None:
                hidden_states = self.current_hidden_states.pop()
            with torch.no_grad():
                seq_len = mha_states_grad.shape[1]
                salience = (mha_states * mha_states_grad.abs().sum(dim=0).mean(dim=0) * seq_len) if mha_states_grad is not None else torch.zeros_like(mha_states)
                head_salience = salience.view(-1, self.attention_head_size).sum(dim=1)
                self.all_head_scores[2].append(head_salience)
                if hidden_states_grad is not None:
                    salience = (hidden_states * hidden_states_grad.abs().sum(dim=0).mean(dim=0) * seq_len)
                    self.hidden_cnt += 1
                else:
                    salience = torch.zeros_like(hidden_states)
                self.all_hidden_scores += salience
        
        def cache_enc_ffn_states(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: torch.Tensor):
            # print("ffn forward hook fired")
            hidden_states = outputs
            ffn_hidden_states = layer_inputs[0]
            if ffn_hidden_states is not None and ffn_hidden_states.requires_grad:
                # print("ffn state is not None and not zero")
                with torch.no_grad():
                    ffn_state = ffn_hidden_states.abs().sum(dim=0).mean(dim=0)
                    self.current_ffn_states[0].append(ffn_state)
                    self.current_hidden_states.append(hidden_states.abs().sum(dim=0).mean(dim=0))
            elif ffn_hidden_states is None and hidden_states is not None and hidden_states.requires_grad:
                # print("ffn state is None, is input state None?", input_states is None)
                with torch.no_grad():
                    self.current_hidden_states.append(None)

        def calculate_enc_ffn_score(module: nn.Module, grad_layer_inputs, grad_outputs):
            # print("ffn backward hook fired")
            hidden_states_grad = grad_outputs[0]
            ffn_states = self.current_ffn_states[0].pop()
            while not isinstance(ffn_states, torch.Tensor):
                self.all_ffn_scores[0].append(ffn_states[1])
                ffn_states = self.current_ffn_states[0].pop()
            ffn_states_grad = grad_layer_inputs[0]
            hidden_states = self.current_hidden_states.pop()
            while hidden_states is None:
                hidden_states = self.current_hidden_states.pop()
            with torch.no_grad():
                seq_len = ffn_states_grad.shape[1]
                neuron_salience = (ffn_states * ffn_states_grad.abs().sum(dim=0).mean(dim=0) * seq_len) if ffn_states_grad is not None else torch.zeros_like(ffn_states)
                self.all_ffn_scores[0].append(neuron_salience)
                if hidden_states_grad is not None:
                    salience = (hidden_states * hidden_states_grad.abs().sum(dim=0).mean(dim=0) * seq_len)
                    self.hidden_cnt += 1
                else:
                    salience = torch.zeros_like(hidden_states)
                self.all_hidden_scores += salience

        def cache_dec_ffn_states(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: torch.Tensor):
            # print("ffn forward hook fired")
            hidden_states = outputs
            ffn_hidden_states = layer_inputs[0]
            if ffn_hidden_states is not None and ffn_hidden_states.requires_grad:
                # print("ffn state is not None and not zero")
                with torch.no_grad():
                    ffn_state = ffn_hidden_states.abs().sum(dim=0).mean(dim=0)
                    self.current_ffn_states[1].append(ffn_state)
                    self.current_hidden_states.append(hidden_states.abs().sum(dim=0).mean(dim=0))
            elif ffn_hidden_states is None and hidden_states is not None and hidden_states.requires_grad:
                # print("ffn state is None, is input state None?", input_states is None)
                with torch.no_grad():
                    self.current_hidden_states.append(None)

        def calculate_dec_ffn_score(module: nn.Module, grad_layer_inputs, grad_outputs):
            # print("ffn backward hook fired")
            hidden_states_grad = grad_outputs[0]
            ffn_states = self.current_ffn_states[1].pop()
            while not isinstance(ffn_states, torch.Tensor):
                self.all_ffn_scores[1].append(ffn_states[1])
                ffn_states = self.current_ffn_states[1].pop()
            ffn_states_grad = grad_layer_inputs[0]
            hidden_states = self.current_hidden_states.pop()
            while hidden_states is None:
                hidden_states = self.current_hidden_states.pop()
            with torch.no_grad():
                seq_len = ffn_states_grad.shape[1]
                neuron_salience = (ffn_states * ffn_states_grad.abs().sum(dim=0).mean(dim=0) * seq_len) if ffn_states_grad is not None else torch.zeros_like(ffn_states)
                self.all_ffn_scores[1].append(neuron_salience)
                if hidden_states_grad is not None:
                    salience = (hidden_states * hidden_states_grad.abs().sum(dim=0).mean(dim=0) * seq_len)
                    self.hidden_cnt += 1
                else:
                    salience = torch.zeros_like(hidden_states)
                self.all_hidden_scores += salience

                
        for layer in range(self.num_layers):
            mha_layer = self.param_controller.get_layer(layer, 'enc_self_output')
            if mha_layer is not None:
                mha_handler = mha_layer.register_forward_hook(cache_enc_mha_states)
                self.mha_handlers.append(mha_handler)
                mha_handler = mha_layer.register_full_backward_hook(calculate_enc_mha_score)
                self.mha_handlers.append(mha_handler)
            mha_layer = self.param_controller.get_layer(layer, 'dec_self_output')
            if mha_layer is not None:
                mha_handler = mha_layer.register_forward_hook(cache_dec_mha_states)
                self.mha_handlers.append(mha_handler)
                mha_handler = mha_layer.register_full_backward_hook(calculate_dec_mha_score)
                self.mha_handlers.append(mha_handler)
            mha_layer = self.param_controller.get_layer(layer, 'cross_output')
            if mha_layer is not None:
                mha_handler = mha_layer.register_forward_hook(cache_cross_mha_states)
                self.mha_handlers.append(mha_handler)
                mha_handler = mha_layer.register_full_backward_hook(calculate_cross_mha_score)
                self.mha_handlers.append(mha_handler)
            ffn_layer = self.param_controller.get_layer(layer, 'encoder_io')
            if ffn_layer is not None:
                ffn_handler = ffn_layer.register_forward_hook(cache_enc_ffn_states)
                self.ffn_handlers.append(ffn_handler)
                ffn_handler = ffn_layer.register_full_backward_hook(calculate_enc_ffn_score)
                self.ffn_handlers.append(ffn_handler)
            ffn_layer = self.param_controller.get_layer(layer, 'decoder_io')
            if ffn_layer is not None:
                ffn_handler = ffn_layer.register_forward_hook(cache_dec_ffn_states)
                self.ffn_handlers.append(ffn_handler)
                ffn_handler = ffn_layer.register_full_backward_hook(calculate_dec_ffn_score)
                self.ffn_handlers.append(ffn_handler)
            
        self.hooks_registered = True
            
            
    def _gather_score(self):
        if not self.hooks_registered:
            self._register_hooks()
            return # Skip the first gathering step
        with torch.no_grad():            
            for v in self.all_head_scores: # reverse scores because in backward propagation, the last layer is calculated first
                v.reverse()
            for v in self.all_ffn_scores:
                v.reverse()
            # Check if there're any bottom layers' hs that does not require grad, and resulting in zero states and salience
            if self.model.virtual_pruned:
                # If virtually pruned, the salience scores should covers only the retained indices
                head_mask_use, intermediate_mask_use = self.model.split_mask_or_score(self.model.backup_head_mask, self.model.backup_intermediate_mask)
                expected_enc_mha_num = sum([(v != 0).sum().item() for v in head_mask_use[0]])
                expected_dec_mha_num = sum([(v != 0).sum().item() for v in head_mask_use[1]])
                expected_cross_mha_num = sum([(v != 0).sum().item() for v in head_mask_use[2]])
                expected_enc_ffn_num = sum([(v != 0).sum().item() for v in intermediate_mask_use[0]])
                expected_dec_ffn_num = sum([(v != 0).sum().item() for v in intermediate_mask_use[1]])
            else:
                # Else, the salience scores' shape should be the same as masks
                head_mask_use, intermediate_mask_use = self.model.split_mask_or_score(self.model.head_mask, self.model.intermediate_mask)
                expected_enc_mha_num = sum([v.shape[0] for v in head_mask_use[0]])
                expected_dec_mha_num = sum([v.shape[0] for v in head_mask_use[1]])
                expected_cross_mha_num = sum([v.shape[0] for v in head_mask_use[2]])
                expected_enc_ffn_num = sum([v.shape[0] for v in intermediate_mask_use[0]])
                expected_dec_ffn_num = sum([v.shape[0] for v in intermediate_mask_use[1]])

            # Padding the bottom layers' salience scores with zeros
            mha_shape_differences = [expected_enc_mha_num - sum([v.shape[0] for v in self.all_head_scores[0]]), expected_dec_mha_num - sum([v.shape[0] for v in self.all_head_scores[1]]), expected_cross_mha_num - sum([v.shape[0] for v in self.all_head_scores[2]])]
            ffn_shape_differences = [expected_enc_ffn_num - sum([v.shape[0] for v in self.all_ffn_scores[0]]), expected_dec_ffn_num - sum([v.shape[0] for v in self.all_ffn_scores[1]])]
            self.all_head_scores[0] = ([torch.zeros(mha_shape_differences[0], device=self.model.device)] if mha_shape_differences[0] > 0 else []) + self.all_head_scores[0]
            self.all_head_scores[1] = ([torch.zeros(mha_shape_differences[1], device=self.model.device)] if mha_shape_differences[1] > 0 else []) + self.all_head_scores[1]
            self.all_head_scores[2] = ([torch.zeros(mha_shape_differences[2], device=self.model.device)] if mha_shape_differences[2] > 0 else []) + self.all_head_scores[2]
            self.all_ffn_scores[0] = ([torch.zeros(ffn_shape_differences[0], device=self.model.device)] if ffn_shape_differences[0] > 0 else []) + self.all_ffn_scores[0]
            self.all_ffn_scores[1] = ([torch.zeros(ffn_shape_differences[1], device=self.model.device)] if ffn_shape_differences[1] > 0 else []) + self.all_ffn_scores[1]
            
            # Enforce the top-salient top-decoder-cross head to be retained
            self.all_head_scores[2][-1][self.all_head_scores[2][-1].argmax()] = 1e10
            
            self.all_head_scores =self.all_head_scores[0] + self.all_head_scores[1] + self.all_head_scores[2]
            self.all_ffn_scores = self.all_ffn_scores[0] + self.all_ffn_scores[1]
            
            if not len(self.all_head_scores) and not len(self.all_ffn_scores):
                print("No salience is calculated in this step. Skip score gathering", flush=True)
                return
            current_salience_dict = {
                'head_mask': torch.cat(self.all_head_scores),
                'intermediate_mask': torch.cat(self.all_ffn_scores),
                # 'hidden_mask': self.all_hidden_scores / self.hidden_cnt * self.expected_hidden_cnt,
                'hidden_mask': self.all_hidden_scores,
            }
            if not (all([len(v) == 0 for v in self.current_mha_states]) and all([len(v) == 0 for v in self.current_ffn_states]) and len(self.current_hidden_states) == 0):
                # make sure all states are popped so there's no mismatch
                raise RuntimeError("Not all states are popped. Length of current states: {}, {}, {}".format([len(v) for v in self.current_mha_states], [len(v) for v in self.current_ffn_states], len(self.current_hidden_states)))
            
            self.hidden_cnt = 0
            for m in ['head_mask', 'intermediate_mask', 'hidden_mask']:
                current_salience = current_salience_dict[m]
                retained_indices = self.retained_indices.get(m, None)
                if retained_indices is not None and self.model.virtual_pruned:
                    if self.block_normalize_dict is not None:
                        current_salience *= self.block_normalize_dict[m]
                    if self.use_uncertainty and not self.static:
                        current_uncertainty = (current_salience - self.salience_dict[m]['s'][retained_indices]).abs()
                        self.salience_dict[m]['u'][retained_indices] = self.salience_dict[m]['u'][retained_indices] * self.beta_2 + (1 - self.beta_2) * current_uncertainty
                    if self.static:
                        self.salience_dict[m]['s'][retained_indices] += current_salience
                    else:
                        self.salience_dict[m]['s'][retained_indices] = self.salience_dict[m]['s'][retained_indices] * self.beta_1 + (1 - self.beta_1) * current_salience
                else:
                    if self.block_normalize_dict is not None:
                        current_salience *= self.block_normalize_dict[m]
                    if self.use_uncertainty and not self.static:
                        current_uncertainty = (current_salience - self.salience_dict[m]['s']).abs()
                        self.salience_dict[m]['u'] = self.salience_dict[m]['u'] * self.beta_2 + (1 - self.beta_2) * current_uncertainty
                    if self.static:
                        self.salience_dict[m]['s'] += current_salience
                    else:
                        self.salience_dict[m]['s'] = self.salience_dict[m]['s'] * self.beta_1 + (1 - self.beta_1) * current_salience
            self._gather_module_score()
        self.all_head_scores = [[], [], []] # encoder, decoder, cross
        self.all_ffn_scores = [[], []] # encoder, decoder
        self.all_hidden_scores = 0


class BackwardLlamaRunningHiddenStatesSalienceScorer(BackwardRunningHiddenStatesSalienceScorer):
    def __init__(self, *args, **kwargs):
        # Add hooks to the model (MHA, FFN, LayerNorm)
        # Add MHA hooks
        super().__init__(*args, **kwargs)
        self.expected_hidden_cnt = 2 * self.num_layers + 1 # each layer's selfmha, (decdoer cross mha), and ffn outputs, plus embedding output
        # self.all_head_saliences = []
        # self.all_ffn_saliences = []
        # self.mha_kurtosis = []
        # self.ffn_kurtosis = []
        # self.mha_kurtosis_history = []
        # self.ffn_kurtosis_history = []

    def _register_hooks(self):
        print("Registering hooks in LLaMA like model with hidden states salience scorer...", flush=True)
        def cache_mha_states(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]):
            # print("mha forward hook fired")
            hidden_states = outputs
            mha_hidden_states = layer_inputs[0]
            if mha_hidden_states is not None and mha_hidden_states.requires_grad:
                # print("mha state is not None and not zero")
                with torch.no_grad():
                    mha_state = mha_hidden_states.abs().sum(dim=0).sum(dim=0)
                    self.current_mha_states.append(mha_state)
                    self.current_hidden_states.append(hidden_states.abs().sum(dim=0).sum(dim=0))
                    # mha_hidden_states shape: (batch_size, seq_len, num_heads x head_size)
                    # module weight shape: (hidden_size, num_heads x head_size)
                    mha_hidden_unsqueezed = mha_hidden_states.mean(0).mean(0).unsqueeze(0) # (1, num_heads x head_size)
                    mha_hidden_unsqueezed = mha_hidden_unsqueezed.view(1, -1, self.attention_head_size) # Convert to (1, num_heads, head_size)
                    
                    weight_merged = (module.weight + (module.lora_B @ module.lora_A) * module.scaling) if isinstance(module, lora.Linear) else module.weight # shape: (hidden_size, num_heads x head_size)
                    weight_unsqueezed = weight_merged.view(weight_merged.shape[0], -1, self.attention_head_size) # Convert to (hidden_size, num_heads, head_size)
                    activation = (mha_hidden_unsqueezed * weight_unsqueezed) # shape: (hidden size, num_heads, head_size) 
                    act_kurtosis = kurtosis(activation.permute(1, 2, 0).reshape(-1, activation.shape[1]))
                    self.all_mha_kurtosis.append(act_kurtosis)
                    
                    # Calculating hidden states kurtosis
                    hidden_kurtosis = kurtosis(activation.view(activation.shape[0], -1).permute(1, 0))
                    self.all_hidden_kurtosis += hidden_kurtosis.clamp(min=0).sqrt()
                    if self.log_kurtosis:
                        self.all_hidden_kurtosis_log.append(hidden_kurtosis)
            elif mha_hidden_states is None and hidden_states is not None and hidden_states.requires_grad:
                # print("mha state is None, is input state None?", input_states is None)
                with torch.no_grad():
                    self.current_hidden_states.append(None)
                    
        def calculate_mha_score(module: nn.Module, grad_layer_inputs, grad_outputs):
            # print("mha backward hook fired")
            hidden_states_grad = grad_outputs[0]
            mha_states = self.current_mha_states.pop()
            while not isinstance(mha_states, torch.Tensor):
                self.all_head_scores.append(mha_states[1])
                mha_states = self.current_mha_states.pop()
            mha_states_grad = grad_layer_inputs[0]
            hidden_states = self.current_hidden_states.pop()
            while hidden_states is None:
                hidden_states = self.current_hidden_states.pop()
            with torch.no_grad():
                salience = (mha_states * mha_states_grad.abs().sum(dim=0).sum(dim=0)) if mha_states_grad is not None else torch.zeros_like(mha_states)
                head_salience = salience.view(-1, self.attention_head_size).sum(dim=1)
                self.all_head_scores.append(head_salience)
                if hidden_states_grad is not None:
                    salience = (hidden_states * hidden_states_grad.abs().sum(dim=0).sum(dim=0))
                    self.hidden_cnt += 1
                else:
                    salience = torch.zeros_like(hidden_states)
                self.all_hidden_scores += salience

        def cache_ffn_states(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: torch.Tensor):
            # print("ffn forward hook fired")
            hidden_states = outputs
            ffn_hidden_states = layer_inputs[0]
            if ffn_hidden_states is not None and ffn_hidden_states.requires_grad:
                # print("ffn state is not None and not zero")
                with torch.no_grad():
                    ffn_state = ffn_hidden_states.abs().sum(dim=0).sum(dim=0)
                    self.current_ffn_states.append(ffn_state)
                    self.current_hidden_states.append(hidden_states.abs().sum(dim=0).sum(dim=0))
                    # ffn_hidden_states shape: (batch_size, seq_len, num_neurons)
                    # module weight shape: (hidden_size, num_neurons)
                    ffn_hidden_unsqueezed = ffn_hidden_states.mean(0).mean(0).unsqueeze(0) # (1,  num_neurons)
                    weight_merged = (module.weight + (module.lora_B @ module.lora_A) * module.scaling) if isinstance(module, lora.Linear) else module.weight # shape: (hidden_size, num_neurons)
                    # Convert to (hidden_size num_neurons)
                    activation = (ffn_hidden_unsqueezed * weight_merged) # shape: (hidden size, num_neurons) 
                    act_kurtosis = kurtosis(activation)
                    self.all_ffn_kurtosis.append(act_kurtosis)
                    
                    # Calculating hidden states kurtosis
                    hidden_kurtosis = kurtosis(activation.permute(1, 0))
                    self.all_hidden_kurtosis += hidden_kurtosis.clamp(min=0).sqrt()
                    if self.log_kurtosis:
                        self.all_hidden_kurtosis_log.append(hidden_kurtosis)
            elif ffn_hidden_states is None and hidden_states is not None and hidden_states.requires_grad:
                # print("ffn state is None, is input state None?", input_states is None)
                with torch.no_grad():
                    self.current_hidden_states.append(None)

        def calculate_ffn_score(module: nn.Module, grad_layer_inputs, grad_outputs):
            # print("ffn backward hook fired")
            hidden_states_grad = grad_outputs[0]
            ffn_states = self.current_ffn_states.pop()
            while not isinstance(ffn_states, torch.Tensor):
                self.all_ffn_scores.append(ffn_states[1])
                ffn_states = self.current_ffn_states.pop()
            ffn_states_grad = grad_layer_inputs[0]
            hidden_states = self.current_hidden_states.pop()
            while hidden_states is None:
                hidden_states = self.current_hidden_states.pop()
            with torch.no_grad():
                neuron_salience = (ffn_states * ffn_states_grad.abs().sum(dim=0).sum(dim=0)) if ffn_states_grad is not None else torch.zeros_like(ffn_states)
                self.all_ffn_scores.append(neuron_salience)
                if hidden_states_grad is not None:
                    salience = (hidden_states * hidden_states_grad.abs().sum(dim=0).sum(dim=0))
                    self.hidden_cnt += 1
                else:
                    salience = torch.zeros_like(hidden_states)
                self.all_hidden_scores += salience

                
        for layer in range(self.num_layers):
            mha_layer = self.param_controller.get_layer(layer, 'dec_self_output')
            if mha_layer is not None:
                mha_handler = mha_layer.register_forward_hook(cache_mha_states)
                self.mha_handlers.append(mha_handler)
                mha_handler = mha_layer.register_full_backward_hook(calculate_mha_score)
                self.mha_handlers.append(mha_handler)
            ffn_layer = self.param_controller.get_layer(layer, 'decoder_io')
            if ffn_layer is not None:
                ffn_handler = ffn_layer.register_forward_hook(cache_ffn_states)
                self.ffn_handlers.append(ffn_handler)
                ffn_handler = ffn_layer.register_full_backward_hook(calculate_ffn_score)
                self.ffn_handlers.append(ffn_handler)
            
        self.hooks_registered = True
        
    # def _gather_score(self):
    #     if len(self.all_head_scores):
    #         self.all_head_saliences.append(torch.stack(self.all_head_scores[::-1]))
    #         self.all_ffn_saliences.append(torch.stack(self.all_ffn_scores[::-1]))
    #     super()._gather_score()
    #     if len(self.mha_kurtosis):
    #         self.mha_kurtosis_history.append(torch.stack(self.mha_kurtosis))
    #         self.ffn_kurtosis_history.append(torch.stack(self.ffn_kurtosis))
    #         self.mha_kurtosis = []
    #         self.ffn_kurtosis = []
            
class RunningHiddenStatesMagnitudeScorer(RunningSalienceScorer):
    def __init__(self, model: PreTrainedModel, param_controller: ParamController, state: TrainerState, dataloader: Optional[DataLoader] = None, gather_freq: int = 1, beta_1: float = 0.85, beta_2: float = 0.85, use_uncertainty: bool = False, block_normalize_dict: Optional[Dict[str, float]] = None, **kwargs):
        # Add hooks to the model (MHA, FFN, LayerNorm)
        # Add MHA hooks
        self.mha_handlers = []
        self.ffn_handlers = []
        self.hidden_handlers = []
        self.current_score_dict = {
            'head_mask': [],
            'intermediate_mask': [],
            'hidden_mask': 0,
        }
        self.hooks_registered = False
        super().__init__(model, param_controller, state, dataloader, gather_freq, beta_1, beta_2, use_uncertainty, block_normalize_dict)

    def _register_hooks(self):
        def calculate_mha_score(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]):
            mha_hidden_states = outputs[0]
            input_states = layer_inputs[0]
            with torch.no_grad():
                if mha_hidden_states is not None:
                    mha_score = mha_hidden_states.abs().sum(dim=0).sum(dim=0).view(-1, self.attention_head_size).sum(dim=1)
                    self.current_score_dict['head_mask'].append(mha_score)
                if input_states is not None:
                    hidden_score = input_states.abs().sum(dim=0).sum(dim=0)
                    self.current_score_dict['hidden_mask'] += hidden_score
        
        def calculate_ffn_score(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: torch.Tensor):
            ffn_hidden_states = outputs
            input_states = layer_inputs[0]
            with torch.no_grad():
                if ffn_hidden_states is not None:
                    ffn_score = ffn_hidden_states.abs().sum(dim=0).sum(dim=0)
                    self.current_score_dict['intermediate_mask'].append(ffn_score)
                if input_states is not None:
                    hidden_score = input_states.abs().sum(dim=0).sum(dim=0)
                    self.current_score_dict['hidden_mask'] += hidden_score
            
        def model_output_hook(module: nn.Module, layer_inputs: Tuple[torch.Tensor], outputs: torch.Tensor):
            output_hidden_states = outputs[0]
            with torch.no_grad():
                if output_hidden_states is not None:
                    hidden_score = output_hidden_states.abs().sum(dim=0).sum(dim=0)
                    self.current_score_dict['hidden_mask'] += hidden_score
        
        for layer in range(self.num_layers):
            mha_layer = self.param_controller.get_parent_layer(layer, 'query')
            if mha_layer is not None:
                mha_handler = mha_layer.register_forward_hook(calculate_mha_score)
                self.mha_handlers.append(mha_handler)
            
            ffn_layer = self.param_controller.get_parent_layer(layer, 'intermediate')
            if ffn_layer is not None:
                ffn_handler = self.param_controller.get_parent_layer(layer, 'intermediate').register_forward_hook(calculate_ffn_score)
                self.ffn_handlers.append(ffn_handler)

        self.hidden_handlers.append(getattr(self.model,self.model.base_model_prefix).encoder.register_forward_hook(model_output_hook))
        self.hooks_registered = True
            
            
    def _gather_score(self):
        if not self.hooks_registered:
            self._register_hooks()
            return # Skip the first gathering step
        with torch.no_grad():
            self.current_score_dict['head_mask'] = torch.cat(self.current_score_dict['head_mask'])
            self.current_score_dict['intermediate_mask'] = torch.cat(self.current_score_dict['intermediate_mask'])

            for m in ['head_mask', 'intermediate_mask', 'hidden_mask']:
                current_salience = self.current_score_dict[m]
                if self.block_normalize_dict is not None:
                    current_salience *= self.block_normalize_dict[m]
                if self.use_uncertainty:
                    current_uncertainty = (current_salience - self.salience_dict[m]['s']).abs()
                    self.salience_dict[m]['u'] = self.salience_dict[m]['u'] * self.beta_2 + (1 - self.beta_2) * current_uncertainty
                self.salience_dict[m]['s'] = self.salience_dict[m]['s'] * self.beta_1 + (1 - self.beta_1) * current_salience
            self.current_score_dict = {
                'head_mask': [],
                'intermediate_mask': [],
                'hidden_mask': 0,
            }
            self._gather_module_score()
            
    def _remove_hooks(self):
        for handle in self.mha_handlers:
            handle.remove()
        for handle in self.ffn_handlers:
            handle.remove()
        for handle in self.hidden_handlers:
            handle.remove()
        self.mha_handlers = []
        self.ffn_handlers = []
        self.hidden_handlers = []
        self.hooks_registered = False
        
    def _reset_hooks(self):
        self._remove_hooks()
        self._register_hooks()
        
    def end(self):
        self._remove_hooks()
        
    def reset_module_scores(self):
        super().reset_module_scores()
        if len(self.mha_handlers) or len(self.ffn_handlers) or len(self.hidden_handlers):
            self._reset_hooks() # When salience is pruned, it means the model layers are changed, so we need to re-register the hooks