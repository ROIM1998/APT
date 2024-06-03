import os
import json
import dataclasses
import torch
import transformers
import numpy as np
import loralib as lora
import torch.nn as nn
from typing import Dict, List, Tuple
from models.modeling_bert import CoFiBertForSequenceClassification, CoFiBertLayer


def get_select_hidden_dim_states(hidden_states: Tuple[torch.Tensor], layer: int, select_indices: torch.Tensor) -> torch.Tensor:
    states = hidden_states[layer+1]
    if states.device != 'cpu':
        states = states.cpu().clone()
    return states.index_select(2, select_indices)    


def get_pruned_and_retained_hiddens(outputs: transformers.modeling_outputs.SequenceClassifierOutput, zs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    returned_dict = {}
    hidden_states= outputs['hidden_states']
    layer_num = len(hidden_states) - 1
    # Get hidden states of those pruned hidden dimensions
    hidden_z = zs['hidden_z']
    pruned_hiddens, retained_hiddens = torch.where(hidden_z == 0)[0], torch.where(hidden_z == 1)[0]

    # Get embeddings of the pruned and retained hidden dimensions
    embedding_pruned_hiddens = get_select_hidden_dim_states(hidden_states, -1, pruned_hiddens)
    embedding_retained_hiddens = get_select_hidden_dim_states(hidden_states, -1, retained_hiddens)
    returned_dict['pruned_embedding'] = [embedding_pruned_hiddens]
    returned_dict['retained_embedding'] = [embedding_retained_hiddens]

    # Get each layer's output of the pruned and retained hidden dimensions
    pruned_layer_hiddens = []
    retained_layer_hiddens = []
    for layer in range(layer_num):
        layer_pruned_hiddens, layer_retained_hiddens = get_select_hidden_dim_states(hidden_states, layer, pruned_hiddens), get_select_hidden_dim_states(hidden_states, layer, retained_hiddens)
        pruned_layer_hiddens.append(layer_pruned_hiddens)
        retained_layer_hiddens.append(layer_retained_hiddens)
    returned_dict['pruned_layer_hiddens'] = pruned_layer_hiddens
    returned_dict['retained_layer_hiddens'] = retained_layer_hiddens
    return returned_dict


def get_select_attention(attentions: Tuple[torch.Tensor], layer: int, head_indices: torch.Tensor) -> torch.Tensor:
    attention = attentions[layer]
    if not attention.device == 'cpu':
        attention = attention.cpu().clone()
    return attention.index_select(1, head_indices)


def get_pruned_and_retained_heads(outputs: transformers.modeling_outputs.SequenceClassifierOutput, zs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    attentions = outputs['attentions']
    head_z, head_layer_z = zs['head_z'], zs['head_layer_z']
    head_layer_mask = head_z.T.mul(head_layer_z).T
    hidden_states = outputs['hidden_states']
    layer_num = len(hidden_states) - 1
    pruned_attentions, retained_attentions = [], []
    for layer in range(layer_num):
        layer_mask = head_layer_mask[layer].cpu().clone()
        pruned_heads, retained_heads = torch.where(layer_mask == 0)[0], torch.where(layer_mask == 1)[0]
        heads_pruned_states, heads_retained_states = get_select_attention(attentions, layer, pruned_heads), get_select_attention(attentions, layer, retained_heads)
        pruned_attentions.append(heads_pruned_states)
        retained_attentions.append(heads_retained_states)
    returned_dict = {
        'pruned_attentions': pruned_attentions,
        'retained_attentions': retained_attentions,
    }
    return returned_dict


def get_attention_output(model: CoFiBertForSequenceClassification, hidden_states: Tuple[torch.Tensor], attention_mask: torch.Tensor, layer: int):
    bert_layer: CoFiBertLayer = model.bert.encoder.layer[layer]
    if bert_layer.training:
        bert_layer.eval()
    hidden = hidden_states[layer]
    attention_mask = model.bert.get_extended_attention_mask(
        attention_mask,
        hidden.shape[:-1],
        model.device,
    )
    self_attention_outputs = bert_layer.attention(
        hidden, # since the zero-th hidden_states is from the embedding layer, each hidden_states actually corresponds to its layer's input by index
        attention_mask,
    ) # returns a tuple with the size of 1, if not output attentions
    attention_output = self_attention_outputs[0]
    return attention_output


def get_intermediate(model: CoFiBertForSequenceClassification, hidden_states: Tuple[torch.Tensor], attention_mask: torch.Tensor, layer: int) -> torch.Tensor:
    bert_layer: CoFiBertLayer = model.bert.encoder.layer[layer]
    if bert_layer.training:
        bert_layer.eval()
    attention_output = get_attention_output(model, hidden_states, attention_mask, layer)
    intermediate_output = bert_layer.intermediate(attention_output)
    return intermediate_output


def test_intermediate(model, hidden_states, attention_mask ,layer):
    bert_layer: CoFiBertLayer = model.bert.encoder.layer[layer]
    bert_layer.eval()
    attention_output = get_attention_output(model, hidden_states, attention_mask ,layer)
    intermediate_output = get_intermediate(model, hidden_states, attention_mask, layer)
    layer_output = bert_layer.output(
        intermediate_output,
        attention_output,
        mlp_z=None,
        hidden_z=None
    )
    return (layer_output == hidden_states[layer+1]).all(), layer_output


def get_pruned_and_retained_intermediate(model: CoFiBertForSequenceClassification, hidden_states: Tuple[torch.Tensor], attention_mask: torch.Tensor, zs: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
    intermediate_z, mlp_z = zs['intermediate_z'], zs['mlp_z']
    layer_num = len(hidden_states) - 1
    intermediate_mask = intermediate_z.T.mul(mlp_z).T
    pruned_intermediate_states, retained_intermediate_states = [], []
    for layer in range(layer_num):
        intermediate_hidden = get_intermediate(model, hidden_states, attention_mask, layer).cpu()
        pruned_intermediate, retained_intermediate = torch.where(intermediate_mask[layer] == 0)[0], torch.where(intermediate_mask[layer] == 1)[0]
        pruned_intermediate_state, retained_intermediate_state = intermediate_hidden.index_select(2, pruned_intermediate), intermediate_hidden.index_select(2, retained_intermediate)
        pruned_intermediate_states.append(pruned_intermediate_state)
        retained_intermediate_states.append(retained_intermediate_state)
    return {
        'pruned_intermediate_states': pruned_intermediate_states,
        'retained_intermediate_states': retained_intermediate_states,
    }
    
def get_pruned_and_retained(model, inputs, zs):
    outputs = model(**inputs, return_dict=True, output_hidden_states=True, output_attentions=True)
    hiddens = get_pruned_and_retained_hiddens(outputs, zs)
    hiddens = {**hiddens, **get_pruned_and_retained_heads(outputs, zs)}
    hiddens = {**hiddens, **get_pruned_and_retained_intermediate(model, outputs['hidden_states'], inputs['attention_mask'], zs)}
    hiddens['mask'] = inputs['attention_mask']
    return hiddens


def gen_run_report(save_path):
    if os.path.exists(os.path.join(save_path, 'eval_results.json')):
        eval_results = json.load(open(os.path.join(save_path, 'eval_results.json')))
    else:
        eval_results = {}
    if os.path.exists(os.path.join(save_path, 'best_model')) and os.path.exists(os.path.join(save_path, 'best_model', 'eval_results.json')):
        best_eval_results = json.load(open(os.path.join(save_path, 'best_model', 'eval_results.json')))
    else:
        best_eval_results = eval_results
    train_results = json.load(open(os.path.join(save_path, 'train_results.json')))
    if os.path.exists(os.path.join(save_path, 'efficiency_results.json')):
        efficiency_results = json.load(open(os.path.join(save_path, 'efficiency_results.json')))
    else:
        efficiency_results = {}
    results = {**eval_results, **train_results, **efficiency_results}
    if 'eval_accuracy' in best_eval_results:
        results['eval_accuracy'] = best_eval_results['eval_accuracy']
    elif 'eval_pearson' in best_eval_results:
        results['eval_pearson'] = best_eval_results['eval_pearson']
    if 'best_epoch' in best_eval_results:
        results['best_epoch'] = best_eval_results['epoch']
    trainer_state = json.load(open(os.path.join(save_path, 'trainer_state.json')))
    log_history = trainer_state['log_history']
    mem_use = [log['step_end_mem'] for log in log_history if 'step_end_mem' in log if log['step_end_mem'] is not None]
    if len(mem_use) > 0:
        results['peak_mem'] = max(mem_use)
        results['avg_mem'] = sum(mem_use) / len(mem_use)
    else:
        results['peak_mem'] = None
    if 'best_poch' in results:
        if results['best_epoch'] == train_results['epoch']:
            results['time_to_best'] = train_results['train_runtime']
        else:
            results['time_to_best'] = [v['training_time'] for v in log_history if v['epoch'] == results['best_epoch'] and 'training_time' in v][0]
    training_args = torch.load(os.path.join(save_path, 'training_args.bin'))
    results['args'] = dataclasses.asdict(training_args)
    return results

def get_all_logs(root_dir: str):
    results = {}
    for root, _, files in os.walk(root_dir):
        if not root.endswith('_model') and 'eval_results.json' in files:
            results[root] = gen_run_report(root)
            results[root]['path'] = root
    return results