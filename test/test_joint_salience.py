import seaborn as sns
sns.set_theme(style="darkgrid")

import torch
import pandas as pd

from utils.utils import *
from matplotlib import pyplot as plt
from utils.fisher_utils.efficiency.param import *

def main():

    salience = torch.load('output/bert-base-uncased_lora_minus_rte_cubic_gradual_running_fisher_alloc_running_fisher_momentum_mapping_static_teacher_dynamic_cofi_student_distill_tophalf_limited_originaltest/mac0.4/epoch120/bz32/numprune5/paramq:0-11,v:0-11,i:0-11/lora_r4/pruning_start-1/distill_epoch96/first_salience.pt', map_location='cpu')
    new_salience = torch.load('output/bert-base-uncased_lora_minus_rte_cubic_gradual_running_fisher_alloc_running_fisher_momentum_mapping_static_teacher_dynamic_cofi_student_distill_tophalf_limited_nomainmask_mean/mac0.4/epoch120/bz32/numprune5/paramq:0-11,v:0-11,i:0-11/lora_r4/pruning_start-1/distill_epoch96/first_salience.pt', map_location='cpu')
    
    attention_head_size = 64
    num_layers = 12
    
    salience['score']['head_mask'].mean() / salience['score']['intermediate_mask'].mean()
    salience['score']['head_mask'].mean() / salience['score']['hidden_mask'].mean()
    
    new_salience['score']['head_mask'].mean() / new_salience['score']['intermediate_mask'].mean()
    new_salience['score']['head_mask'].mean() / new_salience['score']['hidden_mask'].mean()
    
    queries = [new_salience['salience']['modules'][layer]['query']['output_mask']['s'].view(-1, attention_head_size).abs().sum(dim=1) for layer in range(num_layers) if isinstance(new_salience['salience']['modules'][layer]['query']['output_mask']['s'], torch.Tensor)]
    values = [new_salience['salience']['modules'][layer]['value']['output_mask']['s'].view(-1, attention_head_size).abs().sum(dim=1) for layer in range(num_layers) if isinstance(new_salience['salience']['modules'][layer]['value']['output_mask']['s'], torch.Tensor)]
    query_tuning_scores = torch.cat(queries)
    value_tuning_scores = torch.cat(values)
    calculated_head_pseudo_salience = (query_tuning_scores + value_tuning_scores)
    
    intermediates = [new_salience['salience']['modules'][layer]['intermediate']['output_mask']['s'].abs() for layer in range(12) if isinstance(new_salience['salience']['modules'][layer]['intermediate']['output_mask']['s'], torch.Tensor)]
    calculated_intermediate_pseudo_salience = torch.cat(intermediates)
    
    hiddens = [
        val['input_mask']['s']
        for v in new_salience['salience']['modules'].values() for val in v.values() if isinstance(val['input_mask']['s'], torch.Tensor)
    ]
    calculated_hidden_pseudo_salience = torch.stack(hiddens).sum(dim=0)
    """
    Pruning score:
        head: 0.0003
        intermediate: 2.8067e-06
        hidden: 0.0048
    
    Tuning score:
        head query: 0.0008
        head value: 0.0036
        intermediate: 5.6331e-05
        hidden: 0.0031
    """
    
    # Plotting the correlation between main mask saliences and LoRA mask saliences
    plt.clf()
    hiddens = [
        val['input_mask']['s']
        for v in salience['salience']['modules'].values() for val in v.values() if isinstance(val['input_mask']['s'], torch.Tensor)
    ]
    hidden_pseudo_salience = torch.stack(hiddens).abs().sum(dim=0)
    
    queries = [salience['salience']['modules'][layer]['query']['output_mask']['s'].view(-1, attention_head_size).abs().sum(dim=1) for layer in range(num_layers) if isinstance(salience['salience']['modules'][layer]['query']['output_mask']['s'], torch.Tensor)]
    values = [salience['salience']['modules'][layer]['value']['output_mask']['s'].view(-1, attention_head_size).abs().sum(dim=1) for layer in range(num_layers) if isinstance(salience['salience']['modules'][layer]['value']['output_mask']['s'], torch.Tensor)]
    query_tuning_scores = torch.cat(queries)
    value_tuning_scores = torch.cat(values)
    head_pseudo_salience = (query_tuning_scores + value_tuning_scores)
    
    intermediates = [salience['salience']['modules'][layer]['intermediate']['output_mask']['s'].abs() for layer in range(12) if isinstance(salience['salience']['modules'][layer]['intermediate']['output_mask']['s'], torch.Tensor)]
    intermediate_pseudo_salience = torch.cat(intermediates)
    
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 12))
    sns.scatterplot(x=salience['salience']['head_mask']['s'].log10(), y=head_pseudo_salience.log10(), label='head', ax=axs[0])
    sns.scatterplot(x=salience['salience']['intermediate_mask']['s'].log10(), y=intermediate_pseudo_salience.log10(), label='intermediate', ax=axs[1])
    sns.scatterplot(x=salience['salience']['hidden_mask']['s'].log10(), y=hidden_pseudo_salience.log10(), label='hidden', ax=axs[2])
    plt.savefig('salience_correlation.png')
    print("Head mask mean scale", salience['salience']['head_mask']['s'].mean() / head_pseudo_salience.mean())
    print("Intermediate mask mean scale", salience['salience']['intermediate_mask']['s'].mean() / intermediate_pseudo_salience.mean())
    print("Hidden mask mean scale", salience['salience']['hidden_mask']['s'].mean() / hidden_pseudo_salience.mean())
    """
        RTE head: 1.9992
        intermediate neurons: 6.0920
        hidden: 13.4282
        
        
    """
    
    plt.clf()
    # Plotting query, value, and intermediates' input and output mask salience distributions
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 16))
    df_data = []
    for layer in salience['grafting_mask_salience']['modules']:
        for mask_type in ['input_mask', 'output_mask']:
            for val in salience['grafting_mask_salience']['modules'][layer]['query'][mask_type]['s'].log10().tolist():
                df_data.append({
                    'layer': layer,
                    'mask_type': mask_type,
                    'salience': val,
                })
    df_query = pd.DataFrame(df_data)
    sns.violinplot(data=df_query, x="layer", y="salience", hue="mask_type", split=True, ax=axs[0])
    axs[0].set_title('Query Mask Salience')
    axs[0].set_ylim(-8, -1)
    
    df_data = []
    for layer in salience['grafting_mask_salience']['modules']:
        for mask_type in ['input_mask', 'output_mask']:
            for val in salience['grafting_mask_salience']['modules'][layer]['value'][mask_type]['s'].log10().tolist():
                df_data.append({
                    'layer': layer,
                    'mask_type': mask_type,
                    'salience': val,
                })
    df_value = pd.DataFrame(df_data)
    sns.violinplot(data=df_value, x="layer", y="salience", hue="mask_type", split=True, ax=axs[1])
    axs[1].set_title('Value Mask Salience')
    axs[1].set_ylim(-8, -1)
    
    df_data = []
    for layer in salience['grafting_mask_salience']['modules']:
        for mask_type in ['input_mask', 'output_mask']:
            for val in salience['grafting_mask_salience']['modules'][layer]['intermediate'][mask_type]['s'].log10().tolist():
                df_data.append({
                    'layer': layer,
                    'mask_type': mask_type,
                    'salience': val,
                })
    df_neuron = pd.DataFrame(df_data)
    sns.violinplot(data=df_neuron, x="layer", y="salience", hue="mask_type", split=True, ax=axs[2])
    axs[2].set_title('Intermediate Mask Salience')
    axs[2].set_ylim(-8, -1)
    
    df_data = []
    for layer in salience['grafting_mask_salience']['modules']:
        for mask_type in ['head_mask', 'intermediate_mask']:
            for layer in range(12):
                for val in salience['model_mask_salience']['salience'][mask_type]['s'].view(12, -1)[layer].log10().tolist():
                    df_data.append({
                        'layer': layer,
                        'mask_type': mask_type,
                        'salience': val,
                    })
    df_neuron = pd.DataFrame(df_data)
    sns.violinplot(data=df_neuron, x="layer", y="salience", hue="mask_type", split=True, ax=axs[3])
    axs[2].set_title('Intermediate Mask Salience')
    axs[2].set_ylim(-8, -1)
    
    
    plt.savefig('salience_distributions.png')
    
    attention_head_size = 64
    param_per_block =  {'head_mask': 196800, 'intermediate_mask': 1537, 'hidden_mask': 110664}
    original_pruning_scores = [salience['model_mask_salience']['salience'][m]['s'] / param_per_block[m] for m in ['head_mask', 'intermediate_mask', 'hidden_mask']]
    score_types = [torch.ones_like(original_pruning_scores[i]) for i in range(3)]
    original_pruning_scores, score_types = torch.cat(original_pruning_scores), torch.cat(score_types)
    sorted_scores, sorted_indices = torch.sort(original_pruning_scores, descending=False)
    sorted_score_types = score_types[sorted_indices]
    
    query_tuning_scores = torch.cat([salience['grafting_mask_salience']['modules'][layer]['query']['output_mask']['s'].view(-1, attention_head_size).mean(dim=1) for layer in range(12)])
    value_tuning_scores = torch.cat([salience['grafting_mask_salience']['modules'][layer]['value']['output_mask']['s'].view(-1, attention_head_size).mean(dim=1) for layer in range(12)])
    head_tuning_scores = (query_tuning_scores + value_tuning_scores) / 2
    intermediate_tuning_scores = torch.cat([salience['grafting_mask_salience']['modules'][layer]['intermediate']['output_mask']['s'] for layer in range(12)])
    hidden_tuning_scores = torch.stack([salience['grafting_mask_salience']['modules'][layer]['query']['input_mask']['s'] for layer in range(12)] + [salience['grafting_mask_salience']['modules'][layer]['value']['input_mask']['s'] for layer in range(12)] + [salience['grafting_mask_salience']['modules'][layer]['intermediate']['input_mask']['s'] for layer in range(12)]).mean(dim=0)
    tuning_scores = {
        'head_mask': head_tuning_scores,
        'intermediate_mask': intermediate_tuning_scores,
        'hidden_mask': hidden_tuning_scores,
    }
    
    
    joint_pruning_scores = torch.cat([salience['model_mask_salience']['salience'][m]['s'] * tuning_scores[m] / param_per_block[m] for m in ['head_mask', 'intermediate_mask', 'hidden_mask']])
    sorted_joint_scores, sorted_joint_indices = torch.sort(joint_pruning_scores, descending=False)
    sorted_joint_score_types = score_types[sorted_joint_indices]
    