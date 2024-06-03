import seaborn as sns
sns.set_theme(style='darkgrid')
import os
import json
import sys
import torch
import re
import pandas as pd
import loralib as lora

from math import pi
from matplotlib import pyplot as plt
from post_analysis import get_relative_metrics
from utils.plot_utils import plot_acc_speedup, multiplot_acc_metrics
from models import build_model
from torch.utils.data import Subset
from utils.utils import *
from transformers import (HfArgumentParser)
from args import DataTrainingArguments
from models.model_args import ModelArguments
from args import MinusTrainingArguments
from trainer.param_control import ParamController
from prune.pruner import AdapterPruner
from utils import build_trainer, build_dataloader
from prune.fisher import collect_grads_by_suffix, collect_weight_saliency
from collect_reports import gather_core_info

key2metrics = {
    'speedup': ['training_speedup', 'inference_speedup'],
    'memory': ['relative_eval_memory_usage', 'relative_training_memory_usage'],
    'density': ['attn_density', 'ffn_density'],
}

def plot_radar(df):    
    # ------- PART 1: Create background
    
    # number of variable
    categories=list(df)[1:]
    print(categories)
    N = len(categories)
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    y_max = max([v for v in df.max() if isinstance(v, float)])
    plt.yticks([y_max / 4, y_max / 2, y_max * 3 / 4], [str(round(y_max / 4, 1)), str(round(y_max / 2, 1)), str(round(y_max * 3 / 4, 1))], color="grey", size=7)
    plt.ylim(0,y_max)
    

    # ------- PART 2: Add plots
    
    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable
    
    # Ind1
    values=df.loc[0].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="ft")
    ax.fill(angles, values, 'b', alpha=0.1)
    
    # Ind2
    values=df.loc[1].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="lora")
    ax.fill(angles, values, 'y', alpha=0.1)
    
    # Ind3
    values=df.loc[2].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="elastic-distill")
    ax.fill(angles, values, 'g', alpha=0.1)
    
    # Ind3
    values=df.loc[3].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="elastic-nodistill-cubic")
    ax.fill(angles, values, 'r', alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.savefig('radar.png')
    

if __name__ == '__main__':
    # Plot radar map of existing models
    df = gather_core_info('output', 'roberta', 'sst2')
    df['model_flops'] = df['model_flops'].apply(lambda x: float(x.replace('G', '').strip()))
    df = df[['dir', 'eval_accuracy', 'peak_mem', 'train_runtime_per_epoch', 'time_to_best', 'model_flops', 'num_parameters']]
    old_df['model_flops'] = old_df['model_flops'].apply(lambda x: float(x.replace('G', '').strip()))
    old_df = old_df[['dir', 'eval_accuracy', 'peak_mem', 'train_runtime_per_epoch', 'time_to_best', 'model_flops', 'num_parameters']]
    concat_df = pd.concat([df, old_df])
    concat_df = concat_df[concat_df['dir'].apply(lambda x: 'roberta-base' in x)]
    concat_df.sort_values(by='eval_accuracy', ascending=False, inplace=True)
    concat_df = concat_df[concat_df['dir'].apply(lambda x: 'mac0.6' not in x)]
    concat_df.to_excel('model_info.xlsx', index=False)
    
    plt.clf()
    plt.figure(figsize=(12, 12))
    new_df = pd.read_excel('new_model_info.xlsx')
    baseline = new_df.iloc[0]
    for k in new_df:
        if k != 'group':
            if k == 'eval_accuracy':
                new_df[k] = new_df[k] / baseline[k]
            else:
                new_df[k] = baseline[k] / new_df[k]
    plot_radar(new_df)
    
    # Salience tendency plot
    plt.clf()
    salience = torch.load('output/bert-base-uncased_lora_minus_mnli_once_global_none_nodistill_restore/mac0.4/epoch20/bz32/warmup5/paramq:0-11,v:0-11,i:0-11/lora_r8/lora_alpha16/salience_collected_999.pt', map_location='cpu')
    head_saliences=  torch.stack([v['head_mask_grad'] for v in salience], dim=0)
    # Compare the first head in different layers
    head_saliences = head_saliences[:, :, 0].pow(2)
    # Moving average to smooth the curve
    head_saliences = head_saliences.cumsum(dim=0) / torch.arange(1, head_saliences.shape[0] + 1).unsqueeze(1)
    sns.lineplot(data=pd.DataFrame(head_saliences.log10().numpy().tolist(), columns=[f'layer_{i}' for i in range(12)]))
    plt.savefig('head_salience_tendency.png')