import pickle
import json
import torch
import numpy as np
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set(font_scale=2)
import pandas as pd
from matplotlib import pyplot as plt
from typing import Dict, List, Union


def violin_plot():
    pass


def line_plot(data: Union[Dict[str, float], Dict[float, float]], filename:str = None):
    if isinstance(list(data)[0], str):
        data = {float(k): v for k, v in data.items()}
    sns.lineplot(x=list(data.keys()), y=list(data.values()))
    if filename:
        plt.savefig(filename)
    plt.clf()
    

def draw_mean_and_vars(input_files: List[str]):
    for f in input_files:
        if f.endswith('.json'):
            data = json.load(open(f, 'r'))
        elif f.endswith('.pkl'):
            data = pickle.load(open(f, 'rb'))
        line_plot({float(k): np.mean(v) for k, v in data.items()}, f.replace('.json', '_mean.png'))
        line_plot({float(k): np.var(v) for k, v in data.items()}, f.replace('.json', '_var.png'))
    

def plot_multiple_jsonl(file_path, x_col):
    with open(file_path, 'r') as f:
        json_lines = f.read().splitlines()
    data = [json.loads(jl) for jl in json_lines]
    acc_keys = ['searched', 'rearranged', 'rescaled']
    acc_df = pd.DataFrame([
        {
            'category': k,
            'constraint': d['constraint'],
            'val': d[k],
        }
        for d in data for k in acc_keys
    ])
    _ = plt.figure(figsize=(8, 6))
    sns.lineplot(data=acc_df, x="constraint", y="val", hue="category")


def plot_acc_speedup(df, x, y, hue, label):
    ax = sns.lineplot(data=df, x=x, y=y, hue=hue, markers=True, style=hue)
    ax.set(title='Correlation between %s and %s' % (x, y), xlabel=x, ylabel=y)
    # label points on the plot
    if label is not None:
        for x_val, y_val, l in zip(df[x].tolist(), df[y].tolist(), df[label].tolist()):
            # the position of the data label relative to the data point can be adjusted by adding/subtracting a value from the x &/ y coordinates
            plt.text(
                x = x_val - 0.01, # x-coordinate position of data label
                y = y_val+0.05, # y-coordinate position of data label, adjusted to be 150 below the data point
                s = str(l), # data label, formatted to ignore decimals
                color = 'purple'
            ) # set colour of line


def multiplot_acc_metrics(df, y, metrics):
    # specify plot layouts with different width using subplots()
    f, axs = plt.subplots(1, len(metrics),
        figsize=(8 * len(metrics),6),
        sharey=True,
    )
    for i, metric in enumerate(metrics):
        # makde line plot along x-axis
        sns.lineplot(data=df,
            x=metric,
            y=y,
            ax=axs[i],
            marker="o",
        )
    f.tight_layout()

def plot_score_by_layer(scores: torch.Tensor):
    num_layers = scores.shape[0]
    plt.figure(
        figsize=(12, 3 * num_layers),
    )
    sns.violinplot(data=[torch.log(scores[i]).cpu().numpy() for i in range(num_layers)], orient='h')


def plot_loss_comparison(data, remove_distill_loss: bool = False):
    sns.lineplot(x=[v['epoch'] for v in data if 'loss' in v], y=[v['loss'] for v in data if 'loss' in v] if not remove_distill_loss else [v['loss'] if v['distill_ce_loss'] == 0 else (v['loss'] - (v['distill_ce_loss'] * 0.1 + v['distill_loss'] * 0.9) * 0.5) * 2 for v in data if 'loss' in v], label='train')
    sns.lineplot(x=[v['epoch'] for v in data if 'eval_loss' in v], y=[v['eval_loss'] for v in data if 'eval_loss' in v], label='eval')
    plt.legend()
    
def plot_training_time_per_epoch(all_fn, all_labels):
    for fn, label in zip(all_fn , all_labels):
        data = json.load(open(fn))['log_history']
        time_diff = np.diff([v['training_time'] for v in data if 'training_time' in v])
        epochs = [v['epoch'] for v in data if 'training_time' in v]
        epochs_diff = np.diff(epochs)
        sns.lineplot(x=epochs[1:], y=time_diff / epochs_diff, label=label)
    plt.legend()
    
    
def plot_training_mem_per_epoch(all_fn, all_labels):
    for fn, label in zip(all_fn , all_labels):
        data = json.load(open(fn))['log_history']
        mem = [v['end_mem'] for v in data if 'end_mem' in v]
        epochs = [v['epoch'] for v in data if 'training_time' in v]
        sns.lineplot(x=epochs, y=mem, label=label)
    plt.legend()
    

def plot_eval_loss(all_fn, all_labels):
    for fn, label in zip(all_fn , all_labels):
        data = json.load(open(fn))['log_history']
        mem = [v['eval_loss'] for v in data if 'eval_loss' in v]
        epochs = [v['epoch'] for v in data if 'eval_loss' in v]
        sns.lineplot(x=epochs, y=mem, label=label)
    plt.legend()
    
if __name__ == '__main__':
    all_fn = [
        'output/roberta-base_lora_minus_mnli_linear_gradual_global_distill_full_newversion/mac0.4/lora_r64/lora_alpha16/trainer_state.json',
        'output/roberta-base_lora_minus_mnli_linear_gradual_global_distill_full_newversion_directfinal/mac0.4/lora_r64/lora_alpha16/trainer_state.json',
        'output/roberta-base_lora_minus_mnli_once_global_distill_full_newversion/mac0.4/lora_r64/lora_alpha16/trainer_state.json',
        'output/roberta-base_lora_minus_mnli_once_global_distill_full_shortdistill_newversion/mac0.4/lora_r64/lora_alpha16/trainer_state.json',
        'output/roberta-base_lora_minus_mnli_once_global_nodistill_full_newversion/mac0.4/lora_r64/lora_alpha16/trainer_state.json',
        'output/roberta-base_lora_minus_mnli_once_global_distill_full_exp/mac0.4/lora_r64/lora_alpha16/trainer_state.json'
    ]
    all_labels = [
        'gradual',
        'gradual_directfinal',
        'once_distill',
        'once_shortdistill',
        'nodistill',
        'old_distill'
    ]
    plt.figure(figsize=(16,12))
    plot_training_time_per_epoch(all_fn, all_labels)
    plt.savefig('training_time_comparison.png')
    plt.clf()
    plt.figure(figsize=(16,12))
    plot_training_mem_per_epoch(all_fn, all_labels)
    plt.savefig('training_mem_comparison.png')
    plt.clf()
    plt.figure(figsize=(16,12))
    plot_eval_loss(all_fn, all_labels)
    plt.savefig('eval_loss_comparison.png')
    plt.clf()