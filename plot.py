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

def violin_plot_by_suffix(model, dataloader, suffix, status):
    # Create the violin plot grouped with #layer and attr (query or value)
    roberta_regex = r'^roberta\.encoder\.layer\.(\d+?)\.attention\.self\.(.+?)\..+$'
    grads = collect_grads_by_suffix(model, dataloader, suffix)
    scores = {k: v.pow(2).sum(dim=0) for k, v in grads.items() if suffix in k}
    scores = [
        {
            'layer_id': int(re.match(roberta_regex, k).group(1)),
            'attr': re.match(roberta_regex, k).group(2),
            'score': np.log10(val.item()),
        }
        for k, v in scores.items() for val in v
    ]
    score_df = pd.DataFrame(scores)
    sns.violinplot(data=score_df, x='layer_id', y='score', hue='attr', split=True)
    plt.savefig('%s_%s_scores_violin.png' % (status, suffix[1:].replace('_mask', '')))
    plt.clf()
    return score_df

def get_block_salience(model, dataloader):
    for m in model.modules():
        if isinstance(m, lora.LoRALayer):
            m.eval()
    layer_names = [n for n, p in model.named_modules() if 'query' in n or 'value' in n]
    all_grads = collect_weight_saliency(model, dataloader, layer_names)
    num_heads = model.config.num_attention_heads
    all_salience = {
        k: v.view(v.shape[0], num_heads, -1, v.shape[-1]).sum(dim=-1).sum(dim=-1).pow(2).sum(dim=0)
        for k, v in all_grads.items()
    }
    for m in model.modules():
        if isinstance(m, lora.LoRALayer):
            m.train()
    return all_salience, all_grads

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

def main():
    sys.argv = ['test_pre_tuning_prune.py',
            '--output_dir',
            './output/test_model_grafting_dynamic_all_dependent_pruned_test/',
            '--model_name_or_path',
            'old_output/roberta-base_lora_minus_mnli_once_global_nodistill/mac0.4/epoch20/bz128/numprune5/lora_r64/lora_alpha16/pre_pruning_model',
            '--task_name',
            'mnli',
            '--do_train',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '128',
            '--per_device_eval_batch_size',
            '128',
            '--apply_lora',
            '--lora_r',
            '8',
            '--lora_alpha',
            '16',
            '--save_strategy',
            'no',
            '--evaluation_strategy',
            'steps',
            '--num_train_epochs',
            '30',
            '--learning_rate',
            '5e-4',
            '--weight_decay',
            '0.1',
            '--warmup_ratio',
            '0.06',
            '--report_to',
            'none',
            ]
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MinusTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # training_args.disable_tqdm = False
    t_name, raw_datasets = get_raw_datasets(model_args, data_args, training_args)
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)

    model.head_mask, model.intermediate_mask = model.head_mask.to(training_args.device), model.intermediate_mask.to(training_args.device)
    # trainer.evaluate()

    pruning_batch_size = 4
    num_pruning_batches = 64
    dataloader = build_dataloader(Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]), pruning_batch_size, data_args, training_args, tokenizer)
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset)
    all_salience, all_grads = get_block_salience(model, dataloader)
    model_args.model_name_or_path = 'old_output/roberta-base_mnli_full/epoch1/bz128'
    config, tokenizer, ft_model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    ft_model = ft_model.to(training_args.device)
    ft_model.head_mask, ft_model.intermediate_mask = ft_model.head_mask.to(training_args.device), ft_model.intermediate_mask.to(training_args.device)
    all_ft_salience, all_ft_grads = get_block_salience(ft_model, dataloader)
    
    saliences = []
    for k in all_salience:
        lora_salience = all_salience[k].detach().log().cpu().numpy().tolist()
        ft_salience = all_ft_salience[k].detach().log().cpu().numpy().tolist()
        for l_s, f_s in zip(lora_salience, ft_salience):
            saliences.append({
                'lora_salience': l_s,
                'ft_salience': f_s,
                'layer_type': k.split('.')[-2],
            })
    saliences = pd.DataFrame(saliences)
    plt.figure(figsize=(12, 12))
    sns.scatterplot(data=saliences, x='lora_salience', y='ft_salience', hue='layer_type')
    plt.savefig('lora_vs_ft_attn_salience.png')
    plt.clf()

    saliences = []
    for i in range(12):
        layer_prefix = 'roberta.encoder.layer.%d' % i + '.attention.self.%s.weight'
        query_saliences, value_saliences = all_salience[layer_prefix % 'query'].detach().log().cpu().numpy().tolist(), all_salience[layer_prefix % 'value'].detach().log().cpu().numpy().tolist()
        for q, v in zip(query_saliences, value_saliences):
            saliences.append({
                'Query Salience': q,
                'Value Salience': v,
                'model_type': 'LoRA',
            })
        query_saliences, value_saliences = all_ft_salience[layer_prefix % 'query'].detach().log().cpu().numpy().tolist(), all_ft_salience[layer_prefix % 'value'].detach().log().cpu().numpy().tolist()
        for q, v in zip(query_saliences, value_saliences):
            saliences.append({
                'Query Salience': q,
                'Value Salience': v,
                'model_type': 'FT',
            })

    saliences = pd.DataFrame(saliences)
    plt.figure(figsize=(12, 12))
    sns.scatterplot(data=saliences, x='Query Salience', y='Value Salience', hue='model_type')
    plt.xlim(-11, 1)
    plt.ylim(-11, 1)
    plt.savefig('query_vs_value_attn_salience.png')
    plt.clf()
    
def plot_mt_preliminary(original_fn, lora_fn):
    original, lora = [json.loads(s) for s in open(original_fn).read().splitlines()], [json.loads(s) for s in open(lora_fn).read().splitlines()]
    original_eval_acc = [(v['constraint'], v['rescaled'] * 100) for v in original]
    lora_eval_acc = [(v['constraint'], v['rescaled'] * 100) for v in lora]
    sns.lineplot(data=pd.DataFrame(original_eval_acc, columns=['constraint', 'accuracy']), x='constraint', y='accuracy', label='Fine-tune')
    sns.lineplot(data=pd.DataFrame(lora_eval_acc, columns=['constraint', 'accuracy']), x='constraint', y='accuracy', label='LoRA')
    plt.xlabel('FLOP constraint')
    plt.ylabel('Accuracy')
    plt.savefig('mt_preliminary.png')
    

if __name__ == '__main__':
    main()
    df, new_df = get_relative_metrics('output/roberta-base_lora_mnli/epoch5/lora_r8/lora_alpha16', 'output/roberta-base_lora_minus_mnli_once_global_distill_full_exp_shorter')

    plt.figure(figsize=(16, 12))
    plot_acc_speedup(new_df, 'accuracy', 'metric', 'metric_type', 'mac_constraint')
    plt.savefig('metrics_acc.png')
    plt.clf()
    df=df[df['lora_r'] == 64]
    
    multiplot_acc_metrics(df, 'mac_constraint', ['eval_accuracy'])
    plt.title(f'mac_constraint vs accuracy metrics')
    plt.savefig(os.path.join('figures/metric_correlation', f'mac_accuracy_corr.png'))
    plt.clf()

    for anchor in ['eval_accuracy', 'mac_constraint']:
        for metric_key, metrics in key2metrics.items():
            multiplot_acc_metrics(df, anchor, metrics)
            plt.title(f'{anchor} vs {metric_key} metrics')
            plt.savefig(os.path.join('figures/metric_correlation', f'{anchor}_{metric_key}_corr.png'))
            plt.clf()
            
    # Plot radar map of existing models
    df = gather_core_info('output', 'roberta', 'mnli')
    old_df = gather_core_info('exp_output', 'roberta', 'mnli')
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