import seaborn as sns
sns.set_theme(style='whitegrid')

import re
import os
import json
import traceback
import pandas as pd

from matplotlib import pyplot as plt

root_dir = 'output'

minus_dir_regex = r'^(?P<model>.+?)_lora_minus_(?P<task>.+?)_(?P<pruning_strategy>.+?)_global_free_inout_nodistill/mac(?P<mac_constraint>.+?)/epoch(?P<epoch>.+?)/bz(?P<batch_size>.+?)/numprune5/param(?P<param_config>.+?)/lora_r(?P<lora_r>.+?)/lora_alpha(?P<lora_alpha>.+)$'
lora_dir_regex = r'^(?P<model>.+?)_lora_(?P<task>.+?)/epoch(?P<epoch>.+?)/bz(?P<batch_size>.+?)/lora_r(?P<lora_r>.+?)/lora_alpha(?P<lora_alpha>.+?)$'
ft_dir_regex = r'^(?P<model>.+?)_(?P<task>.+?)/epoch(?P<epoch>.+?)/bz(?P<batch_size>.+?)$'
subfolder_regex = r'^(?P<key>.+)(?P<value>\d+(\.\d+)?)$'

task2metric = {
    'squad': 'f1',
    'squadv2': 'f1',
    'mnli': 'eval_accuracy',
    'mnli-mm': 'eval_accuracy',
    'sst2': 'eval_accuracy',
    'qqp': 'eval_accuracy',
    'qnli': 'eval_accuracy',
    'rte': 'eval_accuracy',
    'mrpc': 'eval_accuracy',
    'cola': 'eval_matthews_correlation',
    'stsb': 'eval_pearson',
    'xsum': 'eval_rouge1',
}

def gather_raw_report(root_dir: str):
    res = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f == 'run_report.json':
                val = json.load(open(os.path.join(root, f)))
                val = {**val, **val['args']}
                del val['args']
                val['dir'] = root
                res.append(val)
    df = pd.DataFrame(res)
    return df

def gather_report(root_dir: str):
    res = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f == 'run_report.json':
                try:
                    val = json.load(open(os.path.join(root, f)))
                    del val['args']
                    dir_to_parse = '/'.join(root.split('/')[1:])
                    minus_regex_found = re.match(minus_dir_regex, dir_to_parse)
                    
                    if minus_regex_found:
                        named_group = minus_regex_found.groupdict()
                        val.update(named_group)
                        val['tuning_strategy'] = 'free_inout'
                    else:
                        lora_regex_found = re.match(lora_dir_regex, dir_to_parse)
                        if lora_regex_found:
                            named_group = lora_regex_found.groupdict()
                            val.update(named_group)
                            val['pruning_strategy'] = 'none'
                            val['tuning_strategy'] = 'lora'
                        else:
                            ft_regex_found = re.match(ft_dir_regex, dir_to_parse.replace('_full', ''))
                            if ft_regex_found:
                                named_group = ft_regex_found.groupdict()
                                val.update(named_group)
                                val['pruning_strategy'] = 'none'
                                val['tuning_strategy'] = 'ft'
                    if 'task' in val:
                        val['eval_metric'] = val[task2metric[val['task']]]
                        del val[task2metric[val['task']]]
                        val['dir'] = root
                        res.append(val)
                except Exception as e:
                    print(root)
                    traceback.print_exc()
                
    df = pd.DataFrame(res)
    return df

def gather_core_info(root, model_name, task_name):
    raw_df = gather_raw_report(root)
    core_keys = ['dir', 'eval_accuracy', 'peak_mem', 'train_runtime_per_epoch', 'time_to_best', 'bz32_t_mean', 'bz32_em_mean', 'bz128_t_mean', 'bz128_em_mean', 'model_flops', 'num_parameters']
    df = raw_df[core_keys]
    df = df[df['dir'].apply(lambda x: model_name in x and task_name in x)]
    if model_name == 'bert':
        df = df[df['dir'].apply(lambda x: 'roberta' not in x)]
    return df


if __name__ == '__main__':
    # Collect quick and short summary of run reports
    df = gather_core_info('output', 'roberta', 'mnli')
    df.to_excel('roberta_mnli.xlsx', index=False)
    df = gather_core_info('output', 'roberta', 'sst2')
    df.to_excel('roberta_sst2.xlsx', index=False)
    df = gather_core_info('output', 'bert', 'mnli')
    df.to_excel('bert_mnli.xlsx', index=False)

    df = gather_report('exp_output')
    latest_df = gather_report('output')
    df = pd.concat([df, latest_df])
    
    baselines = df[df['tuning_strategy'] == 'ft'].reset_index(drop=True)
    baseline_tasks = baselines['task'].unique()
    baselines.set_index('task', inplace=True)
    
    efficiency_metrics = ['train_runtime_per_epoch', 'peak_mem', 'bz128_t_mean', 'bz128_em_mean']
    baseline_dict = {
        task: {
            k: baselines.loc[task, k]
            for k in efficiency_metrics
        }
        for task in baseline_tasks
    }
    
    for metric in efficiency_metrics:
        df['relative_' + metric] = df.apply(lambda x: round((baseline_dict[x['task']][metric] / x[metric]), 2) if x['task'] in baseline_dict else None, axis=1)
    
    roberta_mnli_once = df[(df['model'] == 'roberta-base') & (df['pruning_strategy'] == 'once') & (df['task'] == 'mnli') & (df['mac_constraint'] == '0.4')]
    roberta_mnli_once['lora_r'] = roberta_mnli_once['lora_r'].astype(int)
    roberta_mnli_once['eval_metric'] = round(roberta_mnli_once['eval_metric'] * 100, 2)
    # Plot eval accuracy v.s. LoRA r use. Using time and memory efficiency as x-axis, while using eval accuracy as y-axis; lora_r as the label
    f, axs = plt.subplots(1, 3, figsize=(8, 3), sharey=True)
    sns.lineplot(data=roberta_mnli_once, x='eval_metric', y='lora_r', style='lora_alpha', markers=True, legend=False, sort=True, ax=axs[0])
    axs[0].set_xlabel('Accuracy')
    axs[0].set_ylabel('LoRA r')
    sns.lineplot(data=roberta_mnli_once, x='relative_train_runtime_per_epoch', y='lora_r', style='lora_alpha', markers=True, legend=False, sort=True, ax=axs[1])
    axs[1].set_xlabel('Training speedup')
    axs[1].set_ylabel('LoRA r')
    sns.lineplot(data=roberta_mnli_once, x='relative_peak_mem', y='lora_r', style='lora_alpha', markers=True, legend=False, sort=True, ax=axs[2])
    axs[2].set_xlabel('Memory reduction')
    axs[2].set_ylabel('LoRA r')
    axs[2].set_xticks([1.35, 1.355, 1.36])
    f.tight_layout()
    plt.savefig('lora_r_corr.pdf', bbox_inches='tight', format='pdf')
    plt.clf()

    df = gather_report('all_res')
    mnli_df = df[df['task'] == 'mnli'].reset_index(drop=True)
    mnli_df['lora_r'] = mnli_df['lora_r'].astype(int)
    mnli_df['eval_metric'] = round(mnli_df['eval_metric'] * 100, 2)
    efficiency_metrics = ['train_runtime_per_epoch', 'peak_mem', 'bz128_t_mean', 'bz128_em_mean']
    baseline = mnli_df[mnli_df['lora_r'].isna()].iloc[0]
    baseline_dict = {
        k: baseline[k]
        for k in efficiency_metrics
    }
    for metric in efficiency_metrics:
        mnli_df['relative_' + metric] = mnli_df.apply(lambda x: round((baseline_dict[metric] / x[metric]), 2), axis=1)
    mnli_df = mnli_df[~mnli_df['lora_r'].isna()]
    mnli_df.loc[mnli_df['mac_constraint'].isna(), 'relative_bz128_em_mean'] = 1.
    mnli_df.loc[mnli_df['mac_constraint'].isna(), 'mac_constraint'] = 1
    mnli_df['mac_constraint'] = mnli_df['mac_constraint'].astype(float)
    mnli_df['encoder_num_parameters'] = mnli_df['encoder_num_parameters'] / 1000000
    # Plot eval accuracy v.s. FLOP constraint.
    f, axs = plt.subplots(1, 5, figsize=(15, 3), sharey=True)
    sns.lineplot(data=mnli_df, x='encoder_num_parameters', y='eval_metric', style='lora_alpha', markers=True, legend=False, sort=True, ax=axs[0])
    axs[0].set_xlabel('Model Size (M)')
    axs[0].set_ylabel('Accuracy')
    sns.lineplot(data=mnli_df, x='relative_train_runtime_per_epoch', y='eval_metric', style='lora_alpha', markers=True, legend=False, sort=True, ax=axs[1])
    axs[1].set_xlabel('Train Speedup')
    axs[1].set_ylabel('Accuracy')
    sns.lineplot(data=mnli_df, x='relative_peak_mem', y='eval_metric', style='lora_alpha', markers=True, legend=False, sort=True, ax=axs[2])
    axs[2].set_xlabel('Train Mem. Reduction')
    axs[2].set_ylabel('Accuracy')
    sns.lineplot(data=mnli_df, x='relative_bz128_t_mean', y='eval_metric', style='lora_alpha', markers=True, legend=False, sort=True, ax=axs[3])
    axs[3].set_xlabel('Inf. Speedup')
    axs[3].set_ylabel('Accuracy')
    sns.lineplot(data=mnli_df, x='relative_bz128_em_mean', y='eval_metric', style='lora_alpha', markers=True, legend=False, sort=True, ax=axs[4])
    axs[4].set_xlabel('Inf. Memory Reduction')
    axs[4].set_ylabel('Accuracy')
    f.tight_layout()
    plt.savefig('flop_corr.pdf', bbox_inches='tight', format='pdf')
    # plt.savefig('flop_corr.png', bbox_inches='tight', format='png')
    plt.clf()