import os
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
import scipy.signal as signal

from tqdm import tqdm
from utils.minus_utils import parse_collected_salience
from utils.cofi_utils import parse_cofi_zs

def filter(array, dim=0):
    # Define the filter parameters
    cutoff_freq = 0.01  # Adjust this value to set the cutoff frequency
    fs = 1.0  # Sampling frequency

    # Calculate the filter coefficients
    order = 4  # Adjust this value to set the filter order
    nyquist_freq = 0.5 * fs
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    b, a = signal.butter(order, normalized_cutoff_freq, btype='low', analog=False, output='ba')

    # Apply the filter along dimension 0 (time) using scipy.signal.lfilter
    filtered_aray = signal.lfilter(b, a, array, axis=dim)
    return filtered_aray

def plot_salience_change(array, savename):
    filtered_query_fisher = filter(array, dim=0)
    df = pd.DataFrame(np.log10(filtered_query_fisher))
    df = df.unstack().reset_index()
    df.columns = ['Head', 'Timestep', 'Importance']
    sns.lineplot(data=df, x='Timestep', y='Importance', hue='Head')
    plt.xlabel('Timestep')
    plt.ylabel('Importance')
    plt.savefig('%s.png' % savename)
    plt.clf()

def plot_salience_change_given_cofi_zs(array, savename):
    cofi_zs = torch.load('/home/zbw/projects/CoFiPruning/out/bert-base/MNLI/CoFi/MNLI_sparsity0.60_lora/zs.pt')
    head_zs, intermediate_zs, hidden_zs = parse_cofi_zs(cofi_zs)
    pruned_heads = head_zs.view(-1).nonzero().squeeze().tolist()
    filtered_query_fisher = filter(array, dim=0)
    df = pd.DataFrame(np.log10(filtered_query_fisher))
    df = df.unstack().reset_index()
    df.columns = ['Head', 'Timestep', 'Importance']
    df['Pruned'] = df['Head'].apply(lambda x: 'Pruned' if x in pruned_heads else 'Retained')
    sns.lineplot(data=df, x='Timestep', y='Importance', size='Head', hue='Pruned')
    plt.xlabel('Timestep')
    plt.ylabel('Importance')
    plt.legend()
    plt.savefig('%s.png' % savename)
    plt.clf()
    

if __name__ == '__main__':
    root_dir = 'output/bert-base-uncased_lora_minus_mnli_once_global_none_nodistill_restore/mac0.4/epoch20/bz32/warmup5/paramq:0-11,v:0-11,i:0-11/lora_r8/lora_alpha16/'
    salience_collected_filenames = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.startswith('salience_collected')]
    salience_collected_filenames = sorted(salience_collected_filenames, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    parsed = parse_collected_salience('output/bert-base-uncased_lora_minus_mnli_once_global_none_nodistill_restore/mac0.4/epoch20/bz32/warmup5/paramq:0-11,v:0-11,i:0-11/lora_r8/lora_alpha16/salience_collected_999.pt')
    query_output_fisher: torch.Tensor = parsed['query_fisher']
    
    # Filter the fisher signal on dim 0 (timestep) by low-pass filtering
    query_output_fisher_numpy = query_output_fisher.numpy()
    plot_salience_change(query_output_fisher_numpy.reshape(1000, -1)[:, 0:144:12], 'query_salience_change')
    plot_salience_change(parsed['value_fisher'].numpy().reshape(1000, -1)[:, 0:144:12], 'value_salience_change')
    plot_salience_change(parsed['head_fisher'].numpy().reshape(1000, -1), 'head_salience_change')
    
    total_head_fisher = []
    for fn in tqdm(salience_collected_filenames):
        parsed = parse_collected_salience(fn)
        parsed_head_fisher = parsed['head_fisher']
        total_head_fisher.append(parsed_head_fisher)
    
    total_head_fisher = torch.cat(total_head_fisher, dim=0)
    
    plt.figure(figsize=(16, 9))
    plot_salience_change(total_head_fisher.numpy().reshape(total_head_fisher.shape[0], -1), 'head_salience_change')
    plot_salience_change_given_cofi_zs(total_head_fisher.numpy().reshape(total_head_fisher.shape[0], -1), 'head_salience_change_given_cofi_zs')