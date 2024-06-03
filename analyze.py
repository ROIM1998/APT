import sys
import os
import json

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

if __name__ == '__main__':
    path = sys.argv[1]
    if os.path.exists(os.path.join(path, 'trainer_state.json')):
        data = json.load(open(os.path.join(path, 'trainer_state.json')))
    else:
        path = os.path.join(path, '../')
        data = json.load(open(os.path.join(path, 'trainer_state.json')))
    if 'squad' in path:
        eval_key = 'f1'
    elif 'cola' in path:
        eval_key = 'eval_matthews_correlation'
    elif 'stsb' in path:
        eval_key = 'eval_spearmanr'
    elif 'cnndm' in path or 'xsum' in path:
        eval_key = 'eval_loss'
    else:
        eval_key = 'eval_accuracy'
    
    log_history = data['log_history']
    starting_param_num = [v for v in log_history if 'total_params' in v][0]['total_params']
    log_history = pd.DataFrame(log_history)
    log_history['total_params'] = log_history['total_params'] / starting_param_num
    log_history['tuning_params'] = log_history['tuning_params'] / starting_param_num
    
    if eval_key not in log_history.columns:
        eval_key = 'eval_loss'
    # Calculating the between-step training time

    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(8, 20))
        
    # Plot training loss and eval loss
    sns.set_theme(style="darkgrid")
    sns.lineplot(x='epoch', y='eval_loss', data=log_history, label='eval_loss', ax=axs[0])
    sns.lineplot(x='epoch', y='loss', data=log_history, label='train_loss', ax=axs[0])
    sns.lineplot(x='epoch', y='distill_loss', data=log_history, label='distill_loss', ax=axs[0])
    sns.lineplot(x='epoch', y='distill_ce_loss', data=log_history, label='distill_ce_loss', ax=axs[0])
    axs[0].axis(ymin=-0.1, ymax=1.)
    sns.lineplot(x='epoch', y=eval_key, data=log_history, label=eval_key, ax=axs[1])
    sns.lineplot(x='epoch', y='learning_rate', data=log_history, label='learning_rate', ax=axs[2])
    # Plot pruning, tuning, and distillation schedule
    sns.lineplot(x='epoch', y='tuning_params', data=log_history, label='tuning_params', ax=axs[3])
    if 'total_params' in log_history.columns:
        sns.lineplot(x='epoch', y='total_params', data=log_history, label='total_params', ax=axs[3])
    if 'moving_term' in log_history.columns and not log_history['moving_term'].isna().all():
        sns.lineplot(x='epoch', y='moving_term', data=log_history, label='moving_term', ax=axs[3])
    sns.lineplot(x='epoch', y='step_end_mem', data=log_history, label='step_end_mem', ax=axs[4])
    training_speed = log_history[~log_history['training_time'].isna()]['step'].diff()[1:] / log_history[~log_history['training_time'].isna()]['training_time'].diff()[1:]
    training_speed_epoch = log_history[~log_history['training_time'].isna()]['epoch'][1:]
    sns.lineplot(x=training_speed_epoch, y=training_speed, label='training_speed', ax=axs[5])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'analysis.png'))