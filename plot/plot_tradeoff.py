import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid")

from matplotlib import pyplot as plt

def multiplot_acc_metrics(dfs, y_keys, metrics, label_key=None):
    # specify plot layouts with different width using subplots()
    num_dfs = len(dfs) if isinstance(dfs, list) else 1
    f, axs = plt.subplots(num_dfs, len(metrics),
        figsize=(3 * len(metrics), 3 * num_dfs),
        sharey='row' if num_dfs > 1 else True,
    )
    if not isinstance(dfs, list):
        dfs = [dfs]
    for j, df in enumerate(dfs):
        for i, metric in enumerate(metrics):
            # makde line plot along x-axis
            sns.lineplot(data=df,
                x=metric,
                y=y_keys[j] if num_dfs > 1 and isinstance(y_keys, list) else y_keys,
                ax=axs[i] if num_dfs == 1 else axs[j][i],
                marker="o",
            )
            # label points on the plot
            if label_key is not None:
                scale = max(df[metric].tolist()) - min(df[metric].tolist())
                for x, y, t in zip(df[metric].tolist(), df[y_keys[j]].tolist(), df[label_key].tolist()):
                    # the position of the data label relative to the data point can be adjusted by adding/subtracting a value from the x &/ y coordinates
                    ax = axs[i] if num_dfs == 1 else axs[j][i]
                    ax.text(x = x + scale * 0.05, # x-coordinate position of data label
                    y = y - 0.5, # y-coordinate position of data label, adjusted to be 150 below the data point
                    s = '{:.1f}'.format(t), # data label, formatted to ignore decimals
                    color = 'black',
                    size = 8,
                    ) # set colour of line
    f.tight_layout()
    # Set bigger font size for x and y labels and tick labels
    # for ax in axs:
    #     ax.set_xlabel(ax.get_xlabel(), fontsize=15)
    #     ax.set_ylabel(ax.get_ylabel(), fontsize=15)
    #     ax.tick_params(axis='both', which='major', labelsize=15)
    return f, axs
        
def multiplot_metrics(df, y, metrics, n_rol: int = 1):
    # specify plot layouts with different width using subplots()
    assert len(metrics) % n_rol == 0
    f, axs = plt.subplots(n_rol, len(metrics) // n_rol,
        figsize=(4 * len(metrics) / n_rol, 2 * n_rol),
        sharey=True,
    )
    for i, metric in enumerate(metrics):
        # makde line plot along x-axis
        sns.lineplot(data=df,
            x=metric,
            y=y,
            ax=axs[i // n_rol][i % n_rol],
            marker="o",
        )
    f.tight_layout()


if __name__ == '__main__':
    df = pd.read_csv('data/tuning_tradeoff.csv')
    plt.clf()
    df.rename(columns={'lora_r': 'LoRA init rank', 'acc': 'Accuracy', 'peak mem': 'Train Peak Memory'}, inplace=True)
    df['97% TTA'] = 127 / df['97% TTA']
    df['Train Peak Memory'] /= 2696
    multiplot_acc_metrics(df, 'LoRA init rank', ['Accuracy', 'Train Peak Memory', '97% TTA'])
    plt.savefig('data/tuning_tradeoff.pdf', dpi=300, bbox_inches="tight")
    
    plt.clf()
    roberta_df = pd.read_csv('data/roberta_pruning_tradeoff.csv')
    t5_df = pd.read_csv('data/t5_pruning_tradeoff.csv')
    llama_df = pd.read_csv('data/llama2_7b_tradeoff.csv')
    roberta_df['RoBERTa Relative Accuracy'] = roberta_df['Accuracy'] / 94.5 * 100
    t5_df['T5 Relative Accuracy'] = t5_df['Accuracy'] / 95 * 100
    baseline = llama_df.iloc[0]
    llama_df['LLaMA2 Relative Performance'] = llama_df['Average'] / baseline['Average'] * 100
    llama_df['Inf. Speedup'] = llama_df['Inference Throughput'] / baseline['Inference Throughput']
    llama_df['Inf. Memory'] = llama_df['Inference Memory'] / baseline['Inference Memory']
    roberta_df['Inf. Mem. Red.'] = 1 / roberta_df['Inf. Memory']
    t5_df['Inf. Mem. Red.'] = 1 / t5_df['Inf. Memory']
    llama_df['Inf. Mem. Red.'] = 1 / llama_df['Inf. Memory']
    llama_df['Training Peak Memory'] = llama_df['Training Peak Memory'] / baseline['Training Peak Memory']
    f, axs = multiplot_acc_metrics([roberta_df, t5_df, llama_df], ['RoBERTa Relative Accuracy', 'T5 Relative Accuracy', 'LLaMA2 Relative Performance'], ['Inf. Speedup', 'Inf. Mem. Red.'])
    # f, axs = multiplot_acc_metrics([roberta_df, t5_df], ['RoBERTa Relative Accuracy', 'T5 Relative Accuracy'], ['Inf. Speedup', 'Inf. Memory', '95% TTA'])
    # Add points for FT+CoFi, LoRA+CoFi, and LoRA+MT for RoBERTa; and LoRA+MT for T5
    axs[0][0].scatter([2.707561773], [94.5/94.5*100], marker='o', label='APT', zorder=1)
    axs[0][0].scatter([2.629], [93.0/94.5*100], marker='x', label='LoRA+Prune', color='green', zorder=10)
    axs[0][1].scatter([1 / 0.751], [93.0/94.5*100], marker='x', label='LoRA+Prune', color='green', zorder=10)
    # axs[0][2].scatter([0.02], [93.0/94.5*100], marker='x', label='LoRA+Prune', color='green', zorder=10)
    axs[1][0].scatter([2.125], [92.3/95*100], marker='x', label='LoRA+Prune', color='green', zorder=10)
    axs[1][1].scatter([1 / 0.734], [92.3/95*100], marker='x', label='LoRA+Prune', color='green', zorder=10)
    # axs[1][2].scatter([0.025], [92.3/95*100], marker='x', label='LoRA+Prune', color='green', zorder=10)
    axs[0][0].scatter([2.592], [94.5/94.5*100], marker='+', label='Prune+Distill', color='red', zorder=10)
    axs[0][1].scatter([1 / 0.792], [94.5/94.5*100], marker='+', label='Prune+Distill', color='red', zorder=10)
    # axs[0][2].scatter([0.067], [94.5/94.5*100], marker='+', label='Prune+Distill', color='red', zorder=10)
    axs[0][0].scatter([2.537], [91.9/94.5*100], marker='^', label='LoRA+Prune+Distill', color='orange', zorder=10)
    axs[0][1].scatter([1 / 0.823], [91.9/94.5*100], marker='^', label='LoRA+Prune+Distill', color='orange', zorder=10)
    # axs[0][2].scatter([0.015], [91.9/94.5*100], marker='^', label='LoRA+Prune+Distill', color='orange', zorder=10)
    axs[2][0].scatter([1.155], [45.5/53.4*100], marker='x', label='LoRA+Prune', color='green', zorder=10)
    axs[2][1].scatter([1 / 0.689], [45.5/53.4*100], marker='x', label='LoRA+Prune', color='green', zorder=10)
    # axs[2][2].scatter([1.0], [45.5/53.4*100], marker='x', label='LoRA+Prune', color='green', zorder=10)
    axs[2][0].scatter([1.148], [42.9/53.4*100], marker='v', label='LLMPruner', color='purple', zorder=10)
    axs[2][1].scatter([1 / 0.742], [42.9/53.4*100], marker='v', label='LLMPruner', color='purple', zorder=10)
    # axs[2][2].scatter([2.536], [42.9/53.4*100], marker='+', label='LLMPruner', color='red', zorder=10)
    def plot_arrow(ax, rotation=45, alpha=1):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.text((xmin + xmax) / 2, (ymin + ymax) / 2, "Better",
                    ha="center", va="center", rotation=rotation, size=12,
                    bbox=dict(boxstyle="rarrow,pad=0.3",
                            fc="lightblue", ec="steelblue", lw=2), zorder=1)
    plot_arrow(axs[0][0])
    plot_arrow(axs[0][1])
    # plot_arrow(axs[0][2])
    plot_arrow(axs[1][0])
    plot_arrow(axs[1][1])
    plot_arrow(axs[2][0])
    plot_arrow(axs[2][1])
    # plot_arrow(axs[1][2])
    h,l = axs[0][0].get_legend_handles_labels()
    llama_h, llama_l = axs[2][0].get_legend_handles_labels()
    handlers = h + llama_h[1:2]
    labels = l + llama_l[1:2]
    plt.legend(handlers, labels, loc='upper center', bbox_to_anchor=(-0.1, 4), ncol=len(handlers) // 2)
    plt.savefig('data/pruning_tradeoff.pdf', dpi=300, bbox_inches="tight")
    
    
    # plt.clf()
    # f, axs = multiplot_acc_metrics(llama_df, 'Relative Performance', ['Inf. Speedup', 'Inf. Memory', 'Training Peak Memory'])
    # axs[0].scatter([1.155], [45.5/53.4*100], marker='x', label='LoRA+Prune', color='green', zorder=10)
    # axs[1].scatter([0.689], [45.5/53.4*100], marker='x', label='LoRA+Prune', color='green', zorder=10)
    # # axs[2].scatter([1.0], [45.5/53.4*100], marker='x', label='LoRA+Prune', color='green', zorder=10)
    # axs[0].scatter([1.148], [42.9/53.4*100], marker='+', label='LLMPruner', color='red', zorder=10)
    # axs[1].scatter([0.742], [42.9/53.4*100], marker='+', label='LLMPruner', color='red', zorder=10)
    # # axs[2].scatter([2.536], [42.9/53.4*100], marker='+', label='LLMPruner', color='red', zorder=10)
    # h, l = axs[0].get_legend_handles_labels()
    # axs[0].legend(loc='lower left', handles=h, fontsize=8)
    # plot_arrow(axs[0], rotation=45)
    # plot_arrow(axs[1], rotation=135)
    # plot_arrow(axs[2], rotation=135)
    # plt.savefig('data/llama2_7b_tradeoff.pdf', dpi=300)
    
    df = pd.read_csv('data/preliminary.csv')
    plt.clf()
    plt.figure(figsize=(4, 3))
    sns.lineplot(data=df, x='Density', y='FT+MT', marker='X', label='FT+MT', color='red')
    sns.lineplot(data=df, x='Density', y='LoRA+MT', marker='X', label='LoRA+MT', color='blue')
    # Add points for FT+CoFi: (0.4, 0.945); LoRA+CoFi: (0.4, 0.919)
    sns.scatterplot(x=[0.4], y=[0.945], marker='o', label='FT+CoFi', color='red')
    sns.scatterplot(x=[0.4], y=[0.919], marker='o', label='LoRA+CoFi', color='blue')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('data/preliminary.pdf', dpi=300)
    
    df = pd.read_csv('data/llama_init_density.csv')
    tasks = ['ARC', 'HellaSwag', 'MMLU', 'TruthfulQA', 'Avg.']
    df.loc[df['Backbone'] == 'LLaMA 2 7B', 'ARC'] = df[df['Backbone'] == 'LLaMA 2 7B']['ARC'] / 55.6 * 100
    df.loc[df['Backbone'] == 'LLaMA 2 7B', 'HellaSwag'] = df[df['Backbone'] == 'LLaMA 2 7B']['HellaSwag'] / 79.3 * 100
    df.loc[df['Backbone'] == 'LLaMA 2 7B', 'MMLU'] = df[df['Backbone'] == 'LLaMA 2 7B']['MMLU'] / 46.9 * 100
    df.loc[df['Backbone'] == 'LLaMA 2 7B', 'TruthfulQA'] = df[df['Backbone'] == 'LLaMA 2 7B']['TruthfulQA'] / 49.9 * 100
    df.loc[df['Backbone'] == 'LLaMA 2 7B', 'Avg.'] = df[df['Backbone'] == 'LLaMA 2 7B']['Avg.'] / 57.9 * 100
    df.loc[df['Backbone'] == 'LLaMA 2 13B', 'ARC'] = df[df['Backbone'] == 'LLaMA 2 13B']['ARC'] / 60.8 * 100
    df.loc[df['Backbone'] == 'LLaMA 2 13B', 'HellaSwag'] = df[df['Backbone'] == 'LLaMA 2 13B']['HellaSwag'] / 82.8 * 100
    df.loc[df['Backbone'] == 'LLaMA 2 13B', 'MMLU'] = df[df['Backbone'] == 'LLaMA 2 13B']['MMLU'] / 56.0 * 100
    df.loc[df['Backbone'] == 'LLaMA 2 13B', 'TruthfulQA'] = df[df['Backbone'] == 'LLaMA 2 13B']['TruthfulQA'] / 46.5 * 100
    df.loc[df['Backbone'] == 'LLaMA 2 13B', 'Avg.'] = df[df['Backbone'] == 'LLaMA 2 13B']['Avg.'] / 61.5 * 100
    
    
    _7b_df = pd.DataFrame(columns=['Init. Density', 'Task', 'Relative Accuracy (%)'])
    for i, task in enumerate(tasks):
        _7b_df = pd.concat([_7b_df, df[['Init. Density', task]][:3].rename(columns={task: 'Relative Accuracy (%)'}).assign(Task=task)])
    _13b_df = pd.DataFrame(columns=['Init. Density', 'Task', 'Relative Accuracy (%)'])
    for i, task in enumerate(tasks):
        _13b_df = pd.concat([_13b_df, df[['Init. Density', task]][-3:].rename(columns={task: 'Relative Accuracy (%)'}).assign(Task=task)])
        
    plt.clf()
    f, axs = plt.subplots(1, 2, sharey=True, figsize=(6, 3))
    sns.barplot(data=_7b_df, x='Init. Density', y='Relative Accuracy (%)', hue='Task', ax=axs[0])
    sns.barplot(data=_13b_df, x='Init. Density', y='Relative Accuracy (%)', hue='Task', ax=axs[1])
    # Remove sns legend
    axs[0].get_legend().remove()
    axs[1].get_legend().remove()
    axs[0].set_title('LLaMA2 7B')
    axs[1].set_title('LLaMA2 13B')
    axs[0].set_ylim(75, 102)
    axs[1].set_ylim(75, 102)
    # Make legend
    handles, labels = axs[1].get_legend_handles_labels()
    f.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.1), fontsize=10)
    plt.tight_layout()
    plt.savefig('data/llama_init_density.pdf', dpi=300, bbox_inches="tight")