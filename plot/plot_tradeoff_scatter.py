import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
sns.color_palette("tab10")
sns.set_theme(style="darkgrid")
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

label_dict = {
    'FT': 'Fine-tuning',
    'LoRA': 'LoRA',
    'FT + MT + retrain': 'FT + $\\text{MT}_{\\text{train}}$',
    'FT + MT + redistill': 'FT + $\\text{MT}_{\\text{distill}}$',
    'LoRA + MT': 'LoRA + MT',
    'LoRA + MT + retrain': 'LoRA + $\\text{MT}_{\\text{train}}$',
    'LoRA + MT + redistill': 'LoRA + $\\text{MT}_{\\text{distill}}$',
    'CoFi': 'CoFi',
    'CoFi + LoRA': 'CoFi + LoRA',
    'LLMPruner': 'LLMPruner',
    'APT': 'APT',
}

palette = {
    "Fine-tuning":"tab:red",
    "LoRA":"tab:blue", 
    "LoRA + MT":"tab:pink",
    "LoRA + $\\text{MT}_{\\text{train}}$":"tab:orange",
    "LoRA + $\\text{MT}_{\\text{distill}}$":"tab:brown",
    "CoFi + LoRA": "tab:purple",
    "LLMPruner": "tab:cyan",
    "APT": "tab:green",
}

def expand_df(df):
    baseline = df[df['Method'] == 'LoRA'].iloc[0]
    df['Accuracy'] = df['Acc.']
    df['Relative Accuracy (\\%)'] = df['Acc.'] / baseline['Acc.'] * 100
    if '97% TTA' in df.columns:
        df['97\\% TTA (Second)'] = df['97% TTA']
        df['Relative Training Speed'] = baseline['97% TTA'] / df['97% TTA']
    elif 'Training Time' in df.columns:
        df['Training Time (Second)'] = df['Training Time']
        df['Relative Training Speed'] = baseline['Training Time'] / df['Training Time']
    df['Training Peak Memory (MB)'] = df['Training Peak Mem.']
    df['Relative Training Peak Memory'] = df['Training Peak Mem.'] / baseline['Training Peak Mem.']
    df['Inference time (Second)'] = df['Inf. time']
    df['Inference Speedup'] = baseline['Inf. time'] / df['Inf. time']
    df['Inference memory'] = df['Inf. mem']
    df['Relative Inference memory'] = df['Inf. mem'] / baseline['Inf. mem']
    # df['hue'] = df['Method'].apply(lambda x: hue_dict[x])
    df['Method'] = df['Method'].apply(lambda x: label_dict[x])
    return df

def plot_arrow(ax, rotation=45):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.text((xmin + xmax) / 2, (ymin + ymax) / 2, "Better",
                ha="center", va="center", rotation=rotation, size=12,
                bbox=dict(boxstyle="rarrow,pad=0.3",
                        fc="lightblue", ec="steelblue", lw=2))
    

if __name__ == '__main__':
    df = pd.read_csv('data/perf_efficiency.csv')
    df = expand_df(df)
    t5_df = pd.read_csv('data/t5_perf_efficiency.csv')
    t5_df = expand_df(t5_df)
    llama_df = pd.read_csv('data/llama_perf_efficiency.csv')
    llama_df = expand_df(llama_df)
    
    plt.clf()
    plt.figure(figsize=(3, 3))
    ax = sns.scatterplot(data=df, x="Accuracy", y="97\\% TTA (Second)", size="Training Peak Memory (MB)", style="Method", hue='hue', sizes=(50, 200), legend='brief')
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    h,l = ax.get_legend_handles_labels()
    plt.legend(loc='upper left', handles=h[-10:] + [h[7], h[8], h[10], h[12]],  bbox_to_anchor=(1, 1))
    # Create a custom legend using the plt.legend() function
    plt.savefig('plot_training_tradeoff_scatter.pdf', bbox_inches="tight")
    
    plt.clf()
    plt.figure(figsize=(3, 3))
    ax = sns.scatterplot(data=df, x="Accuracy", y="Inference time", size="Inference memory", style="Method", sizes=(50, 200), legend='brief', hue='hue')
    h,l = ax.get_legend_handles_labels()
    plt.legend(loc='upper left', handles=h[-10:] + [h[7], h[8], h[10], h[12]], bbox_to_anchor=(1, 1))
    plt.savefig('plot_inference_tradeoff_scatter.pdf', bbox_inches="tight")
    
    plt.clf()
    fig, (ax1, bx1, cx1) = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.05, wspace=0.05)
    ax2, bx2, cx2 = ax1.twinx(), bx1.twinx(), cx1.twinx()
    selected_df = df.iloc[[0, 1, 4, 5, 7, 8]]
    selected_t5_df = t5_df
    sns.scatterplot(data=selected_df, x="Relative Accuracy (\\%)", y="Relative Training Speed", size="Relative Training Peak Memory", hue='Method', sizes=(50, 400), legend='brief', ax=ax1, palette=palette)
    plt.subplots_adjust(wspace=0.05)
    h,l = ax1.get_legend_handles_labels()
    # Then remove ax1's legend and put a new one in the figure
    ax1.legend([],[], frameon=False)
    sns.scatterplot(data=selected_df, x="Relative Accuracy (\\%)", y="Inference Speedup", size="Relative Inference memory", sizes=(50, 400), legend=False, hue='Method', ax=ax2, palette=palette, marker='s')
    ax1.set_title('RoBERTa-base SST2')
    # ax2.set_title('RoBERTa-base SST2')
    ax1.set_xlim(96, 100.5)
    # ax2.set_xlim(96, 100.5)
    ax2.grid(None)
    plot_arrow(ax1, rotation=45)
    # plot_arrow(ax2, rotation=45)

    sns.scatterplot(data=selected_t5_df, x="Relative Accuracy (\\%)", y="Relative Training Speed", size="Relative Training Peak Memory", hue='Method', sizes=(50, 400), legend=False, ax=bx1, palette=palette)
    plt.subplots_adjust(wspace=0.05)
    # h,l = ax1.get_legend_handles_labels()
    # Then remove ax1's legend and put a new one in the figure
    # ax1.legend([],[], frameon=False)
    sns.scatterplot(data=selected_t5_df, x="Relative Accuracy (\\%)", y="Inference Speedup", size="Relative Inference memory", sizes=(50, 400), legend=False, hue='Method', ax=bx2, palette=palette, marker='s')
    bx1.set_title('T5-base SST2')
    # bx2.set_title('T5-base SST2')
    bx1.set_xlim(96, 100.5)
    # bx2.set_xlim(96, 100.5)
    bx2.grid(None)
    plot_arrow(bx1, rotation=45)
    # plot_arrow(bx2, rotation=45)
    # plt.legend(loc='upper center', handles=h[-9:], bbox_to_anchor=(-0.1, 1.45), ncol=3)

    selected_llama_df = llama_df
    sns.scatterplot(data=selected_llama_df, x="Relative Accuracy (\\%)", y="Relative Training Speed", size="Relative Training Peak Memory", hue='Method', sizes=(50, 400), legend='brief', ax=cx1, palette=palette)
    plt.subplots_adjust(wspace=0.05)
    h2,l2 = cx1.get_legend_handles_labels()
    # Then remove ax1's legend and put a new one in the figure
    cx1.legend([],[], frameon=False)
    sns.scatterplot(data=selected_llama_df, x="Relative Accuracy (\\%)", y="Inference Speedup", size="Relative Inference memory", sizes=(50, 400), legend=False, hue='Method', ax=cx2, palette=palette, marker='s')
    cx1.set_title('LLaMA2 7B Alpaca')
    # cx2.set_title('LLaMA2 7B Alpaca')
    cx1.set_xlim(60, 105)
    cx2.grid(None)
    # cx2.set_xlim(60, 105)
    plot_arrow(cx1, rotation=45)
    # plot_arrow(cx2, rotation=45)
    
    # plt.legend(loc='upper right', handles=h[:6] + [h2[2], h2[4], h2[5]], bbox_to_anchor=(1.8, 1), ncol=1)
    ax1.legend(loc='center left', handles=h[:6] + [h2[2], h2[4], h2[5]], ncol=1, prop = { "size": 8})
    plt.savefig('plot_tradeoff_scatter.pdf', bbox_inches="tight")
    
    # Vertical version
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 7))
    sns.scatterplot(data=df, x="97\\% TTA (Second)", y="Accuracy", size="Training Peak Memory (MB)", style="Method", hue='hue', sizes=(50, 400), legend='brief', ax=ax1)
    plt.subplots_adjust(hspace=0.3)
    h,l = ax1.get_legend_handles_labels()
    # Then remove ax1's legend and put a new one in the figure
    ax1.legend([],[], frameon=False)
    sns.scatterplot(data=df, x="Inference time", y="Accuracy", size="Inference memory", style="Method", sizes=(50, 200), legend=False, hue='hue', ax=ax2)
    ax1.set_title('Training efficiency and performance trade-off')
    ax2.set_title('Inference efficiency and performance trade-off')
    plt.legend(loc='upper center', handles=h[-9:], bbox_to_anchor=(0.5, 2.6), ncol=3, fontsize=6)
    plt.savefig('plot_tradeoff_scatter.pdf', bbox_inches="tight")
    
    # Make preliminary plot
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2), sharey=True)
    preliminary_df = df[df['Method'].apply(lambda x: 'LoRA' in x or 'APT' in x)][1:]
    sns.scatterplot(data=preliminary_df, x="97\\% TTA (Second)", y="Accuracy", style="Method", hue='hue', sizes=(50, 400), legend=False, ax=ax1)
    scale = max(preliminary_df['97\\% TTA (Second)'].tolist()) - min(preliminary_df['97\\% TTA (Second)'].tolist())
    x_bias = [-0.5, -0.5, -0.4, 0.05]
    y_bias = [0, 0, 0, -0.1]
    for x, y, t, xb, yb in zip(preliminary_df['97\\% TTA (Second)'].tolist(), preliminary_df['Accuracy'].tolist(), preliminary_df['Method'].tolist(), x_bias, y_bias):
        # the position of the data label relative to the data point can be adjusted by adding/subtracting a value from the x &/ y coordinates
        ax1.text(x = x + scale * xb, # x-coordinate position of data label
        y = y + yb, # y-coordinate position of data label, adjusted to be 150 below the data point
        s = t, # data label, formatted to ignore decimals
        size = 7,
        ) # set colour of line
    sns.scatterplot(data=preliminary_df, x="Training Peak Memory (MB)", y="Accuracy", style="Method", sizes=(50, 200), legend=False, hue='hue', ax=ax2)
    scale = max(preliminary_df['Training Peak Memory (MB)'].tolist()) - min(preliminary_df['Training Peak Memory (MB)'].tolist())
    x_bias = [0.05, 0.05, -0.4, 0.05]
    y_bias = [0, 0, 0, -0.1]
    for x, y, t, xb, yb in zip(preliminary_df['Training Peak Memory (MB)'].tolist(), preliminary_df['Accuracy'].tolist(), preliminary_df['Method'].tolist(), x_bias, y_bias):
        # the position of the data label relative to the data point can be adjusted by adding/subtracting a value from the x &/ y coordinates
        ax2.text(x = x + scale * xb, # x-coordinate position of data label
        y = y + yb, # y-coordinate position of data label, adjusted to be 150 below the data point
        s = t, # data label, formatted to ignore decimals
        size = 7,
        ) # set colour of line
    plt.savefig('plot_preliminary_scatter.pdf', bbox_inches="tight")