import os
import torch
import seaborn as sns

from tqdm import tqdm
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 0. Configs
    num_layers = 40
    num_heads = 40
    num_intermediate = 13824
    d_model = 5120
    dim_per_head = 5120 // 40
    
    # 1. Load WANDA weights
    wanda_dir = '/mmfs1/home/bowen98/projects/wanda/wanda_output/llama2_13b/unstructured/wanda'
    wanda_weight_paths = [
        os.path.join(wanda_dir, f) for f in os.listdir(wanda_dir) if f.endswith('.bin') and f.startswith('pytorch_model')
    ]
    wanda_weights = [torch.load(f, map_location='cpu') for f in tqdm(wanda_weight_paths)]
    wanda_weights = {
        k: v for w in wanda_weights for k, v in w.items()
    }
    
    # 2. Load MT masks
    mt_dir = 'llama_output/meta-llama/Llama-2-13b-hf/alpaca_gpt4/bz4/lora/teacher_dq:0-39,dv:0-39/epoch5/lora_r8/lora_alpha16/lr1e-4/seed42/best_model/pruned/constraint_0.5/batches_256'
    mt_head_mask = torch.load(os.path.join(mt_dir, 'pruning_head_mask.pt'), map_location='cpu')
    mt_intermediate_mask = torch.load(os.path.join(mt_dir, 'pruning_intermediate_mask.pt'), map_location='cpu')
    
    # 3. Load APT pre-tuning masks

    
    # 4. Load APT gradual masks
    apt_gradual_head_mask = torch.load('llama_output/meta-llama/Llama-2-13b-hf/alpaca_gpt4/bz4/elastictuning_virtualprune_pre-tuning-prune-kurtosis-0.9-nodistill/mac0.5/epoch15/numprune10/sparsity_warmup1/pruning_start-1/pruning_stop3/lora_r8/lora_alpha16/warmup_paramdq:0-39,dv:0-39/teacher_paramdq:0-39,dv:0-39/final_head_mask.pt', map_location='cpu').view(40, 40)
    apt_gradual_intermediate_mask = torch.load('llama_output/meta-llama/Llama-2-13b-hf/alpaca_gpt4/bz4/elastictuning_virtualprune_pre-tuning-prune-kurtosis-0.9-nodistill/mac0.5/epoch15/numprune10/sparsity_warmup1/pruning_start-1/pruning_stop3/lora_r8/lora_alpha16/warmup_paramdq:0-39,dv:0-39/teacher_paramdq:0-39,dv:0-39/final_intermediate_mask.pt', map_location='cpu').view(40, 13824)
    
    # 5. Compare Wanda with APT
    # 5.1 Where are the overlaps between Wanda and APT?
    # 5.1.1 Wanda q, k, v, o pruning ratios
    attn_ratios = [
        [
            (wanda_weights['model.layers.%d.self_attn.%s.weight' % (i, attr)] == 0).sum().item() / wanda_weights['model.layers.%d.self_attn.%s.weight' % (i, attr)].numel()
            for attr in ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        ]
        for i in tqdm(range(num_layers))
    ] # should be uniformly 0.5
    ffn_ratios = [
        [
            (wanda_weights['model.layers.%d.mlp.%s.weight' % (i, attr)] == 0).sum().item() / wanda_weights['model.layers.%d.mlp.%s.weight' % (i, attr)].numel()
            for attr in ['up_proj', 'gate_proj', 'down_proj']
        ]
        for i in tqdm(range(num_layers))
    ] # should be uniformly 0.5
    
    # 5.1.2 Attn comparison
    wanda_head_prune_ratios = []
    wanda_ffn_prune_ratios = []
    for i in tqdm(range(num_layers)):
        q = wanda_weights['model.layers.%d.self_attn.q_proj.weight' % i]
        k = wanda_weights['model.layers.%d.self_attn.k_proj.weight' % i]
        v = wanda_weights['model.layers.%d.self_attn.v_proj.weight' % i]
        o = wanda_weights['model.layers.%d.self_attn.o_proj.weight' % i].view(-1, num_heads)
        wanda_head_prune_ratios.append(
            (
                (q == 0).sum(dim=0) / d_model,
                (k == 0).sum(dim=0) / d_model,
                (v == 0).sum(dim=0) / d_model,
                (o == 0).sum(dim=0) / (d_model * dim_per_head)
            )
        )
        
        up = wanda_weights['model.layers.%d.mlp.up_proj.weight' % i]
        gate = wanda_weights['model.layers.%d.mlp.gate_proj.weight' % i]
        down = wanda_weights['model.layers.%d.mlp.down_proj.weight' % i]
        wanda_ffn_prune_ratios.append(
            (
                (up == 0).sum(dim=0) / (num_intermediate),
                (gate == 0).sum(dim=0) / (num_intermediate),
                (down == 0).sum(dim=0) / (d_model)
            )
        )
    
    head_ratios = torch.stack([v[-1] for v in wanda_head_prune_ratios])
    ffn_ratios = torch.stack([v[-1] for v in wanda_ffn_prune_ratios])
    
    sns.heatmap(head_ratios.numpy() * 100)
    plt.xlabel('Head')
    plt.ylabel('Layer')
    plt.title('Wanda head in o_proj pruning ratios')
    plt.savefig('head_ratios.png')
    plt.clf()
    
    sns.heatmap(ffn_ratios.numpy() * 100)
    plt.xlabel('FFN')
    plt.ylabel('Layer')
    plt.title('Wanda ffn in down_proj pruning ratios')
    plt.savefig('ffn_ratios.png')
    plt.clf()
    
    head_hidden_dim_ratio = torch.stack([(v[0] + v[1] + v[2]) / 3 for v in wanda_head_prune_ratios])
    ffn_hidden_dim_ratio = torch.stack([(v[0] + v[1]) / 2 for v in wanda_ffn_prune_ratios])
    
    sns.heatmap(head_hidden_dim_ratio.numpy() * 100)
    plt.xlabel('Hidden Dim')
    plt.ylabel('Layer')
    plt.title('Wanda hidden dim in q, k, v pruning ratios')
    plt.savefig('head_hidden_dim_ratios.png')
    plt.clf()
    
    sns.heatmap(ffn_hidden_dim_ratio.numpy() * 100)
    plt.xlabel('Hidden Dim')
    plt.ylabel('Layer')
    plt.title('Wanda hidden dim in up, gate pruning ratios')
    plt.savefig('ffn_hidden_dim_ratios.png')
    plt.clf()
    
    # 5.2 What does APT prune that Wanda doesn't?