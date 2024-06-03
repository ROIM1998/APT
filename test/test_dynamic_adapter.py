import seaborn as sns
sns.set_theme(style="darkgrid")
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
import re
import loralib as lora
import pandas as pd

from transformers import (HfArgumentParser)
from args import DataTrainingArguments
from models.model_args import ModelArguments
from utils.utils import *
from utils.minus_utils import lora_to_prunelora, shrink_pruning_lora, shrink_pruning_lora_outdim, shrink_pruning_lora_indim, shrink_pruning_lora_bottleneckdim, expand_pruning_lora_bottleneckdim, lora_to_linear
from utils import build_trainer, build_dataloader
from args import MinusTrainingArguments
from models import build_model
from prune.fisher import collect_grads_by_suffix
from prune.pruner import AdapterPruner
from torch.utils.data import Subset
from trainer.param_control import ParamController
from ortools.algorithms import pywrapknapsack_solver
from prune import build_pruner, build_scorer, build_pruning_scheduler, BetterFisherPruner
from matplotlib import pyplot as plt
from trainer.allocation_strategy import binary_knapsack_search


def test_expand_or_shrink(prune_ratio: float = 0.5, init_dim: int = 8, init_num: int = 24):
    dims = [torch.ones(init_dim) for i in range(init_num)]
    print(f'init_dims: {[len(v) for v in dims]}')
    for _ in range(10):
        summed = int(sum([v.sum().item() for v in dims]))
        num_pruned = int(summed * prune_ratio)
        mask = torch.ones(summed)
        mask[torch.randperm(summed)[:num_pruned]] = 0
        new_dims = []
        i = 0
        for dim in dims:
            status = mask[i:i+len(dim)]
            if status.all():
                new_dims.append(torch.ones(len(dim) * 2))
            elif status.any():
                new_dims.append(torch.ones(int(len(dim) - (1 - status).sum().item())))
            i += len(dim)
        print(f'new_dims: {[len(v) for v in new_dims]}')
        dims = new_dims
        
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

def collect_weight_saliency(model, dataloader, layer_names):
    all_grads = {}
    for n, p in model.named_parameters():
        if n.rsplit('.', 1)[0] in layer_names and 'weight' in n:
            p.requires_grad = True
            all_grads[n] = []
        else:
            p.requires_grad = False
    model.zero_grad()
    for batch in dataloader:
        for k, v in batch.items():
            batch[k] = v.to(model.device, non_blocking=True) if isinstance(v, torch.Tensor) else torch.stack(v, dim=1).to(model.device) if isinstance(v, list) else v
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        for n, p in model.named_parameters():
            if n in all_grads:
                all_grads[n].append(p.grad.detach())
                p.grad = None
    for n, p in model.named_parameters():
        if n in all_grads:
            all_grads[n] = torch.stack(all_grads[n], dim=0)
    return all_grads

def main():
    sys.argv = ['test_pre_tuning_prune.py',
            '--output_dir',
            './output/test_model_grafting_dynamic_all_dependent_pruned_test/',
            '--model_name_or_path',
            'output/bert-base-uncased_lora_minus_mnli_once_global_free_inout_nodistill/mac0.4/epoch30/bz128/numprune5/paramq:0-11,v:0-11,i:0-11/lora_r8/lora_alpha16/pre_pruning_model',
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
    os.makedirs(training_args.output_dir, exist_ok=True)
    # training_args.disable_tqdm = False
    t_name, raw_datasets = get_raw_datasets(model_args, data_args, training_args)
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)

    model.head_mask, model.intermediate_mask = model.head_mask.to(training_args.device), model.intermediate_mask.to(training_args.device)
    # trainer.evaluate()

    pruning_batch_size = 4
    num_pruning_batches = 64
    dataloader = build_dataloader(Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]), pruning_batch_size, data_args, training_args, tokenizer)

    # Also add ffn input layers to teacher config
    teacher_keys = ['query', 'value', 'intermediate']
    teacher_config  = {
        k: [i for i in range(model.config.num_hidden_layers)] for k in teacher_keys
    }

    adapter_pruner = AdapterPruner(model, dataloader)
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        student_config=None,
        lora_with_bias=False,
        adapter_pruner=adapter_pruner,
    )
    param_controller.convert_to_pre_pruning_lora_teacher()
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    param_controller.model_as_teacher()
    tuning_param_num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    
    # Pruning the model before re-allocating the dimensions
    training_args.seq_len = 128
    training_args.cls_task = True
    scorer = build_scorer('gradient_l2', model, dataloader)
    pruner = BetterFisherPruner(model, ['head_mask', 'intermediate_mask'], {'head_mask': scorer, 'intermediate_mask': scorer}, training_args.seq_len, training_args.cls_task, ['search', 'better_rearrange', 'global'])

    pre_prune_rank_df = violin_plot_by_suffix(model, dataloader, '.bottleneck_mask', 'pre_prune')
    pre_prune_input_df = violin_plot_by_suffix(model, dataloader, '.input_mask', 'pre_prune')
    pre_prune_output_df = violin_plot_by_suffix(model, dataloader, '.output_mask', 'pre_prune')
    
    # fulfill fine-tuned model input and output score calculation (set grafting masks before and after the actual weight)
    test_ft_model = False
    if test_ft_model:
        roberta_regex = r'^roberta\.encoder\.layer\.(\d+?)\.attention\.self\.(.+?)\..+$'
        model_args.model_name_or_path = 'output/roberta-base_mnli_full/epoch1/bz128'
        config, tokenizer, ft_model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
        ft_model.head_mask, ft_model.intermediate_mask = None, None
        layer_names = [n for n, p in ft_model.named_modules() if 'query' in n or 'value' in n]
        all_ft_grads = collect_weight_saliency(model, dataloader, layer_names)
        all_ft_input_grads = {k: v.sum(dim=1) for k, v in all_ft_grads.items()}
        all_ft_input_scores = {k: v.pow(2).sum(dim=0) for k, v in all_ft_input_grads.items()}
        all_ft_output_grads = {k: v.sum(dim=2) for k, v in all_ft_grads.items()}
        all_ft_output_scores = {k: v.pow(2).sum(dim=0) for k, v in all_ft_output_grads.items()}
        ft_input_scores = [
            {
                'layer_id': int(re.match(roberta_regex, k).group(1)),
                'attr': re.match(roberta_regex, k).group(2),
                'score': np.log10(val.item()),
            }
            for k, v in all_ft_input_scores.items() for val in v
        ]
        score_df = pd.DataFrame(ft_input_scores)
        sns.violinplot(data=score_df, x='layer_id', y='score', hue='attr', split=True)
        plt.savefig('%s_%s_scores_violin.png' % ('ft_pre_prune', 'input'))
        plt.clf()
        ft_output_scores = [
            {
                'layer_id': int(re.match(roberta_regex, k).group(1)),
                'attr': re.match(roberta_regex, k).group(2),
                'score': np.log10(val.item()),
            }
            for k, v in all_ft_output_scores.items() for val in v
        ]
        score_df = pd.DataFrame(ft_output_scores)
        sns.violinplot(data=score_df, x='layer_id', y='score', hue='attr', split=True)
        plt.savefig('%s_%s_scores_violin.png' % ('ft_pre_prune', 'output'))
        plt.clf()
        ft_scorer = build_scorer('gradient_l2', ft_model, dataloader)
        ft_pruner = BetterFisherPruner(ft_model, ['head_mask', 'intermediate_mask'], {'head_mask': ft_scorer, 'intermediate_mask': ft_scorer}, training_args.seq_len, training_args.cls_task, ['search', 'better_rearrange', 'global'])
        ft_model.reset_masks()
        ft_masks = ft_pruner.generate_mask(0.4)
    
    masks = pruner.generate_mask(0.4)
    head_mask, intermediate_mask = model.head_mask.clone(), model.intermediate_mask.clone()
    model.prune_model_with_masks()
    # bottleneck_names, output_dim_masks, all_bottleneck_mask, input_dim_masks, target_rs, all_scores = param_controller.allocate_dims(0.8)
    named_modules = dict(model.named_modules())
    for n, p in model.named_modules():
        if isinstance(p, lora.PruningLinear):
            p.set_grafting_mask()
            
    # Collecting mask saliences
    bottleneck_grads = collect_grads_by_suffix(model, dataloader, '.bottleneck_mask')
    input_grads = collect_grads_by_suffix(model, dataloader, '.input_mask')
    output_grads = collect_grads_by_suffix(model, dataloader, '.output_mask')

    torch.cat([v.view(-1) for v in bottleneck_grads.values()]).abs().mean()
    torch.cat([v.view(-1) for v in input_grads.values()]).abs().mean()
    torch.cat([v.view(-1) for v in output_grads.values()]).abs().mean()
    

    bottleneck_names, output_dim_masks, all_bottleneck_mask, input_dim_masks, target_rs, all_scores = param_controller.allocate_dims(0, 0.8, final_head_mask=None, final_intermediate_mask=None, alloc_target='student')
    current_tuning_param_num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    pruned_tuning_param_num = tuning_param_num - current_tuning_param_num

    post_prune_rank_df = violin_plot_by_suffix(model, dataloader, '.bottleneck_mask', 'post_prune')
    post_prune_input_df = violin_plot_by_suffix(model, dataloader, '.input_mask', 'post_prune')
    post_prune_output_df = violin_plot_by_suffix(model, dataloader, '.output_mask', 'post_prune')
    
    # Edition: test better fisher pruning instead of a simple search
    # Question to solve: how many dimensions to expand, if needed
    # Grads shape: [num_pruning_batches, num_blocks]
    bottleneck_names, all_bottleneck_mask, all_bottleneck_grads = param_controller.adapter_pruner.prune_by_suffix(0.8, '.bottleneck_mask')
    # Try adding pseudo bottleneck dimensions to expand some of the layers
    relative_ratio = current_tuning_param_num / tuning_param_num
    if relative_ratio > 1:
        decay_factor = (1 / relative_ratio) ** 0.5
    else:
        decay_factor = (1 / relative_ratio)
    # Add pseudo bottleneck dimensions to the layers, with their scores set as the average of the corresponding layers' scores
    # Only adding pseudo bottleneck dimensions if all of the current dimensions are relatively high
    all_bottleneck_scores = {k: v.pow(2).sum(dim=0) for k, v in all_bottleneck_grads.items()}
    all_bottleneck_scores = {
        k: torch.cat([v, torch.tensor([v.mean() - v.std()] * int(v.numel() * 0.5 * decay_factor), dtype=v.dtype, device=v.device)])
        for k, v in all_bottleneck_scores.items()
    }
    # Using a single binary knapsack search to find the optimal bottleneck allocation
    named_modules = dict(model.named_modules())
    weights_tensor = torch.tensor([int(named_modules[k.rsplit('.', 1)[0]].in_features + named_modules[k.rsplit('.', 1)[0]].out_features) for k in all_bottleneck_scores for _ in range(all_bottleneck_scores[k].numel())])
    lens = [len(all_bottleneck_scores[k]) for k in all_bottleneck_scores]
    capacities = [int(current_tuning_param_num * decay_factor)]
    # Using our customized search function (solve it quicker instead of better)
    values_tensor = torch.cat([v for v in all_bottleneck_scores.values()]).cpu()
    all_pseudo_bottleneck_mask = binary_knapsack_search(values_tensor, weights_tensor, capacities)
    all_pseudo_bottleneck_mask = torch.split(all_pseudo_bottleneck_mask, lens)
    target_lora_rs = {
        k.rsplit('.', 1)[0]: int(m.sum().item())
        for k, m in zip(bottleneck_names, all_pseudo_bottleneck_mask)
    }
    # Then fix the bottleneck dimensions and search for the input-output dimension masks
    # Set the masks back to get the dependent results
    new_bottleneck_masks = []
    for original_mask, pseudo_mask in zip(all_bottleneck_mask, all_pseudo_bottleneck_mask):
        mask = torch.zeros(original_mask.numel()).to(original_mask.device)
        mask[pseudo_mask[:original_mask.numel()].nonzero().squeeze()] = 1
        new_bottleneck_masks.append(mask)
    dependent=True
    if dependent:
        for name, mask in zip(bottleneck_names, new_bottleneck_masks):
            layer_name = name.rsplit('.', 1)[0]
            named_modules[layer_name].bottleneck_mask = torch.nn.Parameter(mask, requires_grad=False)
    output_grads = collect_grads_by_suffix(model, adapter_pruner.dataloader, '.output_mask')
    output_scores = {k: v.pow(2).sum(dim=0) for k, v in output_grads.items() if 'output_mask' in k}
    input_grads = collect_grads_by_suffix(model, adapter_pruner.dataloader, '.input_mask')
    input_scores = {k: v.pow(2).sum(dim=0) for k, v in input_grads.items() if 'input_mask' in k}
    weights = torch.tensor([int(target_lora_rs[k.rsplit('.', 1)[0]]) for k in output_scores for _ in range(output_scores[k].numel())] + [int(target_lora_rs[k.rsplit('.', 1)[0]]) for k in input_scores for _ in range(input_scores[k].numel())])
    values_tensor = torch.cat([v for v in output_scores.values()] + [v for v in input_scores.values()]).cpu()
    lens = [len(output_scores[k]) for k in output_scores] + [len(input_scores[k]) for k in input_scores]
    capacities = [tuning_param_num]
    dim_masks = binary_knapsack_search(values_tensor, weights, capacities)
    dim_masks = dim_masks.int().tolist()
    splitted_dim_masks = torch.split(torch.tensor(dim_masks), lens)
    output_dim_masks = splitted_dim_masks[:len(output_scores)]
    input_dim_masks = splitted_dim_masks[len(output_scores):]
    
    pruning_bottleneck_param_num = sum([
        (1 - v).sum() * (named_modules[name.rsplit('.', 1)[0]].lora_A.shape[1] + named_modules[name.rsplit('.', 1)[0]].lora_B.shape[0])
        for name, v in zip(bottleneck_names, all_bottleneck_mask)
    ]).item()
    extra_bottleneck_params = sum([
        v.all().int() * v.numel() * (named_modules[name.rsplit('.', 1)[0]].lora_A.shape[1] + named_modules[name.rsplit('.', 1)[0]].lora_B.shape[0])
        for name, v in zip(bottleneck_names, all_bottleneck_mask)
    ]).item()
    expanding_scale = int((pruning_bottleneck_param_num + pruned_tuning_param_num) // extra_bottleneck_params + 2)
    print(f'expanding_scale: {expanding_scale}')
    target_lora_rs = {
        k.rsplit('.', 1)[0]: expanding_scale * m.numel() if m.all() else m.sum().item()
        for k, m in zip(bottleneck_names, all_bottleneck_mask)
    }
    # Edition: set the bottleneck masks back to the layers, so the input & output dimension selection will be settled depdent on the bottleneck pruning
    old_output_grads = collect_grads_by_suffix(model, dataloader, '.output_mask')
    old_output_scores = {k: v.pow(2).sum(dim=0) for k, v in old_output_grads.items() if 'intermediate' not in k and 'output_mask' in k}
    old_input_grads = collect_grads_by_suffix(model, dataloader, '.input_mask')
    old_input_scores = {k: v.pow(2).sum(dim=0) for k, v in old_input_grads.items() if 'intermediate' not in k and 'input_mask' in k}
    plt.hist(np.log10(torch.cat([v for v in old_output_scores.values()]).cpu().numpy()), bins=100)
    plt.savefig('old_output_scores.png')
    plt.clf()
    plt.hist(np.log10(torch.cat([v for v in old_input_scores.values()]).cpu().numpy()), bins=100)
    plt.savefig('old_input_scores.png')
    plt.clf()

    named_modules = dict(model.named_modules())
    for name, mask in zip(bottleneck_names, all_bottleneck_mask):
        layer_name = name.rsplit('.', 1)[0]
        named_modules[layer_name].bottleneck_mask = torch.nn.Parameter(mask, requires_grad=False)
    output_grads = collect_grads_by_suffix(model, dataloader, '.output_mask')
    output_scores = {k: v.pow(2).sum(dim=0) for k, v in output_grads.items() if 'intermediate' not in k and 'output_mask' in k}
    input_grads = collect_grads_by_suffix(model, dataloader, '.input_mask')
    input_scores = {k: v.pow(2).sum(dim=0) for k, v in input_grads.items() if 'intermediate' not in k and 'input_mask' in k}
    
    # Group the grads by LoRA layers
    all_grads = {}
    for name in all_bottleneck_grads:
        layer_name = name.rsplit('.', 1)[0]
        bottleneck_grad, output_grad, input_grad = all_bottleneck_grads[layer_name + '.bottleneck_mask'], output_grads[layer_name + '.output_mask'], input_grads[layer_name + '.input_mask']
        all_grads[name] = {
            'bottleneck': bottleneck_grad,
            'output': output_grad,
            'input': input_grad,
        }
    
    # Calculate the remaining capacities when lora_A are fixed
    # Firstly, find the extra tuning parameters involved by expaning the bottleneck dimensions
    # tuning_param_lora_a = sum([768 * int(target_lora_rs[k.rsplit('.', 1)[0]]) for k, v in output_scores.items()])
    # tuning_param_left_for_b = tuning_param_num - tuning_param_lora_a
    # The cost of each output dimension is exactly the lora_r corresponding to it
    # Solve it output_dimension mask as the knapsack problem
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 
        'KnapsackExample'
    )
    values = (torch.cat([v for v in output_scores.values()] + [v for v in input_scores.values()]) * 1e8).int().tolist()
    scaled_values = (torch.cat([v for v in output_scores.values()] + [v for v in input_scores.values()]) * 1e10).int().tolist()
    weights = [[int(target_lora_rs[k.rsplit('.', 1)[0]]) for k in output_scores for _ in range(output_scores[k].numel())] + [int(target_lora_rs[k.rsplit('.', 1)[0]]) for k in input_scores for _ in range(input_scores[k].numel())]]
    lens = [len(output_scores[k]) for k in output_scores] + [len(input_scores[k]) for k in input_scores]
    capacities = [tuning_param_num]

    # Using our customized search function (solve it quicker instead of better)
    values_tensor = torch.cat([v for v in output_scores.values()] + [v for v in input_scores.values()]).cpu()
    weights_tensor = torch.tensor(weights[0])
    sorted_values, sorted_indices = torch.sort(values_tensor, descending=True)
    # Binary search for the best solution
    max_i, min_i = len(sorted_values), 0
    next_i = len(sorted_values) // 2
    while max_i - min_i > 1:
        selected_indices = sorted_indices[:next_i]
        weights_sum = torch.index_select(weights_tensor, 0, selected_indices).sum()
        if weights_sum < capacities[0]:
            min_i = next_i
        else:
            max_i = next_i
        next_i = (max_i + min_i) // 2
    if weights_sum > capacities[0]:
        next_i -= 1
    dim_masks = torch.zeros(len(sorted_values))
    dim_masks[sorted_indices[:next_i]] = 1
    dim_masks = dim_masks.int().tolist()

    # solver.Init(values, weights, capacities)
    # computed_value = solver.Solve()
    # packed_items = []
    # packed_weights = []
    # total_weight = 0
    # dim_masks = [] 
    # print('Total value =', computed_value)
    # for i in range(len(values)):
    #     if solver.BestSolutionContains(i):
    #         packed_items.append(i)
    #         packed_weights.append(weights[0][i])
    #         total_weight += weights[0][i]
    #         dim_masks.append(1)
    #     else:
    #         dim_masks.append(0)
    splitted_dim_masks = torch.split(torch.tensor(dim_masks), lens)
    output_dim_masks = splitted_dim_masks[:len(output_scores)]
    input_dim_masks = splitted_dim_masks[len(output_scores):]

    for bottleneck_name, input_mask, output_mask, bottleneck_mask in zip(bottleneck_names, input_dim_masks, output_dim_masks, all_bottleneck_mask):
        layer_name = bottleneck_name.rsplit('.', 1)[0]
        layer = named_modules[layer_name]
        parent_layer_name, layer_attr = layer_name.rsplit('.', 1)
        parent_layer = named_modules[parent_layer_name]
        if not output_mask.any() or not bottleneck_mask.any():
            # mask all equals to 0
            shrinked_layer = lora_to_linear(layer)
        else:
            pruned_bottleneck_dim = (bottleneck_mask == 0).nonzero().squeeze()
            pruned_out_dim = (output_mask == 0).nonzero().squeeze()
            pruned_in_dim = (input_mask == 0).nonzero().squeeze()
            pruned_bottleneck_dim = pruned_bottleneck_dim.tolist() if pruned_bottleneck_dim.dim() else [pruned_bottleneck_dim.item()]
            pruned_out_dim = pruned_out_dim.tolist() if pruned_out_dim.dim() else [pruned_out_dim.item()]
            pruned_in_dim = pruned_in_dim.tolist() if pruned_in_dim.dim() else [pruned_in_dim.item()]
            if bottleneck_mask.all():
                if output_mask.all():
                    shrinked_layer = layer
                else:
                    shrinked_layer = shrink_pruning_lora_outdim(layer, pruned_out_dim)
                if not input_mask.all():
                    shrinked_layer = shrink_pruning_lora_indim(shrinked_layer, pruned_in_dim)
                shrinked_layer = expand_pruning_lora_bottleneckdim(shrinked_layer, layer.r * expanding_scale)
            else:
                if output_mask.all():
                    shrinked_layer = shrink_pruning_lora_bottleneckdim(layer, pruned_bottleneck_dim)
                else:
                    shrinked_layer = shrink_pruning_lora(layer, pruned_out_dim, pruned_bottleneck_dim, pruned_in_dim)   
        setattr(parent_layer, layer_attr, shrinked_layer)
    
    param_controller.set_grafting_mask(mode=False, target='teacher')
    param_controller.clear_states()
    param_controller.model_as_teacher()
    print("Tuning parameter number:", sum([p.numel() for p in model.parameters() if p.requires_grad]))
    print("LoRA parameter number:", sum([p.numel() for n, p in model.named_parameters() if 'lora' in n]))
    for i in range(12):
        print("query r in layer %d:" % i, model.roberta.encoder.layer[i].attention.self.query.r if hasattr(model.roberta.encoder.layer[i].attention.self.query, 'r') else 0, 'query input dim: ', model.roberta.encoder.layer[i].attention.self.query.lora_A.shape[1] if hasattr(model.roberta.encoder.layer[i].attention.self.query, 'lora_A') else 0, 'query output dim: ', model.roberta.encoder.layer[i].attention.self.query.lora_B.shape[0] if hasattr(model.roberta.encoder.layer[i].attention.self.query, 'lora_B') else 0)
        print("value r in layer %d:" % i, model.roberta.encoder.layer[i].attention.self.value.r if hasattr(model.roberta.encoder.layer[i].attention.self.value, 'r') else 0, 'value input dim: ', model.roberta.encoder.layer[i].attention.self.value.lora_A.shape[1] if hasattr(model.roberta.encoder.layer[i].attention.self.value, 'lora_A') else 0, 'value output dim: ', model.roberta.encoder.layer[i].attention.self.value.lora_B.shape[0] if hasattr(model.roberta.encoder.layer[i].attention.self.value, 'lora_B') else 0)

    trainer.auto_layer_conversion = False
    # trainer.dynamic_grafting = True
    train_result = trainer.train()

if __name__ == '__main__':
    main()