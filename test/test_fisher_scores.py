import seaborn as sns
sns.set_theme(style="darkgrid")
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
import loralib as lora

from transformers import (HfArgumentParser)
from args import DataTrainingArguments
from models.model_args import ModelArguments
from utils.utils import *
from utils.minus_utils import lora_to_prunelora, lora_to_linear
from utils import build_trainer, build_dataloader
from args import MinusTrainingArguments
from models import build_model
from prune.fisher import collect_grads_by_suffix, collect_mask_grads
from prune import build_scorer, BetterFisherPruner, AdapterPruner
from torch.utils.data import Subset
from trainer.param_control import ParamController
from matplotlib import pyplot as plt

def main():
    sys.argv = ['test_fisher_scores.py',
            '--output_dir',
            './output/test_fisher_scores/',
            '--model_name_or_path',
            '/data0/bowen/minus/bert_base_mnli_pre_pruning_model',
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
    
    named_modules = dict(model.named_modules())
    for n, p in model.named_modules():
        if isinstance(p, lora.Linear):
            parent_layer_attr, attr = n.rsplit('.', 1)
            parent_layer = named_modules[parent_layer_attr]
            if 'intermediate' in n:
                setattr(parent_layer, attr, lora_to_linear(p))
            else:
                new_layer = lora_to_prunelora(p, r=p.r, lora_alpha=p.lora_alpha)
                new_layer.set_grafting_mask()
                setattr(parent_layer, attr, new_layer)

    model.head_mask, model.intermediate_mask = model.head_mask.to(training_args.device), model.intermediate_mask.to(training_args.device)
    # trainer.evaluate()

    pruning_batch_size = 4
    num_pruning_batches = 64
    dataloader = build_dataloader(Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]), pruning_batch_size, data_args, training_args, tokenizer)

    teacher_keys = ['query', 'value']
    teacher_config  = {
        k: [i for i in range(model.config.num_hidden_layers)] for k in teacher_keys
    }
    student_keys = ['query', 'value', 'intermediate']
    student_config = {
        k: [i for i in range(model.config.num_hidden_layers)] for k in student_keys
    }
    adapter_pruner = AdapterPruner(model, dataloader)
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        student_config=student_config,
        lora_with_bias=False,
        adapter_pruner=adapter_pruner,
    )
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    param_controller.model_as_teacher()
    tuning_param_num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    
    # Pruning the model before re-allocating the dimensions
    training_args.seq_len = 128
    training_args.cls_task = True
    for p in model.parameters():
            p.requires_grad = False
    
    head_grads, intermediate_grads = collect_mask_grads(model, dataloader)

    # bottleneck_names, output_dim_masks, all_bottleneck_mask, input_dim_masks, target_rs , all_scores= param_controller.allocate_dims(0.8)
    named_modules = dict(model.named_modules())
    for n, p in model.named_modules():
        if isinstance(p, lora.PruningLinear):
            p.set_grafting_mask()
    param_controller.clear_states()
    param_controller.model_as_teacher()
    pruned_tuning_param_num = tuning_param_num - sum([p.numel() for p in model.parameters() if p.requires_grad])
    
    old_bottleneck_grads = collect_grads_by_suffix(model, dataloader, '.bottleneck_mask')
    old_bottleneck_scores = {k: v.pow(2).sum(dim=0) for k, v in old_bottleneck_grads.items() if 'intermediate' not in k and 'bottleneck_mask' in k}
    plt.hist(np.log10(torch.cat([v for v in old_bottleneck_scores.values()]).cpu().numpy()), bins=100)
    plt.savefig('old_bottleneck_scores.png')
    plt.clf()

    # Edition: set the bottleneck masks back to the layers, so the input & output dimension selection will be settled depdent on the bottleneck pruning
    old_output_grads = collect_grads_by_suffix(model, dataloader, '.output_mask')
    old_output_scores = {k: v.pow(2).sum(dim=0) for k, v in old_output_grads.items() if 'intermediate' not in k and 'output_mask' in k}
    old_input_grads = collect_grads_by_suffix(model, dataloader, '.input_mask')
    old_input_scores = {k: v.pow(2).sum(dim=0) for k, v in old_input_grads.items() if 'intermediate' not in k and 'input_mask' in k}
    
    # Testing the correlation between the tuning parameters' fisher information and the overall parameters' fisher information
    scorer = build_scorer('gradient_l2', model, dataloader)
    pruner = BetterFisherPruner(model, ['head_mask', 'intermediate_mask'], {'head_mask': scorer, 'intermediate_mask': scorer}, training_args.seq_len, training_args.cls_task, ['search', 'better_rearrange', 'global'])
    masks = pruner.generate_mask(0.4)
    
    merged_head_grads = scorer.head_grads
    merged_output_grads = torch.stack([
        (old_output_scores['roberta.encoder.layer.%d.attention.self.query.output_mask' % i] + old_output_scores['roberta.encoder.layer.%d.attention.self.value.output_mask' % i]).view(model.config.num_attention_heads, -1)
        for i in range(merged_head_scores.shape[0])
    ]).permute(2, 0, 1)
    merged_freeze_grads = merged_head_grads - merged_output_grads
    
    merged_head_scores = scorer.head_score()
    merged_output_scores = merged_output_grads.pow(2).sum(dim=0)
    merged_freeze_scores = merged_freeze_grads.pow(2).sum(dim=0)
    
    sns.scatterplot(x=np.log10(merged_head_scores.view(-1).cpu().numpy()), y=np.log10(merged_output_scores.view(-1).cpu().numpy()), hue=masks['head_mask'].long().view(-1).cpu().numpy())
    plt.xlabel('Overall scores')
    plt.ylabel('Tuning scores')
    plt.savefig('all_vs_tuning.png')
    plt.clf()
    sns.scatterplot(x=np.log10(merged_freeze_scores.view(-1).cpu().numpy()), y=np.log10(merged_output_scores.view(-1).cpu().numpy()), hue=masks['head_mask'].long().view(-1).cpu().numpy())
    plt.xlabel('Freeze scores')
    plt.ylabel('Tuning scores')
    plt.savefig('freeze_vs_tuning.png')
    plt.clf()
    sns.scatterplot(x=merged_head_scores.view(-1).cpu().numpy(), y=merged_freeze_scores.view(-1).cpu().numpy(), hue=masks['head_mask'].long().view(-1).cpu().numpy())
    plt.xlabel('Overall scores')
    plt.ylabel('Freeze scores')
    plt.savefig('all_vs_freeze.png')
    plt.clf()
    
    
    plt.hist(np.log10(torch.cat([v for v in old_output_scores.values()]).cpu().numpy()), bins=100)
    plt.savefig('old_output_scores.png')
    plt.clf()
    plt.hist(np.log10(torch.cat([v for v in old_input_scores.values()]).cpu().numpy()), bins=100)
    plt.savefig('old_input_scores.png')
    plt.clf()
 

if __name__ == '__main__':
    main()