import seaborn as sns
sns.set_theme(style="darkgrid")
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
import time
import pandas as pd
import traceback

from transformers import (HfArgumentParser, EvalPrediction, default_data_collator, DataCollatorWithPadding)
from args import DataTrainingArguments
from models.model_args import ModelArguments
from trainer.trainer_minus import MinusTrainer
from utils.utils import *
from utils.minus_utils import sum_fisher_score
from args import MinusTrainingArguments
from models import build_model
from datasets import load_metric
from prune.scorer import GradientScorer
from prune.pruner import FisherPruner, BetterFisherPruner
from prune.search import search_mac
from torch.utils.data import DataLoader, Subset
from matplotlib import pyplot as plt

def test_mask_strategy(model, dataloader, trainer, flop_constraint=0.6):
    scorer = GradientScorer(model, dataloader)
    search_pruner = BetterFisherPruner(model, ['intermediate_mask', 'head_mask'], {'intermediate_mask': scorer, 'head_mask': scorer}, 128, True, ['search'])
    fisher_pruner = FisherPruner(model, ['intermediate_mask', 'head_mask'], {'intermediate_mask': scorer, 'head_mask': scorer}, 128, True)
    better_fisher_pruner = BetterFisherPruner(model, ['intermediate_mask', 'head_mask'], {'intermediate_mask': scorer, 'head_mask': scorer}, 128, True, ['search', 'rearrange', 'better_rearrange'])
    better_shorter_pruner = BetterFisherPruner(model, ['intermediate_mask', 'head_mask'], {'intermediate_mask': scorer, 'head_mask': scorer}, 128, True, ['search', 'better_rearrange'])
    layerwise_rearrange_pruner = BetterFisherPruner(model, ['intermediate_mask', 'head_mask'], {'intermediate_mask': scorer, 'head_mask': scorer}, 128, True, ['search', 'better_rearrange', 'layerwise_rearrange'])
    layerwise_shorter_pruner = BetterFisherPruner(model, ['intermediate_mask', 'head_mask'], {'intermediate_mask': scorer, 'head_mask': scorer}, 128, True, ['search', 'layerwise_rearrange'])
    
    global_pruner = BetterFisherPruner(model, ['intermediate_mask', 'head_mask'], {'intermediate_mask': scorer, 'head_mask': scorer}, 128, True, ['search', 'better_rearrange', 'global'])
    global_layerwise_pruner = BetterFisherPruner(model, ['intermediate_mask', 'head_mask'], {'intermediate_mask': scorer, 'head_mask': scorer}, 128, True, ['search', 'better_rearrange', 'layerwise_rearrange', 'global'])
    
    results = []
    for name, pruner in zip(['search', 'fisher', 'better_fisher', 'better_shorter', 'layerwise_rearrange', 'layerwise_shorter', 'global', 'global_layerwise'], [search_pruner, fisher_pruner, better_fisher_pruner, better_shorter_pruner, layerwise_rearrange_pruner, layerwise_shorter_pruner, global_pruner, global_layerwise_pruner]):
        start_time = time.time()
        fisher_masks = pruner.generate_mask(flop_constraint=flop_constraint)
        pruning_time = time.time() - start_time
        model.head_mask, model.intermediate_mask = fisher_masks['head_mask'], fisher_masks['intermediate_mask']
        eval_metrics = trainer.evaluate()
        fisher_score = sum_fisher_score(scorer.head_grads, scorer.intermediate_grads, model.head_mask, model.intermediate_mask)
        results.append({
            'eval_loss': eval_metrics['eval_loss'],
            'eval_accuracy': eval_metrics['eval_accuracy'],
            'fisher_score': fisher_score,
            'label_type': name,
            'pruning_time': pruning_time,
            'batch_size': dataloader.batch_size,
            'num_batches': len(dataloader),
        })
        print(f'{name} finished, with loss {eval_metrics["eval_loss"]}, accuracy {eval_metrics["eval_accuracy"]}, fisher score {fisher_score}, pruning time {pruning_time}')
    
    return results


def main():
    sys.argv = ['test_fisher_pruning.py',
            '--output_dir',
            './output/neuron_importance/',
            '--model_name_or_path',
            '/home/zbw/projects/AdaptPruning/old_output/roberta-base_lora_mnli/epoch30/bz128/lora_r8/lora_alpha16/',
            '--task_name',
            'mnli',
            '--do_train',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '32',
            '--per_device_eval_batch_size',
            '32',
            '--apply_lora',
            '--lora_r',
            '8',
            '--do_distill',
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
    
    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("./glue_metric.py", data_args.task_name)
    else:
        metric = load_metric("accuracy")
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
    
    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    
    model = model.to(training_args.device)
    model.head_mask, model.intermediate_mask = model.head_mask.to(training_args.device), model.intermediate_mask.to(training_args.device)
    # model.head_mask = torch.ones([model.config.num_hidden_layers, model.config.num_attention_heads], device=model.device)
    # model.intermediate_mask = torch.ones([model.config.num_hidden_layers, model.config.intermediate_size], device=model.device)
    trainer = MinusTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    benchmark_fisher = True
    if benchmark_fisher:
        results = []
        for flop_constraint in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
            pruning_batch_size = 32
            num_pruning_batches = 64
            dataloader = DataLoader(
                Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]),
                batch_size=pruning_batch_size,
                collate_fn=data_collator,
            )
            scorer = GradientScorer(model, dataloader)
            pruner = FisherPruner(model, ['intermediate_mask', 'head_mask'], {'intermediate_mask': scorer, 'head_mask': scorer}, 128, True, do_rescale=True)
            fisher_masks = pruner.generate_mask(flop_constraint=flop_constraint)
            model.head_mask, model.intermediate_mask = fisher_masks['head_mask'], fisher_masks['intermediate_mask']
            eval_metrics = trainer.evaluate()
            results.append({
                'eval_loss': eval_metrics['eval_loss'],
                'eval_accuracy': eval_metrics['eval_accuracy'],
                'flop_constraint': flop_constraint,
            })
        
    
    accumulated_results = []
    for pruning_batch_size in [1, 4, 16, 64]:
        for num_pruning_batches in [32, 64, 128, 256]:
            for i in range(5):
                dataloader = DataLoader(
                    Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]),
                    batch_size=pruning_batch_size,
                    collate_fn=data_collator,
                )
                try:
                    results = test_mask_strategy(model, dataloader, trainer)
                    accumulated_results += [
                        {
                            **r,
                            'pruning_batch_size': pruning_batch_size,
                            'num_pruning_batches': num_pruning_batches,
                        } for r in results
                    ]
                except Exception as e:
                    traceback.print_exc()
    df = pd.DataFrame(accumulated_results)
    df['fisher_score'] = df['fisher_score'].apply(lambda x: float(x.split('(')[1].split(',')[0]) if isinstance(x, str) else x.item())
    df['log_fisher_score'] = np.log(df['fisher_score'])
    df.to_csv('results.csv', index=False)
    
    do_random_testing = False
    if do_random_testing:
        results = []
        scorer = GradientScorer(model, dataloader)
        fisher_pruner = FisherPruner(model, ['intermediate_mask', 'head_mask'], {'intermediate_mask': scorer, 'head_mask': scorer}, 128, True)
        fisher_masks = fisher_pruner.generate_mask()
        head_grads, intermediate_grads = scorer.head_grads, scorer.intermediate_grads
        searched_head_mask, searched_intermediate_mask = search_mac(model.config, scorer.head_score(), scorer.intermediate_score(), 128, 0.6)
        model.head_mask, model.intermediate_mask = searched_head_mask, searched_intermediate_mask
        eval_metrics = trainer.evaluate()
        fisher_score = sum_fisher_score(head_grads, intermediate_grads, model.head_mask, model.intermediate_mask)
        results.append({
            'eval_loss': eval_metrics['eval_loss'],
            'eval_accuracy': eval_metrics['eval_accuracy'],
            'fisher_score': fisher_score,
            'label_type': 'searched'
        })
        
        model.head_mask, model.intermediate_mask = fisher_masks['head_mask'], fisher_masks['intermediate_mask']
        eval_metrics = trainer.evaluate()
        fisher_score = sum_fisher_score(head_grads, intermediate_grads, model.head_mask, model.intermediate_mask)
        results.append({
            'eval_loss': eval_metrics['eval_loss'],
            'eval_accuracy': eval_metrics['eval_accuracy'],
            'fisher_score': fisher_score,
            'label_type': 'rearranged'
        })
        for i in range(100):
            pruned_heads, unpruned_heads = (model.head_mask == 0).nonzero().squeeze(), (model.head_mask != 0).nonzero().squeeze()
            pruned_intermdiate, unpruned_intermediate = (model.intermediate_mask == 0).nonzero().squeeze(), (model.intermediate_mask != 0).nonzero().squeeze()
            select_pruned_head, select_pruned_intermediate, select_unpruned_head, select_unpruned_intermediate = pruned_heads[torch.randint(len(pruned_heads), (1,))].squeeze(), pruned_intermdiate[torch.randint(len(pruned_intermdiate), (1,))].squeeze(), unpruned_heads[torch.randint(len(unpruned_heads), (1,))].squeeze(), unpruned_intermediate[torch.randint(len(unpruned_intermediate), (1,))].squeeze()
            model.head_mask[select_pruned_head[0], select_pruned_head[1]], model.intermediate_mask[select_pruned_intermediate[0], select_pruned_intermediate[1]], model.head_mask[select_unpruned_head[0], select_unpruned_head[1]], model.intermediate_mask[select_unpruned_intermediate[0], select_unpruned_intermediate[1]] = 1, 1, 0, 0
            eval_metrics = trainer.evaluate()
            fisher_score = sum_fisher_score(head_grads.cpu(), intermediate_grads.cpu(), model.head_mask.cpu(), model.intermediate_mask.cpu())
            results.append({
                'eval_loss': eval_metrics['eval_loss'],
                'eval_accuracy': eval_metrics['eval_accuracy'],
                'fisher_score': fisher_score,
                'label_type': 'test'
            })
            model.head_mask[select_pruned_head[0], select_pruned_head[1]], model.intermediate_mask[select_pruned_intermediate[0], select_pruned_intermediate[1]], model.head_mask[select_unpruned_head[0], select_unpruned_head[1]], model.intermediate_mask[select_unpruned_intermediate[0], select_unpruned_intermediate[1]] = 0, 0, 1, 1
    df = pd.DataFrame(results)
    df['log_fisher_score'] = np.log10(df['fisher_score'])
    # sns.scatterplot(data=df, x='eval_loss', y='fisher_score', hue='label_type')
    df.to_csv('results.csv')

if __name__ == '__main__':
    main()