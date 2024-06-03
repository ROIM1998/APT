import sys
import os
import json
import torch

from tqdm import tqdm
from datasets import load_metric
from glue import avg_seq_length
from transformers import HfArgumentParser, TrainingArguments, DataCollatorWithPadding
from torch.utils.data import DataLoader, RandomSampler
from args import AdditionalArguments, DataTrainingArguments
from models.model_args import ModelArguments
from utils.utils import *
from models import build_model
from prune.pruner import RandomPruner

from utils.fisher_utils.efficiency.mac import compute_mask_mac
from prune.fisher import collect_mask_grads
from prune.search import search_mac
from prune.rearrange import rearrange_mask
from prune.rescale import rescale_mask
from utils.fisher_utils.schedule import get_pruning_schedule

def test(model, eval_dataloader, head_mask, intermediate_mask, metric, data_args):
    for i, inputs in enumerate(eval_dataloader):
        labels = inputs['labels']
        inputs['head_z'] = head_mask
        inputs['intermediate_z'] = intermediate_mask
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(
            **inputs,
            )
        if data_args.task_name == 'stsb':
            predictions = outputs.logits.squeeze()
        else:
            predictions = outputs.logits.argmax(dim=-1)
        metric.add_batch(
            predictions=predictions,
            references=labels,
        )
    eval_results = metric.compute()
    return eval_results


if __name__ == '__main__':
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, AdditionalArguments)
    )
    # sys.argv = ['run_pruning.py',
    #         '--output_dir',
    #         './output',
    #         '--model_name_or_path',
    #         # 'output/roberta_lora_mnli/checkpoint-36500',
    #         'output/roberta_mnli/checkpoint-36500',
    #         '--task_name',
    #         'mnli',
    #         '--do_train',
    #         '--do_eval',
    #         '--max_seq_length',
    #         '128',
    #         '--per_device_train_batch_size',
    #         '32',
    #         '--per_device_eval_batch_size',
    #         '32',
    #         # '--apply_lora'
    #         '--prune_mode',
    #         'fisher',
    #         ]
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, additional_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, additional_args = parser.parse_args_into_dataclasses()
        
    t_name, raw_datasets = get_raw_datasets(model_args, data_args, training_args)
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    train_dataset, eval_dataset, predict_dataset, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)
    collate_fn = DataCollatorWithPadding(tokenizer)
    test_mode, mask_mode, prune_mode = additional_args.test_mode, additional_args.mask_mode, additional_args.prune_mode

    if prune_mode == 'fisher':
        assert training_args.do_train
        sample_dataloader = DataLoader(
            train_dataset,
            batch_size = training_args.per_device_train_batch_size,
            sampler=RandomSampler(train_dataset, replacement=True, num_samples=int(additional_args.sample_rate * len(train_dataset))),
            collate_fn=collate_fn,
        )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        sampler=RandomSampler(eval_dataset, replacement=True, num_samples=training_args.per_device_eval_batch_size*50), # prevent from killing because of memory out of usage
        collate_fn=collate_fn,
    )
    model.to(training_args.device)
    model.eval()
    for p in model.parameters():
        _ = p.requires_grad_(False)

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("./glue_metric.py", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    
    if prune_mode == 'random':
        ratio_range = torch.arange(0, additional_args.ratio_bound, additional_args.ratio_step).tolist()
    elif prune_mode == 'fisher':
        ratio_range = torch.arange(additional_args.ratio_bound, 1, additional_args.ratio_step).tolist()
    if test_mode == 'correlation':
        inner_rep = 1
    else:
        inner_rep = 20
    accuracy_by_ratios = {}
    with tqdm(total=len(ratio_range) * inner_rep) as pbar:
        for ratio in ratio_range:
            accuracy_ratio = []
            for _ in range(inner_rep):
                pbar.update(1)
                # If using random pruner to prune the model
                if prune_mode == 'random':
                    mask_required = ['head_mask', 'intermediate_mask'] if mask_mode == 'all' else ['head_mask'] if mask_mode == 'head' else ['intermediate_mask'] if mask_mode == 'intermediate' else []
                    pruner = RandomPruner(model, mask_required)
                    masks = pruner.random_mask(mask_ratio={
                        s: ratio for s in mask_required
                    })
                    eval_results = test(model, eval_dataloader, masks['head_mask'] if 'head_mask' in mask_required else torch.ones(pruner.mask_to_shape['head_mask']), masks['intermediate_mask'] if 'intermediate_mask' in mask_required else torch.ones(pruner.mask_to_shape['intermediate_mask']), metric, data_args)
                    accuracy_ratio.append(eval_results['accuracy'])
                elif prune_mode == 'fisher':
                    IS_SQUAD = 'squad' in data_args.task_name
                    mask_required = ['head_mask', 'intermediate_mask']
                    seq_len = 170 if IS_SQUAD else avg_seq_length(data_args.task_name)
                    full_head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads).cuda()
                    full_neuron_mask = torch.ones(config.num_hidden_layers, config.intermediate_size).cuda()
                    head_grads, neuron_grads = collect_mask_grads(
                        model,
                        full_head_mask,
                        full_neuron_mask,
                        sample_dataloader,
                    )
                    teacher_constraint = get_pruning_schedule(target=ratio, num_iter=2)[0]
                    teacher_head_mask, teacher_neuron_mask = search_mac(
                        config,
                        head_grads,
                        neuron_grads,
                        seq_len,
                        teacher_constraint,
                    )
                    head_mask, neuron_mask = search_mac(
                        config,
                        head_grads,
                        neuron_grads,
                        seq_len,
                        ratio,
                    )
                    pruned_mac, orig_mac = compute_mask_mac(head_mask, neuron_mask, seq_len, config.hidden_size)
                    print(f"Pruned Model MAC: {pruned_mac / orig_mac * 100.0:.2f} %")
                    searched_eval_metrics = test(model, eval_dataloader, head_mask, neuron_mask, metric, data_args)
                    # Rearrange the mask
                    head_mask = rearrange_mask(head_mask, head_grads)
                    neuron_mask = rearrange_mask(neuron_mask, neuron_grads)
                    rearranged_eval_metrics = test(model, eval_dataloader, head_mask, neuron_mask, metric, data_args)
                    # Rescale the mask by solving a least squares problem
                    head_mask, neuron_mask = rescale_mask(
                        model,
                        config,
                        teacher_head_mask,
                        teacher_neuron_mask,
                        head_mask,
                        neuron_mask,
                        sample_dataloader,
                        classification_task=not IS_SQUAD,
                    )
                    rescaled_eval_metrics = test(model, eval_dataloader, head_mask, neuron_mask, metric, data_args)
                    accuracy_ratio.append({
                        "searched": searched_eval_metrics['accuracy'],
                        "rearranged": rearranged_eval_metrics['accuracy'],
                        "rescaled": rescaled_eval_metrics['accuracy'],
                        "head_mask_ratio": (head_mask != 0).sum() / 144,
                        "intermediate_mask_ratio": (neuron_mask != 0).sum() / 36864,
                        "overall_ratio": (head_mask != 0).sum() + (neuron_mask != 0).sum() / (144 + 36864),
                    })
                    # print("Accuracy after mask rescaling is:", eval_metrics['accuracy'])
            accuracy_by_ratios[ratio] = accuracy_ratio
    print('Test after-pruning eval_results with %f pruning:' % (ratio / 10), eval_results['accuracy'])
    # post-pruning eval_results: {'accuracy': 0.299375}
    # pre-pruning eval_results: {'accuracy': 0.84625}
    # cofi eval_results: {'accuracy': 0.8052}
        
    json.dump(accuracy_by_ratios, open(os.path.join(training_args.output_dir, '%s_mask_%s_%s_nolora.json' % (prune_mode, mask_mode, test_mode)), 'w'))