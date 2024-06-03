import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
from itertools import combinations
from transformers import (HfArgumentParser, EvalPrediction, default_data_collator, DataCollatorWithPadding)
from args import DataTrainingArguments
from models.model_args import ModelArguments
from trainer.trainer_minus import MinusTrainer
from utils.utils import *
from args import MinusTrainingArguments
from models import build_model
from datasets import load_metric
from prune.scorer import PredictivityScorer, GradientScorer
from prune.pruner import GreedyPruner, FisherPruner
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import seaborn as sns
sns.set_theme(style='darkgrid')
from matplotlib import pyplot as plt
from utils.fisher_utils.efficiency.mac import compute_mac, mac_per_head, mac_per_neuron

@torch.no_grad()
def log_head_neuron_performance(
    trainer,
    config,
    head_importance,
    neuron_importance,
    seq_len,
    mac_constraint,
):
    assert mac_constraint < 1

    num_hidden_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    hidden_size = config.hidden_size
    attention_head_size = int(hidden_size / num_attention_heads)

    original_mac = compute_mac(
        [num_attention_heads] * num_hidden_layers,
        [intermediate_size] * num_hidden_layers,
        seq_len,
        hidden_size,
        attention_head_size,
    )
    max_mac = mac_constraint * original_mac

    # Globally rank heads and neurons
    sorted_head_importance, sorted_head_indicies = head_importance.view(-1).sort(descending=True)
    sorted_neuron_importance, sorted_neuron_indicies = neuron_importance.view(-1).sort(descending=True)

    max_importance = 0
    results = []
    for num_heads in range(1, num_hidden_layers * num_attention_heads + 1):
        heads_mac = mac_per_head(seq_len, hidden_size, attention_head_size) * num_heads
        neurons_mac = max_mac - heads_mac
        num_neurons = int(neurons_mac / mac_per_neuron(seq_len, hidden_size))
        num_neurons = max(num_neurons, 0)

        total_importance = sorted_head_importance[:num_heads].sum() + sorted_neuron_importance[:num_neurons].sum()
        head_indicies = sorted_head_indicies[:num_heads]
        neuron_indicies = sorted_neuron_indicies[:num_neurons]

        head_mask = torch.zeros(num_hidden_layers * num_attention_heads).cuda()
        head_mask[head_indicies] = 1.0
        head_mask = head_mask.view(num_hidden_layers, num_attention_heads)
        neuron_mask = torch.zeros(num_hidden_layers * intermediate_size).cuda()
        neuron_mask[neuron_indicies] = 1.0
        neuron_mask = neuron_mask.view(num_hidden_layers, intermediate_size)
        trainer.model.head_mask = head_mask
        trainer.model.intermediate_mask = neuron_mask
        metrics = trainer.evaluate()
        metrics['num_attention_heads'] = num_heads
        metrics['num_neurons'] = num_neurons
        metrics['total_importance'] = total_importance
        results.append(metrics)

    return results


# TODO: regorganize the functions for neuron importance calculation
def main():
    sys.argv = ['neuron_importance.py',
            '--output_dir',
            './output/neuron_importance/',
            '--model_name_or_path',
            'output/roberta-base_lora_minus_mnli_once_fisher_distill_full_lorawithbias/mac0.6/lora_r128/lora_alpha16/pre_distillation_model',
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
            '128',
            '--do_distill'
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
        
    dataloader = DataLoader(
        Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:training_args.per_device_eval_batch_size * 64]),
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    model = model.to(training_args.device)
    model.head_mask = (model.head_mask != 0).float().to(model.device)
    model.intermediate_mask = (model.intermediate_mask != 0).float().to(model.device)
    # model.head_mask = torch.ones([model.config.num_hidden_layers, model.config.num_attention_heads], device=model.device)
    # model.intermediate_mask = torch.ones([model.config.num_hidden_layers, model.config.intermediate_size], device=model.device)
    scorer = GradientScorer(model, dataloader)
    greedy_pruner = GreedyPruner(model, ['intermediate_mask', 'head_mask'], {'intermediate_mask': scorer, 'head_mask': scorer})
    fisher_pruner = FisherPruner(model, ['intermediate_mask', 'head_mask'], {'intermediate_mask': scorer, 'head_mask': scorer}, 128, True)
    greedy_masks = greedy_pruner.generate_mask()
    fisher_masks = fisher_pruner.generate_mask()
    trainer = MinusTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    results = log_head_neuron_performance(
        trainer,
        model.config,
        scorer.head_score(),
        scorer.intermediate_score(),
        128,
        0.6,
    )
    _, indices = scores.view(-1).sort(descending=True)
    indices = [(i.item() // model.config.intermediate_size, i.item() % model.config.intermediate_size) for i in indices]
    trainer.evaluate()
    unmasked_top = [
        v for v in indices if v[0] > 3 and model.intermediate_mask[v] == 1
    ]
    results = {
        0: trainer.evaluate()
    }
    for i in range(1, len(unmasked_top) // 100):
        for v in unmasked_top[- i * 100:]:
            model.intermediate_mask[v] = 0
        results[i] = trainer.evaluate()
    
    best_by_mutual = list(set([d.flatten().sort(descending=True)[1][0].item() for d in divergence]))
    for i, index in enumerate(best_by_mutual):
        plt.clf()
        layer, neuron_i = index // model.config.intermediate_size, index % model.config.intermediate_size
        for l in range(len(model.config.label2id)):
            sns.kdeplot(activations[:, layer, neuron_i].index_select(0, (labels == l).nonzero().squeeze()), label=model.config.id2label[l])
        plt.savefig(os.path.join(training_args.output_dir, "skilled-neuron-activation-%d.png" % index))
    
    for i, j in combinations(best_by_mutual, 2):
        plt.clf()
        layer_i, neuron_i = i // model.config.intermediate_size, i % model.config.intermediate_size
        layer_j, neuron_j = j // model.config.intermediate_size, j % model.config.intermediate_size
        sns.scatterplot(x=activations[:, layer_i, neuron_i], y=activations[:, layer_j, neuron_j], hue=labels)
        plt.savefig(os.path.join(training_args.output_dir, "skilled-neuron-comparison-%d-%d.png" % (i, j)))
    
    torch.save(scores, os.path.join(training_args.output_dir, "skilled-neuron-scores.pt"))
    torch.save(divergence, os.path.join(training_args.output_dir, "skilled-neuron-divergence.pt"))
    intermediate_mask = fisher_pruner.generate_mask()['intermediate_mask']
    torch.save(intermediate_mask, os.path.join(training_args.output_dir, "intermediate-mask.pt"))
    
    i = 31698
    j = 26023
    with torch.no_grad():
        for folder in tqdm([v for v in os.listdir('output/roberta_lora_mnli_new') if 'checkpoint' in v]):
            iteration = folder.split('-')[-1]
            model_args.model_name_or_path = os.path.join('output/roberta_lora_mnli_new', folder)
            config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
            model = model.to(training_args.device)
            model.head_mask = None
            model.intermediate_mask = None
            scorer = PredictivityScorer(model, dataloader)
            scores, divergence, activations, labels = scorer.get_activation_divergence()
            plt.clf()
            layer_i, neuron_i = i // model.config.intermediate_size, i % model.config.intermediate_size
            layer_j, neuron_j = j // model.config.intermediate_size, j % model.config.intermediate_size
            sns.scatterplot(x=activations[:, layer_i, neuron_i], y=activations[:, layer_j, neuron_j], hue=labels)
            plt.savefig(os.path.join(training_args.output_dir, "skilled-neuron-comparison-%d-%d-iter%s.png" % (i, j, iteration)))

if __name__ == '__main__':
    main()