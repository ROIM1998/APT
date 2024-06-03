import os
import sys
import torch

from transformers import (HfArgumentParser)
from torch.utils.data import Subset
from args import DataTrainingArguments
from models.model_args import ModelArguments
from args import MinusTrainingArguments
from utils.utils import *
from utils import build_trainer, build_dataloader
from models import build_model
from trainer.param_control import ParamController
from prune import build_scorer, build_pruner

MB = 1024 * 1024

def main():
    sys.argv = ['test_bert.py',
            '--output_dir',
            './output/test_magnitude_scorer/',
            '--model_name_or_path',
            'bert-base-uncased',
            '--task_name',
            'mnli',
            '--do_train',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '32',
            '--per_device_eval_batch_size',
            '128',
            '--apply_lora',
            '--do_distill',
            '--lora_r',
            '8',
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
    model.head_mask = model.head_mask.to(training_args.device).view(-1)
    model.intermediate_mask = model.intermediate_mask.to(training_args.device).view(-1)
    model.hidden_mask = model.hidden_mask.to(training_args.device).view(-1)
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)
    model = model.to(training_args.device)
    pruning_batch_size = 32
    num_pruning_batches = 64
    dataloader = build_dataloader(Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]), pruning_batch_size, data_args, training_args, tokenizer)
    dataloader = build_dataloader(data_args, training_args, model, tokenizer, raw_datasets)

    # Also add ffn input layers to teacher config
    teacher_keys = ['query', 'value', 'intermediate']
    teacher_config  = {
        k: [i for i in range(model.config.num_hidden_layers)] for k in teacher_keys
    }
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        student_config=None,
        lora_with_bias=False,
        param_allocation_strategy=training_args.param_allocation_strategy,
    )
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)

    scorer = build_scorer('magnitude', model, param_controller=param_controller, state=None)
    head_score = scorer.get_score('head_mask')
    intermediate_score = scorer.get_score('intermediate_mask')
    hidden_score = scorer.get_score('hidden_mask')
    
    # Testing head score equality
    for layer in range(12):
        get_score = head_score[12 * layer: 12 * (layer + 1)]
        attention_layer = model.bert.encoder.layer[layer].attention.self
        attention_output_layer = model.bert.encoder.layer[layer].attention.output
        qkv_score = attention_layer.query.weight.abs().sum(dim=1) + attention_layer.key.weight.abs().sum(dim=1) + attention_layer.value.weight.abs().sum(dim=1)
        qkv_score += attention_layer.query.bias.abs() + attention_layer.key.bias.abs() + attention_layer.value.bias.abs()
        qkv_score += attention_output_layer.dense.weight.abs().sum(dim=0)
        score = qkv_score.view(12, -1).sum(dim=1)
        print("Equality:", torch.allclose(get_score, score, atol=1e-6))
        
    for layer in range(12):
        get_score = intermediate_score[3072 * layer: 3072 * (layer + 1)]
        ffn_layer = model.bert.encoder.layer[layer].intermediate
        ffn_output_layer = model.bert.encoder.layer[layer].output
        score = ffn_layer.dense.weight.abs().sum(dim=1)
        score += ffn_layer.dense.bias.abs()
        score += ffn_output_layer.dense.weight.abs().sum(dim=0)
        print("Equality:", torch.allclose(get_score, score, atol=1e-6))
    
    calculated_hidden_score = 0
    for layer in range(12):
        attention_layer = model.bert.encoder.layer[layer].attention.self
        attention_output_layer = model.bert.encoder.layer[layer].attention.output
        qkv_score = attention_layer.query.weight.abs().sum(dim=0) + attention_layer.key.weight.abs().sum(dim=0) + attention_layer.value.weight.abs().sum(dim=0)
        qkv_score += attention_output_layer.dense.weight.abs().sum(dim=1)
        qkv_score +=attention_output_layer.dense.bias.abs()
        
        ffn_layer = model.bert.encoder.layer[layer].intermediate
        ffn_output_layer = model.bert.encoder.layer[layer].output
        score = ffn_layer.dense.weight.abs().sum(dim=0)
        score += ffn_output_layer.dense.weight.abs().sum(dim=1)
        score += ffn_output_layer.dense.bias.abs()
        calculated_hidden_score += qkv_score + score
    print("Equality:", torch.allclose(hidden_score, calculated_hidden_score, atol=1e-6))
    
    # Test pruning with magnitude scorer and binary-search density pruner
    training_args.pruner_type = 'running'
    pruner = build_pruner(training_args.pruner_type, training_args, model, scorer)
    pruner.update_mask(0.8, is_last=True)
    
    # Now testing wanda scorers
    scorer = build_scorer('wanda', model, dataloader=dataloader, param_controller=param_controller, state=trainer.state, gather_freq=1, beta_1=0.85, beta_2=0.85, use_uncertainty=False, block_normalize_dict=None)
    print("Pre-step max memory allocated:", torch.cuda.max_memory_allocated() / MB)
    with torch.no_grad():
        scorer.step()
    print("Post-step max memory allocated:", torch.cuda.max_memory_allocated() / MB)
        
    head_wanda_score = scorer.get_score('head_mask')
    intermediate_wanda_score = scorer.get_score('intermediate_mask')
    hidden_wanda_score = scorer.get_score('hidden_mask')
    training_args.pruner_type = 'running'
    pruner = build_pruner(training_args.pruner_type, training_args, model, scorer)
    pruner.update_mask(0.6, is_last=True)

if __name__ == '__main__':
    main()