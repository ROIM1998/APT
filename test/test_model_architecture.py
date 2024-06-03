import os
import torch
os.environ["WANDB_DISABLED"] = "true"
import sys
from transformers import (HfArgumentParser, EvalPrediction, default_data_collator, DataCollatorWithPadding, AutoConfig, AutoTokenizer)
from args import DataTrainingArguments
from models import build_model
from models.model_args import ModelArguments
from trainer.trainer_minus import MinusTrainer
from utils.utils import *
from datasets import load_metric
from args import MinusTrainingArguments
from torch.utils.data import DataLoader
from utils.cofi_utils import prune_model_with_z
from models.modeling_roberta import CoFiRobertaForSequenceClassification
from models.modeling_roberta_old import CoFiRobertaForSequenceClassification as CoFiRobertaForSequenceClassificationOld

def main():
    sys.argv = ['neuron_importance.py',
            '--output_dir',
            './output/neuron_importance/',
            '--model_name_or_path',
            'output/roberta-base_lora_minus_mnli_once_global_oldselfdistill/mac0.4/epoch10/bz32/numprune5/lora_r64/lora_alpha16/pruning_1_model',
            '--task_name',
            'mnli',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '32',
            '--per_device_eval_batch_size',
            '32',
            '--lora_r',
            '64',
            '--apply_lora',
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
    t_name, raw_datasets = get_raw_datasets(model_args, data_args, training_args)

    model_path = model_args.model_name_or_path
    label_list, num_labels, _ = get_label(data_args, raw_datasets)

    # Build up config and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_path,
        num_labels=num_labels,
        finetuning_task=t_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.apply_lora=model_args.apply_lora
    config.lora_alpha=model_args.lora_alpha
    config.lora_r=model_args.lora_r
    config.do_distill = training_args.do_distill
    config.ft_from_lora = training_args.ft_from_lora
    if config.apply_lora:
        print("LoRA r using is %d" % config.lora_r)
    else:
        print("No LoRA is using. Fine-tuning the model!")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        local_files_only=True,
    )
    model = CoFiRobertaForSequenceClassification.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    old_model = CoFiRobertaForSequenceClassificationOld.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)    
    model = model.to(training_args.device)
    old_model = old_model.to(training_args.device)
    model.head_mask, model.intermediate_mask = torch.load(os.path.join(model_path, 'head_mask.pt')), torch.load(os.path.join(model_path, 'intermediate_mask.pt'))
    old_model.head_mask, old_model.intermediate_mask = torch.load(os.path.join(model_path, 'head_mask.pt')), torch.load(os.path.join(model_path, 'intermediate_mask.pt'))

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    dataloader = DataLoader(
        eval_dataset,
        batch_size=32,
        collate_fn=data_collator,
    )
    inputs = next(iter(dataloader))
    inputs = {k: v.to(training_args.device) for k, v in inputs.items()}
    
    model.eval()
    for p in model.parameters():
        _ = p.requires_grad_(False)
    old_model.eval()
    for p in old_model.parameters():
        _ = p.requires_grad_(False)

    new_teacher_outputs = model(**inputs, output_hidden_states=True, return_dict=False, pass_mask=False)
    old_teacher_outputs = old_model(**inputs, output_hidden_states=True, return_dict=False, pass_mask=False)
    new_student_outputs = model(**inputs, output_hidden_states=True, return_dict=False, pass_mask=True)
    old_student_outputs = old_model(**inputs, output_hidden_states=True, return_dict=False, pass_mask=True)