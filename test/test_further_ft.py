import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
from transformers import HfArgumentParser, EvalPrediction, default_data_collator, DataCollatorWithPadding
from args import DataTrainingArguments
from models import build_model
from models.model_args import ModelArguments
from datasets import load_metric
from trainer.trainer_minus import MinusTrainer
from utils.utils import *
from args import MinusTrainingArguments
from torch.utils.data import DataLoader, Subset

def main():
    sys.argv = ['test_further_ft.py',
            '--output_dir',
            'output/roberta-base_lora_minus_mnli_once_global_distill_full_exp_shorter/mac0.05/lora_r8/lora_alpha16/best_model/finetuned',
            '--model_name_or_path',
            'output/roberta-base_lora_minus_mnli_once_global_distill_full_exp_shorter/mac0.05/lora_r8/lora_alpha16/best_model',
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
            '--ft_from_lora',
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
    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    dataloader = DataLoader(
        Subset(eval_dataset, torch.randperm(len(eval_dataset)).tolist()[:training_args.per_device_eval_batch_size * 64]),
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
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

    trainer = MinusTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    result = trainer.evaluate()
    
if __name__ == '__main__':
    main()