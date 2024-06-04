import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
from typing import Dict
from transformers import HfArgumentParser, default_data_collator, DataCollatorWithPadding, EvalPrediction
from args import DataTrainingArguments
from models import build_model
from models.model_args import ModelArguments
from utils.utils import *
from args import MinusTrainingArguments
from torch.utils.data import DataLoader
from trainer.param_control import ParamController
from trainer.trainer_minus import MinusTrainer
from datasets import load_metric
from prune import build_scorer, build_pruner

def distill(trainer: MinusTrainer, inputs: Dict[str, torch.Tensor], controller: ParamController):
    model = trainer.model
    model.train()
    controller.model_as_teacher()
    teacher_outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=False,
        pass_mask=False,
    )
    teacher_loss = teacher_outputs[0]
    
    controller.model_teacher_with_student()
    student_outputs = model(
        **inputs, output_hidden_states=True,
        return_dict=False
    )
    zs = {
        'intermediate_z': model.intermediate_mask,
        'head_z': model.head_mask,
    }
    distill_loss, distill_ce_loss, new_loss = trainer.calculate_distillation_loss(
        [v.detach() if isinstance(v, torch.Tensor) else [vv.detach() for vv in v] for v in teacher_outputs],
        student_outputs,
        zs,
    )
    loss = new_loss * 0.5 + teacher_loss * 0.5
    loss.backward()
    return loss

def main():
    sys.argv = ['neuron_importance.py',
            '--output_dir',
            './output/neuron_importance/',
            '--model_name_or_path',
            'output/roberta-base_lora_minus_mnli_once_fisher_distill_full_teacherlearning/mac0.6/lora_r128/lora_alpha16/pre_distillation_model',
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
            '--lora_r',
            '128',
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

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
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
    
    teacher_param_str = 'q:0-11,v:0-11'
    student_param_str = 'i:0-11'
    teacher_param_config, student_param_config = ParamController.parse_tuning_param_str(teacher_param_str), ParamController.parse_tuning_param_str(student_param_str)
    controller = ParamController(model, teacher_param_config, student_param_config)
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    inputs = next(iter(dataloader))
    trainer = MinusTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    inputs = trainer._prepare_inputs(inputs)
    model.head_mask, model.intermediate_mask = model.head_mask.to(trainer.args.device), model.intermediate_mask.to(trainer.args.device)
    trainer.args.head_scorer_type, trainer.args.intermediate_scorer_type = 'gradient_l2', 'gradient_l2'
    for p in model.parameters():
        p.requires_grad_(False)
    
    head_scorer = build_scorer(trainer.args.head_scorer_type, trainer.model, trainer.pruning_dataloader)
    intermediate_scorer = head_scorer
    scorer_dict = {
        'head_mask': head_scorer,
        'intermediate_mask': intermediate_scorer,
    }
    pruner = build_pruner(trainer.args, trainer.model, scorer_dict)
    masks = pruner.generate_mask(0.6)
    model.head_mask, model.intermediate_mask = masks['head_mask'], masks['intermediate_mask']
    distill(trainer, inputs, controller)

if __name__ == '__main__':
    main()