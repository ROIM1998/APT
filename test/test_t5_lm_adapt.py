import torch
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import time

from transformers import HfArgumentParser
from args import DataTrainingArguments
from models import build_model
from models.model_args import ModelArguments
from trainer.trainer_minus import ParamController
from utils import build_trainer
from utils.utils import *
from args import MinusTrainingArguments

def main():
    sys.argv = ['test_t5_lm_adapt.py',
            '--output_dir',
            './output/test_t5_lm_adapt/',
            '--model_name_or_path',
            'output/google/t5-base-lm-adapt/sst2/bz32/lora/epoch60/lora_r8/lora_alpha16/lr1e-3/seed42/best_model',
            '--task_name',
            'sst2',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '32',
            '--per_device_eval_batch_size',
            '32',
            '--eval_accumulation_steps',
            '1',
            '--lora_r',
            '8',
            '--lora_alpha',
            '16',
            '--apply_lora',
            '--pruner_type',
            'none',
            '--head_scorer_type',
            'gradient_l2',
            '--intermediate_scorer_type',
            'gradient_l2',
            '--pruning_batch_size',
            '4',
            '--pruning_batches',
            '64',
            '--pruning_scheduler',
            'once',
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
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    MODEL_GENERATIVE = True
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets, MODEL_GENERATIVE)

    model.to(training_args.device)

    teacher_keys = ['enc_self_query', 'enc_self_value', 'dec_self_query', 'dec_self_value', 'cross_query', 'cross_value']
    teacher_config = {
        k: [i for i in range(config.num_layers)] for k in teacher_keys
    }
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        lora_with_bias=False,
    )
    training_args.seq_len = 128
    training_args.output_seq_len = 2
    training_args.cls_task = True
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    model.head_mask, model.intermediate_mask = None, None
    model.hidden_mask = None
    trainer.args.predict_with_generate = True
    trainer.evaluate()
    
    eval_dataloader = trainer.get_eval_dataloader(eval_dataset)
    inputs = next(iter(eval_dataloader))
    inputs = trainer._prepare_inputs(inputs)
    outputs = model(**inputs, output_hidden_states=True)
    
    with torch.no_grad():
        total, correct = 0, 0
        for step, inputs in enumerate(eval_dataloader):
            inputs = trainer._prepare_inputs(inputs)
            labels = inputs.pop('labels')
            inputs['decoder_input_ids'] = model._shift_right(labels)
            inputs['decoder_input_ids'][:] = 0
            outputs = model(**inputs)
            total += len(labels)
            correct += (outputs[0].argmax(dim=-1)[:, 0] == labels[:, 0]).sum().item()
            
        # Test with using .generate()
        total, correct = 0, 0
        for step, inputs in enumerate(eval_dataloader):
            inputs = trainer._prepare_inputs(inputs)
            labels = inputs.pop('labels')
            outputs = model.generate(**inputs, output_hidden_states=True, num_beams=1, do_sample=False)
            total += len(labels)
            correct += (outputs[:, 1] == labels[:, 0]).sum().item()
        
        # Test with original model calling
        total, correct = 0, 0
        for step, inputs in enumerate(eval_dataloader):
            inputs = trainer._prepare_inputs(inputs)
            labels = inputs['labels']
            outputs = model(**inputs)
            total += len(labels)
            correct += (outputs[1].argmax(dim=-1)[:, 0] == labels[:, 0]).sum().item()
            
        
        # Compare the results of the two methods
        inputs = next(iter(eval_dataloader))
        inputs = trainer._prepare_inputs(inputs)
        outputs = model(**inputs, output_hidden_states=True)
        labels = inputs['labels']
        generate_outputs = model.generate(**inputs, output_hidden_states=True)