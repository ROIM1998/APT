import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import transformers
import torch
import nltk
import numpy as np
transformers.logging.set_verbosity_error()

from transformers import (HfArgumentParser, DataCollatorForSeq2Seq)
from models.model_args import ModelArguments
from utils.utils import *
from trainer.trainer_minus import MinusTrainer
from args import MinusTrainingArguments, Seq2SeqDataTrainingArguments
from models import build_model
from trainer.param_control import ParamController

MB = 1024 * 1024

def main():
    sys.argv = ['test_t5.py',
            '--output_dir',
            './output/test_t5_grafting/',
            '--model_name_or_path',
            't5-3b',
            '--task_name',
            'xsum',
            '--do_train',
            '--do_eval',
            '--max_input_length',
            '512',
            '--max_target_length',
            '128',
            '--per_device_train_batch_size',
            '4',
            '--per_device_eval_batch_size',
            '4',
            '--learning_rate',
            '1e-3',
            '--eval_accumulation_steps',
            '1',
            '--lora_r',
            '8',
            '--lora_alpha',
            '16',
            '--apply_lora',
            '--pruner_type',
            'search',
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
            '--report_to',
            'none',
            '--pruner_type',
            'running_fisher',
            ]
    parser = HfArgumentParser(
        (ModelArguments, Seq2SeqDataTrainingArguments, MinusTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    config, tokenizer, model = build_model(model_args, data_args, training_args)
    train_dataset, eval_dataset, _, datasets = build_seq2seq_data(data_args, training_args, tokenizer)
    model.to(training_args.device)

    teacher_keys = ['enc_self_query', 'enc_self_value', 'dec_self_query', 'dec_self_value', 'cross_query', 'cross_value', 'encoder_i', 'decoder_i']
    teacher_config = {
        k: [i for i in range(config.num_layers)] for k in teacher_keys
    }
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        lora_with_bias=False,
    )
    model.head_mask = model.head_mask.to(training_args.device)
    model.intermediate_mask = model.intermediate_mask.to(training_args.device)
    model.hidden_mask = model.hidden_mask.to(training_args.device)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args.disable_tqdm = False
    trainer = MinusTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        param_controller=param_controller,
        teacher_model=None,
    )
    param_controller.convert_to_pre_pruning_lora_teacher()
    dataloader = trainer.get_eval_dataloader(eval_dataset)
    inputs = next(iter(dataloader))
    inputs = trainer._prepare_inputs(inputs)
    
    # Test model pruning consistency
    model.eval()
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=False,
        ) # loss, logits, decoder.past_key_values, decoder.hidden_states, encoder.last_hidden_states, encoder.hidden_states
        unmasked_outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=False,
            use_teacher=True,
            pass_mask=False,
        )
    
    model.prune_model_with_masks()
    
    with torch.no_grad():
        pruned_outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=False,
        )
        generated_outputs = model.generate(**inputs)
        decoded_input = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
        decoded_output = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
        
    # Test model memory consumption with and without parameters, masks gradients
    for p in model.parameters():
        p.requires_grad = False
    model.head_mask.requires_grad = True
    model.intermediate_mask.requires_grad = False
    model.hidden_mask.requires_grad = False
    
    print(torch.cuda.memory_allocated() / MB)
    with torch.no_grad():
        outputs = model(
            **inputs,
        )
    print(torch.cuda.max_memory_allocated() / MB)
    del outputs
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    outputs = model(**inputs)
    outputs[0].backward()
    model.zero_grad()
    print(torch.cuda.max_memory_allocated() / MB)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model.head_mask.requires_grad = False
    model.intermediate_mask.requires_grad = True
    outputs = model(**inputs)
    outputs[0].backward()
    model.zero_grad()
    print(torch.cuda.max_memory_allocated() / MB)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model.intermediate_mask.requires_grad = False
    model.hidden_mask.requires_grad = True
    outputs = model(**inputs)
    outputs[0].backward()
    model.zero_grad()
    print(torch.cuda.max_memory_allocated() / MB)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model.clear_masks()
    for n, p in model.named_parameters():
        if 'lora' in n:
            p.requires_grad = True
    outputs = model(**inputs)
    outputs[0].backward()
    model.zero_grad()
    print(torch.cuda.max_memory_allocated() / MB)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()