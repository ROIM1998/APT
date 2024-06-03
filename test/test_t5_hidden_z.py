import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import transformers
import torch
import nltk
import numpy as np
transformers.logging.set_verbosity_error()

from transformers import (HfArgumentParser, DataCollatorForSeq2Seq)
from datasets import load_metric
from models.model_args import ModelArguments
from utils.utils import *
from trainer.trainer_minus import MinusTrainer
from args import MinusTrainingArguments, Seq2SeqDataTrainingArguments
from loralib.layers import LoRALayer
from models import build_model
from loralib.layers import PruningLinear, DistillLinear
from trainer.param_control import ParamController    

def main():
    sys.argv = ['test_t5_running.py',
            '--output_dir',
            './output/test_t5_grafting/',
            '--model_name_or_path',
            'output/t5-base_lora_minus_xsum_once_global_none_co_learning_mapping_static_teacher_dynamic_cofi_student_distill/mac0.4/epoch5/bz16/numprune5/parameq:9-11,ev:9-11,dq:9-11,dv:9-11,cq:9-11,cv:9-11,ei:9-11,di:9-11/lora_r8/lora_alpha/pre_pruning_model',
            '--task_name',
            'xsum',
            '--do_train',
            '--do_eval',
            '--max_input_length',
            '512',
            '--max_target_length',
            '128',
            '--per_device_train_batch_size',
            '16',
            '--per_device_eval_batch_size',
            '16',
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
    model = model.to(training_args.device)
    print(model.config)

    teacher_keys = ['enc_self_query', 'enc_self_value', 'dec_self_query', 'dec_self_value', 'cross_query', 'cross_value']
    teacher_config = {
        k: [i for i in range(config.num_layers)] for k in teacher_keys
    }
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        lora_with_bias=False,
    )
    
    model.reset_masks()    
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
        seq_len=data_args.max_input_length,
        output_seq_len=data_args.max_target_length,
        cls_task=False,
    )
    training_args.predict_with_generate = True
    model = model.double() # Using double to reduce the numerical error
    
    model.hidden_mask[:20] = 0
    dataloader = trainer.get_train_dataloader()
    inputs = next(iter(dataloader))
    inputs = trainer._prepare_inputs(inputs)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    for i in range(13):
        print("Decoder", outputs[3][i][:, :, :20].any())
        print("Encoder", outputs[5][i][:, :, :20].any())
    
        
    model.prune_model_with_masks()
    
    model.eval()
    with torch.no_grad():
        pruned_outputs = model(**inputs, output_hidden_states=True)
        
    for i in range(13):
        print("Encoder %d" % i, (outputs[5][i][:, :, 20:] - pruned_outputs[5][i]).abs().mean())
        print("Decoder %d" % i, (outputs[3][i][:, :, 20:] - pruned_outputs[3][i]).abs().mean())