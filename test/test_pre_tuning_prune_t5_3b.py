import torch
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import transformers
transformers.logging.set_verbosity_error()

from transformers import (HfArgumentParser, DataCollatorForSeq2Seq, set_seed)
from models.model_args import ModelArguments
from utils.utils import *
from trainer.trainer_minus import MinusTrainer
from args import MinusTrainingArguments, Seq2SeqDataTrainingArguments
from models import build_model
from trainer.param_control import ParamController
from prune import RandomBSMaskPruner
from utils.minus_utils import count_params

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
            '32',
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
    set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)

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
    original_param_num, original_param_vars = count_params(model, mode='main')
    
    pre_tuning_pruner = RandomBSMaskPruner(model, training_args, None)
    pre_tuning_pruner.update_mask(0.8, 1, is_last=True)
    
    model = model.cpu()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated() / MB)
    model.prune_model_with_masks()
    model = model.to(training_args.device)
    pruned_model_param_num, pruned_model_param_vars = count_params(model, mode='main')
    print("Current model param num ratio: {:.4f}".format(pruned_model_param_num / original_param_num))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    trainer.param_controller.set_grafting_mask(False)
    
    trainer.train()