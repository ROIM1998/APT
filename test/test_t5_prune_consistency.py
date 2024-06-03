import os
os.environ["WANDB_DISABLED"] = "true"
import sys

from transformers import HfArgumentParser
from args import Seq2SeqDataTrainingArguments
from models import build_model
from models.model_args import ModelArguments
from utils.utils import *
from args import MinusTrainingArguments
from utils.minus_utils import efficiency_testing, input_constructor

def main():
    sys.argv = ['test_t5.py',
            '--output_dir',
            './output/test_t5_grafting/',
            '--model_name_or_path',
            'output/t5-large_lora_minus_xsum_once_global_free_inout_nodistill/mac0.05/epoch3/bz4/numprune3/parameq:0-23,ev:0-23,dq:0-23,dv:0-23,cq:0-23,cv:0-23,ei:0-23,di:0-23/lora_r8/prunestart0.01/pre_pruning_model',
            '--task_name',
            'xsum',
            '--do_train',
            '--do_eval',
            '--max_input_length',
            '936',
            '--max_target_length',
            '38',
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
            'global',
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
        (ModelArguments, Seq2SeqDataTrainingArguments, MinusTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    config, tokenizer, model = build_model(model_args, data_args, training_args)

    efficiency_results = efficiency_testing(model, tokenizer, training_args.device)