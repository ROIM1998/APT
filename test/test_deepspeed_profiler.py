import torch
import os
import sys
os.environ["WANDB_DISABLED"] = "true"
from deepspeed.profiling.flops_profiler import get_model_profile, get_module_duration
from transformers import HfArgumentParser
from args import DataTrainingArguments, MinusTrainingArguments
from models import build_model
from models.model_args import ModelArguments
from utils.utils import *
from trainer.model_arch import get_layers
from utils.cofi_utils import update_params, prune_model_with_z
from utils.minus_utils import input_constructor

def main():
    sys.argv = ['neuron_importance.py',
            '--output_dir',
            './output/neuron_importance/',
            '--model_name_or_path',
            'roberta-base',
            '--task_name',
            'mnli',
            '--do_train',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '128',
            '--per_device_eval_batch_size',
            '128',
            '--apply_lora',
            '--do_distill',
            '--lora_r',
            '8'
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
    if model.head_mask is not None:
        mask_prefix = 'final_' if os.path.exists(os.path.join(model_args.model_name_or_path, 'final_head_mask.pt')) else ''
        zs = {
            'head_z': torch.load(os.path.join(model_args.model_name_or_path, mask_prefix + 'head_mask.pt'), map_location='cpu'),
            'intermediate_z': torch.load(os.path.join(model_args.model_name_or_path, mask_prefix + 'intermediate_mask.pt'), map_location='cpu'),
        }
        update_params(model, zs)
        prune_model_with_z(zs, model)
        model.head_mask, model.intermediate_mask = None, None
    model.eval()
    for i in range(model.config.num_hidden_layers):
        module = get_layers(model)[i].intermediate.dense
        module.eval()
        module.weight.data += (module.lora_B @ module.lora_A)* module.scaling
        module.merged=True

    with torch.cuda.device(0):
        model=model.cuda()
        batch_size = training_args.per_device_eval_batch_size
        seq_len = 128
        enable_profile = True
        if enable_profile:
            flops, macs, params = get_model_profile(
                model,
                kwargs={k: v.to(model.device) for k, v in input_constructor(batch_size, seq_len, tokenizer).items()},
                print_profile=True,
                detailed=True,
                output_file='roberta-base-profile.txt'
            )
        else:
            inputs = input_constructor((batch_size, seq_len), tokenizer)
            outputs = model(inputs)