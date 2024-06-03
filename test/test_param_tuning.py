import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
from transformers import HfArgumentParser
from args import DataTrainingArguments
from models import build_model
from models.model_args import ModelArguments
from utils.utils import *
from args import MinusTrainingArguments
from utils import build_trainer

def main():
    sys.argv = ['neuron_importance.py',
            '--output_dir',
            './output/neuron_importance/',
            '--model_name_or_path',
            'output/debug_output',
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
    t_name, raw_datasets = get_raw_datasets(model_args, data_args, training_args)
        
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=None)
    
    model_args.model_name_or_path = os.path.join(model_args.model_name_or_path, 'best_model')
    # training_args.disable_tqdm = False
    config, tokenizer, best_model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    best_trainer = build_trainer(data_args, training_args, best_model, tokenizer, train_dataset, eval_dataset, param_controller=None)
    
    final_model_params = dict(model.named_parameters())
    best_model_params = dict(best_model.named_parameters())
    tuned_params = [
        k for k in final_model_params if not torch.allclose(final_model_params[k], best_model_params[k])
    ]
    sum_tuned_params = sum([torch.numel(final_model_params[k]) for k in tuned_params])
    sum_changed_params = sum([(final_model_params[k] != best_model_params[k]).sum() for k in tuned_params])
    
if __name__ == '__main__':
    main()