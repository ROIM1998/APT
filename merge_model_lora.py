import seaborn as sns
sns.set_theme(style="darkgrid")
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
import loralib as lora

from transformers import (HfArgumentParser)
from args import DataTrainingArguments
from models.model_args import ModelArguments
from utils.utils import *
from utils import build_trainer
from utils.minus_utils import lora_to_linear
from args import MinusTrainingArguments
from models import build_model
from torch.utils.data import Subset
from utils.fisher_utils.efficiency.param import *

def main():
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
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets, force_model_shape_deduction=True)
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)
    IS_SQUAD = 'squad' in data_args.task_name.lower()
    model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt')).to(training_args.device)
    model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt')).to(training_args.device)
    model.hidden_mask = torch.load(os.path.join(model_args.model_name_or_path, 'hidden_mask.pt')).to(training_args.device)

    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=None)
    fixed_scaling = True
    if fixed_scaling:
        for m in model.modules():
            if isinstance(m, lora.Linear):
                m.scaling = model_args.lora_alpha / model_args.lora_r
    model_param_num = sum(p.numel() for p in model.parameters())
    print("Unmerged model's performance: ", trainer.evaluate())
    for n, m in dict(model.named_modules()).items():
        for child_name, child in dict(m.named_children()).items():
            if isinstance(child, lora.Linear):
                print("Merging layer {}".format(n + '.' + child_name))
                delattr(m, child_name)
                merged_layer = lora_to_linear(child)
                setattr(m, child_name, merged_layer)
    
    model_param_num_merged = sum(p.numel() for p in model.parameters())
    print("Merged model's performance: ", trainer.evaluate())
    print("Parmeter number reduced from {} to {}, with {} parameters removed".format(model_param_num, model_param_num_merged, model_param_num - model_param_num_merged))
    
    trainer.save_model()

if __name__ == '__main__':
    main()