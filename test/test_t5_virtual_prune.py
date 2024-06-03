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
from utils import build_dataloader, build_trainer
from utils.utils import *
from args import MinusTrainingArguments
from loralib.layers import LoRALayer, DistillLinear, PruningLinear
from tqdm import tqdm
from torch.utils.data import Subset
    
def main():
    sys.argv = ['test_t5.py',
            '--output_dir',
            './output/test_t5_3b_pretuning_prune/',
            '--model_name_or_path',
            'output/debug_t5_elastictuning/pre_virtual_pruning_model_step4480',
            '--task_name',
            'sst2',
            '--do_train',
            '--do_eval',
            '--do_distill',
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
            'running_fisher',
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

    teacher_keys = ['enc_self_query', 'enc_self_value', 'dec_self_query', 'dec_self_value', 'cross_query', 'cross_value']
    teacher_config = {
        k: [i for i in range(config.num_layers)] for k in teacher_keys
    }
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        lora_with_bias=False,
    )
    if os.path.exists(os.path.join(model_args.model_name_or_path, 'head_mask.pt')):
        model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt')).to(training_args.device)
        model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt')).to(training_args.device)
        model.hidden_mask = torch.load(os.path.join(model_args.model_name_or_path, 'hidden_mask.pt')).to(training_args.device)
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    
    model.head_mask = model.head_mask.to(training_args.device).view(-1)
    model.intermediate_mask = model.intermediate_mask.to(training_args.device).view(-1)
    model.hidden_mask = model.hidden_mask.to(training_args.device).view(-1)
    
    inputs = next(iter(trainer.get_train_dataloader()))
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    teacher_outputs = model(**inputs, return_dict=False, use_teacher=True, pass_mask=False)
    outputs = model(**inputs, return_dict=False)
    model.virtual_prune()
    virtual_pruned_teacher_outputs = model(**inputs, return_dict=False, use_teacher=True, pass_mask=False)
    virtual_pruned_student_outputs = model(**inputs, return_dict=False)
    
    one_masked_metrics = trainer.evaluate()
    model.head_mask[torch.randperm(model.head_mask.numel())[:int(model.head_mask.numel() * 0.2)]] = 0
    model.intermediate_mask[torch.randperm(model.intermediate_mask.numel())[:int(model.intermediate_mask.numel() * 0.2)]] = 0
    model.hidden_mask[torch.randperm(model.hidden_mask.numel())[:int(model.hidden_mask.numel() * 0.1)]] = 0
    model.prune_model_with_masks()
    
    inputs = next(iter(trainer.get_train_dataloader()))
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    teacher_metrics = trainer.evaluate()
    model.head_mask[torch.randperm(model.head_mask.numel())[:int(model.head_mask.numel() * 0.2)]] = 0
    model.intermediate_mask[torch.randperm(model.intermediate_mask.numel())[:int(model.intermediate_mask.numel() * 0.2)]] = 0
    model.hidden_mask[torch.randperm(model.hidden_mask.numel())[:int(model.hidden_mask.numel() * 0.1)]] = 0
    
    trainer.teacher_model = model
    masked_metrics = trainer.evaluate()
    masked_outputs = model(**inputs, return_dict=False)
    model.virtual_prune()
    for i in range(12):
        model.encoder.block[i].layer[0].SelfAttention.register_forward_pre_hook(lambda x, i: print('forwarded', flush=True))
        print(model.encoder.block[i].layer[0].SelfAttention.n_heads, model.encoder.block[i].layer[0].SelfAttention.n_teacher_heads)
    
    
    virtual_pruned_outputs = model(**inputs, return_dict=False)
    teacher_outputs = model(**inputs, return_dict=False, use_teacher=True)
    virtual_pruned_metrics = trainer.evaluate()
    model.virtual_prune_restore()