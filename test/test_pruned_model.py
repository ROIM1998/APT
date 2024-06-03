import seaborn as sns
sns.set_theme(style="darkgrid")
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
import loralib as lora

from copy import deepcopy
from transformers import (HfArgumentParser)
from args import DataTrainingArguments
from models.model_args import ModelArguments
from utils.utils import *
from utils import build_trainer, build_dataloader
from args import MinusTrainingArguments
from models import build_model, convert_layers_based_on_ckpt
from prune import AdapterPruner
from torch.utils.data import Subset
from trainer.param_control import ParamController
from matplotlib import pyplot as plt
from utils.fisher_utils.efficiency.param import *
from utils.minus_utils import compare_module_inputs_equality, load_grafting_masks

def main():
    sys.argv = ['test_pre_tuning_prune.py',
            '--output_dir',
            './output/test_pruned_model/',
            '--model_name_or_path',
            'output/bert-base-uncased_lora_minus_sst2_cubic_gradual_running_fisher_alloc_running_fisher_momentum_mapping_static_teacher_dynamic_cofi_student_distill_tophalf_limited_resizing/mac0.4/epoch20/bz32/numprune10/paramq:0-11,v:0-11,i:0-11/lora_r8/pruning_start-1/distill_epoch10/best_model',
            '--task_name',
            'sst2',
            '--do_train',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '32',
            '--per_device_eval_batch_size',
            '128',
            '--apply_lora',
            '--lora_r',
            '8',
            '--lora_alpha',
            '16',
            '--save_strategy',
            'no',
            '--evaluation_strategy',
            'steps',
            '--num_train_epochs',
            '20',
            '--learning_rate',
            '2e-4',
            '--weight_decay',
            '0.1',
            '--warmup_ratio',
            '0.06',
            '--report_to',
            'none',
            '--teacher_param_tuning_config',
            'q:0-11,v:0-11,i:0-11',
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
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets, force_model_shape_deduction=True)
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)


    pruning_batch_size = 4
    num_pruning_batches = 64
    dataloader = build_dataloader(Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]), pruning_batch_size, data_args, training_args, tokenizer)

    teacher_keys = ['query', 'value', 'intermediate']
    teacher_config  = {
        k: [i for i in range(model.config.num_hidden_layers)] for k in teacher_keys
    }
    student_keys = ['query', 'value', 'intermediate']
    student_config = {
        k: [i for i in range(model.config.num_hidden_layers)] for k in student_keys
    }
    adapter_pruner = AdapterPruner(model, dataloader)
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        student_config=student_config,
        lora_with_bias=False,
        adapter_pruner=adapter_pruner,
    )
    param_controller.convert_to_pre_pruning_lora_teacher()
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    param_controller.model_as_teacher()
    
    load_grafting_masks(model, torch.load(os.path.join(model_args.model_name_or_path, 'grafting_masks.pt')))
    model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt'))
    model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt'))
    model.hidden_mask = torch.load(os.path.join(model_args.model_name_or_path, 'hidden_mask.pt'))
    for m in model.modules():
        if isinstance(m, lora.PruningLinear):
            m.scaling = 2
    
    model.prune_model_with_masks()
    trainer.evaluate()


    model_args.model_name_or_path = 'output/debug_running_fisher_dynamic_distill/post_pruning_model_step1152'
    config, tokenizer, pruned_model = build_model(model_args, data_args, training_args, t_name, raw_datasets, force_model_shape_deduction=True)
    pruned_param_controller = ParamController(
        pruned_model,
        teacher_config=teacher_config,
        student_config=student_config,
        lora_with_bias=False,
        adapter_pruner=adapter_pruner,
    )
    pruned_trainer = build_trainer(data_args, training_args, pruned_model, tokenizer, train_dataset, eval_dataset, param_controller=pruned_param_controller)
    pruned_param_controller.convert_to_pre_pruning_lora_teacher()
    load_grafting_masks(pruned_model, torch.load(os.path.join(model_args.model_name_or_path, 'grafting_masks.pt')))
    pruned_model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt'))
    pruned_model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt'))
    pruned_model.hidden_mask = torch.load(os.path.join(model_args.model_name_or_path, 'hidden_mask.pt'))
    pruned_trainer.evaluate()
    
    print((torch.cat(pruned_model.head_mask) == torch.cat(model.head_mask)).all())
    print((torch.cat(pruned_model.intermediate_mask) == torch.cat(model.intermediate_mask)).all())
    print((pruned_model.hidden_mask == model.hidden_mask).all())
    
    
    model_params = dict(model.named_parameters())
    pruned_model_params = dict(pruned_model.named_parameters())
    [k for k in model_params if model_params[k].shape != pruned_model_params[k].shape or not (model_params[k] == pruned_model_params[k]).all()]