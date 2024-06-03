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
from utils.minus_utils import lora_to_prunelora, lora_to_linear
from utils import build_trainer, build_dataloader
from args import MinusTrainingArguments
from models import build_model
from prune.pruner import AdapterPruner
from torch.utils.data import Subset
from trainer.param_control import ParamController
from prune import build_scorer, BetterFisherPruner

def main():
    sys.argv = ['test_weight_restoration.py',
            '--output_dir',
            './output/test_model_weight_restoration/',
            '--model_name_or_path',
            'output/roberta-base_lora_minus_mnli_once_global_nodistill/mac0.4/epoch20/bz128/numprune5/lora_r64/lora_alpha16/pre_pruning_model',
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
            '--lora_r',
            '8',
            '--lora_alpha',
            '16',
            '--save_strategy',
            'no',
            '--evaluation_strategy',
            'steps',
            '--num_train_epochs',
            '0.1',
            '--learning_rate',
            '5e-4',
            '--weight_decay',
            '0.1',
            '--warmup_ratio',
            '0.06',
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
    os.makedirs(training_args.output_dir, exist_ok=True)
    # training_args.disable_tqdm = False
    t_name, raw_datasets = get_raw_datasets(model_args, data_args, training_args)
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)
    
    named_modules = dict(model.named_modules())
    for n, p in model.named_modules():
        if isinstance(p, lora.Linear):
            parent_layer_attr, attr = n.rsplit('.', 1)
            parent_layer = named_modules[parent_layer_attr]
            if 'intermediate' in n:
                setattr(parent_layer, attr, lora_to_linear(p))
            else:
                new_layer = lora_to_prunelora(p, r=p.r, lora_alpha=p.lora_alpha)
                new_layer.set_grafting_mask()
                setattr(parent_layer, attr, new_layer)

    model.head_mask, model.intermediate_mask = model.head_mask.to(training_args.device), model.intermediate_mask.to(training_args.device)

    pruning_batch_size = 4
    num_pruning_batches = 64
    dataloader = build_dataloader(Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]), pruning_batch_size, data_args, training_args, tokenizer)

    teacher_keys = ['query', 'value']
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
    
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)    
    param_controller.model_as_teacher()
    tuning_param_num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f'tuning_param_num: {tuning_param_num}')
    param_controller.set_fixed_tuning_param_number()
    
    # Pruning the model before re-allocating the dimensions
    training_args.seq_len = 128
    training_args.cls_task = True
    for p in model.parameters():
            p.requires_grad = False
    scorer = build_scorer('gradient_l2', model, dataloader)
    pruner = BetterFisherPruner(model, ['head_mask', 'intermediate_mask'], {'head_mask': scorer, 'intermediate_mask': scorer}, training_args.seq_len, training_args.cls_task, ['search', 'better_rearrange', 'global'])
    masks = pruner.generate_mask(0.8)
    head_mask, intermediate_mask = model.head_mask.clone(), model.intermediate_mask.clone()
    model.prune_model_with_masks()

    # Further pruning the model
    for p in model.parameters():
            p.requires_grad = False
    model.head_mask, model.intermediate_mask = [torch.ones(layer.attention.self.num_attention_heads).to(model.device) for layer in model.roberta.encoder.layer], [torch.ones(layer.intermediate.dense.out_features).to(model.device) for layer in model.roberta.encoder.layer]
    # new_head_grads, new_intermediate_grads = collect_mask_grads(model, dataloader)
    inputs = next(iter(dataloader))
    inputs = trainer._prepare_inputs(inputs)
    scorer = build_scorer('gradient_l2', model, dataloader)
    pruner = BetterFisherPruner(model, ['head_mask', 'intermediate_mask'], {'head_mask': scorer, 'intermediate_mask': scorer}, training_args.seq_len, training_args.cls_task, ['search', 'better_rearrange', 'global'])
    masks = pruner.generate_mask(0.6)
    head_mask, intermediate_mask = [m.clone() for m in model.head_mask], [m.clone() for m in model.intermediate_mask]
    pre_pruned_outputs = model(**inputs, output_hidden_states=True, return_dict=False)
    pre_pruned_metrics = trainer.evaluate()
    model.prune_model_with_masks()
    post_pruned_outputs = model(**inputs, output_hidden_states=True, return_dict=False)
    post_pruned_metrics = trainer.evaluate()

if __name__ == '__main__':
    main()