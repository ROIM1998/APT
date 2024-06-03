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
from utils import build_trainer, build_dataloader
from args import MinusTrainingArguments
from models import build_model
from prune.pruner import AdapterPruner
from torch.utils.data import Subset
from trainer.param_control import ParamController
from prune import build_scorer, BetterFisherPruner

def main():
    sys.argv = ['test_adapt_pruning.py',
            '--output_dir',
            './output/test_adapt_pruning/',
            '--model_name_or_path',
            'output/roberta-base_lora_minus_mnli_once_global_free_inout_nodistill/mac0.4/epoch30/bz128/numprune5/paramq:0-11,v:0-11,i:0-11/lora_r32/lora_alpha16/pre_pruning_model',
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
            '32',
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
            '--do_distill',
            '--continuous_allocation',
            '--continuous_alloc_interval',
            '1',
            '--distillation_type',
            'self_interleave',
            '--distill_mapping_strategy',
            'none',
            '--param_allocation_strategy',
            'free_inout'
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
        param_allocation_strategy=training_args.param_allocation_strategy,
    )
    
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    if isinstance(model.head_mask, torch.Tensor) and isinstance(model.intermediate_mask, torch.Tensor):
        model.head_mask, model.intermediate_mask = model.head_mask.to(training_args.device), model.intermediate_mask.to(training_args.device)
    else:
        model.reset_masks()
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
    masks = pruner.generate_mask(0.4)
    head_mask, intermediate_mask = [v.clone() for v in model.head_mask], [v.clone() for v in model.intermediate_mask]
    # model.prune_model_with_masks()
    # post_pruning_metrics = trainer.evaluate()
    param_controller.convert_to_distill(head_mask, intermediate_mask)
    bottleneck_names, output_dim_masks, all_bottleneck_mask, input_dim_masks, target_rs, all_scores = param_controller.allocate_dims(0.8)
    param_controller.restore_dims(target='student')
    param_controller.set_grafting_mask(mode=True, target='student')
    inputs = next(iter(dataloader))
    inputs = trainer._prepare_inputs(inputs)
    suffix = '.bottleneck_mask'
    all_grads = {}
    pre_collect_tuning_status = {}
    for n, p in model.named_parameters():
        if n.endswith(suffix):
            p.requires_grad = True
            p.retain_grad()
            all_grads[n] = []
        else:
            pre_collect_tuning_status[n] = p.requires_grad
            p.requires_grad = False
    outputs = model(**inputs, output_hidden_states=True, return_dict=False)
    loss = outputs[0]
    loss.backward()
    # post_conversion_metrics = trainer.evaluate()
    
            
    # Testing the restore_dim functions for the lora layers
    trainer.auto_layer_conversion = False
    train_result = trainer.train()
    
    post_training_metrics = trainer.evaluate()
    named_modules = dict(model.named_modules())
    for bottleneck_name in bottleneck_names:
        layer_name = bottleneck_name.rsplit('.', 1)[0]
        layer = named_modules[layer_name]
        layer.restore_dims()
    post_restore_metrics = trainer.evaluate()

    # Further pruning the model
    for p in model.parameters():
            p.requires_grad = False
    model.head_mask, model.intermediate_mask = [torch.ones(layer.attention.self.num_attention_heads).to(model.device) for layer in model.roberta.encoder.layer], [torch.ones(layer.intermediate.dense.out_features).to(model.device) for layer in model.roberta.encoder.layer]
    # new_head_grads, new_intermediate_grads = collect_mask_grads(model, dataloader)
    scorer = build_scorer('gradient_l2', model, dataloader)
    pruner = BetterFisherPruner(model, ['head_mask', 'intermediate_mask'], {'head_mask': scorer, 'intermediate_mask': scorer}, training_args.seq_len, training_args.cls_task, ['search', 'better_rearrange', 'global'])
    masks = pruner.generate_mask(0.6)
    new_head_mask, new_intermediate_mask = [m.clone() for m in model.head_mask], [m.clone() for m in model.intermediate_mask]
    pre_prune_masked_metrics = trainer.evaluate()
    model.prune_model_with_masks()
    post_prune_masked_metrics = trainer.evaluate()
    

if __name__ == '__main__':
    main()