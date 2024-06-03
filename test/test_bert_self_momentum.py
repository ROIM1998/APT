import os
import sys
import torch
import torch.nn as nn
import loralib as lora

from copy import deepcopy
from typing import Tuple
from transformers import (HfArgumentParser)
from torch.utils.data import Subset
from args import DataTrainingArguments
from models.model_args import ModelArguments
from args import MinusTrainingArguments
from utils.utils import *
from utils import build_trainer, build_dataloader
from models import build_model
from trainer.param_control import ParamController
from utils.minus_utils import load_grafting_masks, compare_module_inputs_equality

MB = 1024 * 1024

def main():
    sys.argv = ['test_bert.py',
            '--output_dir',
            './output/test_magnitude_scorer/',
            '--model_name_or_path',
            'output/bert-base-uncased/sst2/bz32/elastictuning/mac0.5/epoch40/distill_epoch20/numprune10/sparsity_warmup1/pruning_start-1/pruning_stop3/lora_r8/lora_alpha16/warmup_paramq:0-11,v:0-11,i:0-11/teacher_paramq:0-11,v:0-11,i:0-11/distill_self_momentum/distill_mapping_dynamic_block_teacher_dynamic_student/post_pruning_model_step3057',
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
            '--do_distill',
            '--lora_r',
            '8',
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
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets, force_model_shape_deduction=True)
    
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)
    model = model.to(training_args.device)
    model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt'), map_location='cpu')
    model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt'), map_location='cpu')
    model.hidden_mask = torch.load(os.path.join(model_args.model_name_or_path, 'hidden_mask.pt'), map_location='cpu')
    
    pruning_batch_size = 32
    num_pruning_batches = 64
    dataloader = build_dataloader(Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]), pruning_batch_size, data_args, training_args, tokenizer)
    dataloader = build_dataloader(data_args, training_args, model, tokenizer, raw_datasets)

    # Also add ffn input layers to teacher config
    warmup_keys = ['query', 'value']
    warmup_config  = {
        k: [i for i in range(model.config.num_hidden_layers)] for k in warmup_keys
    }
    teacher_keys = ['query', 'value', 'intermediate']
    teacher_config  = {
        k: [i for i in range(model.config.num_hidden_layers)] for k in teacher_keys
    }
    param_controller = ParamController(
        model,
        warmup_config=warmup_config,
        teacher_config=teacher_config,
        student_config=None,
        lora_with_bias=False,
        param_allocation_strategy=training_args.param_allocation_strategy,
    )
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    model.head_mask = model.head_mask.to(training_args.device)
    model.intermediate_mask = model.intermediate_mask.to(training_args.device)
    model.hidden_mask = model.hidden_mask.to(training_args.device)


    # Reset scalings for LoRA layers
    for m in model.modules():
        if isinstance(m, lora.Linear):
            m.scaling = 2.0
            m.lora_alpha = 2 * m.r
            if isinstance(m, lora.DistillLinear):
                m.teacher_scaling = 2.0
                m.teacher_lora_alpha = 2 * m.r

    trainer.distilling = True
    trainer.evaluate()
                
    trainer.teacher_model_masks = torch.load(os.path.join(model_args.model_name_or_path, 'teacher_model_masks.pt'), map_location=training_args.device)

    load_grafting_masks(model, torch.load(os.path.join(model_args.model_name_or_path, 'grafting_masks.pt')))
    loaded_weights = torch.load(os.path.join(model_args.model_name_or_path, 'pytorch_model.bin'), map_location=training_args.device)
    all((loaded_weights[k] == v).all() for k, v in model.state_dict().items() if k in loaded_weights)
    trainer.distilling = True
    pre_pruning_metrics = trainer.evaluate()
    
    inputs = next(iter(dataloader))
    inputs = trainer._prepare_inputs(inputs)
    pre_pruning_outputs = trainer.model(**inputs, output_hidden_states=True, return_dict=False)
    pre_pruning_teacher_outputs = trainer.teacher_model(**inputs, output_hidden_states=True, return_dict=False, use_teacher=True, head_z=trainer.teacher_model_masks['head_mask'], intermediate_z=trainer.teacher_model_masks['intermediate_mask'], hidden_z=trainer.teacher_model_masks['hidden_mask'])
    # original_model = deepcopy(model)

    model_args.model_name_or_path = 'output/bert-base-uncased_lora_minus_sst2_cubic_gradual_running_fisher_alloc_running_fisher_self_momentum_mapping_static_teacher_dynamic_cofi_student_distill_tophalf_limited_resizing_noffnstart/mac0.4/epoch40/bz32/numprune10/paramq:0-11,v:0-11,i:0-11/lora_r8/pruning_start-1/distill_epoch20/post_converting_model_step4209'
    config, tokenizer, pretune_model = build_model(model_args, data_args, training_args, t_name, raw_datasets, force_model_shape_deduction=True)
    pretune_model.to(training_args.device)
    pretune_model.eval()
    pretune_model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt'), map_location=training_args.device)
    pretune_model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt'), map_location=training_args.device)
    pretune_model.hidden_mask = torch.load(os.path.join(model_args.model_name_or_path, 'hidden_mask.pt'), map_location=training_args.device)
    # Reset scalings for LoRA layers
    for m in pretune_model.modules():
        if isinstance(m, lora.Linear):
            m.scaling = 2.0
            m.lora_alpha = 2 * m.r
            if isinstance(m, lora.DistillLinear):
                m.teacher_scaling = 2.0
                m.teacher_lora_alpha = 2 * m.r
    pretune_param_controller = ParamController(
        pretune_model,
        warmup_config=warmup_config,
        teacher_config=teacher_config,
        student_config=None,
        lora_with_bias=False,
        param_allocation_strategy=training_args.param_allocation_strategy,
    )
    pretune_trainer = build_trainer(data_args, training_args, pretune_model, tokenizer, train_dataset, eval_dataset, param_controller=pretune_param_controller)
    pretune_trainer.distilling = True
    pretune_trainer.evaluate()

    pretune_teacher_outputs = pretune_model(**inputs, output_hidden_states=True, return_dict=False, use_teacher=True, head_z=trainer.teacher_model_masks['head_mask'], intermediate_z=trainer.teacher_model_masks['intermediate_mask'], hidden_z=trainer.teacher_model_masks['hidden_mask'])
    for i in range(13):
        print(torch.norm(pre_pruning_teacher_outputs[-1][i] - pretune_teacher_outputs[-1][i]))
    with torch.no_grad():
        res = compare_module_inputs_equality(
            [model, pretune_model],
            inputs,
            lambda x: x.bert.encoder.layer[0].intermediate.intermediate_act_fn,
            use_teacher=True,
            head_z=trainer.teacher_model_masks['head_mask'],
            intermediate_z=trainer.teacher_model_masks['intermediate_mask'],
            hidden_z=trainer.teacher_model_masks['hidden_mask'],
        )
        print(torch.norm(res[0] - res[1]))

    # If can, pruning the model before setting it as teacher
    trainer.prune_model()
    for k in ['head_mask', 'intermediate_mask', 'hidden_mask']:
        mask = getattr(model, k, None)
        if mask is not None:
            mask = mask.view(-1).detach().clone() if isinstance(mask, torch.Tensor) else torch.cat([m.view(-1).detach().clone() for m in mask])
        # mask.requires_grad = True
        # mask.retain_grad()
        setattr(model, k, mask)
        # Update accumulated salience and uncertainty
    trainer.teacher_model = trainer.model
    trainer.teacher_model_masks = {
        'head_mask': trainer.model.head_mask.detach().clone(),
        'intermediate_mask': trainer.model.intermediate_mask.detach().clone(),
        'hidden_mask': trainer.model.hidden_mask.detach().clone()
    }
    post_pruning_outputs = trainer.model(**inputs, output_hidden_states=True, return_dict=False)
    post_pruning_teacher_outputs = trainer.teacher_model(**inputs, output_hidden_states=True, return_dict=False, use_teacher=True, head_z=trainer.teacher_model_masks['head_mask'], intermediate_z=trainer.teacher_model_masks['intermediate_mask'], hidden_z=trainer.teacher_model_masks['hidden_mask'])
    
    # Compare the hidden states of the pre-pruning and post-pruning student outputs
    for i, (pre_hidden_states, post_hidden_states) in enumerate(zip(pre_pruning_outputs[-1], post_pruning_outputs[-1])):
        print(i, torch.norm(pre_hidden_states - post_hidden_states))

    # Compare the hidden states of the pre-pruning and post-pruning teacher outputs
    for i, (pre_hidden_states, post_hidden_states) in enumerate(zip(pre_pruning_teacher_outputs[-1], post_pruning_teacher_outputs[-1])):
        print(i, torch.norm(pre_hidden_states - post_hidden_states))
    
    param_controller.convert_to_self_momentum_distill()
    trainer.set_tuning_params(trainer.state.epoch, trainer.state.global_step)
    post_pruning_metrics = trainer.evaluate()
    # Compare the converted teacher hidden states to the post-pruning student hidden states
    new_teacher_outputs = trainer.teacher_model(**inputs, output_hidden_states=True, return_dict=False, use_teacher=True, head_z=trainer.teacher_model_masks['head_mask'], intermediate_z=trainer.teacher_model_masks['intermediate_mask'], hidden_z=trainer.teacher_model_masks['hidden_mask'])
    for i, (new_teacher_hidden_states, post_hidden_states) in enumerate(zip(new_teacher_outputs[-1], post_pruning_outputs[-1])):
        print(i, torch.norm(new_teacher_hidden_states - post_hidden_states))
    
    [n for n, m in model.named_modules() if isinstance(m, lora.Linear) and not hasattr(m, 'teacher_lora_A')]
    all((m.lora_A == m.teacher_lora_A).all() for m in model.modules() if isinstance(m, lora.DistillLinear))
    all((m.lora_B == m.teacher_lora_B).all() for m in model.modules() if isinstance(m, lora.DistillLinear))
    
    save_dir = 'output/bert-base-uncased_lora_minus_sst2_cubic_gradual_running_fisher_alloc_running_fisher_self_momentum_mapping_static_teacher_dynamic_cofi_student_distill_tophalf_limited_resizing_noffnstart/mac0.4/epoch40/bz32/numprune10/paramq:0-11,v:0-11,i:0-11/lora_r8/pruning_start-1/distill_epoch20/'
    
    pre_pruning_dirs = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if 'pre_pruning' in f]
    pre_pruning_dirs.sort(key=lambda x: int(x.split('step')[-1]))
    post_pruning_dirs = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if 'post_pruning' in f]
    post_pruning_dirs.sort(key=lambda x: int(x.split('step')[-1]))
    
    tuned_parameter_history = []
    # Compare tuning, meaning the first post-pruned model with the second pre-pruned model, and ...
    for pre_tune_dir, post_tune_dir in zip(post_pruning_dirs[1:-1], pre_pruning_dirs[2:]):
        model_args.model_name_or_path = pre_tune_dir
        config, tokenizer, pre_prune_model = build_model(model_args, data_args, training_args, t_name, raw_datasets, force_model_shape_deduction=True)
        model_args.model_name_or_path = post_tune_dir
        config, tokenizer, post_prune_model = build_model(model_args, data_args, training_args, t_name, raw_datasets, force_model_shape_deduction=True)
        pre_weights = torch.load(os.path.join(pre_tune_dir, 'pytorch_model.bin'), map_location='cpu')
        post_weights = torch.load(os.path.join(post_tune_dir, 'pytorch_model.bin'), map_location='cpu')
        tuned_parameters = {
            k: pre_weights[k].numel() for k in post_weights if pre_weights[k].shape != post_weights[k].shape or not torch.allclose(pre_weights[k], post_weights[k])
        }
        mismatch_parameters = {
            k: pre_weights[k].numel() for k in post_weights if pre_weights[k].shape != post_weights[k].shape
        }
        all([torch.allclose(pre_weights[k], post_weights[k]) for k in post_weights if 'teacher' in k])
        print(pre_tune_dir.split('step')[-1], sum(tuned_parameters.values()))
        tuned_parameter_history.append(tuned_parameters)
        break
    
if __name__ == '__main__':
    main()