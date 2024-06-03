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
    adapter_pruner = AdapterPruner(model, dataloader)
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        student_config=None,
        lora_with_bias=False,
        adapter_pruner=adapter_pruner,
        param_allocation_strategy=training_args.param_allocation_strategy,
    )
    
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)    
    param_controller.model_as_teacher()
    tuning_param_num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f'tuning_param_num: {tuning_param_num}')
    param_controller.set_fixed_tuning_param_number()
    
    # Pruning the model before re-allocating the dimensions
    training_args.seq_len = 128
    training_args.cls_task = True
    scorer = build_scorer('gradient_l2', model, dataloader)
    pruner = BetterFisherPruner(model, ['head_mask', 'intermediate_mask'], {'head_mask': scorer, 'intermediate_mask': scorer}, training_args.seq_len, training_args.cls_task, ['search', 'better_rearrange', 'global'])
    masks = pruner.generate_mask(0.4)
    head_mask, intermediate_mask = model.head_mask.clone(), model.intermediate_mask.clone()
    model.prune_model_with_masks()
    # post_pruning_metrics = trainer.evaluate()
    bottleneck_names, output_dim_masks, all_bottleneck_mask, input_dim_masks, target_rs, all_scores = param_controller.allocate_dims(0.8)
    for name, mask in zip(bottleneck_names, all_bottleneck_mask):
        print(mask.sum().item() / mask.numel(), target_rs[name.rsplit('.', 1)[0]] - mask.sum().item())
    # post_conversion_metrics = trainer.evaluate()
    
    # Vaildate the output value consistency
    validate_output_consistency = False
    named_modules = dict(model.named_modules())
    if validate_output_consistency:
        config, tokenizer, ref_model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
        ref_model.head_mask, ref_model.intermediate_mask = head_mask, intermediate_mask
        ref_model = ref_model.to(training_args.device)
        ref_model.prune_model_with_masks()
        ref_model.eval()
        inputs = next(iter(dataloader))
        inputs = trainer._prepare_inputs(inputs)
        ref_outputs = ref_model(**inputs, output_hidden_states=True, return_dict=False)
        outputs = model(**inputs, output_hidden_states=True, return_dict=False)
        for i, (ref_o, o) in enumerate(zip(ref_outputs[-1], outputs[-1])):
            print(f'layer {i}', (ref_o - o).abs().max().item())
        
        ref_modules = dict(ref_model.named_modules())
        for n, p in ref_model.named_modules():
            if isinstance(p, lora.Linear):
                parent_layer_attr, attr = n.rsplit('.', 1)
                parent_layer = ref_modules[parent_layer_attr]
                if 'intermediate' in n:
                    setattr(parent_layer, attr, lora_to_linear(p))
                else:
                    new_layer = lora_to_prunelora(p, r=p.r, lora_alpha=p.lora_alpha)
                    new_layer.set_grafting_mask()
                    setattr(parent_layer, attr, new_layer)
        
        ref_modules = dict(ref_model.named_modules())
        for name in bottleneck_names:
            layer_name = name.rsplit('.', 1)[0]
            layer = named_modules[layer_name]
            layer.eval()
            assert layer.merged
            history = layer.history[0]
            print(f'{name} weight difference max:', ((history['scaling'] * (history['lora_B'] @ history['lora_A']) + history['weight']) - layer.weight.cpu()).abs().max())
            layer.train()

    # Validate the consistency between the masks and the current model layers
    named_modules = dict(model.named_modules())
    for bottleneck_name, input_mask, output_mask, bottleneck_mask in zip(bottleneck_names, input_dim_masks, output_dim_masks, all_bottleneck_mask):
        layer_name = bottleneck_name.rsplit('.', 1)[0]
        layer = named_modules[layer_name]
        pruned_bottleneck_dim = (bottleneck_mask == 0).nonzero().squeeze()
        pruned_out_dim = (output_mask == 0).nonzero().squeeze()
        pruned_in_dim = (input_mask == 0).nonzero().squeeze()
        pruned_bottleneck_dim = pruned_bottleneck_dim.tolist() if pruned_bottleneck_dim.dim() else [pruned_bottleneck_dim.item()]
        pruned_out_dim = pruned_out_dim.tolist() if pruned_out_dim.dim() else [pruned_out_dim.item()]
        pruned_in_dim = pruned_in_dim.tolist() if pruned_in_dim.dim() else [pruned_in_dim.item()]
        if (not input_mask.any()) or (not output_mask.any()) or (not bottleneck_mask.any()):
            assert layer.lora_A is None and layer.lora_B is None and layer.out_transformation is None and layer.in_transformation is None and layer.r == 0
        else:
            assert layer.lora_A.shape[1] == layer.in_features - len(pruned_in_dim)
            assert layer.lora_B.shape[0] == layer.out_features - len(pruned_out_dim)
            assert layer.out_transformation.nonzero()[:, 1].tolist() == sorted(list(set(range(layer.out_features)) - set(pruned_out_dim)))
            assert layer.in_transformation.nonzero()[:, 0].tolist() == sorted(list(set(range(layer.in_features)) - set(pruned_in_dim)))
            
    # Testing the restore_dim functions for the lora layers
    trainer.auto_layer_conversion = False
    train_result = trainer.train()
    
    # post_training_metrics = trainer.evaluate()
    param_controller.restore_dims(target='teacher')
    # post_restore_metrics = trainer.evaluate()

    # Further pruning the model
    model.reset_masks()
    # new_head_grads, new_intermediate_grads = collect_mask_grads(model, dataloader)
    scorer = build_scorer('gradient_l2', model, dataloader)
    pruner = BetterFisherPruner(model, ['head_mask', 'intermediate_mask'], {'head_mask': scorer, 'intermediate_mask': scorer}, training_args.seq_len, training_args.cls_task, ['search', 'better_rearrange', 'global'])
    masks = pruner.generate_mask(0.6)
    new_head_mask, new_intermediate_mask = [m.clone() for m in model.head_mask], [m.clone() for m in model.intermediate_mask]
    pre_prune_masked_metrics = trainer.evaluate()
    model.prune_model_with_masks()
    post_prune_masked_metrics = trainer.evaluate()
    
    # Compare with a direct pruning model
    config, tokenizer, original_model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    original_model.head_mask, original_model.intermediate_mask = new_head_mask, new_intermediate_mask
    original_model.prune_model_with_masks()
    original_model_trainer = build_trainer(data_args, training_args, original_model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    original_model_trainer.evaluate()
    inputs = next(iter(dataloader))
    inputs = trainer._prepare_inputs(inputs)
    original_outputs = original_model(**inputs, output_hidden_states=True, return_dict=False)
    outputs = model(**inputs, output_hidden_states=True, return_dict=False)
    for i in range(13):
        print(f'layer {i}', (original_outputs[-1][i] - outputs[-1][i]).abs().max().item())
    
    # Test continuous allocating
    bottleneck_names, output_dim_masks, all_bottleneck_mask, input_dim_masks, target_rs, all_scores = param_controller.allocate_dims(0.8)
    for name, mask in zip(bottleneck_names, all_bottleneck_mask):
        print(name, ":", mask.sum().item(), "out of", mask.numel(), "dimensions are retained")

if __name__ == '__main__':
    main()