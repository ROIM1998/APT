import os
import sys
import torch
import torch.nn as nn
import loralib as lora

from tqdm import tqdm
from typing import Tuple
from transformers import (HfArgumentParser, set_seed)
from torch.utils.data import Subset
from args import DataTrainingArguments
from models.model_args import ModelArguments
from args import MinusTrainingArguments
from utils.utils import *
from utils import build_trainer, build_dataloader
from models import build_model
from trainer.param_control import ParamController

MB = 1024 * 1024

def main():
    sys.argv = ['test_bert.py',
            '--output_dir',
            './output/test_magnitude_scorer/',
            '--model_name_or_path',
            'google/t5-xl-lm-adapt',
            '--task_name',
            'sst2',
            '--do_train',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '32',
            '--per_device_eval_batch_size',
            '32',
            '--apply_lora',
            '--do_distill',
            '--lora_r',
            '8',
            '--report_to',
            'none',
            '--pruner_type',
            'running_fisher',
            '--pre_tuning_scorer',
            'backward_running_hidden_states_salience',
            '--pre_tuning_pruner',
            'running_fisher',
            '--pruning_batch_size',
            '4',
            '--pruning_batches',
            '64',
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
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets, generative=True)
    set_seed(128)
    
    
    model = model.to(training_args.device)
    model.head_mask = model.head_mask.to(training_args.device).view(-1).to(model.device)
    model.intermediate_mask = model.intermediate_mask.to(training_args.device).view(-1).to(model.device)
    model.hidden_mask = model.hidden_mask.to(training_args.device).view(-1).to(model.device)

    # Also add ffn input layers to teacher config
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
        model.head_mask, model.intermediate_mask, model.hidden_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt')).to(training_args.device), torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt')).to(training_args.device), torch.load(os.path.join(model_args.model_name_or_path, 'hidden_mask.pt')).to(training_args.device)
    
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    dataloader = trainer.pruning_dataloader
    
    # param_controller.convert_to_pruning_lora_teacher()
    # param_controller.model_as_teacher()
    
    inputs = next(iter(dataloader))
    inputs = trainer._prepare_inputs(inputs)
    trainer.shortens_inputs(inputs)
    
    outputs = model(**inputs, return_dict=False)
    outputs[0].backward()
    print(torch.cuda.max_memory_allocated() / 1024 / 1024)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    trainer.pre_tuning_pruning_scorer.step()
    print(torch.cuda.max_memory_allocated() / 1024 / 1024)
    trainer.pre_tuning_pruning_pruner.update_mask(0.5, is_last=True)
    trainer.pre_tuning_pruning_scorer.end()
    trainer.prune_model()

    param_controller.model_as_teacher()
    
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    for m in model.modules():
        if isinstance(m, lora.Linear):
            m.scaling = 2
            if isinstance(m, lora.DistillLinear):
                m.teacher_scaling = 2
    
    
    inputs = next(iter(dataloader))
    inputs = trainer._prepare_inputs(inputs)

    # old_scorer = build_scorer('running_hidden_states_salience', model, None, param_controller = param_controller, state=trainer.state, gather_freq=1, beta_1=0.85, beta_2=0.85, use_uncertainty=False, block_normalize_dict=None)
    
    scorer = trainer.scorer
    
    outputs = model(**inputs, return_dict=False)
    print([len(v) for v in trainer.scorer.current_mha_states], [len(v) for v in trainer.scorer.current_ffn_states], len(trainer.scorer.current_hidden_states))
    print([len(v) for v in trainer.scorer.all_head_scores], [len(v) for v in trainer.scorer.all_ffn_scores])
    mha_shapes = [[v.shape[0] for v in trainer.scorer.current_mha_states[0]], [v.shape[0] for v in trainer.scorer.current_mha_states[1]], [v.shape[0] for v in trainer.scorer.current_mha_states[2]]]
    ffn_shapes = [[v.shape[0] if isinstance(v, torch.Tensor) else v[1].shape[0] for v in trainer.scorer.current_ffn_states[0]], [v.shape[0] if isinstance(v, torch.Tensor) else v[1].shape[0] for v in trainer.scorer.current_ffn_states[1]]]
    hidden_shapes = [v.shape[0] for v in trainer.scorer.current_hidden_states]
    
    outputs[0].backward()
    print([len(v) for v in trainer.scorer.current_mha_states], [len(v) for v in trainer.scorer.current_ffn_states], len(trainer.scorer.current_hidden_states))
    print([len(v) for v in trainer.scorer.all_head_scores], [len(v) for v in trainer.scorer.all_ffn_scores])
    mha_score_shapes = [[v.shape[0] for v in trainer.scorer.all_head_scores[0]], [v.shape[0] for v in trainer.scorer.all_head_scores[1]], [v.shape[0] for v in trainer.scorer.all_head_scores[2]]]
    ffn_score_shapes = [[v.shape[0] if isinstance(v, torch.Tensor) else v[1].shape[0] for v in trainer.scorer.all_ffn_scores[0]], [v.shape[0] if isinstance(v, torch.Tensor) else v[1].shape[0] for v in trainer.scorer.all_ffn_scores[1]]]
    ffn_score_shapes[0].reverse()
    ffn_score_shapes[1].reverse()
    ffn_real_shapes = [
        [model.encoder.block[i].layer[1].DenseReluDense.wo.weight.shape[1] if model.encoder.block[i].layer[1].DenseReluDense.wo is not None else 0 for i in range(24)],
        [model.decoder.block[i].layer[2].DenseReluDense.wo.weight.shape[1] if model.decoder.block[i].layer[2].DenseReluDense.wo is not None else 0 for i in range(24)],
    ]
    trainer.scorer.step()
    
    def log_io(module, inputs, outputs):
        module.logged = (inputs, outputs)
    
    handlers = []
    for i in range(24):
        handler = model.decoder.block[i].layer[2].DenseReluDense.wo.register_forward_hook(
            log_io
        )
        handlers.append(handler)
        handler = model.decoder.block[i].layer[2].DenseReluDense.register_forward_hook(
            log_io
        )
        handlers.append(handler)
        handler = model.decoder.block[i].layer[2].register_forward_hook(
            log_io
        )
        handlers.append(handler)
    
    for i in range(24):
        print(hasattr(model.decoder.block[i].layer[2].DenseReluDense.wo, 'logged'))
    
    
    print([len(v) for v in trainer.scorer.current_mha_states], [len(v) for v in trainer.scorer.current_ffn_states], len(trainer.scorer.current_hidden_states))
    print([len(v) for v in trainer.scorer.all_head_scores], [len(v) for v in trainer.scorer.all_ffn_scores])
    print(torch.cuda.memory_allocated() / MB)
    print(torch.cuda.max_memory_allocated() / MB)
    
    # Create scenarios where some blocks are totally pruned, while some are totally masked to zeros
    # 1. Prune some random blocks with 10% probability
    model.head_mask[torch.randperm(model.head_mask.numel())[:int(model.head_mask.numel() * 0.1)]] = 0
    model.intermediate_mask[torch.randperm(model.intermediate_mask.numel())[:int(model.intermediate_mask.numel() * 0.1)]] = 0
    model.hidden_mask[torch.randperm(model.hidden_mask.numel())[:int(model.hidden_mask.numel() * 0.1)]] = 0
    masked_outputs = model(**inputs, return_dict=False, output_hidden_states=True)
    
    model.prune_model_with_masks()
    pruned_outputs = model(**inputs, return_dict=False, output_hidden_states=True)
    print(torch.norm(masked_outputs[1] - pruned_outputs[1]))
    
    # 2. Prune some random blocks with 100% probability
    head_masks, intermediate_masks = model.split_mask_or_score()
    head_masks[0][3][:] = 0
    head_masks[1][7][:] = 0
    head_masks[2][11][:] = 0
    intermediate_masks[0][3][:] = 0
    intermediate_masks[1][8][:] = 0
    model.prune_model_with_masks()
    
    # 3. Mask some random blocks with 100% probability
    model.head_mask[torch.randperm(model.head_mask.numel())[:int(model.head_mask.numel() * 0.1)]] = 0
    model.intermediate_mask[torch.randperm(model.intermediate_mask.numel())[:int(model.intermediate_mask.numel() * 0.1)]] = 0
    model.hidden_mask[torch.randperm(model.hidden_mask.numel())[:int(model.hidden_mask.numel() * 0.1)]] = 0
    head_masks, intermediate_masks = model.split_mask_or_score()
    head_masks[0][6][:] = 0
    head_masks[1][3][:] = 0
    head_masks[2][10][:] = 0
    intermediate_masks[0][6][:] = 0
    intermediate_masks[1][5][:] = 0

    inputs = next(iter(dataloader))
    inputs = trainer._prepare_inputs(inputs)
    outputs = model(**inputs, return_dict=False)
    print([len(v) for v in trainer.scorer.current_mha_states], [len(v) for v in trainer.scorer.current_ffn_states], len(trainer.scorer.current_hidden_states))
    print([len(v) for v in trainer.scorer.all_head_scores], [len(v) for v in trainer.scorer.all_ffn_scores])
    outputs[0].backward()
    print([len(v) for v in trainer.scorer.current_mha_states], [len(v) for v in trainer.scorer.current_ffn_states], len(trainer.scorer.current_hidden_states))
    print([len(v) for v in trainer.scorer.all_head_scores], [len(v) for v in trainer.scorer.all_ffn_scores])
    scorer.step()
    
    for k in ['head_mask', 'intermediate_mask', 'hidden_mask']:
        retained_indices = getattr(model, k).nonzero().squeeze()
        # Update accumulated salience and uncertainty
        scorer.set_retained_indices(k, retained_indices, clean_to_zero=False)

    model.virtual_prune()
    inputs = next(iter(dataloader))
    inputs = trainer._prepare_inputs(inputs)

    trainer.scorer.current_mha_states = [[], [], []]
    trainer.scorer.current_ffn_states = [[], []]
    trainer.scorer.current_hidden_states = []
    trainer.scorer.all_head_scores = [[], [], []]
    trainer.scorer.all_ffn_scores = [[], []]
    with torch.no_grad():
        teacher_outputs = model(**inputs, return_dict=False, use_teacher=True, pass_mask=False)
    
    outputs = model(**inputs, return_dict=False)
    outputs[0].backward()
    scorer.step()

    head_scores, intermediate_scores = model.split_mask_or_score(scorer.salience_dict['head_mask']['s'], scorer.salience_dict['intermediate_mask']['s'])
    
    model.virtual_prune_restore()
    
    for i in tqdm(range(32)):
        model.zero_grad()
        inputs = next(iter(dataloader))
        inputs = trainer._prepare_inputs(inputs)
        outputs = model(**inputs, return_dict=False)
        outputs[0].backward()
        scorer.step()
    
    trainer.scorer._remove_hooks()


if __name__ == '__main__':
    main()