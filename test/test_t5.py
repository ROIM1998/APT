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
from utils.minus_utils import efficiency_testing, compare_module_inputs_equality, collect_module_inputs
from tqdm import tqdm
from torch.utils.data import Subset
from prune import build_pruner, build_scorer, build_pruning_scheduler, BetterFisherPruner
from prune.search import search_encoder_decoder_mac

def collect_lora_info(model):
    lora_vars = [n for n, p in model.named_parameters() if 'lora' in n]
    lora_param_num = sum([p.numel() for n, p in model.named_parameters() if 'lora' in n])
    lora_layers = [n for n, p in model.named_modules() if isinstance(p, LoRALayer)]
    prune_layers = [n for n, p in model.named_modules() if isinstance(p, PruningLinear)]
    distill_layers = [n for n, p in model.named_modules() if isinstance(p, DistillLinear)]
    return {
        'lora_vars': lora_vars,
        'lora_param_num': lora_param_num,
        'lora_layers': lora_layers,
        'prune_layers': prune_layers,
        'distill_layers': distill_layers,
    }
    

def main():
    sys.argv = ['test_t5.py',
            '--output_dir',
            './output/test_t5_grafting/',
            '--model_name_or_path',
            'output/test_model_t5',
            '--task_name',
            'mnli',
            '--do_train',
            '--do_eval',
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
            'search',
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

    pruning_batch_size, num_pruning_batches = 4, 64
    # print(trainer.evaluate())
    dataloader = build_dataloader(Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]), training_args.per_device_train_batch_size, data_args, training_args, tokenizer)
    model.to(training_args.device)

    teacher_keys = ['enc_self_query', 'enc_self_value', 'dec_self_query', 'dec_self_value', 'cross_query', 'cross_value']
    teacher_config = {
        k: [i for i in range(config.num_layers)] for k in teacher_keys
    }
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        lora_with_bias=False,
    )
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    model.head_mask, model.intermediate_mask = None, None
    # trainer.evaluate()

    # Test model fisher scoring & pruning functions
    mac_constraint = 0.4
    training_args.seq_len = 128
    training_args.cls_task = True
    model.head_mask, model.intermediate_mask = torch.ones(3, 12, 12).to(model.device), torch.ones(2, 12, 3072).to(model.device)
    for p in model.parameters():
        p.requires_grad = False
    scorer = build_scorer('gradient_l2', model, dataloader)
    pruner = BetterFisherPruner(model, ['head_mask', 'intermediate_mask'], {'head_mask': scorer, 'intermediate_mask': scorer}, training_args.seq_len, training_args.cls_task, ['search'])
    
    masks = pruner.generate_mask(mac_constraint)
    searched_head_mask, searched_intermediate_mask = model.head_mask.clone(), model.intermediate_mask.clone()
    
    global_pruner = BetterFisherPruner(model, ['head_mask', 'intermediate_mask'], {'head_mask': scorer, 'intermediate_mask': scorer}, training_args.seq_len, training_args.cls_task, ['search', 'better_rearrange', 'global'])
    global_masks = global_pruner.generate_mask(mac_constraint)
    print("Global head mask changed: ", (global_masks['head_mask'] != searched_head_mask).sum())
    print("Global intermediate mask changed: ", (global_masks['intermediate_mask'] != searched_intermediate_mask).sum())
    
    pruning_scheduler = build_pruning_scheduler(training_args, model, masks.get('head_mask', None), masks.get('intermediate_mask', None), scorer.head_grads, scorer.intermediate_grads)
    pruning_scheduler.gen_schedule()

    # Using simple mask based on the score within each partition (encoder-self-heads, encoder-intermediate, decoder-self-heads, decoder-cross-heads, decoder-intermediate)
    head_scores, intermediate_scores = scorer.head_score(), scorer.intermediate_score()
    head_mask, intermediate_mask = torch.zeros_like(head_scores), torch.zeros_like(intermediate_scores)
    encoder_self_head_scores, decoder_self_head_scores, decoder_cross_head_scores = head_scores
    encoder_intermediate_scores, decoder_intermediate_scores = intermediate_scores
    _, encoder_self_head_indices = torch.sort(encoder_self_head_scores.view(-1), descending=True)
    _, decoder_self_head_indices = torch.sort(decoder_self_head_scores.view(-1), descending=True)
    _, decoder_cross_head_indices = torch.sort(decoder_cross_head_scores.view(-1), descending=True)
    _, encoder_intermediate_indices = torch.sort(encoder_intermediate_scores.view(-1), descending=True)
    _, decoder_intermediate_indices = torch.sort(decoder_intermediate_scores.view(-1), descending=True)
    head_mask[0].view(-1)[encoder_self_head_indices[:int(encoder_self_head_indices.numel() * mac_constraint)]] = 1
    head_mask[1].view(-1)[decoder_self_head_indices[:int(decoder_self_head_indices.numel() * mac_constraint)]] = 1
    head_mask[2].view(-1)[decoder_cross_head_indices[:int(decoder_cross_head_indices.numel() * mac_constraint)]] = 1
    intermediate_mask[0].view(-1)[encoder_intermediate_indices[:int(encoder_intermediate_indices.numel() * mac_constraint)]] = 1
    intermediate_mask[1].view(-1)[decoder_intermediate_indices[:int(decoder_intermediate_indices.numel() * mac_constraint)]] = 1
    model.head_mask, model.intermediate_mask = head_mask, intermediate_mask
    pre_prune_results = trainer.evaluate()
    model.head_mask, model.intermediate_mask = search_encoder_decoder_mac(model.config, head_scores, intermediate_scores, 128, 2, mac_constraint)
    searched_results = trainer.evaluate()
    
    model.prune_model_with_masks()
    post_prune_results = trainer.evaluate()

    inputs = next(iter(dataloader))
    inputs = {k: v.to(training_args.device) for k, v in inputs.items()}
    model.head_mask, model.intermediate_mask = model.head_mask.to(training_args.device), model.intermediate_mask.to(training_args.device)
    unshape_head_mask = model.head_mask.view(-1)
    unshape_intermediate_mask = model.intermediate_mask.view(-1)
    unshape_head_mask[torch.randperm(unshape_head_mask.numel())[:int(unshape_head_mask.numel() * 0.5)]] = 0
    unshape_intermediate_mask[torch.randperm(unshape_intermediate_mask.numel())[:int(unshape_intermediate_mask.numel() * 0.5)]] = 0
    model.head_mask = unshape_head_mask.view(model.head_mask.shape)
    model.intermediate_mask = unshape_intermediate_mask.view(model.intermediate_mask.shape)

    # Create original non-pruning model
    config, tokenizer, original_model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    original_model.head_mask, original_model.intermediate_mask = model.head_mask.to(training_args.device), model.intermediate_mask.to(training_args.device)
    original_model.eval()
    model.eval()
    original_model = original_model.to(training_args.device)

    with torch.no_grad():
        original_outputs = original_model(**inputs, output_attentions=True, output_hidden_states=True)

    # Set up LoRA training
    param_controller.convert_to_pre_pruning_lora_teacher()
    
    converted_lora_info = collect_lora_info(model)
    print("LoRA param num: ", converted_lora_info['lora_param_num'])
    print("LoRA var num: ", len(converted_lora_info['lora_vars']))
    print("LoRA layer num", len(converted_lora_info['lora_layers']))
    print("LoRA Pruning layer num", len(converted_lora_info['prune_layers']))
    print("DistillLinear layer num", len(converted_lora_info['distill_layers']))
    param_controller.tune_lora_only()
    print("Tuning paramteres", sum(p.numel() for p in model.parameters() if p.requires_grad))
    param_controller.model_as_teacher()
    print("Teacher tuning parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    with torch.no_grad():
        converted_outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
    
    original_params, converted_params = dict(original_model.named_parameters()), dict(model.named_parameters())
    mismatch_params = [k for k in converted_params if k not in original_params or not torch.allclose(original_params[k], converted_params[k])]
    print("Mismatch params", mismatch_params)
    
    model.prune_model_with_masks()
    with torch.no_grad():
        pruned_outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
    results = compare_module_inputs_equality([model, original_model], inputs, lambda x: x.encoder.block[0].layer[0].SelfAttention.o)
    index_select = original_model.head_mask[0][0].repeat_interleave(768//12).nonzero().squeeze()
    print("Diff:", (results[0] - results[1][:,:,index_select]).abs().max())
    
    results = compare_module_inputs_equality([model, original_model], inputs, lambda x: x.encoder.block[0].layer[1].DenseReluDense.wo)
    index_select = original_model.intermediate_mask[0][0].nonzero().squeeze()
    print("Diff:", (results[0] - results[1][:,:,index_select]).abs().max())
    
    results = compare_module_inputs_equality([model, original_model], inputs, lambda x: x.encoder.block[1].layer[0].SelfAttention.o)
    index_select = original_model.head_mask[0][1].repeat_interleave(768//12).nonzero().squeeze()
    print("Diff:", (results[0] - results[1][:, :, index_select]).abs().max())
    original_inputs = collect_module_inputs(original_model, inputs, lambda x: x.encoder.block[1].layer[0].SelfAttention)[0]
    converted_inputs = collect_module_inputs(model, inputs, lambda x: x.encoder.block[1].layer[0].SelfAttention)[0]

    for i in range(12):
        print("Encoder diff layer %d" % i, (original_outputs.encoder_hidden_states[i+1] - pruned_outputs.encoder_hidden_states[i+1]).abs().max())

    all_outputs = []
    for inputs in tqdm(dataloader):
        if 'labels' in inputs:
            del inputs['labels']
        inputs = {k: v.to(training_args.device) for k, v in inputs.items()}
        all_outputs.append(model.generate(**inputs))
    

    # loaded_metrics = trainer.evaluate()
    efficiency_results = efficiency_testing(model, tokenizer, training_args.device)
    print(efficiency_results)