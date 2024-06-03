import os
import sys
import time
import torch

from transformers import (HfArgumentParser)
from args import DataTrainingArguments
from models.model_args import ModelArguments
from args import MinusTrainingArguments
from utils.utils import *
from utils import build_dataloader, build_trainer
from utils.minus_utils import input_constructor
from models import build_model
from tqdm import tqdm

def bench_latency(model, inputs, warm_steps=3, num_reps=10, apply_virtual_prune=False):
    model.train()
    timings = []
    start_mems = []
    end_mems = []
    diff_mems = []
    MB = 1024.0 * 1024.0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    for _ in range(warm_steps):
        model(**inputs, return_dict=True)
    
    for _ in range(num_reps):
        start_mem = torch.cuda.max_memory_allocated() / MB
        torch.cuda.synchronize()
        start = time.perf_counter()
        if apply_virtual_prune:
            model.virtual_prune()
        outputs = model(**inputs, return_dict=True)
        loss = outputs.loss
        loss.backward()
        model.zero_grad()
        if apply_virtual_prune:
            model.virtual_prune_restore()
        torch.cuda.synchronize()
        end = time.perf_counter()
        inference_time = end - start
        end_mem = torch.cuda.max_memory_allocated() / MB
        diff_mem = end_mem - start_mem
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        timings.append(inference_time)
        start_mems.append(start_mem)
        end_mems.append(end_mem)
        diff_mems.append(diff_mem)

    timings = torch.as_tensor(timings, dtype=torch.float32)
    start_mems = torch.as_tensor(start_mems, dtype=torch.float32)
    end_mems = torch.as_tensor(end_mems, dtype=torch.float32)
    diff_mems = torch.as_tensor(diff_mems, dtype=torch.float32)
    t_mean = timings.mean().item()
    t_std = timings.std().item()
    sm_mean = start_mems.mean().item()
    em_mean = end_mems.mean().item()
    dm_mean = diff_mems.mean().item()
    sm_std = start_mems.std().item()
    em_std = end_mems.std().item()
    dm_std = diff_mems.std().item()
    result = {
        't_mean': t_mean,
        't_std': t_std,
        'sm_mean': sm_mean,
        'sm_std': sm_std,
        'em_mean': em_mean,
        'em_std': em_std,
        'dm_mean': dm_mean,
        'dm_std': dm_std,
    }
    return result



if __name__ == 'main':
    sys.argv = ['test_mask_efficiency.py',
            '--output_dir',
            './output/test_mask_efficiency/',
            '--model_name_or_path',
            'roberta-large',
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
    
    
    print("LoRA param num:", sum(p.numel() for n, p in model.named_parameters() if 'lora' in n))
    print("LoRA param names:", [n for n, p in model.named_parameters() if 'lora' in n])
    
    model = model.cuda()
    model.head_mask, model.intermediate_mask = model.head_mask.cuda(), model.intermediate_mask.cuda()
    
    dataloader = build_dataloader(train_dataset, training_args.per_device_train_batch_size, data_args, training_args, tokenizer)
    inputs = next(iter(dataloader))
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    for n, p in model.named_parameters():
        p.requires_grad = True
    
    # Full mask to one bench
    mask_one_result = bench_latency(model, inputs, warm_steps=3, num_reps=10)
    
    # Full mask to zero bench
    model.head_mask, model.intermediate_mask = model.head_mask * 0, model.intermediate_mask * 0
    mask_zero_result = bench_latency(model, inputs, warm_steps=3, num_reps=10)
    
    # Partial mask bench
    model.head_mask = torch.load('output/roberta-large_lora_minus_sst2_once_global_free_inout_nodistill/mac0.4/epoch10/bz32/numprune5/paramq:0-23,v:0-23,i:0-23/lora_r8/lora_alpha16/final_head_mask.pt')
    model.intermediate_mask = torch.load('output/roberta-large_lora_minus_sst2_once_global_free_inout_nodistill/mac0.4/epoch10/bz32/numprune5/paramq:0-23,v:0-23,i:0-23/lora_r8/lora_alpha16/final_intermediate_mask.pt')
    mask_partial_result = bench_latency(model, inputs, warm_steps=3, num_reps=10)
    
    # Partial mask bench  w/ virtual prune
    model.head_mask = torch.load('output/roberta-large_lora_minus_sst2_once_global_free_inout_nodistill/mac0.4/epoch10/bz32/numprune5/paramq:0-23,v:0-23,i:0-23/lora_r8/lora_alpha16/final_head_mask.pt')
    model.intermediate_mask = torch.load('output/roberta-large_lora_minus_sst2_once_global_free_inout_nodistill/mac0.4/epoch10/bz32/numprune5/paramq:0-23,v:0-23,i:0-23/lora_r8/lora_alpha16/final_intermediate_mask.pt')
    for p in model.parameters():
        p.grad = None
    mask_virtprune_partial_result = bench_latency(model, inputs, warm_steps=3, num_reps=10, apply_virtual_prune=True)
    
    # Mask to none bench
    model.head_mask, model.intermediate_mask = None, None
    mask_none_result = bench_latency(model, inputs, warm_steps=3, num_reps=10)