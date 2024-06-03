import os
import sys
import time
import torch
import loralib as lora

from transformers import (HfArgumentParser)
from args import DataTrainingArguments
from models.model_args import ModelArguments
from args import MinusTrainingArguments
from utils.utils import *
from utils import build_dataloader, build_trainer
from models import build_model
from tqdm import tqdm
from trainer.param_control import ParamController


if __name__ == '__main__':
    sys.argv = ['test_mask_efficiency.py',
            '--output_dir',
            './output/test_mask_efficiency/',
            '--model_name_or_path',
            'output/roberta-base/sst2/bz32/elastictuning_virtualprune_revisedscorer/mac0.4/epoch40/distill_epoch20/numprune10/sparsity_warmup1/pruning_start-1/pruning_stop3/lora_r8/lora_alpha/warmup_paramq:0-11,v:0-11,i:0-11/teacher_paramq:0-11,v:0-11,i:0-11/best_distilled_model',
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
            '--report_to',
            'none',
            '--learning_rate',
            '5e-5',
            '--warmup_ratio',
            '0.06',
            '--weight_decay',
            '0.1',
            '--apply_lora',
            '--lora_r',
            '8',
            '--lora_alpha',
            '16',
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
    
    model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt')).to(training_args.device)
    model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt')).to(training_args.device)
    model.hidden_mask = torch.load(os.path.join(model_args.model_name_or_path, 'hidden_mask.pt')).to(training_args.device)
    model = model.to(training_args.device)
    
    dataloader = build_dataloader(train_dataset, training_args.per_device_train_batch_size, data_args, training_args, tokenizer)
    inputs = next(iter(dataloader))
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    for m in model.modules():
        if isinstance(m, lora.Linear):
            m.scaling = 2
            if isinstance(m, lora.DistillLinear):
                m.teacher_scaling = 2
    
    with torch.no_grad():
        outputs = model(**inputs, return_dict=False)
        

    # Also add ffn input layers to teacher config
    teacher_keys = ['query', 'value', 'intermediate']
    teacher_config  = {
        k: [i for i in range(model.config.num_hidden_layers)] for k in teacher_keys
    }
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        student_config=None,
        lora_with_bias=False,
        param_allocation_strategy=training_args.param_allocation_strategy,
    )
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    trainer.distilling = True
    param_controller.model_as_teacher()
        
    inputs = next(iter(dataloader))
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
    pre_virtual_prune_metrics = trainer.evaluate()
    print(pre_virtual_prune_metrics)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    outputs = model(**inputs)
    outputs[0].backward()
    model.zero_grad()
    print(torch.cuda.max_memory_allocated() / 1024 / 1024)
    
    
    model.virtual_prune()
    
    virtual_pruned_outputs = model(**inputs, return_dict=False, output_hidden_states=True, use_teacher=True, pass_mask=False)
    post_virtual_prune_metrics = trainer.evaluate()
    print(post_virtual_prune_metrics)
    
    model.virtual_prune_restore()
    original_outputs = model(**inputs, return_dict=False, output_hidden_states=True, use_teacher=True, pass_mask=False)
    restore_metrics = trainer.evaluate()
    print(restore_metrics)
    
    for original_hs, virtual_pruned_hs in zip(original_outputs[2], virtual_pruned_outputs[2]):
        print(torch.norm(original_hs - virtual_pruned_hs))
    
    trainer.create_optimizer_and_scheduler(num_training_steps=1000)
    optimizer = trainer.optimizer
    scheduler = trainer.lr_scheduler
    
    torch.cuda.reset_peak_memory_stats()
    i = 0
    for inputs in tqdm(dataloader, desc="Iteration"):
        i += 1
        inputs = trainer._prepare_inputs(inputs)
        with torch.no_grad():
            teacher_outputs = model(**inputs, return_dict=False, output_hidden_states=True, use_teacher=True, pass_mask=False)
        student_outputs = model(**inputs, return_dict=False, output_hidden_states=True)
        distill_loss, distill_ce_loss, loss = trainer.calculate_distillation_loss(
            teacher_outputs,
            student_outputs,
        )        
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i == 10:
            break
    print(torch.cuda.max_memory_allocated() / 1024 / 1024)