import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import transformers
import torch
import nltk
import numpy as np
transformers.logging.set_verbosity_error()

from transformers import (HfArgumentParser, DataCollatorForSeq2Seq)
from datasets import load_metric
from models.model_args import ModelArguments
from utils.utils import *
from trainer.trainer_minus import MinusTrainer
from args import MinusTrainingArguments, Seq2SeqDataTrainingArguments
from loralib.layers import LoRALayer
from models import build_model
from loralib.layers import PruningLinear, DistillLinear
from trainer.param_control import ParamController
from torch.utils.data import Subset
from utils import build_dataloader
from prune import build_pruner, build_scorer

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
    sys.argv = ['test_t5_running.py',
            '--output_dir',
            './output/test_t5_grafting/',
            '--model_name_or_path',
            'output/t5-base_lora_xsum/epoch5/bz16/lora_r8/lora_alpha16/parameq:0-11,ev:0-11,dq:0-11,dv:0-11,cq:0-11,cv:0-11/best_model',
            '--task_name',
            'xsum',
            '--do_train',
            '--do_eval',
            '--max_input_length',
            '512',
            '--max_target_length',
            '128',
            '--per_device_train_batch_size',
            '16',
            '--per_device_eval_batch_size',
            '16',
            '--learning_rate',
            '1e-3',
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
            '--report_to',
            'none',
            ]
    parser = HfArgumentParser(
        (ModelArguments, Seq2SeqDataTrainingArguments, MinusTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    config, tokenizer, model = build_model(model_args, data_args, training_args)
    train_dataset, eval_dataset, _, datasets = build_seq2seq_data(data_args, training_args, tokenizer)
    model = model.to(training_args.device)
    print(model.config)

    teacher_keys = ['enc_self_query', 'enc_self_value', 'dec_self_query', 'dec_self_value', 'cross_query', 'cross_value']
    teacher_config = {
        k: [i for i in range(config.num_layers)] for k in teacher_keys
    }
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        lora_with_bias=False,
    )

    model.reset_masks()
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args.disable_tqdm = False
    training_args.pruner_type = 'running_fisher'
    trainer = MinusTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        param_controller=param_controller,
        teacher_model=None,
        seq_len=data_args.max_input_length,
        output_seq_len=data_args.max_target_length,
        cls_task=False,
    )
    training_args.eval_accumulation_steps = None
    # training_args.predict_with_generate = True
    # trainer.evaluate()
    
    model.head_mask = model.head_mask.view(-1)
    model.intermediate_mask = model.intermediate_mask.view(-1)
    
    model.head_mask[torch.randperm(model.head_mask.numel())[:int(model.head_mask.numel() * 0.1)]] = 0
    model.intermediate_mask[torch.randperm(model.intermediate_mask.numel())[:int(model.intermediate_mask.numel() * 0.1)]] = 0
    model.hidden_mask[torch.randperm(model.hidden_mask.numel())[:2]] = 0
    
    history_head_mask, history_intermediate_mask, history_hidden_mask = model.head_mask.clone(), model.intermediate_mask.clone(), model.hidden_mask.clone()
    
    history_head_mask = torch.split(history_head_mask, [sum(model.enc_selfattn_headnum), sum(model.dec_selfattn_headnum), sum(model.dec_crossattn_headnum)])
    history_head_mask = (history_head_mask[0].split(model.enc_selfattn_headnum), history_head_mask[1].split(model.dec_selfattn_headnum), history_head_mask[2].split(model.dec_crossattn_headnum))
    history_intermediate_mask = torch.split(history_intermediate_mask, [sum(model.enc_neuron_nums), sum(model.dec_neuron_nums)])
    
    
    dataloader = trainer.get_train_dataloader()
    inputs = next(iter(dataloader))
    inputs = trainer._prepare_inputs(inputs)
    
    model = model.double()
    
    
    model = model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
    print("Peak memory usage: %f MB" % (torch.cuda.max_memory_allocated() / 1024 / 1024))
    
    model.prune_model_with_masks()
    
    # Check pruning shape consistency
    
    for i in range(model.config.num_layers):
        if model.encoder.block[i].layer[0].SelfAttention.n_heads == history_head_mask[0][i].shape[0] - (history_head_mask[0][i] == 0).sum().item() and (model.enc_selfattn_headnum[i] == model.encoder.block[i].layer[0].SelfAttention.n_heads):
            print("Layer %d encoder self-attention check complete" % i)
        else:
            raise ValueError("Layer %d encoder self-attention check failed" % i)
        if model.decoder.block[i].layer[0].SelfAttention.n_heads == history_head_mask[1][i].shape[0] - (history_head_mask[1][i] == 0).sum().item() and (model.dec_selfattn_headnum[i] == model.decoder.block[i].layer[0].SelfAttention.n_heads):
            print("Layer %d decoder self-attention check complete" % i)
        else:
            raise ValueError("Layer %d decoder self-attention check failed" % i)
        if model.decoder.block[i].layer[1].EncDecAttention.n_heads == history_head_mask[2][i].shape[0] - (history_head_mask[2][i] == 0).sum().item() and (model.dec_crossattn_headnum[i] == model.decoder.block[i].layer[1].EncDecAttention.n_heads):
            print("Layer %d decoder self-attention check complete" % i)
        else:
            raise ValueError("Layer %d decoder self-attention check failed" % i)
        
        
        
        
    with torch.no_grad():
        pruned_outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
    
    for i in range(13):
        print((outputs[3][i][:, :, 20:] - pruned_outputs[3][i]).abs().mean())

    for i in range(13):
        print((outputs[7][i][:, :, 20:] - pruned_outputs[7][i]).abs().mean())