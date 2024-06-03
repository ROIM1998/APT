# Test ElasticLlama model with pruning consistency and tuning consistency
import os
import sys
import torch

from utils.minus_utils import compare_module_inputs_equality
from transformers import (HfArgumentParser)
from args import InstructionDataTrainingArguments
from models.model_args import ModelArguments
from args import MinusTrainingArguments
from utils.utils import *
from utils import build_dataloader, build_trainer
from models import build_model
from prune.pruner import AdapterPruner
from torch.utils.data import Subset
from trainer.param_control import ParamController
from copy import deepcopy

def main():
    sys.argv = ['test_bert.py',
            '--output_dir',
            './output/test_llama_elastictuning/',
            '--model_name_or_path',
            'meta-llama/Llama-2-7b-hf',
            '--do_train',
            '--task_name',
            'alpaca',
            '--data_path', 
            'data/sft/alpaca_data.json',
            '--bf16',
            'True',
            '--output_dir',
            'output/llama_lora_alpaca/epoch_30',
            '--num_train_epochs',
            '30',
            '--per_device_train_batch_size',
            '4',
            '--per_device_eval_batch_size',
            '4',
            '--gradient_accumulation_steps',
            '8',
            '--evaluation_strategy',
            "no",
            '--save_strategy',
            "steps",
            '--save_steps',
            '2000',
            '--save_total_limit',
            '1',
            '--learning_rate',
            '2e-4', # LoRA learning rate
            '--weight_decay',
            '0.',
            '--warmup_ratio',
            '0.03',
            '--lr_scheduler_type',
            "cosine",
            '--logging_steps',
            '1',
            '--apply_lora',
            '--do_distill',
            '--lora_r',
            '8',
            '--lora_alpha',
            '16',
            '--report_to',
            'none',
            # '--fsdp',
            # "full_shard auto_wrap",
            # '--fsdp_transformer_layer_cls_to_wrap',
            # 'LlamaDecoderLayer',
            '--tf32',
            'True',
            ]
    parser = HfArgumentParser(
        (ModelArguments, InstructionDataTrainingArguments, MinusTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)
    # training_args.disable_tqdm = False
    config, tokenizer, model = build_model(model_args, data_args, training_args, token=os.environ.get('HF_TOKEN', None))
    train_dataset, _, _, _ = build_seq2seq_data(data_args, training_args, tokenizer)
    dataloader = build_dataloader(train_dataset, training_args.per_device_train_batch_size, data_args, training_args, tokenizer)
    
    inputs = next(iter(dataloader))
    model = model.to(training_args.device)
    inputs = {k: v.to(training_args.device) for k, v in inputs.items()}
    
    model.head_mask = model.head_mask.to(training_args.device)
    model.intermediate_mask = model.intermediate_mask.to(training_args.device)
    model.hidden_mask = model.hidden_mask.to(training_args.device)
    
    model.eval()
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Randomly masking 10% of the heads, intermediate layers, and hidden layers
    model.head_mask[torch.randperm(model.head_mask.shape[0])[:int(model.head_mask.shape[0] * 0.1)]] = 0
    model.intermediate_mask[torch.randperm(model.intermediate_mask.shape[0])[:int(model.intermediate_mask.shape[0] * 0.1)]] = 0
    model.hidden_mask[torch.randperm(model.hidden_mask.shape[0])[:int(model.hidden_mask.shape[0] * 0.1)]] = 0
    retained_hidden_indices = model.hidden_mask.nonzero().squeeze()
    head_mask, intermediate_mask = model.split_mask_or_score()
    head_size = model.config.hidden_size // model.config.num_attention_heads
    head_mask = [v.repeat_interleave(head_size) for v in head_mask]
    
    with torch.no_grad():
        masked_outputs = model(**inputs, output_hidden_states=True)
    
    retained_masked_hidden_states = [
        hs.index_select(-1, retained_hidden_indices)
        for hs in masked_outputs[-1]
    ]
    pruned_model = deepcopy(model)
    pruned_model.prune_model_with_masks()
    
    with torch.no_grad():
        pruned_outputs = pruned_model(**inputs, output_hidden_states=True)
    
    for i in range(32):
        print((retained_masked_hidden_states[i] - pruned_outputs[-1][i]).abs().mean())
        
    collected_inputs = []
    handlers = []
    module_func = lambda x: x.model.layers[0]
    handlers.append(module_func(model).register_forward_hook(lambda self, input, output: collected_inputs.append(output)))
    handlers.append(module_func(pruned_model).register_forward_hook(lambda self, input, output: collected_inputs.append(output)))
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        pruned_outputs = pruned_model(**inputs, output_hidden_states=True)
        while len(handlers) > 0:
            handlers.pop().remove()            
        # print((collected_inputs[0][0].index_select(-1, head_mask[0].nonzero().squeeze()) - collected_inputs[1][0]).abs().mean())
        # print((collected_inputs[0][0].index_select(-1, intermediate_mask[0].nonzero().squeeze()) - collected_inputs[1][0]).abs().mean())
        # TODO: figure this out: the difference of out hidden-states are way bigger than collected outputs
        print((collected_inputs[0][0].index_select(-1, retained_hidden_indices) - collected_inputs[1][0]).abs().mean())
    
    print((collected_inputs[1][0] - pruned_outputs[-1][1]).abs().mean())
    print((collected_inputs[0][0] - masked_outputs[-1][1]).abs().mean())
    
    print((collected_inputs[1][0] - retained_masked_hidden_states[1]).abs().mean())
    print((collected_inputs[0][0].index_select(-1, retained_hidden_indices) - retained_masked_hidden_states[1]).abs().mean())
    print((collected_inputs[0][0].index_select(-1, retained_hidden_indices) - pruned_outputs[-1][1]).abs().mean())
    print((collected_inputs[1][0] - pruned_outputs[-1][1]).abs().mean())
    print((collected_inputs[0][0].index_select(-1, retained_hidden_indices) - collected_inputs[1][0]).abs().mean())
    
    
    pruned_hidden_indices = (model.hidden_mask == 0).nonzero().squeeze()
    print(masked_outputs[-1][1].index_select(-1, pruned_hidden_indices).any())
    
    pruned_indices = (head_mask[0] == 0).nonzero().squeeze()
    print(collected_inputs[0][0].index_select(-1, pruned_indices))
    
    
    pruned_hidden = (model.hidden_mask == 0).nonzero().squeeze()
    
if __name__ == '__main__':
    main()