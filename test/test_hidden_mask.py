import os
import sys
import torch

from transformers import (HfArgumentParser)
from args import DataTrainingArguments
from models.model_args import ModelArguments
from args import MinusTrainingArguments
from utils.utils import *
from utils import build_dataloader, build_trainer
from models import build_model
from prune.pruner import AdapterPruner
from torch.utils.data import Subset
from trainer.param_control import ParamController
from prune.fisher import collect_hidden_mask_grads

if __name__ == 'main':
    sys.argv = ['test_hidden_mask.py',
            '--output_dir',
            './output/test_bert_hidden_mask/',
            '--model_name_or_path',
            'output/bert-base-uncased_lora_minus_sst2_once_global_alloc_none_self_student_mapping_dynamic_block_teacher_dynamic_student_distill_fixedteacher/mac0.4/epoch40/bz32/numprune5/paramq:0-11,v:0-11,i:0-11/lora_r8/pruning_start1/distill_epoch19/pre_pruning_model',
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
    
    model = model.to(training_args.device)


    
    # Also add ffn input layers to teacher config
    teacher_keys = ['query', 'value', 'intermediate']
    teacher_config  = {
        k: [i for i in range(model.config.num_hidden_layers)] for k in teacher_keys
    }

    pruning_batch_size = 4
    num_pruning_batches = 64
    dataloader = build_dataloader(Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]), pruning_batch_size, data_args, training_args, tokenizer)
    adapter_pruner = AdapterPruner(model, dataloader)
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        student_config=None,
        lora_with_bias=False,
        adapter_pruner=adapter_pruner,
        param_allocation_strategy=training_args.param_allocation_strategy,
    )
    param_controller.convert_to_pre_pruning_lora_teacher()
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    
    model.head_mask, model.intermediate_mask = model.head_mask.cuda(), model.intermediate_mask.cuda()
    model.hidden_mask = torch.ones(768).cuda()
    # trainer.evaluate()
    
    hidden_grads = collect_hidden_mask_grads(model, dataloader)
    print(torch.cuda.max_memory_allocated() / (1024 ** 2))
    hidden_score = hidden_grads.pow(2).sum(dim=0)
    sorted_score, sorted_idx = torch.sort(hidden_score, descending=True)
    model.hidden_mask[sorted_idx[-5:]] = 0
    # pre_prune_masked_metrics = trainer.evaluate()
    
    # Test pruning consistecy of hidden mask pruning
    model.eval()
    model = model.double()

    inputs = next(iter(dataloader))
    inputs = trainer._prepare_inputs(inputs)
    masked_outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=False,
    )
    original_outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=False,
        pass_mask=False,
    )
    
    pruned_hidden = (model.hidden_mask == 0).nonzero().squeeze()
    preserved_hidden = (model.hidden_mask == 1).nonzero().squeeze()
    model.prune_model_with_masks()
    model = model.double()
    # post_prune_masked_metrics = trainer.evaluate()
    post_prune_masked_outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=False,
    )

    # Compare the hidden states of the pruned model with the masked model
    masked_emb_out = masked_outputs[2][0][:, :, preserved_hidden]
    masked_emb_pruned = masked_outputs[2][0][:, :, pruned_hidden]
    pruned_emb_out = post_prune_masked_outputs[2][0]
    for i in range(13):
        print((masked_outputs[2][i][:, :,  preserved_hidden] - post_prune_masked_outputs[2][i]).abs().max())