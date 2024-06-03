import seaborn as sns
sns.set_theme(style="darkgrid")
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
import json

from transformers import (HfArgumentParser)
from utils.utils import *
from utils import build_trainer, build_dataloader
from args import MinusTrainingArguments, InstructionDataTrainingArguments
from models import build_model
from models.model_args import ModelArguments
from prune import build_scorer, BetterFisherPruner
from torch.utils.data import Subset
from utils.fisher_utils.efficiency.param import *

def main():
    sys.argv = ['test_bert.py',
            '--output_dir',
            'output/meta-llama/Llama-2-7b-hf/alpaca_gpt4/bz4/lora/teacher_dq:0-31,dv:0-31/epoch5/lora_r8/lora_alpha16/lr2e-4/seed42/checkpoint-8000/pruned/constraint_0.8/batches_64',
            '--model_name_or_path',
            'output/meta-llama/Llama-2-7b-hf/alpaca_gpt4/bz4/lora/teacher_dq:0-31,dv:0-31/epoch5/lora_r8/lora_alpha16/lr2e-4/seed42/checkpoint-8000',
            '--do_train',
            '--do_eval',
            '--task_name',
            'alpaca',
            '--data_path', 
            'data/sft/alpaca_data_gpt4.json',
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
            '--apply_lora',
            '--do_distill',
            '--lora_r',
            '8',
            '--lora_alpha',
            '16',
            '--report_to',
            'none',
            '--tf32',
            'True',
            '--pruning_batch_size',
            '4',
            '--pruning_batches',
            '64',
            '--mac_constraint',
            '0.8',
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

    config, tokenizer, model = build_model(model_args, data_args, training_args)
    model_generative = True
    train_dataset, eval_dataset, _, _ = build_seq2seq_data(data_args, training_args, tokenizer)

    pruning_batch_size = training_args.pruning_batch_size
    num_pruning_batches = training_args.pruning_batches
    dataloader = build_dataloader(Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]), pruning_batch_size, data_args, training_args, tokenizer, model=model)

    training_args.predict_with_generate=True
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=None)
    print("Trainer type: ", type(trainer))

    # Convert model's mask to post-training pruning mask

    model.head_mask = model.head_mask.view(model.config.num_hidden_layers, model.config.num_attention_heads).to(model.device)
    model.intermediate_mask = model.intermediate_mask.view(model.config.num_hidden_layers, model.config.intermediate_size).to(model.device)
    model.hidden_mask = None # post-training pruning cannot prune hidden dimensions
    
    print("Unpruned model's performance: ", trainer.evaluate())
    scorer = build_scorer('gradient_l2', model, dataloader)
    constraint = training_args.mac_constraint
    model.head_mask = torch.ones_like(model.head_mask)
    model.intermediate_mask = torch.ones_like(model.intermediate_mask)
    for p in model.parameters():
        p.requires_grad = False # disable gradient computation when generating masks
    
    pre_pruning_param_num = sum(p.numel() for n, p in model.named_parameters() if 'embed' not in n and 'transform' not in n and 'lm_head' not in n)
    pruner = BetterFisherPruner(model, ['head_mask', 'intermediate_mask'], {'head_mask': scorer, 'intermediate_mask': scorer}, 221, False, ['search', 'rearrange'], gated=True)

    masks = pruner.generate_mask(constraint)
    # {'eval_loss': 1.0787718296051025, 'eval_runtime': 65.5152, 'eval_samples_per_second': 7.952, 'eval_steps_per_second': 2.0}
    model.prune_model_with_masks()
    post_pruning_param_num = sum(p.numel() for n, p in model.named_parameters() if 'embed' not in n and 'transform' not in n and 'lm_head' not in n)
    post_pruning_performance = trainer.evaluate(ignore_keys=['past_key_values'])
    trainer.save_model()
    print("Pre-pruning param num %d, post-pruning param num %d. Remaining density %.4f" % (pre_pruning_param_num, post_pruning_param_num, post_pruning_param_num / pre_pruning_param_num))
    json.dump(post_pruning_performance, open(os.path.join(training_args.output_dir, 'eval_results.json'), 'w'), indent=4, sort_keys=True)
    print("Post-pruning performance: ", post_pruning_performance)
    print("Max memory usage: ", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")
    
if __name__ == '__main__':
    main()