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
from args import MinusTrainingArguments, Seq2SeqDataTrainingArguments
from models import build_model
from models.model_args import ModelArguments
from prune import build_scorer, BetterFisherPruner
from torch.utils.data import Subset
from utils.fisher_utils.efficiency.param import *
from glue import avg_seq_length

def main():
    parser = HfArgumentParser(
        (ModelArguments, Seq2SeqDataTrainingArguments, MinusTrainingArguments))
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
    if 't5' in model.config.model_type:
        model.head_mask = model.head_mask.view(3, model.config.num_layers, model.config.num_heads).to(model.device)
        model.intermediate_mask = model.intermediate_mask.view(2, model.config.num_layers, model.config.d_ff).to(model.device)
    else:
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
        
    if 't5' in model.config.model_type:
        pruner = BetterFisherPruner(model, ['head_mask', 'intermediate_mask'], {'head_mask': scorer, 'intermediate_mask': scorer}, 170, False, ['search', 'rearrange'], gated='gated' in getattr(model.config, 'feed_forward_proj', ''))
    else:
        pruner = BetterFisherPruner(model, ['head_mask', 'intermediate_mask'], {'head_mask': scorer, 'intermediate_mask': scorer}, 170, False, ['search', 'rearrange', 'rescale'], 'gated' in getattr(model.config, 'feed_forward_proj', ''))
    masks = pruner.generate_mask(constraint)
    # Saving the masks
    if model.head_mask is not None:
        torch.save(model.head_mask, os.path.join(training_args.output_dir, 'pruning_head_mask.pt'))
    if model.intermediate_mask is not None:
        torch.save(model.intermediate_mask, os.path.join(training_args.output_dir, 'pruning_intermediate_mask.pt'))
    # Ignore hidden_mask
    model.prune_model_with_masks()
    post_pruning_performance = trainer.evaluate(ignore_keys=['past_key_values'])
    trainer.save_model()
    json.dump(post_pruning_performance, open(os.path.join(training_args.output_dir, 'eval_results.json'), 'w'), indent=4, sort_keys=True)
    print("Post-pruning performance: ", post_pruning_performance)
    print("Max memory usage: ", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")

if __name__ == '__main__':
    main()