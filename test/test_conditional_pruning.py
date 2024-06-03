import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
from models import build_model
from transformers import HfArgumentParser
from args import DataTrainingArguments
from models.model_args import ModelArguments
from trainer.param_control import ParamController
from args import MinusTrainingArguments
from utils.utils import *
from utils import build_trainer, build_dataloader
from args import MinusTrainingArguments
from models import build_model
from prune import build_scorer
from prune.pruner import AdapterPruner, BetterFisherPruner
from prune.scheduler import SaliencyPruningScheduler
from prune.search import search_mac, search_mac_reverse
from torch.utils.data import Subset
from trainer.param_control import ParamController

def main():
    sys.argv = ['neuron_importance.py',
            '--output_dir',
            './output/neuron_importance/',
            '--model_name_or_path',
            'output/debug_output/pruning_1_model',
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
            '--apply_lora',
            '--do_distill',
            '--lora_r',
            '64',
            '--report_to',
            'none'
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

    pruning_batch_size = 4
    num_pruning_batches = 64
    dataloader = build_dataloader(Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:pruning_batch_size * num_pruning_batches]), pruning_batch_size, data_args, training_args, tokenizer)

    model_args.model_name_or_path = 'output/debug_output/pre_pruning_model'
    config, tokenizer, original_model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    training_args.seq_len = 128
    training_args.cls_task = True
    for p in model.parameters():
            p.requires_grad = False
    scorer = build_scorer('gradient_l2', original_model, dataloader)
    pruner = BetterFisherPruner(original_model, ['head_mask', 'intermediate_mask'], {'head_mask': scorer, 'intermediate_mask': scorer}, training_args.seq_len, training_args.cls_task, ['search', 'better_rearrange', 'global'])
    head_mask, intermediate_mask = search_mac(model.config, scorer.head_score(), scorer.intermediate_score(), 128, 0.4)
    head_mask_reversed, intermediate_mask_reversed = search_mac_reverse(model.config, scorer.head_score(), scorer.intermediate_score(), 128, 0.4)
    print("Head mask match: ", torch.all(head_mask == head_mask_reversed))
    print("Intermediate mask match: ", torch.all(intermediate_mask == intermediate_mask_reversed))
    head_mask_conditioned, intermediate_mask_conditioned = search_mac_reverse(model.config, scorer.head_score(), scorer.intermediate_score(), 128, 0.6, head_mask, intermediate_mask)
    print("Head mask follows condition: ", torch.all((head_mask_conditioned - head_mask) >= 0))
    print("Intermediate mask follows condition: ", torch.all((intermediate_mask_conditioned - intermediate_mask) >= 0))
    
    masks = pruner.generate_mask(0.4)
    head_mask, intermediate_mask = masks['head_mask'], masks['intermediate_mask']
    original_model.head_mask, original_model.intermediate_mask = head_mask, intermediate_mask
    original_model.prune_model_with_masks()

    teacher_keys = ['query', 'value']
    teacher_config  = {
        k: [i for i in range(model.config.num_hidden_layers)] for k in teacher_keys
    }
    student_keys = ['query', 'value', 'intermediate']
    student_config = {
        k: [i for i in range(model.config.num_hidden_layers)] for k in student_keys
    }
    adapter_pruner = AdapterPruner(model, dataloader)
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        student_config=student_config,
        lora_with_bias=False,
        adapter_pruner=adapter_pruner,
    )

    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    model.clear_masks()
    trainer.evaluate()
    
    pruning_steps, pruning_schedule = param_controller.generate_pruning_schedule(100, 10000, 0.4, 5)
    pruning_schedule = pruning_schedule[1:]
    scheduler = SaliencyPruningScheduler(model, head_mask, intermediate_mask, None, None, dataloader, pruning_schedule)
    scheduler.gen_schedule()
    for i in range(5):
        masks = scheduler.gen_next_mask()
        model.head_mask, model.intermediate_mask = masks['head_mask'], masks['intermediate_mask']
        model.prune_model_with_masks()

    for i in range(12):
        print("Layer: ", i, model.roberta.encoder.layer[i].intermediate.dense.weight.shape == original_model.roberta.encoder.layer[i].intermediate.dense.weight.shape)

if __name__ == '__main__':
    main()