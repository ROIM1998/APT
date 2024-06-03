import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
from models import build_model
from transformers import HfArgumentParser
from args import DataTrainingArguments
from models.model_args import ModelArguments
from args import MinusTrainingArguments
from prune.scheduler import RandomPruningScheduler, SequentialPruningScheduler
from utils import build_trainer
from utils.utils import get_raw_datasets, build_data
from utils.cofi_utils import update_params, prune_model_with_z

def main():
    sys.argv = ['test_sequential_pruning.py',
            '--output_dir',
            './output/neuron_importance/',
            '--model_name_or_path',
            'output/roberta-base_lora_minus_mnli_linear_gradual_global_newself_sequentialdistill/mac0.4/epoch10/bz128/numprune5/lora_r64/lora_alpha16/pruning_5_model',
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
            '64'
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
    mask_prefix = 'final_' if os.path.exists(os.path.join(model_args.model_name_or_path, 'final_head_mask.pt')) else ''
    zs = {
        'head_z': torch.load(os.path.join(model_args.model_name_or_path, mask_prefix + 'head_mask.pt'), map_location='cpu'),
        'intermediate_z': torch.load(os.path.join(model_args.model_name_or_path, mask_prefix + 'intermediate_mask.pt'), map_location='cpu'),
    }
    head_mask, intermediate_mask = torch.stack(zs['head_z']), torch.stack(zs['intermediate_z'])
    trainer = build_trainer(data_args, training_args, model, tokenizer, eval_dataset=eval_dataset)
    pruning_scheduler = SequentialPruningScheduler(model, head_mask, intermediate_mask, None, None)
    pruning_scheduler.gen_schedule(5)
    zs_schedule = pruning_scheduler.zs_schedule
    
    update_params(model, zs)
    prune_model_with_z(zs, model)
    head_mask, intermediate_mask = zs['head_z'], zs['intermediate_z']
    model.head_mask, model.intermediate_mask = None, None
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)

    pruning_scheduler = RandomPruningScheduler(model, head_mask, intermediate_mask, None, None)
    pruning_scheduler.gen_schedule(5)
    for now_zs in pruning_scheduler.zs_schedule:
        update_params(model, now_zs)
        prune_model_with_z(now_zs, model)

    once_pruned_params = {
        n: p for n, p in model.named_parameters()
    }
    sequential_pruned_params = {
        n: p for n, p in model.named_parameters()
    }
    equal_params = [k for k in once_pruned_params.keys() if torch.equal(once_pruned_params[k], sequential_pruned_params[k])]
    unequal_params = [k for k in once_pruned_params.keys() if not torch.equal(once_pruned_params[k], sequential_pruned_params[k])]
    print(f'equal params: {len(equal_params)}')
    print(f'unequal params: {len(unequal_params)}')

    
if __name__ == '__main__':
    main()