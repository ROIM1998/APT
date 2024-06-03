import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
from transformers import HfArgumentParser, EvalPrediction, default_data_collator, DataCollatorWithPadding
from args import DataTrainingArguments
from models import build_model
from models.model_args import ModelArguments
from datasets import load_metric
from trainer.trainer_minus import MinusTrainer
from utils.utils import *
from args import MinusTrainingArguments
from utils.cofi_utils import update_params, prune_model_with_z
from torch.utils.data import DataLoader, Subset
from utils.minus_utils import count_params

def main():
    sys.argv = ['neuron_importance.py',
            '--output_dir',
            './output/neuron_importance/',
            '--model_name_or_path',
            'output/roberta-base_lora_minus_mnli_once_global_alloc_none_co_learning_mapping_dynamic_block_teacher_dynamic_student_distill_fixedteacher/mac0.4/epoch40/bz32/numprune5/paramq:9-11,v:9-11,i:9-11/lora_r8/pruning_start1/distill_epoch19/post_distillation_model',
            '--task_name',
            'mnli',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '32',
            '--per_device_eval_batch_size',
            '32',
            '--lora_r',
            '64',
            '--apply_lora',
            '--do_distill'
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
    
    zs_schedule = torch.load(os.path.join(model_args.model_name_or_path, 'zs_schedule.pt'), 'cpu')
    folders = sorted([v for v in os.listdir(model_args.model_name_or_path) if 'pruning' in v and 'pre' not in v and 'post' not in v])
    
    model_path = model_args.model_name_or_path
    models = []
    model_names = []
    for f in folders:
        model_names.append(f)
        model_args.model_name_or_path = os.path.join(model_args.model_name_or_path, f)
        config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
        pruning_index = int(f.split('_')[1]) if len(f.split('_')) == 3 else 0
        for now_zs in zs_schedule[pruning_index:]:
            update_params(model, now_zs)
            prune_model_with_z(now_zs, model)
        models.append(model)
        model_args.model_name_or_path = model_path

    model_args.model_name_or_path = os.path.join(model_args.model_name_or_path, 'best_model')
    # training_args.disable_tqdm = False
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    models.append(model)
    model_names.append('best_model')
    
    model_args.model_name_or_path = 'roberta-base'
    config, tokenizer, pt_model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    for now_zs in zs_schedule:
        update_params(pt_model, now_zs)
        prune_model_with_z(now_zs, pt_model)

    print("Shape consistency check")
    for model, name in zip(models, model_names):
        model_params = dict(model.named_parameters())
        pt_model_params = dict(pt_model.named_parameters())
        assert len(model_params) == len(pt_model_params)
        shape_matched_params = [n for n in model_params if model_params[n].shape == pt_model_params[n].shape]
        shape_mismatch_params = [n for n in model_params if model_params[n].shape != pt_model_params[n].shape]
        print(f"Model: {name}")
        print(f"Shape matched params: {len(shape_matched_params)}")
        print(f"Shape mismatch params: {len(shape_mismatch_params)}")
    
    for model, name in zip(models, model_names):
        model_params = dict(model.named_parameters())
        pt_model_params = dict(pt_model.named_parameters())
        print(f"Model: {name}")
        print("Attn query dense equality:", (model_params['roberta.encoder.layer.0.attention.self.query.weight'] - pt_model_params['roberta.encoder.layer.0.attention.self.query.weight']).abs().max())
    
    print("Param value check")
    value_matched_params = [n for n in shape_matched_params if torch.allclose(model_params[n], pt_model_params[n], atol=1e-7)]
    value_mismatch_params = [n for n in shape_matched_params if not torch.allclose(model_params[n], pt_model_params[n], atol=1e-7)]
    non_lora_mismatch = [n for n in value_mismatch_params if 'lora' not in n]
    print("Value matched params: ", len(value_matched_params))
    print("Value mismatch params: ", len(value_mismatch_params))
    print("Non-LoRA mismatch params: ", len(non_lora_mismatch))
    diff_with_name = [(n, torch.abs(model_params[n] - pt_model_params[n]).max()) for n in non_lora_mismatch]
    print("Non-LoRA mismatch difference:\n", "\n".join([f"{n}: {d}" for n, d in diff_with_name]))
    print("Non-LoRA, non-bias mismatch params:\n ", "\n".join([f"{n}: {d}" for n, d in diff_with_name if 'bias' not in n]))
    
if __name__ == '__main__':
    main()