from post_analysis import compare_module_inputs_equality
import torch
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import time

from transformers import HfArgumentParser
from args import DataTrainingArguments
from models import build_model
from models.model_args import ModelArguments
from trainer.trainer_minus import ParamController
from utils import build_dataloader, build_trainer
from utils.utils import *
from args import MinusTrainingArguments
from utils.cofi_utils import prune_model_with_z
from loralib.layers import LoRALayer, DistillLinear, PruningLinear

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
    sys.argv = ['test_layer_conversion.py',
            '--output_dir',
            './output/test/test_layer_conversion/',
            '--model_name_or_path',
            'output/debug_roberta_co_learning_output/pruning_1_model',
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
            '--lora_r',
            '8',
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
    os.makedirs(training_args.output_dir, exist_ok=True)
    t_name, raw_datasets = get_raw_datasets(model_args, data_args, training_args)
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)
    training_args.disable_tqdm = True
    mask_prefix = 'final_' if os.path.exists(os.path.join(model_args.model_name_or_path, 'final_head_mask.pt')) else ''
    head_mask, intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, mask_prefix + 'head_mask.pt'), map_location='cpu'), torch.load(os.path.join(model_args.model_name_or_path, mask_prefix + 'intermediate_mask.pt'), map_location='cpu')
    model.head_mask, model.intermediate_mask = [v.to(training_args.device) for v in head_mask], [v.to(training_args.device) for v in intermediate_mask]
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset)
    result = trainer.evaluate()
    
    dataloader = build_dataloader(train_dataset, training_args.per_device_train_batch_size, data_args, training_args, tokenizer)
    model.to(training_args.device)
    inputs = next(iter(dataloader))
    inputs = {k: v.to(training_args.device) for k, v in inputs.items()}

    teacher_config  = ParamController.parse_tuning_param_str(training_args.teacher_param_tuning_config)
    student_config = ParamController.parse_tuning_param_str(training_args.student_param_tuning_config)
    student_config = {**student_config, **teacher_config}
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        student_config=student_config,
        lora_with_bias=False,
    )
    param_controller.convert_to_pre_pruning_lora_teacher()
    param_controller.model_as_teacher()
    loaded_lora_info = collect_lora_info(model)
    print("LoRA param num: ", loaded_lora_info['lora_param_num'])
    print("LoRA var num: ", len(loaded_lora_info['lora_vars']))
    print("LoRA layer num", len(loaded_lora_info['lora_layers']))
    print("LoRA Pruning layer num", len(loaded_lora_info['prune_layers']))
    print("DistillLinear layer num", len(loaded_lora_info['distill_layers']))
    print("Tuning parameter num", sum([p.numel() for n, p in model.named_parameters() if p.requires_grad]))
    print("Tuning var num", len([p.numel() for n, p in model.named_parameters() if p.requires_grad]))
    print("Non-tuning lora parameters", [n for n, p in model.named_parameters() if 'lora' in n and not p.requires_grad])

    param_controller.convert_to_distill(head_mask, intermediate_mask)
    param_controller.model_teacher_with_student()
    distill_lora_info = collect_lora_info(model)
    print("LoRA param num: ", distill_lora_info['lora_param_num'])
    print("LoRA var num: ", len(distill_lora_info['lora_vars']))
    print("LoRA layer num", len(distill_lora_info['lora_layers']))
    print("LoRA Pruning layer num", len(distill_lora_info['prune_layers']))
    print("DistillLinear layer num", len(distill_lora_info['distill_layers']))
    print("Tuning parameter num", sum([p.numel() for n, p in model.named_parameters() if p.requires_grad]))
    print("Tuning var num", len([p.numel() for n, p in model.named_parameters() if p.requires_grad]))
    print("Non-tuning lora parameters", [n for n, p in model.named_parameters() if 'lora' in n and not p.requires_grad])

    
    param_controller.convert_to_post_distillation_lora_student()
    converted_lora_info = collect_lora_info(model)
    print("LoRA param num: ", converted_lora_info['lora_param_num'])
    print("LoRA var num: ", len(converted_lora_info['lora_vars']))
    print("LoRA layer num", len(converted_lora_info['lora_layers']))
    print("LoRA Pruning layer num", len(converted_lora_info['prune_layers']))
    print("DistillLinear layer num", len(converted_lora_info['distill_layers']))
    param_controller.model_as_student()
    print("Tuning parameter num", sum([p.numel() for n, p in model.named_parameters() if p.requires_grad]))
    print("Tuning var num", len([p.numel() for n, p in model.named_parameters() if p.requires_grad]))
    print("Non-tuning lora parameters", [n for n, p in model.named_parameters() if 'lora' in n and not p.requires_grad])


    zs = {
        'head_z': head_mask,
        'intermediate_z': intermediate_mask,
    }
    prune_model_with_z(zs, model)
    model.head_mask, model.intermediate_mask = None, None
    pruned_lora_info = collect_lora_info(model)
    print("LoRA param num: ", pruned_lora_info['lora_param_num'])
    print("LoRA var num: ", len(pruned_lora_info['lora_vars']))
    print("LoRA layer num", len(pruned_lora_info['lora_layers']))
    print("LoRA Pruning layer num", len(pruned_lora_info['prune_layers']))
    print("DistillLinear layer num", len(pruned_lora_info['distill_layers']))