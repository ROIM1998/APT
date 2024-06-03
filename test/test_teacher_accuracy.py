import torch
import os
os.environ["WANDB_DISABLED"] = "true"
import sys

from transformers import HfArgumentParser
from args import DataTrainingArguments
from models import build_model
from models.model_args import ModelArguments
from trainer.trainer_minus import ParamController
from utils import build_dataloader
from utils.utils import *
from args import MinusTrainingArguments


def main():
    sys.argv = ['neuron_importance.py',
            '--output_dir',
            './output/neuron_importance/',
            '--model_name_or_path',
            'roberta-base',
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
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    _, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)
    
    model_path = 'output/roberta-base_lora_minus_mnli_once_global_self_interleavedistill/mac0.4/epoch10/bz128/numprune5/lora_r64/lora_alpha16/pruning_1_model'
    head_mask, intermediate_mask = torch.load(os.path.join(model_path, 'head_mask.pt')), torch.load(os.path.join(model_path, 'intermediate_mask.pt'))
    
    # print(trainer.evaluate())
    eval_dataloader = build_dataloader(eval_dataset, training_args.per_device_train_batch_size, data_args, training_args, tokenizer)
    model.to(training_args.device)
    model.head_mask, model.intermediate_mask = None, None
    teacher_config  = ParamController.parse_tuning_param_str(training_args.teacher_param_tuning_config)
    student_config = ParamController.parse_tuning_param_str(training_args.student_param_tuning_config)
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        student_config=student_config,
        lora_with_bias=True,
    )
    param_controller.convert_to_distill(head_mask, intermediate_mask)
    
    
    loaded_weights = torch.load(os.path.join(model_path, 'pytorch_model.bin'))
    best_weights = torch.load(os.path.join(model_path.replace('pruning_1_model', 'best_model'), 'pytorch_model.bin'))
    mismatch_weights = [k for k in loaded_weights.keys() if not (loaded_weights[k] == best_weights[k]).all()]
    
    model.load_state_dict(loaded_weights)
    model.head_mask, model.intermediate_mask = head_mask, intermediate_mask
    
    with torch.no_grad():
        all_labels, all_preds = [], []
        all_loss = []
        for step, inputs in enumerate(eval_dataloader):
            inputs = {k: v.to(training_args.device) for k, v in inputs.items()}
            outputs = model(**inputs, pass_mask=False, use_teacher=True)
            all_labels += inputs['labels'].detach().cpu().numpy().tolist()
            all_preds += outputs[1].argmax(dim=1).detach().cpu().numpy().tolist()
            all_loss.append(outputs[0].item())
        accuracy = (np.array(all_labels) == np.array(all_preds)).mean()
        loss = np.mean(all_loss)
        print("Teacher accuracy:", accuracy, "loss:", loss)
        
        
        all_labels, all_preds = [], []
        all_loss = []
        for step, inputs in enumerate(eval_dataloader):
            inputs = {k: v.to(training_args.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            all_labels += inputs['labels'].detach().cpu().numpy().tolist()
            all_preds += outputs[1].argmax(dim=1).detach().cpu().numpy().tolist()
            all_loss.append(outputs[0].item())
        accuracy = (np.array(all_labels) == np.array(all_preds)).mean()
        loss = np.mean(all_loss)
        print("Student accuracy:", accuracy, "loss:", loss)