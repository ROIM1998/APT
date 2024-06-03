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

def main():
    sys.argv = ['neuron_importance.py',
            '--output_dir',
            './output/neuron_importance/',
            '--model_name_or_path',
            'output/roberta-base_lora_minus_mnli_once_global_self_student_loratransform_distill/mac0.4/epoch25/bz32/numprune5/lora_r64/lora_alpha16/best_model',
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
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)
    training_args.disable_tqdm = True
    head_mask, intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt')), torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt'))

    # print(trainer.evaluate())
    dataloader = build_dataloader(train_dataset, training_args.per_device_train_batch_size, data_args, training_args, tokenizer)
    model.to(training_args.device)
    inputs = next(iter(dataloader))
    inputs = {k: v.to(training_args.device) for k, v in inputs.items()}

    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset)
    model.head_mask, model.intermediate_mask = None, None
    
    # previous_model_params = dict(model.named_parameters())
    # pre_switch_metrics = trainer.evaluate()
    teacher_config  = ParamController.parse_tuning_param_str(training_args.teacher_param_tuning_config)
    student_config = ParamController.parse_tuning_param_str(training_args.student_param_tuning_config)
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        student_config=student_config,
        lora_with_bias=False,
    )
    param_controller.convert_to_distill(head_mask, intermediate_mask)
    # converted_model_params = dict(model.named_parameters())
    # post_switch_metrics = trainer.evaluate()
    
    model.head_mask, model.intermediate_mask = head_mask, intermediate_mask
    load_best = False
    if load_best:
        model_args.model_name_or_path = 'output/roberta-base_lora_mnli/epoch30/lora_r8/lora_alpha16/best_model'
        config, tokenizer, lt_model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
        model_args.model_name_or_path = 'roberta-base'
        config, tokenizer, pt_model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
        lt_model.layer_transformation=None
        lt_trainer = build_trainer(data_args, training_args, lt_model, tokenizer, train_dataset, eval_dataset)
        lt_model.head_mask, lt_model.intermediate_mask = None, None
        lt_trainer.evaluate()
        model.layer_transformation=None
        for i in range(12):
            model.roberta.encoder.layer[i].attention.self.query.weight.data = lt_model.roberta.encoder.layer[i].attention.self.query.weight.data
            model.roberta.encoder.layer[i].attention.self.value.weight.data = lt_model.roberta.encoder.layer[i].attention.self.value.weight.data
            model.roberta.encoder.layer[i].attention.self.query.teacher_lora_A.data = lt_model.roberta.encoder.layer[i].attention.self.query.lora_A.data
            model.roberta.encoder.layer[i].attention.self.query.teacher_lora_B.data = lt_model.roberta.encoder.layer[i].attention.self.query.lora_B.data
            model.roberta.encoder.layer[i].attention.self.value.teacher_lora_A.data = lt_model.roberta.encoder.layer[i].attention.self.value.lora_A.data
            model.roberta.encoder.layer[i].attention.self.value.teacher_lora_B.data = lt_model.roberta.encoder.layer[i].attention.self.value.lora_B.data
            model.classifier.dense.weight.data = lt_model.classifier.dense.weight.data
            model.classifier.out_proj.weight.data = lt_model.classifier.out_proj.weight.data
        trainer.distilling = True
        trainer.evaluate()
        
    
    training_args.learning_rate = 5e-4
    optimizer = trainer.create_optimizer()
    model.head_mask, model.intermediate_mask = head_mask, intermediate_mask
    trainer.distilling = True
    # pre_distillation_metrics = trainer.evaluate()
    
    param_controller.model_decouple_as_teacher()
    training_steps = 0
    training_test = True
    if training_test:
        for inputs in dataloader:
            inputs = trainer._prepare_inputs(inputs)
            _ = model.train()
            outputs = model(**inputs, pass_mask=False, use_teacher=True)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            model.zero_grad()
            training_steps += 1
            if training_steps % 200 == 0:
                print(model.roberta.encoder.layer[0].attention.self.query.teacher_lora_A.sum().item())
                print(model.roberta.encoder.layer[0].attention.self.query.teacher_lora_B.sum().item())
                print(trainer.evaluate())
                
    
    conducting_pruning_test = False
    if conducting_pruning_test:
        zs = {
            'head_z': [v.to('cpu') for v in head_mask],
            'intermediate_z': [v.to('cpu') for v in intermediate_mask],
        }
        prune_model_with_z(zs, model)
        
    layer_transformation_test = True
    if layer_transformation_test:
        eval_dataloader = build_dataloader(eval_dataset, training_args.per_device_eval_batch_size, data_args, training_args, tokenizer)
        all_labels, all_preds, all_trans_preds = [], [], []
        with torch.no_grad():
            for inputs in eval_dataloader:
                inputs = trainer._prepare_inputs(inputs)
                _ = model.eval()
                student_outputs = model(**inputs, output_hidden_states=True)
                student_last_hidden_state = student_outputs[2][-1]
                transformed_student_last_hidden_state = model.layer_transformation(student_last_hidden_state)
                transformed_cls_output = model.classifier(transformed_student_last_hidden_state)
                all_labels.append(inputs['labels'])
                all_preds.append(student_outputs[1].argmax(dim=1))
                all_trans_preds.append(transformed_cls_output.argmax(dim=1))

        non_trans_acc = (torch.cat(all_labels) == torch.cat(all_preds)).sum().item() / len(torch.cat(all_labels))
        trans_acc = (torch.cat(all_labels) == torch.cat(all_trans_preds)).sum().item() / len(torch.cat(all_labels))
        print("Transformed acc: ", trans_acc, "Non-transform acc", non_trans_acc)
        
    model.train()
    param_controller.model_decouple_as_teacher()
    teacher_outputs = model(**inputs, pass_mask=False, use_teacher=True)
    loss = teacher_outputs[0]
    loss.backward()
    print("Num of variables requires grad", len([n for n, p in model.named_parameters() if p.grad is not None]))
    model.zero_grad()
    for p in model.parameters():
        p.grad = None

    param_controller.model_decouple_as_student()
    for n, p in model.named_parameters():
        if 'lora' in n and 'teacher' not in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
    with torch.no_grad():
        teacher_outputs = model(**inputs, output_hidden_states=True, return_dict=False, pass_mask=False, use_teacher=True)
    student_outputs = model(**inputs, output_hidden_states=True, return_dict=False)
    distill_loss, distill_ce_loss, loss = trainer.calculate_distillation_loss(teacher_outputs, student_outputs, zs)
    loss = loss * 0.5 + student_outputs[0] * 0.5
    loss.backward()
    print("Num of variables requires grad", len([n for n, p in model.named_parameters() if p.grad is not None]))