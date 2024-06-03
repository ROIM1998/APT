import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import loralib as lora

from tqdm import tqdm
from typing import Tuple
from transformers import (HfArgumentParser, set_seed)
from torch.utils.data import Subset
from args import Seq2SeqDataTrainingArguments
from models.model_args import ModelArguments
from args import MinusTrainingArguments
from utils.utils import *
from utils import build_trainer, build_dataloader
from models import build_model
from trainer.param_control import ParamController

MB = 1024 * 1024

def main():
    sys.argv = ['test_t5_lm_adapt_elastictuning.py',
            '--output_dir',
            './output/test_magnitude_scorer/',
            '--model_name_or_path',
            'output/google/t5-base-lm-adapt/cnndm/bz16/elastictuning_virtualprune/mac0.4/epoch10/distill_epoch5/numprune10/sparsity_warmup1/pruning_start-1/pruning_stop3/lora_r8/lora_alpha/warmup_parameq:0-11,ev:0-11,dq:0-11,dv:0-11,cq:0-11,cv:0-11,ei0:0-11,di0:0-11/teacher_parameq:0-11,ev:0-11,dq:0-11,dv:0-11,cq:0-11,cv:0-11,ei0:0-11,di0:0-11/pre_distillation_model',
            '--task_name',
            'cnndm',
            '--do_train',
            '--do_eval',
            '--max_input_length',
            '512',
            '--max_target_length',
            '128',
            '--per_device_train_batch_size',
            '16',
            '--per_device_eval_batch_size',
            '16',
            '--apply_lora',
            '--do_distill',
            '--lora_r',
            '8',
            '--report_to',
            'none',
            '--pruner_type',
            'running_fisher',
            '--pre_tuning_scorer',
            'backward_running_hidden_states_salience',
            '--pre_tuning_pruner',
            'running_fisher',
            '--pruning_batch_size',
            '4',
            '--pruning_batches',
            '64',
            '--distill_mapping_strategy',
            'dynamic_block_teacher_dynamic_student'
            ]
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
    set_seed(128)
    # training_args.disable_tqdm = False
    config, tokenizer, model = build_model(model_args, data_args, training_args, force_model_shape_deduction=True)
    train_dataset, eval_dataset, _, _ = build_seq2seq_data(data_args, training_args, tokenizer)

    # Also add ffn input layers to teacher config
    teacher_keys = ['enc_self_query', 'enc_self_value', 'dec_self_query', 'dec_self_value', 'cross_query', 'cross_value']
    teacher_config = {
        k: [i for i in range(config.num_layers)] for k in teacher_keys
    }
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        lora_with_bias=False,
    )
    if os.path.exists(os.path.join(model_args.model_name_or_path, 'head_mask.pt')):
        model.head_mask, model.intermediate_mask, model.hidden_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt')).to(training_args.device), torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt')).to(training_args.device), torch.load(os.path.join(model_args.model_name_or_path, 'hidden_mask.pt')).to(training_args.device)
    
    for m in model.modules():
        if isinstance(m, lora.LoRALayer):
            m.scaling = 2
            if isinstance(m, lora.DistillLinear):
                m.teacher_scaling = 2
    
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    dataloader = trainer.pruning_dataloader
    inputs = next(iter(dataloader))
    inputs = {k: v.to(training_args.device) for k, v in inputs.items()}

    with torch.no_grad():
        teacher_outputs = model(**inputs, use_teacher=True, pass_mask=False, output_hidden_states=True)
    
    student_outputs = model(**inputs, output_hidden_states=True)
    distill_loss, distill_ce_loss, loss = trainer.calculate_distillation_loss(
        (teacher_outputs[0], teacher_outputs[1], teacher_outputs[5], teacher_outputs[3]),
        (student_outputs[0], student_outputs[1], student_outputs[5], student_outputs[3]),
    )
    
    with torch.no_grad():
        inputs = F.log_softmax(student_outputs[1] / training_args.distill_temp, dim=-1)  #! logits: [32,3]
        target = F.softmax(teacher_outputs[1] / training_args.distill_temp, dim=-1)  #! distill_temp: 2.0
        loss = F.kl_div(inputs[:, :2, :], target[:, :2, :], reduction="batchmean") * (training_args.distill_temp ** 2)
    
    ce_distill_loss = F.kl_div(
        input=F.log_softmax(
            student_outputs[1] / training_args.distill_temp, dim=-1), #! logits: [32,3]
        target=F.softmax(
            teacher_outputs[1] / training_args.distill_temp, dim=-1), #! distill_temp: 2.0
        reduction="batchmean") * (training_args.distill_temp ** 2)
    
    losses = []
    with torch.no_grad():
        eval_dataloader= trainer.get_eval_dataloader(eval_dataset)
        for inputs in tqdm(eval_dataloader):
            inputs = {k: v.to(training_args.device) for k, v in inputs.items()}
            trainer.shortens_inputs(inputs)
            losses.append(model(**inputs, use_teacher=True, pass_mask=False, output_hidden_states=True)[0].detach())

if __name__ == '__main__':
    main()