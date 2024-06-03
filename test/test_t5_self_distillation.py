from utils.minus_utils import compare_module_inputs_equality, collect_module_inputs

import torch
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import time
from copy import deepcopy

from transformers import HfArgumentParser
from args import DataTrainingArguments
from models import build_model
from models.model_args import ModelArguments
from utils import build_trainer
from utils.utils import *
from args import MinusTrainingArguments
from transformers.trainer_pt_utils import nested_detach

def main():
    sys.argv = ['test_t5_lm_adapt.py',
            '--output_dir',
            './output/test_t5_lm_adapt/',
            '--model_name_or_path',
            'output/debug_t5_lm_adapt_glue/pre_pruning_model',
            '--task_name',
            'mnli',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '32',
            '--per_device_eval_batch_size',
            '32',
            '--eval_accumulation_steps',
            '1',
            '--lora_r',
            '8',
            '--lora_alpha',
            '16',
            '--apply_lora',
            '--do_distill',
            '--pruner_type',
            'search',
            '--head_scorer_type',
            'gradient_l2',
            '--intermediate_scorer_type',
            'gradient_l2',
            '--pruning_batch_size',
            '4',
            '--pruning_batches',
            '64',
            '--pruning_scheduler',
            'once',
            '--teacher_param_tuning_config',
            'eq:9-11,ev:9-11,dq:9-11,dv:9-11,cq:9-11,cv:9-11,ei0:9-11,di0:9-11',
            '--student_param_tuning_config',
            'eq:6-11,ev:6-11,dq:6-11,dv:6-11,cq:6-11,cv:6-11,ei0:6-11,di0:6-11',
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
    MODEL_GENERATIVE = True
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets, MODEL_GENERATIVE)
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset)
    param_controller = trainer.param_controller
    trainer.args.predict_with_generate = True

    dataloader = trainer.get_eval_dataloader(eval_dataset)
    inputs = next(iter(dataloader))
    inputs = trainer._prepare_inputs(inputs)
    
    model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_head_mask.pt'))
    model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_intermediate_mask.pt'))
    model.hidden_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_hidden_mask.pt'))
    
    param_controller.convert_to_distill(None, None)
    param_controller.model_teacher_with_student()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print([n for n, p in model.named_parameters() if p.requires_grad])

    model.eval()
    
    with torch.no_grad():
        teacher_outputs = trainer.model(
            **inputs,
            output_hidden_states=True,
            return_dict=False,
            use_teacher=True,
            pass_mask=False,
        ) # loss, logits, decoder.past_key_values, decoder.hidden_states, encoder.last_hidden_states, encoder.hidden_states
        
    with torch.no_grad():
        student_outputs = trainer.model(
            **inputs,
            output_hidden_states=True,
            return_dict=False,
            use_teacher=False,
            pass_mask=True,
        )
    # teacher_loss = teacher_outputs[0]
    teacher_outputs = nested_detach(teacher_outputs)
    # distill_loss, distill_ce_loss, loss = trainer.calculate_distillation_loss(
    #     (teacher_outputs[0], teacher_outputs[1], teacher_outputs[3], teacher_outputs[5]),
    #     (student_outputs[0], student_outputs[1], student_outputs[3], student_outputs[5]),
    # )
    # trainer.now_distill_loss = distill_loss.item()
    # trainer.now_distill_ce_loss = distill_ce_loss.item()
    # loss = loss * 0.5 + teacher_loss * 0.5
    
    # Print encoder layer-by-layer mse
    for student_encoder_layer, teacher_encoder_layer in zip(student_outputs[5][1:], teacher_outputs[5][1:]):
        print((student_encoder_layer - teacher_encoder_layer).pow(2).mean())
        
    # Print decoder layer-by-layer mse
    for student_decoder_layer, teacher_decoder_layer in zip(student_outputs[3][1:], teacher_outputs[3][1:]):
        print((student_decoder_layer - teacher_decoder_layer).pow(2).mean())
    

    teacher_model = deepcopy(trainer.model)
    model.prune_model_with_masks()
    
    with torch.no_grad():
        pruned_outputs = trainer.model(
            **inputs,
            output_hidden_states=True,
            return_dict=False,
        )
        
    # Print pruned encoder layer-by-layer mse
    for pruned_encoder_layer, student_encoder_layer in zip(pruned_outputs[5], student_outputs[5]):
        print((pruned_encoder_layer - student_encoder_layer).pow(2).mean())
        
    # Print pruned decoder layer-by-layer mse
    for pruned_decoder_layer, student_decoder_layer in zip(pruned_outputs[3], student_outputs[3]):
        print((pruned_decoder_layer - student_decoder_layer).pow(2).mean())
    
    model.eval()
    teacher_model.eval()
    
    with torch.no_grad():
        res = compare_module_inputs_equality([teacher_model, model], inputs, lambda x: x.encoder.block[9].layer[0].SelfAttention.o)
        print((res[0] - res[1]).abs().mean() if res[0].shape == res[1].shape else False)
        
    remained_heads = teacher_model.head_mask[0][9].nonzero().squeeze()
    (model.encoder.block[9].layer[0].SelfAttention.tracked_scores - teacher_model.encoder.block[9].layer[0].SelfAttention.tracked_scores[:, remained_heads, :, :]).abs().max()
    (model.encoder.block[9].layer[0].SelfAttention.tracked_position_bias - teacher_model.encoder.block[9].layer[0].SelfAttention.tracked_position_bias[:, remained_heads, :, :]).any()
    
    (model.encoder.block[11].layer[0].SelfAttention.tracked_scores - teacher_model.encoder.block[11].layer[0].SelfAttention.tracked_scores[:, [10], :, :]).abs().max()
    (model.encoder.block[11].layer[0].SelfAttention.tracked_position_bias - teacher_model.encoder.block[11].layer[0].SelfAttention.tracked_position_bias[:, [10], :, :]).abs().max()
    
    (model.encoder.block[11].layer[0].SelfAttention.tracked_query_states - teacher_model.encoder.block[11].layer[0].SelfAttention.tracked_query_states[:, [10], :, :]).abs().max()
    (model.encoder.block[11].layer[0].SelfAttention.tracked_key_states - teacher_model.encoder.block[11].layer[0].SelfAttention.tracked_key_states[:, [10], :, :]).abs().max()
    
    query_states = teacher_model.encoder.block[11].layer[0].SelfAttention.tracked_query_states
    key_states = teacher_model.encoder.block[11].layer[0].SelfAttention.tracked_key_states
    computed_scores = torch.matmul(query_states, key_states.transpose(3, 2))
    scores = teacher_model.encoder.block[11].layer[0].SelfAttention.tracked_scores
        
    head_mask_use = teacher_model.head_mask[0][9].repeat_interleave(64, dim=0)
    selected_masked_inputs = res[0][:, :, head_mask_use.nonzero().squeeze()]
    pruned_inputs = res[1]
    print(selected_masked_inputs - pruned_inputs)
        
        
    with torch.no_grad():
        masked_inputs = collect_module_inputs(teacher_model, inputs, lambda x: x.encoder.dropout)
        pruned_inputs = collect_module_inputs(model, inputs, lambda x: x.encoder.dropout)
        print((masked_inputs[1][0] - pruned_inputs[1][0]).pow(2).mean())
    
    print((teacher_model.encoder.block[0].layer[0].SelfAttention.tracked_scores[:, 0, :, :] - model.encoder.block[0].layer[0].SelfAttention.tracked_scores[:, 0, :, :]).abs().mean())