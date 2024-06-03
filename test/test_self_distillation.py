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
from utils.utils import *
from utils import build_trainer, build_dataloader
from args import MinusTrainingArguments
from utils.cofi_utils import prune_model_with_z
from trainer.model_arch import hijack_input
from transformers.trainer import nested_detach

def main():
    sys.argv = ['neuron_importance.py',
            '--output_dir',
            './output/neuron_importance/',
            '--model_name_or_path',
            'output/debug_co_learning/pre_pruning_model',
            '--task_name',
            'mnli',
            '--do_train',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '128',
            '--per_device_eval_batch_size',
            '128',
            '--lora_r',
            '64',
            '--apply_lora',
            '--do_distill',
            '--report_to',
            'none',
            '--do_distill',
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
    head_mask, intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt'), map_location='cpu'), torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt'), map_location='cpu')
    teacher_keys = ['query', 'value']
    teacher_config  = {
        k: [i for i in range(9, 12)] for k in teacher_keys
    }
    student_keys = ['query', 'value', 'intermediate']
    student_config = {
        k: [i for i in range(6, 12)] for k in student_keys
    }
    student_config = {**teacher_config, **student_config}
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        student_config=student_config,
        lora_with_bias=False,
    )
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    
    model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_head_mask.pt'))
    model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_intermediate_mask.pt'))
    model.hidden_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_hidden_mask.pt'))
    dataloader = trainer.get_eval_dataloader(eval_dataset)
    inputs = next(iter(dataloader))
    inputs = trainer._prepare_inputs(inputs)
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
    for student_encoder_layer, teacher_encoder_layer in zip(student_outputs[2], teacher_outputs[2]):
        print((student_encoder_layer - teacher_encoder_layer).pow(2).mean())
        

    
    config, tokenizer, original_model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    optimizer = trainer.create_optimizer()
    param_controller.convert_to_distill(head_mask, intermediate_mask)
    param_controller.model_teacher_with_student()
    # print(trainer.evaluate())
    dataloader = build_dataloader(train_dataset, 32, data_args, training_args, tokenizer)
    original_inputs = next(iter(dataloader))
    original_inputs = trainer._prepare_inputs(original_inputs)
    trainer.optimizer = None
    optimizer = trainer.create_optimizer()
    model.head_mask, model.intermediate_mask = head_mask.to(training_args.device), intermediate_mask.to(training_args.device)
    
    model.eval()
    original_model.eval()
    original_model = original_model.to(training_args.device)
    original_model.head_mask, original_model.intermediate_mask = None, None
    
    with torch.no_grad():
        original_crossed_outputs = model(**original_inputs, output_hidden_states=True, output_masked_states=True, return_dict=False, use_cross_masked_states=True)
    original_outputs = original_model(**inputs, output_hidden_states=True, return_dict=False)

    for i, inputs in enumerate(dataloader):
        _ = model.eval()
        
        _ = model.train()
        inputs = trainer._prepare_inputs(inputs)
        crossed_outputs = model(**inputs, output_hidden_states=True, output_masked_states=True, return_dict=False, use_cross_masked_states=True)
        teacher_outputs = (crossed_outputs[0], crossed_outputs[2], crossed_outputs[4])
        student_outputs = (crossed_outputs[1], crossed_outputs[3], crossed_outputs[5])
        teacher_loss = teacher_outputs[0]
        teacher_outputs = nested_detach(teacher_outputs)
        zs = {
            'head_z': model.head_mask,
            'intermediate_z': model.intermediate_mask,
        }
        distillation_loss, distillation_ce_loss, loss = trainer.calculate_distillation_loss(teacher_outputs, student_outputs, zs)
        # loss = 0.5 * teacher_loss + 0.5 * loss
        loss.backward()
        optimizer.step()
        _ = model.eval()
        with torch.no_grad():
            optimized_crossed_outputs = model(**original_inputs, output_hidden_states=True, output_masked_states=True, return_dict=False, use_cross_masked_states=True)
        print("Teacher loss change:", (optimized_crossed_outputs[0] - original_crossed_outputs[0]))
        if i > 10:
            break

    all_zs = torch.load(os.path.join(model_args.model_name_or_path, '../zs_schedule.pt'), map_location='cpu')
    zs = all_zs[-2]
    prune_model_with_z(
        zs,
        model,
    )

    with torch.no_grad():
        masks =  torch.load(os.path.join(model_args.model_name_or_path, '../masks_schedule.pt'), map_location=model.device)
        head_mask, intermediate_mask = masks[-1]['head_mask'], masks[-1]['intermediate_mask']
        model.head_mask, model.intermediate_mask = None, None
        original_outputs = model(**inputs, output_hidden_states=True, return_dict=False, use_teacher=True)
        
        model.head_mask, model.intermediate_mask = head_mask, intermediate_mask
        masked_outputs = model(**inputs, output_hidden_states=True, return_dict=False, use_teacher=False)
        crossed_outputs = model(**inputs, output_hidden_states=True, output_masked_states=True, return_dict=False, use_cross_masked_states=True)
        teacher_outputs = (crossed_outputs[0], crossed_outputs[2], crossed_outputs[4])
        student_outputs = (crossed_outputs[1], crossed_outputs[3], crossed_outputs[5])
        outputs = model(**inputs, output_hidden_states=True, output_masked_states=True, return_dict=False)
        inversed_outputs = model(**inputs, output_hidden_states=True, output_masked_states=True, return_dict=False, use_unmasked_states=False)
        
        print("Overall mask hidden states check: ", torch.allclose(masked_outputs[2][1] , outputs[3][0]))
        print("Overall original hidden states check: ", [
                torch.allclose(original_outputs[2][i] , outputs[2][i])
                for i in range(model.config.num_hidden_layers + 1) 
        ])
        
        def get_cross_outputs():
            start_time = time.time()
            crossed_outputs = model(**inputs, output_hidden_states=True, output_masked_states=True, return_dict=False, use_cross_masked_states=True)
            torch.cuda.synchronize()
            self_distill_time = time.time() - start_time
            return crossed_outputs, self_distill_time
        
        def get_inlayer_outputs():
            start_time = time.time()
            outputs = model(**inputs, output_hidden_states=True, output_masked_states=True, return_dict=False)
            torch.cuda.synchronize()
            self_distill_time = time.time() - start_time
            return crossed_outputs, self_distill_time
        
        times = []
        for i in range(10):
            crossed_outputs, self_distill_time = get_cross_outputs()
            times.append(self_distill_time)
            
        inlayer_times = []
        for i in range(10):
            inlayer_outputs, self_distill_time = get_inlayer_outputs()
            inlayer_times.append(self_distill_time)

        print("Crossed loss and logits check: ", torch.allclose(crossed_outputs[0] , original_outputs[0]), torch.allclose(crossed_outputs[2] , original_outputs[1]), torch.allclose(crossed_outputs[1] , masked_outputs[0]), torch.allclose(crossed_outputs[3] , masked_outputs[1]))
        print("Crossed unmasked hidden states check: ", [torch.allclose(crossed_outputs[4][i] , original_outputs[2][i]) for i in range(13)])
        print("Crossed unmasked hidden states check: ", [torch.allclose(crossed_outputs[4][i] , original_outputs[2][i]) for i in range(13)])
        print("Crossed masked hidden states check: ", [torch.allclose(crossed_outputs[5][i] , masked_outputs[-1][i+1]) for i in range(12)])

        
        all_inputs = []
        handles = []
        for i in range(12):
            got_inputs = []
            handle = hijack_input(model.roberta.encoder.layer[i], got_inputs)
            handles.append(handle)
            all_inputs.append(got_inputs)
        model.head_mask, model.intermediate_mask = None, None
        original_outputs = model(**inputs, output_hidden_states=True, return_dict=False)
        model.head_mask, model.intermediate_mask = head_mask, intermediate_mask
        outputs = model(**inputs, output_hidden_states=True, output_masked_states=True, return_dict=False)
        for handle in handles:
            handle.remove()
        print("Layer inputs check:", [(i[0][0] - i[1][0]).abs().max() if i[0][0] is not None else (i[0][0] is None and i[1][0] is None) for i in all_inputs if len(i)])


        model.head_mask, model.intermediate_mask = head_mask, intermediate_mask
        all_inputs = []
        handles = []
        for i in range(12):
            got_inputs = []
            handle = hijack_input(model.roberta.encoder.layer[i].attention.output.dense, got_inputs)
            handles.append(handle)
            all_inputs.append(got_inputs)
        masked_outputs = model(**inputs, output_hidden_states=True, return_dict=False)
        inversed_outputs = model(**inputs, output_hidden_states=True, output_masked_states=True, return_dict=False, use_unmasked_states=False)
        for handle in handles:
            handle.remove()
        print("Layer inputs check:", [(i[0][0] - i[2][0]).abs().max() if i[0][0] is not None else (i[0][0] is None and i[1][0] is None) for i in all_inputs if len(i)])
        
        

    
    head_z = model.roberta.get_head_mask(head_mask, 12)
    intermediate_z = intermediate_mask
    
    for i in range(model.config.num_hidden_layers):
        hidden_states = torch.randn([32, 128, 768]).to('cuda')
        outputs = model.roberta.encoder.layer[i](
            hidden_states,
            output_masked_states=True,
            head_z=head_z[i],
            intermediate_z=intermediate_z[i],
        )
        original_outputs = model.roberta.encoder.layer[i](
            hidden_states,
            head_z=None,
            intermediate_z=None,
        )
        masked_outputs = model.roberta.encoder.layer[i](
            hidden_states,
            head_z=head_z[i],
            intermediate_z=intermediate_z[i],
        )
        
        print("Original outputs layer %d check: " % i, torch.allclose(original_outputs[0], outputs[0]))
        print("Masked outputs layer %d check: " % i, torch.allclose(masked_outputs[0], outputs[-1]))