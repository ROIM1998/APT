import seaborn as sns
sns.set_theme(style="darkgrid")
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
import loralib as lora
import random

from transformers import (HfArgumentParser)
from args import DataTrainingArguments
from models.model_args import ModelArguments
from utils.utils import *
from utils import build_trainer, build_dataloader
from args import MinusTrainingArguments
from models import build_model
from prune.pruner import AdapterPruner
from torch.utils.data import Subset
from trainer.param_control import ParamController
from prune import build_scorer, BetterFisherPruner
from tqdm import tqdm
from transformers.trainer import nested_detach

def main():
    sys.argv = ['test_adapt_pruning.py',
            '--output_dir',
            './output/test_adapt_pruning/',
            '--model_name_or_path',
            'output/roberta-base_lora_minus_mnli_once_global_free_inout_nodistill/mac0.6/epoch30/bz128/numprune5/paramq:0-11,v:0-11,i:0-11/lora_r8/lora_alpha16/pre_pruning_model',
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
            '--apply_lora',
            '--lora_r',
            '8',
            '--lora_alpha',
            '16',
            '--save_strategy',
            'no',
            '--evaluation_strategy',
            'steps',
            '--num_train_epochs',
            '0.1',
            '--learning_rate',
            '5e-4',
            '--weight_decay',
            '0.1',
            '--warmup_ratio',
            '0.06',
            '--report_to',
            'none',
            '--do_distill',
            '--distillation_type',
            'self_interleave',
            '--distill_mapping_strategy',
            'dynamic_block_teacher_dynamic_student',
            '--param_allocation_strategy',
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

    # Freeze the bottom 6 layers, while tuning the top 6 layers only. Teacher is tuned with supervised fine-tuning objectives, while the student is tuned with distillation objectives.
    teacher_keys = ['query', 'value', 'intermediate']
    teacher_config  = {
        k: [i for i in range(9, 12)] for k in teacher_keys
    }
    student_keys = ['query', 'value', 'intermediate']
    student_config = {
        k: [i for i in range(6, 12)] for k in student_keys
    }
    adapter_pruner = AdapterPruner(model, dataloader)
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        student_config=student_config,
        lora_with_bias=False,
        adapter_pruner=adapter_pruner,
        param_allocation_strategy=training_args.param_allocation_strategy,
    )
    
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_head_mask.pt'))
    model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_intermediate_mask.pt'))
    
    param_controller.convert_to_distill(model.head_mask, model.intermediate_mask)
    param_controller.model_teacher_with_student()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    inputs = next(iter(dataloader))
    inputs = trainer._prepare_inputs(inputs)
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=False,
        output_masked_states=True,
        use_cross_masked_states=True,
    )
    print(torch.cuda.max_memory_allocated() / 1024 / 1024)
    len(outputs)
    teacher_outputs = (outputs[0], outputs[2], outputs[4])
    student_outputs = (outputs[1], outputs[3], outputs[5])
    teacher_loss = teacher_outputs[0]
    
    teacher_outputs = nested_detach(teacher_outputs)
    distill_loss, ce_distill_loss, loss = trainer.calculate_distillation_loss(teacher_outputs, student_outputs)
    loss = 0.5 * loss + 0.5 * teacher_loss
    loss.backward()
    print(torch.cuda.max_memory_allocated() / 1024 / 1024)
    param_with_grad = [n for n, p in model.named_parameters() if p.grad is not None]

    # Compare the layer outputs between the student and the teacher
    result = []
    for inputs in tqdm(dataloader):
        inputs = trainer._prepare_inputs(inputs)
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=False,
            output_masked_states=True,
            use_cross_masked_states=True,
        )
        teacher_outputs = (outputs[0], outputs[2], outputs[4])
        student_outputs = (outputs[1], outputs[3], outputs[5])
        teacher_layer_output = teacher_outputs[2][-6:]
        student_layer_output = student_outputs[2][-12:]

        mse_loss = torch.nn.MSELoss(reduction="mean")
        l = []
        distill_student_layer_selection = {}
        specified_teacher_layers = range(6)
        sampled_layer_pair_num = 4
        specified_student_layer_idx = sorted(random.sample(list(param_controller.student_tuning_layers), sampled_layer_pair_num))
        if hasattr(model, 'layer_transformation') and model.layer_transformation is not None:
            student_layer_output = [model.layer_transformation(student_layer_output[i]) for i in specified_student_layer_idx]
        else:
            student_layer_output = [student_layer_output[i] for i in specified_student_layer_idx]
        specified_teacher_layer_reps = []
        for i in specified_teacher_layers:
            specified_teacher_layer_reps.append(teacher_layer_output[i]) #! teacher: 12x[32,113,768]
        device = student_layer_output[0].device
        
        for t_layer_o in specified_teacher_layer_reps:
            for i, s_layer_o in enumerate(student_layer_output): #! student: 4x[32,113,768]
                l.append(mse_loss(t_layer_o, s_layer_o))
        layerwiseloss = torch.stack(l).reshape(
            len(student_layer_output), len(specified_teacher_layer_reps)) # [4 (student),12 (teacher)]
        
        last_aligned_layer = 6
        alignment = []
        for search_index in range(len(specified_student_layer_idx)-1, -1, -1):
            indexes = layerwiseloss[search_index].sort()[1]
            align = indexes[indexes < last_aligned_layer]
            if len(align) > 0:
                align = align[0]
            else:
                align = last_aligned_layer
            if align not in distill_student_layer_selection:
                distill_student_layer_selection[align.item()] = []
            alignment.append(align)
            last_aligned_layer = align
        alignment.reverse()
        for i, align in enumerate(alignment):
            distill_student_layer_selection[align.item()].append(specified_student_layer_idx[i])
        alignment = torch.tensor(alignment).to(device)

        layerwise = torch.arange(len(specified_student_layer_idx)).to(device)
        layer_loss = layerwiseloss[layerwise, alignment]
        layer_losses = layer_loss.detach().cpu().numpy().tolist()
        layer_loss = layer_loss.sum() #! layerwise: teacher (specified layers) / alignment: student (min loss layers) / layerwiseloss: [4,12]
        result.append({
            'layer_loss': layer_loss.item(),
            'layer_losses': layer_losses,
            'distill_student_layer_selection': distill_student_layer_selection,
            'alignment': alignment.detach().cpu(),
            'layerwise_loss': layerwiseloss.detach().cpu(),
        })

if __name__ == '__main__':
    main()