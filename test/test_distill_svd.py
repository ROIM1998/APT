import seaborn as sns
sns.set_theme(style="darkgrid")
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
import loralib as lora

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
from post_analysis import compare_module_inputs_equality

def main():
    sys.argv = ['test_adapt_pruning.py',
            '--output_dir',
            './output/test_adapt_pruning/',
            '--model_name_or_path',
            'output/roberta-base_lora_minus_mnli_once_global_free_inout_nodistill/mac0.4/epoch30/bz128/numprune5/paramq:0-11,v:0-11,i:0-11/lora_r32/lora_alpha16/pre_pruning_model',
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
            '32',
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
            '--continuous_allocation',
            '--continuous_alloc_interval',
            '1',
            '--distillation_type',
            'self_interleave',
            '--distill_mapping_strategy',
            'none',
            '--param_allocation_strategy',
            'free_inout'
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

    teacher_keys = ['query', 'value', 'intermediate']
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
        param_allocation_strategy=training_args.param_allocation_strategy,
    )
    
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    
    lora_A, lora_B = model.roberta.encoder.layer[0].attention.self.query.lora_A.data.clone(), model.roberta.encoder.layer[0].attention.self.query.lora_B.data.clone()
    weight = model.roberta.encoder.layer[0].attention.self.query.weight.data.clone()
    scaling = model.roberta.encoder.layer[0].attention.self.query.scaling
    teacher_scaling = 16 / 8
    equivalent = scaling * (lora_B @ lora_A) / teacher_scaling
    u, s, v = torch.linalg.svd(equivalent)
    teacher_lora_B = u[:, :8]
    teacher_lora_A = v[:8, :]
    teacher_s = s[:8]
    teacher_lora_A =  teacher_s.unsqueeze(1) * teacher_lora_A
    error = torch.norm(teacher_lora_B @ teacher_lora_A * teacher_scaling - lora_B @ lora_A * scaling)
    
    param_controller.convert_to_distill(None, None)
    trainer.distilling = True
    model.is_distilling = True
    model.head_mask, model.intermediate_mask = None, None
    trainer.evaluate()
    
    compare_consistency = False
    if compare_consistency:
        config, tokenizer, original_model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
        original_model.head_mask, original_model.intermediate_mask = None, None
        inputs = next(iter(dataloader))
        inputs = trainer._prepare_inputs(inputs)
        original_model.eval().to(training_args.device)
        model.eval()
        outputs, original_outputs = model(**inputs, output_hidden_states=True), original_model(**inputs, output_hidden_states=True)
        original_hidden_states = original_outputs.hidden_states
        hidden_states=  outputs.hidden_states
        for i in range(len(hidden_states)):
            print(torch.norm(hidden_states[i] - original_hidden_states[i]))
            
        named_modules = dict(model.named_parameters())
        for n, p in original_model.named_parameters():
            if n in named_modules and not (p == named_modules[n]).all():
                print(n)
                print(torch.norm(p - named_modules[n]))
            elif n not in named_modules:
                print(n, 'not found')

        q = model.roberta.encoder.layer[0].attention.self.query
        original_q = original_model.roberta.encoder.layer[0].attention.self.query
        a = hidden_states[0]
        b = (a @ ((q.lora_B @ q.lora_A) * q.scaling + q.weight).T) + q.bias
        original_b = (a @ ((original_q.lora_B @ original_q.lora_A) * original_q.scaling + original_q.weight).T) + original_q.bias

        res = compare_module_inputs_equality([model, original_model], inputs, lambda x: x.roberta.encoder.layer[0].output.dense)
        
        
        
if __name__ == '__main__':
    main()