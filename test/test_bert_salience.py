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
from prune import build_scorer, BetterFisherPruner
from torch.utils.data import Subset
from trainer.param_control import ParamController
from utils.fisher_utils.efficiency.param import *
from utils.minus_utils import load_grafting_masks

def main():
    sys.argv = ['test_pre_tuning_prune.py',
            '--output_dir',
            './output/test_bert_salience/',
            '--model_name_or_path',
            'output/bert-base-uncased_lora_minus_sst2_cubic_gradual_running_fisher_alloc_running_fisher_momentum_mapping_dynamic_block_teacher_dynamic_student_distill_tophalf_limited_resizing/mac0.4/epoch40/bz32/numprune10/paramq:0-11,v:0-11,i:0-11/lora_r8/pruning_start-1/distill_epoch20/best_model',
            '--task_name',
            'sst2',
            '--do_train',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '32',
            '--per_device_eval_batch_size',
            '32',
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
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets, force_model_shape_deduction=True)
    train_dataset, eval_dataset, _, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)


    pruning_batch_size = 32
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
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        student_config=student_config,
        lora_with_bias=False,
    )
    param_controller.convert_to_pruning_lora_teacher()
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    param_controller.model_as_teacher()
    inputs = trainer._prepare_inputs(next(iter(dataloader)))
    
    load_grafting_masks(model, torch.load(os.path.join(model_args.model_name_or_path, 'grafting_masks.pt')))
    model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt')).to(model.device)
    model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt')).to(model.device)
    model.hidden_mask = torch.load(os.path.join(model_args.model_name_or_path, 'hidden_mask.pt')).to(model.device)
    for m in model.modules():
        if isinstance(m, lora.PruningLinear):
            m.scaling = 2
    
    trainer.evaluate()
    param_controller.set_grafting_mask(True, target='teacher', requires_grad=True)

    outputs = model(**inputs)
    print(outputs.loss)
    outputs.loss.backward()
    for i in range(12):
        tuning_input_salience = (model.bert.encoder.layer[i].attention.self.query.lora_A * model.bert.encoder.layer[i].attention.self.query.lora_A.grad).sum(dim=0)
        mask_input_salience = model.bert.encoder.layer[i].attention.self.query.input_mask * model.bert.encoder.layer[i].attention.self.query.input_mask.grad
        print("Query input allclose:", torch.allclose(tuning_input_salience, mask_input_salience))
        tuning_rank_salience = (model.bert.encoder.layer[i].attention.self.query.lora_A * model.bert.encoder.layer[i].attention.self.query.lora_A.grad).sum(dim=1)
        mask_rank_salience = model.bert.encoder.layer[i].attention.self.query.bottleneck_mask * model.bert.encoder.layer[i].attention.self.query.bottleneck_mask.grad
        print("Query rank allclose:", torch.allclose(tuning_rank_salience, mask_rank_salience))
        tuning_input_salience = (model.bert.encoder.layer[i].attention.self.value.lora_A * model.bert.encoder.layer[i].attention.self.value.lora_A.grad).sum(dim=0)
        mask_input_salience = model.bert.encoder.layer[i].attention.self.value.input_mask * model.bert.encoder.layer[i].attention.self.value.input_mask.grad
        print("Value input allclose:", torch.allclose(tuning_input_salience, mask_input_salience))
        tuning_rank_salience = (model.bert.encoder.layer[i].attention.self.value.lora_A * model.bert.encoder.layer[i].attention.self.value.lora_A.grad).sum(dim=1)  
        mask_rank_salience = model.bert.encoder.layer[i].attention.self.value.bottleneck_mask * model.bert.encoder.layer[i].attention.self.value.bottleneck_mask.grad
        print("Value rank allclose:", torch.allclose(tuning_rank_salience, mask_rank_salience))
        tuning_input_salience = (model.bert.encoder.layer[i].intermediate.dense.lora_A * model.bert.encoder.layer[i].intermediate.dense.lora_A.grad).sum(dim=0)
        mask_input_salience = model.bert.encoder.layer[i].intermediate.dense.input_mask * model.bert.encoder.layer[i].intermediate.dense.input_mask.grad
        print("Intermediate input allclose:", torch.allclose(tuning_input_salience, mask_input_salience))
        tuning_rank_salience = (model.bert.encoder.layer[i].intermediate.dense.lora_A * model.bert.encoder.layer[i].intermediate.dense.lora_A.grad).sum(dim=1)
        mask_rank_salience = model.bert.encoder.layer[i].intermediate.dense.bottleneck_mask * model.bert.encoder.layer[i].intermediate.dense.bottleneck_mask.grad
        print("Intermediate rank allclose:", torch.allclose(tuning_rank_salience, mask_rank_salience))