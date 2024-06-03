import seaborn as sns
sns.set_theme(style="darkgrid")
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
import loralib as lora

from copy import deepcopy
from transformers import (HfArgumentParser)
from args import DataTrainingArguments
from models.model_args import ModelArguments
from utils.utils import *
from utils import build_trainer, build_dataloader
from args import MinusTrainingArguments
from models import build_model, convert_layers_based_on_ckpt
from prune import AdapterPruner
from torch.utils.data import Subset
from trainer.param_control import ParamController
from matplotlib import pyplot as plt
from utils.fisher_utils.efficiency.param import *
from utils.minus_utils import compare_module_inputs_equality, load_grafting_masks

def main():
    sys.argv = ['test_pre_tuning_prune.py',
            '--output_dir',
            './output/test_model_grafting_dynamic_all_dependent_pruned_test/',
            '--model_name_or_path',
            'output/debug_running_fisher_dynamic/pre_pruning_model_step811',
            '--task_name',
            'sst2',
            '--do_train',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '32',
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
            '20',
            '--learning_rate',
            '2e-4',
            '--weight_decay',
            '0.1',
            '--warmup_ratio',
            '0.06',
            '--report_to',
            'none',
            '--teacher_param_tuning_config',
            'q:0-11,v:0-11,i:0-11',
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

    model.head_mask, model.intermediate_mask = model.head_mask.to(training_args.device), model.intermediate_mask.to(training_args.device)
    model.hidden_mask = model.hidden_mask.to(training_args.device)
    # trainer.evaluate()

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
    )
    param_controller.convert_to_pre_pruning_lora_teacher()
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    param_controller.model_as_teacher()
    tuning_param_num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    
    trainer.salience_to_be_collected = True
    trainer.salience_collecting_start = 200
    trainer.salience_collecting_end = 1000
    trainer.args.pruner_type = 'running_fisher'
    trainer.param_dynamic_allocation = True
    trainer.args.param_allocation_strategy == 'running_fisher'
    trainer.args.mac_constraint = 0.9
    load_grafting_masks(model, torch.load(os.path.join(model_args.model_name_or_path, 'grafting_masks.pt')))
    model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt'))
    model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt'))
    model.hidden_mask = torch.load(os.path.join(model_args.model_name_or_path, 'hidden_mask.pt'))
    
    pruned_model = deepcopy(model)
    pruned_model_controller = ParamController(
        pruned_model,
        teacher_config=teacher_config,
        student_config=student_config,
        lora_with_bias=False,
        adapter_pruner=adapter_pruner,
    )
    pruned_model.head_mask = pruned_model.head_mask.view(12, -1)
    pruned_model.intermediate_mask = pruned_model.intermediate_mask.view(12, -1)
    pruned_trainer = build_trainer(data_args, training_args, pruned_model, tokenizer, train_dataset, eval_dataset, param_controller=pruned_model_controller)
    pruned_model.prune_model_with_masks()
    pruned_model_controller.adjust_lora_with_masks(0, 2., target='teacher')
    pruned_trainer.evaluate()
    
    model.prune_model_with_masks()
    trainer.evaluate()
    
    load_post_pruned_model = False
    if load_post_pruned_model:
        model_args.model_name_or_path = 'bert-base-uncased'
        _, _, pruned_model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
        loaded_weights = torch.load('output/debug_running_fisher_dynamic/pre_pruning_model_step1423/pytorch_model.bin')
        pruned_model.head_mask = torch.load('output/debug_running_fisher_dynamic/pre_pruning_model_step811/head_mask.pt')
        pruned_model.intermediate_mask = torch.load('output/debug_running_fisher_dynamic/pre_pruning_model_step811/intermediate_mask.pt')
        pruned_model.hidden_mask = torch.load('output/debug_running_fisher_dynamic/pre_pruning_model_step811/hidden_mask.pt')
        # pruned_model.head_mask, pruned_model.intermediate_mask, pruned_model.hidden_mask = model.head_mask, model.intermediate_mask, model.hidden_mask
        pruned_model.prune_model_with_masks()
        
        convert_layers_based_on_ckpt(pruned_model, loaded_weights)
        pruned_model.load_state_dict(loaded_weights, strict=False)
        pruned_param_controller = ParamController(
            pruned_model,
            teacher_config=teacher_config,
            student_config=student_config,
            lora_with_bias=False,
            adapter_pruner=adapter_pruner,
        )
        pruned_param_controller.convert_to_pre_pruning_lora_teacher()
        load_grafting_masks(pruned_model, torch.load(os.path.join('output/debug_running_fisher_dynamic/pre_pruning_model_step1423', 'grafting_masks.pt')))
        
        pruned_trainer = build_trainer(data_args, training_args, pruned_model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
        
        # Compare based on inputs
        inputs = trainer._prepare_inputs(next(iter(dataloader)))
        model = model.eval()
        pruned_model = pruned_model.eval()
        
        for i in range(12):
            with torch.no_grad():
                res = compare_module_inputs_equality([model, pruned_model], inputs, lambda x: x.bert.encoder.layer[i].intermediate)
                print((res[0] - res[1]).abs().mean())
        
        # Compare merged weights
        for m in model.modules():
            if isinstance(m, lora.LoRALayer):
                _ = m.eval()
        
        for m in pruned_model.modules():
            if isinstance(m, lora.LoRALayer):
                _ = m.eval()
                
        for i in range(12):
            print("Query diff: ", (pruned_model.bert.encoder.layer[i].attention.self.query.weight - model.bert.encoder.layer[i].attention.self.query.weight).abs().mean())
            print("Value diff: ", (pruned_model.bert.encoder.layer[i].attention.self.value.weight - model.bert.encoder.layer[i].attention.self.value.weight).abs().mean())
            print("Up diff: ", (pruned_model.bert.encoder.layer[i].intermediate.dense.weight - model.bert.encoder.layer[i].intermediate.dense.weight).abs().mean())
        
        # Comparing mask consistency
        for i in range(12):
            print("Layer ", i)
            print("Query input mask", (model.bert.encoder.layer[i].attention.self.query.input_mask > 0).sum().item(), pruned_model.bert.encoder.layer[i].attention.self.query.input_mask.shape[0])
            print("Query output mask", (model.bert.encoder.layer[i].attention.self.query.output_mask > 0).sum().item(), pruned_model.bert.encoder.layer[i].attention.self.query.output_mask.shape[0])
            print("Value input mask", (model.bert.encoder.layer[i].attention.self.value.input_mask > 0).sum().item(), pruned_model.bert.encoder.layer[i].attention.self.value.input_mask.shape[0])
            print("Value output mask", (model.bert.encoder.layer[i].attention.self.value.output_mask > 0).sum().item(), pruned_model.bert.encoder.layer[i].attention.self.value.output_mask.shape[0])
            print("Up input mask", (model.bert.encoder.layer[i].intermediate.dense.input_mask > 0).sum().item(), pruned_model.bert.encoder.layer[i].intermediate.dense.input_mask.shape[0])
            print("Up output mask", (model.bert.encoder.layer[i].intermediate.dense.output_mask > 0).sum().item(), pruned_model.bert.encoder.layer[i].intermediate.dense.output_mask.shape[0])

    
    trainer.train()
    # Manually intterupt the training after 1000 steps
    
    original_model = deepcopy(model)
    original_model.head_mask = original_model.head_mask.view(12, 12)
    original_model.intermediate_mask = original_model.intermediate_mask.view(12, 3072)
    original_trainer = build_trainer(data_args, training_args, original_model, tokenizer, train_dataset, eval_dataset, param_controller=param_controller)
    

    model.prune_model_with_masks()

    param_controller.adjust_lora_with_masks(trainer.state.global_step, 2., target='teacher')
    
    # Compare based on inputs
    inputs = trainer._prepare_inputs(next(iter(dataloader)))
    model = model.eval()
    original_model = original_model.eval()
    
    for i in range(12):
        with torch.no_grad():
            res = compare_module_inputs_equality([model, original_model], inputs, lambda x: x.bert.encoder.layer[i].attention.output)
            print((res[0] - res[1]).abs().mean())
    
    for m in model.modules():
        if isinstance(m, lora.PruningLinear):
            print((m.input_mask == 0).sum().item(), (m.output_mask == 0).sum().item(), (m.bottleneck_mask == 0).sum().item())
    
    # Compare merged weights
    for m in model.modules():
        if isinstance(m, lora.PruningLinear):
            _ = m.train()
    
    for m in original_model.modules():
        if isinstance(m, lora.PruningLinear):
            _ = m.train()
            
    for i in range(12):
        head_mask = torch.repeat_interleave(original_model.head_mask[i], 64)
        print("Query diff: ", (model.bert.encoder.layer[i].attention.self.query.weight - original_model.bert.encoder.layer[i].attention.self.query.weight[head_mask.nonzero().squeeze()]).abs().mean())
        print("Key diff: ", (model.bert.encoder.layer[i].attention.self.key.weight - original_model.bert.encoder.layer[i].attention.self.key.weight[head_mask.nonzero().squeeze()]).abs().mean())
        print("Up diff: ", (model.bert.encoder.layer[i].intermediate.dense.weight - original_model.bert.encoder.layer[i].intermediate.dense.weight[original_model.intermediate_mask[i].nonzero().squeeze()]).abs().mean())
    
    dataloader = trainer.get_train_dataloader()
    for step, inputs in enumerate(dataloader):
        tr_loss_step = trainer.training_step(model, inputs)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), trainer.args.max_grad_norm)
        
        # Optimizer step
        trainer.optimizer.step()
        trainer.lr_scheduler.step()
        model.zero_grad()
        if step == 100:
            break
        
    print("Salience collected length:", len(trainer.salience_collected))
    
    head_grad_change = torch.stack([salience['head_mask_grad'].pow(2) for salience in trainer.salience_collected])
    plt.clf()
    plt.plot(head_grad_change.view(head_grad_change.shape[0], -1).cpu().numpy())
    plt.savefig('head_grad_change.png')
    
    head_mask_change = torch.stack([salience['head_mask'] for salience in trainer.salience_collected])
    plt.clf()
    plt.plot(head_mask_change.cpu().numpy())
    plt.savefig('head_mask_change.png')

    hidden_mask_change = torch.stack([salience['hidden_mask'] for salience in trainer.salience_collected])
    plt.clf()
    plt.plot(hidden_mask_change.cpu().numpy())
    plt.savefig('hidden_mask_change.png')
    
    head_mask_var, head_mask_mean = head_mask_change[200:].var(dim=0), head_mask_change[200:].mean(dim=0)
    const_prune_heads = ((head_mask_var < 0.1) & (head_mask_mean < 0.1))
    
    num_param_per_head = param_per_head(768, 64)
    num_param_per_neuron = param_per_neuron(768)
    num_param_per_hidden = param_per_hidden_dim(12 * [64 * 12], 12 * [768])

    salience_density = {}
    salience_density['head_mask'] = trainer.salience_acc['head_mask'] / num_param_per_head
    salience_density['hidden_mask'] = trainer.salience_acc['hidden_mask'] / num_param_per_hidden
    salience_density['intermediate_mask'] = trainer.salience_acc['intermediate_mask'] / num_param_per_neuron
    
    current_target = 0.4 ** 0.5
    densities = torch.cat([salience_density['head_mask'].view(-1), salience_density['hidden_mask'].view(-1), salience_density['intermediate_mask'].view(-1)])
    total_mask = torch.zeros_like(densities)
    sorted_densities, sorted_indices = torch.sort(densities, descending=True)
    total_mask[sorted_indices[:int(current_target * len(densities))]] = 1
    head_mask, intermediate_mask, hidden_mask = torch.split(total_mask, [len(salience_density['head_mask'].view(-1)), len(salience_density['intermediate_mask'].view(-1)), len(salience_density['hidden_mask'].view(-1))])
    head_mask = head_mask.view(12, 12)
    intermediate_mask = intermediate_mask.view(12, 3072)
    

if __name__ == '__main__':
    main()