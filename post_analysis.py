import os
import re
import sys
import json
import seaborn as sns
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from matplotlib import pyplot as plt
# from transformers.file_utils import hf_bucket_url, cached_path
from transformers import HfArgumentParser, TrainingArguments
from transformers.trainer import Trainer
from torch.utils.data import DataLoader, RandomSampler

from args import DataTrainingArguments
from utils.minus_utils import flatten_states
from utils.analysis_utils import get_pruned_and_retained
from utils.utils import build_data, get_raw_datasets
from trainer.model_arch import get_layers
from utils.cofi_utils import update_params, prune_model_with_z
from utils.minus_utils import compare_module_inputs_equality
from models import build_model
from models.model_args import ModelArguments
from models.modeling_bert import CoFiBertForSequenceClassification
from utils.fisher_utils.efficiency.mac import *

logger = logging.getLogger(__name__)

def get_all_training_results(dir):
    data_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file == "train_results.json":
                data_files.append(os.path.join(root, file))
    data = {
        '/'.join(f.split('/')[-4:-1]).replace('/batchuse64/', ''): json.load(open(f, 'r'))
        for f in data_files
    }
    return data

def get_cofi_weights(pretrained_model_name_or_path):
    if os.path.exists(pretrained_model_name_or_path):
        weights = torch.load(os.path.join(pretrained_model_name_or_path, "pytorch_model.bin"), map_location=torch.device("cpu"))
    else:
        archive_file = hf_bucket_url(pretrained_model_name_or_path, filename="pytorch_model.bin") 
        resolved_archive_file = cached_path(archive_file)
        weights = torch.load(resolved_archive_file, map_location="cpu")
    # Convert old format to new format if needed from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in weights.keys():
        new_key = None
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        weights[new_key] = weights.pop(old_key)
    return weights


def get_cofi_zs(config, weights):
    zs = {}
    dim_per_head = config.hidden_size // config.num_attention_heads
    architecture = config.architectures[0].lower()
    bert_name = "roberta" if "roberta" in architecture else "bert"

    hidden_z = torch.zeros(config.hidden_size)
    hidden_z[:weights[f"{bert_name}.embeddings.word_embeddings.weight"].shape[1]] = 1
    zs["hidden_z"] = hidden_z

    head_z = torch.zeros(config.num_hidden_layers, config.num_attention_heads)
    head_layer_z = torch.zeros(config.num_hidden_layers)
    for i in range(config.num_hidden_layers):
        key = f"{bert_name}.encoder.layer.{i}.attention.output.dense.weight"
        if key in weights:
            remaining_heads = weights[key].shape[-1] // dim_per_head
            head_z[i, :remaining_heads] = 1
            head_layer_z[i] = 1
    zs["head_z"] = head_z
    zs["head_layer_z"] = head_layer_z

    int_z = torch.zeros(config.num_hidden_layers, config.intermediate_size)
    mlp_z = torch.zeros(config.num_hidden_layers)
    for i in range(config.num_hidden_layers):
        key = f"bert.encoder.layer.{i}.output.dense.weight"
        if key in weights:
            remaining_int_dims = weights[key].shape[-1]
            int_z[i, :remaining_int_dims] = 1
            mlp_z[i] = 1
    zs["intermediate_z"] = int_z
    zs["mlp_z"] = mlp_z
    return zs


def compare_main():
    sys.argv = ['post_analysis.py',
            '--output_dir',
            './output/roberta_lora_minus_mnli/freq0.1/batchuse64/mac0.6/',
            '--model_name_or_path',
            './output/roberta_lora_minus_mnli/freq0.1/batchuse64/mac0.6/',
            '--task_name',
            'mnli',
            # '--do_train',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '32',
            '--per_device_eval_batch_size',
            '32',
            '--apply_lora',
            ]
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    IS_SQUAD = 'squad' in data_args.task_name
    t_name, raw_datasets = get_raw_datasets(model_args, data_args, training_args)
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    config, tokenizer, new_model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    train_dataset, eval_dataset, predict_dataset, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        sampler=RandomSampler(eval_dataset, replacement=True, num_samples=training_args.per_device_eval_batch_size*50), # prevent from killing because of memory out of usage
    )

    if os.path.exists(os.path.join(model_args.model_name_or_path, 'head_mask.pt')):
        model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt'))
    if os.path.exists(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt')):
        model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt'))

    zs = {
        'head_z': model.head_mask,
        'intermediate_z': model.intermediate_mask,
    }
    update_params(model, zs)
    prune_model_with_z(zs, model)
    model.head_mask = None
    model.intermediate_mask = None
    model = model.to(training_args.device)

    if os.path.exists(os.path.join(model_args.model_name_or_path, 'head_mask.pt')):
        new_model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt'))
    if os.path.exists(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt')):
        new_model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt'))

    new_model = new_model.to(training_args.device)
    new_model.head_mask = new_model.head_mask.to(training_args.device)
    new_model.intermediate_mask = new_model.intermediate_mask.to(training_args.device)
    
    inputs = next(iter(eval_dataloader))
    for i in range(model.config.num_hidden_layers):
        layer_func = lambda x: get_layers(x)[i]
        result = compare_module_inputs_equality([model, new_model], inputs, layer_func)
        print("Layer %d:" % i, "the biggest difference is", (result[0] - result[1]).abs().max().item())
    module_func = lambda x: get_layers(x)[0].output.dense
    result = compare_module_inputs_equality([model, new_model], inputs, module_func)
    hidden_states = module_func(model)(result[0])
    new_hidden_states = module_func(new_model)(result[1])
    print("Pre dense-layer hidden_states equality:",
        (result[1][:,:, new_model.intermediate_mask[0].nonzero().T.squeeze()] == result[0]).all().item()
    )
    print("Post dense-layer hidden states equality:", (hidden_states == new_hidden_states).all().item())
    print("Post dense-layer hidden states equality (with float64):", (module_func(model).double()(result[0].double()) == module_func(new_model).double()(result[1].double())).all().item())
    
    original_weights = get_layers(model)[0].output.dense.weight
    selected_weights = get_layers(new_model)[0].output.dense.weight[:, new_model.intermediate_mask[0].nonzero().T.squeeze()]
    post_dense_states = result[0].matmul(original_weights.T) + get_layers(model)[0].output.dense.bias
    other_post_dense_states = result[0].matmul(selected_weights.T) + get_layers(model)[0].output.dense.bias
    print("Layer forward equal to pruned-weight-matmul:", ((post_dense_states == hidden_states).all().item()))
    print("Layer forward equal to selected-weight-matmul:", ((other_post_dense_states == hidden_states).all().item()))
    
    original_post_dense_states = result[1].matmul(get_layers(new_model)[0].output.dense.weight.T) + get_layers(new_model)[0].output.dense.bias
    select_states_original_layer = result[1][:,:, new_model.intermediate_mask[0].nonzero().T.squeeze()].matmul(get_layers(new_model)[0].output.dense.weight[:, new_model.intermediate_mask[0].nonzero().T.squeeze()].T) + get_layers(new_model)[0].output.dense.bias
    zeroed_layer_weight = get_layers(new_model)[0].output.dense.weight.clone()
    zeroed_layer_weight[:,(new_model.intermediate_mask[0] == 0).nonzero().T.squeeze()] = 0
    new_output_states = result[1].matmul(zeroed_layer_weight.T) + get_layers(new_model)[0].output.dense.bias
    

def get_results(folder):
    data = {}
    if os.path.exists(os.path.join(folder, 'train_results.json')):
        data = {
            **data,
            **json.load(open(os.path.join(folder, 'train_results.json'), 'r'))
        }
    if os.path.exists(os.path.join(folder, 'best_model', 'eval_results.json')):
        data = {
            **data,
            **json.load(open(os.path.join(folder, 'best_model', 'eval_results.json'), 'r'))
        }
    elif os.path.exists(os.path.join(folder, 'eval_results.json')):
        data = {
            **data,
            **json.load(open(os.path.join(folder, 'eval_results.json'), 'r'))
        }
    if os.path.exists(os.path.join(folder, 'efficiency_results.json')):
        data = {
            **data,
            **json.load(open(os.path.join(folder, 'efficiency_results.json')))
        }
    elif os.path.exists(os.path.join(folder, 'best_model', 'efficiency_results.json')):
        data = {
            **data,
            **json.load(open(os.path.join(folder, 'best_model', 'efficiency_results.json')))
        }
    if os.path.exists(os.path.join(folder, 'trainer_state.json')):
        training_mem = max([v['end_mem'] for v in json.load(open(os.path.join(folder, 'trainer_state.json'), 'r'))['log_history'] if 'end_mem' in v])
        data['training_mem'] = training_mem
    return data

def get_torch_saved_dirs(root: str, searching_fn: str = 'trainer_state.json') -> List[str]:
    trained_output_folders = []
    for root, _, files in os.walk(root):
        for f in files:
            if f == searching_fn and 'checkpoint' not in root and 'finetuned' not in root:
                trained_output_folders.append(root)
    return trained_output_folders

def gather_model_infos(output_home_dir: str) -> pd.DataFrame:
    trained_output_folders = get_torch_saved_dirs(output_home_dir)
    hyperparameter_regex = r'mac(\d+\.*\d+)/lora_r(\d+)/lora_alpha(\d+)'
    collected_data = []
    for folder in trained_output_folders:
        search_result = re.search(hyperparameter_regex, folder)
        if search_result is None:
            data = {
                'frequency' : None,
                'batch_use': None,
                'mac_constraint': 1,
            }
        else:
            data = {
                'mac_constraint': float(search_result.group(1)),
                'lora_r': int(search_result.group(2)),
                'lora_alpha': int(search_result.group(3)),
            }
        data = {
            **data,
            **get_results(folder)
        }
        if os.path.exists(os.path.join(folder, 'final_head_mask.pt')):
            head_mask = torch.load(os.path.join(folder, 'final_head_mask.pt'), map_location='cpu')
            data['attn_density'] = (head_mask != 0).sum().item() / head_mask.numel()
        else:
            data['attn_density'] = 1
        if os.path.exists(os.path.join(folder, 'final_intermediate_mask.pt')):
            intermediate_mask = torch.load(os.path.join(folder, 'final_intermediate_mask.pt'), map_location='cpu')
            data['ffn_density'] = (intermediate_mask != 0).sum().item() / intermediate_mask.numel()
        else:
            data['ffn_density'] = 1
        collected_data.append(data)
    return pd.DataFrame(collected_data)


def get_relative_metrics(baseline_dir, output_root_dir):
    df = gather_model_infos(output_root_dir)
    baseline_metrics = get_results(baseline_dir)
    # lora_df = gather_model_infos('output/roberta-base_lora_mnli/epoch5/lora_r8/lora_alpha16')
    # df = pd.concat([df, lora_df], ignore_index=True)
    baseline_traintime, baseline_inftime, baseline_memusage, baseline_trainingmem = baseline_metrics['train_runtime'], baseline_metrics['bz128_t_mean'], baseline_metrics['bz128_em_mean'], baseline_metrics['training_mem']
    df['relative_training_runtime'] = df['train_runtime'] / baseline_traintime
    df['training_speedup'] = 1 / df['relative_training_runtime']
    df['relative_inference_runtime'] = df['bz128_t_mean'] / baseline_inftime
    df['inference_speedup'] = 1 / df['relative_inference_runtime']
    df['relative_eval_memory_usage'] = df['bz128_em_mean'] / baseline_memusage
    df['relative_training_memory_usage'] = df['training_mem'] / baseline_trainingmem
    new_df = pd.DataFrame()
    new_df['accuracy'] =  df['eval_accuracy'].tolist() * 6
    new_df['metric'] = df['training_speedup'].tolist() + df['inference_speedup'].tolist() + df['relative_eval_memory_usage'].tolist() + df['relative_training_memory_usage'].tolist() + df['attn_density'].tolist() + df['ffn_density'].tolist()
    
    new_df['metric_type'] = ['training_speedup'] * len(df) + ['inference_speedup'] * len(df) + ['eval_memory_usage'] * len(df) + ['training_memory_usage'] * len(df) + ['attn_density'] * len(df) + ['ffn_density'] * len(df)
    new_df['lora_r'] = df['lora_r'].tolist() * 6
    new_df['mac_constraint'] = df['mac_constraint'].tolist() * 6
    return df, new_df


def get_all_metrics(baseline_dir, output_root_dir):
    df = gather_model_infos(output_root_dir)
    lora_df = gather_model_infos('output/roberta_lora_mnli')
    df = pd.concat([df, lora_df], ignore_index=True)
    baseline_metrics = get_results(baseline_dir)
    baseline_traintime, baseline_inftime, baseline_memusage = baseline_metrics['train_runtime'], baseline_metrics['t_mean'], baseline_metrics['em_mean']
    df['relative_training_runtime'] = df['train_runtime'] / baseline_traintime
    df['training_speedup'] = 1 / df['relative_training_runtime']
    df['relative_inference_runtime'] = df['t_mean'] / baseline_inftime
    df['inference_speedup'] = 1 / df['relative_inference_runtime']
    df['relative_eval_memory_usage'] = df['em_mean'] / baseline_memusage
    new_df = pd.DataFrame()
    new_df['metric'] = df['training_speedup'].tolist() + df['inference_speedup'].tolist() + (df['eval_accuracy'] / baseline_metrics['eval_accuracy']).tolist() + df['relative_eval_memory_usage'].tolist()
    new_df['metric_type'] = ['training'] * len(df) + ['inference'] * len(df) + ['accuracy'] * len(df) + ['memory_usage'] * len(df)
    parameter_name = 'frequency' if 'frequency' in df.columns else 'steppoint'
    new_df[parameter_name] = df[parameter_name].tolist() * 4
    new_df['mac_constraint'] = df['mac_constraint'].tolist() * 4
    new_df['attn_density'] = df['attn_density'].tolist() * 4
    new_df['ffn_density'] = df['ffn_density'].tolist() * 4
    return df, new_df



def get_mask_consistency(folder, training_samples=392702, epochs=3, batch_size=32, saving_steps=500, pruning_freq=0.5):
    batches_num = training_samples // batch_size + 1
    batches_total = batches_num * epochs
    pruning_steps = int(pruning_freq * batches_num)
    pruning_occurring_steps = np.arange(pruning_steps, batches_total, pruning_steps)
    saved_pruning_occurring_steps = [
        ((step // saving_steps) * saving_steps, (step // saving_steps + 1) * saving_steps)
        for step in pruning_occurring_steps
    ]
    head_masks = [os.path.join(root, f) for root, _, files in os.walk(folder) for f in files if 'checkpoint' in root and 'head_mask' in f]
    intermediate_masks = [os.path.join(root, f) for root, _, files in os.walk(folder) for f in files if 'checkpoint' in root and 'intermediate_mask' in f]
    head_masks = {
        int(re.search(r'checkpoint-(\d+)/', f)[1]): torch.load(f).to('cpu')
        for f in tqdm(head_masks)
    }
    intermediate_masks = {
        int(re.search(r'checkpoint-(\d+)/', f)[1]): torch.load(f).to('cpu')
        for f in tqdm(intermediate_masks)
    }
    head_mask_pruning_changed_num = [
        (head_masks[step[0]] - head_masks[step[1]]).abs().sum().item()
        for step in saved_pruning_occurring_steps
    ]
    intermediate_mask_pruning_changed_num = [
        (intermediate_masks[step[0]] - intermediate_masks[step[1]]).abs().sum().item()
        for step in saved_pruning_occurring_steps
    ]
    return head_mask_pruning_changed_num, intermediate_mask_pruning_changed_num

def get_training_speed_each_saving_step(folder, saving_step=500):
    data = json.load(open(os.path.join(folder, 'trainer_state.json')))
    log_history = data['log_history']
    times = [v['training_time'] if 'training_time' in v else v['train_runtime'] for v in log_history]
    times = [times[0] if i == 0 else times[i] - times[i-1] for i in range(len(times))]
    speed = [saving_step / training_time for training_time in times]
    return speed


def get_outputs_on_dataset(model, tokenizer, dataset, batch_size=32):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
    )
    output_data = {
        "input_sentence": [],
        "label": [],
        "prediction": [],
    }
    for batch in tqdm(dataloader):
        batch = {
            k: torch.stack(v, dim=1).to(model.device) if isinstance(v, list) else v.to(model.device)
            for k, v in batch.items()
        }
        output = model(**batch)
        predictions = output['logits'].argmax(dim=-1)
        for input_ids, attention_mask in zip(batch['input_ids'], batch['attention_mask']):
            output_data['input_sentence'].append(tokenizer.decode(input_ids[:sum(attention_mask)]))
        output_data['label'] += batch['labels'].tolist()
        output_data['prediction'] += predictions.tolist()
    return pd.DataFrame(output_data)


def gather_trainer_states(root: str) -> List[str]:
    trainer_state_files = []
    for r, _, files in os.walk(root):
        for f in files:
            if ('checkpoint' not in r) and (f == 'trainer_state.json'):
                trainer_state_files.append(os.path.join(r, f))
    return trainer_state_files


def plot_loss_and_lr(files: List[str], types: List[str]):
    all_data = {
        'epochs': [],
        'losses': [],
        'lr': [],
        'type': [],
    }
    for i, f in enumerate(files):
        data = json.load(open(f))
        epochs = [v['epoch'] for v in data['log_history'] if 'loss' in v]
        all_data['epochs'] += epochs
        all_data['losses'] += [v['loss'] for v in data['log_history'] if 'loss' in v]
        all_data['lr'] += [v['learning_rate'] for v in data['log_history'] if 'learning_rate' in v]
        all_data['type'] += [types[i]] * len(epochs)
    df = pd.DataFrame(all_data)
    fig, ax =plt.subplots(2, 1, figsize=(16, 18))
    sns.lineplot(data=df, x='epochs', y='losses', hue='type', ax=ax[0])
    sns.lineplot(data=df, x='epochs', y='lr', hue='type', ax=ax[1])
    plt.savefig('loss_lr_compared.png')
    plt.clf()
    return True

def load_gen_mask_with_restoration(root_dir):
    dirs = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if '_step' in f]
    masks = []
    calc_mac_per_head = mac_per_head(128, 768, 768 // 12)
    calc_mac_per_neuron = mac_per_neuron(128, 768)
    total_mac = compute_mac([12] * 12, [3072] * 12, 128, 768, 768 // 12)
    param_per_head = 768 // 12 * 768 * 4
    param_per_neuron = 768 * 2
    for d in dirs:
        step_num = int(d.split('_step')[-1])
        head_mask = torch.load(os.path.join(d, 'head_mask.pt'), map_location='cpu')
        intermediate_mask = torch.load(os.path.join(d, 'intermediate_mask.pt'), map_location='cpu')
        head_density = head_mask.sum().item() / head_mask.numel()
        intermediate_density = intermediate_mask.sum().item() / intermediate_mask.numel()
        overall_density = (head_mask.sum().item() * param_per_head + intermediate_mask.sum().item() * param_per_neuron) / (head_mask.numel() * param_per_head + intermediate_mask.numel() * param_per_neuron)
        current_mac = head_mask.sum().item() * calc_mac_per_head + intermediate_mask.sum().item() * calc_mac_per_neuron
        mac_ratio = current_mac / total_mac
        masks.append({
            'step': step_num,
            'head_mask': head_mask,
            'intermediate_mask': intermediate_mask,
            'head_density': head_density,
            'intermediate_density': intermediate_density,
            'overall_density': overall_density,
            'mac_ratio': mac_ratio,
        })
    return sorted(masks, key=lambda x: x['step'], reverse=False)

if __name__ == '__main__':
    sys.argv = ['cofi_diff.py',
                '--output_dir',
                './output',
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
                ]
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    t_name, raw_datasets = get_raw_datasets(model_args, data_args, training_args)
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    train_dataset, eval_dataset, predict_dataset, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets)

    os.makedirs(training_args.output_dir, exist_ok=True)
    cofi_model = CoFiBertForSequenceClassification.from_pretrained("princeton-nlp/CoFi-MNLI-s95")
    bert_model = CoFiBertForSequenceClassification.from_pretrained("/data/zbw/delta/out-test/MNLI/MNLI_bert-mnli/")
    zs = get_cofi_zs(cofi_model.config, get_cofi_weights("princeton-nlp/CoFi-MNLI-s95"))

    trainer = Trainer(model)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        sampler=RandomSampler(eval_dataset, replacement=True, num_samples=training_args.per_device_eval_batch_size*50), # prevent from killing because of memory out of usage
    )
    cofi_model.eval()
    bert_model.eval()
    bert_model.to('cuda')
    input_keys = [
        'input_ids',
        'token_type_ids',
        'attention_mask',
        'position_ids',
    ]
    
    with torch.no_grad():
        hiddens_by_batch = []
        for i, inputs in enumerate(tqdm(eval_dataloader)):
            inputs = trainer._prepare_inputs(inputs)
            labels = inputs['labels']
            hiddens = get_pruned_and_retained(bert_model, inputs, zs)
            hiddens_by_batch.append(hiddens)

    concatenated_hiddens = {
        k: [torch.cat([
            item[k][i]
            for item in hiddens_by_batch
        ], dim=0) for i in range(len(hiddens_by_batch[0][k]))]
        for k in hiddens_by_batch[0].keys() if 'mask' not in k
    }
    concatenated_hiddens['mask'] = torch.cat([v['mask'] for v in hiddens_by_batch], dim=0)
    # Using the first-layer hidden states as the test case
    flattened_pruned_layer_hiddens = flatten_states(concatenated_hiddens['pruned_layer_hiddens'][0], concatenated_hiddens['mask'])
    flattened_retained_layer_hiddens = flatten_states(concatenated_hiddens['retained_layer_hiddens'][0], concatenated_hiddens['mask'])
    
    func_to_test = {
        'var': lambda x: x.var().item(),
        
    }
    hidden_keys = ['pruned_embedding', 'retained_embedding', 'pruned_layer_hiddens', 'retained_layer_hiddens', 'pruned_intermediate_states', 'retained_intermediate_states']
    state_keys = ['embedding', 'layer_hiddens', 'intermediate_states']
    pruned_vars, retained_vars = {k: [] for k in state_keys}, {k: [] for k in state_keys}
    for k in hidden_keys:
        for layer in tqdm(range(len(concatenated_hiddens[k]))):
            if 'pruned' in k:
                pruned_vars['_'.join(k.split('_')[1:])].append([func_to_test['var'](v) for v in flatten_states(concatenated_hiddens[k][layer], concatenated_hiddens['mask'])])
            else:
                retained_vars['_'.join(k.split('_')[1:])].append([func_to_test['var'](v) for v in flatten_states(concatenated_hiddens[k][layer], concatenated_hiddens['mask'])])
    
    for k in pruned_vars.keys():
        if len(pruned_vars[k]) == 12:
            fig, axs = plt.subplots(nrows=3, ncols=4,figsize=(16, 12))
            for layer in range(12):
                pruned, retained = list(filter(lambda x: x < 2, pruned_vars[k][layer])), list(filter(lambda x: x < 2, retained_vars[k][layer]))
                axs[layer // 4, layer % 4].violinplot([t for t in [pruned, retained] if len(t)], showmeans=True, showextrema=True, showmedians=True)
                axs[layer // 4, layer % 4].set_xticks(np.arange(1, 3 if len(pruned) and len(retained) else 2), labels=['pruned', 'retained'] if len(pruned) and len(retained) else ['pruned'] if len(pruned) else ['retained'])
                axs[layer // 4, layer % 4].set_title(f'layer {layer}')
            fig.suptitle(f"{k} variance")
            fig.savefig(f"{k}_variance.png")
            plt.clf()
        elif len(pruned_vars[k]) == 1:
            fig = plt.figure()
            pruned, retained = list(filter(lambda x: x < 2, pruned_vars[k][0])), list(filter(lambda x: x < 2, retained_vars[k][0]))
            plt.violinplot([pruned, retained], showmeans=True, showextrema=True, showmedians=True)
            ax = plt.gca()
            ax.set_xticks(np.arange(1, 3 if len(pruned) and len(retained) else 2), labels=['pruned', 'retained'] if len(pruned) and len(retained) else ['pruned'] if len(pruned) else ['retained'])
            fig.suptitle(f"{k} variance")
            fig.savefig(f"{k}_variance.png")
    