# Test ElasticLlama model with pruning consistency and tuning consistency
import torch
import os
import sys
import gc
import torch.nn as nn
import numpy as np
import seaborn as sns
import loralib as lora

from transformers import (HfArgumentParser)
from args import InstructionDataTrainingArguments
from models.model_args import ModelArguments
from args import MinusTrainingArguments
from utils.utils import *
from utils import build_dataloader
from models import build_model
from trainer.param_control import ParamController
from trainer.trainer_minus import MinusTrainer
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils.minus_utils import kurtosis

def main():
    sys.argv = ['test_llama_kurtosis.py',
            '--output_dir',
            './output/test_llama_virtual_pruning/',
            '--model_name_or_path',
            'meta-llama/Llama-2-7b-hf',
            '--do_train',
            '--task_name',
            'alpaca',
            '--data_path', 
            'data/sft/alpaca_data_gpt4.json',
            '--bf16',
            'True',
            '--output_dir',
            'output/llama_lora_alpaca/epoch_30',
            '--num_train_epochs',
            '30',
            '--per_device_train_batch_size',
            '4',
            '--per_device_eval_batch_size',
            '4',
            '--gradient_accumulation_steps',
            '8',
            '--evaluation_strategy',
            "no",
            '--save_strategy',
            "steps",
            '--save_steps',
            '2000',
            '--save_total_limit',
            '1',
            '--learning_rate',
            '2e-4', # LoRA learning rate
            '--weight_decay',
            '0.',
            '--warmup_ratio',
            '0.03',
            '--lr_scheduler_type',
            "cosine",
            '--logging_steps',
            '1',
            '--apply_lora',
            '--do_distill',
            '--lora_r',
            '8',
            '--lora_alpha',
            '16',
            '--report_to',
            'none',
            # '--fsdp',
            # "full_shard auto_wrap",
            # '--fsdp_transformer_layer_cls_to_wrap',
            # 'LlamaDecoderLayer',
            '--tf32',
            'True',
            '--pruner_type',
            'running_fisher',
            '--pre_tuning_scorer',
            'backward_running_hidden_states_salience',
            '--pre_tuning_constraint',
            '0.8',
            '--pre_tuning_pruner',
            'running_fisher',
            ]
    parser = HfArgumentParser(
        (ModelArguments, InstructionDataTrainingArguments, MinusTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)
    # training_args.disable_tqdm = False
    config, tokenizer, model = build_model(model_args, data_args, training_args, token=os.environ.get('HF_TOKEN', None))
    train_dataset, eval_dataset, _, _ = build_seq2seq_data(data_args, training_args, tokenizer)
    dataloader = build_dataloader(train_dataset, training_args.per_device_train_batch_size, data_args, training_args, tokenizer)
    
    inputs = next(iter(dataloader))
    model = model.to(training_args.device)
    inputs = {k: v.to(training_args.device) for k, v in inputs.items()}
    
    model.head_mask = model.head_mask.to(training_args.device)
    model.intermediate_mask = model.intermediate_mask.to(training_args.device)
    model.hidden_mask = model.hidden_mask.to(training_args.device)
    
    
    teacher_keys = ['dec_self_query', 'dec_self_value']
    teacher_config = {
        k: [i for i in range(config.num_hidden_layers)] for k in teacher_keys
    }
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        lora_with_bias=False,
    )
        
    @dataclass
    class DataCollatorForSupervisedDataset(object):
        """Collate examples for supervised fine-tuning."""

        tokenizer: transformers.PreTrainedTokenizer

        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )
    data_collator = DataCollatorForSupervisedDataset(tokenizer)
    
    trainer = MinusTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        teacher_model=None,
        param_controller=param_controller,
        seq_len=512,
        cls_task=False,
    )
    param_controller.convert_to_pruning_lora_teacher()
    param_controller.model_as_teacher()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(sum(p.numel() for p in model.parameters()))
    
    scorer = trainer.pre_tuning_pruning_scorer
    scorer.log_kurtosis = True
    pruner = trainer.pre_tuning_pruning_pruner
    
    current_mha_states = []
    forward_mha_states = []
    all_mha_kurtosis = []
    all_head_scores = []
    attention_head_size = model.config.hidden_size // model.config.num_attention_heads
    
    def cache_mha_states(module: nn.Module, layer_inputs, outputs):
        # print("mha forward hook fired")
        mha_hidden_states = layer_inputs[0]
        if mha_hidden_states is not None and mha_hidden_states.requires_grad:
            # print("mha state is not None and not zero")
            with torch.no_grad():
                mha_state = mha_hidden_states.abs().sum(dim=0).sum(dim=0)
                current_mha_states.append(mha_state)
                # mha_hidden_states shape: (batch_size, seq_len, num_heads x head_size)
                # module weight shape: (hidden_size, num_heads x head_size)
                mha_hidden_unsqueezed = mha_hidden_states.mean(0).mean(0).unsqueeze(0) # (1, num_heads x head_size)
                mha_hidden_unsqueezed = mha_hidden_unsqueezed.view(1, -1, attention_head_size) # Convert to (1, num_heads, head_size)
                
                weight_merged = (module.weight + (module.lora_B @ module.lora_A) * module.scaling) if isinstance(module, lora.Linear) else module.weight # shape: (hidden_size, num_heads x head_size)
                weight_unsqueezed = weight_merged.view(weight_merged.shape[0], -1, attention_head_size) # Convert to (hidden_size, num_heads, head_size)
                activation = (mha_hidden_unsqueezed * weight_unsqueezed) # shape: (hidden size, num_heads, head_size) 
                forward_mha_states.append(activation.permute(1, 2, 0).reshape(-1, activation.shape[1]))
                act_kurtosis = kurtosis(activation.permute(1, 2, 0).reshape(-1, activation.shape[1]))
                all_mha_kurtosis.append(act_kurtosis)
    
    def calculate_mha_score(module: nn.Module, grad_layer_inputs, grad_outputs):
        # print("mha backward hook fired")
        mha_states = current_mha_states.pop()
        while not isinstance(mha_states, torch.Tensor):
            all_head_scores.append(mha_states[1])
            mha_states = current_mha_states.pop()
        mha_states_grad = grad_layer_inputs[0]
        with torch.no_grad():
            salience = (mha_states * mha_states_grad.abs().sum(dim=0).sum(dim=0)) if mha_states_grad is not None else torch.zeros_like(mha_states)
            head_salience = salience.view(-1, attention_head_size).sum(dim=1)
            all_head_scores.append(head_salience)
    
    mha_handlers = []
    
    num_layers = model.config.num_hidden_layers
    for layer in range(num_layers):
        mha_layer = param_controller.get_layer(layer, 'dec_self_output')
        if mha_layer is not None:
            mha_handler = mha_layer.register_forward_hook(cache_mha_states)
            mha_handlers.append(mha_handler)
            mha_handler = mha_layer.register_full_backward_hook(calculate_mha_score)
            mha_handlers.append(mha_handler)
    
    inputs = next(iter(dataloader))
    inputs = {k: v.to(training_args.device) for k, v in inputs.items()}
    outputs = model(**inputs, use_cache=False)
    
    plt.clf()
    plt.figure(figsize=(10, 2))
    sns.kdeplot(forward_mha_states[0][:, 0].detach().float().cpu().numpy(), multiple="stack")
    plt.yscale('log')
    plt.savefig("forward_state.png")
    
    plt.clf()
    dist1 = np.concatenate([np.random.normal(50, 10, 470000), np.random.normal(150, 5, 30000)])
    # Plotting the data
    plt.figure(figsize=(10, 1))
    # KDE plot for the distribution with outliers
    sns.kdeplot(dist1, shade=True)
    plt.axis('off')
    plt.savefig("kde.pdf", bbox_inches="tight", pad_inches=0)
    
    outputs[0].backward()
    
    mha_saliences = torch.cat(scorer.all_head_scores)
    mha_kurtosis = torch.cat(scorer.all_mha_kurtosis)
    # Plotting the scatter plot
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=mha_saliences.cpu().float().log10().numpy(), y=mha_kurtosis.cpu().float().log10().numpy())
    plt.xlabel("MHA salience")
    plt.ylabel("MHA kurtosis")
    plt.savefig("mha_kurtosis.png")
    plt.clf()