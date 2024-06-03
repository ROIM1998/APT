#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import sys
import copy
import logging
import torch
import transformers
import loralib as lora

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import utils.alpaca_utils as utils
from torch.utils.data import Dataset
from transformers import Trainer
from tqdm import tqdm

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)
        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    sys.argv = [
        'train.py',
        '--model_name_or_path',
        'meta-llama/Llama-2-7b-hf',
        '--data_path', 
        'data/sft/alpaca_data_gpt4.json',
        '--bf16',
        'True',
        '--output_dir',
        'output/llama2_lora_alpaca_qkv-gated/epoch_5',
        '--num_train_epochs',
        '5',
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
        # '--fsdp',
        # "full_shard auto_wrap",
        # '--fsdp_transformer_layer_cls_to_wrap',
        # 'LlamaDecoderLayer',
        '--tf32',
        'True',
    ]
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        token=os.environ.get('HF_TOKEN', None),
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        token=os.environ.get('HF_TOKEN', None),
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # inputs = data_module['train_dataset'][0]
    # inputs['input_ids'] = inputs['input_ids'].unsqueeze(0)
    # inputs['labels'] = inputs['labels'].unsqueeze(0)
    # inputs = {k: v.to(training_args.device) for k, v in inputs.items()}
    
    # with torch.no_grad():
    #     outputs = model.generate(input_ids=inputs['input_ids'][:, :45], use_cache=False, do_sample=False, num_beams=1, max_length=100)
                
    # Converting q, v, and gate to LoRA layers
    for i in tqdm(range(model.config.num_hidden_layers)):
        q, k = model.model.layers[i].self_attn.q_proj, model.model.layers[i].self_attn.k_proj
        q_lora = lora.Linear(q.in_features, q.out_features, bias=q.bias is not None, r=8, lora_alpha=16, lora_dropout=0.05)
        k_lora = lora.Linear(k.in_features, k.out_features, bias=k.bias is not None, r=8, lora_alpha=16, lora_dropout=0.05)
        q_lora.weight.data = q.weight.data
        k_lora.weight.data = k.weight.data
        del q.weight
        del k.weight
        if q.bias is not None:
            q_lora.bias.data = q.bias.data
            k_lora.bias.data = k.bias.data
            del q.bias
            del k.bias
        model.model.layers[i].self_attn.q_proj = q_lora
        model.model.layers[i].self_attn.k_proj = k_lora
        
        gate = model.model.layers[i].mlp.gate_proj
        gate_lora = lora.Linear(gate.in_features, gate.out_features, bias=gate.bias is not None, r=8, lora_alpha=16, lora_dropout=0.05)
        gate_lora.weight.data = gate.weight.data
        del gate.weight
        if gate.bias is not None:
            gate_lora.bias.data = gate.bias.data
            del gate.bias
        model.model.layers[i].mlp.gate_proj = gate_lora
        
    for n, p in model.named_parameters():
        if 'lora' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
    
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    print("Total model parameters:", sum(p.numel() for p in model.parameters()))
    print("Tuning model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    print("Max memory used:", torch.cuda.max_memory_allocated() / 1024 ** 2, "MB")


if __name__ == "__main__":
    train()