# Test ElasticLlama model with pruning consistency and tuning consistency
import os
import sys
import torch
import loralib as lora

from transformers import (HfArgumentParser, set_seed)
from args import InstructionDataTrainingArguments
from models.model_args import ModelArguments
from args import MinusTrainingArguments
from utils.utils import *
from models import build_model
from trainer.trainer_minus import MinusTrainer
import numpy as np
import pandas as pd
from tqdm import tqdm
from eval.mmlu.categories import subcategories, categories
from eval.utils import get_next_word_predictions

choices = ["A", "B", "C", "D"]

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

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval_hf_model(args, subject, model, tokenizer, dev_df, test_df, batch_size=1):
    prompts = []
    for i in range(0, test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        
        tokenized_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        # make sure every prompt is less than 2048 tokens
        while tokenized_prompt.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            tokenized_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids

        if args.use_chat_format:
            prompt = "<|user|>\n" + prompt.strip() + "\n<|assistant|>\nThe answer is:"
            
        prompts.append(prompt)

    # get the answer for all examples
    # note: here we cannot directly use convert_tokens_to_ids because the some tokenizers will automatically add space prefix.
    answer_choice_ids = [tokenizer.encode(answer_choice, add_special_tokens=False)[0] for answer_choice in choices]
    pred_indices, all_probs = get_next_word_predictions(
        model, tokenizer, prompts, candidate_token_ids=answer_choice_ids, return_token_predictions=False, batch_size=batch_size
    )

    # get the metrics
    cors = []
    groud_truths = test_df.iloc[:, -1].values
    for i in range(len(pred_indices)):
        prediction = choices[pred_indices[i]]
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)
        
    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs

def main():
    sys.argv = ['test_bert.py',
            '--output_dir',
            './output/test_llama_elastictuning/',
            '--model_name_or_path',
            'output/meta-llama/Llama-2-7b-hf/alpaca_gpt4/bz4/elastictuning_virtualprune_pre-tuning-prune0.8/mac0.6/epoch10/distill_epoch5/numprune10/sparsity_warmup1/pruning_start-1/pruning_stop3/lora_r8/lora_alpha16/warmup_paramdq:0-31,dv:0-31/teacher_paramdq:0-31,dv:0-31',
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
    set_seed(128)
    config, tokenizer, model = build_model(model_args, data_args, training_args, token=os.environ.get('HF_TOKEN', None))
    model.head_mask, model.intermediate_mask, model.hidden_mask = None, None, None
    train_dataset, eval_dataset, _, _ = build_seq2seq_data(data_args, training_args, tokenizer)
        
    inputs = tokenizer("Please answer the following question based on your knowledge: who is the president of the United States?", return_tensors="pt")
    inputs = next(iter(train_dataset))
    inputs = {k: v.unsqueeze(0).to(model.device) for k, v in inputs.items()}
        
    with torch.no_grad():
        # use greedy decoding
        gen = model.generate(
            input_ids = inputs['input_ids'],
            num_beams=1, max_length=128, do_sample=False
        )

    tokenizer.decode(inputs['input_ids'][:, :inputs['labels'][0].tolist().index(29896)].tolist()[0])
    tokenizer.decode(gen[0].tolist())
        

    for m in model.modules():
        if isinstance(m, lora.LoRALayer):
            m.scaling = 2
    
    model = model.cuda()
    
    inputs = train_dataset[0]
    inputs = {k: v.unsqueeze(0).to(model.device) for k, v in inputs.items()}
    
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        teacher_model=None,
        param_controller=None,
        seq_len=512,
        cls_task=False,
    )
    
    data_dir = '/mmfs1/home/bowen98/projects/AdaptPruning/data/eval/mmlu'
    
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}
    
    args = training_args
    args.ntrain = 0
    args.use_chat_format = True
    args.n_instances = None

    for subject in tqdm(subjects, desc=f"Evaluating subjects: "):
        
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )
        if args.n_instances and args.n_instances < test_df.shape[0]:
            test_df = test_df.sample(args.n_instances, random_state=42)

        cors, acc, probs = eval_hf_model(args, subject, model, tokenizer, dev_df, test_df, args.eval_batch_size)

            
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["correct"] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["choice{}_probs".format(choice)] = probs[:, j]

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))
    
if __name__ == '__main__':
    main()