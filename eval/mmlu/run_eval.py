import argparse
import os
import torch
import transformers
import numpy as np
import pandas as pd
import time
import json
import loralib as lora
from tqdm import tqdm
import time
from typing import Dict
from eval.mmlu.categories import subcategories, categories
from eval.utils import get_next_word_predictions, load_hf_lm_and_tokenizer, query_openai_chat_model

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


def eval_openai_chat_engine(args, subject, engine, dev_df, test_df, batch_size=1):
    
    import tiktoken
    gpt_tokenizer = tiktoken.get_encoding("cl100k_base")
    answer_choice_ids = [gpt_tokenizer.encode(" " + x)[0] for x in choices]  # be careful, the tokenizer will tokenize " A" and "A" differently.

    prompts = []
    for i in range(0, test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end        
        prompts.append(prompt)

    instances = [{"id": prompt, "prompt": prompt} for _, prompt in enumerate(prompts)]
    results = query_openai_chat_model(
        engine=args.openai_engine,
        instances=instances,
        batch_size=args.eval_batch_size if args.eval_batch_size else 10,
        output_path=os.path.join(args.save_dir, f"{subject}_openai_results.jsonl"),
        logit_bias={token_id: 100 for token_id in answer_choice_ids},
        max_tokens=1,
    )
    
    # get the metrics
    cors = []
    groud_truths = test_df.iloc[:, -1].values
    for i in range(len(test_df)):
        prediction = results[i]["output"].strip()
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)
        
    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array([[0.25, 0.25, 0.25, 0.25] for _ in range(len(test_df))]) # dummy probs, just don't want to dig into the openai probs

    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs

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

def main(args):

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path, 
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit, 
            load_in_half=False,
            gptq_model=args.gptq
        )
    
    special_tokens_dict = dict()
    special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    # Converting q, v, and gate to LoRA layers
    for i in tqdm(range(model.config.num_hidden_layers)):
        q, v = model.model.layers[i].self_attn.q_proj, model.model.layers[i].self_attn.v_proj
        q_lora = lora.Linear(q.in_features, q.out_features, bias=q.bias is not None, r=8, lora_alpha=16, lora_dropout=0.05)
        v_lora = lora.Linear(v.in_features, v.out_features, bias=v.bias is not None, r=8, lora_alpha=16, lora_dropout=0.05)
        del q.weight
        del v.weight
        if q.bias is not None:
            del q.bias
            del v.bias
        model.model.layers[i].self_attn.q_proj = q_lora
        model.model.layers[i].self_attn.v_proj = v_lora
        
        gate = model.model.layers[i].mlp.gate_proj
        gate_lora = lora.Linear(gate.in_features, gate.out_features, bias=gate.bias is not None, r=8, lora_alpha=16, lora_dropout=0.05)
        del gate.weight
        if gate.bias is not None:
            del gate.bias
        model.model.layers[i].mlp.gate_proj = gate_lora
        
    model_dict_keys = list(model.state_dict().keys())
    real_dir = '/mmfs1/home/bowen98/projects/AdaptPruning/output/meta-llama/Llama-2-7b-hf/alpaca_gpt4/bz4/lora/teacher_dq:0-31,dv:0-31,di0:0-31/epoch5/lora_r8/lora_alpha16/lr2e-4/seed42/'
    weight_files = [v for v in os.listdir(real_dir) if v.endswith('00003.bin')]
    loaded_keys = []
    for weight_file in tqdm(weight_files):
        weight_file = os.path.join(real_dir, weight_file)
        weight_dict = torch.load(weight_file)
        model.load_state_dict(weight_dict, strict=False)
        loaded_keys += list(weight_dict.keys())
    loaded_keys = set(loaded_keys)
    for key in model_dict_keys:
        if key not in loaded_keys:
            print(key, 'not loaded')
            
    model = model.cuda()
    
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if args.subjects:
        assert all(subj in subjects for subj in args.subjects), f"Some of the subjects you specified are not valid: {args.subjects}"
        subjects = args.subjects

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir)):
        os.makedirs(os.path.join(args.save_dir))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in tqdm(subjects, desc=f"Evaluating subjects: "):
        
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )
        if args.n_instances and args.n_instances < test_df.shape[0]:
            test_df = test_df.sample(args.n_instances, random_state=42)

        if args.model_name_or_path:
            cors, acc, probs = eval_hf_model(args, subject, model, tokenizer, dev_df, test_df, args.eval_batch_size)
        else:
            cors, acc, probs = eval_openai_chat_engine(args, subject, args.openai_engine, dev_df, test_df, args.eval_batch_size)
            
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
        test_df.to_csv(
            os.path.join(
                args.save_dir, "{}.csv".format(subject)
            ),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

    # save results
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "average_acc": weighted_acc,
                "subcat_acc": {
                    subcat: np.mean(np.concatenate(subcat_cors[subcat]))
                    for subcat in subcat_cors
                },
                "cat_acc": {
                    cat: np.mean(np.concatenate(cat_cors[cat]))
                    for cat in cat_cors
                },
            },
            f,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="data/mmlu")
    parser.add_argument("--save_dir", type=str, default="results/mmlu/llama-7B/")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="if specified, we will load the tokenizer from here.")
    parser.add_argument("--openai_engine", type=str, default=None, help="if specified, we will use the OpenAI API to generate the predictions.")
    parser.add_argument("--subjects", nargs="*", help="which subjects to evaluate. If not specified, all the 57 subjects will be evaluated.")
    parser.add_argument("--n_instances", type=int, help="if specified, a maximum of n_instances per subject will be used for the evaluation.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--load_in_8bit", action="store_true", help="load model in 8bit mode, which will reduce memory and speed up inference.")
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument("--use_chat_format", action="store_true", help="If given, the prompt will be encoded as a chat format with the roles in prompt.")
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
