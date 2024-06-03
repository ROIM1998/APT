import argparse
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import numpy as np
import pandas as pd
import time
import json
from tqdm import tqdm
import time
from eval.mmlu.categories import subcategories, categories
from eval.utils import get_next_word_predictions, load_hf_lm_and_tokenizer, query_openai_chat_model

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    q_info = df.iloc[idx, 1]
    q_options = list(q_info.keys())

    for j in range(len(q_options)):
        prompt += "\n{}. {}".format(choices[j], q_options[j])
    prompt += "\nAnswer:"

    if include_answer:
        for i, option in enumerate(q_options):
            if q_info[option] == 1:
                answer_choice = choices[i]
                break

        prompt += " {}\n\n".format(answer_choice)

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
def eval_hf_model(args, model, tokenizer, test_df, batch_size=1):
    prompts = []
    
    for i in range(0, test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        # train_prompt = gen_prompt(dev_df, subject, k)
        # prompt = train_prompt + prompt_end
        prompt = prompt_end
        # tokenized_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        # # make sure every prompt is less than 2048 tokens
        # while tokenized_prompt.shape[-1] > 2048:
        #     k -= 1
        #     train_prompt = gen_prompt(dev_df, subject, k)
        #     prompt = train_prompt + prompt_end
        #     tokenized_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids

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
    mc1_column = test_df.iloc[:,1].values
    ground_truths = [choices[list(q_info.values()).index(1)] for q_info in mc1_column]

    for i in range(len(pred_indices)):
        prediction = choices[pred_indices[i]]
        ground_truth = ground_truths[i]
        print(prediction, ground_truth)
        cors.append(prediction == ground_truth)
        
    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f}".format(acc))
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

def main(args):
    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path, 
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit, 
            load_in_half=True,
            gptq_model=args.gptq
        )
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir)):
        os.makedirs(os.path.join(args.save_dir))
    
    test_df = pd.read_json(
        os.path.join(args.data_dir, "truthfulqa_mc.json")
    )

    if args.n_instances and args.n_instances < test_df.shape[0]:
        test_df = test_df.sample(args.n_instances, random_state=42)

    # Shuffle answers
    for i, q_info in enumerate(test_df.iloc[:,1].values):
        q_options = list(q_info.items())
        np.random.shuffle(q_options)
        test_df.iat[i, 1] = dict(q_options)

    if args.model_name_or_path:
        cors, acc, probs = eval_hf_model(args, model, tokenizer, test_df, args.eval_batch_size)
    else:
        cors, acc, probs = eval_openai_chat_engine(args, args.openai_engine, test_df, args.eval_batch_size)
    
    print("Average accuracy: {:.3f}".format(acc))

    # save results
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "average_acc": acc,
            },
            f,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="data/truthfulqa")
    parser.add_argument("--save_dir", type=str, default="results/truthfulqa/llama-7B/")
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
