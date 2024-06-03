import random
import numpy as np
from collections import defaultdict
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Subset
from transformers import PretrainedConfig
from datasets import load_dataset, DatasetDict
from typing import Sequence, Dict
from utils import alpaca_utils
from dataclasses import dataclass, field
import logging
import transformers 
import torch
import copy

logger = logging.getLogger(__name__)


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

abbr_to_lang = {
    'en': 'English',
    'de': 'German',
    'ro': 'Romanian',
}

def get_raw_datasets(model_args, data_args, training_args):
    t_name = None 
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "./glue.py", data_args.task_name.replace("-", ""), cache_dir=model_args.cache_dir)
        t_name = data_args.task_name
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
        t_name = data_args.dataset_name
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        t_name = data_args.t_name
        data_files = {"train": data_args.train_file,
                        "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError(
                    "Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv", data_files=data_files, cache_dir=model_args.cache_dir)
        elif data_args.train_file.endswith(".tsv"):
            dataset_dict = {}
            for key in data_files:
                dataset_dict[key] = load_from_tsv(data_files[key])
            raw_datasets = DatasetDict(dataset_dict)
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json", data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    return t_name, raw_datasets




def log_all_parameters(logger, model_args, data_args, training_args, additional_args):
    logger.info("Model Arguments:")
    for arg in vars(model_args):
        logger.info(f"{arg} = {getattr(model_args, arg)}")

    logger.info("Data Arguments:")
    for arg in vars(data_args):
        logger.info(f"{arg} = {getattr(data_args, arg)}")

    logger.info("Training Arguments:")
    for arg in vars(training_args):
        logger.info(f"{arg} = {getattr(training_args, arg)}")

    logger.info("Additional Arguments:")
    for arg in vars(additional_args):
        logger.info(f"{arg} = {getattr(additional_args, arg)}")

def calculate_parameters(module):
    keys = ["embedding", "layer_transformation", "classifier", "pooler"]
    return sum(p.numel() for n, p in module.named_parameters() if not any(key in n for key in keys))

def load_from_tsv(file):
    lines = open(file, "r").readlines()
    data = [line.strip().split("\t") for line in lines[1:]]
    headers = lines[0].strip().split("\t")
    d = defaultdict(list)
    for i, head in enumerate(headers):
        for j, dd in enumerate(data):
            d[head].append(dd[i])

    dataset = Dataset.from_dict(d)
    return dataset


def get_label(data_args, raw_datasets):
    # Labels (regression or classification)
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            label_list = None
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in [
            "float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    return label_list, num_labels, is_regression


def build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets, generative=False):
    label_list, num_labels, is_regression = get_label(data_args, raw_datasets)
    # Loading preprocess column names (keys)
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
        and not generative
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression and not generative:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif generative:
        model.label2id = {tuple(tokenizer(label, truncation=True, padding='max_length', max_length=2)['input_ids']):i for i, label in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}
        model.config.label2id = {str(label):i for i, label in model.config.id2label.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # preprocessing function to load raw datasets
    def preprocess_function(examples):
        # Tokenize the texts
        if generative:
            args = (
                ([sentence1_key + ": " + v for v in  examples[sentence1_key]],) if sentence2_key is None else ([sentence1_key + ": " + s1 + ' ' + sentence2_key + ": " + s2 for s1, s2 in zip(examples[sentence1_key], examples[sentence2_key])],)
            )
        else:
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if generative:
            target_text = [label_list[v] for v in examples["label"]]
            result['labels'] = tokenizer(target_text, truncation=True, padding='max_length', max_length=2)['input_ids']
        elif label_to_id is not None and "label" in examples:
            result["labels"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        elif "label" in examples:
            result["labels"] = examples["label"]
        return result

    # preprocessing raw datasets into torch Dataset
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
            remove_columns=raw_datasets["train"].column_names
        ) #! get dataset
    
    train_dataset, eval_dataset, predict_dataset = None, None, None
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    return train_dataset, eval_dataset, predict_dataset, is_regression


def build_squad_data(data_args, training_args, tokenizer, is_v2: bool = False, generative=False):
    if is_v2:
        datasets = load_dataset("squad_v2")
    else:
        datasets = load_dataset("squad")

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.py.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    # Training preprocessing
    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name if pad_on_right else context_column_name] = [q.lstrip() for q in examples[question_column_name if pad_on_right else context_column_name]]
        
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=data_args.max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(
                        token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(
                        token_end_index + 1)

        return tokenized_examples

    if training_args.do_train:
        train_dataset = datasets["train"].map(
            prepare_train_features,
            batched=True,
            remove_columns=column_names,
        )

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=data_args.max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation.py, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(
                examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    if training_args.do_eval:
        # validation_dataset = load_and_cache_examples(training_args, data_args, model_args, tokenizer, evaluate=True, output_examples=False)
        validation_dataset = datasets["validation"].map(
            prepare_validation_features,
            batched=True,
            remove_columns=column_names,
        )
    
    return train_dataset, validation_dataset, None, datasets, answer_column_name


def build_seq2seq_data(data_args, training_args, tokenizer, **kwargs):
    if data_args.task_name == 'xsum':
        return build_xsum_data(data_args, training_args, tokenizer)
    elif data_args.task_name == 'cnndm':
        return build_cnn_dm_data(data_args, training_args, tokenizer)
    elif data_args.task_name == 'wmt16':
        return build_wmt16_data(data_args, training_args, tokenizer)
    elif 'alpaca' in data_args.task_name:
        return build_alpaca_data(data_args, training_args, tokenizer, **kwargs)
    else:
        raise NotImplementedError

def build_xsum_data(data_args, training_args, tokenizer):
    raw_datasets = load_dataset("xsum")
    max_input_length, max_target_length = data_args.max_input_length, data_args.max_target_length
    
    def preprocess_function(examples):
        inputs = ["summarize: " + doc for doc in examples["document"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(raw_datasets["train"].column_names)

    return tokenized_datasets["train"] if training_args.do_train else None, tokenized_datasets["validation"] if training_args.do_eval else None, tokenized_datasets["test"] if training_args.do_predict else None, raw_datasets

def build_cnn_dm_data(data_args, training_args, tokenizer):
    raw_datasets = load_dataset("cnn_dailymail", '3.0.0')
    max_input_length, max_target_length = data_args.max_input_length, data_args.max_target_length
    
    def preprocess_function(examples):
        inputs = ["summarize: " + doc for doc in examples["article"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["highlights"], max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(raw_datasets["train"].column_names)

    return tokenized_datasets["train"] if training_args.do_train else None, tokenized_datasets["validation"] if training_args.do_eval else None, tokenized_datasets["test"] if training_args.do_predict else None, raw_datasets

def build_wmt16_data(data_args, training_args, tokenizer):
    source_lang, target_lang = data_args.source_lang, data_args.target_lang
    if target_lang == 'en':
        data_args.lang_pair = '%s-%s' % (source_lang, target_lang) # Reverse the language pair and make sure ends with en
    raw_datasets = load_dataset("wmt16", data_args.lang_pair)
    max_input_length, max_target_length = data_args.max_input_length, data_args.max_target_length
    
    def preprocess_function(examples):
        inputs = ["translate %s to %s: " % (abbr_to_lang[source_lang], abbr_to_lang[target_lang]) + doc[source_lang] for doc in examples['translation']]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer([doc[target_lang] for doc in examples['translation']], max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(raw_datasets["train"].column_names)

    return tokenized_datasets["train"] if training_args.do_train else None, tokenized_datasets["validation"] if training_args.do_eval else None, tokenized_datasets["test"] if training_args.do_predict else None, raw_datasets


def build_alpaca_data(data_args, training_args, tokenizer, do_split=True, no_output=False):
    IGNORE_INDEX = -100
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
        if no_output:
            examples = [s for s in sources]
        else:
            examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)


    class SupervisedDataset(TorchDataset):
        """Dataset for supervised fine-tuning."""

        def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
            super(SupervisedDataset, self).__init__()
            logging.info("Loading data...")
            list_data_dict = alpaca_utils.jload(data_path)
            logging.info("Formatting inputs...")
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
            sources = [
                prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                for example in list_data_dict
            ]
            targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

            logging.info("Tokenizing inputs... This may take some time...")
            data_dict = preprocess(sources, targets, tokenizer)

            self.input_ids = data_dict["input_ids"]
            self.labels = data_dict["labels"]

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    total_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    
    if do_split:
        # Randomly sampling 1% of the training data as the validation set.
        total_samples = len(total_dataset)
        num_train_samples = int(0.99 * total_samples)

        # Split the dataset into two disjoint datasets.
        train_dataset, eval_dataset = torch.utils.data.random_split(
            total_dataset,
            [num_train_samples, total_samples - num_train_samples]
        )

        print("Sampled %d examples from %d training examples for the validation set." % (len(eval_dataset), len(train_dataset)))
        return train_dataset, eval_dataset, None, None # (no eval, no test, no raw_datasets)
    else:
        return total_dataset, None, None, None

def build_mmlu_data():
    pass