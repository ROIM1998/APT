from transformers import (EvalPrediction, default_data_collator, DataCollatorWithPadding, DataCollatorForSeq2Seq)
from trainer.trainer_minus import MinusTrainer
from trainer.trainer_seq2seq_minus import MinusSeq2SeqTrainer
from args import Seq2SeqDataTrainingArguments
from utils.utils import *
from datasets import load_metric
from torch.utils.data import DataLoader
from utils.qa_utils import postprocess_qa_predictions
from glue import avg_seq_length
from dataclasses import dataclass

IGNORE_INDEX = -100
GLUE_TASKS = set(["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"])

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

def build_dataloader(dataset, batch_size, data_args, training_args, tokenizer, model=None):
    if 'alpaca' in data_args.task_name:
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    elif isinstance(data_args, Seq2SeqDataTrainingArguments):
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    elif 'squad' in data_args.task_name or data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    return dataloader

def build_trainer(data_args, training_args, model, tokenizer, train_dataset=None, eval_dataset=None, teacher_model=None, param_controller=None):
    model_generative = 't5' in model.config.model_type or 'bart' in model.config.model_type or 'llama' in model.config.model_type
    IS_SQUAD = 'squad' in data_args.task_name
    if model.config.model_type == 'llama':
        metric, compute_metrics = None, None
    elif data_args.task_name is not None and data_args.task_name in GLUE_TASKS:
        label2id = model.label2id if hasattr(model, 'label2id') else None
        # Get the metric function
        metric = load_metric("./glue_metric.py", data_args.task_name)
        # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
        # predictions and label_ids field) and has to return a dictionary string to float.
        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            labels = p.label_ids
            if model_generative:
                # Cropping padded sequence tokens (for T5, it's 0, <pad>)
                preds_nonzero = (preds != 0).any(axis=0)
                preds = preds[:, preds_nonzero]
                labels_nonzero = (labels != 0).any(axis=0)
                labels = labels[:, labels_nonzero]
            else:
                preds = np.argmax(preds, axis=-1) if model_generative else np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
            if model_generative:
                preds = list(map(lambda x: label2id[tuple(x)] if tuple(x) in label2id else -1, preds.tolist()))
                labels = list(map(lambda x: label2id[tuple(x)], labels.tolist()))
            if data_args.task_name is not None:
                result = metric.compute(predictions=preds, references=labels)
                if len(result) > 1:
                    result["combined_score"] = np.mean(list(result.values())).item()
                return result
            elif is_regression:
                return {"mse": ((preds - labels) ** 2).mean().item()}
            elif model_generative:
                return {"accuracy": (preds == labels).all(dim=1).astype(np.float32).mean().item()}
            else:
                return {"accuracy": (preds == labels).astype(np.float32).mean().item()}
    else:
        metric, compute_metrics = None, None
    
    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if model.config.model_type == 'llama':
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    elif isinstance(data_args, Seq2SeqDataTrainingArguments):
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    elif IS_SQUAD or data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    
    is_regression = data_args.task_name == "stsb"
    seq_len = 170 if IS_SQUAD else avg_seq_length(data_args.task_name) if data_args.task_name in GLUE_TASKS else 512
    trainer_cls = MinusSeq2SeqTrainer if model_generative else MinusTrainer

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        teacher_model=teacher_model,
        param_controller=param_controller,
        seq_len=seq_len,
        cls_task=data_args.task_name in GLUE_TASKS,
    )
    return trainer