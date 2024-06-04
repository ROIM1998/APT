import os
import json
os.environ["WANDB_DISABLED"] = "true"
import sys
import logging
import transformers
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import time
transformers.logging.set_verbosity_error()


from transformers import (HfArgumentParser, EvalPrediction, default_data_collator, DataCollatorWithPadding, set_seed)
from deepspeed.profiling.flops_profiler import get_model_profile
from datasets import load_metric
from args import DataTrainingArguments
from models.model_args import ModelArguments
from utils.utils import *
from utils.minus_utils import efficiency_testing, input_constructor, compare_parameters
from utils.analysis_utils import gen_run_report
from trainer.trainer_minus import MinusTrainer
from trainer.trainer_seq2seq_minus import MinusSeq2SeqTrainer
from args import MinusTrainingArguments
from loralib.layers import LoRALayer
from models import build_model
from utils import avg_seq_length

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


def test(eval_datasets, tasks, trainer, data_args):
    for eval_dataset, task in zip(eval_datasets, tasks):
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_val_samples = data_args.max_val_samples if hasattr(data_args, 'max_val_samples') and data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
    return metrics


def main():
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
    set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    
    fileHandler = logging.FileHandler("{0}/{1}.log".format(training_args.output_dir, data_args.task_name))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    logger.info("MiNUS training arguments: %s", str(training_args))
    
    # save args
    torch.save(data_args, os.path.join(
        training_args.output_dir, "data_args.bin"))
    torch.save(model_args, os.path.join(
        training_args.output_dir, "model_args.bin"))
    
    training_args.disable_tqdm = False
    IS_SQUAD = 'squad' in data_args.task_name
    t_name, raw_datasets = get_raw_datasets(model_args, data_args, training_args)
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets, force_model_shape_deduction=os.path.exists(model_args.model_name_or_path))
    MODEL_GENERATIVE = any(['decoder' in n for n, _ in model.named_parameters()])
    train_dataset, eval_dataset, predict_dataset, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets, generative=MODEL_GENERATIVE)
    if MODEL_GENERATIVE:
        training_args.eval_accumulation_steps=1
        training_args.predict_with_generate=True
    label2id = model.label2id if hasattr(model, 'label2id') else None
    if training_args.teacher_path is None:
        teacher_model = None
    else:
        _, _, teacher_model = build_model(model_args, data_args, training_args, t_name, raw_datasets, determined_model_path=training_args.teacher_path)
        teacher_model.head_mask = torch.load(os.path.join(training_args.teacher_path, 'head_mask.pt')).to(training_args.device) if os.path.exists(os.path.join(training_args.teacher_path, 'head_mask.pt')) else None
        if (teacher_model.head_mask == 1).all():
            teacher_model.head_mask = None
        teacher_model.intermediate_mask = torch.load(os.path.join(training_args.teacher_path, 'intermediate_mask.pt')).to(training_args.device) if os.path.exists(os.path.join(training_args.teacher_path, 'intermediate_mask.pt')) else None
        if (teacher_model.intermediate_mask == 1).all():
            teacher_model.intermediate_mask = None
        teacher_model.hidden_mask = torch.load(os.path.join(training_args.teacher_path, 'hidden_mask.pt')).to(training_args.device) if os.path.exists(os.path.join(training_args.teacher_path, 'hidden_mask.pt')) else None
        if (teacher_model.hidden_mask == 1).all():
            teacher_model.hidden_mask = None

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name, experiment_id='elastictuning' + data_args.task_name + str(time.time()))
    else:
        metric = load_metric("accuracy", experiment_id='elastictuning' + data_args.task_name + str(time.time()))
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        labels = p.label_ids
        if MODEL_GENERATIVE:
            # Cropping padded sequence tokens (for T5, it's 0, <pad>)
            preds_nonzero = (preds != 0).any(axis=0)
            preds = preds[:, preds_nonzero]
            labels_nonzero = (labels != 0).any(axis=0)
            labels = labels[:, labels_nonzero]
        else:
            preds = np.argmax(preds, axis=-1) if MODEL_GENERATIVE else np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if MODEL_GENERATIVE:
            preds = list(map(lambda x: label2id[tuple(x)] if tuple(x) in label2id else -1, preds.tolist()))
            labels = list(map(lambda x: label2id[tuple(x)], labels.tolist()))
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=labels)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - labels) ** 2).mean().item()}
        elif MODEL_GENERATIVE:
            return {"accuracy": (preds == labels).all(dim=1).astype(np.float32).mean().item()}
        else:
            return {"accuracy": (preds == labels).astype(np.float32).mean().item()}
    
    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    
    if os.path.exists(model_args.model_name_or_path):
        if os.path.exists(os.path.join(model_args.model_name_or_path, 'head_mask.pt')):
            model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt'))
        if os.path.exists(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt')):
            model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt'))
        if os.path.exists(os.path.join(model_args.model_name_or_path, 'hidden_mask.pt')):
            model.hidden_mask = torch.load(os.path.join(model_args.model_name_or_path, 'hidden_mask.pt'))
        model.prune_model_with_masks()
        for m in model.modules():
            if isinstance(m, LoRALayer):
                m.scaling = model_args.lora_alpha / model_args.lora_r

    model = model.to(training_args.device)
    if hasattr(model, 'head_mask') and hasattr(model, 'intermediate_mask'):
        if isinstance(model.head_mask, torch.Tensor):
            model.head_mask = model.head_mask.to(training_args.device)
        elif isinstance(model.head_mask, list):
            model.head_mask = [v.to(training_args.device) for v in model.head_mask]
        if isinstance(model.intermediate_mask, torch.Tensor):
            model.intermediate_mask = model.intermediate_mask.to(training_args.device)
        elif isinstance(model.intermediate_mask, list):
            model.intermediate_mask = [v.to(training_args.device) for v in model.intermediate_mask]
    
    
    if getattr(model, 'hidden_mask', None) is not None:
        model.hidden_mask = model.hidden_mask.to(training_args.device)
    
    flops, macs, params = get_model_profile(
        model,
        kwargs={k: v.to(model.device) for k, v in input_constructor(training_args.per_device_eval_batch_size, data_args.max_seq_length, tokenizer, output_seq_len=2).items()} if MODEL_GENERATIVE else {k: v.to(model.device) for k, v in input_constructor(training_args.per_device_eval_batch_size, data_args.max_seq_length, tokenizer).items()},
        print_profile=True,
        detailed=True,
        output_file=os.path.join(training_args.output_dir, 'pretrain_deepspeed_profile.txt'),
    )
    torch.cuda.reset_peak_memory_stats()
    
    seq_len = 170 if IS_SQUAD else avg_seq_length(data_args.task_name)
    training_args.task_name = data_args.task_name
    trainer_cls = MinusTrainer if not MODEL_GENERATIVE else MinusSeq2SeqTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        teacher_model=teacher_model,
        seq_len=seq_len,
        cls_task=not IS_SQUAD,
    )

    # Training
    if training_args.do_train:
        if os.path.exists(model_args.model_name_or_path):
            print("Evaluating pre-trained model...")
            print(trainer.evaluate())
        train_result = trainer.train()
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        # trainer.save_param_allocation()
        # trainer.save_allocation_history()
    
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        _ = model.eval()
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        metrics = test(eval_datasets, tasks, trainer, data_args)
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)

    # TODO: merge LoRA layers after training for efficiency during efficiency & deepspeed profiler testing
    model.eval()
    efficiency_results = efficiency_testing(model, tokenizer, training_args.device, model_generative=MODEL_GENERATIVE)

    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.eval()

    flops, macs, params = get_model_profile(
        model,
        kwargs={k: v.to(model.device) for k, v in input_constructor(training_args.per_device_eval_batch_size, data_args.max_seq_length, tokenizer, output_seq_len=2).items()} if MODEL_GENERATIVE else {k: v.to(model.device) for k, v in input_constructor(training_args.per_device_eval_batch_size, data_args.max_seq_length, tokenizer).items()},        print_profile=True,
        detailed=True,
        output_file=os.path.join(training_args.output_dir, 'deepspeed_profile.txt'),
    )
    efficiency_results['model_flops'] = flops
    efficiency_results['model_macs'] = macs
    
    json.dump(efficiency_results, open(os.path.join(training_args.output_dir, 'efficiency_results.json'), 'w'), indent=4, sort_keys=True)
    run_report = gen_run_report(training_args.output_dir)
    run_report['train_runtime_per_epoch'] = run_report['train_runtime'] / training_args.num_train_epochs
    json.dump(run_report, open(os.path.join(training_args.output_dir, 'run_report.json'), 'w'), indent=4, sort_keys=True)
    
    if os.path.exists(os.path.join(training_args.output_dir, 'pre_pruning_model')):
        model_args.model_name_or_path = os.path.join(training_args.output_dir, 'pre_pruning_model')
        config, tokenizer, pre_pruning_model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
        pre_pruning_model.head_mask = torch.load(os.path.join(training_args.output_dir, 'final_head_mask.pt'), map_location='cpu')
        pre_pruning_model.intermediate_mask = torch.load(os.path.join(training_args.output_dir, 'final_intermediate_mask.pt'), map_location='cpu')
        pre_pruning_model.hidden_mask = torch.load(os.path.join(training_args.output_dir, 'final_hidden_mask.pt'), map_location='cpu') if os.path.exists(os.path.join(training_args.output_dir, 'final_hidden_mask.pt')) else None
        pre_pruning_model.prune_model_with_masks()
        model = model.cpu()
        same_param_num, same_vars = compare_parameters(model, pre_pruning_model)
        logger.info(f"Num parameters not changed after pruning: {same_param_num}")
        logger.info(f"Parameter variables not changed after pruning: {same_vars}")

if __name__ == '__main__':
    main()