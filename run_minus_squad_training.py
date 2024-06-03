import os
import json
os.environ["WANDB_DISABLED"] = "true"
import sys
import logging
import transformers
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
transformers.logging.set_verbosity_error()


from transformers import (HfArgumentParser, EvalPrediction, default_data_collator, set_seed)
from deepspeed.profiling.flops_profiler import get_model_profile
from datasets import load_metric
from transformers import SquadDataTrainingArguments
from models.model_args import ModelArguments
from utils.utils import *
from utils.minus_utils import efficiency_testing, input_constructor, compare_parameters
from utils.analysis_utils import gen_run_report
from trainer.trainer_qa_minus import MinusQATrainer
from args import MinusTrainingArguments
from loralib.layers import LoRALayer
from models import build_model
from utils.qa_utils import postprocess_qa_predictions

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
        (ModelArguments, SquadDataTrainingArguments, MinusTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)
    IS_SQUAD_V2 = data_args.version_2_with_negative
    logger.info("IS_SQUAD_V2: " + str(IS_SQUAD_V2))
    task_name = 'squad' if not IS_SQUAD_V2 else 'squad_v2'
    data_args.task_name = task_name
    set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    
    # save args
    torch.save(data_args, os.path.join(
        training_args.output_dir, "data_args.bin"))
    torch.save(model_args, os.path.join(
        training_args.output_dir, "model_args.bin"))
    
    fileHandler = logging.FileHandler("{0}/{1}.log".format(training_args.output_dir, task_name))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    logger.info("MiNUS training arguments: %s", str(training_args))
    
    training_args.disable_tqdm = False
    config, tokenizer, model = build_model(model_args, data_args, training_args)
    
    MODEL_GENERATIVE = any(['decoder' in n for n, _ in model.named_parameters()])
    train_dataset, eval_dataset, _, datasets, answer_column_name = build_squad_data(data_args, training_args, tokenizer, is_v2=IS_SQUAD_V2, generative=MODEL_GENERATIVE)
    if MODEL_GENERATIVE:
        training_args.eval_accumulation_steps=1
        training_args.predict_with_generate=True

    if training_args.teacher_path is None:
        teacher_model = None
    else:
        _, _, teacher_model = build_model(model_args, data_args, training_args, determined_model_path=training_args.teacher_path)
        teacher_model.head_mask, teacher_model.intermediate_mask, teacher_model.hidden_mask = None, None, None
    
    
    if os.path.exists(model_args.model_name_or_path):
        if model.pruned_history is None:
            model.head_mask, model.intermediate_mask = None, None
            model.hidden_mask = None
        else:
            if os.path.exists(os.path.join(model_args.model_name_or_path, '../final_head_mask.pt')):
                model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_head_mask.pt'))
            else:
                model.head_mask = None
            if os.path.exists(os.path.join(model_args.model_name_or_path, '../final_intermediate_mask.pt')):
                model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_intermediate_mask.pt'))
            else:
                model.intermediate_mask = None
            if os.path.exists(os.path.join(model_args.model_name_or_path, '../final_hidden_mask.pt')):
                model.hidden_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_hidden_mask.pt'))
            else:
                model.hidden_mask = None
            model.prune_model_with_masks()
    else:
        if os.path.exists(os.path.join(model_args.model_name_or_path, 'head_mask.pt')):
            model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt'))
        if os.path.exists(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt')):
            model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt'))

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
    
    if hasattr(model, 'hidden_mask') and model.hidden_mask is not None:
        model.hidden_mask = model.hidden_mask.to(training_args.device)

    # Post-processing:
    def post_processing_function(examples, features, predictions):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            is_world_process_zero=trainer.is_world_process_zero(),
        )
        # Format the result to the format the metric expects.
        if data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in predictions.items()]
        references = [{"id": ex["id"], "answers": ex[answer_column_name]}
                      for ex in datasets["validation"]]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    # Get the metric function
    if IS_SQUAD_V2:
        metric = load_metric("squad_v2")
    else:
        metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)
    
    flops, macs, params = get_model_profile(
        model,
        kwargs={k: v.to(model.device) for k, v in input_constructor(training_args.per_device_eval_batch_size, data_args.max_seq_length, tokenizer).items()},
        print_profile=True,
        detailed=True,
        output_file=os.path.join(training_args.output_dir, 'pretrain_deepspeed_profile.txt'),
    )
    torch.cuda.reset_peak_memory_stats()
    
    seq_len = 170
    training_args.task_name = data_args.task_name
    trainer = MinusQATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        post_processing_function=post_processing_function,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        teacher_model=teacher_model,
        seq_len=seq_len,
        cls_task=False,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

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
        tasks = [task_name]
        eval_datasets = [eval_dataset]
        metrics = test(eval_datasets, tasks, trainer, data_args)
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)

    # TODO: merge LoRA layers after training for efficiency during efficiency & deepspeed profiler testing
    model.eval()
    efficiency_results = efficiency_testing(model, tokenizer, training_args.device)

    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.eval()

    flops, macs, params = get_model_profile(
        model,
        kwargs={k: v.to(model.device) for k, v in input_constructor(training_args.per_device_eval_batch_size, data_args.max_seq_length, tokenizer).items()},
        print_profile=True,
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
        config, tokenizer, pre_pruning_model = build_model(model_args, data_args, training_args)
        pre_pruning_model.head_mask = torch.load(os.path.join(training_args.output_dir, 'final_head_mask.pt'), map_location='cpu')
        pre_pruning_model.intermediate_mask = torch.load(os.path.join(training_args.output_dir, 'final_intermediate_mask.pt'), map_location='cpu')
        pre_pruning_model.hidden_mask = torch.load(os.path.join(training_args.output_dir, 'final_hidden_mask.pt'), map_location='cpu') if os.path.exists(os.path.join(training_args.output_dir, 'final_hidden_mask.pt')) else None
        pre_pruning_model.hidden_mask = torch.load(os.path.join(training_args.output_dir, 'final_hidden_mask.pt'), map_location='cpu') if os.path.exists(os.path.join(training_args.output_dir, 'final_hidden_mask.pt')) else None
        pre_pruning_model.prune_model_with_masks()
        model = model.cpu()
        same_param_num, same_vars = compare_parameters(model, pre_pruning_model)
        logger.info(f"Num parameters not changed after pruning: {same_param_num}")
        logger.info(f"Parameter variables not changed after pruning: {same_vars}")

if __name__ == '__main__':
    main()