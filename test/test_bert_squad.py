import os
import json
os.environ["WANDB_DISABLED"] = "true"
import sys
import transformers
import torch


from transformers import (HfArgumentParser, EvalPrediction, default_data_collator, set_seed)
from datasets import load_metric
from transformers import SquadDataTrainingArguments
from models.model_args import ModelArguments
from utils.utils import *
from trainer.trainer_qa_minus import MinusQATrainer
from args import MinusTrainingArguments
from models import build_model
from utils.qa_utils import postprocess_qa_predictions

def main():
    sys.argv = ['test_bert_squad.py',
            '--output_dir',
            './output/test_bert_squad/',
            '--model_name_or_path',
            'output/bert-base-uncased_lora_minus_squad_cubic_gradual_running_fisher_alloc_running_fisher_self_student_mapping_static_teacher_dynamic_cofi_student_distill_momentum/mac0.4/epoch10/bz16/numprune10/paramq:0-11,v:0-11/lora_r8/pruning_start-1/distill_epoch5/pre_distillation_model',
            '--do_train',
            '--do_eval',
            '--doc_stride',
            '128',
            '--max_seq_length',
            '384',
            '--per_device_train_batch_size',
            '16',
            '--per_device_eval_batch_size',
            '16',
            '--apply_lora',
            '--do_distill',
            '--lora_r',
            '8',
            '--teacher_param_tuning_config',
            'q:0-11,v:0-11,i:0-11',
            '--report_to',
            'none',
            ]
    parser = HfArgumentParser(
        (ModelArguments, SquadDataTrainingArguments, MinusTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    IS_SQUAD_V2 = False
    
    task_name = 'squad' if not IS_SQUAD_V2 else 'squad_v2'
    data_args.task_name = task_name
    
    set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    
    training_args.disable_tqdm = False
    config, tokenizer, model = build_model(model_args, data_args, training_args)
    
    MODEL_GENERATIVE = False
    train_dataset, eval_dataset, _, datasets, answer_column_name = build_squad_data(data_args, training_args, tokenizer, is_v2=IS_SQUAD_V2, generative=MODEL_GENERATIVE)

    teacher_model = None
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
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        labels = p.label_ids
        if MODEL_GENERATIVE:
            # Cropping padded sequence tokens (for T5, it's 0, <pad>)
            preds_nonzero = (preds != 0).any(axis=0)
            preds = preds[:, preds_nonzero]
            labels_nonzero = (labels != 0).any(axis=0)
            labels = labels[:, labels_nonzero]
        else:
            preds = np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=labels)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif MODEL_GENERATIVE:
            return {"accuracy": (preds == labels).all(dim=1).astype(np.float32).mean().item()}
        else:
            return {"accuracy": (preds == labels).astype(np.float32).mean().item()}
    
    seq_len = 170
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
    
    trainer.param_controller.set_grafting_mask(target='teacher')

if __name__ == '__main__':
    main()