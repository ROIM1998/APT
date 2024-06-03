import os
import json
os.environ["WANDB_DISABLED"] = "true"
import sys
import logging
import transformers
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import nltk
import numpy as np
transformers.logging.set_verbosity_error()

from tqdm import tqdm
from transformers import (HfArgumentParser, EvalPrediction, DataCollatorForSeq2Seq, set_seed)
from torch.nn.utils.rnn import pad_sequence
from deepspeed.profiling.flops_profiler import get_model_profile
from datasets import load_metric
from models.model_args import ModelArguments
from utils.utils import *
from utils.minus_utils import efficiency_testing, input_constructor, compare_parameters
from utils.analysis_utils import gen_run_report
from trainer.trainer_seq2seq_minus import MinusSeq2SeqTrainer
from args import MinusTrainingArguments, Seq2SeqDataTrainingArguments
from loralib.layers import LoRALayer
from models import build_model

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)



def main():
    parser = HfArgumentParser(
        (ModelArguments, Seq2SeqDataTrainingArguments, MinusTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)
    task_name = data_args.task_name
    
    fileHandler = logging.FileHandler("{0}/{1}.log".format(training_args.output_dir, task_name))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    logger.info("MiNUS training arguments: %s", str(training_args))
    set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    
    # save args
    torch.save(data_args, os.path.join(
        training_args.output_dir, "data_args.bin"))
    torch.save(model_args, os.path.join(
        training_args.output_dir, "model_args.bin"))
    
    training_args.disable_tqdm = False
    training_args.predict_with_generate=True
    config, tokenizer, model = build_model(model_args, data_args, training_args)
    train_dataset, eval_dataset, _, datasets = build_seq2seq_data(data_args, training_args, tokenizer)
    
    if training_args.teacher_path is None:
        teacher_model = None
    else:
        _, _, teacher_model = build_model(model_args, data_args, training_args, determined_model_path=training_args.teacher_path)
        teacher_model.head_mask, teacher_model.intermediate_mask, teacher_model.hidden_mask = None, None, None
    
    
    # if os.path.exists(model_args.model_name_or_path):
    #     if getattr(model, 'pruned_history', None) is not None:
    #         model.head_mask, model.intermediate_mask = None, None
    #         model.hidden_mask = None
    #     else:
    #         if os.path.exists(os.path.join(model_args.model_name_or_path, '../final_head_mask.pt')):
    #             model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_head_mask.pt'))
    #         else:
    #             model.head_mask = None
    #         if os.path.exists(os.path.join(model_args.model_name_or_path, '../final_intermediate_mask.pt')):
    #             model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_intermediate_mask.pt'))
    #         else:
    #             model.intermediate_mask = None
    #         if os.path.exists(os.path.join(model_args.model_name_or_path, '../final_hidden_mask.pt')):
    #             model.hidden_mask = torch.load(os.path.join(model_args.model_name_or_path, '../final_hidden_mask.pt'))
    #         else:
    #             model.hidden_mask = None
    #         model.prune_model_with_masks()

    if model.head_mask is None and model.intermediate_mask is None and model.hidden_mask is None and training_args.pruner_type != 'none':
        model.reset_masks()

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

    if 'wmt' in task_name:
        metric = load_metric("sacrebleu")
        gen_prefix = "eval"

        def postprocess_text(preds, labels):
            str_preds = [pred.strip() for pred in preds]
            str_labels = [label.strip() for label in labels]

            preds = [pred.strip() for pred in preds]
            labels = [[label.strip()] for label in labels]
            return preds, labels, str_preds, str_labels

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            if data_args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Some simple post-processing
            decoded_preds, decoded_labels, str_decoded_preds, str_decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            result = {"bleu": result["score"]}

            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            return result
    else:
        metric = load_metric("rouge")
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Rouge expects a newline after each sentence
            decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
            decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
            
            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            # Extract a few results
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
            
            # Add mean generated length
            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
            result["gen_len"] = np.mean(prediction_lens)
            
            return {k: round(v, 4) for k, v in result.items()}

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    flops, macs, params = get_model_profile(
        model,
        kwargs={k: v.to(model.device) for k, v in input_constructor(training_args.per_device_eval_batch_size, data_args.max_input_length, tokenizer, output_seq_len=data_args.max_target_length).items()},
        print_profile=True,
        detailed=True,
        output_file=os.path.join(training_args.output_dir, 'pretrain_deepspeed_profile.txt'),
    )
    torch.cuda.reset_peak_memory_stats()
    
    training_args.task_name = data_args.task_name
    trainer = MinusSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        teacher_model=teacher_model,
        seq_len=data_args.max_input_length,
        output_seq_len=data_args.max_target_length,
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

    if model.head_mask is not None and (model.head_mask == 1).all():
        model.head_mask = None
    if model.intermediate_mask is not None and (model.intermediate_mask == 1).all():
        model.intermediate_mask = None
    if model.hidden_mask is not None and (model.hidden_mask == 1).all():
        model.hidden_mask = None
    
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        _ = model.eval()
        predictions = []
        references = []
        eval_dataloader = trainer.get_eval_dataloader(eval_dataset)
        model = model.to('cuda')
        with torch.no_grad():
            for example in tqdm(eval_dataloader):
                example = {k: v.to('cuda') for k, v in example.items()}
                output_ids = model.generate(input_ids=example['input_ids'], attention_mask=example['attention_mask'], max_length=data_args.max_target_length)
                predictions.extend(output_ids.cpu())
                references.extend(example['labels'].cpu())
        
        predictions = pad_sequence(predictions, batch_first=True, padding_value=tokenizer.pad_token_id)
        references = pad_sequence(references, batch_first=True, padding_value=tokenizer.pad_token_id)
        eval_pred = EvalPrediction(predictions=predictions, label_ids=references)
        metrics = compute_metrics(eval_pred)
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
        kwargs={k: v.to(model.device) for k, v in input_constructor(training_args.per_device_eval_batch_size, data_args.max_input_length, tokenizer, output_seq_len=data_args.max_target_length).items()},
        print_profile=True,
        detailed=True,
        output_file=os.path.join(training_args.output_dir, 'deepspeed_profile.txt'),
    )
    efficiency_results['model_flops'] = flops
    efficiency_results['model_macs'] = macs
    
    json.dump(efficiency_results, open(os.path.join(training_args.output_dir, 'efficiency_results.json'), 'w'), indent=4, sort_keys=True)
    if not os.path.exists(model_args.model_name_or_path):
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