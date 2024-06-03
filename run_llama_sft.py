import os
import sys
import json
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import loralib as lora

from tqdm import tqdm
from transformers import (HfArgumentParser, set_seed)
from args import InstructionDataTrainingArguments
from models.model_args import ModelArguments
from args import MinusTrainingArguments
from utils.utils import *
from utils import build_dataloader, build_trainer
from utils.minus_utils import efficiency_testing
from utils.analysis_utils import gen_run_report
from models import build_model
from loralib.layers import LoRALayer

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


def main():
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
    config, tokenizer, model = build_model(model_args, data_args, training_args, token=os.environ.get('HF_TOKEN', None))
    train_dataset, eval_dataset, _, _ = build_seq2seq_data(data_args, training_args, tokenizer)
    # TODO: using MMLU validate set as the eval set for generalization testing
    
    if training_args.teacher_path is None:
        teacher_model = None
    else:
        _, _, teacher_model = build_model(model_args, data_args, training_args, token=os.environ.get('HF_TOKEN', None))
        teacher_model.head_mask = torch.load(os.path.join(training_args.teacher_path, 'head_mask.pt')).to(training_args.device) if os.path.exists(os.path.join(training_args.teacher_path, 'head_mask.pt')) else None
        if (teacher_model.head_mask == 1).all():
            teacher_model.head_mask = None
        teacher_model.intermediate_mask = torch.load(os.path.join(training_args.teacher_path, 'intermediate_mask.pt')).to(training_args.device) if os.path.exists(os.path.join(training_args.teacher_path, 'intermediate_mask.pt')) else None
        if (teacher_model.intermediate_mask == 1).all():
            teacher_model.intermediate_mask = None
        teacher_model.hidden_mask = torch.load(os.path.join(training_args.teacher_path, 'hidden_mask.pt')).to(training_args.device) if os.path.exists(os.path.join(training_args.teacher_path, 'hidden_mask.pt')) else None
        if (teacher_model.hidden_mask == 1).all():
            teacher_model.hidden_mask = None

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
    
    training_args.task_name = data_args.task_name
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset)

    # Training
    if training_args.do_train:
        if os.path.exists(model_args.model_name_or_path):
            print("Evaluating pre-trained model...")
            print(trainer.evaluate())
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
        metrics = trainer.evaluate()
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)

    # TODO: merge LoRA layers after training for efficiency during efficiency & deepspeed profiler testing
    model.eval()
    efficiency_results = efficiency_testing(model, tokenizer, training_args.device)

    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.eval()
    
    json.dump(efficiency_results, open(os.path.join(training_args.output_dir, 'efficiency_results.json'), 'w'), indent=4, sort_keys=True)
    run_report = gen_run_report(training_args.output_dir)
    run_report['train_runtime_per_epoch'] = run_report['train_runtime'] / training_args.num_train_epochs
    json.dump(run_report, open(os.path.join(training_args.output_dir, 'run_report.json'), 'w'), indent=4, sort_keys=True)


if __name__ == "__main__":
    main()