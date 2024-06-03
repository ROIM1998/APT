import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import torch
import json
from transformers import HfArgumentParser
from deepspeed.profiling.flops_profiler import get_model_profile
from args import DataTrainingArguments
from models import build_model
from utils import build_dataloader, build_trainer
from models.model_args import ModelArguments
from utils.utils import *
from utils.minus_utils import efficiency_testing, input_constructor, compare_parameters
from utils.analysis_utils import gen_run_report
from args import MinusTrainingArguments
from loralib.layers import LoRALayer

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
        
    t_name, raw_datasets = get_raw_datasets(model_args, data_args, training_args)
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
    MODEL_GENERATIVE = any(['decoder' in n for n, _ in model.named_parameters()])
    train_dataset, eval_dataset, predict_dataset, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets, generative=MODEL_GENERATIVE)

    model = model.to(training_args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALayer):
            m.eval()
    
    model.eval()
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=None)
    model.clear_masks()
    efficiency_results = efficiency_testing(model, tokenizer, training_args.device)

    flops, macs, params = get_model_profile(
        model,
        kwargs={k: v.to(model.device) for k, v in input_constructor(training_args.per_device_eval_batch_size, data_args.max_seq_length, tokenizer, output_seq_len=2).items()} if MODEL_GENERATIVE else {k: v.to(model.device) for k, v in input_constructor(training_args.per_device_eval_batch_size, data_args.max_seq_length, tokenizer).items()},        print_profile=True,
        detailed=True,
        output_file=os.path.join(training_args.output_dir, 'deepspeed_profile.txt'),
    )
    efficiency_results['model_flops'] = flops
    efficiency_results['model_macs'] = macs
    json.dump(efficiency_results, open(os.path.join(training_args.output_dir, 'efficiency_results.json'), 'w'), indent=4, sort_keys=True)
    # run_report = gen_run_report(training_args.output_dir)
    # run_report['train_runtime_per_epoch'] = run_report['train_runtime'] / training_args.num_train_epochs
    # json.dump(run_report, open(os.path.join(training_args.output_dir, 'run_report.json'), 'w'), indent=4, sort_keys=True)
    
    result = trainer.evaluate()
    json.dump(result, open(os.path.join(training_args.output_dir, 'eval_results.json'), 'w'), indent=4, sort_keys=True)    

    
if __name__ == '__main__':
    main()