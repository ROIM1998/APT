import os
import json
os.environ["WANDB_DISABLED"] = "true"
import sys
import transformers
import torch
import loralib as lora
transformers.logging.set_verbosity_error()


from transformers import HfArgumentParser
from deepspeed.profiling.flops_profiler import get_model_profile
from args import DataTrainingArguments
from models.model_args import ModelArguments
from utils import build_trainer
from utils.utils import *
from utils.minus_utils import efficiency_testing, input_constructor, lora_to_linear
from args import MinusTrainingArguments
from models import build_model

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
    
    training_args.disable_tqdm = False
    t_name, raw_datasets = get_raw_datasets(model_args, data_args, training_args)
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets, force_model_shape_deduction=True)
    if os.path.exists(os.path.join(model_args.model_name_or_path, 'head_mask.pt')):
        model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt'))
    if os.path.exists(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt')):
        model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt'))
    if os.path.exists(os.path.join(model_args.model_name_or_path, 'hidden_mask.pt')):
        model.hidden_mask = torch.load(os.path.join(model_args.model_name_or_path, 'hidden_mask.pt'))
    model.prune_model_with_masks()
    if isinstance(model.head_mask, torch.Tensor) and (model.head_mask == 1).all().item():
        model.head_mask = None
    if isinstance(model.intermediate_mask, torch.Tensor) and (model.intermediate_mask == 1).all().item():
        model.intermediate_mask = None
    if  isinstance(model.hidden_mask, torch.Tensor) and (model.hidden_mask == 1).all().item():
        model.hidden_mask = None
    
    MODEL_GENERATIVE = any(['decoder' in n for n, _ in model.named_parameters()])
    train_dataset, eval_dataset, predict_dataset, is_regression = build_data(model_args, data_args, training_args, model, tokenizer, config, raw_datasets, generative=MODEL_GENERATIVE)

    model.eval()
    for module in model.modules():
        if isinstance(module, lora.Linear):
            module.scaling = model_args.lora_alpha / model_args.lora_r
    
    model_param_num = sum(p.numel() for p in model.parameters())
    for n, m in dict(model.named_modules()).items():
        for child_name, child in dict(m.named_children()).items():
            if isinstance(child, lora.Linear):
                print("Merging layer {}".format(n + '.' + child_name))
                delattr(m, child_name)
                merged_layer = lora_to_linear(child)
                setattr(m, child_name, merged_layer)
    
    model = model.to(training_args.device)
    for m_attr in ['head_mask', 'intermediate_mask', 'hidden_mask']:
        if getattr(model, m_attr, None) is not None:
            setattr(model, m_attr, getattr(model, m_attr).to(training_args.device))
    
    model_param_num_merged = sum(p.numel() for p in model.parameters())
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=None)
    
    # print("Merged model's performance: ", trainer.evaluate(), flush=True)
    print("Parmeter number reduced from {} to {}, with {} parameters removed".format(model_param_num, model_param_num_merged, model_param_num - model_param_num_merged), flush=True)
    
    efficiency_results = efficiency_testing(model, tokenizer, training_args.device, model_generative=MODEL_GENERATIVE)
    flops, macs, params = get_model_profile(
        model,
        kwargs={k: v.to(model.device) for k, v in input_constructor(training_args.per_device_eval_batch_size, data_args.max_seq_length, tokenizer, output_seq_len=2).items()} if MODEL_GENERATIVE else {k: v.to(model.device) for k, v in input_constructor(training_args.per_device_eval_batch_size, data_args.max_seq_length, tokenizer).items()},        print_profile=True,
        detailed=True,
        output_file=os.path.join(training_args.output_dir, 'deepspeed_profile.txt'),
    )
    efficiency_results['model_flops'] = flops
    efficiency_results['model_macs'] = macs
    
    json.dump(efficiency_results, open(os.path.join(training_args.output_dir, 'efficiency_results.json'), 'w'), indent=4, sort_keys=True)

if __name__ == '__main__':
    main()