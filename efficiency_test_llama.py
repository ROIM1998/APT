import os
import json
os.environ["WANDB_DISABLED"] = "true"
import sys
import transformers
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import loralib as lora
transformers.logging.set_verbosity_error()


from transformers import HfArgumentParser
from deepspeed.profiling.flops_profiler import get_model_profile
from args import InstructionDataTrainingArguments
from models.model_args import ModelArguments
from utils import build_trainer
from utils.utils import *
from utils.minus_utils import efficiency_testing, input_constructor, lora_to_linear
from args import MinusTrainingArguments
from models import build_model

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
    
    training_args.disable_tqdm = False
    config, tokenizer, model = build_model(model_args, data_args, training_args, token=os.environ.get('HF_TOKEN', None))
    # Explicitly converting the model to tf32
    model = model.to(dtype=torch.float32)
    print("Model parameter dtypes:", set([p.dtype for p in model.parameters()]))
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
    
    train_dataset, eval_dataset, _, _ = build_seq2seq_data(data_args, training_args, tokenizer)

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
        setattr(model, m_attr, None)
    model = model.to(dtype=torch.float32)
    
    model_param_num_merged = sum(p.numel() for p in model.parameters())
    trainer = build_trainer(data_args, training_args, model, tokenizer, train_dataset, eval_dataset, param_controller=None)
    
    # print("Merged model's performance: ", trainer.evaluate(), flush=True)
    print("Parmeter number reduced from {} to {}, with {} parameters removed".format(model_param_num, model_param_num_merged, model_param_num - model_param_num_merged), flush=True)
    
    efficiency_results = efficiency_testing(model, tokenizer, training_args.device, batch_sizes=[training_args.per_device_eval_batch_size], seq_len=512)
    json.dump(efficiency_results, open(os.path.join(training_args.output_dir, 'efficiency_results.json'), 'w'), indent=4, sort_keys=True) # save it first before risking to crash during deepspeed profiling
    flops, macs, params = get_model_profile(
        model,
        kwargs={k: v.to(model.device) for k, v in input_constructor(training_args.per_device_eval_batch_size, data_args.model_max_length, tokenizer).items()},
        print_profile=True,
        detailed=True,
        output_file=os.path.join(training_args.output_dir, 'deepspeed_profile.txt'),
    )
    efficiency_results['model_flops'] = flops
    efficiency_results['model_macs'] = macs
    
    json.dump(efficiency_results, open(os.path.join(training_args.output_dir, 'efficiency_results.json'), 'w'), indent=4, sort_keys=True)

if __name__ == '__main__':
    main()