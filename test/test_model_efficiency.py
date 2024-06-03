import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import json
import re
from transformers import HfArgumentParser
from args import DataTrainingArguments
from models import build_model
from models.model_args import ModelArguments
from utils.utils import *
from utils.minus_utils import bench_latency, compute_throughput
from trainer.model_arch import get_layers, get_encoder
from args import MinusTrainingArguments
from loralib.layers import DistillLinear

def main():
    sys.argv = ['neuron_importance.py',
            '--output_dir',
            './output/neuron_importance/',
            '--model_name_or_path',
            'output/roberta-base_lora_mnli/epoch5/lora_r8/lora_alpha16/best_model',
            '--task_name',
            'mnli',
            '--do_train',
            '--do_eval',
            '--max_seq_length',
            '128',
            '--per_device_train_batch_size',
            '32',
            '--per_device_eval_batch_size',
            '32',
            '--apply_lora',
            '--do_distill',
            '--lora_r',
            '8'
            ]
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
    t_name, raw_datasets = get_raw_datasets(model_args, data_args, training_args)
    
    for root, _, _ in os.walk('output/roberta-base_lora_minus_mnli_once_global_distill_full_exp_shorter'):
        if 'best_model' in root and 'finetuned' not in root and not os.path.exists(os.path.join(root, 'efficiency_results.json')):
            model_args.model_name_or_path = os.path.join(root)
            model_args.lora_r = int(re.search(r'lora_r(\d+)/', root)[1])
            config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets)
            model.head_mask, model.intermediate_mask = None, None
            model = model.to(training_args.device)
            _ = model.eval()
            for p in model.parameters():
                p.requires_grad = False
            for layer in get_layers(model):
                if isinstance(layer.intermediate.dense, DistillLinear):
                    layer.intermediate.dense.eval(merge=True)

            bz32_bench = bench_latency(model, batch_size=32, seq_len=128, tokenizer=tokenizer)
            bz32_throughput = compute_throughput(model, tokenizer, batch_size=32, seq_len=128)
            bz128_bench = bench_latency(model, batch_size=128, seq_len=128, tokenizer=tokenizer)
            bz128_throughput = compute_throughput(model, tokenizer, batch_size=128, seq_len=128)
            overall_results = {
                **{'bz32_' + k: v for k, v in bz32_bench.items()},
                **{'bz128_' + k: v for k, v in bz128_bench.items()},
                'bz32_throughput': bz32_throughput,
                'bz128_throughput': bz128_throughput,
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'encoder_num_parameters': sum(p.numel() for p in get_encoder(model).parameters()),
            }
            json.dump(overall_results, open(os.path.join(model_args.model_name_or_path, 'efficiency_results.json'), 'w'), indent=4, sort_keys=True)
    
if __name__ == '__main__':
    main()