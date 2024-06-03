import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import json
from transformers import HfArgumentParser, AutoTokenizer, AutoConfig
from args import DataTrainingArguments
from models.cofi_modeling_bert import CoFiBertForSequenceClassification
from models.model_args import ModelArguments
from utils.utils import *
from utils.minus_utils import bench_latency, compute_throughput
from trainer.model_arch import get_layers, get_encoder
from args import MinusTrainingArguments
from loralib.layers import DistillLinear

def main():
    sys.argv = ['test_cofi.py',
            '--output_dir',
            './output/neuron_importance/',
            '--model_name_or_path',
            '/home/zbw/projects/CoFiPruning/out/cofi-mnli-s95',
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
    
    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in [
            "float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=t_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    Model = CoFiBertForSequenceClassification
    model = Model.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    ) #! inside the function, we get the original struct  #! CofiBertForSequenceClassification
        
    model = model.to(training_args.device)
    model.eval()

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