import os
os.environ["WANDB_DISABLED"] = "true"
import torch
import sys
import json

from tqdm import tqdm
from transformers import (HfArgumentParser)
from utils.utils import *
from args import MinusTrainingArguments, InstructionDataTrainingArguments
from models import build_model
from models.model_args import ModelArguments
from torch.utils.data import DataLoader
from utils.fisher_utils.efficiency.param import *

IGNORE_INDEX = -100

if __name__ == '__main__':
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
    # training_args.disable_tqdm = False

    config, tokenizer, model = build_model(model_args, data_args, training_args)
    train_dataset, eval_dataset, _, _ = build_seq2seq_data(data_args, training_args, tokenizer, do_split=False, no_output=True)
    
    # Avg davinci-003 token length: 127.85, max: 1050
    _ = model.eval().cuda()
    model.clear_masks()
    batch_size = training_args.per_device_eval_batch_size
    
    @dataclass
    class DataCollatorForAlpacaEval(object):
        """Collate examples for AlpacaEval. The only difference is that we pad to left instead of right."""

        tokenizer: transformers.PreTrainedTokenizer

        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
            pad_id = self.tokenizer.pad_token_id
            max_length = max(len(i) for i in input_ids)
            input_ids = torch.stack([torch.cat([torch.full((max_length - len(i),), pad_id, dtype=torch.long), i]) for i in input_ids])
            labels = torch.stack([torch.cat([torch.full((max_length - len(i),), pad_id, dtype=torch.long), i]) for i in labels])
            
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )
    data_collator = DataCollatorForAlpacaEval(tokenizer=tokenizer)
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    
    all_raw_gens = []
    all_gens = []
    
    if os.path.exists(os.path.join(training_args.output_dir, 'alpaca_eval_tmp.txt')):
        loaded_gens = open(os.path.join(training_args.output_dir, 'alpaca_eval_tmp.txt'), 'r').readlines()
        loaded_gens = [json.loads(g) for g in loaded_gens]
    else:
        loaded_gens = []
    # Support checkpointing with interrupted sbatch jobs
    tmp_file = open(os.path.join(training_args.output_dir, 'alpaca_eval_tmp.txt'), 'w')
    with torch.no_grad():
        for i, inputs in tqdm(enumerate(dataloader)):
            if i < len(loaded_gens):
                all_gens.append(loaded_gens[i])
                continue
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs   ['attention_mask'], max_length=512, pad_token_id=tokenizer.pad_token_id)
            gen = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_raw_gens.append(gen)
            # Find the answer by getting the results in between "###Response:\n" and "\n###"
            gen = [g.split("### Response:")[1].split("\n###")[0].strip() if "### Response:" in g else "" for g in gen]
            if len(all_gens) == 0:
                print("First batch of generations: ")
                for g in gen:
                    print(g)
            all_gens.append(gen)
            tmp_file.write(json.dumps(gen) + '\n')
    
    tmp_file.close()
    
    all_gens = [item for sublist in all_gens for item in sublist]
    
    eval_data = json.load(open(data_args.data_path, 'r'))
    suffix = model_args.model_name_or_path.split('/')[-1]
    assert len(all_gens) == len(eval_data)
    for v in eval_data:
        v['output'] = all_gens.pop(0)
        v['generator'] = 'llama2_%s' % suffix
        
    json.dump(eval_data, open(os.path.join(training_args.output_dir, 'alpaca_eval_output.json'), 'w'), indent=4)