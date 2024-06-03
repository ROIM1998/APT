# Test ElasticLlama model with pruning consistency and tuning consistency
import os
import sys
import torch

from transformers import (HfArgumentParser)
from args import InstructionDataTrainingArguments
from models.model_args import ModelArguments
from args import MinusTrainingArguments
from utils.utils import *
from utils import build_dataloader, build_trainer
from models import build_model
from trainer.param_control import ParamController
from trainer.trainer_seq2seq_minus import MinusSeq2SeqTrainer

def main():
    sys.argv = ['test_bert.py',
            '--output_dir',
            './output/test_llama_elastictuning/',
            '--model_name_or_path',
            'meta-llama/Llama-2-7b-hf',
            '--do_train',
            '--task_name',
            'alpaca',
            '--data_path', 
            'data/sft/alpaca_data.json',
            '--bf16',
            'True',
            '--output_dir',
            'output/llama_lora_alpaca/epoch_30',
            '--num_train_epochs',
            '30',
            '--per_device_train_batch_size',
            '4',
            '--per_device_eval_batch_size',
            '4',
            '--gradient_accumulation_steps',
            '8',
            '--evaluation_strategy',
            "no",
            '--save_strategy',
            "steps",
            '--save_steps',
            '2000',
            '--save_total_limit',
            '1',
            '--learning_rate',
            '2e-4', # LoRA learning rate
            '--weight_decay',
            '0.',
            '--warmup_ratio',
            '0.03',
            '--lr_scheduler_type',
            "cosine",
            '--logging_steps',
            '1',
            '--apply_lora',
            '--do_distill',
            '--lora_r',
            '8',
            '--lora_alpha',
            '16',
            '--report_to',
            'none',
            # '--fsdp',
            # "full_shard auto_wrap",
            # '--fsdp_transformer_layer_cls_to_wrap',
            # 'LlamaDecoderLayer',
            '--tf32',
            'True',
            '--distill_mapping_strategy',
            'dynamic_block_teacher_dynamic_student',
            ]
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
    config, tokenizer, model = build_model(model_args, data_args, training_args, token=os.environ.get('HF_TOKEN', None))
    train_dataset, eval_dataset, _, _ = build_seq2seq_data(data_args, training_args, tokenizer)
    dataloader = build_dataloader(train_dataset, training_args.per_device_train_batch_size, data_args, training_args, tokenizer)
    
    inputs = next(iter(dataloader))
    model = model.to(training_args.device)
    inputs = {k: v.to(training_args.device) for k, v in inputs.items()}
    
    model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt'))
    model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt'))
    model.hidden_mask = torch.load(os.path.join(model_args.model_name_or_path, 'hidden_mask.pt'))
    
    model.train()
    
    teacher_keys = ['dec_self_query', 'dec_self_value']
    teacher_config = {
        k: [i for i in range(config.num_hidden_layers)] for k in teacher_keys
    }
    param_controller = ParamController(
        model,
        teacher_config=teacher_config,
        lora_with_bias=False,
    )
        
    @dataclass
    class DataCollatorForSupervisedDataset(object):
        """Collate examples for supervised fine-tuning."""

        tokenizer: transformers.PreTrainedTokenizer

        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )
    data_collator = DataCollatorForSupervisedDataset(tokenizer)
    
    trainer = MinusSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        teacher_model=None,
        param_controller=param_controller,
        seq_len=512,
        cls_task=False,
    )
    param_controller.convert_to_pruning_lora_teacher()
    param_controller.model_as_teacher()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(sum(p.numel() for p in model.parameters()))
    
    
    with torch.no_grad():
        teacher_outputs = model(**inputs, output_hidden_states=True, use_cache=False, use_teacher=True, pass_mask=False)
        
    student_outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    
    # De-variance the hidden states for stable distillation
    decoder_teacher_states = [v * torch.rsqrt(v.pow(2).mean(dim=-1, keepdim=True) + 1e-6) for v in teacher_outputs[-1]]
    decoder_student_states = [v * torch.rsqrt(v.pow(2).mean(dim=-1, keepdim=True) + 1e-6) for v in student_outputs[-1]]
    
    decoder_layer_loss = trainer.calculate_layer_distillation_loss(decoder_teacher_states, decoder_student_states, is_decoder=True)
    
if __name__ == '__main__':
    main()