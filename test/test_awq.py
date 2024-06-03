# Test ElasticLlama model with pruning consistency and tuning consistency
import torch
import os
import sys
import gc
import torch.nn as nn
import seaborn as sns
import loralib as lora

from transformers import (HfArgumentParser)
from args import InstructionDataTrainingArguments
from models.model_args import ModelArguments
from args import MinusTrainingArguments
from utils.utils import *
from utils import build_dataloader
from models import build_model
from trainer.param_control import ParamController
from trainer.trainer_minus import MinusTrainer
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.stats import entropy, kurtosis
from utils.minus_utils import kurtosis as minus_kurtosis

def pseudo_quantize_tensor(w, n_bit=8,
                           zero_point=True, q_group_size=-1,
                           inplace=False,
                           get_scale_zp=False
                           ):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = - 2 ** (n_bit - 1)
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        ((w.div_(scales).round_().add_(zeros)).clamp_(
            min_int, max_int).sub_(zeros)).mul_(scales)
    else:
        w = (torch.clamp(torch.round(w / scales) +
                         zeros, min_int, max_int) - zeros) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w

# weight quantization
@torch.no_grad()
def auto_clip_layer(w, input_feat, n_bit, q_config,
                    n_grid=20,
                    max_shrink=0.5,
                    n_sample_token=512):
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    group_size = q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0::input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)

    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []

    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size: (i_b + 1) * oc_batch_size]

        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = - max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize_tensor(cur_w, n_bit=n_bit, **q_config)
            cur_out = (input_feat * q_w).sum(dim=-1)

            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)

    del input_feat
    del org_out
    gc.collect()
    torch.cuda.empty_cache()
    return best_max_val.squeeze(1)

@torch.no_grad()
def minus_kurtosis(a: torch.Tensor, axis: int = 0, fisher: bool = True, bias: bool = True):
    """Compute the kurtosis (Fisher or Pearson) of a distribution.

    Kurtosis is the fourth central moment divided by the square of the
    variance. If Fisher's definition is used, then 3.0 is subtracted from
    the result to give 0.0 for a normal distribution.

    If bias is False then the kurtosis is calculated using k statistics to
    eliminate bias coming from biased moment estimators

    Use `kurtosistest` to see if result is close enough to normal.

    Parameters
    ----------
    a : torch.Tensor
        Data for which the kurtosis is calculated.
    axis : int or None, optional
        Axis along which the kurtosis is calculated. Default is 0.
        If None, compute over the whole array `a`.

    Returns
    -------
    kurtosis : torch.Tensor
        The kurtosis of values along an axis, returning NaN where all values
        are equal.

    """
    # Compute the mean of the tensor
    mu = torch.mean(a, axis, keepdim=True)

    # Compute the centered values
    centered = a - mu

    # Compute the zscore
    zscores = centered / torch.std(a, axis, keepdim=True)

    # Compute kurtosis
    kurt = torch.mean(zscores.pow(4), axis, keepdim=True)

    return kurt - 3 if fisher else kurt

def main():
    sys.argv = ['test_bert.py',
            '--output_dir',
            './output/test_llama_virtual_pruning/',
            '--model_name_or_path',
            'meta-llama/Llama-2-7b-hf',
            '--do_train',
            '--task_name',
            'alpaca',
            '--data_path', 
            'data/sft/alpaca_data_gpt4.json',
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
            '--pruner_type',
            'running_fisher',
            '--pre_tuning_scorer',
            'backward_running_hidden_states_salience',
            '--pre_tuning_constraint',
            '0.8',
            '--pre_tuning_pruner',
            'running_fisher',
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
    model.layer_transformation.weight = nn.Parameter(torch.eye(model.config.hidden_size))
    train_dataset, eval_dataset, _, _ = build_seq2seq_data(data_args, training_args, tokenizer)
    dataloader = build_dataloader(train_dataset, training_args.per_device_train_batch_size, data_args, training_args, tokenizer)
    
    inputs = next(iter(dataloader))
    model = model.to(training_args.device)
    inputs = {k: v.to(training_args.device) for k, v in inputs.items()}
    
    model.head_mask = model.head_mask.to(training_args.device)
    model.intermediate_mask = model.intermediate_mask.to(training_args.device)
    model.hidden_mask = model.hidden_mask.to(training_args.device)
    
    
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
    
    trainer = MinusTrainer(
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
    
    # Test out input-activation & weight magnitude
    collected_inputs = []
    def collect_input_hook(module, input, output):
        collected_inputs.append(input[0])
        
    handler = model.model.layers[0].self_attn.o_proj.register_forward_hook(collect_input_hook)
    with torch.no_grad():
        outputs = model(**inputs)
    
    handler.remove()
    module, mha_hidden_states = model.model.layers[0].self_attn.o_proj, collected_inputs[0]
    
    mha_kurtosis = []
    mha_hidden_unsqueezed = mha_hidden_states.detach().cpu().mean(0).mean(0).unsqueeze(0) # (1, batch_size, seq_len, num_heads x head_size)
    # Convert to (1, batch_size, seq_len, num_heads, head_size)
    # mha_hidden_unsqueezed = mha_hidden_unsqueezed.view(mha_hidden_unsqueezed.shape[0], mha_hidden_unsqueezed.shape[1], mha_hidden_unsqueezed.shape[2], -1, 4096 // 32)
    mha_hidden_unsqueezed = mha_hidden_unsqueezed.view(1, -1, 4096 // 32)
    
    weight_merged = (module.weight.detach().cpu() + (module.lora_B.detach().cpu() @ module.lora_A.detach().cpu()) * module.scaling) if isinstance(module, lora.Linear) else module.weight.detach().cpu() # shape: (hidden_size, num_heads x head_size)
    # Convert to (hidden_size, num_heads, head_size)
    weight_unsqueezed = weight_merged
    weight_unsqueezed = weight_unsqueezed.view(weight_unsqueezed.shape[0], -1, 4096 // 32)
    activation = (mha_hidden_unsqueezed * weight_unsqueezed) # shape: (hidden size, num_heads, head_size) 
    act_kurtosis = kurtosis(activation.permute(1, 2, 0).reshape(-1, activation.shape[1]).numpy())
    minus_act_kurtosis = minus_kurtosis(activation.permute(1, 2, 0).reshape(-1, activation.shape[1]))
    mha_kurtosis.append(act_kurtosis)
    
    scorer = trainer.pre_tuning_pruning_scorer
    pruner = trainer.pre_tuning_pruning_pruner
    scorer.step()
    
    print(scorer.mha_kurtosis_history[0])
    pruner.update_mask(trainer.starting_mac_constraint, is_last=True)
    # if getattr(trainer.model, 'layer_transformation', None) is not None:
    #     index = torch.LongTensor(trainer.model.hidden_mask.nonzero().squeeze().tolist()).to(trainer.args.device)
    #     trainer.model.layer_transformation = prune_layer(trainer.model.layer_transformation, index, dim=0)
    scorer.end()
    

    # Check the distribution of each weight and see if there are any outliers
    weight = model.model.layers[0].self_attn.q_proj.weight.detach().cpu().numpy().flatten()
    sns.histplot(weight)
    plt.savefig('query_layer_0.png')
    plt.clf()
    weight = model.model.layers[1].self_attn.q_proj.weight.detach().cpu().numpy().flatten()
    sns.histplot(weight)
    plt.savefig('query_layer_1.png')
    plt.clf()
    torch.save(model.head_mask, 'apt_head_mask.pt')
    torch.save(model.intermediate_mask, 'apt_intermediate_mask.pt')
    
    mt_head_mask = torch.load('llama_output/meta-llama/Llama-2-7b-hf/mt_pruned/constraint_0.8/batches_64/pruning_head_mask.pt', map_location='cpu')
    mt_intermediate_mask = torch.load('llama_output/meta-llama/Llama-2-7b-hf/mt_pruned/constraint_0.8/batches_64/pruning_intermediate_mask.pt', map_location='cpu')
    apt_head_mask = model.head_mask.view(model.config.num_hidden_layers, -1)
    apt_intermediate_mask = model.intermediate_mask.view(model.config.num_hidden_layers, -1)
    apt_gradual_head_mask = torch.load('llama_output/meta-llama/Llama-2-7b-hf/alpaca_gpt4/bz4/elastictuning_virtualprune_pre-tuning-prune-1.0-nodistill/mac0.8/epoch15/numprune10/sparsity_warmup1/pruning_start-1/pruning_stop3/lora_r8/lora_alpha16/warmup_paramdq:0-31,dv:0-31/teacher_paramdq:0-31,dv:0-31/final_head_mask.pt', map_location='cpu')
    apt_gradual_intermediate_mask = torch.load('llama_output/meta-llama/Llama-2-7b-hf/alpaca_gpt4/bz4/elastictuning_virtualprune_pre-tuning-prune-1.0-nodistill/mac0.8/epoch15/numprune10/sparsity_warmup1/pruning_start-1/pruning_stop3/lora_r8/lora_alpha16/warmup_paramdq:0-31,dv:0-31/teacher_paramdq:0-31,dv:0-31/final_intermediate_mask.pt', map_location='cpu')
    
    # Determining a heuristic outlier rule
    @torch.no_grad()
    def outlier_count(weight: torch.Tensor, num_heads=32):
        weight_per_head = weight.view(num_heads, -1)
        avgs = weight_per_head.mean(dim=1, keepdim=True)
        stds = weight_per_head.std(dim=1, keepdim=True)
        outliers = torch.sum(torch.abs(weight_per_head - avgs) > 10 * stds, dim=1)
        return outliers
    
    for i in range(model.config.num_hidden_layers):
        print("query layer {}".format(i), end=" ")
        print(outlier_count(model.model.layers[i].self_attn.q_proj.weight))
        
    for i in range(model.config.num_hidden_layers):
        print("key layer {}".format(i), end=" ")
        print(outlier_count(model.model.layers[i].self_attn.k_proj.weight))
        
    for i in range(model.config.num_hidden_layers):
        print("value layer {}".format(i), end=" ")
        print(outlier_count(model.model.layers[i].self_attn.v_proj.weight))
        
    for i in range(model.config.num_hidden_layers):
        print("gate layer {}".format(i), end=" ")
        print(outlier_count(model.model.layers[i].mlp.gate_proj.weight))
        
    def count_outliers_iqr(data: torch.Tensor, num_heads=32):
        """
        Count the number of outliers in a dataset using the IQR method.

        Args:
        - data (np.array): An array of data points.

        Returns:
        - int: The number of outliers.
        """
        # Convert the data into chunks of numpy arrays
        data = data.view(num_heads, -1).numpy()
        
        # Compute the first and third quartiles (Q1, Q3)
        Q1 = np.percentile(data, 25, axis=1, keepdims=True)
        Q3 = np.percentile(data, 75, axis=1, keepdims=True)
        
        # Compute the Interquartile Range (IQR)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count values outside of the bounds
        outlier_count = np.sum((data < lower_bound) | (data > upper_bound), axis=1)
        
        return outlier_count
    
    count_outliers_iqr(model.model.layers[3].self_attn.q_proj.weight.detach().cpu()).sum() / (4096 ** 2)
    entropy(model.model.layers[0].self_attn.q_proj.weight.detach().cpu().view(32, -1).numpy())
    kurtosis(model.model.layers[0].self_attn.q_proj.weight.detach().cpu().view(32, -1).T.numpy())
    kurtosis(model.model.layers[1].self_attn.q_proj.weight.detach().cpu().view(32, -1).T.numpy())
    
    # Static analysis
    # Comparing the kurtosis of the mt-pruned model and the apt-pruned model
    mt_kurtosis = []
    
if __name__ == '__main__':
    main()