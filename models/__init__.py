import os
import torch
import gc
import json
from transformers import AutoConfig, AutoTokenizer
from loralib.layers import LoRALayer, PruningLinear, DistillLinear
from tqdm import tqdm
from utils.utils import get_label
from utils.minus_utils import model_layer_switch, lora_to_linear, lora_to_distill, lora_to_prunelora, linear_to_lora
from trainer.model_arch import get_ffn1, get_mha_proj
from utils.alpaca_utils import smart_tokenizer_and_embedding_resize
from .modeling_bert import CoFiBertForSequenceClassification, AdaPBertForQuestionAnswering
# from .modeling_roberta_backup import CoFiRobertaForSequenceClassification, NewRobertaForQuestionAnswering
from .modeling_roberta import CoFiRobertaForSequenceClassification, NewRobertaForQuestionAnswering
# from .modeling_t5_backup import AdaPT5ForConditionalGeneration
from .modeling_t5 import AdaPT5ForConditionalGeneration
from .modeling_mt5 import AdaPMT5ForConditionalGeneration
from .modeling_llama import ElasticLlamaForCausalLM

def check_model_weights_shape_matching(model, weights):
    if weights is None:
        return False
    if 'bert' in model.config.model_type:
        for i in range(model.config.num_hidden_layers):
            if get_mha_proj(model, i).weight.shape != weights['%s.encoder.layer.%d.attention.output.dense.weight' % (model.config.model_type, i)].shape:
                return False
            if get_ffn1(model, i).dense.weight.shape != weights['%s.encoder.layer.%d.intermediate.dense.weight' % (model.config.model_type, i)].shape:
                return False
        return True
    elif 't5' in  model.config.model_type:
        for i in range(model.config.num_layers):
            if 'encoder.block.%d.layer.0.SelfAttention.o.weight' % i not in weights or get_mha_proj(model, i).weight.shape != weights['encoder.block.%d.layer.0.SelfAttention.o.weight' % i].shape:
                return False
            if 'encoder.block.%d.layer.1.DenseReluDense.wo.weight' % i not in weights or get_ffn1(model, i).wo.weight.shape != weights['encoder.block.%d.layer.1.DenseReluDense.wo.weight' % i].shape:
                return False
        for i in range(model.config.num_decoder_layers):
            if 'decoder.block.%d.layer.0.SelfAttention.o.weight' % i not in weights or get_mha_proj(model, i, use_decoder=True).weight.shape != weights['decoder.block.%d.layer.0.SelfAttention.o.weight' % i].shape:
                return False
            if 'decoder.block.%d.layer.1.EncDecAttention.o.weight' % i not in weights or get_mha_proj(model, i, use_decoder=True, use_cross_attention=True).weight.shape != weights['decoder.block.%d.layer.1.EncDecAttention.o.weight' % i].shape:
                return False
            if 'decoder.block.%d.layer.2.DenseReluDense.wo.weight' % i not in weights or get_ffn1(model, i, use_decoder=True).wo.weight.shape != weights['decoder.block.%d.layer.2.DenseReluDense.wo.weight' % i].shape:
                return False
    elif model.config.model_type == 'llama':
        for i in range(model.config.num_hidden_layers):
            if get_mha_proj(model, i, use_decoder=True).weight.shape != weights['model.layers.%d.self_attn.o_proj.weight' % i].shape:
                return False
            if get_ffn1(model, i, use_decoder=True).gate_proj.weight.shape != weights['model.layers.%d.mlp.gate_proj.weight' % i].shape:
                return False
        return True


def convert_layers_based_on_ckpt(model, params):
    loras = [n for n in params if 'lora' in n]
    layer_names = list(set(['.'.join(n.split('.')[:-1]) for n in loras]))
    named_modules = dict(model.named_modules())
    if getattr(model, 'layer_transformation', None) is not None:
        model.layer_transformation.weight = torch.nn.Parameter(torch.zeros(model.layer_transformation.weight.shape).to(device=model.device, dtype=model.dtype))
    for layer_name in layer_names:
        ckpt_lora_r = params[layer_name + '.lora_B'][1] if layer_name + '.lora_B' in params else 0
        if '.' in layer_name:
            layer_key = '.'.join(layer_name.split('.')[:-1])
            if layer_key in named_modules:
                parent_layer = named_modules[layer_key]
            else:
                continue
            layer_attr = layer_name.split('.')[-1]
        else:
            parent_layer = model
            layer_attr = layer_name
        if layer_name not in named_modules:
            continue
        if not isinstance(named_modules[layer_name], LoRALayer) or named_modules[layer_name].r != ckpt_lora_r:
            # print("Converting layer %s to LoRA" % layer_name)
            setattr(parent_layer, layer_attr, linear_to_lora(getattr(parent_layer, layer_attr), r=ckpt_lora_r, lora_alpha=model.config.lora_alpha))
        # Check if any transformation or teacher layers exist in the checkpoint
        if layer_name + '.lora_B' in params:
            r = params[layer_name + '.lora_B'][1]
        else:
            r = 0
        if layer_name + '.out_transformation' in params:
            out_retained_indices = params[layer_name + '.out_transformation'].nonzero().cpu()[:, 1].tolist()
        else:
            out_retained_indices = None
        if layer_name + '.in_transformation' in params:
            in_retained_indices = params[layer_name + '.in_transformation'].nonzero().cpu()[:, 0].tolist()
        else:
            in_retained_indices = None
        if layer_name + '.teacher_lora_B' in params:
            teacher_r = params[layer_name + '.teacher_lora_B'][1]
            if layer_name + '.teacher_out_transformation' in params:
                teacher_out_retained_indices = params[layer_name + '.teacher_out_transformation'].nonzero().cpu()[:, 1].tolist()
            else:
                teacher_out_retained_indices = None
            if layer_name + '.teacher_in_transformation' in params:
                teacher_in_retained_indices = params[layer_name + '.teacher_in_transformation'].nonzero().cpu()[:, 0].tolist()
            else:
                teacher_in_retained_indices = None
            setattr(parent_layer, layer_attr, lora_to_distill(getattr(parent_layer, layer_attr), r=r, lora_alpha=model.config.lora_alpha, in_retained_indices=in_retained_indices, out_retained_indices=out_retained_indices, teacher_r=teacher_r, teacher_out_retained_indices=teacher_out_retained_indices, teacher_in_retained_indices=teacher_in_retained_indices))
        else:
            setattr(parent_layer, layer_attr, lora_to_prunelora(getattr(parent_layer, layer_attr), r=r, lora_alpha=model.config.lora_alpha, out_retained_indices=out_retained_indices, in_retained_indices=in_retained_indices))
    model.print_lora_info_by_layer()
    
def model_post_processing(model, model_args, data_args, training_args, tokenizer, **kwargs):
    if model.config.model_type == 'llama':
        DEFAULT_PAD_TOKEN = "[PAD]"
        DEFAULT_EOS_TOKEN = "</s>"
        DEFAULT_BOS_TOKEN = "<s>"
        DEFAULT_UNK_TOKEN = "<unk>"
        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )

def build_model(model_args, data_args, training_args, t_name=None, raw_datasets=None, determined_model_path=None, force_model_shape_deduction: bool = True, retain_scaling: bool = True, **kwargs):
    model_path = determined_model_path if determined_model_path is not None else model_args.model_name_or_path
    if raw_datasets is not None:
        label_list, num_labels, _ = get_label(data_args, raw_datasets)
    else:
        num_labels = 2 # SQuAD

    # Build up config and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_path,
        num_labels=num_labels,
        finetuning_task=t_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=kwargs.get('token', None),
    )
    print("Config: ", config)
    pruned_heads =  getattr(config, 'pruned_heads', {})
    if pruned_heads:
        config.adap_pruned_heads = pruned_heads
        config.pruned_heads = {}
    if not hasattr(config, 'adap_pruned_heads'):
        config.adap_pruned_heads = {}
    elif 't5' in config.model_type:
        for t in list(config.adap_pruned_heads.keys()):
            v = config.adap_pruned_heads[t]
            for layer in list(v.keys()):
                v[int(layer)] = v[layer]
                del v[layer]
    if not getattr(config, 'apply_lora', False):
        config.apply_lora=model_args.apply_lora
    if not hasattr(config, 'lora_alpha'):
        config.lora_alpha=model_args.lora_alpha
    if not hasattr(config, 'lora_r'):
        config.lora_r=model_args.lora_r if determined_model_path is None else 8
    if not getattr(config, 'do_distill', False):
        config.do_distill = training_args.do_distill
    if config.apply_lora:
        print("LoRA r using is %d" % config.lora_r)
    else:
        print("No LoRA is using. Fine-tuning the model!")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=kwargs.get('token', None),
        model_max_length=data_args.max_seq_length if hasattr(data_args, 'max_seq_length') else data_args.max_input_length if hasattr(data_args, 'max_input_length') else getattr(data_args, 'model_max_length', 128),
        # local_files_only=True,
    )
    model_type = config.model_type
    
    if model_type == 'mt5':
        Model = AdaPMT5ForConditionalGeneration
    elif model_type == 't5':
        Model = AdaPT5ForConditionalGeneration
    elif model_type == 'llama':
        Model = ElasticLlamaForCausalLM
        # set padding side to left for batch generation
        tokenizer.padding_side = "left" # TODO: WARNING: this brings up lots of confusions, but we are only doing left-padding for inference, while right-padding for training in DataCollatorForSFT
    elif model_type == 'roberta':
        if 'squad' in data_args.task_name:
            Model = NewRobertaForQuestionAnswering
        else:
            Model = CoFiRobertaForSequenceClassification
    elif model_type == 'bert':
        if 'squad' in data_args.task_name:
            Model = AdaPBertForQuestionAnswering
        else:
            Model = CoFiBertForSequenceClassification
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented yet!")

    
    if os.path.exists(model_path):
        original_config_path = json.load(open(os.path.join(model_path, 'config.json'), 'r'))['_name_or_path']
        model_config = config
        print("Pruned heads:", config.pruned_heads, flush=True)
        model = Model.from_pretrained(
            original_config_path,
            from_tf=bool(".ckpt" in model_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=kwargs.get('token', None),
            low_cpu_mem_usage='llama' in config.model_type,
            torch_dtype = torch.bfloat16 if 'llama' in config.model_type else 'auto',
        )
        if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
            bin_fns = []
            p = os.path.join(model_path, "pytorch_model.bin")
            loaded_weights = torch.load(p, map_location="cpu")
            loaded_shapes = {k: v.shape for k, v in loaded_weights.items()}
        else:
            bin_fns = [fn for fn in os.listdir(model_path) if 'pytorch_model' in fn and fn.endswith(".bin")]
            loaded_weights = None
            loaded_shapes = {}
            print("Loading weight shapes from %s" % model_path)
            for fn in tqdm(bin_fns):
                loaded_shapes.update({k: v.shape for k, v in torch.load(os.path.join(model_path, fn), map_location="cpu").items()})
                gc.collect()
        post_processed = False
        # wrong_named_keys = [k for k in loaded_weights if 'layers' in k]
        # for k in wrong_named_keys:
        #     loaded_weights[k.replace('layers', 'layer')] = loaded_weights.pop(k)
        if not check_model_weights_shape_matching(model, loaded_weights):
            zs = None
            if os.path.exists(os.path.join(model_path, "final_head_mask.pt")) and os.path.exists(os.path.join(model_path, "final_intermediate_mask.pt")):
                intermediate_mask = torch.load(os.path.join(model_path, "final_intermediate_mask.pt"), map_location="cpu")
                head_mask = torch.load(os.path.join(model_path, "final_head_mask.pt"), map_location="cpu")
                hidden_mask = torch.load(os.path.join(model_path, "final_hidden_mask.pt"), map_location="cpu") if os.path.exists(os.path.join(model_path, "final_hidden_mask.pt")) else None
                zs = {
                    'intermediate_z': intermediate_mask,
                    'head_z': head_mask,
                    'hidden_z': hidden_mask,
                }
            elif os.path.exists(os.path.join(model_path, "..", "final_head_mask.pt")) and os.path.exists(os.path.join(model_path, "..", "final_intermediate_mask.pt")):
                intermediate_mask = torch.load(os.path.join(model_path, "..", "final_intermediate_mask.pt"), map_location="cpu")
                head_mask = torch.load(os.path.join(model_path, "..", "final_head_mask.pt"), map_location="cpu")
                hidden_mask = torch.load(os.path.join(model_path, "..", "final_hidden_mask.pt"), map_location="cpu") if os.path.exists(os.path.join(model_path, "..", "final_hidden_mask.pt")) else None
                zs = {
                    'intermediate_z': intermediate_mask,
                    'head_z': head_mask,
                    'hidden_z': hidden_mask,
                }
            elif os.path.exists(os.path.join(model_path, "head_mask.pt")) and os.path.exists(os.path.join(model_path, "intermediate_mask.pt")):
                intermediate_mask = torch.load(os.path.join(model_path, "intermediate_mask.pt"), map_location="cpu")
                head_mask = torch.load(os.path.join(model_path, "head_mask.pt"), map_location="cpu")
                hidden_mask = torch.load(os.path.join(model_path, "hidden_mask.pt"), map_location="cpu") if os.path.exists(os.path.join(model_path, "hidden_mask.pt")) else None
                zs = {
                    'intermediate_z': intermediate_mask,
                    'head_z': head_mask,
                    'hidden_z': hidden_mask,
                }
            if force_model_shape_deduction:
                zs = None
            if zs is None:
                # Deducting masks from the difference between model and weights
                print("Deducting masks from the difference between model and weights")
                if 't5' in model_type:
                    # T5 model deduction
                    adap_pruned_heads = config.adap_pruned_heads
                    head_size = model_config.d_model // model_config.num_heads
                    head_mask = torch.ones([3, model_config.num_layers, model_config.num_heads])
                    if adap_pruned_heads:
                        encoder_pruned_heads = adap_pruned_heads['encoder']
                        decoder_pruned_heads = adap_pruned_heads['decoder']
                        cross_pruned_heads = adap_pruned_heads['cross']
                        for i in range(model_config.num_layers):
                            head_mask[0][i][encoder_pruned_heads[i]] = 0
                            head_mask[1][i][decoder_pruned_heads[i]] = 0
                            head_mask[2][i][cross_pruned_heads[i]] = 0
                    intermediate_mask = torch.ones([2, model_config.num_layers, model_config.d_ff])
                    encoder_intermediates_left = [
                        loaded_weights['encoder.block.%d.layer.1.DenseReluDense.wo.weight' % i].shape[1] if 'encoder.block.%d.layer.1.DenseReluDense.wo.weight' % i in loaded_weights else 0
                        for i in range(model_config.num_layers)
                    ]
                    decoder_intermediates_left = [
                        loaded_weights['decoder.block.%d.layer.2.DenseReluDense.wo.weight' % i].shape[1] if 'decoder.block.%d.layer.2.DenseReluDense.wo.weight' % i in loaded_weights else 0
                        for i in range(model_config.num_layers)
                    ]
                    for i in range(model_config.num_layers):
                        intermediate_mask[0][i][:model_config.d_ff - encoder_intermediates_left[i]] = 0
                        intermediate_mask[1][i][:model_config.d_ff - decoder_intermediates_left[i]] = 0
                    head_mask = head_mask.view(-1)
                    intermediate_mask = intermediate_mask.view(-1)
                    hidden_size = loaded_weights['shared.weight'].shape[1]
                    if hidden_size != model_config.d_model:
                        hidden_mask = torch.zeros(model_config.d_model)
                        hidden_mask[:hidden_size] = 1
                    else:
                        hidden_mask = None
                elif 'bert' in model_type:
                    # Bert-like model deduction
                    intermediate_mask, head_mask = torch.ones([model_config.num_hidden_layers, model_config.intermediate_size]), torch.ones([model_config.num_hidden_layers, model_config.num_attention_heads])
                    head_size = model_config.hidden_size // model_config.num_attention_heads
                    heads_left = [
                        loaded_weights['%s.encoder.layer.%s.attention.self.value.weight' % (model_type, i)].shape[0] // head_size if '%s.encoder.layer.%s.attention.self.value.weight' % (model_type, i) in loaded_weights else 0
                        for i in range(model_config.num_hidden_layers)
                    ]
                    intermediates_left = [
                    loaded_weights['%s.encoder.layer.%s.intermediate.dense.weight' % (model_type, i)].shape[0] if '%s.encoder.layer.%s.intermediate.dense.weight' % (model_type, i) in loaded_weights else 0
                        for i in range(model_config.num_hidden_layers)
                    ]
                    if getattr(model_config, 'adap_pruned_heads', None) is not None:
                        for i in model_config.adap_pruned_heads:
                            for j in model_config.adap_pruned_heads[i]:
                                head_mask[i][j] = 0
                    else:
                        for i in range(model_config.num_hidden_layers):
                            head_mask[i][:model_config.num_attention_heads - heads_left[i]] = 0
                    for i in range(model_config.num_hidden_layers):
                        intermediate_mask[i][:model_config.intermediate_size - intermediates_left[i]] = 0
                    hidden_size = loaded_weights['%s.embeddings.word_embeddings.weight' % model_type].shape[1]
                    if hidden_size != model_config.hidden_size:
                        hidden_mask = torch.zeros(model_config.hidden_size)
                        hidden_mask[:hidden_size] = 1
                    else:
                        hidden_mask = None
                elif model_type == 'llama':
                    # LLaMA 1/2 model deduction
                    intermediate_mask, head_mask = torch.ones([model_config.num_hidden_layers, model_config.intermediate_size]), torch.ones([model_config.num_hidden_layers, model_config.num_attention_heads])
                    head_size = model_config.hidden_size // model_config.num_attention_heads
                    heads_left = [
                        loaded_shapes['model.layers.%d.self_attn.v_proj.weight' % i][0] // head_size if 'model.layers.%d.self_attn.v_proj.weight' % i in loaded_shapes else 0
                        for i in range(model_config.num_hidden_layers)
                    ]
                    intermediates_left = [
                    loaded_shapes['model.layers.%d.mlp.up_proj.weight' % i][0] if 'model.layers.%d.mlp.up_proj.weight' % i in loaded_shapes else 0
                        for i in range(model_config.num_hidden_layers)
                    ]
                    if getattr(model_config, 'adap_pruned_heads', None) is not None and len(model_config.adap_pruned_heads) > 0:
                        for i in model_config.adap_pruned_heads:
                            for j in model_config.adap_pruned_heads[i]:
                                head_mask[i][j] = 0
                    else:
                        for i in range(model_config.num_hidden_layers):
                            head_mask[i][:model_config.num_attention_heads - heads_left[i]] = 0
                    for i in range(model_config.num_hidden_layers):
                        intermediate_mask[i][:model_config.intermediate_size - intermediates_left[i]] = 0
                    hidden_size = loaded_shapes['model.embed_tokens.weight'][1]
                    if hidden_size != model_config.hidden_size:
                        hidden_mask = torch.zeros(model_config.hidden_size)
                        hidden_mask[:hidden_size] = 1
                    else:
                        hidden_mask = None

            model.head_mask, model.intermediate_mask = head_mask, intermediate_mask
            model.hidden_mask = hidden_mask
            model.prune_model_with_masks(continual_pruning=False)
            if model.config.model_type == 'llama' and loaded_shapes['lm_head.weight'][0] != model.lm_head.weight.shape[0]:
                model_post_processing(model, model_args, data_args, training_args, tokenizer, **kwargs)
                post_processed = True
            convert_layers_based_on_ckpt(model, loaded_shapes)
            if loaded_weights is None:
                print("Real loading weights from %s" % model_path)
                for fn in tqdm(bin_fns):
                    model.load_state_dict(torch.load(os.path.join(model_path, fn), map_location="cpu"), strict=False)
                    gc.collect()
            else:
                model.load_state_dict(loaded_weights, strict=False)
        else:
            convert_layers_based_on_ckpt(model, {k: v.shape for k, v in loaded_weights.items()})
            model.load_state_dict(loaded_weights, strict=False)
            if os.path.exists(os.path.join(model_path, "final_head_mask.pt")) and os.path.exists(os.path.join(model_path, "final_intermediate_mask.pt")):
                if model_args.do_auto_pruning:
                    model.intermediate_mask = torch.load(os.path.join(model_path, "final_intermediate_mask.pt"), map_location="cpu")
                    model.head_mask = torch.load(os.path.join(model_path, "final_head_mask.pt"), map_location="cpu")
                    model.prune_model_with_masks(continual_pruning=False)
                    if (model.head_mask.int() == model.head_mask).all():
                        model.head_mask = None
                    else:
                        model.head_mask = [model.head_mask[i].index_select(dim=0, index=model.head_mask[i].nonzero().squeeze()) for i in range(model.head_mask.shape[0])]
                    if (model.intermediate_mask.int() == model.intermediate_mask).all():
                        model.intermediate_mask = None
                    else:
                        model.intermediate_mask = [model.intermediate_mask[i].index_select(dim=0, index=model.intermediate_mask[i].nonzero().squeeze()) for i in range(model.intermediate_mask.shape[0])]
                elif os.path.exists(os.path.join(model_path, 'head_mask.pt')):
                    model.head_mask = torch.load(os.path.join(model_path, 'head_mask.pt'), map_location="cpu")
                    model.intermediate_mask = torch.load(os.path.join(model_path, 'intermediate_mask.pt'), map_location="cpu")
                else:
                    model.head_mask = None
                    model.intermediate_mask = None
        if not post_processed:
            model_post_processing(model, model_args, data_args, training_args, tokenizer, **kwargs)
        if retain_scaling:
            scaling = config.lora_alpha / config.lora_r
            for m in model.modules():
                if isinstance(m, LoRALayer):
                    m.scaling = scaling
                    if isinstance(m, DistillLinear):
                        m.teacher_scaling = scaling
        print(f"Load weights from {model_path}")
    else:
        model = Model.from_pretrained(
            model_path,
            from_tf=bool(".ckpt" in model_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=kwargs.get('token', None),
            low_cpu_mem_usage='llama' in config.model_type,
            torch_dtype = torch.bfloat16 if 'llama' in config.model_type else 'auto',
        ) #! inside the function, we get the original struct  #! CofiBertForSequenceClassification
        model_post_processing(model, model_args, data_args, training_args, tokenizer, **kwargs)
    if getattr(model, 'layer_transformation', None) is not None and model.layer_transformation.weight.device == torch.device('meta'):
        model.layer_transformation.weight = torch.nn.Parameter(torch.eye(model.layer_transformation.in_features, device=model.device).to(device=model.device, dtype=model.dtype))
        model.layer_transformation.reset_parameters()
    return config, tokenizer, model