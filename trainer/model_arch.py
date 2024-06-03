import torch
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig

NAME2TEMPLATE = {
    'query': '.encoder.layer.%d.attention.self.query',
    'key': '.encoder.layer.%d.attention.self.key',
    'value': '.encoder.layer.%d.attention.self.value',
    'attention.output': '.encoder.layer.%d.attention.output.dense',
    'intermediate': '.encoder.layer.%d.intermediate.dense',
    'intermediate.output': '.encoder.layer.%d.output.dense',
}

T5NAME2TEMPLATE = {
    'enc_self_query': 'encoder.block.%d.layer.0.SelfAttention.q',
    'enc_self_key': 'encoder.block.%d.layer.0.SelfAttention.k',
    'enc_self_value': 'encoder.block.%d.layer.0.SelfAttention.v',
    'enc_self_output': 'encoder.block.%d.layer.0.SelfAttention.o',
    'dec_self_query': 'decoder.block.%d.layer.0.SelfAttention.q',
    'dec_self_key': 'decoder.block.%d.layer.0.SelfAttention.k',
    'dec_self_value': 'decoder.block.%d.layer.0.SelfAttention.v',
    'dec_self_output': 'decoder.block.%d.layer.0.SelfAttention.o',
    'cross_query': 'decoder.block.%d.layer.1.EncDecAttention.q',
    'cross_key': 'decoder.block.%d.layer.1.EncDecAttention.k',
    'cross_value': 'decoder.block.%d.layer.1.EncDecAttention.v',
    'cross_output': 'decoder.block.%d.layer.1.EncDecAttention.o',
    'encoder_i': 'encoder.block.%d.layer.1.DenseReluDense.wi',
    'encoder_i0': 'encoder.block.%d.layer.1.DenseReluDense.wi_0',
    'encoder_i1': 'encoder.block.%d.layer.1.DenseReluDense.wi_1',
    'encoder_io': 'encoder.block.%d.layer.1.DenseReluDense.wo',
    'decoder_i': 'decoder.block.%d.layer.2.DenseReluDense.wi',
    'decoder_i0': 'decoder.block.%d.layer.2.DenseReluDense.wi_0',
    'decoder_i1': 'decoder.block.%d.layer.2.DenseReluDense.wi_1',
    'decoder_io': 'decoder.block.%d.layer.2.DenseReluDense.wo',
}

LLAMA_NAME2TEMPLATE = {
    'dec_self_query': 'model.layers.%d.self_attn.q_proj',
    'dec_self_key': 'model.layers.%d.self_attn.k_proj',
    'dec_self_value': 'model.layers.%d.self_attn.v_proj',
    'dec_self_output': 'model.layers.%d.self_attn.o_proj',
    'decoder_i0': 'model.layers.%d.mlp.gate_proj',
    'decoder_i1': 'model.layers.%d.mlp.up_proj',
    'decoder_io': 'model.layers.%d.mlp.down_proj',
}

NAME2CATEGORY = {
    'query': 'attn',
    'key': 'attn',
    'value': 'attn',
    'attention.output': 'attn_o',
    'intermediate': 'ffn',
    'intermediate.output': 'ffn_o',
    'classifier': 'classifier',
    'classifier.out': 'classifier',
    'enc_self_query': 'attn',
    'enc_self_key': 'attn',
    'enc_self_value': 'attn',
    'enc_self_output': 'attn',
    'dec_self_query': 'dec_attn',
    'dec_self_key': 'dec_attn',
    'dec_self_value': 'dec_attn',
    'dec_self_output': 'dec_attn',
    'cross_query': 'cross_attn',
    'cross_key': 'cross_attn',
    'cross_value': 'cross_attn',
    'cross_output': 'cross_attn',
    'encoder_i': 'ffn',
    'encoder_i0': 'ffn',
    'encoder_i1': 'ffn',
    'encoder_io': 'ffn',
    'decoder_i': 'dec_ffn',
    'decoder_i0': 'dec_ffn',
    'decoder_i1': 'dec_ffn',
    'decoder_io': 'dec_ffn',
}

NAME2ATTR = {
    'query': {
        'bert': 'query',
        't5': 'q',
    },
    'key': {
        'bert': 'key',
        't5': 'k',
    },
    'value': {
        'bert': 'value',
        't5': 'v',
    },
    'attention.output': {
        'bert': 'dense',
        't5': 'o',
    },
    'intermediate': {
        'bert': 'dense',
        't5': 'wi_0',
    },
    'intermediate.output': {
        'bert': 'dense',
        't5': 'wo',
    },
    'enc_self_query': {'t5': 'q'},
    'enc_self_key': {'t5': 'k'},
    'enc_self_value': {'t5': 'v'},
    'enc_self_output': {'t5': 'o'},
    'dec_self_query': {
        't5': 'q',
        'llama': 'q_proj',
    },
    'dec_self_key': {
        't5': 'k',
        'llama': 'k_proj',
    },
    'dec_self_value': {
        't5': 'v',
        'llama': 'v_proj',
    },
    'dec_self_output': {
        't5': 'o',
        'llama': 'o_proj',
    },
    'cross_query': {'t5': 'q'},
    'cross_key': {'t5': 'k'},
    'cross_value': {'t5': 'v'},
    'cross_output': {'t5': 'o'},
    'encoder_i': {'t5': 'wi'},
    'encoder_i0': {'t5': 'wi_0'},
    'encoder_i1': {'t5': 'wi_1'},
    'encoder_io': {'t5': 'wo'},
    'decoder_i': {'t5': 'wi'},
    'decoder_i0': {
        't5': 'wi_0',
        'llama': 'gate_proj',
    },
    'decoder_i1': {
        't5': 'wi_1',
        'llama': 'up_proj',
    },
    'decoder_io': {
        't5': 'wo',
        'llama': 'down_proj',
    },
}


AUTOENCODER = 0
AUTOREGRESSIVE = 1
ENCODER_DECODER = 2

MODEL2ARCH = {
    'bert': AUTOENCODER,
    'roberta': AUTOENCODER,
    't5': ENCODER_DECODER,
    'mt5': ENCODER_DECODER,
    'bart': ENCODER_DECODER,
    'gpt2': AUTOREGRESSIVE,
    'llama': AUTOREGRESSIVE,
}

def get_backbone(model):
    model_type = model.base_model_prefix
    backbone = getattr(model, model_type)
    return backbone


def get_encoder(model):
    if hasattr(model, 'encoder'):
        backbone = model
    else:
        backbone = get_backbone(model)
    if hasattr(backbone, 'encoder'):
        return backbone.encoder
    else:
        return None

def get_decoder(model):
    if hasattr(model, 'decoder'):
        backbone = model
    else:
        backbone = get_backbone(model)
    if hasattr(backbone, 'decoder'):
        return backbone.decoder
    elif model.config.model_type == 'llama':
        return backbone
    else:
        return None


def get_layers(model, use_decoder=False):
    if use_decoder:
        module = get_decoder(model)
    else:
        module = get_encoder(model)
    if hasattr(module, 'layer'):
        layers = module.layer
    elif hasattr(module, 'block'):
        layers = module.block
    elif hasattr(module, 'layers'):
        layers = module.layers
    else:
        raise ValueError('Unknown module layer type')
    return layers

def get_mha_qkv(model, index, use_decoder=False, use_cross_attention=False):
    layer = get_layers(model, use_decoder=use_decoder)[index]
    if hasattr(layer, 'attention') and hasattr(layer.attention, 'self'):
        mha_qkv = layer.attention.self
    elif hasattr(layer, 'layer'):
        if use_cross_attention:
            mha_qkv = layer.layer[1].EncDecAttention
        else:
            mha_qkv = layer.layer[0].SelfAttention
    elif hasattr(layer, 'self_attn'):
        mha_qkv = layer.self_attn
    else:
        raise ValueError('Unknown attention type')
    return mha_qkv

def get_mha_proj(model, index, use_decoder=False, use_cross_attention=False, return_parent=False):
    layer = get_layers(model, use_decoder=use_decoder)[index]
    if hasattr(layer, 'attention') and hasattr(layer.attention, 'output'):
        if return_parent:
            mha_proj = layer.attention.output
        else:
            mha_proj = layer.attention.output.dense
    elif hasattr(layer, 'layer'):
        if use_cross_attention:
            if return_parent:
                mha_proj = layer.layer[1].EncDecAttention
            else:
                mha_proj = layer.layer[1].EncDecAttention.o
        else:
            if return_parent:
                mha_proj = layer.layer[0].SelfAttention
            else:
                mha_proj = layer.layer[0].SelfAttention.o
    elif hasattr(layer, 'self_attn'):
        if return_parent:
            mha_proj = layer.self_attn
        else:
            mha_proj = layer.self_attn.o_proj
    else:
        raise ValueError('Unknown attention type')
    return mha_proj


def get_ffn1(model, index, use_decoder=False):
    layer = get_layers(model, use_decoder=use_decoder)[index]
    if hasattr(layer, 'intermediate'):
        ffn1 = layer.intermediate
    elif hasattr(layer, 'layer'):
        ffn1 = layer.layer[-1].DenseReluDense
    elif hasattr(layer, 'mlp'):
        ffn1 = layer.mlp
    else:
        raise ValueError('Unknown ffn1 type')
    return ffn1


def get_ffn2(model, index, use_decoder=False):
    layer = get_layers(model, use_decoder=use_decoder)[index]
    if hasattr(layer, 'output'):
        ffn2 = layer.output
    elif hasattr(layer, 'layer') and hasattr(layer.layer[-1], 'DenseReluDense'):
        ffn2 = layer.layer[-1].DenseReluDense.wo
    elif hasattr(layer, 'mlp'):
        ffn2 = layer.mlp.down_proj
    else:
        raise ValueError('Unknown ffn2 type')
    return ffn2


def get_classifier(model):
    backbone = get_backbone(model)
    if backbone.pooler is not None:
        classifier = model.classifier
    else:
        classifier = model.classifier.out_proj
    return classifier

def hijack_input(module, list_to_append, input_index=None, device='cuda'):
    if input_index is None:
        hook = lambda _, inputs: list_to_append.append([v.to(device) if isinstance(v, torch.Tensor) else v for v in inputs])
    else:
        hook = lambda _, inputs: list_to_append.append(inputs[input_index].to(device) if isinstance(inputs[input_index], torch.Tensor) else inputs[input_index])
    handle = module.register_forward_pre_hook(hook)
    return handle

def hijack_output(module, list_to_append, output_index=None, device='cuda'):
    if output_index is None:
        hook = lambda _, __, outputs: list_to_append.append([v.to(device) if isinstance(v, torch.Tensor) else v for v in outputs])
    else:
        hook = lambda _, __, outputs: list_to_append.append(outputs[output_index].to(device) if isinstance(outputs[output_index], torch.Tensor) else outputs[output_index])
    handle = module.register_forward_hook(hook)
    return handle


class ModelArch(object):
    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.config: PretrainedConfig = model.config
        self.model_type = self.config.model_type
        self.model_category = 'bert' if 'bert' in model.config.model_type else 't5' if 't5' in model.config.model_type else model.config.model_type
        self.arch_type = MODEL2ARCH[self.config.model_type]
        if 'bert' in self.model_type:
           self.name_template = NAME2TEMPLATE
        elif 't5' in self.model_type:
            self.name_template = T5NAME2TEMPLATE
        elif 'llama' in self.model_type:
            self.name_template = LLAMA_NAME2TEMPLATE
        else:
            raise NotImplementedError
        self.name2category = NAME2CATEGORY
        
    def get_name2template(self):
        if 'bert' in self.model_type:
            name2template = {
                k: self.model.base_model_prefix + v if v.startswith('.') else v
                for k, v in NAME2TEMPLATE.items()
            }
            return name2template
        elif 't5' in self.model_type:
            return T5NAME2TEMPLATE
        elif 'llama' in self.model_type:
            return LLAMA_NAME2TEMPLATE
        else:
            raise NotImplementedError
        
    def get_parent_layer(self, i, k):
        if NAME2CATEGORY[k] == 'attn':
            parent_layer = get_mha_qkv(self.model, i, use_decoder=False)
        elif NAME2CATEGORY[k] == 'attn_o':
            parent_layer = get_mha_proj(self.model, i, return_parent=True)
        elif NAME2CATEGORY[k] == 'ffn':
            parent_layer = get_ffn1(self.model, i, use_decoder=False)
        elif NAME2CATEGORY[k] == 'ffn_o':
            parent_layer = get_ffn2(self.model, i)
        elif NAME2CATEGORY[k] == 'dec_attn':
            parent_layer = get_mha_qkv(self.model, i, use_decoder=True, use_cross_attention=False)
        elif NAME2CATEGORY[k] == 'cross_attn':
            parent_layer = get_mha_qkv(self.model, i, use_decoder=True, use_cross_attention=True)
        elif NAME2CATEGORY[k] == 'dec_ffn':
            parent_layer = get_ffn1(self.model, i, use_decoder=True)
        else:
            raise ValueError("Unknown category %s" % NAME2CATEGORY[k])
        return parent_layer      
    
    def get_layer(self, i, k):
        # For output layers, we directly return the Linear layer
        if 'io' in k or 'output' in k:
            if k == 'attention.output':
                return get_mha_proj(self.model, i)
            if k == 'intermediate.output':
                return get_ffn2(self.model, i).dense
            if k == 'enc_self_output':
                return get_mha_proj(self.model, i)
            if k == 'dec_self_output':
                return get_mha_proj(self.model, i, use_decoder=True)
            if k == 'cross_output':
                return get_mha_proj(self.model, i, use_decoder=True, use_cross_attention=True)
            if k == 'encoder_i':
                return get_ffn1(self.model, i)
            if k == 'decoder_i':
                return get_ffn1(self.model, i, use_decoder=True)
            if k == 'encoder_io':
                return get_ffn2(self.model, i)
            if k == 'decoder_io':
                return get_ffn2(self.model, i, use_decoder=True)
        k_attr = NAME2ATTR[k][self.model_category]
        parent_layer = self.get_parent_layer(i, k)
        layer = getattr(parent_layer, k_attr)
        return layer