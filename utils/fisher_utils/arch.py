import torch

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
    else:
        raise ValueError('Unknown attention type')
    return mha_proj


def get_ffn1(model, index, use_decoder=False):
    layer = get_layers(model, use_decoder=use_decoder)[index]
    if hasattr(layer, 'intermediate'):
        ffn1 = layer.intermediate
    elif hasattr(layer, 'layer'):
        ffn1 = layer.layer[-1].DenseReluDense
    else:
        raise ValueError('Unknown ffn1 type')
    return ffn1


def get_ffn2(model, index, use_decoder=False):
    layer = get_layers(model, use_decoder=use_decoder)[index]
    if hasattr(layer, 'output'):
        ffn2 = layer.output
    elif hasattr(layer, 'layer') and hasattr(layer.layer[-1], 'DenseReluDense'):
        ffn2 = layer.layer[-1].DenseReluDense.wo
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


def register_mask(module, mask):
    hook = lambda _, inputs: (inputs[0] * mask, inputs[1])
    handle = module.register_forward_pre_hook(hook)
    return handle


def apply_neuron_mask(model, neuron_mask):
    num_hidden_layers = neuron_mask.shape[0]
    handles = []
    for layer_idx in range(num_hidden_layers):
        ffn2 = get_ffn2(model, layer_idx)
        handle = register_mask(ffn2, neuron_mask[layer_idx])
        handles.append(handle)
    return handles


class MaskNeurons:
    def __init__(self, model, neuron_mask):
        self.handles = apply_neuron_mask(model, neuron_mask)

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        for handle in self.handles:
            handle.remove()


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