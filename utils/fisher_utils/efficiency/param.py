def param_per_head(
    hidden_size,
    attention_head_size,
    *args,
    **kwargs,
):
    
    weight_params = 4 * hidden_size * attention_head_size # qkvo
    bias_params = 3 * attention_head_size # qkv
    return weight_params + bias_params

def param_per_hidden_dim(
    head_sizes,
    intermediate_sizes,
    num_hidden_layers = None,
    ffn_gated: bool = False,
    *args,
    **kwargs,
):
    if num_hidden_layers is None:
        total_params_per_hidden = 0
        for head_size, intermediate_size in zip(head_sizes, intermediate_sizes):
            params_per_head_hidden = 4 * head_size + 1 # qkvo
            params_per_intermediate_hidden = (3 if ffn_gated else 2) * intermediate_size  + 1 # up & down
            params_per_layernorm = 4 # gamma & beta for MHA & FFN
            total_params_per_hidden += params_per_head_hidden + params_per_intermediate_hidden + params_per_layernorm
        return total_params_per_hidden
    else:
        # head_sizes and intermediate_sizes are the total number of heads and neurons
        return 4 * head_sizes + (3 if ffn_gated else 2) * intermediate_sizes + 6 * num_hidden_layers

def param_per_neuron(
    hidden_size,
    gated: bool = False,
    *args,
    **kwargs,
):
    if gated:
        return 3 * hidden_size + 1 # up & down + gate
    else:
        return 2 * hidden_size + 1 # up & down

def param_per_t5_head(
    hidden_size,
    attention_head_size,
):
    
    weight_params = 4 * hidden_size * attention_head_size # qkvo
    return weight_params

def param_per_t5_hidden_dim(
    head_sizes,
    intermediate_sizes,
    ffn_gated: bool = False,
    num_hidden_layers = None,
):
    if num_hidden_layers is None:
        total_params_per_hidden = 2 # encoder final layer norm & decoder final layer norm
        for head_size, intermediate_size in zip(head_sizes, intermediate_sizes):
            params_per_head_hidden = 4 * head_size# qkvo
            params_per_intermediate_hidden = (3 if ffn_gated else 2) * intermediate_size # up & down, or plus gate
            params_per_layernorm = 2.5 # gamma for MHA & FFN
            total_params_per_hidden += params_per_head_hidden + params_per_intermediate_hidden + params_per_layernorm
        return total_params_per_hidden
    else:
        # head_sizes and intermediate_sizes are the total number of heads and neurons
        return 4 * head_sizes + (3 if ffn_gated else 2) * intermediate_sizes + 2.5 * num_hidden_layers + 2

def param_per_t5_neuron(
    hidden_size,
    gated: bool = False,
):
    if gated:
        return 3 * hidden_size # up & down + gate
    else:
        return 2 * hidden_size# up & down

def compute_param(
    num_heads_per_layer,
    num_neurons_per_layer,
    hidden_size,
    attention_head_size,
    num_hidden_layers = None,
    is_t5 = False,
    ffn_gated = False,
):
    if num_hidden_layers is None:
        param = 0
        for num_heads, num_neurons in zip(num_heads_per_layer, num_neurons_per_layer):
            if is_t5:
                attention_param = num_heads * param_per_t5_head(hidden_size, attention_head_size)
                ffn_param = num_neurons * param_per_t5_neuron(hidden_size, ffn_gated)
                layer_norm_param = 2 * hidden_size # gamma & beta for MHA & FFN
                param += attention_param + ffn_param + layer_norm_param + output_bias_param
            else:
                attention_param = num_heads * param_per_head(hidden_size, attention_head_size)
                ffn_param = num_neurons * param_per_neuron(hidden_size, ffn_gated)
                layer_norm_param = 4 * hidden_size # gamma & beta for MHA & FFN
                output_bias_param = 2 * hidden_size # MHA & FFN
                param += attention_param + ffn_param + layer_norm_param + output_bias_param
        return param
    elif is_t5:
        # num_heads_per_layer and num_neurons_per_layer are the total number of heads and neurons
        return hidden_size * (4 * num_heads_per_layer * attention_head_size + (3 if ffn_gated else 2) * num_neurons_per_layer + 4 * num_hidden_layers) # no bias and possibly ffn gated, with encoder and decoder layer norms
    else:
        # num_heads_per_layer and num_neurons_per_layer are the total number of heads and neurons
        return hidden_size * (4 * num_heads_per_layer * attention_head_size + (3 if ffn_gated else 2) * num_neurons_per_layer + 6 * num_hidden_layers) +  num_heads_per_layer * attention_head_size * 3 + num_neurons_per_layer