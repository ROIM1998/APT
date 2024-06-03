

def mac_per_head(
    seq_len,
    hidden_size,
    attention_head_size,
):
    per_head_qkv = lambda seq_len: 3 * seq_len * hidden_size * attention_head_size
    per_head_attn = lambda seq_len: 2 * seq_len * seq_len * attention_head_size
    per_head_output = lambda seq_len: seq_len * attention_head_size * hidden_size
    mac = per_head_qkv(seq_len) + per_head_attn(seq_len) + per_head_output(seq_len)
    return mac

def mac_per_cross_head(
    input_seq_len,
    output_seq_len,
    hidden_size,
    attention_head_size,
):
    per_head_qkv = 2 * input_seq_len * hidden_size * attention_head_size + output_seq_len * hidden_size * attention_head_size
    per_head_attn = 2 * input_seq_len * output_seq_len * attention_head_size
    per_head_output = output_seq_len * attention_head_size * hidden_size
    mac = per_head_qkv + per_head_attn + per_head_output
    return mac

def mac_per_hidden_dim(
    seq_len,
    head_sizes,
    intermediate_sizes,
):
    total_mac_per_hidden = 0
    for head_size, intermediate_size in zip(head_sizes, intermediate_sizes):
        mac_per_head_hidden = 4 * seq_len * head_size
        mac_per_intermediate_hidden = 2 * seq_len * intermediate_size
        total_mac_per_hidden += mac_per_head_hidden + mac_per_intermediate_hidden
    return total_mac_per_hidden


def mac_per_neuron(seq_len, hidden_size, gated=False):
    if gated:
        return 3 * seq_len * hidden_size
    else:
        return 2 * seq_len * hidden_size


def compute_mac(
    num_heads_per_layer,
    num_neurons_per_layer,
    seq_len,
    hidden_size,
    attention_head_size,
    gated=False,
):
    mac = 0.0
    for num_heads, num_neurons in zip(num_heads_per_layer, num_neurons_per_layer):
        attention_mac = num_heads * mac_per_head(seq_len, hidden_size, attention_head_size)
        ffn_mac = num_neurons * mac_per_neuron(seq_len, hidden_size, gated=gated)
        mac += attention_mac + ffn_mac
    return mac

def compute_encoder_decoder_mac(
    num_head_per_layer,
    num_neurons_per_layer,
    input_seq_len,
    output_seq_len,
    hidden_size,
    attention_head_size,
    gated=True,
):
    mac = 0.0
    assert len(num_head_per_layer) == 3 and len(num_neurons_per_layer) == 2
    assert len(num_head_per_layer[0]) == len(num_neurons_per_layer[0])
    assert len(num_head_per_layer[1]) == len(num_neurons_per_layer[1]) and len(num_head_per_layer[2]) == len(num_neurons_per_layer[1])

    for num_self_heads, num_neurons in zip(num_head_per_layer[0], num_neurons_per_layer[0]):
        attention_mac = num_self_heads * mac_per_head(input_seq_len, hidden_size, attention_head_size)
        ffn_mac = num_neurons * mac_per_neuron(input_seq_len, hidden_size, gated=gated)
        mac += attention_mac + ffn_mac
    
    for num_self_heads, num_cross_heads, num_neurons in zip(num_head_per_layer[1], num_head_per_layer[2], num_neurons_per_layer[1]):
        self_attention_mac = num_self_heads * mac_per_head(output_seq_len, hidden_size, attention_head_size)
        cross_attention_mac = num_cross_heads * mac_per_cross_head(input_seq_len, output_seq_len, hidden_size, attention_head_size)
        ffn_mac = num_neurons * mac_per_neuron(output_seq_len, hidden_size, gated=gated)
        mac += self_attention_mac + cross_attention_mac + ffn_mac
    return mac


def compute_mask_mac(head_mask, neuron_mask, seq_len, hidden_size):
    num_hidden_layers = head_mask.shape[0]
    num_attention_heads = head_mask.shape[1]
    intermediate_size = neuron_mask.shape[1]
    attention_head_size = int(hidden_size / num_attention_heads)

    original_mac = compute_mac(
        [num_attention_heads] * num_hidden_layers,
        [intermediate_size] * num_hidden_layers,
        seq_len,
        hidden_size,
        attention_head_size,
    )
    pruned_mac = compute_mac(
        (head_mask != 0).sum(dim=1),
        (neuron_mask != 0).sum(dim=1),
        seq_len,
        hidden_size,
        attention_head_size,
    )
    return pruned_mac, original_mac
