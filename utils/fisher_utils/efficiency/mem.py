from typing import List

MB = 1024 * 1024

def bert_forward(batch_size: int = 32, seq_len: int = 128, num_heads: List[int] = [12] * 12, num_neurons: List[int] = [3072] * 12, hidden_size: int = 768, intermediate_size: int = 3072, attn_head_size: int = 64, output_hidden_states: bool = True, output_attention: bool = False, dtype=32)-> float:
    assert len(num_heads) == len(num_neurons)
    mha_size = sum(num_heads) * ((hidden_size * attn_head_size) + 1) * 4
    ffn_size = sum(num_neurons) * hidden_size * 2 + sum(num_neurons)
    total = mha_size + ffn_size
    return total * dtype / 8 / MB