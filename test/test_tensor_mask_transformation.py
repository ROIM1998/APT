import time
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class SelectLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, in_retained_indices=None, out_retained_indices=None):
        super().__init__(in_features, out_features, bias)
        self.in_retained_indices = in_retained_indices
        self.out_retained_indices = out_retained_indices
        
    def forward(self, input: Tensor) -> Tensor:
        if self.in_retained_indices is not None:
            input = torch.index_select(input, -1, self.in_retained_indices)
            selected_weight = self.weight.index_select(1, self.in_retained_indices)
        else:
            selected_weight = self.weight

        if self.out_retained_indices is not None:
            selected_weight = selected_weight.index_select(0, self.out_retained_indices)
            selected_bias = self.bias.index_select(0, self.out_retained_indices)
        else:
            selected_bias = self.bias
            
        output = F.linear(input, selected_weight, selected_bias)
        
        # Padding non-retained indices with zeros
        if self.out_retained_indices is not None:
            # output = torch.zeros(input.shape[:-1] + (self.out_features,), device=input.device).scatter_add(-1, self.out_retained_indices.unsqueeze(0).expand(output.shape[:-1] + (-1,)), output)
            padded_output = torch.zeros(input.shape[:-1] + (self.out_features,), device=input.device)
            padded_output[..., self.out_retained_indices] = output
            return padded_output
        return output
    
    def to(self, device):
        if self.in_retained_indices is not None:
            self.in_retained_indices = self.in_retained_indices.to(device)
        if self.out_retained_indices is not None:
            self.out_retained_indices = self.out_retained_indices.to(device)
        return super().to(device)


def time_test():
    mask_in = torch.ones(768, device='cuda')
    # Mask 20% of the input
    mask_out = torch.ones(3072, device='cuda')
    mask_out[torch.randperm(3072)[:614]] = 0
    in_retained_indices = mask_in.nonzero().squeeze()
    out_retained_indices = mask_out.nonzero().squeeze()
    
    linear = nn.Linear(768, 3072).cuda()
    select_linear = SelectLinear(768, 3072, in_retained_indices=in_retained_indices, out_retained_indices=out_retained_indices).to('cuda')
    select_linear.weight.data.copy_(linear.weight.data)
    select_linear.bias.data.copy_(linear.bias.data)
    x = torch.randn(32, 128, 768, device='cuda')
    
    for i in range(100): # warm up
        loss = linear(x).sum()
        loss.backward()
        torch.cuda.synchronize()
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(100):
        select_linear.zero_grad()
        y_a = select_linear(x)
        y_a.sum().backward()
    torch.cuda.synchronize()
    
    print(f'SelectLinear {i} iterations take {time.time() - start_time} seconds')
    print(f'Max memory allocated {torch.cuda.max_memory_allocated() / 1024 / 1024} MB')
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(100):
        linear.zero_grad()
        y_b = linear(x * mask_in)  * mask_out
        y_b.sum().backward()
    torch.cuda.synchronize()
    print(f'Linear {i} iterations take {time.time() - start_time} seconds')
    print(f'Max memory allocated {torch.cuda.max_memory_allocated() / 1024 / 1024} MB')
    
    print(torch.allclose(y_a, y_b))
    print(torch.allclose(select_linear.weight.grad, linear.weight.grad))

def main():
    time_test()
    

if __name__ == '__main__':
    main()