import sys
import os
import torch
from tqdm import tqdm

if __name__ == '__main__':
    root = sys.argv[1]
    weights = [os.path.join(root, v) for v in os.listdir(root) if v.endswith('.bin') and 'arg' not in v]
    total_param_nums = 0
    param_nums = 0
    for weight in tqdm(weights):
        state_dict = torch.load(weight, map_location='cpu')
        for k, v in state_dict.items():
            if 'lora' in k or 'transform' in k:
                continue
            total_param_nums += v.numel()
            if 'lm_head' in k or 'embed' in k or 'shared' in k or 'classifier' in k or 'pooler' in k or 'qa_output' in k:
                print("Excluding %s with number of params %d" % (k, v.numel()))
                continue
            param_nums += v.numel()
            
    print("Total param nums: {}".format(total_param_nums))
    print("Encoder/decoder param nums: {}".format(param_nums))