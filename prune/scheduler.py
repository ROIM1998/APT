import torch
from utils.minus_utils import sequential_head_mask_generation, sequential_neuron_mask_generation
from .search import search_mac_reverse
from .fisher import collect_mask_grads, compute_fisher_info

class BasePruningScheduler:
    def __init__(self, model, head_mask: torch.Tensor, intermediate_mask: torch.Tensor, head_grads: torch.Tensor, intermediate_grads: torch.Tensor):
        self.model = model
        self.head_mask = head_mask
        self.intermediate_mask = intermediate_mask
        self.current_head_mask = head_mask
        self.current_intermediate_mask = intermediate_mask
        self.head_grads = head_grads
        self.intermediate_grads = intermediate_grads
        self.head_mask_priority = None
        self.intermediate_mask_priority = None
        self.zs_schedule = None
        self.masks_schedule = None
        self.final_mask_schedule = []
        
    def gen_priority(self):
        raise NotImplementedError

    def gen_schedule(self, num_steps: int = 5):
        if self.head_mask_priority is None or self.intermediate_mask_priority is None:
            self.gen_priority(num_steps)
        head_masks, intermediate_masks = sequential_neuron_mask_generation(self.head_mask_priority), sequential_neuron_mask_generation(self.intermediate_mask_priority)
        head_zs = sequential_head_mask_generation(self.head_mask_priority)
        self.zs_schedule = [
            {
                'head_z': head_z,
                'intermediate_z': intermediate_z
            }
            for head_z, intermediate_z in zip(head_zs, intermediate_masks)
        ]
        self.masks_schedule = [
            {
                'head_mask': [v.to(self.model.device) for v in head_mask],
                'intermediate_mask': [v.to(self.model.device) for v in intermediate_mask],
            }
            for head_mask, intermediate_mask in zip(head_masks, intermediate_masks)
        ]
        now_head_mask, now_intermediate_mask = [v for v in self.head_mask.clone()], [v for v in self.intermediate_mask.clone()]
        for head_mask, intermediate_mask in zip(head_masks, intermediate_masks):
            self.final_mask_schedule.append({
                'head_mask': [v.to(self.model.device) if v is not None else None for v in now_head_mask],
                'intermediate_mask': [v.to(self.model.device) if v is not None else None for v in now_intermediate_mask],
            })
            if self.head_mask.ndim == 2 and self.intermediate_mask.ndim == 2:
                now_head_mask = [now_h[h.nonzero().squeeze()] if now_h is not None and now_h.size() else None for now_h, h in zip(now_head_mask, head_mask)]
                now_intermediate_mask = [now_i[i.nonzero().squeeze()] if now_i.size() else None for now_i, i in zip(now_intermediate_mask, intermediate_mask)]
            elif self.head_mask.ndim == 3 and self.intermediate_mask.ndim == 3:
                now_head_mask = [[now_h[h.nonzero().squeeze()] if now_h is not None and now_h.size() else None for now_h, h in zip(current_now_head_mask, current_head_mask)] for current_now_head_mask, current_head_mask in zip(now_head_mask, head_mask)]
                now_intermediate_mask = [[now_i[i.nonzero().squeeze()] if now_i.size() else None for now_i, i in zip(current_now_intermediate_mask, current_intermediate_mask)] for current_now_intermediate_mask, current_intermediate_mask in zip(now_intermediate_mask, intermediate_mask)]
            else:
                raise ValueError(f'Unsupported head_mask and intermediate_mask shape: {self.head_mask.shape}, {self.intermediate_mask.shape}')
        if self.head_mask.ndim == 2 and self.intermediate_mask.ndim == 2:
            assert all([len(m['head_mask']) == self.model.config.num_hidden_layers and len(m['intermediate_mask']) == self.model.config.num_hidden_layers for m in self.final_mask_schedule])
            assert all([h is None or (h == 1).all() for h in now_head_mask]) and all([i is None or (i == 1).all() for i in now_intermediate_mask])
            assert all([(h_m.sum() == self.head_mask[i].sum()) and (i_m.sum() == self.intermediate_mask[i].sum()) for m in self.final_mask_schedule for i, (h_m, i_m) in enumerate(zip(m['head_mask'], m['intermediate_mask']))])
        elif self.head_mask.ndim == 3 and self.intermediate_mask.ndim == 3:
            assert all([all([len(m) == self.model.config.num_hidden_layers for m in masks['head_mask']]) for masks in self.final_mask_schedule])
            assert all([all([len(m) == self.model.config.num_hidden_layers for m in masks['intermediate_mask']]) for masks in self.final_mask_schedule])
            assert all([h is None or (h == 1).all() for head_masks in now_head_mask for h in head_masks]) and all([i is None or (i == 1).all() for intermediate_masks in now_intermediate_mask for i in intermediate_masks])
            assert all([(h_m.sum() == self.head_mask[i].sum()) and (i_m.sum() == self.intermediate_mask[i].sum()) for m in self.final_mask_schedule for i, (h_m, i_m) in enumerate(zip(m['head_mask'], m['intermediate_mask']))])
        
    def gen_next_mask(self, mask_direct_final: bool = False):
        if self.masks_schedule is not None and len(self.masks_schedule) > 0:
            current_masks = self.final_mask_schedule.pop(0)
            self.current_head_mask = current_masks['head_mask']
            self.current_intermediate_mask = current_masks['intermediate_mask']
            if not mask_direct_final:
                return self.masks_schedule.pop(0)
            else:
                self.masks_schedule.pop(0)
                return current_masks
        elif self.masks_schedule is None:
            raise ValueError('Please call gen_schedule() first.')
        else:
            return {
                'head_mask': None,
                'intermediate_mask': None,
            }
        
    def gen_next_z(self):
        if self.zs_schedule is not None and len(self.zs_schedule) > 0:
            return self.zs_schedule.pop(0)
        elif self.zs_schedule is None:
            raise ValueError('Please call gen_schedule() first.')
        else:
            return None


class RandomPruningScheduler(BasePruningScheduler):
    def __init__(self, model, head_mask, intermediate_mask, head_grads, intermediate_grads):
        super().__init__(model, head_mask, intermediate_mask, head_grads, intermediate_grads)
        
        
    def gen_priority(self, num_steps: int = 5):
        head_mask, intermediate_mask = self.head_mask * num_steps, self.intermediate_mask * num_steps
        head_zeros, intermediate_zeros = (head_mask == 0).nonzero(), (intermediate_mask == 0).nonzero()
        head_zeros, intermediate_zeros = head_zeros[torch.randperm(head_zeros.size(0))], intermediate_zeros[torch.randperm(intermediate_zeros.size(0))]
        head_zero_chunks, intermediate_zero_chunks = torch.chunk(head_zeros, num_steps), torch.chunk(intermediate_zeros, num_steps)
        if head_mask.ndim == 2 and intermediate_mask.ndim == 2:
            for i in range(num_steps):
                head_mask[head_zero_chunks[i][:, 0], head_zero_chunks[i][:, 1]] = i
                intermediate_mask[intermediate_zero_chunks[i][:, 0], intermediate_zero_chunks[i][:, 1]] = i
        elif head_mask.ndim == 3 and intermediate_mask.ndim == 3:
            for i in range(num_steps):
                head_mask[head_zero_chunks[i][:, 0], head_zero_chunks[i][:, 1], head_zero_chunks[i][:, 2]] = i
                intermediate_mask[intermediate_zero_chunks[i][:, 0], intermediate_zero_chunks[i][:, 1], intermediate_zero_chunks[i][:, 2]] = i
        else:
            raise ValueError('head_mask and intermediate_mask should be 2D or 3D.')
        self.head_mask_priority = head_mask
        self.intermediate_mask_priority = intermediate_mask
        return head_mask, intermediate_mask
    
class SequentialPruningScheduler(BasePruningScheduler):
    def __init__(self, model, head_mask, intermediate_mask, head_grads, intermediate_grads):
        super().__init__(model, head_mask, intermediate_mask, head_grads, intermediate_grads)
        
    def _chunk_by_layer(self, mask, num_chunks: int, num_layers: int = 12):
        layer_to_chunks = []
        base, base_size, residual = 0, num_layers // num_chunks, num_layers % num_chunks
        for i in range(num_chunks):
            step_chunk_size = base_size if i < num_chunks - residual else base_size + 1
            layer_to_chunks.append(list(range(base, base + step_chunk_size)))
            base += step_chunk_size
        mask_chunks = []
        for i in range(num_chunks):
            mask_chunks.append(mask[(mask[:, 0] >= layer_to_chunks[i][0]) & (mask[:, 0] <= layer_to_chunks[i][-1])])
        mask_chunks.reverse()
        return mask_chunks
        
        
    def gen_priority(self, num_steps: int = 5):
        head_mask, intermediate_mask = self.head_mask * num_steps, self.intermediate_mask * num_steps
        head_zeros, intermediate_zeros = (head_mask == 0).nonzero(), (intermediate_mask == 0).nonzero()
        head_zeros, intermediate_zeros = head_zeros[torch.randperm(head_zeros.size(0))], intermediate_zeros[torch.randperm(intermediate_zeros.size(0))]
        head_zero_chunks, intermediate_zero_chunks = self._chunk_by_layer(head_zeros, num_steps, self.model.config.num_hidden_layers), self._chunk_by_layer(intermediate_zeros, num_steps, self.model.config.num_hidden_layers)
        for i in range(num_steps):
            head_mask[head_zero_chunks[i][:, 0], head_zero_chunks[i][:, 1]] = i
            intermediate_mask[intermediate_zero_chunks[i][:, 0], intermediate_zero_chunks[i][:, 1]] = i
        self.head_mask_priority = head_mask
        self.intermediate_mask_priority = intermediate_mask
        return head_mask, intermediate_mask


class OncePruningScheduler(RandomPruningScheduler):
    def __init__(self, model, head_mask, intermediate_mask, head_grads, intermediate_grads):
        super().__init__(model, head_mask, intermediate_mask, head_grads, intermediate_grads)
        
    def gen_priority(self, num_steps: int = 5):
        return super().gen_priority(1)


class SaliencyPruningScheduler(BasePruningScheduler):
    def __init__(self, model, head_mask, intermediate_mask, head_grads, intermediate_grads, dataloader, mac_constraints, seq_len: int = 128):
        super().__init__(model, head_mask, intermediate_mask, head_grads, intermediate_grads)
        self.dataloader = dataloader
        self.current_head_score = None
        self.current_intermediate_score = None
        self.mac_constraints = mac_constraints
        self.seq_len = seq_len
    
    def gen_priority(self, num_steps: int = 5):
        
        return super().gen_priority()
    
    def gen_schedule(self, num_steps: int = 5):
        self.masks_schedule = self.mac_constraints
        
    def calculate_saliency(self):
        self.model.reset_masks()
        head_grads, intermediate_grads = collect_mask_grads(self.model, self.dataloader)
        self.current_head_score = compute_fisher_info(head_grads)
        self.current_intermediate_score = compute_fisher_info(intermediate_grads)
        
    def gen_next_mask(self):
        mac_constraint = self.masks_schedule.pop(0)
        assert 0 < mac_constraint < 1
        if self.current_head_score is None or self.current_intermediate_score is None:
            self.calculate_saliency()
            
        # Reversed search, compared with the original mask-tuning paper
        gen_head_mask, gen_intermediate_mask = search_mac_reverse(
            self.model.config,
            self.current_head_score,
            self.current_intermediate_score,
            self.seq_len,
            mac_constraint,
            head_mask_condition=self.current_head_mask,
            neuron_mask_condition=self.current_intermediate_mask,
        )
        
        self.current_head_mask = [
            current_mask[gen_mask.nonzero().squeeze()].view(-1).contiguous().clone() if current_mask is not None and current_mask.size() else torch.tensor([]).to(self.model.device)
            for gen_mask, current_mask in zip(gen_head_mask, self.current_head_mask)
        ] # Using .view(-1) to convert potential 0D tensor to 1D tensor
        self.current_intermediate_mask = [
            current_mask[gen_mask.nonzero().squeeze()].view(-1).contiguous().clone() if current_mask is not None and current_mask.size() else torch.tensor([]).to(self.model.device)
            for gen_mask, current_mask in zip(gen_intermediate_mask, self.current_intermediate_mask)
        ] # Using .view(-1) to convert potential 0D tensor to 1D tensor
        
        self.current_head_score = None
        self.current_intermediate_score = None
        return {
            'head_mask': gen_head_mask,
            'intermediate_mask': gen_intermediate_mask,
        }
    
    def gen_next_z(self):
        pass