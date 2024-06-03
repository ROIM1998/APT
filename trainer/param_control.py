import sys
import torch
import logging
import loralib as lora
import torch.nn as nn

from typing import List, Dict, Optional, Tuple
from utils.minus_utils import lora_to_distill, linear_to_lora, distill_to_lora, shrink_and_expand_pruning_lora, lora_to_prunelora, lora_to_linear
from .model_arch import ModelArch, NAME2ATTR
from .allocation_strategy import *
from tqdm import tqdm

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

ABBREVIATIONS = {
    'q': 'query',
    'k': 'key',
    'v': 'value',
    'ao': 'attention.output',
    'i': 'intermediate',
    'io': 'intermediate.output',
    'eq': 'enc_self_query',
    'ek': 'enc_self_key',
    'ev': 'enc_self_value',
    'eo': 'enc_self_output',
    'dq': 'dec_self_query',
    'dk': 'dec_self_key',
    'dv': 'dec_self_value',
    'do': 'dec_self_output',
    'cq': 'cross_query',
    'ck': 'cross_key',
    'cv': 'cross_value',
    'co': 'cross_output',
    'ei': 'encoder_i',
    'ei0': 'encoder_i0',
    'ei1': 'encoder_i1',
    'eio': 'encoder_io',
    'di': 'decoder_i',
    'di0': 'decoder_i0',
    'di1': 'decoder_i1',
    'dio': 'decoder_io',
}

ACTFN = {
    'lora': None,
    'pa': 'relu',
}


PARAM_ALLOCATION_FUNC = {
    'free_inout': adjust_r_then_shrink_inout,
    'static_inout': adjust_r_with_static_inout,
    'preserved_inout': adjust_r_with_preserved_inout,
    'none': None
}

class ParamController:
    def __init__(self, model: torch.nn.Module, args = None, teacher_config: Optional[Dict[str, List[int]]] = None, student_config: Optional[Dict[str, List[int]]] = None, warmup_config: Optional[Dict[str, List[int]]] = None, bias_only: bool = False, lora_with_bias: bool = False, param_allocation_strategy: str = 'none', adapter_type: str = 'lora', max_lora_r: int = -1, **kwargs):
        if args is not None:
            fileHandler = logging.FileHandler("{0}/{1}.log".format(args.output_dir, 'trainer'))
            fileHandler.setFormatter(logFormatter)
            logger.addHandler(fileHandler)
        self.model = model
        self.named_modules = dict(self.model.named_modules())
        self.bias_only = bias_only
        self.lora_with_bias = lora_with_bias
        self.model_arch = ModelArch(self.model)
        self.name2template = self.model_arch.get_name2template()
        self.teacher_config = teacher_config
        self.student_config = student_config
        if warmup_config is None:
            self.warmup_config = teacher_config
        else:
            self.warmup_config = warmup_config
        self.param_config = {}
        self.warmup_param_config = {}
        self.teacher_tuning_layers = set()
        self.student_tuning_layers = set()
        for i in range(self.model.config.num_hidden_layers):
            self.param_config[i] = {}
            self.warmup_param_config[i] = set()
            for k in self.name2template:
                module_is_teacher = self.teacher_config is not None and k in self.teacher_config and i in self.teacher_config[k]
                module_is_student = self.student_config is not None and k in self.student_config and i in self.student_config[k]
                if module_is_teacher and i not in self.teacher_tuning_layers:
                    self.teacher_tuning_layers.add(i)
                if module_is_student and i not in self.student_tuning_layers:
                    self.student_tuning_layers.add(i)
                module_in_warmup = self.warmup_config is not None and k in self.warmup_config and i in self.warmup_config[k]
                if module_in_warmup:
                    self.warmup_param_config[i].add(k)
                if module_is_teacher and module_is_student:
                    self.param_config[i][k] = 'teacher_student'
                elif module_is_teacher:
                    self.param_config[i][k] = 'teacher'
                elif module_is_student:
                    self.param_config[i][k] = 'student'
                else:
                    self.param_config[i][k] = 'none'
        self.warmup_params = self._config_to_params(self.warmup_config)
        self.teacher_params = self._config_to_params(teacher_config) if teacher_config is not None else None
        self.student_params = self._config_to_params(student_config) if student_config is not None else None
        self.adapter_type = adapter_type
        # self.teacher_modules = self._config_to_module(teacher_config) if teacher_config is not None else None
        # self.student_modules = self._config_to_module(student_config) if student_config is not None else None
        self.tuning_param_number = 0
        self.tuning_param_number_fixed = False
        self.schedule = None
        self.tuning_schedule = None
        self.next_tuning_param_num = None
        self.next_tuning_step, self.next_tuning_param_num = None, None
        self.target_tuning_num = None
        self.next_pruning_step = None
        self.next_pruning_ratio = None
        self.model.is_teacher = False
        self.model.is_student = False
        self.model.is_colearning = False
        self.model.is_distilling = False
        self.param_allocation_func = PARAM_ALLOCATION_FUNC[param_allocation_strategy] if param_allocation_strategy in PARAM_ALLOCATION_FUNC else None
        self.max_lora_r = max_lora_r
        self.beta_1 = 0.85
        self.beta_2 = 0.85
    
    @classmethod
    def parse_tuning_param_str(cls, parsing_str: Optional[str]):
        if parsing_str is None:
            return None
        returned_dict = {}
        params = parsing_str.split(',')
        for param in params:
            name, layers = param.split(':')
            layers = list(range(int(layers.split('-')[0]), int(layers.split('-')[1]) + 1)) if '-' in layers else []
            returned_dict[ABBREVIATIONS[name]] = layers
        return returned_dict
    
    def _config_to_params(self, config: Optional[Dict[str, List[int]]]):
        if config is None:
            return set()
        params = set()
        num_layers = self.model.config.num_hidden_layers
        for k, v in config.items():
            if "%d" in self.name2template[k]:
                composed_ks = []
                for layer_i in v:
                    if layer_i >= num_layers:
                        raise ValueError(f"Layer {layer_i} is out of range for model {self.model_arch.model_category}")                            
                    else:
                        composed_ks.append(self.name2template[k] % (layer_i))
            else:
                composed_ks =[ self.name2template[k]]
            for composed_k in composed_ks:
                module = self.named_modules.get(composed_k, None)
                if module is None:
                    continue   
                if isinstance(module, lora.Linear):
                    params.add(composed_k + '.lora_A')
                    params.add(composed_k + '.lora_B')
                    if self.lora_with_bias:
                        params.add(composed_k + '.bias')
                else:
                    params.add(composed_k + '.bias')
                    if not self.bias_only:
                        params.add(composed_k + '.weight')
        return params
    
    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
            
    def unfreeze(self):
        for p in self.model.parameters():
            p.requires_grad_(True)
    
    def clear_states(self):
        self.model.is_teacher = False
        self.model.is_student = False
        self.model.is_colearning = False

    def clear_grads(self):
        for p in self.model.parameters():
            p.grad = None
            
    def tune_lora_only(self):
        for n, p in self.model.named_parameters():
            if 'lora_' in n:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
                
    def model_as_teacher(self):
        # if self.model.is_teacher:
        #     return
        for n, p in self.model.named_parameters():
            if n in self.teacher_params:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
        self.model.is_teacher = True
        self.model.is_student = False
        self.model.is_colearning = False
                
    def model_as_student(self):
        # if self.model.is_student:
        #     return
        for n, p in self.model.named_parameters():
            if n in self.student_params:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
        self.model.is_teacher = False
        self.model.is_student = True
        self.model.is_colearning = False
        
    def model_teacher_with_student(self):
        # if self.model.is_colearning:
        #     return
        for n, p in self.model.named_parameters():
            if n in self.teacher_params or n in self.student_params:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
                p.grad = None
        self.model.is_teacher = False
        self.model.is_student = False
        self.model.is_colearning = True
        
    def finetune(self):
        for n, p in self.model.named_parameters():
            if 'embeddings' not in n:
                p.requires_grad_(True)
                if 'lora_' in n:
                    logger.warning("WARNING: Found lora parameter %s for fine-tuning!" % n)
            else:
                p.requires_grad_(False)
    
    def get_layer(self, i, k):
        return self.model_arch.get_layer(i, k)

    def get_parent_layer(self, i, k):
        return self.model_arch.get_parent_layer(i, k)

    def convert_to_pre_pruning_lora_teacher(self):
        logger.info("Converting to pre-pruning LoRA warmup tuning layers")
        self.teacher_params = set()
        self.student_params = set()
        r_using = self.model.config.lora_r
        alpha_using = 16 if self.model.config.do_distill else self.model.config.lora_alpha
        for i in tqdm(self.warmup_param_config):
            # layer_ffn_o = get_ffn2(self.model, i)
            for k in self.warmup_param_config[i]:
                k_attr = NAME2ATTR[k][self.model_arch.model_category]
                parent_layer = self.get_parent_layer(i, k)
                composed_k = self.name2template[k] % i
                self.teacher_params.add(composed_k + '.lora_A')
                self.teacher_params.add(composed_k + '.lora_B')
                if self.lora_with_bias:
                    self.teacher_params.add(composed_k + '.bias')
                src_layer = getattr(parent_layer, k_attr)
                if isinstance(src_layer, lora.PruningLinear) or src_layer is None:
                    continue
                if not isinstance(src_layer, lora.LoRALayer) or getattr(src_layer, 'r', 0) == 0:
                    # Convert Linear layer to vanilla LoRALinear, and then to PruningLinear
                    lora_layer = linear_to_lora(src_layer, r=r_using, lora_alpha=alpha_using)
                else:
                    lora_layer = src_layer
                setattr(parent_layer, k_attr, lora_to_prunelora(lora_layer, lora_layer.r, lora_layer.lora_alpha, act_fn=ACTFN[self.adapter_type]))
        self.clear_states()
        
    def convert_to_pruning_lora_teacher(self):
        logger.info("Converting to pruning LoRA teacher")
        self.teacher_params = set()
        self.student_params = set()
        r_using = self.model.config.lora_r
        alpha_using = 16 if self.model.config.do_distill else self.model.config.lora_alpha
        for i in tqdm(self.param_config):
            # layer_ffn_o = get_ffn2(self.model, i)
            for k in self.param_config[i]:
                k_attr = NAME2ATTR[k][self.model_arch.model_category]
                parent_layer = self.get_parent_layer(i, k)
                composed_k = self.name2template[k] % i
                if 'teacher' in self.param_config[i][k] or self.param_config[i][k] == 'shared':
                    self.teacher_params.add(composed_k + '.lora_A')
                    self.teacher_params.add(composed_k + '.lora_B')
                    if self.lora_with_bias:
                        self.teacher_params.add(composed_k + '.bias')
                    src_layer = getattr(parent_layer, k_attr)
                    if isinstance(src_layer, lora.PruningLinear) or src_layer is None:
                        continue
                    if not isinstance(src_layer, lora.LoRALayer):
                        lora_layer = linear_to_lora(src_layer, r=r_using, lora_alpha=alpha_using)
                    else:
                        lora_layer = src_layer
                    new_layer = lora_to_prunelora(lora_layer, lora_layer.r, lora_layer.lora_alpha, act_fn=ACTFN[self.adapter_type])
                    if any([isinstance(m, lora.DistillLinear) for m in self.model.modules()]):
                        # If self-distillation has been applied, we need to convert the teacher LoRA layer to DistillLinear to maintain the teacher model
                        new_layer = lora_to_distill(new_layer, r=new_layer.r, teacher_r=new_layer.r, lora_alpha=new_layer.lora_alpha, teacher_lora_alpha=new_layer.lora_alpha, retained_indices=None, teacher_retained_indices=None, copy_to_teacher=True)
                    setattr(parent_layer, k_attr, new_layer)
                else:
                    # Convert non-teacher tuning LoRA layers to frozen Linear layer
                    layer = getattr(parent_layer, k_attr, None)
                    if isinstance(layer, lora.LoRALayer):
                        print('Converting %s to Linear' % composed_k)
                        setattr(parent_layer, k_attr, lora_to_linear(layer))
        self.clear_states()

    def convert_to_distill(self, head_mask: Optional[torch.Tensor] = None, intermediate_mask: Optional[torch.Tensor] = None):
        self.teacher_params = set()
        self.student_params = set()
        if not self.model.config.do_distill:
            raise ValueError("No need to do DistillLinear conversion when do_distill is False")
        model = self.model
        # hidden_per_head = model.config.hidden_size // model.config.num_attention_heads
        for i in tqdm(self.param_config):
            for k in self.param_config[i]:                
                if 'teacher' in self.param_config[i][k] or 'student' in self.param_config[i][k]:
                    k_attr = NAME2ATTR[k][self.model_arch.model_category]
                    parent_layer = self.get_parent_layer(i, k)
                    composed_k = self.name2template[k] % i
                    if 'student' in self.param_config[i][k]:
                        self.student_params.add(composed_k + '.lora_A')
                        self.student_params.add(composed_k + '.lora_B')
                        # if is_ffn:
                        #     lora_r_use = max(8, model.config.lora_r // 8)
                        # else:
                        lora_r_use = model.config.lora_r
                    else:
                        lora_r_use = 0
                    if 'teacher' in self.param_config[i][k]:
                        self.teacher_params.add(composed_k + '.teacher_lora_A')
                        self.teacher_params.add(composed_k + '.teacher_lora_B')
                        if self.lora_with_bias:
                            self.teacher_params.add(composed_k + '.bias')
                        teacher_r_use = 8
                    else:
                        teacher_r_use = 0
                    # TODO: support different adapter types
                    setattr(parent_layer, k_attr, lora_to_distill(getattr(parent_layer, k_attr), r=lora_r_use, teacher_r=teacher_r_use, lora_alpha=model.config.lora_alpha, teacher_lora_alpha=16, retained_indices=None, teacher_retained_indices=None))
        self._tuning_layer_transformation()
        self._tuning_head()
        self.clear_states()
        self.model.is_distilling = True

    def convert_to_self_momentum_distill(self):
        self.teacher_params = set()
        self.student_params = set()
        if not self.model.config.do_distill:
            raise ValueError("No need to do DistillLinear conversion when do_distill is False")
        for i in tqdm(self.param_config):
            for k in self.param_config[i]:                
                if 'teacher' in self.param_config[i][k] or self.param_config[i][k] == 'shared':
                    k_attr = NAME2ATTR[k][self.model_arch.model_category]
                    parent_layer = self.get_parent_layer(i, k)
                    current_layer: nn.Linear = getattr(parent_layer, k_attr)
                    composed_k = self.name2template[k] % i
                    self.teacher_params.add(composed_k + '.lora_A')
                    self.teacher_params.add(composed_k + '.lora_B')
                    if self.lora_with_bias:
                        self.teacher_params.add(composed_k + '.bias')
                    if isinstance(current_layer, lora.DistillLinear):
                        # Already converted, just copying the parameters
                        del current_layer.teacher_lora_A
                        del current_layer.teacher_lora_B
                        current_layer.teacher_lora_A = nn.Parameter(current_layer.lora_A.detach().clone())
                        current_layer.teacher_lora_B = nn.Parameter(current_layer.lora_B.detach().clone())
                        if getattr(current_layer, 'teacher_out_transformation', None) is not None:
                            del current_layer.teacher_out_transformation
                            current_layer.teacher_out_transformation = nn.Parameter(current_layer.out_transformation.detach().clone())
                        if getattr(current_layer, 'teacher_in_transformation', None) is not None:
                            del current_layer.teacher_in_transformation
                            current_layer.teacher_in_transformation = nn.Parameter(current_layer.in_transformation.detach().clone())
                        if getattr(current_layer, 'teacher_in_retained_indices', None) is not None:
                            current_layer.teacher_in_retained_indices = current_layer.in_retained_indices
                        if getattr(current_layer, 'teacher_out_retained_indices', None) is not None:
                            current_layer.teacher_out_retained_indices = current_layer.out_retained_indices
                    elif isinstance(current_layer, lora.Linear):
                        # Not-yet converted, convert to DistillLinear with copy_to_teacher=True
                        # TODO: support different adapter types
                        new_layer = lora_to_distill(current_layer, r=current_layer.r, teacher_r=current_layer.r, lora_alpha=current_layer.lora_alpha, teacher_lora_alpha=current_layer.lora_alpha, retained_indices=None, teacher_retained_indices=None, copy_to_teacher=True)
                        # Make sure the scaling is correct
                        new_layer.scaling = current_layer.scaling
                        new_layer.teacher_scaling = current_layer.scaling
                        setattr(parent_layer, k_attr, new_layer)

        self._tuning_layer_transformation()
        self._tuning_head(self_distill=True)
        self.clear_states()
        self.model.is_distilling = True
                    
    def convert_to_post_distillation_lora_student(self):
        self.teacher_params = set()
        self.student_params = set()
        for i in tqdm(self.param_config):
            for k in self.param_config[i]:
                if 'student' in self.param_config[i][k] or self.param_config[i][k] == 'shared':
                    k_attr = NAME2ATTR[k][self.model_arch.model_category]
                    parent_layer = self.get_parent_layer(i, k)
                    src_layer = getattr(parent_layer, k_attr)
                    if isinstance(src_layer, lora.DistillLinear):
                        # TODO: support different adapter types
                        setattr(parent_layer, k_attr, distill_to_lora(src_layer, r=src_layer.r, lora_alpha=src_layer.lora_alpha))
                    composed_k = self.name2template[k] % i
                    self.student_params.add(composed_k + '.lora_A')
                    self.student_params.add(composed_k + '.lora_B')
                    if self.lora_with_bias:
                        self.student_params.add(composed_k + '.bias')
        self._tuning_head()
        self.clear_states()
        self.model.is_distilling = False
        
    def convert_to_post_distillation_ft_student(self):
        self.teacher_params = set()
        self.student_params = set()
        for i in tqdm(self.param_config):
            for k in self.param_config[i]:
                if 'student' in self.param_config[i][k] or self.param_config[i][k] == 'shared':
                    k_attr = NAME2ATTR[k][self.model_arch.model_category]
                    parent_layer = self.get_parent_layer(i, k)
                    src_layer = getattr(parent_layer, k_attr)
                    if isinstance(src_layer, lora.Linear):
                        setattr(parent_layer, k_attr, lora_to_linear(src_layer))
                    composed_k = self.name2template[k] % i
                    self.student_params.add(composed_k + '.lora_A')
                    self.student_params.add(composed_k + '.lora_B')
                    if self.lora_with_bias:
                        self.student_params.add(composed_k + '.bias')
        self.finetune()
        self.clear_states()
        self.model.is_distilling = False

    def set_grafting_mask(self, mode: bool = True, target: str = 'teacher', requires_grad: bool = False):
        assert target in ['teacher', 'student']
        # Reset fisher status
        for i in self.param_config:
            for k in self.param_config[i]:
                if target in self.param_config[i][k]:
                    k_attr = NAME2ATTR[k][self.model_arch.model_category]
                    parent_layer = self.get_parent_layer(i, k)
                    src_layer = getattr(parent_layer, k_attr)
                    if isinstance(src_layer, lora.PruningLinear):
                        if mode:
                            src_layer.set_grafting_mask(requires_grad=requires_grad)
                        else:
                            src_layer.remove_grafting_mask()
    
    def restore_dims(self, target: str = 'teacher'):
        assert target in ['teacher', 'student']
        for i in self.param_config:
            for k in self.param_config[i]:
                if target in self.param_config[i][k]:
                    k_attr = NAME2ATTR[k][self.model_arch.model_category]
                    parent_layer = self.get_parent_layer(i, k)
                    src_layer = getattr(parent_layer, k_attr)
                    if isinstance(src_layer, lora.PruningLinear):
                        src_layer.restore_dims()
                        
    def generate_pruning_schedule(self, pruning_start_step: int, pruning_end_step: int, starting_mac_constraint: float = 1., mac_constraint: float = 0.4, num_prunes: int = 5, scheduler_reduction_type: str = 'cubic', scheduler_frequency_type: str = 'linear', warmup_steps: int = -1) -> Tuple[List[float]]:
        assert 0 < mac_constraint <= 1
        if scheduler_frequency_type == 'once':
            pruning_steps = [pruning_start_step]
        elif scheduler_frequency_type == 'linear':
            # if num_prunes == 1:
            #     raise ValueError('num_prunes must be greater than 1 when scheduler_frequency_type is linear. Please use the once frequency type instead.')
            if pruning_start_step >= pruning_end_step:
                raise ValueError('pruning_start_step must be less than pruning_end_step when scheduler_frequency_type is linear.')
            pruning_steps = torch.linspace(pruning_start_step, pruning_end_step, num_prunes + 1).int().tolist()
        
        if len(pruning_steps) == 1:
            pruning_schedule = [starting_mac_constraint, mac_constraint]
        else:
            warmup_end = pruning_start_step if warmup_steps == -1 else warmup_steps + pruning_start_step
            if scheduler_reduction_type == 'cubic':
                pruning_schedule = [min(starting_mac_constraint, mac_constraint + (starting_mac_constraint - mac_constraint) * (((pruning_end_step - step) / (pruning_end_step - warmup_end)) ** 3)) for step in tqdm(pruning_steps)]
            elif scheduler_reduction_type == 'linear':
                pruning_schedule = [min(starting_mac_constraint, mac_constraint + (starting_mac_constraint - mac_constraint) * (pruning_end_step - step) / (pruning_end_step - warmup_end)) for step in tqdm(pruning_steps)]
            pruning_steps, pruning_schedule = pruning_steps, pruning_schedule
        self.schedule = list(zip(pruning_steps, pruning_schedule))
        self.next_pruning_step, self.next_pruning_ratio = self.schedule.pop(0)
        return pruning_steps, pruning_schedule
    
    def gen_next_pruning_step(self) -> Optional[int]:
        if self.schedule:
            self.next_pruning_step, self.next_pruning_ratio = self.schedule.pop(0)
        return self.next_pruning_step, self.next_pruning_ratio
    
    def generate_tuning_schedule(self, tuning_start_step: int, tuning_end_step: int, target_tuning_num: int, num_tunings: int = 5, scheduler_increase_type: str = 'linear', scheduler_frequency_type: str = 'linear', warmup_steps: int = -1) -> Tuple[List[float]]:
        assert target_tuning_num >= 0
        self.set_tuning_param_number()
        assert self.tuning_param_number > 0
        self.target_tuning_num = target_tuning_num
        tuning_param_number = self.tuning_param_number
        if scheduler_frequency_type == 'once':
            tuning_steps = [tuning_start_step]
        elif scheduler_frequency_type == 'linear':
            # if num_prunes == 1:
            #     raise ValueError('num_prunes must be greater than 1 when scheduler_frequency_type is linear. Please use the once frequency type instead.')
            if tuning_start_step >= tuning_end_step:
                raise ValueError('tuning_start_step must be less than tuning_end_step when scheduler_frequency_type is linear.')
            tuning_steps = torch.linspace(tuning_start_step, tuning_end_step, num_tunings + 1).int().tolist()
        
        if len(tuning_steps) == 1:
            tuning_schedule = [tuning_param_number, target_tuning_num]
        else:
            warmup_end = tuning_end_step if warmup_steps == -1 else warmup_steps + tuning_start_step
            if scheduler_increase_type == 'cubic':
                tuning_schedule = [target_tuning_num - max(0, (target_tuning_num - tuning_param_number) * (((warmup_end - step) / (warmup_end - tuning_start_step)) ** 3)) for step in tqdm(tuning_steps)]
            elif scheduler_increase_type == 'linear':
                tuning_schedule = [tuning_param_number + max(0, (target_tuning_num - tuning_param_number) * (step - tuning_start_step) / (warmup_end - tuning_start_step)) for step in tqdm(tuning_steps)]

        self.tuning_schedule = list(zip(tuning_steps, tuning_schedule))[1:]
        self.next_tuning_step, self.next_tuning_param_num = self.tuning_schedule.pop(0)
        return tuning_steps, tuning_schedule
        
    def gen_next_tuning_step(self) -> Optional[int]:
        if self.tuning_schedule:
            self.next_tuning_step, self.next_tuning_param_num = self.tuning_schedule.pop(0)
        return self.next_tuning_step, self.next_tuning_param_num
        
    def _tuning_head(self, target='teacher', self_distill: bool = False):
        model = self.model
        if hasattr(model, 'lm_head'):
            # T5 model
            if not isinstance(model.lm_head, lora.LoRALayer) and model.config.apply_lora:
                new_layer = linear_to_lora(model.lm_head, r=8, lora_alpha=16)
                if model.config.do_distill:
                    new_layer = lora_to_distill(new_layer, r=8, lora_alpha=16, teacher_r=8, teacher_lora_alpha=16, copy_to_teacher=self_distill)
                model.lm_head = new_layer
            elif isinstance(model.lm_head, lora.DistillLinear) and self_distill:
                # Already converted, just copying the parameters
                del model.lm_head.teacher_lora_A
                del model.lm_head.teacher_lora_B
                model.lm_head.teacher_lora_A = nn.Parameter(model.lm_head.lora_A.detach().clone())
                model.lm_head.teacher_lora_B = nn.Parameter(model.lm_head.lora_B.detach().clone())
            if target == 'teacher':
                self.teacher_params.add('lm_head.lora_A')
                self.teacher_params.add('lm_head.lora_B')
                if self.lora_with_bias:
                    self.teacher_params.add('lm_head.bias')
            else:
                self.student_params.add('lm_head.lora_A')
                self.student_params.add('lm_head.lora_B')
                if self.lora_with_bias:
                    self.student_params.add('lm_head.bias')
        elif hasattr(model, 'qa_outputs'):
            # QuestionAnswering models, either BERT or RoBERTa
            if not isinstance(model.qa_outputs, lora.LoRALayer) and model.config.apply_lora:
                new_layer = linear_to_lora(model.qa_outputs, r=8, lora_alpha=16)
                if model.config.do_distill:
                    new_layer = lora_to_distill(new_layer, r=8, lora_alpha=16, teacher_r=8, teacher_lora_alpha=16, copy_to_teacher=self_distill)
                model.qa_outputs = new_layer
            elif isinstance(model.qa_outputs, lora.DistillLinear) and self_distill:
                del model.qa_outputs.teacher_lora_A
                del model.qa_outputs.teacher_lora_B
                model.qa_outputs.teacher_lora_A = nn.Parameter(model.qa_outputs.lora_A.detach().clone())
                model.qa_outputs.teacher_lora_B = nn.Parameter(model.qa_outputs.lora_B.detach().clone())
            if target == 'teacher':
                self.teacher_params.add('qa_outputs.lora_A')
                self.teacher_params.add('qa_outputs.lora_B')
                if self.lora_with_bias:
                    self.teacher_params.add('qa_outputs.bias')
            else:
                self.student_params.add('qa_outputs.lora_A')
                self.student_params.add('qa_outputs.lora_B')
                if self.lora_with_bias:
                    self.student_params.add('qa_outputs.bias')
        elif hasattr(model.classifier, 'dense'):
            # RoBERTa model
            if not isinstance(model.classifier.dense, lora.LoRALayer) and model.config.apply_lora:
                new_layer = linear_to_lora(getattr(model.classifier, 'dense'), r=8, lora_alpha=16)
                if model.config.do_distill:
                    new_layer = lora_to_distill(new_layer, r=8, lora_alpha=16, teacher_r=8, teacher_lora_alpha=16, copy_to_teacher=self_distill)
                setattr(model.classifier, 'dense', new_layer)
            elif isinstance(model.classifier.dense, lora.DistillLinear) and self_distill:
                del model.classifier.dense.teacher_lora_A
                del model.classifier.dense.teacher_lora_B
                model.classifier.dense.teacher_lora_A = nn.Parameter(model.classifier.dense.lora_A.detach().clone())
                model.classifier.dense.teacher_lora_B = nn.Parameter(model.classifier.dense.lora_B.detach().clone())
            if target == 'teacher':
                self.teacher_params.add('classifier.dense.lora_A')
                self.teacher_params.add('classifier.dense.lora_B')
                if self.lora_with_bias:
                    self.teacher_params.add('classifier.dense.bias')
            else:
                self.student_params.add('classifier.dense.lora_A')
                self.student_params.add('classifier.dense.lora_B')
                if self.lora_with_bias:
                    self.student_params.add('classifier.dense.bias')
        else:
            # BERT model, tuning the pooler's dense layer instead of the classifier
            if not isinstance(model.bert.pooler.dense, lora.LoRALayer) and model.config.apply_lora:
                new_layer = linear_to_lora(model.bert.pooler.dense, r=8, lora_alpha=16)
                if model.config.do_distill:
                    new_layer = lora_to_distill(new_layer, r=8, lora_alpha=16, teacher_r=8, teacher_lora_alpha=16, copy_to_teacher=self_distill)
                setattr(model.bert.pooler, 'dense', new_layer)
            elif isinstance(model.bert.pooler.dense, lora.DistillLinear) and self_distill:
                del model.bert.pooler.dense.teacher_lora_A
                del model.bert.pooler.dense.teacher_lora_B
                model.bert.pooler.dense.teacher_lora_A = nn.Parameter(model.bert.pooler.dense.lora_A.detach().clone())
                model.bert.pooler.dense.teacher_lora_B = nn.Parameter(model.bert.pooler.dense.lora_B.detach().clone())
            if target == 'teacher':
                self.teacher_params.add('bert.pooler.dense.lora_A')
                self.teacher_params.add('bert.pooler.dense.lora_B')
                if self.lora_with_bias:
                    self.teacher_params.add('bert.pooler.dense.bias')
            else:
                self.student_params.add('bert.pooler.dense.lora_A')
                self.student_params.add('bert.pooler.dense.lora_B')
                if self.lora_with_bias:
                    self.student_params.add('bert.pooler.dense.bias')
            
    def _tuning_layer_transformation(self, target='student'):
        model = self.model
        if hasattr(model, 'layer_transformation') and model.layer_transformation is not None:
            # if not isinstance(model.layer_transformation, lora.LoRALayer) and model.config.apply_lora:
            #     model.layer_transformation = linear_to_lora(model.layer_transformation, r=8, lora_alpha=16)
            if target == 'teacher':
                self.teacher_params.add('layer_transformation.lora_A')
                self.teacher_params.add('layer_transformation.lora_B')
                if self.lora_with_bias:
                    self.teacher_params.add('layer_transformation.bias')
            else:
                self.student_params.add('layer_transformation.lora_A')
                self.student_params.add('layer_transformation.lora_B')
                if self.lora_with_bias:
                    self.student_params.add('layer_transformation.bias')
    

    def model_decouple_as_teacher(self):
        if not self.model.config.do_distill:
            raise ValueError("No need to decouple model when do_distill is False")
        self.model.is_colearning = False
        for n, p in self.model.named_parameters():
            if n in self.teacher_params:
                p.requires_grad = True
            else:
                p.requires_grad = False
                p.grad = None

    def model_decouple_as_student(self):
        if not self.model.config.do_distill:
            raise ValueError("No need to decouple model when do_distill is False")
        self.model.is_colearning = False
        for n, p in self.model.named_parameters():
            if n in self.student_params:
                p.requires_grad = True
            else:
                p.requires_grad = False
                p.grad = None
                
    def count_model_params(self):
        return sum(p.numel() for p in self.model.parameters())
        
    def count_tuning_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def set_tuning_param_number(self, number: Optional[int] = None):
        if self.tuning_param_number_fixed:
            logger.warning("Warning: tuning_param_number is already fixed, cannot be changed")
            return False
        if number is None:
            self.tuning_param_number = self.count_tuning_params()
        else:
            self.tuning_param_number = number
        return True
    
    def set_fixed_tuning_param_number(self, number: Optional[int] = None):
        if self.set_tuning_param_number(number):
            self.tuning_param_number_fixed = True
            return True
        else:
            return False
    
    # Pruning model's LoRA layers based on learned masks from running fisher, salience, or L0 regularization
    def adjust_lora_with_masks(self, score_dict, current_step: int, target: str ='teacher', expand_mode: str = 'exponential_limited', refill_blocks: List[str] = []):
        model = self.model
        refill_blocks = set(refill_blocks)
        assert target in ['teacher', 'student']
        old_tuning_param_number = self.tuning_param_number
        target_tuning_param_num = self.next_tuning_param_num if self.next_tuning_param_num is not None else self.tuning_param_number
        param_cnt = {}
        rank_cnt = {}
        rank_scores = {}
        
        # First, calculating the tuning parameter num after pruning, with expanded ranks
        post_pruning_tuning_num = 0
        for i in self.param_config:
            for k in self.param_config[i]:
                if target in self.param_config[i][k]:
                    k_attr = NAME2ATTR[k][self.model_arch.model_category]
                    parent_layer = self.get_parent_layer(i, k)
                    layer = getattr(parent_layer, k_attr)
                    if not isinstance(layer, lora.PruningLinear) or layer.r == 0:
                        continue
                    using_r = layer.r if layer.bottleneck_mask is None or 'bottleneck' in refill_blocks else (layer.bottleneck_mask > 0).sum().item()
                    using_in = layer.in_features if layer.input_mask is None or 'in' in refill_blocks else (layer.input_mask > 0).sum().item()
                    using_out = layer.out_features if layer.output_mask is None or 'out' in refill_blocks else (layer.output_mask > 0).sum().item()
                    param_cnt[(i, k)] = (using_in + using_out) * using_r
                    rank_cnt[(i, k)] = using_r
                    post_pruning_tuning_num += param_cnt[(i, k)]
                    if expand_mode.startswith('tophalf'):
                        if isinstance(score_dict[i][k]['bottleneck_mask']['s'], torch.Tensor):
                            score = score_dict[i][k]['bottleneck_mask']['s'].mean().item()
                        else:
                            score = score_dict[i][k]['bottleneck_mask']['s']
                            print('Warning: score is not a tensor, using the mean value in %d-%s' % (i, k))
                        rank_scores[(i, k)] = score

        logger.info("Current tuning parameter num: %d. Post-pruning tuning parameter num: %d. Target tuning parameter num: %d."% (old_tuning_param_number, post_pruning_tuning_num, target_tuning_param_num))
        # Secondly, generating pruning masks and target rs based on the post-pruning tuning-param number and target tuning-param number, given the specific strategy
        logger.info("Using expand mode %s with refill blocks %s" % (expand_mode, refill_blocks))
        
        if expand_mode.startswith('exponential'):
            target_iks = None
            r_expanding_rate = target_tuning_param_num / post_pruning_tuning_num
            logger.info("Expanding rate: %.2f" % r_expanding_rate)
        elif expand_mode.startswith('tophalf'):
            sorted_rank_scores = sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)
            if expand_mode.endswith('limited'):
                sorted_rank_scores = [v for v in sorted_rank_scores if rank_cnt[v[0]] < self.max_lora_r]
                logger.info("Limiting the parameter in layers with type %s maximum r to %d. Pre-filter layer number: %d; Post-filter layer number: %d" % (expand_mode, self.max_lora_r, len(rank_scores), len(sorted_rank_scores)))
            target_iks = [i for i, _ in sorted_rank_scores[:len(param_cnt) // 2]]
            target_param_sum = sum([param_cnt[i] for i in target_iks])
            if len(target_iks) > 0:
                r_expanding_rate = (target_tuning_param_num - post_pruning_tuning_num) / target_param_sum + 1
            else:
                r_expanding_rate = 1
        else:
            target_iks = None
            r_expanding_rate = None
        target_rs = {}
        pruned_dim = {}
        for i in self.param_config:
            target_rs[i] = {}
            pruned_dim[i] = {}
            for k in self.param_config[i]:
                if target in self.param_config[i][k]:
                    pruned_dim[i][k] = {}
                    k_attr = NAME2ATTR[k][self.model_arch.model_category]
                    parent_layer = self.get_parent_layer(i, k)
                    layer = getattr(parent_layer, k_attr)
                    if not isinstance(layer, lora.PruningLinear) or layer.r == 0:
                        continue
                    if 'in' in refill_blocks:
                        layer.refill_input()
                        pruned_dim[i][k]['in'] = []
                    elif layer.input_mask is None:
                        pruned_dim[i][k]['in'] = []
                    else:
                        pruned_dim[i][k]['in'] = (layer.input_mask == 0).nonzero().squeeze()
                        pruned_dim[i][k]['in'] = pruned_dim[i][k]['in'].tolist() if pruned_dim[i][k]['in'].dim() else [pruned_dim[i][k]['in'].item()]
                    if 'out' in refill_blocks:
                        layer.refill_output()
                        pruned_dim[i][k]['out'] = []
                    elif layer.output_mask is None:
                        pruned_dim[i][k]['out'] = []
                    else:
                        pruned_dim[i][k]['out'] = (layer.output_mask == 0).nonzero().squeeze()
                        pruned_dim[i][k]['out'] = pruned_dim[i][k]['out'].tolist() if pruned_dim[i][k]['out'].dim() else [pruned_dim[i][k]['out'].item()]
                    if 'bottleneck' in refill_blocks:
                        layer.refill_bottleneck()
                        pruned_dim[i][k]['bottleneck'] = []
                    elif layer.bottleneck_mask is None:
                        pruned_dim[i][k]['bottleneck'] = []
                    else:
                        pruned_dim[i][k]['bottleneck'] = (layer.bottleneck_mask == 0).nonzero().squeeze()
                        pruned_dim[i][k]['bottleneck'] = pruned_dim[i][k]['bottleneck'].tolist() if pruned_dim[i][k]['bottleneck'].dim() else [pruned_dim[i][k]['bottleneck'].item()]
                    if expand_mode.startswith('exponential'):
                        r_use = layer.r if layer.bottleneck_mask is None or 'bottleneck' in refill_blocks else (layer.bottleneck_mask > 0).sum().item()
                        target_rs[i][k] = int(r_use * r_expanding_rate)
                    elif expand_mode.startswith('uniform'):
                        target_rs[i][k] = int(layer.r * target_tuning_param_num / post_pruning_tuning_num)
                    elif expand_mode.startswith('tophalf'):
                        if (i, k) in target_iks:
                            target_rs[i][k] = int(layer.r * r_expanding_rate)
                        else:
                            target_rs[i][k] = layer.r
                    else:
                        raise ValueError("Unknown expand mode %s" % expand_mode)
                    if expand_mode.endswith('limited'):
                        if self.max_lora_r > 0 and target_rs[i][k] > self.max_lora_r:
                            logger.info("Limiting the parameter in layer %d with type %s maximum r from %d to %d" % (i, k, target_rs[i][k], self.max_lora_r))
                            target_rs[i][k] = self.max_lora_r
        
        with torch.no_grad():
            for i in self.param_config:
                for k in self.param_config[i]:
                    if target in self.param_config[i][k]:
                        k_attr = NAME2ATTR[k][self.model_arch.model_category]
                        parent_layer = self.get_parent_layer(i, k)
                        layer = getattr(parent_layer, k_attr)
                        if not isinstance(layer, lora.PruningLinear) or layer.r == 0:
                            continue   
                        pruned_bottleneck_dim = pruned_dim[i][k]['bottleneck']
                        pruned_out_dim = pruned_dim[i][k]['out']
                        pruned_in_dim = pruned_dim[i][k]['in']
                        
                        if (layer.output_mask is not None and len(pruned_out_dim) == layer.output_mask.ndim) or (layer.input_mask is not None and len(pruned_in_dim) == layer.input_mask.ndim) or (layer.bottleneck_mask is not None and len(pruned_bottleneck_dim) == layer.r):
                            # mask all equals to 0
                            shrinked_layer = layer
                            shrinked_layer.log_history(current_step, pruned_out_dim, pruned_bottleneck_dim, pruned_in_dim, 0)
                            shrinked_layer.eval()
                            shrinked_layer.lora_A, shrinked_layer.lora_B, shrinked_layer.in_transformation, shrinked_layer.out_transformation = None, None, None, None
                            shrinked_layer.input_mask, shrinked_layer.output_mask, shrinked_layer.bottleneck_mask = None, None, None
                            shrinked_layer.r = 0
                        else:
                            layer.log_history(current_step, pruned_out_dim, pruned_bottleneck_dim, pruned_in_dim, target_rs[i][k])
                            shrinked_layer = shrink_and_expand_pruning_lora(layer, target_rs[i][k], pruned_out_dim, pruned_bottleneck_dim, pruned_in_dim)
                            shrinked_layer.history = layer.history
                        setattr(parent_layer, k_attr, shrinked_layer)
        
        # Expand layer_transformation
        if model.layer_transformation is not None and r_expanding_rate is not None: # Only expand when layer_transformation is not None (distilling)
            new_transform_r = int(model.layer_transformation.r * r_expanding_rate)
            if new_transform_r > model.layer_transformation.r and new_transform_r < self.max_lora_r * 2:
                logger.info("Expanding layer_transformation from %d to %d" % (model.layer_transformation.r, new_transform_r))
                if not isinstance(model.layer_transformation, lora.PruningLinear):
                    model.layer_transformation = lora_to_prunelora(model.layer_transformation, r=8, lora_alpha=16)
                self.model.layer_transformation = shrink_and_expand_pruning_lora(self.model.layer_transformation, new_transform_r, None, None, None)
        # Reset the model training states (teacher, student, or co-learning)
        if model.is_teacher:
            self.clear_states()
            self.model_as_teacher()
            logger.info("Reset the model as teacher")
        elif model.is_student:
            self.clear_states()
            self.model_as_student()
            logger.info("Reset the model as student")
        elif model.is_distilling:
            self.clear_states()
            self.model_teacher_with_student()
            self._tuning_head()
            self._tuning_layer_transformation()
            logger.info("Reset the model as distilling")        
            
        current_tuning_param_number = self.count_tuning_params()
        logger.info("Tuning parameter number: %d" % current_tuning_param_number)
        logger.info("LoRA parameter number: %d" % sum([p.numel() for n, p in model.named_parameters() if 'lora' in n]))
        model.print_lora_info_by_layer()
        model.train()
        
        # Update the current tuning parameter number
        self.tuning_param_number = current_tuning_param_number
        logger.info("Tuning parameter number changed from %d to %d" % (old_tuning_param_number, self.tuning_param_number))
        self.next_tuning_step, self.next_tuning_param_num = self.gen_next_tuning_step()
        torch.cuda.empty_cache()
        
    def get_param_allocation(self):
        return {}