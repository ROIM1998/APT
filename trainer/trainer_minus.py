import collections
import torch
import datasets
import logging
import time
import warnings
import math
import os
import sys
import json
import random
import torch.nn.functional as F
import loralib as lora
import numpy as np
from functools import partial

from copy import deepcopy
from tqdm.auto import tqdm
from args import MinusTrainingArguments
from transformers import __version__
from transformers import Trainer
from transformers.trainer import unwrap_model, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.optimization import get_scheduler
from torch.utils.data import DataLoader, Subset, IterableDataset
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers.modeling_utils import PreTrainedModel
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, Callable
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollator
from transformers.trainer_utils import EvalPrediction, TrainOutput, set_seed, get_last_checkpoint, speed_metrics, EvalLoopOutput, denumpify_detensorize
from transformers.trainer_pt_utils import nested_concat, nested_numpify, nested_truncate, IterableDatasetShard, find_batch_size, nested_detach
from transformers.file_utils import is_torch_tpu_available, WEIGHTS_NAME, CONFIG_NAME
from transformers.trainer_callback import TrainerState
from transformers.configuration_utils import PretrainedConfig
from prune import AdapterPruner, build_scorer, build_pruner
from utils.minus_utils import count_params, prune_layer, to_cpu_recursive, lora_to_prunelora
from utils.fisher_utils.efficiency.param import *
from transformers.trainer_callback import TrainerCallback

from .param_control import ParamController
TRAINER_STATE_NAME = "trainer_state.json"
MB = 1024.0 * 1024.0

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

task2evalkey = {
    'squad': 'f1',
    'squad_v2': 'f1',
    'mnli': 'eval_accuracy',
    'sst2': 'eval_accuracy',
    'qqp': 'eval_accuracy',
    'qnli': 'eval_accuracy',
    'rte': 'eval_accuracy',
    'mrpc': 'eval_accuracy',
    'cola': 'eval_matthews_correlation',
    'stsb': 'eval_pearson',
    'cnndm': 'eval_loss',
    'xsum': 'eval_loss',
    'wmt16': 'eval_loss',
    'alpaca': 'eval_loss',
    'alpaca_gpt4': 'eval_loss',
}

def get_minus_linear_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, pruning_step: int, last_epoch: int = -1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < pruning_step:
            return 1
        else:
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - pruning_step))
            )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def _get_minus_linear_cutoff_schedule_with_warmup(current_step: int, num_warmup_steps: int, num_training_steps: int, peak_ratio: float = 1.):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps)) * peak_ratio
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))) * peak_ratio

def get_minus_linear_cutoff_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1, peak_ratio: float = 1.):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_minus_linear_cutoff_schedule_with_warmup,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        peak_ratio=peak_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
 

def get_minus_linear_schedule_with_rewarmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, reset_steps: List[int], last_epoch: int = -1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    steppoints = []
    if not reset_steps[0] == 0:
        reset_steps = [0] + reset_steps
    warmup_starts = set(reset_steps)
    for step in reset_steps:
        steppoints.append(step)
        steppoints.append(step + num_warmup_steps)
    steppoints.append(num_training_steps + 1)
    
    # Determine which range an integer belongs to using binary search
    def find_range(n):
        for idx, step in enumerate(steppoints):
            if step <= n < steppoints[idx + 1]:
                if step in warmup_starts:
                    return step, steppoints[idx + 1], True # is warmup
                else:
                    return step, steppoints[idx + 1], False # is not warmup

    def lr_lambda(current_step: int):
        range_start, range_end, is_warmup = find_range(current_step)
        if is_warmup:
            return float(current_step - range_start) / float(max(1, range_end - range_start))
        else:
            return max(
                0.0, float(range_end - current_step) / float(max(1, range_end - range_start))
            )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class MinusTrainer(Trainer):
    def __init__(
            self,
            model: PreTrainedModel = None,
            args: MinusTrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            eval_examples = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List["TrainerCallback"]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            param_controller: Optional[ParamController] = None,
            teacher_model: Optional[PreTrainedModel] = None,
            seq_len: int = 128,
            output_seq_len: Optional[int] = None,
            cls_task: bool = True,
            pre_tune_head_mask: Optional[torch.Tensor] = None,
            pre_tune_intermediate_mask: Optional[torch.Tensor] = None,
            post_processing_function=None,
    ):

        print("Model is None: ", model is None)
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        # By default, the teacher config should be q:0-12,k:0-12,v:0-12, while the student config should be i:0-12
        fileHandler = logging.FileHandler("{0}/{1}.log".format(args.output_dir, 'trainer'))
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)
        teacher_config = ParamController.parse_tuning_param_str(args.teacher_param_tuning_config)
        logger.info("Teacher config: " + str(teacher_config))
        student_config = ParamController.parse_tuning_param_str(args.student_param_tuning_config)
        logger.info("Student config: " + str(student_config))
        # Only support BERT, RoBERTa, and T5 configurations for now
        self.model_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else model.config.num_layers
        self.model_encoder_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else model.config.num_layers
        self.model_decoder_layers = model.config.num_decoder_layers if hasattr(model.config, 'num_decoder_layers') else None
        if model.base_model_prefix == 'transformer':
            self.attention_head_size = model.config.d_kv
            self.t5_backbone = True
        elif 'bert' in model.base_model_prefix or model.config.model_type == 'llama':
            self.attention_head_size = model.config.hidden_size // model.config.num_attention_heads
            self.t5_backbone = False
        self.gated_ffn = 'gated' in getattr(model.config, 'feed_forward_proj', '') or model.config.model_type == 'llama'
        if args.warmup_param_tuning_config is None:
            warmup_config = {
                k: list(range(self.model_layers)) for k in teacher_config
            } if teacher_config is not None else None
        else:
            warmup_config = ParamController.parse_tuning_param_str(args.warmup_param_tuning_config)
        logger.info("Warmup config: " + str(warmup_config))

        self.pruning_dataloader = DataLoader(
            Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:args.pruning_batch_size * args.pruning_batches]),
            batch_size = args.pruning_batch_size,
            collate_fn=data_collator,
            shuffle=False,
        ) if train_dataset is not None else None
        
        if args.param_allocation_strategy == 'none':
            self.param_dynamic_allocation = False
        else:
            self.param_dynamic_allocation = True
        self.continuous_allocation = args.continuous_allocation
        self.contiuous_alloc_interval = args.continuous_alloc_interval
        self.restore_before_pruning = args.restore_before_pruning
        adapter_pruner = AdapterPruner(self.model, self.pruning_dataloader)
        logger.info("Adapter type: " + args.adapter_type)
        if param_controller is not None:
            self.param_controller = param_controller
        else:
            self.param_controller = ParamController(
                self.model,
                args=args,
                teacher_config=teacher_config,
                student_config=student_config,
                warmup_config=warmup_config,
                lora_with_bias=False,
                adapter_pruner=adapter_pruner,
                param_allocation_strategy=args.param_allocation_strategy,
                adapter_type=args.adapter_type,
                max_lora_r=args.max_lora_r,
            )
        # Post-processing function for SQuAD v2
        self.post_processing_function = post_processing_function
        self.eval_examples = eval_examples
        self.best_eval = None
        self.pruning_conducted = False
        self.distilling = False
        self.distill_finished = False
        self.distill_step_cnt = 0
        self.moving_term = None
        self.now_distill_loss = 0
        self.now_distill_ce_loss = 0
        self.teacher_tuning_params = None
        self.distill_tuning_params = []
        self.pruning_tuning_params = []
        self.current_model_ratio = 1
        self.final_tuning_params = None
        self.distillation_is_self = teacher_model is None and 'momentum' not in self.args.distillation_type # Even if it's self-momentum distillation, distillation_is_self is still set as False
        self.teacher_model = model if self.distillation_is_self else teacher_model # If using momentum distillation, the teacher model is set as None at start
        self.teacher_model_masks = None
        self.pruning_start_step = None
        self.pruning_end_step = None
        self.pruning_steps = None
        self.pre_pruning_tuning_steps = None
        self.last_pruning_step = 0
        self.mask_decaying = False
        self.num_prunings = None
        self.distill_start_step = -1
        self.distill_steps = -1
        self.layer_warmup_steps = -1
        self.teacher_distillation_learning = args.teacher_learning
        self.epoch_start_mem, self.epoch_end_mem = None, None
        self.mac_constraint_schedule = None
        self.current_mac_constraint = None
        self.current_distillation_type = None
        self.distill_mapping_strategy = self.args.distill_mapping_strategy
        self.auto_layer_conversion = True
        self.salience_to_be_collected = self.args.collect_salience
        self.collecting_salience = False
        self.save_salience = False
        self.salience_collecting_start = self.args.salience_collecting_start
        self.salience_collecting_end = self.args.salience_collecting_end
        self.salience_collected = []
        # self.block_normalize_term = {
        #     'head_mask': 12,
        #     'intermediate_mask': 768,
        #     'hidden_mask': 1,
        # }
        self.block_normalize_term = None
        logger.info("Teacher is learning: " + str(self.teacher_distillation_learning))
        if pre_tune_head_mask is not None or pre_tune_intermediate_mask is not None:
            self.masks = {'head_mask': pre_tune_head_mask, 'intermediate_mask': pre_tune_intermediate_mask}
        else:
            self.masks = None
        self.head_layer_z, self.mlp_z = None, None
        self.layer_z = torch.ones(self.model.config.num_hidden_layers).bool().to(self.args.device)
        self.distill_layer_loss_each = None
        self.distill_teacher_layer_selection = None
        self.distill_student_layer_selection = None
        if self.teacher_distillation_learning and not self.distillation_is_self:
            raise ValueError("teacher_distillation_learning can only be set to True when distillation is self")
        if teacher_model is not None:
            teacher_model.to(args.device)
            teacher_model.eval()
            teacher_model.head_mask, teacher_model.intermediate_mask, teacher_model.hidden_mask = None, None, None
        self.accumulated_teacher_loss = 0
        self.teacher_loss = 0
        self.mask_history = []
        self.pruning_history = []
        self.param_allocation_history = []
        self.salience_history = []
        self.hidden_mask_pruning_history = []
        self.beta_1 = 0.85
        self.beta_2 = 0.85
        if self.args.pre_tuning_scorer != 'none' and self.args.pre_tuning_constraint < 1:
            self.pre_tuning_pruning_scorer = build_scorer(self.args.pre_tuning_scorer, self.model, self.pruning_dataloader, param_controller=self.param_controller, state=self.state, gather_freq=1, beta_1=self.beta_1, beta_2=self.beta_2, use_uncertainty=False, block_normalize_dict=self.block_normalize_term, static=True, use_kurtosis=True)
            self.pre_tuning_pruning_pruner = build_pruner(args.pre_tuning_pruner, args, model, self.pre_tuning_pruning_scorer)
            self.starting_mac_constraint = self.args.pre_tuning_constraint
        else:
            self.pre_tuning_pruning_scorer = None
            self.pre_tuning_pruning_pruner = None
            self.starting_mac_constraint = 1
        if self.args.pruner_type != 'none':
            self.scorer = build_scorer('backward_running_hidden_states_salience', self.model, None, param_controller = self.param_controller, state=self.state, gather_freq=1, beta_1=self.beta_1, beta_2=self.beta_2, use_uncertainty=False, block_normalize_dict=self.block_normalize_term, static=False, use_kurtosis=True) # not using kurtosis for running scorer during training
        else:
            self.scorer = None
        self.pruner = build_pruner(args.pruner_type,args, model, self.scorer)
        task_name = getattr(args, 'task_name', None)
        self.eval_key = task2evalkey[task_name] if task_name in task2evalkey else 'eval_loss'
        self.bigger_is_better = self.eval_key != 'eval_loss'
        self.best_teacher_metric = 0. if self.bigger_is_better else 1e10
        
        if (args.fp16 or args.bf16) and self.sharded_ddp is None:
            if args.half_precision_backend == "auto":
                if args.device == torch.device("cpu"):
                    if args.fp16:
                        raise ValueError("Tried to use `fp16` but it is not supported on cpu")
                    else:
                        args.half_precision_backend = "cpu_amp"
                else:
                    args.half_precision_backend = "cuda_amp"

            logger.info(f"Using {args.half_precision_backend} half precision backend")

        self.do_grad_scaling = False
        if (args.fp16 or args.bf16):
            # deepspeed and SageMaker Model Parallel manage their own half precision
            if self.sharded_ddp is None:
                if args.half_precision_backend == "cuda_amp":
                    self.use_cuda_amp = True
                    self.amp_dtype = torch.float16 if args.fp16 else torch.bfloat16
                elif args.half_precision_backend == "cpu_amp":
                    self.use_cpu_amp = True
                    self.amp_dtype = torch.bfloat16
                    
        logger.info("Half precision backend: " + args.half_precision_backend)
        logger.info("Half precision dtype: " + str(getattr(self, 'amp_dtype', None)))
        
    def append_hidden_mask_history(self):
        if 'momentum' in self.args.distillation_type and self.model.hidden_mask is not None and (self.model.hidden_mask == 0).sum().item() > 0:
            self.hidden_mask_pruning_history.append(self.model.hidden_mask.detach().clone())
            
    def update_teacher(self, teacher_eval_metric):
        logger.info("Current teacher %s: %.4f. Best teacher %s: %.4f" % (self.eval_key, teacher_eval_metric, self.eval_key, self.best_teacher_metric))
        better = teacher_eval_metric > self.best_teacher_metric if self.bigger_is_better else teacher_eval_metric < self.best_teacher_metric
        # if self.current_model_ratio < 0.6:
        #     logger.info("Current model ratio %f is less than 0.6. Not saving teacher model." % self.current_model_ratio)
        if better and (self.current_mac_constraint < self.starting_mac_constraint):
            self.best_teacher_metric = teacher_eval_metric
            logger.info("Using momentum distillation. Saving teacher model...")
            original_teacher_model_dimension = self.teacher_model.config.hidden_size if self.teacher_model is not None else self.model.config.hidden_size
            self.teacher_model = None
            torch.cuda.empty_cache()
            if 'self' in self.args.distillation_type:
                # If can, pruning the model before setting it as teacher
                logger.info("Pre-pruning distilling linear count: %d; teacher parameter number %d." % (len([n for n in self.model.modules() if isinstance(n, lora.DistillLinear)]), len([p for n, p in self.model.named_parameters() if 'teacher' in n])))
                # pre_pruning_metrics = self.evaluate()
                # self.save_model(os.path.join(self.args.output_dir, "pre_pruning_model_step%d" % self.state.global_step))
                # json.dump(pre_pruning_metrics, open(os.path.join(self.args.output_dir, "pre_pruning_model_step%d" % self.state.global_step, 'eval_results.json'), 'w'), indent=4, sort_keys=True)
                really_pruned = self.prune_model()
                if really_pruned and self.scorer is not None:
                    self.scorer.reset_module_scores()
                self.teacher_model = self.model
                # Disable teacher model mask saving since it's fully binary
                # self.teacher_model_masks = {
                #     'head_mask': self.model.head_mask.detach().clone(),
                #     'intermediate_mask': self.model.intermediate_mask.detach().clone(),
                #     'hidden_mask': self.model.hidden_mask.detach().clone()
                # }
                self.param_controller.convert_to_self_momentum_distill()
                self.set_tuning_params(self.state.epoch, self.state.global_step)
                logger.info("Post-pruning distilling linear count: %d; teacher parameter number %d." % (len([n for n in self.model.modules() if isinstance(n, lora.DistillLinear)]), len([p for n, p in self.model.named_parameters() if 'teacher' in n])))
                # self.save_model(os.path.join(self.args.output_dir, "post_pruning_model_step%d" % self.state.global_step))
                # post_pruning_metrics = self.evaluate()
                # json.dump(post_pruning_metrics, open(os.path.join(self.args.output_dir, "post_pruning_model_step%d" % self.state.global_step, 'eval_results.json'), 'w'), indent=4, sort_keys=True)
            else:
                self.teacher_model = deepcopy(self.model)
                self.teacher_model.layer_transformation = None
                for p in self.teacher_model.parameters():
                    p.requires_grad = False
                    p.grad = None
            torch.cuda.empty_cache()
        elif better:
            logger.info("Using momentum distillation. Not saving teacher model because the sparsity warmup has not finished.")
        else:
            logger.info("Using momentum distillation. Not saving teacher model because current eval loss is not better than best eval loss.")
        
    def set_salience_collecting_state(self):
        if self.salience_to_be_collected:
            if self.state.global_step == self.salience_collecting_start:
                logger.info("Salience collection started...")
                self.collecting_salience = True
                # self.param_controller.set_grafting_mask(mode=True, target='teacher', requires_grad=True)
                # self.model.head_mask.requires_grad = True
                # self.model.intermediate_mask.requires_grad = True
                # self.model.hidden_mask.requires_grad = True
                # self.model.head_mask.retain_grad()
                # self.model.intermediate_mask.retain_grad()
                # self.model.hidden_mask.retain_grad()
            elif self.state.global_step == self.salience_collecting_end - 1:
                logger.info("Salience collection finished.")
                self.collecting_salience = False
                self.mask_decaying = False
                # if self.args.do_virtual_prune:
                #     self.model.virtual_prune()
                self.scorer.end()
                
            if self.state.global_step - self.last_pruning_step == self.pre_pruning_tuning_steps:
                if self.last_pruning_step == 0:
                    torch.save(self.scorer.get_salience_dict(), os.path.join(self.args.output_dir, 'first_salience.pt'))
                if self.pruner is not None:
                    self.mask_decaying = True
                    self.model.virtual_prune_restore()
                    logger.info("Mask decaying started...")
                if 'momentum' in self.args.distillation_type:
                    current_metrics = self.evaluate()
                    teacher_eval_metric = current_metrics[self.eval_key]
                    self.update_teacher(teacher_eval_metric)

                
    def collect_salience(self, block_normalize: bool = False):
        if self.collecting_salience :                
            self.scorer.step()
            if self.save_salience:
                self.salience_collected.append(self.current_collected_salience)
                if self.state.global_step == self.salience_collecting_end - 1 or (self.state.global_step + 1) % 1000 == 0:
                    logger.info("Saving collected salience...")
                    torch.save(self.salience_collected, os.path.join(self.args.output_dir, 'salience_collected_%d.pt' % self.state.global_step))
                    self.salience_collected = [self.current_collected_salience]
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs, return_dict=False)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # Shorten the inputs first to reduce the memory usage
        self.shortens_inputs(inputs)
        inputs = self._prepare_inputs(inputs)
        self.set_salience_collecting_state()
        
        self.debug_output_step += 1
        with self.compute_loss_context_manager():
            if self.distilling:
                if self.current_distillation_type is None or 'inlayer' not in self.current_distillation_type:
                    teacher_loss = None
                    if self.distillation_is_self:
                        if self.current_distillation_type == 'self_teacher':
                            model.train()
                            self.param_controller.model_decouple_as_teacher()
                            teacher_outputs = model(
                                **inputs,
                                pass_mask=False,
                                use_teacher=True,
                            )
                            teacher_loss = teacher_outputs[0]
                            loss = teacher_loss
                        elif self.current_distillation_type == 'self_student':
                            model.eval() # self.teacher_model is actually the same as self.model
                            with torch.no_grad():
                                teacher_outputs = model(
                                    **inputs,
                                    output_hidden_states=True,
                                    return_dict=False,
                                    pass_mask=False,
                                    use_teacher=True,
                                )
                            model.train()
                            self.param_controller.model_decouple_as_student()
                            student_outputs = model(
                                **inputs,
                                output_hidden_states=True,
                                return_dict=False,
                            ) #! get the two outputs
                            distill_loss, distill_ce_loss, loss = self.calculate_distillation_loss(
                                teacher_outputs,
                                student_outputs,
                            )
                            self.now_distill_loss += distill_loss.detach().item() if distill_loss else 0
                            self.now_distill_ce_loss += distill_ce_loss.detach().item() if distill_ce_loss else 0
                            # loss = loss * 0.5 + student_outputs[0] * 0.
                        elif self.current_distillation_type == 'co_learning':
                            model.train()
                            combined_outputs = self.model(
                                **inputs,
                                output_hidden_states=True,
                                return_dict=False,
                                output_masked_states=True,
                                use_cross_masked_states=True,
                            )
                            teacher_outputs = (combined_outputs[0], combined_outputs[2], combined_outputs[4])
                            student_outputs = (combined_outputs[1], combined_outputs[3], combined_outputs[5])
                            teacher_loss = teacher_outputs[0]
                            teacher_outputs = nested_detach(teacher_outputs)
                            distill_loss, distill_ce_loss, loss = self.calculate_distillation_loss(
                                teacher_outputs,
                                student_outputs,
                            )
                            self.now_distill_loss += distill_loss.detach().item() if distill_loss else 0
                            self.now_distill_ce_loss += distill_ce_loss.detach().item() if distill_ce_loss else 0
                            loss = loss * 0.5 + teacher_loss * 0.5
                    else:
                        # teacher model is not self.model. Traditional teacher-fixed distillation
                        assert self.args.distillation_type == 'self_momentum' or self.teacher_model is not self.model
                        self.teacher_model.eval()
                        with torch.no_grad():
                            teacher_outputs = self.teacher_model(
                                **inputs,
                                output_hidden_states=True,
                                return_dict=False,
                                use_teacher=True,
                                head_z=self.teacher_model_masks.get('head_mask', None) if self.teacher_model_masks is not None else None,
                                intermediate_z=self.teacher_model_masks.get('intermediate_mask', None) if self.teacher_model_masks is not None else None,
                                hidden_z=self.teacher_model_masks.get('hidden_mask', None) if self.teacher_model_masks is not None else None,
                            )
                        model.train()
                        student_outputs = model(
                            **inputs,
                            output_hidden_states=True,
                            return_dict=False,
                        )
                        distill_loss, distill_ce_loss, loss = self.calculate_distillation_loss(
                            teacher_outputs,
                            student_outputs,
                        )
                        self.now_distill_loss += distill_loss.detach().item() if distill_loss else 0
                        self.now_distill_ce_loss += distill_ce_loss.detach().item() if distill_ce_loss else 0
                        if 'momentum' in self.args.distillation_type:
                            # self.moving_term = (self.state.global_step - self.distill_start_step) / self.distill_steps
                            self.moving_term = min(1., (self.state.global_step - self.pruning_start_step) / (self.pruning_end_step - self.pruning_start_step)) # moving from 0 to 1
                            # self.moving_term = max(min(self.moving_term / 2 + 0.5, 1), 0.5) # ranging from 0.5 to 1
                            # self.moving_term = 0.5
                            # Starting with distillation loss only, but linearly changing to SFT loss only
                            loss = loss * self.moving_term + student_outputs[0] * (1 - self.moving_term) # Adding labeled data loss
                            
                    self.accumulated_teacher_loss += teacher_outputs[0].item()
                    self.teacher_loss = teacher_outputs[0].item()
                    if self.args.distillation_type == 'self_interleave':
                        self.current_distillation_type = 'self_student' if self.current_distillation_type == 'self_teacher' else 'self_teacher'
                else:
                    model.train()
                    current_inlayer = 'inlayer' in self.current_distillation_type and 'inverse_inlayer' not in self.current_distillation_type
                    outputs = self.model(
                            **inputs,
                            output_hidden_states=current_inlayer,
                            return_dict=False,
                            output_masked_states=current_inlayer,
                    )
                    if current_inlayer:
                        outputs = outputs[:2] + (nested_detach(outputs[2]),) + outputs[3:]
                        distill_loss, self.distill_layer_loss_each = self.calculate_inlayer_distillation_loss(outputs)
                        self.distill_layer_loss_each = self.distill_layer_loss_each.detach().item()
                        if not self.teacher_distillation_learning:
                            loss = distill_loss
                        else:
                            loss = distill_loss * 0.5 + outputs[0] * 0.5
                            self.accumulated_teacher_loss += outputs[0].item()
                            self.teacher_loss = outputs[0].item()
                        self.now_distill_loss += distill_loss.detach().item()
                    else:
                        loss = outputs[0]
                    if self.args.distillation_type == 'inlayer_interleave':
                        self.current_distillation_type = 'inlayer' if self.current_distillation_type == 'inverse_inlayer' else 'inverse_inlayer'
                self.distill_step_cnt += 1
            else:
                model.train()
                loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()

        self.collect_salience()
        
        return loss.detach()
    
    def shortens_inputs(self, inputs):
        max_length = inputs["attention_mask"].sum(-1).max().item()
        inputs["input_ids"] = inputs["input_ids"][:, :max_length]
        inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]
        if "token_type_ids" in inputs:
            inputs["token_type_ids"] = inputs["token_type_ids"][:, :max_length]

    def reset_optimizer_and_lr_scheduler(self, reset_lr_scheduler: bool = False, distill_status: int = 2):
        self.optimizer = None
        if reset_lr_scheduler:
            self.lr_scheduler = None
            logger.info("Reset optimizer and scheduler!")
        else:
            logger.info("Reset optimizer only!")
        # TODO: might need to figure out if tuning the post-pruning model need smaller learning rate. Currently set as the same.
        if self.model.config.model_type == 'llama':
            if distill_status == 2:
                # Not relevant to distillation
                lr_steps = self.t_total - self.state.global_step
                peak_ratio = 1. if self.distilling else 1.
            elif distill_status == 1:
                # Start distillation
                lr_steps = self.distill_steps
                peak_ratio = 1.
            elif distill_status == 0:
                # End distillation
                lr_steps = self.t_total - self.state.global_step
                peak_ratio = 1
        else:
            if distill_status == 2:
                # Not relevant to distillation
                lr_steps = self.t_total - self.state.global_step
                peak_ratio = 1.
            elif distill_status == 1:
                # Start distillation
                lr_steps = self.distill_steps
                peak_ratio = 1.
            elif distill_status == 0:
                # End distillation
                lr_steps = self.t_total - self.state.global_step
                peak_ratio = 0.1 # Post distillation uses a smaller learning rate
        

        # reset the optimizer
        self.create_optimizer_and_scheduler(num_training_steps=lr_steps, peak_ratio=peak_ratio)
        
    def log_salience_history(self):
        self.mask_history.append({
            'step': self.state.global_step,
            'head_mask': self.model.backup_head_mask.detach().cpu().clone() if self.model.virtual_pruned else self.model.head_mask.detach().cpu().clone(),
            'intermediate_mask': self.model.backup_intermediate_mask.detach().cpu().clone() if self.model.virtual_pruned else self.model.intermediate_mask.detach().cpu().clone(),
            'hidden_mask': self.model.backup_hidden_mask.detach().cpu().clone() if self.model.virtual_pruned else self.model.hidden_mask.detach().cpu().clone(),
        })
        self.salience_history.append({
            'step': self.state.global_step,
            'salience': to_cpu_recursive(self.scorer.get_salience_dict()),
            'param_per_block': self.pruner.param_per_block if self.pruner is not None else None,
        })
        self.param_allocation_history.append(
            self.param_controller.get_param_allocation()
        )

    def set_tuning_params(self, epoch: int, step: int):
        if 'interleave' not in self.args.distillation_type:
            if self.model.config.apply_lora:
                if self.distilling and self.distillation_is_self:
                    if self.args.distillation_type == 'co_learning':
                        self.param_controller.model_teacher_with_student()
                    else:
                        self.param_controller.model_as_student()
                else:
                    self.param_controller.model_as_teacher()
            else:
                self.param_controller.finetune()
            if self.distilling:
                n_tuning_params, n_tuning_param_vars, params = count_params(self.model, mode='tuned', return_names=True)
                self.distill_tuning_params.append({
                    'epoch': epoch,
                    'step': step,
                    'n_params': n_tuning_params,
                    'n_vars': n_tuning_param_vars,
                    'params': params,
                })
                logger.info("Distillation tuning params: %d" % n_tuning_params)
                logger.info("Distillation tuning param vars: %d" % n_tuning_param_vars)
            else:
                n_tuning_params, n_tuning_param_vars, params = count_params(self.model, mode='tuned', return_names=True)
                self.pruning_tuning_params.append({
                    'epoch': epoch,
                    'step': step,
                    'n_params': n_tuning_params,
                    'n_vars': n_tuning_param_vars,
                    'params': params,
                })
                logger.info("Pruning tuning params: %d" % n_tuning_params)
                logger.info("Pruning tuning param vars: %d" % n_tuning_param_vars)
        elif 'self_interleave' in self.args.distillation_type:
            assert self.model.config.apply_lora
            self.param_controller.model_decouple_as_student()
            n_student_tuning_params, n_student_tuning_param_vars, student_params = count_params(self.model, mode='tuned', return_names=True)
            self.param_controller.model_decouple_as_teacher()
            n_teacher_tuning_params, n_teacher_tuning_param_vars, teacher_params = count_params(self.model, mode='tuned', return_names=True)
            self.distill_tuning_params.append({
                'epoch': epoch,
                'step': step,
                'n_student_params': n_student_tuning_params,
                'n_student_vars': n_student_tuning_param_vars,
                'n_teacher_params': n_teacher_tuning_params,
                'n_teacher_vars': n_teacher_tuning_param_vars,
                'student_params': student_params,
                'teacher_params': teacher_params,
            })
            logger.info("Student tuning params: %d, Teacher tuning params: %d" % (n_student_tuning_params, n_teacher_tuning_params))
            logger.info("Student tuning param vars: %d, Teacher tuning param vars: %d" % (n_student_tuning_param_vars, n_teacher_tuning_param_vars))
            # Temporarily tuning both the teacher and the student for adding them into the optimizer
            self.param_controller.model_teacher_with_student()
        
        if self.model.virtual_pruned:
            head_mask, intermediate_mask, hidden_mask = self.model.backup_head_mask, self.model.backup_intermediate_mask, self.model.backup_hidden_mask
        else:
            head_mask, intermediate_mask, hidden_mask = self.model.head_mask, self.model.intermediate_mask, self.model.hidden_mask
        if any([m is not None and (m == 0).any() for m in [head_mask, intermediate_mask, hidden_mask]]):
            # If there're masks set as zero, we need to calculate the current model size as if the masked params are pruned
            logger.info("Calculating pseudo model size, instead of using model.parameters().")
            current_total_params = compute_param(
                (head_mask != 0).sum().item(),
                (intermediate_mask != 0).sum().item(),
                (hidden_mask != 0).sum().item(),
                self.attention_head_size,
                self.model_layers,
                is_t5=self.t5_backbone,
                ffn_gated=self.gated_ffn,
            )
            prefix = 'compute'
        else:
            current_total_params, _ = count_params(self.model, mode='main')
            prefix = 'count'
        
        self.current_model_ratio = current_total_params / self.n_params
        logger.info(("Current %s model params: %d; " % (prefix, current_total_params)) + f'{self.current_model_ratio * 100}% of the original model.')
        if prefix == 'compute':
            logger.info(f"Current counted (teacher) model params: {count_params(self.model, mode='main')[0] / self.n_params * 100}% of the original model.")
            
        # Reset optimizer so that pruned layers can still be updated
        # Remember, only those layers that requires grad are added to the optimizer by default
        self.reset_optimizer_and_lr_scheduler()
        self.param_controller.clear_grads()
        logger.info("Layer transformation type %s" % type(self.model.layer_transformation))
        
    def prune_model(self):
        self.model.virtual_prune_restore()
        if all([m is None or (m==1).all().item() for m in [self.model.head_mask, self.model.intermediate_mask, self.model.hidden_mask]]):
            logger.info("All masks are 1 or None. No pruning is needed.")
            return False
        for k in ['head_mask', 'intermediate_mask', 'hidden_mask']:
            if getattr(self.model, k, None) is None:
                logger.info("No %s mask to prune since it's None." % k)
                return False
        if self.distilling:
            self.append_hidden_mask_history()
        retained_indices = {
            k: getattr(self.model, k).view(-1).nonzero().squeeze()
            for k in ['head_mask', 'intermediate_mask', 'hidden_mask']
        }
        self.pruning_history.append({
            'step': self.state.global_step,
            'head_mask': self.model.head_mask.detach().cpu().clone(),
            'intermediate_mask': self.model.intermediate_mask.detach().cpu().clone(),
            'hidden_mask': self.model.hidden_mask.detach().cpu().clone(),
        })
        self.model.prune_model_with_masks()
        self.head_layer_z = self.model.head_layer_z.to(self.args.device)
        self.mlp_z = self.model.mlp_z.to(self.args.device)
        self.layer_z = self.head_layer_z.bool() & self.mlp_z.bool()
        logger.info("Head layer z: %s" % self.head_layer_z.tolist())
        logger.info("MLP z: %s" % self.mlp_z.tolist())
        logger.info("Layer z: %s" % self.layer_z.tolist())
        if self.pruner is not None:
            self.pruner.set_param_per_block()
            logger.info("New pruner param_per_block %s" % self.pruner.param_per_block)
        for k in ['head_mask', 'intermediate_mask', 'hidden_mask']:
            mask = getattr(self.model, k, None)
            if mask is not None:
                mask = mask.view(-1).detach().clone() if isinstance(mask, torch.Tensor) else torch.cat([m.view(-1).detach().clone() for m in mask])
            # mask.requires_grad = True
            # mask.retain_grad()
            setattr(self.model, k, mask)
            # Update accumulated salience and uncertainty
            if self.scorer is not None:
                self.scorer.prune_salience(k, retained_indices[k], clean_to_zero=False)
        torch.cuda.empty_cache() # Empty cache to save memory after pruning
        # no need to virtual prune again since it's really pruned

        # Adjust current model's layer_transformation if teacher's hidden dimension is pruned
        if self.args.distillation_type == 'self_momentum' and len(self.hidden_mask_pruning_history):
            if getattr(self.model, "layer_transformation", None) is not None:
                for mask in self.hidden_mask_pruning_history:
                    index = torch.LongTensor(mask.squeeze().nonzero().squeeze().tolist()).to(self.args.device)
                    original_transformation_shape = self.model.layer_transformation.weight.shape
                    self.model.layer_transformation = prune_layer(self.model.layer_transformation, index, dim=0)
                logger.info("Pruned layer transformation from %s to %s" % (str(original_transformation_shape), str(self.model.layer_transformation.weight.shape)))
            self.hidden_mask_pruning_history = []
        return True
        
    def start_distilling(self, ignore_keys_for_eval: Optional[List[str]] = None):
        target = 'teacher' if self.distillation_is_self or 'momentum' in self.args.distillation_type else 'student'
        self.save_model(os.path.join(self.args.output_dir, "pre_distillation_model"))
        if self.distillation_is_self:
            pre_convert_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self.param_controller.convert_to_distill()
            # self.param_controller.set_grafting_mask(mode=True, target='teacher', requires_grad=self.args.param_allocation_strategy.startswith('running'))
            json.dump(pre_convert_metrics, open(os.path.join(self.args.output_dir, 'pre_distillation_model', 'pre_convert_eval_results.json'), 'w'), indent=4, sort_keys=True)
            # tuning_layer_transformation and tuning_head is already called in convert_to_distill
            logger.info("##########Status: Self-distillation Starts!##########")
        else:
            if self.model.config.apply_lora:
                self.param_controller._tuning_layer_transformation(target=target)
                self.param_controller._tuning_head(target=target)
            else:
                self.param_controller.finetune()
                assert self.model.layer_transformation.weight.requires_grad
            logger.info("##########Status: Distillation from another teacher model starts!##########")
        self.distilling = True
        metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
        json.dump(metrics, open(os.path.join(self.args.output_dir, 'pre_distillation_model', 'eval_results.json'), 'w'), indent=4, sort_keys=True)
        
        # Setting up the distillation type
        if self.distillation_is_self:
            if 'interleave' not in self.args.distillation_type:
                self.current_distillation_type = self.args.distillation_type
            elif self.args.distillation_type == 'inlayer_interleave':
                self.current_distillation_type = 'inlayer'
            elif self.args.distillation_type == 'self_interleave':
                self.current_distillation_type = 'self_student'
        else:
            self.current_distillation_type = 'self_student'
            
        # Setting up parameters to be tuned
        if self.distillation_is_self:
            pass
        
        self.reset_optimizer_and_lr_scheduler(reset_lr_scheduler=True, distill_status=1)
        
        
    def load_best_model(self):
        if os.path.exists(os.path.join(self.args.output_dir, "best_model")) and os.path.exists(os.path.join(self.args.output_dir, "best_model", "pytorch_model.bin")):
            best_model_path = os.path.join(self.args.output_dir, "best_model", "pytorch_model.bin")
            params = torch.load(best_model_path)
            self.model.load_state_dict(params)
            logger.info("Best model loaded.")
            # Change the dirname of best_model into best_distilled_model
            os.replace(os.path.join(self.args.output_dir, "best_model"), os.path.join(self.args.output_dir, "best_distilled_model"))
        else:
            logger.info("No best model found. Skip loading best model.")
        
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
                training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (:obj:`List[str]`, `optional`)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        # Set model as teacher, and logger.info out param tuning info before training
        
        # Calculate the total model parameters before any pruning conducted
        self.n_params, self.n_param_vars = count_params(self.model, mode='main') # Exclude LoRA layers when counting parameters for later pruning usage
        # Logging mixed-precision status
        logger.info("Use cpu_amp %s; use cuda_amp %s; mixed-precision dtype: %s" % (self.use_cpu_amp, self.use_cuda_amp, getattr(self, 'amp_dtype', None)))
        
        if self.model.config.apply_lora:
            logger.info("Using PEFT with LoRA. Disabling grad for all non-teacher-learning layers.")
            if self.auto_layer_conversion:
                self.param_controller.convert_to_pre_pruning_lora_teacher() # actually is using warmup config here, not the teacher config
                self.param_controller.model_as_teacher()
                if self.args.pruner_type == 'running_fisher' or self.args.pruner_type == 'l0_regularization':
                    # self.param_controller.set_grafting_mask(mode=True, target='teacher', requires_grad=True)
                    self.param_controller.set_fixed_tuning_param_number()
            else:
                self.param_controller.tune_lora_only()
            if self.args.do_distill:
                logger.info("Self-distillation will be conducted. Initializing layer transformation matrix to identity.")
                if getattr(self.model, 'layer_transformation', None) is None:
                    self.model.layer_transformation = lora.Linear(self.model.config.hidden_size, self.model.config.hidden_size, bias=False, r=8, lora_alpha=16, dtype=self.model.dtype).to(self.args.device)
                self.model.layer_transformation.weight.data = torch.eye(self.model.config.hidden_size, device=self.args.device).to(self.model.dtype)
                self.model.layer_transformation = lora_to_prunelora(self.model.layer_transformation, r=8, lora_alpha=16)
        else:
            logger.info('Using standard model without LoRA. All parameters are trainable')
            if self.param_controller.student_config is not None:
                logger.info("Fine-tuning existing distilled model. Converting all LoRA layers to linear layers.")
                self.param_controller.convert_to_post_distillation_ft_student()
                logger.info("Post conversion eval results: %s" % self.evaluate(ignore_keys=ignore_keys_for_eval))
            self.param_controller.finetune()
                
        # Conduct pre-tuning pruning if needed
        if self.pre_tuning_pruning_pruner is not None:
            logger.info("Conducting pre-tuning pruning...")
            logger.info("Peak memory usage before pre-tuning pruning: %f MB" % (torch.cuda.max_memory_allocated() / 1024 / 1024))
            logger.info("Memory usage before pre-tuning pruning: %f MB" % (torch.cuda.memory_allocated() / 1024 / 1024))
            self.pre_tuning_pruning_scorer.step()
            self.pre_tuning_pruning_pruner.update_mask(self.starting_mac_constraint, is_last=True)
            # Saving pre-tuning pruning mask
            if self.model.head_mask is not None:
                torch.save(self.model.head_mask, os.path.join(self.args.output_dir, 'pre_tuning_head_mask.pt'))
            if self.model.intermediate_mask is not None:
                torch.save(self.model.intermediate_mask, os.path.join(self.args.output_dir, 'pre_tuning_intermediate_mask.pt'))
            if self.model.hidden_mask is not None:
                torch.save(self.model.hidden_mask, os.path.join(self.args.output_dir, 'pre_tuning_hidden_mask.pt'))
            if getattr(self.model, 'layer_transformation', None) is not None:
                index = torch.LongTensor(self.model.hidden_mask.nonzero().squeeze().tolist()).to(self.args.device)
                self.model.layer_transformation = prune_layer(self.model.layer_transformation, index, dim=0)
            self.pre_tuning_pruning_scorer.end()
            self.prune_model()
        logger.info("Peak memory usage before training: %f MB" % (torch.cuda.max_memory_allocated() / 1024 / 1024))
        logger.info("Memory usage before training: %f MB" % (torch.cuda.memory_allocated() / 1024 / 1024))
        torch.cuda.reset_peak_memory_stats()
        logger.info("Dtype of model: %s" % self.model.dtype)
        logger.info("Dtype of model's layer transformation: %s" % (self.model.layer_transformation.weight.dtype if getattr(self.model, 'layer_transformation', None) is not None else None))
        
        # Reset tuning status
        if self.model.config.apply_lora:
            if self.auto_layer_conversion:
                self.param_controller.model_as_teacher()
            else:
                self.param_controller.tune_lora_only()
        else:
            self.param_controller.finetune()

        total_n_params, total_n_vars = count_params(self.model, mode='all')
        n_tuning_params, n_tuning_param_vars, params = count_params(self.model, mode='tuned', return_names=True)
        self.teacher_tuning_params = {
            'n_params': n_tuning_params,
            'n_vars': n_tuning_param_vars,
            'params': params,
        }
        # Setup the tuning parameter budget in the param controller
        self.param_controller.set_fixed_tuning_param_number()
        self.debug_output_step = 0
        logger.info("Total number of parameters: %d, with %d tuning and %d non-tuning." % (total_n_params, self.teacher_tuning_params['n_params'], total_n_params - self.teacher_tuning_params['n_params']))
        logger.info("Tuning variables: %d, non-tuning variables: %d" % (n_tuning_param_vars, total_n_vars - n_tuning_param_vars))
        
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args: MinusTrainingArguments = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if args.fp16_full_eval and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            logger.info(f"Loading model from {resume_from_checkpoint}).")

            if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
                config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
                checkpoint_version = config.transformers_version
                if checkpoint_version is not None and checkpoint_version != __version__:
                    logger.warn(
                        f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                        f"Transformers but your current version is {__version__}. This is not recommended and could "
                        "yield to errors or unwanted behaviors."
                    )

            if args.deepspeed:
                # will be resumed in deepspeed_init
                pass
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)

                # release memory
                del state_dict

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = len(self.train_dataset) * args.num_train_epochs
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size

        # Setting up pruning steps for lr_scheduler and training
        epochs_trained = 0

        mac_constraints = None
        epoch_length = len(train_dataloader) // args.gradient_accumulation_steps
        logger.info("Length of training dataloader being used: %d" % epoch_length)
        if args.pre_pruning_tuning_steps > 0:
            logger.info("Using pre-pruning tuning steps: " + str(args.pre_pruning_tuning_steps) + ". Ignoring pre_pruning_tuning_epochs.")
            self.pre_pruning_tuning_steps = args.pre_pruning_tuning_steps # By default is set to 200
        else:
            if args.pre_pruning_tuning_epochs > 0:
                self.pre_pruning_tuning_steps = int(args.pre_pruning_tuning_epochs * epoch_length)
                logger.info("Using pre-pruning tuning epochs: " + str(args.pre_pruning_tuning_epochs) + ". Ignoring pre_pruning_tuning_steps.")
            else:
                logger.info("No pre-pruning tuning steps or epochs specified. Starting with mask decay if pruner is not none.")
                self.pre_pruning_tuning_steps = 0 if self.pruner is not None else -1
        logger.info("Finalized pre-pruning tuning steps: " + str(self.pre_pruning_tuning_steps))
        if self.salience_collecting_start < 0:
            self.salience_collecting_start = self.pre_pruning_tuning_steps
            logger.info("Using pre-pruning tuning steps to set salience collecting start. Ignore salience_collecting_start.")
        if args.pruning_start != -1:
            self.pruning_start_step = int(args.pruning_start * epoch_length)
        else:
            self.pruning_start_step = self.pre_pruning_tuning_steps
        total_training_steps = (num_train_epochs - epochs_trained) * epoch_length
        self.pruning_start_step = min(
            total_training_steps,
            max(
                args.pruning_batches + 1,
                self.pruning_start_step,
            )
        )

        self.pruning_end_step = int(args.pruning_stop * epoch_length)
        self.pruning_end_step = min(self.pruning_end_step, total_training_steps)
        if args.pruning_frequency != -1:
            logger.info("Using pruning frequency to set the schedule. Ignore num_prunings.")
            num_prunings = int((self.pruning_end_step - self.pruning_start_step) / (args.pruning_frequency * epoch_length))
        else:
            num_prunings = args.num_prunings
        sparsity_warmup_steps = int(args.sparsity_warmup_epochs * epoch_length) if args.sparsity_warmup_epochs != -1 else -1
        self.layer_warmup_steps = int(args.pre_pruning_layer_warmup_epochs * epoch_length) if args.pre_pruning_layer_warmup_epochs != -1 else -1
        
        if args.pruning_scheduler == 'none':
            self.pruning_steps = []
        elif args.pruning_scheduler == 'once':
            self.pruning_steps, mac_constraints = self.param_controller.generate_pruning_schedule(
                self.pruning_start_step,
                self.pruning_end_step,
                self.starting_mac_constraint,
                args.mac_constraint,
                num_prunes=1,
                scheduler_reduction_type='linear',
                scheduler_frequency_type='once',
                warmup_steps=sparsity_warmup_steps,
            )
        elif args.pruning_scheduler == 'linear_gradual':
            self.pruning_steps, mac_constraints = self.param_controller.generate_pruning_schedule(
                self.pruning_start_step,
                self.pruning_end_step,
                self.starting_mac_constraint,
                args.mac_constraint,
                num_prunes = num_prunings,
                scheduler_reduction_type='linear',
                scheduler_frequency_type='linear',
                warmup_steps=sparsity_warmup_steps,
            )
        elif args.pruning_scheduler == 'cubic_gradual':
            self.pruning_steps, mac_constraints = self.param_controller.generate_pruning_schedule(
                self.pruning_start_step,
                self.pruning_end_step,
                self.starting_mac_constraint,
                args.mac_constraint,
                num_prunes = num_prunings,
                scheduler_reduction_type='cubic',
                scheduler_frequency_type='linear',
                warmup_steps=sparsity_warmup_steps,
            )
        # Set tuning schedule
        if args.pruning_scheduler != 'none' and self.model.config.apply_lora:
            self.param_controller.generate_tuning_schedule(
                self.pruning_start_step,
                self.pruning_end_step,
                n_tuning_params * args.tuning_expanding_ratio,
                num_tunings=1 if args.pruning_scheduler == 'once' else num_prunings,
                scheduler_increase_type='linear',
                scheduler_frequency_type='linear',
                warmup_steps=-1,
            )
            logger.info("Next tuning adjustment step %d, with tuning parameter number %d, with followed up pruning step %s" % (self.param_controller.next_tuning_step, self.param_controller.next_tuning_param_num, self.param_controller.tuning_schedule))
        if mac_constraints is not None:
            self.mac_constraint_schedule = mac_constraints[1:]
        if not all([isinstance(v, int) for v in self.pruning_steps]):
            logger.info("Warning: several steps are not integer at start, and they are automatically converted.")
            self.pruning_steps = [int(v) for v in self.pruning_steps]
        if len(self.pruning_steps) > 0:
            if self.salience_collecting_end == -1:
                self.salience_collecting_end = self.pruning_steps[-1]
                logger.info("Salience collecting end is set to %d, which is the same as the last pruning step" % self.salience_collecting_end)
            if args.restore_before_pruning:
                self.pruning_steps = self.pruning_steps[:-1]
            if args.pruner_type == 'running_fisher':
                self.pruning_steps = self.pruning_steps[1:]
            print(self.pruning_steps, [round(v / epoch_length, 2) for v in self.pruning_steps], len(self.pruning_steps), args.mac_constraint)
            logger.info("Pruning will be conducted at steps %s, and corresonding to epochs %s. %d prunes to be conducted in total. Using final mac constraint as %f" % (self.pruning_steps, [round(v / epoch_length, 2) for v in self.pruning_steps], len(self.pruning_steps), args.mac_constraint))
            logger.info("Pruning mac constraint schedule %s." % self.mac_constraint_schedule)
        else:
            self.pruning_conducted = True
            logger.info("No pruning conducted.")
        self.num_prunings = len(self.pruning_steps)
        self.pruning_steps.reverse()
        # Setup pruning steps that needs rewarmup
        pruning_rewarmup_steps = [v for v in self.pruning_steps]
        if not args.do_distill and args.pruning_scheduler != 'once' and not args.restore_before_pruning:
            pruning_rewarmup_steps = pruning_rewarmup_steps[:-1]
        logger.info("Pruning re-warmup steps: %s" % pruning_rewarmup_steps)
        
        if len(self.pruning_steps) > 0:
            next_pruning_steps = self.pruning_steps.pop()
        else:
            next_pruning_steps = None
        if args.do_distill:
            if args.distill_start >= 0:
                self.distill_start_step = int(args.distill_start * epoch_length)
            else:
                if 'momentum' in self.args.distillation_type:
                    for prune_step, prune_mac in zip(self.pruning_steps[::-1], self.mac_constraint_schedule):
                        if prune_mac < self.starting_mac_constraint:
                            break
                    logger.info("Using momentum distillation. Must set the distillation starting step the same as the first real pruning step.")
                    self.distill_start_step = prune_step + 1
                else:
                    logger.info("Using self-distillation or fixed distillation. The distillation will be started after first real pruning.")
                    self.distill_start_step = self.pruning_start_step
        
        self.distill_steps = int(args.distill_epoch * epoch_length)
        logger.info("###### Distillation schedule: starting from step %d, ending in step %d. Corresponding to %f and %f epoch." % (self.distill_start_step, self.distill_start_step + self.distill_steps, self.distill_start_step / epoch_length, (self.distill_start_step + self.distill_steps) / epoch_length))
        if (self.restore_before_pruning or args.pruner_type == 'running_fisher') and self.mac_constraint_schedule is not None and len(self.mac_constraint_schedule) > 0:
            self.current_mac_constraint = self.mac_constraint_schedule.pop(0)
        else:
            self.current_mac_constraint = args.mac_constraint

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self.t_total = max_steps

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        num_examples = (
            self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break
                
        if self.salience_to_be_collected:
            if self.model.head_mask is not None:
                self.model.head_mask = self.model.head_mask.view(-1)
            if self.model.intermediate_mask is not None:
                self.model.intermediate_mask = self.model.intermediate_mask.view(-1)
        
        logger.info("Pruning target constraint before entering training: %f" % self.current_mac_constraint)
        if getattr(self.model, 'layer_transformation', None) is not None:
            logger.info("Layer transformation weight before entering training: %s" % self.model.layer_transformation.weight)
            logger.info("LoRA A and B weight nans: %d, %d" % (torch.isnan(self.model.layer_transformation.lora_A).sum(), torch.isnan(self.model.layer_transformation.lora_B).sum()))
            logger.info("Maximum values in LoRA A and B: %f, %f" % (torch.max(self.model.layer_transformation.lora_A), torch.max(self.model.layer_transformation.lora_B)))

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = train_dataloader
            
            torch.cuda.empty_cache()
            self.epoch_start_mem = torch.cuda.memory_allocated() / MB
            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):
                if epoch == epochs_trained and step == 0:
                    logger.info("Input example: %s" % inputs)
                    logger.info("Input shapes: %s" % {k: v.shape for k, v in inputs.items()})
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None
                
                # Setting distillation or not
                if self.args.do_distill and (not self.distilling) and self.state.global_step == self.distill_start_step:
                    self.start_distilling(ignore_keys_for_eval=ignore_keys_for_eval)
                    self.set_tuning_params(epoch, step)
                
                # Converting warmup config to teacher config for tuning layers if needed
                if self.state.global_step == self.layer_warmup_steps and self.param_controller.teacher_config != self.param_controller.warmup_config:
                    self.param_controller.convert_to_pruning_lora_teacher()
                    self.param_controller.model_as_teacher()
                    # if self.args.pruner_type == 'running_fisher' or self.args.pruner_type == 'l0_regularization':
                    #     self.param_controller.set_grafting_mask(mode=True, target='teacher', requires_grad=True)
                    self.reset_optimizer_and_lr_scheduler(reset_lr_scheduler=True) # Reset optimizer to add ffn parameters, while resetting the scheduler because there are new-initialized parameters

                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                tr_loss_step = self.training_step(model, inputs)
                
                # If using running method, update the mask after each training step
                if self.mask_decaying and self.args.pruner_type == 'running_fisher':
                    if not self.collecting_salience:
                        logger.warning("Salience collection is not started. Cannot update mask.")
                    elif self.current_mac_constraint < self.starting_mac_constraint:
                        # self.pruner.update_mask(top_k=self.current_mac_constraint, mask_lr=self.args.mask_lr, is_last=next_pruning_steps is not None and self.state.global_step == next_pruning_steps - 2 and len(self.pruning_steps) == 0)
                        self.pruner.update_mask(top_k=self.current_mac_constraint, mask_lr=self.args.mask_lr, is_last=next_pruning_steps is not None and self.state.global_step == next_pruning_steps - 2) #TODO: probably lead to unstable training if not using the last step

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.args.max_grad_norm)
                
                # Optimizer step
                optimizer_was_run = True
                self.optimizer.step()
                if optimizer_was_run and not self.deepspeed:
                    self.lr_scheduler.step()

                model.zero_grad()
                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / steps_in_epoch
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                self.step_end_mem = torch.cuda.max_memory_allocated() / MB
                self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval, start_time)

                if next_pruning_steps is not None and self.state.global_step == next_pruning_steps:
                    # If using running fisher or l0_regularization for pruning, continuously shrinking the model after each epoch without setting probing or not
                    self.log_salience_history()
                    if self.args.pruner_type == 'running_fisher':
                        # self.save_model(os.path.join(args.output_dir, "pre_virtual_pruning_model_step%d" % self.state.global_step))
                        # pre_pruning_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
                        # json.dump(pre_pruning_metrics, open(os.path.join(self.args.output_dir, "pre_virtual_pruning_model_step%d" % self.state.global_step, 'eval_results.json'), 'w'), indent=4, sort_keys=True)
                        logger.info("Pruning probed based on current running fisher score. Using mac constraint as %f" % (self.current_mac_constraint))
                        if 'self' not in self.args.distillation_type:
                            self.prune_model()
                        elif self.args.do_virtual_prune and (len(self.pruning_steps) == 0):
                            # Prune scorer indices because of virtual prune
                            prefix = 'backup_' if self.model.virtual_pruned else ''
                            mask_attrs = ['head_mask', 'intermediate_mask', 'hidden_mask']
                            retained_indices = {
                                k: getattr(self.model, prefix + k).view(-1).nonzero().squeeze() if not getattr(self.model, prefix + k).view(-1).all() else None
                                for k in mask_attrs
                            }
                            if self.scorer is not None:
                                for k in mask_attrs:
                                    # Update accumulated salience and uncertainty
                                    if retained_indices[k] is not None: # only set retained indices if there are 0s
                                        self.scorer.set_retained_indices(k, retained_indices[k], clean_to_zero=False)
                        # Set student layer masks even though pruning might not be really conducted
                        self.model.update_layer_z()
                        self.head_layer_z = self.model.head_layer_z.to(self.args.device)
                        self.mlp_z = self.model.mlp_z.to(self.args.device)
                        self.layer_z = self.head_layer_z.bool() & self.mlp_z.bool()
                        logger.info("Head layer z: %s" % self.head_layer_z.tolist())
                        logger.info("MLP z: %s" % self.mlp_z.tolist())
                        logger.info("Layer z: %s" % self.layer_z.tolist())
                        self.scorer.end()
                        if self.args.param_allocation_strategy == 'running_fisher':
                            logger.info("Adjusting lora with masks")
                            self.param_controller.adjust_lora_with_masks(self.scorer.get_module_score(), self.state.global_step, target='teacher', expand_mode=self.args.param_resizing_strategy, refill_blocks=self.args.refill_blocks)
                            torch.cuda.empty_cache() # clear cache after adjusting lora
                        self.scorer.reset_module_scores()
                        if self.pruner is not None:
                            self.pruner.set_param_per_block()
                            logger.info("New pruner param_per_block %s" % self.pruner.param_per_block)
                        logger.info("Remove mask decaying...")
                        self.mask_decaying = False # Reset mask decaying
                        if self.args.do_virtual_prune and (len(self.pruning_steps) == 0):
                            self.model.virtual_prune() # Virtually pruning the model to promote efficiency
                        # post_pruning_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
                        # self.save_model(os.path.join(args.output_dir, "post_virtual_pruning_model_step%d" % self.state.global_step))
                        # json.dump(post_pruning_metrics, open(os.path.join(self.args.output_dir, "post_virtual_pruning_model_step%d" % self.state.global_step, 'eval_results.json'), 'w'), indent=4, sort_keys=True)
                        if len(self.pruning_steps) > 0:
                            self.last_pruning_step = self.state.global_step
                            # More prunings to be done in the future, update pruning sparsity/constraint
                            self.current_mac_constraint = self.mac_constraint_schedule.pop(0)
                            # if self.args.param_allocation_strategy == 'running_fisher':
                            #     self.param_controller.set_grafting_mask(mode=True, target='teacher', requires_grad=True)
                            self.pruner.mask_updated_once = False # Reset mask_updated_once for next pruning
                                
                    elif self.args.pruner_type == 'l0_regularization':
                        pass                
                    
                    # Setting tuning schedule
                    self.set_tuning_params(epoch, step)
                    
                    if len(self.pruning_steps) > 0:
                        next_pruning_steps = self.pruning_steps.pop()
                        if self.args.pruner_type.startswith('running') or self.args.pruner_type.startswith('l0'):
                            # Reset pre-pruning fine-tuning step to make sure the mask will be decayed
                            step_before_next_prune = next_pruning_steps - self.state.global_step
                            if step_before_next_prune < 200:
                                raise ValueError("The next pruning step is too close to the current step, which will cause the mask cannot be successfully decayed to 0s!")
                            # if step_before_next_prune - self.pre_pruning_tuning_steps < 200:
                            #     self.pre_pruning_tuning_steps = step_before_next_prune // 2
                            #     if step_before_next_prune - self.pre_pruning_tuning_steps < 200:
                            #         self.pre_pruning_tuning_steps = step_before_next_prune - 200
                            #     logger.info("Changing pre-pruning fine-tuning steps to %d to make sure valid mask decay!" % self.pre_pruning_tuning_steps)
                            self.pre_pruning_tuning_steps = step_before_next_prune - 200
                            logger.info("Setting pre-pruning fine-tuning steps to %d to make sure valid mask decay!" % self.pre_pruning_tuning_steps)
                            # self.pre_pruning_tuning_steps = 1
                    else:
                        self.pruning_conducted = True
                        self.save_model(os.path.join(args.output_dir, "final_model"))
                    
                if self.distilling and self.distill_step_cnt >= self.distill_steps:
                    logger.info("##########Status: Distillation finished ##########")
                    self.distilling = False
                    self.distill_finished = True
                    post_distillation_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
                    # Because of self-momentum-distillation, loading the best model's weights before pruning
                    self.load_best_model()
                    self.prune_model()
                    if self.scorer is not None:
                        self.scorer.reset_module_scores()
                    post_loading_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
                    self.save_model(os.path.join(args.output_dir, "post_distillation_model"))
                    json.dump(post_distillation_metrics, open(os.path.join(self.args.output_dir, "post_distillation_model", 'eval_results.json'), 'w'), indent=4, sort_keys=True)
                    json.dump(post_loading_metrics, open(os.path.join(self.args.output_dir, "post_distillation_model", 'best_eval_results.json'), 'w'), indent=4, sort_keys=True)
                    if self.model.config.apply_lora and self.distillation_is_self:
                        if self.param_dynamic_allocation:
                            self.param_controller.restore_dims(target='student')
                        self.param_controller.convert_to_post_distillation_lora_student()
                        self.param_controller.model_as_student()
                    elif self.model.config.apply_lora:
                        # Distillating from another teacher model
                        if self.param_dynamic_allocation and self.args.param_allocation_strategy != 'running_fisher':
                            self.param_controller.restore_dims(target='teacher')
                        self.param_controller.model_as_teacher()
                    else:
                        self.param_controller.finetune()
                    
                    # Reset optimizer together with lr_scheduler, since the learning objective is changed
                    self.reset_optimizer_and_lr_scheduler(reset_lr_scheduler=True, distill_status=0)
                    self.param_controller.clear_grads()
                    n_tuning_params, n_tuning_param_vars, params = count_params(self.model, mode='tuned', return_names=True)
                    self.final_tuning_params = {
                        'n_params': n_tuning_params,
                        'n_vars': n_tuning_param_vars,
                        'params': params,
                    }
                    logger.info("Post-distillation tuning params: %d, number of tuning variables %d" % (n_tuning_params, n_tuning_param_vars))

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            # Epoch ends here

            self.epoch_end_mem = torch.cuda.memory_allocated() / MB
            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval, start_time)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        # Aggregating the final model masks
        if len(self.pruning_history) > 0:
            final_head_mask, final_intermediate_mask, final_hidden_mask = torch.ones_like(self.pruning_history[0]['head_mask']), torch.ones_like(self.pruning_history[0]['intermediate_mask']), torch.ones_like(self.pruning_history[0]['hidden_mask'])
            head_indices, intermediate_indices, hidden_indices = torch.arange(final_head_mask.numel()), torch.arange(final_intermediate_mask.numel()), torch.arange(final_hidden_mask.numel())
            for mask in self.pruning_history:
                pruned_heads, pruned_intermediates, pruned_hidden = (mask['head_mask'] == 0).nonzero().squeeze(), (mask['intermediate_mask'] == 0).nonzero().squeeze(), (mask['hidden_mask'] == 0).nonzero().squeeze()
                final_head_mask[head_indices[pruned_heads]] = 0
                final_intermediate_mask[intermediate_indices[pruned_intermediates]] = 0
                final_hidden_mask[hidden_indices[pruned_hidden]] = 0
                retained_heads, retained_intermediates, retained_neurons = mask['head_mask'].nonzero().squeeze(), mask['intermediate_mask'].nonzero().squeeze(), mask['hidden_mask'].nonzero().squeeze()
                head_indices, intermediate_indices, hidden_indices = head_indices[retained_heads], intermediate_indices[retained_intermediates], hidden_indices[retained_neurons]
            
            torch.save(final_head_mask, os.path.join(args.output_dir, "final_head_mask.pt"))
            torch.save(final_intermediate_mask, os.path.join(args.output_dir, "final_intermediate_mask.pt"))
            torch.save(final_hidden_mask, os.path.join(args.output_dir, "final_hidden_mask.pt"))
        
        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )

            best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
            if os.path.exists(best_model_path):
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(best_model_path, map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)
            else:
                logger.warn(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)
        
        # Same the generated masks schedule at the end
        torch.save(self.mask_history, os.path.join(args.output_dir, "masks_schedule.pt"))
        torch.save(self.pruning_history, os.path.join(args.output_dir, "pruning_schedule.pt"))
        torch.save(self.salience_history, os.path.join(args.output_dir, "salience_history.pt"))
        torch.save(self.param_allocation_history, os.path.join(args.output_dir, "param_allocation_schedule.pt"))

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        logger.info("Saving model checkpoint to %s", output_dir)
        # Make sure the model is saved in training mode, where all LoRA layers' weights are not merged
        self.model.train()
        super().save_model(output_dir)
        if self.model.virtual_pruned:
            head_mask, intermediate_mask, hidden_mask = self.model.backup_head_mask, self.model.backup_intermediate_mask, self.model.backup_hidden_mask
        else:
            head_mask, intermediate_mask, hidden_mask = self.model.head_mask, self.model.intermediate_mask, self.model.hidden_mask
        if head_mask is not None:
            torch.save(head_mask, os.path.join(output_dir, "head_mask.pt"))
        if intermediate_mask is not None:
            torch.save(intermediate_mask, os.path.join(output_dir, "intermediate_mask.pt"))
        if hidden_mask is not None:
            torch.save(hidden_mask, os.path.join(output_dir, "hidden_mask.pt"))
        json.dump({
            'distill_tuning_params': self.distill_tuning_params,
            'teacher_tuning_params': self.teacher_tuning_params,
            'final_tuning_params': self.final_tuning_params,
            'pruning_tuning_params': self.pruning_tuning_params,
        }, open(os.path.join(output_dir, 'tuning_params.json'), 'w'))
        mask_dict = {}
        for n, m in self.model.named_modules():
            if isinstance(m, lora.PruningLinear):
                for name in ['input_mask', 'output_mask', 'bottleneck_mask']:
                    mask = getattr(m, name)
                    if mask is not None:
                        if n not in mask_dict:
                            mask_dict[n] = {}
                        mask_dict[n][name] = mask
        if mask_dict:
            torch.save(mask_dict, os.path.join(output_dir, 'grafting_masks.pt'))
        if self.scorer is not None:
            torch.save(self.scorer.get_salience_dict(), os.path.join(output_dir, 'salience.pt'))
        if self.teacher_model_masks is not None:
            torch.save(self.teacher_model_masks, os.path.join(output_dir, 'teacher_model_masks.pt'))
        else:
            logger.info("Teacher model masks are not saved because it is None!")


    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval, start_time):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss
            now_distill_loss = self.now_distill_loss.item() if isinstance(self.now_distill_loss, torch.Tensor) else self.now_distill_loss
            now_distill_ce_loss = self.now_distill_ce_loss.item() if isinstance(self.now_distill_ce_loss, torch.Tensor) else self.now_distill_ce_loss
            self.now_distill_loss -= self.now_distill_loss
            self.now_distill_ce_loss -= self.now_distill_ce_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            logs["training_time"] = time.time() - start_time
            logs['distill_loss'] = round(now_distill_loss / (self.state.global_step - self._globalstep_last_logged), 4)
            logs['distill_ce_loss'] = round(now_distill_ce_loss / (self.state.global_step - self._globalstep_last_logged), 4)
            logs['distill_layer_loss_each'] = self.distill_layer_loss_each
            logs['distill_teacher'] = self.distill_teacher_layer_selection
            logs['distill_student'] = self.distill_student_layer_selection
            logs["accumulated_teacher_loss"] = self.accumulated_teacher_loss / self.distill_step_cnt if self.distill_step_cnt > 0 and self.distilling else 0
            logs["teacher_loss"] = self.teacher_loss
            logs['start_mem'] = self.epoch_start_mem
            logs['end_mem'] = self.epoch_end_mem
            logs['step_end_mem'] = self.step_end_mem
            torch.cuda.reset_peak_memory_stats() # reset after logging memory usage
            # TODO: remove this log after testing, cuz it might slow the the training process
            logs['tuning_params'], logs['tuning_vars'] = count_params(self.model, mode='tuned', return_names=False)
            # logger.info("Tuning parameters: " + str(list(count_params(self.model, mode='tuned', return_names=True)[-1])))
            logs['total_params'], logs['total_vars'] = count_params(self.model, mode='main', return_names=False)
            logs['moving_term'] = self.moving_term
            # json.dump([n for n, p in self.model.named_parameters() if p.requires_grad], open(os.path.join(self.args.output_dir, 'tuned_params.json'), 'w'))
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            # if (not self.mask_decaying) and len(self.pruning_steps) > 0 and 'momentum' in self.args.distillation_type:
            #     teacher_eval_metric = metrics[self.eval_key]
            #     self.update_teacher(teacher_eval_metric)
            self._report_to_hp_search(trial, epoch, metrics)
            if self.pruning_conducted and (self.best_eval is None or (metrics[self.eval_key] > self.best_eval and self.bigger_is_better) or (metrics[self.eval_key] < self.best_eval and not self.bigger_is_better)):
                self.best_eval = metrics[self.eval_key]
                self.save_model(os.path.join(self.args.output_dir, 'best_model'))
                json.dump(metrics, open(os.path.join(self.args.output_dir, 'best_model', 'eval_results.json'), 'w'), indent=4, sort_keys=True)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def create_optimizer_and_scheduler(self, num_training_steps: int, peak_ratio: float = 1.):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method (or :obj:`create_optimizer`
        and/or :obj:`create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer, peak_ratio=peak_ratio)

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None, peak_ratio: float = 1.):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            if self.args.lr_scheduler_type == 'linear':
                self.lr_scheduler = get_minus_linear_cutoff_schedule_with_warmup(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    peak_ratio=peak_ratio
                )
            else:
                return super().create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)
        return self.lr_scheduler

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # We might have removed columns from the dataset so we put them back.
        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(type=eval_dataset.format["type"], columns=list(eval_dataset.features.keys()))

        if self.post_processing_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_processing_function(
                self.eval_examples,
                eval_dataset,
                output.predictions
            )
            metrics = self.compute_metrics(eval_preds)
        else:
            metrics = {}
        
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        metrics.update(output.metrics)
        self.log(metrics)

        # if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
        #     # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
        #     xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        # Add teacher loss, preds, and labels when doing self-distillation with DistillLinear
        all_teacher_losses = []
        # all_teacher_preds = []
        # all_teacher_labels = []
        
        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size
            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None
            
            if self.teacher_model is not None and not self.distill_finished:
                inputs = self._prepare_inputs(inputs)
                with torch.no_grad():
                    outputs = self.teacher_model(
                        **inputs,
                        pass_mask=False,
                        use_teacher=True,
                    )
                # all_teacher_labels += inputs['labels'].detach().cpu().numpy().tolist()
                # all_teacher_preds += outputs[1].argmax(dim=1).detach().cpu().numpy().tolist()
                if outputs[0].ndim == 0:
                    all_teacher_losses.append(outputs[0].item())
        
        # efficiency_metrics = bench_latency(self.model, self.get_eval_dataloader())
        # json.dump(efficiency_metrics, open(os.path.join(self.args.output_dir,'efficiency.json'), 'w'))
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        
        if self.distilling and len(all_teacher_losses):
            # teacher_accuracy = (np.array(all_teacher_labels) == np.array(all_teacher_preds)).mean()
            teacher_loss = np.mean(all_teacher_losses)
            # metrics['teacher_eval_accuracy'] = teacher_accuracy
            metrics['teacher_eval_loss'] = teacher_loss
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
    
    # From CoFi: layer-distillation functions
    def calculate_layer_distillation_loss(self, teacher_outputs, student_outputs, is_decoder=False):
        mse_loss = torch.nn.MSELoss(reduction="mean")
        # TODO: fix the bug where LLaMA bfloat16 distillation returns NaN
        if self.args.do_distill: #! only do layer distill
            mlp_z = self.mlp_z
            head_layer_z = self.head_layer_z
            if len(teacher_outputs) == self.model_layers + 1:
                teacher_layer_output = teacher_outputs[1:] 
            else:
                teacher_layer_output = teacher_outputs #! hidden states, with a length of 12. Every has a shape of [32, 65, 768]
            if len(student_outputs) == self.model_layers + 1:
                student_layer_output = student_outputs[1:] 
            else:
                student_layer_output = student_outputs
                
            assert len(teacher_layer_output) == len(student_layer_output) and len(teacher_layer_output) == self.model_layers

            layer_losses = []
            layer_loss = 0
            num_hidden_layers = len(teacher_layer_output)
            sampled_layer_pair_num = 4
            teacher_layer_sample_step = num_hidden_layers // sampled_layer_pair_num
            if self.distill_mapping_strategy == 'static':
                # Static layer mapping, one-to-one mapping for all teacher layers & student layers. Only skipping the student layers with no parameter. Usually works worse than no layer-wise distillation
                for layer_num, (t_layer_o, s_layer_o) in enumerate(zip(teacher_layer_output, student_layer_output)):
                    if mlp_z is None or mlp_z[layer_num] > 0:
                        if self.model.layer_transformation is not None:
                            s_layer_o = self.model.layer_transformation(s_layer_o, in_retained_indices=self.model.retained_indices)
                        l = mse_loss(t_layer_o, s_layer_o)
                        layer_loss += l
                        layer_losses.append(l.item())
            else:
                # teacher static/dynamic control and student static/dynamic control
                if 'all_teacher' in self.distill_mapping_strategy:
                    specified_teacher_layers = list(range(num_hidden_layers))
                if 'all_student' in self.distill_mapping_strategy:
                    specified_teacher_layers = sorted(list(self.param_controller.student_tuning_layers))
                elif 'static_teacher' in self.distill_mapping_strategy:
                    specified_teacher_layers = list(range(teacher_layer_sample_step-1, num_hidden_layers, teacher_layer_sample_step))
                elif 'dynamic_random_teacher' in self.distill_mapping_strategy:
                    specified_teacher_layers = random.sample(list(range(num_hidden_layers)), sampled_layer_pair_num)
                    specified_teacher_layers = sorted(specified_teacher_layers)
                elif 'dynamic_block_teacher' in self.distill_mapping_strategy:
                    layer_blocks = [list(range(i, i+teacher_layer_sample_step)) for i in range(0, num_hidden_layers, teacher_layer_sample_step)]
                    specified_teacher_layers = [random.choice(block) for block in layer_blocks]
                else:
                    raise ValueError("Unknown distill mapping strategy %s" % self.distill_mapping_strategy)
                self.distill_teacher_layer_selection = specified_teacher_layers
                self.distill_student_layer_selection = {}
                if 'static_student' in self.distill_mapping_strategy:
                    for layer_num in specified_teacher_layers:
                        t_layer_o, s_layer_o = teacher_layer_output[layer_num], student_layer_output[layer_num]
                        if mlp_z is None or mlp_z[layer_num] > 0:
                            if self.model.layer_transformation is not None:
                                s_layer_o = self.model.layer_transformation(s_layer_o, in_retained_indices=self.model.retained_indices)
                            l = mse_loss(t_layer_o, s_layer_o)
                            layer_loss += l
                            layer_losses.append(l.item())
                            self.distill_student_layer_selection[layer_num] = [layer_num]
                        else:
                            layer_losses.append(None)
                elif 'dynamic_cofi_student' in self.distill_mapping_strategy:
                    # Using the CoFi layer selection strategy
                    l = []
                    if hasattr(self.model, 'layer_transformation') and self.model.layer_transformation is not None:
                        student_layer_output = [self.model.layer_transformation(s_layer_o, in_retained_indices=self.model.retained_indices) for s_layer_o in student_layer_output]
                    specified_teacher_layer_reps = []
                    for i in specified_teacher_layers:
                        self.distill_student_layer_selection[i] = []
                        specified_teacher_layer_reps.append(teacher_layer_output[i]) #! teacher: 4x[32,113,768]
                    device = student_layer_output[0].device
                    
                    for t_layer_o in specified_teacher_layer_reps:
                        for i, s_layer_o in enumerate(student_layer_output): #! student: 12x[32,113,768]
                            l.append(mse_loss(t_layer_o, s_layer_o))
                    layerwiseloss = torch.stack(l).reshape(
                        len(specified_teacher_layer_reps), len(student_layer_output)) #! [4,12]
                    
                    last_aligned_layer = len(student_layer_output)
                    alignment = []
                    for search_index in range(len(specified_teacher_layers)-1, -1, -1):
                        indexes = layerwiseloss[search_index].sort()[1]
                        if self.layer_z is not None:
                            if len(self.layer_z) == indexes.shape[0]:
                                layer_z = self.layer_z
                            elif is_decoder:
                                layer_z = self.layer_z[-self.model_decoder_layers:]
                            else:
                                layer_z = self.layer_z[:self.model_encoder_layers]
                            existing_layer_z = layer_z[indexes]
                            align = indexes[(
                                indexes < last_aligned_layer) & existing_layer_z]
                        else:
                            align = indexes[indexes < last_aligned_layer]
                        if len(align) > 0:
                            align = align[0]
                        else:
                            align = last_aligned_layer
                        alignment.append(align)
                        last_aligned_layer = align
                    alignment.reverse()
                    for i, align in enumerate(alignment):
                        self.distill_student_layer_selection[specified_teacher_layers[i]].append(align.item())
                    alignment = torch.tensor(alignment).to(device)

                    layerwise = torch.arange(len(specified_teacher_layers)).to(device)
                    layer_loss = layerwiseloss[layerwise, alignment]
                    layer_losses = layer_loss.detach().cpu().numpy().tolist()
                    layer_loss = layer_loss.sum() #! layerwise: teacher (specified layers) / alignment: student (min loss layers) / layerwiseloss: [4,12]
                elif 'dynamic_aware_student' in self.distill_mapping_strategy:
                    # Using the teacher-aware layer selection strategy
                    # The selected student layers is the tuning layers, according to the status in ParamController
                    # Meanwhile, all 12 layers of the teacher can be selected
                    l = []
                    specified_student_layer_idx = sorted(random.sample(list(self.param_controller.student_tuning_layers), sampled_layer_pair_num))
                    if hasattr(self.model, 'layer_transformation') and self.model.layer_transformation is not None:
                        student_layer_output = [self.model.layer_transformation(student_layer_output[i], in_retained_indices=self.model.retained_indices) for i in specified_student_layer_idx]
                    else:
                        student_layer_output = [student_layer_output[i] for i in specified_student_layer_idx]
                    specified_teacher_layer_reps = []
                    for i in specified_teacher_layers:
                        specified_teacher_layer_reps.append(teacher_layer_output[i]) #! teacher: 12x[32,113,768]
                    device = student_layer_output[0].device
                    
                    for t_layer_o in specified_teacher_layer_reps:
                        for i, s_layer_o in enumerate(student_layer_output): #! student: 4x[32,113,768]
                            l.append(mse_loss(t_layer_o, s_layer_o))
                    layerwiseloss = torch.stack(l).reshape(
                        len(student_layer_output), len(specified_teacher_layer_reps)) # [4 (student),12 (teacher)]
                    
                    last_aligned_layer = len(teacher_layer_output)
                    alignment = []
                    for search_index in range(len(specified_student_layer_idx)-1, -1, -1):
                        indexes = layerwiseloss[search_index].sort()[1]
                        align = indexes[indexes < last_aligned_layer]
                        if len(align) > 0:
                            align = align[0]
                        else:
                            align = last_aligned_layer
                        if align not in self.distill_student_layer_selection:
                            self.distill_student_layer_selection[align.item()] = []
                        alignment.append(align)
                        last_aligned_layer = align
                    alignment.reverse()
                    for i, align in enumerate(alignment):
                        self.distill_student_layer_selection[align.item()].append(specified_student_layer_idx[i])
                    alignment = torch.tensor(alignment).to(device)

                    layerwise = torch.arange(len(specified_student_layer_idx)).to(device)
                    layer_loss = layerwiseloss[layerwise, alignment]
                    layer_losses = layer_loss.detach().cpu().numpy().tolist()
                    layer_loss = layer_loss.sum() #! layerwise: teacher (specified layers) / alignment: student (min loss layers) / layerwiseloss: [4,12]
                else:
                    assert 'dynamic_student' in self.distill_mapping_strategy
                    for teacher_layer_num in specified_teacher_layers:
                        if teacher_layer_num not in self.distill_student_layer_selection:
                            self.distill_student_layer_selection[teacher_layer_num] = []
                        if self.layer_z is not None:
                            if len(self.layer_z) == num_hidden_layers:
                                layer_z = self.layer_z
                            elif is_decoder:
                                layer_z = self.layer_z[-self.model_decoder_layers:]
                            else:
                                layer_z = self.layer_z[:self.model_encoder_layers]
                        else:
                            layer_z = None
                        t_layer_o = teacher_layer_output[teacher_layer_num]
                        if layer_z is None or layer_z[teacher_layer_num] > 0:
                            student_layer_num = teacher_layer_num
                            s_layer_o = student_layer_output[student_layer_num]
                            if self.model.layer_transformation is not None:
                                s_layer_o = self.model.layer_transformation(s_layer_o, in_retained_indices=self.model.retained_indices)
                            l = mse_loss(t_layer_o, s_layer_o)
                            layer_loss += l
                            layer_losses.append(l.item())
                            self.distill_student_layer_selection[teacher_layer_num].append(student_layer_num)
                        else:
                            # Find the nearest student layer by MSE loss
                            neighbors = []
                            lower, upper = list(range(teacher_layer_num)), list(range(teacher_layer_num+1, num_hidden_layers))
                            lower.reverse()
                            for i in lower:
                                if layer_z[i] > 0:
                                    if self.model.layer_transformation is not None:
                                        s_layer_o = self.model.layer_transformation(student_layer_output[i], in_retained_indices=self.model.retained_indices)
                                    else:
                                        s_layer_o = student_layer_output[i]
                                    neighbors.append((i, mse_loss(t_layer_o, s_layer_o)))
                                    break
                            for i in upper:
                                if layer_z[i] > 0:
                                    if self.model.layer_transformation is not None:
                                        s_layer_o = self.model.layer_transformation(student_layer_output[i], in_retained_indices=self.model.retained_indices)
                                    else:
                                        s_layer_o = student_layer_output[i]
                                    neighbors.append((i, mse_loss(t_layer_o, s_layer_o)))
                                    break
                            if len(neighbors) == 0:
                                continue
                            else:
                                neighbors.sort(key=lambda x: x[1])
                                student_layer_num = neighbors[0][0]
                                layer_loss += neighbors[0][1]
                                layer_losses.append((teacher_layer_num, student_layer_num, neighbors[0][1].item()))
                                self.distill_student_layer_selection[teacher_layer_num].append(student_layer_num)
            self.distill_layer_loss_each = layer_losses
            return layer_loss
        else:
            return None
        

    def calculate_inlayer_distillation_loss(self, outputs):
        mse_loss = torch.nn.MSELoss(reduction="mean")
        if self.args.do_distill: #! only do layer distill
            teacher_layer_output = outputs[2][1:] #! hidden states, with a length of 12. Every has a shape of [32, 128, 768]
            student_layer_output = outputs[3]
            # only using static distillation from CoFi
            layer_loss = 0
            losses = []
            for layer_num, (t_layer_o, s_layer_o) in enumerate(zip(teacher_layer_output, student_layer_output)):
                # Removing layer transformation for now
                if self.model.layer_transformation is not None:
                    s_layer_o = self.model.layer_transformation(s_layer_o, in_retained_indices=self.model.retained_indices)
                if self.layer_z[layer_num]:
                    l = mse_loss(s_layer_o, t_layer_o)
                    layer_loss += l
                    losses.append(l.item())
            return layer_loss, losses
        else:
            return None, None
          

    def calculate_distillation_loss(self, teacher_outputs, student_outputs):
        if teacher_outputs[1] is not None and student_outputs[1] is not None:
            ce_distill_loss = F.kl_div(
                input=F.log_softmax(
                    student_outputs[1] / self.args.distill_temp, dim=-1), #! logits: [32,3]
                target=F.softmax(
                    teacher_outputs[1] / self.args.distill_temp, dim=-1), #! distill_temp: 2.0
                reduction="batchmean") * (self.args.distill_temp ** 2)
        else:
            ce_distill_loss = 0
            
        ce_distill_loss *= 10 # We believe this is helpful for classification models
        
        if self.distill_mapping_strategy == 'none':
            return None, ce_distill_loss, ce_distill_loss
        else:
            layer_loss = self.calculate_layer_distillation_loss(teacher_outputs[-1], student_outputs[-1])
            distill_loss = layer_loss
        
            loss = self.args.distill_ce_loss_alpha * ce_distill_loss
            if distill_loss is not None:
                loss += self.args.distill_loss_alpha * distill_loss

            return distill_loss, ce_distill_loss, loss