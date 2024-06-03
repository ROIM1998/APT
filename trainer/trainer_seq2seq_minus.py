import os
import sys
import torch
import logging
import torch.nn.functional as F

from typing import Union, Dict, Any
from transformers import Seq2SeqTrainer
from .trainer_minus import MinusTrainer, nested_detach

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


class MinusSeq2SeqTrainer(MinusTrainer, Seq2SeqTrainer):
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        # keys_to_drop = ['teacher_model', 'seq_len', 'output_seq_len', 'cls_task', 'pre_tune_head_mask', 'pre_tune_intermediate_mask', 'post_processing_function', 'param_controller']
        # print("Original model is None: ", kwargs['model'] is None)
        # cleaned_kwargs = {k: v for k, v in kwargs.items() if k not in keys_to_drop}
        super().__init__(*args, **kwargs)
        # MinusTrainer().__init__(self, *args, **kwargs)
        self.has_encoder = 't5' in self.model.config.model_type or 'bart' in self.model.config.model_type
        
    def shortens_inputs(self, inputs):
        max_length = inputs["attention_mask"].sum(-1).max().item()
        inputs["input_ids"] = inputs["input_ids"][:, :max_length]
        inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]
        if "token_type_ids" in inputs:
            inputs["token_type_ids"] = inputs["token_type_ids"][:, :max_length]
        if self.has_encoder:
            # Should specifically shorten the decoder inputs
            if "decoder_attention_mask" in inputs:
                max_decoder_length = inputs["decoder_attention_mask"].sum(-1).max().item()
                inputs["decoder_attention_mask"] = inputs["decoder_attention_mask"][:, :max_decoder_length+1]
            else:
                max_decoder_length = inputs['labels'].shape[-1] - (inputs['labels'] == -100).sum(-1).min().item()
            if 'decoder_input_ids' in inputs:
                inputs["decoder_input_ids"] = inputs["decoder_input_ids"][:, :max_decoder_length+1] # +1 for shifted right                
            inputs['labels'] = inputs['labels'][:, :max_decoder_length+1]
        
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
                                use_cache=False,
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
                                    use_cache=False,
                                )
                            model.train()
                            self.param_controller.model_decouple_as_student()
                            student_outputs = model(
                                **inputs,
                                output_hidden_states=True,
                                return_dict=False,
                                use_cache=False,
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
                            teacher_outputs = self.model(
                                **inputs,
                                output_hidden_states=True,
                                return_dict=False,
                                use_teacher=True,
                                pass_mask=False,
                                use_cache=False,
                            ) # loss, logits, decoder.past_key_values, decoder.hidden_states, encoder.last_hidden_states, encoder.hidden_states
                            student_outputs = self.model(
                                **inputs,
                                output_hidden_states=True,
                                return_dict=False,
                                use_teacher=False,
                                pass_mask=True,
                                use_cache=False,
                            )
                            teacher_loss = teacher_outputs[0]
                            teacher_outputs = nested_detach(teacher_outputs)
                            distill_loss, distill_ce_loss, loss = self.calculate_distillation_loss(
                                (teacher_outputs[0], teacher_outputs[1], teacher_outputs[4], teacher_outputs[2]),
                                (student_outputs[0], student_outputs[1], student_outputs[4], student_outputs[2]),
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
                                pass_mask=False,
                                use_cache=False,
                            )
                        model.train()
                        student_outputs = model(
                            **inputs,
                            output_hidden_states=True,
                            return_dict=False,
                            use_cache=False,
                        )
                        if self.has_encoder:
                            distill_loss, distill_ce_loss, loss = self.calculate_distillation_loss(
                                (teacher_outputs[0], teacher_outputs[1], teacher_outputs[4], teacher_outputs[2]),
                                (student_outputs[0], student_outputs[1], student_outputs[4], student_outputs[2]),
                            )
                        else:
                            distill_loss, distill_ce_loss, loss = self.calculate_distillation_loss(
                                (teacher_outputs[0], teacher_outputs[1], None, teacher_outputs[-1]),
                                (student_outputs[0], student_outputs[1], None, student_outputs[-1]),
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
                    raise NotImplementedError('inlayer distillation is not implemented yet.')
                self.distill_step_cnt += 1
            else:
                model.train()
                inputs['use_cache'] = False
                loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
        self.collect_salience()
        
        return loss.detach()

    def calculate_distillation_loss(self, teacher_outputs, student_outputs):
        if teacher_outputs[1] is not None and student_outputs[1] is not None:
            ce_distill_loss = F.kl_div(
                input=F.log_softmax(
                    student_outputs[1] / self.args.distill_temp, dim=-1), #! logits: [32,3]
                target=F.softmax(
                    teacher_outputs[1] / self.args.distill_temp, dim=-1), #! distill_temp: 2.0
                reduction="batchmean") * (self.args.distill_temp ** 2)
            # To keep in line with existing distillation loss, we divide the ce loss by the sequence length
            ce_distill_loss = ce_distill_loss / teacher_outputs[1].shape[1]
        else:
            ce_distill_loss = 0
        
        if self.distill_mapping_strategy == 'none':
            return None, ce_distill_loss, ce_distill_loss
        else:
            # De-variance the hidden states for stable distillation
            decoder_teacher_states = [v * torch.rsqrt(v.pow(2).mean(dim=-1, keepdim=True) + 1e-6) for v in teacher_outputs[3]]
            decoder_student_states = [v * torch.rsqrt(v.pow(2).mean(dim=-1, keepdim=True) + 1e-6) for v in student_outputs[3]]
            decoder_layer_loss = self.calculate_layer_distillation_loss(decoder_teacher_states, decoder_student_states, is_decoder=True)
            if self.has_encoder:
                encoder_teacher_states = [v * torch.rsqrt(v.pow(2).mean(dim=-1, keepdim=True) + 1e-6) for v in  teacher_outputs[2]]
                encoder_student_states = [v * torch.rsqrt(v.pow(2).mean(dim=-1, keepdim=True) + 1e-6) for v in student_outputs[2]]
                encoder_layer_loss = self.calculate_layer_distillation_loss(encoder_teacher_states, encoder_student_states, is_decoder=False)
                distill_loss = (encoder_layer_loss + decoder_layer_loss) / 2
            else:
                encoder_layer_loss = 0
                distill_loss = decoder_layer_loss
        
            loss = self.args.distill_ce_loss_alpha * ce_distill_loss
            if distill_loss is not None:
                loss += self.args.distill_loss_alpha * distill_loss

            return distill_loss, ce_distill_loss, loss