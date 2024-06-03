import os
import sys
import torch
import logging
import torch.nn.functional as F

from typing import Union, Dict, Any
from .trainer_minus import MinusTrainer, nested_detach

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


class MinusQATrainer(MinusTrainer):
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eval_key = 'f1'
        self.bigger_is_better = True
        self.best_teacher_metric = 0

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
                            teacher_outputs = (combined_outputs[0], combined_outputs[2], combined_outputs[4], combined_outputs[6])
                            student_outputs = (combined_outputs[1], combined_outputs[3], combined_outputs[5], combined_outputs[7])
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
                                pass_mask=False,
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
                        self.now_distill_loss += distill_loss.detach().item() if distill_loss else 0
                        if not self.teacher_distillation_learning:
                            loss = distill_loss
                        else:
                            loss = distill_loss * 0.5 + outputs[0] * 0.5
                            self.accumulated_teacher_loss += outputs[0].item()
                            self.teacher_loss = outputs[0].item()
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

    def calculate_distillation_loss(self, teacher_outputs, student_outputs):
        if teacher_outputs[1] is not None and student_outputs[1] is not None:
            start_ce_distill_loss = F.kl_div(
                input=F.log_softmax(
                    student_outputs[1] / self.args.distill_temp, dim=-1), #! logits: [32,3]
                target=F.softmax(
                    teacher_outputs[1] / self.args.distill_temp, dim=-1), #! distill_temp: 2.0
                reduction="batchmean") * (self.args.distill_temp ** 2)
            end_ce_distill_loss = F.kl_div(
                input=F.log_softmax(
                    student_outputs[2] / self.args.distill_temp, dim=-1), #! logits: [32,3]
                target=F.softmax(
                    teacher_outputs[2] / self.args.distill_temp, dim=-1), #! distill_temp: 2.0
                reduction="batchmean") * (self.args.distill_temp ** 2)
            ce_distill_loss = (start_ce_distill_loss + end_ce_distill_loss) / 2
        else:
            ce_distill_loss = 0
            
        ce_distill_loss *= 10
        
        if self.distill_mapping_strategy == 'none':
            return None, ce_distill_loss, ce_distill_loss
        else:
            layer_loss = self.calculate_layer_distillation_loss(teacher_outputs[3], student_outputs[3])
            distill_loss = layer_loss
        
            loss = self.args.distill_ce_loss_alpha * ce_distill_loss
            if distill_loss is not None:
                loss += self.args.distill_loss_alpha * distill_loss

            return distill_loss, ce_distill_loss, loss