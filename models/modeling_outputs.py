import torch

from dataclasses import dataclass
from typing import Optional, Tuple

from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import SequenceClassifierOutput, QuestionAnsweringModelOutput, BaseModelOutputWithPastAndCrossAttentions


@dataclass
class NewQuestionAnsweringModelOutput(QuestionAnsweringModelOutput):
    masked_loss: Optional[torch.FloatTensor] = None
    masked_start_logits: Optional[torch.FloatTensor] = None
    masked_end_logits: Optional[torch.FloatTensor] = None
    masked_states: Optional[Tuple[torch.FloatTensor]] = None
    

@dataclass
class NewBaseModelOutputWithPooling(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_layers: Optional[Tuple[torch.FloatTensor]] = None
    masked_states: Optional[Tuple[torch.FloatTensor]] = None
    masked_pooler_output: Optional[torch.FloatTensor] = None

@dataclass
class NewBaseModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_layers: Optional[Tuple[torch.FloatTensor]] = None
    masked_states: Optional[Tuple[torch.FloatTensor]] = None
    
@dataclass 
class NewSequenceClassifierOutput(SequenceClassifierOutput):
    masked_states: Optional[Tuple[torch.FloatTensor]] = None
    masked_logits: Optional[torch.FloatTensor] = None
    masked_loss: Optional[torch.FloatTensor] = None
    
class AdaPBaseModelOutputWithPastAndCrossAttentions(BaseModelOutputWithPastAndCrossAttentions):
    masked_hidden_states: torch.FloatTensor = None