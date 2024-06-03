from transformers import Seq2SeqTrainingArguments
from dataclasses import dataclass, field
from typing import Optional, List


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}       

seq2seq_task_to_keys = {
    'xsum': ('document', 'summary'),
    'cnndm': ('article', 'highlights'),
    'wmt16': ('en', 'ro'),
}

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    
    t_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the training and validation files."}
    )

    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json", "tsv"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            self.t_name = self.t_name.lower()
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."

@dataclass
class Seq2SeqDataTrainingArguments:
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(seq2seq_task_to_keys.keys())},
    )
    max_input_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total output sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    lang_pair: Optional[str] = field(
        default=None,
        metadata={"help": "The language pair of the dataset to use (via the datasets library). For example, en-ro for English-Romanian translation"},
    )
    source_lang: Optional[str] = field(
        default=None,
        metadata={"help": "Source language of the dataset to use (via the datasets library). For example, en for English-Romanian translation"},
    )
    target_lang: Optional[str] = field(
        default=None,
        metadata={"help": "Target language of the dataset to use (via the datasets library). For example, ro for English-Romanian translation"},
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    
    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in seq2seq_task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(seq2seq_task_to_keys.keys()))
        else:
            raise ValueError("Need either a Seq2Seq task from " + ",".join(seq2seq_task_to_keys.keys()))
        
@dataclass
class InstructionDataTrainingArguments:
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on"},
    )
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

@dataclass
class MMLUDataTrainingArguments:
    data_dir: str = field(
        default="data/mmlu",
        metadata={"help": "The input data dir. Should contain the .csv files (or other data files) for the task."},
    )
    openai_engine: Optional[str] = field(
        default=None,
        metadata={"help": "If specified, we will use the OpenAI API to generate the predictions."},
    )
    subjects: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Which subjects to evaluate. If not specified, all the 57 subjects will be evaluated."},
    )
    ntrain: int = field(
        default=0,
        metadata={"help": "Number of training instances to use in-context."},
    )
    use_chat_format: bool = field(
        default=False,
        metadata={"help": "Whether to use chat format or not."},
    )
    eval_batch_size: int = field(
        default=2,
        metadata={"help": "Batch size for evaluation."},
    )
    n_instances: Optional[int] = field(
        default=None,
        metadata={"help": "Number of evaluation instances to use in-context."},
    )
            
@dataclass
class MinusTrainingArguments(Seq2SeqTrainingArguments):
    adapter_type: str = field(default="lora", metadata={"help": "Adapter type for adapter training"})
    pruning_frequency: float = field(default=-1, metadata={"help": "How many epochs between each pruning"})
    num_prunings: int = field(default=5, metadata={"help": "How many times to probe or prune during training. If set to one, it means only probe once without pruning before distillation ends."})
    pruning_batch_size: int = field(default=4, metadata={"help": "Batch size for pruning"})
    pruning_batches: int = field(default=256, metadata={"help": "How many batches to use for pruning"})
    mac_constraint: float = field(default=1., metadata={"help": "MAC constraints for pruning during training"})
    pruning_scheduler: str = field(default="none", metadata={"help": "Pruning scheduler which controls mac constraint throughout training"})
    pruning_scheduler_strategy: str = field(default="random", metadata={"help": "Pruning scheduler strategy which controls mask changing path throughout training"})
    param_allocation_strategy: str = field(default="none", metadata={"help": "Parameter allocation strategy for elastic LoRA layers"})
    param_resizing_strategy: str = field(default="exponential_limited", metadata={"help": "Parameter resizing strategy for elastic LoRA layers"})
    refill_blocks: List[str] = field(default=None, metadata={"help": "Refill strategy for elastic LoRA layers"})
    continuous_allocation: bool = field(default=False, metadata={"help": "Whether to continuously allocate parameters for elastic LoRA layers"})
    continuous_alloc_interval: int = field(default=1, metadata={"help": "How many epochs between each parameter allocation"})
    restore_before_pruning: bool = field(default=False, metadata={"help": "Whether to restore model to its original shape before each pruning"})
    minus_scheduler: bool = field(default=False, metadata={"help": "When setting to True, the learning rate would be drop after pruning, while remaining constant pre-pruning"})
    head_scorer_type: str = field(default="gradient_l2", metadata={"help": "Attention head scorer type for pruning"})
    intermediate_scorer_type: str = field(default="gradient_l2", metadata={"help": "Intermediate layer scorer type for pruning"})
    pruner_type: str = field(default="none", metadata={"help": "Pruner type for pruning"})
    distillation_type: str = field(default="none", metadata={"help": "Distillation type for distillation"})
    distill_mapping_strategy: str = field(default="none", metadata={"help": "Distillation mapping strategy for distillation. If none, only prediction distribution distillation will be conducted."})
    head_mask_path: str = field(default=None, metadata={"help": "Path to head mask for fixed pruner"})
    intermediate_mask_path: str = field(default=None, metadata={"help": "Path to intermediate mask for fixed pruner"})
    hidden_mask_path: str = field(default=None, metadata={"help": "Path to hidden mask for fixed pruner"})
    pre_pruning_tuning_steps: int = field(default=-1, metadata={"help": "How many steps to tune the model before pruning"})
    pre_pruning_tuning_epochs: float = field(default=-1, metadata={"help": "How many epochs to tune the model before pruning"})
    collect_salience: bool = field(default=False, metadata={"help": "Whether to collect salience scores for each layer"})
    salience_collecting_start: float = field(default=-1, metadata={"help": "When to start collecting salience scores"})
    salience_collecting_end: float = field(default=-1, metadata={"help": "When to stop collecting salience scores. When set to -1, it is dependent on the number of pruning steps."})
    mask_lr: float = field(default=1e-2, metadata={"help": "Learning rate for mask parameters"})
    grafting_mask_lr: Optional[float] = field(default=None, metadata={"help": "Learning rate for grafting mask parameters"})
    grafting_top_k: int = field(default=0.8, metadata={"help": "Top k heads to be grafted"})
    pruning_start: float = field(default=1, metadata={"help": "When to start pruning"})
    pruning_stop: float = field(default=9, metadata={"help": "Last timestep of pruning"})
    sparsity_warmup_epochs: float = field(default=-1, metadata={"help": "How many epochs to warmup sparsity"})
    pre_pruning_layer_warmup_epochs: float = field(default=-1, metadata={"help": "How many epochs to warmup pre-pruning layers (using warmup config for tuning layers, and then teacher config). If set as -1, it will always use teacher config, and warmup config must be set as None."})
    do_distill: bool = field(default=False, metadata={"help": "Whether to conduct distillation"})
    do_virtual_prune: bool = field(default=False, metadata={"help": "Whether to conduct virtual pruning"})
    distill_start: float = field(default=-1, metadata={"help": "When to start distillation"})
    distill_epoch: float = field(default=0.5, metadata={"help": "Epochs used for distillation"})
    distill_temp: float = field(default=2./3., metadata={"help": "Distillation temperature"})
    distill_loss_alpha: float = field(default=0.9, metadata={"help": "Distillation loss weight"})
    distill_ce_loss_alpha: float = field(default=0.1, metadata={"help": "Distillation cross entrypy loss weight"})
    teacher_loss_alpha: float = field(default=0.5, metadata={"help": "Teacher loss weight"})
    teacher_param_tuning_config: Optional[str] = field(default=None, metadata={"help": "Teacher parameter tuning config"})
    student_param_tuning_config: Optional[str] = field(default=None, metadata={"help": "Student parameter tuning config"})
    warmup_param_tuning_config: Optional[str] = field(default=None, metadata={"help": "Warmup parameter tuning config"})
    teacher_path: str = field(default=None, metadata={"help": "Path to teacher model"})
    teacher_learning: bool = field(default=False, metadata={"help": "Whether the teacher modules are updating during self-distillation"})
    max_lora_r: int = field(default=64, metadata={"help": "Maximum number of lora layers for the student model"})
    tuning_expanding_ratio: float = field(default=4.0, metadata={"help": "Expanding ratio for number of parameters to be tuned"})
    pre_tuning_scorer: str = field(default="none", metadata={"help": "Pruner type for before tuning"})
    pre_tuning_pruner: str = field(default="none", metadata={"help": "Pruner type for before tuning"})
    pre_tuning_constraint: float = field(default=0.6, metadata={"help": "MAC constraint for before tuning"})
    post_tuning_scorer: str = field(default="none", metadata={"help": "Pruner type for after tuning"})
    
    def __post_init__(self):
        super().__post_init__()
        if self.refill_blocks is None:
            self.refill_blocks = []
        else:
            self.refill_blocks = self.refill_blocks.split(':')
        if self.grafting_mask_lr is None:
            self.grafting_mask_lr = self.mask_lr
            print('Grafting mask learning rate is set to be the same as mask learning rate.')
        if self.param_resizing_strategy.startswith('uniform') and self.grafting_top_k > 0 and self.grafting_mask_lr > 0:
            raise ValueError('Grafting mask learning rate or top_k should be set to 0 when using uniform parameter resizing strategy.') 
        if self.param_resizing_strategy.startswith('tophalf') and self.grafting_top_k > 0 and self.grafting_mask_lr > 0:
            raise ValueError('Grafting mask learning rate or top_k should be set to 0 when using tophalf parameter resizing strategy.') 
        if self.pre_pruning_layer_warmup_epochs == -1 and self.warmup_param_tuning_config is not None:
            raise ValueError('When pre_pruning_layer_warmup_epochs is set to -1, warmup_param_tuning_config must be set to None, because it will always use teacher config and no layer warmup is applied.')