from typing import Union, Dict, Optional
from transformers import PreTrainedModel
from torch.utils.data import DataLoader
from args import MinusTrainingArguments
from .scorer import BaseScorer, GradientScorer, PredictivityScorer, RunningMaskSalienceScorer, JointRunningMaskSalienceScorer, T5JointRunningMaskSalienceScorer, MagnitudeScorer, RunningWandaScorer, RunningSalienceScorer, RunningHiddenStatesSalienceScorer, RunningHiddenStatesMagnitudeScorer, RunningT5HiddenStatesSalienceScorer, BackwardRunningHiddenStatesSalienceScorer, BackwardT5RunningHiddenStatesSalienceScorer, BackwardLlamaRunningHiddenStatesSalienceScorer
from .pruner import RandomPruner, GreedyPruner, FisherPruner, FixedPruner, BetterFisherPruner, DensityBSMaskPruner, RandomBSMaskPruner, DensityUniformPruner, AdapterPruner, RuleMixSaliencePruner
from .scheduler import BasePruningScheduler, RandomPruningScheduler, SaliencyPruningScheduler, OncePruningScheduler, SequentialPruningScheduler

def build_scorer(score_type: str, model: PreTrainedModel, dataloader: Optional[DataLoader] = None, **kwargs) -> Union[GradientScorer, PredictivityScorer, RunningSalienceScorer, None]:
    if score_type == 'predictivity':
        print("Using predictivity scorer.")
        return PredictivityScorer(model, dataloader)
    elif score_type == 'magnitude':
        return MagnitudeScorer(model, kwargs['param_controller'], kwargs['state'])
    elif score_type == 'wanda':
        return RunningWandaScorer(model, kwargs['param_controller'], kwargs['state'], dataloader, kwargs['gather_freq'], kwargs['beta_1'], kwargs['beta_2'], kwargs['use_uncertainty'], kwargs['block_normalize_dict'])
    elif score_type == 'gradient_l1':
        return GradientScorer(model, dataloader, norm='l1')
    elif score_type == 'gradient_l2':
        return GradientScorer(model, dataloader, norm='l2')
    elif score_type == 'running_salience':
        return RunningMaskSalienceScorer(model, kwargs['param_controller'], kwargs['state'], dataloader, kwargs['gather_freq'], kwargs['beta_1'], kwargs['beta_2'], kwargs['use_uncertainty'], kwargs['block_normalize_dict'])
    elif score_type == 'running_hidden_states_salience':
        if 'bert' in model.base_model_prefix:
            return RunningHiddenStatesSalienceScorer(model, kwargs['param_controller'], kwargs['state'], dataloader, kwargs['gather_freq'], kwargs['beta_1'], kwargs['beta_2'], kwargs['use_uncertainty'], kwargs['block_normalize_dict'])
        else:
            return RunningT5HiddenStatesSalienceScorer(model, kwargs['param_controller'], kwargs['state'], dataloader, kwargs['gather_freq'], kwargs['beta_1'], kwargs['beta_2'], kwargs['use_uncertainty'], kwargs['block_normalize_dict'])
    elif score_type == 'backward_running_hidden_states_salience':
        if 'bert' in model.config.model_type:
            return BackwardRunningHiddenStatesSalienceScorer(model, kwargs['param_controller'], kwargs['state'], dataloader, kwargs['gather_freq'], kwargs['beta_1'], kwargs['beta_2'], kwargs['use_uncertainty'], kwargs['block_normalize_dict'], kwargs['static'] if 'static' in kwargs else False, use_kurtosis=kwargs['use_kurtosis'] if 'use_kurtosis' in kwargs else False)
        elif 't5' in model.config.model_type:
            return BackwardT5RunningHiddenStatesSalienceScorer(model, kwargs['param_controller'], kwargs['state'], dataloader, kwargs['gather_freq'], kwargs['beta_1'], kwargs['beta_2'], kwargs['use_uncertainty'], kwargs['block_normalize_dict'], kwargs['static'] if 'static' in kwargs else False, use_kurtosis=kwargs['use_kurtosis'] if 'use_kurtosis' in kwargs else False)
        elif 'llama' in model.config.model_type:
            return BackwardLlamaRunningHiddenStatesSalienceScorer(model, kwargs['param_controller'], kwargs['state'], dataloader, kwargs['gather_freq'], kwargs['beta_1'], kwargs['beta_2'], kwargs['use_uncertainty'], kwargs['block_normalize_dict'], kwargs['static'] if 'static' in kwargs else False, use_kurtosis=kwargs['use_kurtosis'] if 'use_kurtosis' in kwargs else False)
        else :
            raise NotImplementedError(f"Scorer type {score_type} not implemented.")
    elif score_type == 'running_hidden_states_magnitude':
        return RunningHiddenStatesMagnitudeScorer(model, kwargs['param_controller'], kwargs['state'], dataloader, kwargs['gather_freq'], kwargs['beta_1'], kwargs['beta_2'], kwargs['use_uncertainty'], kwargs['block_normalize_dict'])
    elif score_type == 'joint_running_salience':
        if 'bert' in model.base_model_prefix:
            return JointRunningMaskSalienceScorer(model, kwargs['param_controller'], kwargs['state'], dataloader, kwargs['gather_freq'], kwargs['beta_1'], kwargs['beta_2'], kwargs['use_uncertainty'], kwargs['block_normalize_dict'])
        elif model.base_model_prefix == 'transformer':
            return T5JointRunningMaskSalienceScorer(model, kwargs['param_controller'], kwargs['state'], dataloader, kwargs['gather_freq'], kwargs['beta_1'], kwargs['beta_2'], kwargs['use_uncertainty'], kwargs['block_normalize_dict'])
    elif score_type == 'none':
        return None
    else:
        raise NotImplementedError(f"Scorer type {score_type} not implemented.")


def build_pruner(pruner_type: str, args: MinusTrainingArguments, model: PreTrainedModel, scorer_dict: Union[BaseScorer, Dict[str, BaseScorer]], **kwargs) -> Union[RandomPruner, GreedyPruner, FisherPruner, FixedPruner, DensityBSMaskPruner]:
    if pruner_type == 'none':
        return None
    elif pruner_type.startswith('running'):
        return DensityBSMaskPruner(model, args, scorer_dict)
    elif pruner_type == 'rule_mix':
        return RuleMixSaliencePruner(model, args, scorer_dict, **kwargs)
    elif pruner_type == 'random':
        return RandomBSMaskPruner(model, args, scorer_dict)
    elif pruner_type == 'uniform':
        return DensityUniformPruner(model, args, scorer_dict)
    elif pruner_type == 'greedy':
        return GreedyPruner(model, ['head_mask', 'intermediate_mask'], scorer_dict)
    elif pruner_type == 'fisher':
        return FisherPruner(model, ['head_mask', 'intermediate_mask'], scorer_dict, args.seq_len, args.cls_task)
    elif pruner_type == 'fixed':
        return FixedPruner(model, ['head_mask', 'intermediate_mask', 'hidden_mask'], head_mask_path=args.head_mask_path, intermediate_mask_path=args.intermediate_mask_path, hidden_mask_path=args.hidden_mask_path)
    elif pruner_type == 'search':
        return BetterFisherPruner(model, ['head_mask', 'intermediate_mask'], scorer_dict, args.seq_len, args.cls_task, ['search'], args.output_seq_len)
    elif pruner_type == 'layerwise':
        return BetterFisherPruner(model, ['head_mask', 'intermediate_mask'], scorer_dict, args.seq_len, args.cls_task, ['search', 'better_rearrange', 'layerwise_rearrange'], args.output_seq_len)
    elif pruner_type == 'better_fisher':
        return BetterFisherPruner(model, ['head_mask', 'intermediate_mask'], scorer_dict, args.seq_len, args.cls_task, ['search', 'better_rearrange'], args.output_seq_len)
    elif pruner_type == 'global':
        return BetterFisherPruner(model, ['head_mask', 'intermediate_mask'], scorer_dict, args.seq_len, args.cls_task, ['search', 'better_rearrange', 'global'], args.output_seq_len)
    elif pruner_type == 'global_normalized':
        for k in scorer_dict:
            scorer_dict[k].normalize = True
        return BetterFisherPruner(model, ['head_mask', 'intermediate_mask'], scorer_dict, args.seq_len, args.cls_task, ['search', 'better_rearrange', 'global'], args.output_seq_len)
    elif pruner_type == 'topdown':
        return BetterFisherPruner(model, ['head_mask', 'intermediate_mask', 'hidden_mask'], scorer_dict, args.seq_len, args.cls_task, ['topdown_search', 'better_rearrange'], args.output_seq_len)
    else:
        raise NotImplementedError(f"Pruner type {pruner_type} not implemented.") 
    
def build_pruning_scheduler(args: MinusTrainingArguments, model: PreTrainedModel, head_mask, intermediate_mask, head_grads, intermediate_grads, dataloader=None, mac_constraints=None) ->BasePruningScheduler:
    if 'gradual' not in args.pruning_scheduler:
        return OncePruningScheduler(model, head_mask, intermediate_mask, head_grads, intermediate_grads)
    elif args.pruning_scheduler_strategy == 'random':
        return RandomPruningScheduler(model, head_mask, intermediate_mask, head_grads, intermediate_grads)
    elif args.pruning_scheduler_strategy == 'saliency':
        return SaliencyPruningScheduler(model, head_mask, intermediate_mask, head_grads, intermediate_grads, dataloader, mac_constraints, args.seq_len)
    elif args.pruning_scheduler_strategy == 'sequential':
        return SequentialPruningScheduler(model, head_mask, intermediate_mask, head_grads, intermediate_grads)
    else:
        raise NotImplementedError(f"Pruning scheduler {args.pruning_scheduler_strategy} not implemented.")