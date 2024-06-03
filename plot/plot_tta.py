import os
import json
import seaborn as sns
sns.set_theme(style="whitegrid")
from matplotlib import pyplot as plt

dirs = {
    'elastictuning': 'output/roberta-base_lora_minus_sst2_cubic_gradual_running_fisher_alloc_running_fisher_self_momentum_mapping_dynamic_block_teacher_dynamic_student_distill_tophalf_limited_resizing_latelongdistill/mac0.4/epoch40/bz32/numprune5/paramq:0-11,v:0-11,i:0-11/lora_r8/pruning_start-1/distill_epoch20/best_model',
    'elastictuning-new': 'output/roberta-base_lora_minus_sst2_cubic_gradual_running_fisher_alloc_running_fisher_self_momentum_mapping_dynamic_block_teacher_dynamic_student_distill_tophalf_limited_resizing_longdistill_backscore-abs-sum-sum-prod_prunewarmup/mac0.4/epoch40/bz32/numprune10/paramq:0-11,v:0-11,i:0-11/lora_r8/pruning_start-1/distill_epoch20/best_model',
    'ft': 'output/roberta-base_sst2_full/epoch60/bz32',
    'lora': 'output/roberta-base_lora_sst2/epoch60/bz32/lora_r8/lora_alpha16',
    'ft-mt-retrain': 'output/roberta-base_sst2_full/epoch60/bz32/best_model/pruned/constraint_0.4/batches_64/finetuned/epoch30/bz32/lr2e-5',
    'ft-mt-redistill': 'output/roberta-base_sst2_full/epoch60/bz32/best_model/pruned/constraint_0.4/batches_64/finetune_distilled/epoch20/bz32/lr2e-5/static_teacher_static_student',
    'lora-mt-retrain': 'output/roberta-base_lora_sst2/epoch60/bz32/lora_r8/lora_alpha16/checkpoint-10525/pruned/constraint_0.4/batches_64/finetuned/epoch30/bz32/lr2e-4',
    'lora-mt-redistill': 'output/roberta-base_lora_sst2/epoch60/bz32/lora_r8/lora_alpha16/checkpoint-10525/pruned/constraint_0.4/batches_64/lora_distilled/epoch20/bz32/lr2e-4/paramq:0-11,v:0-11/lora_r8/lora_alpha16/static_teacher_static_student',
    'lora-mt-redistill-retrain': 'output/roberta-base_lora_sst2/epoch60/bz32/lora_r8/lora_alpha16/checkpoint-10525/pruned/constraint_0.4/batches_64/lora_distilled/epoch20/bz32/lr2e-4/paramq:0-11,v:0-11/lora_r8/lora_alpha16/static_teacher_static_student/best_model/loraed/epoch30/bz32/lr2e-5',
}

def read_tta(path):
    data = json.load(open(
        os.path.join(
            os.path.dirname(path) if path.endswith('best_model') else path,
            'tta_history.json',
        ),
        'r',
    ))
    xs = [x[0] for x in data]
    ys = [x[1] for x in data]
    return xs, ys

if __name__ == '__main__':
    fig = plt.figure(figsize=(10, 10))
    plt.clf()
    # FT baseline
    xs, ys = read_tta(dirs['ft'])
    plt.plot(xs, ys, label='FT', color='red', marker='o')
    # LoRA baseline
    xs, ys = read_tta(dirs['lora'])
    plt.plot(xs, ys, label='LoRA', color='blue', marker='x')
    # FT + MT + retrain
    xs, ys = read_tta(dirs['ft-mt-retrain'])
    plt.plot(xs, ys, label='FT + MT + retrain', color='orange', marker='o', linestyle='--')
    # FT + MT + redistill
    xs, ys = read_tta(dirs['ft-mt-redistill'])
    plt.plot(xs, ys, label='FT + MT + redistill', color='red', marker='o', linestyle='-.')
    # LoRA + MT + retrain
    xs, ys = read_tta(dirs['lora-mt-retrain'])
    plt.plot(xs, ys, label='LoRA + MT + retrain', color='blue', marker='x', linestyle='--')
    # LoRA + MT + redistill
    xs, ys = read_tta(dirs['lora-mt-redistill'])
    xs_further, ys_further = read_tta(dirs['lora-mt-redistill-retrain'])
    while ys_further[0] < ys[-1]:
        xs_further.pop(0)
        ys_further.pop(0)
    xs, ys = xs + xs_further, ys + ys_further
    plt.plot(xs, ys, label='LoRA + MT + redistill + retrain', color='purple', marker='x', linestyle='-.')
    # ElasticTuning
    xs, ys = read_tta(dirs['elastictuning'])
    plt.plot(xs, ys, label='ElasticTuning', color='green', marker='*', linestyle='-.')
    # ElasticTuning new
    xs, ys = read_tta(dirs['elastictuning-new'])
    plt.plot(xs, ys, label='ElasticTuning (new)', color='blue', marker='*', linestyle='-.')
    plt.ylim(0.85, 0.95)
    plt.legend()
    fig.axes[0].set_xlabel('Time (s)')
    fig.axes[0].set_ylabel('Accuracy')
    plt.title('RoBERTa-sst2 time to accuracy')
    plt.savefig('tta.png')