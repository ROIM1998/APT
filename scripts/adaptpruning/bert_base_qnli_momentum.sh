#!/bin/bash
#SBATCH -p gpu-rtx6k
#SBATCH -A h2lab
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node (1 in this case)
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --mem=32G                 # Memory per node (total memory)
#SBATCH --gres=gpu:1               # Number of GPUs requested
#SBATCH --time=300:00:00             # Walltime (hh:mm:ss)

if [ "$#" -eq 0 ]; then
    mac_constraint=0.4
    lora_r=8
    pruning_start=-1
    pruning_scheduler=cubic_gradual
    pruner_type=running_fisher
    param_allocation_strategy=running_fisher
    distillation_type=self_momentum
    distill_mapping_strategy=dynamic_block_teacher_dynamic_student
elif [ "$#" -eq 8 ]; then
    mac_constraint=$1
    lora_r=$2
    pruning_start=$3    
    pruning_scheduler=$4
    pruner_type=$5
    param_allocation_strategy=$6
    distillation_type=$7
    distill_mapping_strategy=$8
elif [ "$#" -eq 9 ]; then
    mac_constraint=$1
    lora_r=$2
    pruning_start=$3
    pruning_scheduler=$4
    pruner_type=$5
    param_allocation_strategy=$6
    distillation_type=$7
    distill_mapping_strategy=$8
    gpu_id=$9
    export CUDA_VISIBLE_DEVICES=$gpu_id
fi

model_name=bert-base-uncased
param_resizing_strategy=tophalf_limited
task_name=qnli
adapter_type=lora
pruning_start=-1
pruning_stop=3
distill_start=-1 # about 60%, between 3.4 and 3.8, but after 3.6 where the teacher is updated
distill_epoch=20
pruning_batches=64
num_prunings=10
pruning_batch_size=4
# pre_pruning_tuning_epochs=1
pre_pruning_tuning_steps=200
sparsity_warmup_epochs=1

learning_rate=2e-4
training_batch_size=32
num_train_epochs=40
warmup_param_tuning_config=q:0-11,v:0-11,i:0-11
teacher_param_tuning_config=q:0-11,v:0-11,i:0-11
student_param_tuning_config=q:0-11,v:0-11,i:0-11
lora_alpha=$(($lora_r * 2))

output_dir="output/${model_name}/${task_name}/bz${training_batch_size}/elastictuning/mac${mac_constraint}/epoch${num_train_epochs}/distill_epoch${distill_epoch}/numprune${num_prunings}/sparsity_warmup${sparsity_warmup_epochs}/pruning_start${pruning_start}/pruning_stop${pruning_stop}/lora_r${lora_r}/lora_alpha${lora_alpha}/warmup_param${warmup_param_tuning_config}/teacher_param${teacher_param_tuning_config}/distill_${distillation_type}/distill_mapping_${distill_mapping_strategy}"
echo $output_dir
mkdir -p $output_dir

python run_minus_training.py \
    --output_dir ${output_dir}\
    --task_name ${task_name} \
    --model_name_or_path ${model_name} \
    --do_train \
    --do_eval \
    --save_strategy no \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --logging_steps 200 \
    --log_level info \
    --log_level_replica info \
    --eval_steps 1000 \
    --max_seq_length 128 \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${training_batch_size} \
    --per_device_eval_batch_size ${training_batch_size} \
    --tf32 True \
    --lr_scheduler_type linear\
    --distillation_type ${distillation_type} \
    --distill_mapping_strategy ${distill_mapping_strategy} \
    --warmup_ratio 0.06\
    --learning_rate ${learning_rate}\
    --weight_decay 0.1\
    --seed 128 \
    --apply_lora \
    --lora_alpha ${lora_alpha} \
    --lora_r ${lora_r} \
    --report_to none \
    --pruning_batches ${pruning_batches} \
    --pruning_batch_size ${pruning_batch_size} \
    --mac_constraint ${mac_constraint} \
    --pruning_scheduler ${pruning_scheduler} \
    --sparsity_warmup_epochs ${sparsity_warmup_epochs} \
    --param_allocation_strategy ${param_allocation_strategy} \
    --teacher_param_tuning_config ${teacher_param_tuning_config} \
    --student_param_tuning_config ${student_param_tuning_config} \
    --head_scorer_type gradient_l1 \
    --intermediate_scorer_type gradient_l1 \
    --pruner_type ${pruner_type} \
    --do_distill \
    --do_virtual_prune \
    --distill_start ${distill_start} \
    --distill_epoch ${distill_epoch} \
    --pruning_start ${pruning_start} \
    --pruning_stop ${pruning_stop} \
    --num_prunings ${num_prunings} \
    --pruning_scheduler_strategy saliency \
    --collect_salience \
    --salience_collecting_start 200 \
    --salience_collecting_end -1 \
    --pre_pruning_tuning_steps ${pre_pruning_tuning_steps} \
    --mask_lr 0.01 \
    --grafting_top_k -1 \
    --param_resizing_strategy ${param_resizing_strategy} \
    --tuning_expanding_ratio 4.0 \
    --max_lora_r $(($lora_r * 8)) \
    | tee ${output_dir}/log.txt