#!/bin/bash
#SBATCH -p gpu-rtx6k
#SBATCH -A h2lab
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node (1 in this case)
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --mem=64G                 # Memory per node (total memory)
#SBATCH --gres=gpu:1               # Number of GPUs requested
#SBATCH --time=48:00:00             # Walltime (hh:mm:ss)

if [ "$#" -eq 0 ]; then
    model_name=roberta-base
    teacher_path=textattack/roberta-base-SST-2
    task_name=sst2
    learning_rate=2e-5
    training_batch_size=32
    num_train_epochs=20
    distill_mapping_strategy=static_teacher_static_student
elif [ "$#" -eq 3 ]; then
    model_name=$1
    teacher_path=$2
    task_name=$3
    learning_rate=2e-5
    training_batch_size=32
    num_train_epochs=20
    distill_mapping_strategy=static_teacher_static_student
elif [ "$#" -eq 7 ]; then
    model_name=$1
    teacher_path=$2
    task_name=$3
    learning_rate=$4
    training_batch_size=$5
    num_train_epochs=$6
    distill_mapping_strategy=$7
fi

lora_r=8
lora_alpha=16

if [ -d $model_name ]
then
    output_dir="${model_name}/finetune_distilled/epoch${num_train_epochs}/bz${training_batch_size}/lr${learning_rate}/${distill_mapping_strategy}"
else
    output_dir="output/${model_name}_minus_${task_name}_mapping_${distill_mapping_strategy}_distill_fixedteacher/epoch${num_train_epochs}/bz${training_batch_size}/"
fi


echo $output_dir
mkdir -p $output_dir

python run_minus_seq2seq_training.py \
    --output_dir ${output_dir}\
    --task_name ${task_name} \
    --model_name_or_path ${model_name} \
    --do_train \
    --do_eval \
    --save_strategy no \
    --log_level info \
    --log_level_replica info \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --logging_steps 1000 \
    --eval_steps 5000 \
    --task_name ${task_name} \
    --max_input_length 512 \
    --max_target_length 128 \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${training_batch_size} \
    --per_device_eval_batch_size ${training_batch_size} \
    --tf32 True \
    --distillation_type self_student \
    --distill_mapping_strategy ${distill_mapping_strategy} \
    --warmup_ratio 0.06\
    --learning_rate ${learning_rate}\
    --weight_decay 0.1\
    --lora_alpha ${lora_alpha} \
    --lora_r ${lora_r} \
    --report_to none \
    --do_distill \
    --distill_start 0 \
    --distill_epoch ${num_train_epochs} \
    --teacher_path ${teacher_path} | tee ${output_dir}/log.txt