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
    model_name='roberta-base'
    task_name=sst2
    num_epochs=30
    learning_rate=2e-5
    batch_size=32
elif [ "$#" -eq 5 ]; then
    model_name=$1
    task_name=$2
    num_epochs=$3
    learning_rate=$4
    batch_size=$5
fi

lora_alpha=16
lora_r=8
student_param_tuning_config=q:0-11,v:0-11,i:0-11
suffix=''

if [ -d $model_name ]
then
    output_dir="${model_name}/finetuned/epoch${num_epochs}/bz${batch_size}/lr${learning_rate}"
else
    output_dir="output/${model_name}_${adapter_type}_${task_name}${suffix}/epoch${num_epochs}/bz${batch_size}/lora_r${lora_r}/lora_alpha${lora_alpha}"
fi

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
    --logging_steps 1000 \
    --eval_steps 5000 \
    --max_seq_length 128 \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --student_param_tuning_config ${student_param_tuning_config} \
    --warmup_ratio 0.06\
    --learning_rate ${learning_rate}\
    --weight_decay 0.1\
    --lora_alpha ${lora_alpha} \
    --lora_r ${lora_r} \
    --report_to none | tee ${output_dir}/log.txt