#!/bin/bash
#SBATCH -p ckpt
#SBATCH -A h2lab
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node (1 in this case)
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --mem=64G                 # Memory per node (total memory)
#SBATCH --gpus=a100:1               # Number of GPUs requested
#SBATCH --time=24:00:00             # Walltime (hh:mm:ss)

if [ "$#" -eq 0 ]; then
    model_name='bert-base-uncased'
    lora_r=8
    lora_alpha=16
    learning_rate=1e-4
    teacher_param_tuning_config=dq:0-31,dv:0-31
elif [ "$#" -eq 5 ]; then
    model_name=$1
    lora_r=$2
    lora_alpha=$3
    learning_rate=$4
    teacher_param_tuning_config=$5
fi

adapter_type=lora
task_name=alpaca_gpt4
num_epochs=2
batch_size=4
suffix=''

if [ -d $model_name ]
then
    output_dir="${model_name}/loraed/epoch${num_epochs}/bz${batch_size}/lr${learning_rate}"
else
    output_dir="output/${model_name}_${adapter_type}_${task_name}${suffix}/epoch${num_epochs}/bz${batch_size}/lora_r${lora_r}/lora_alpha${lora_alpha}"
fi

echo $output_dir
mkdir -p $output_dir

python run_llama_sft.py \
    --output_dir ${output_dir}\
    --task_name ${task_name} \
    --model_name_or_path ${model_name} \
    --bf16 True \
    --data_path 'data/sft/alpaca_data_cleaned.json' \
    --do_train \
    --do_eval \
    --save_strategy no \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --logging_steps 100 \
    --eval_steps 500 \
    --model_max_length 512 \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps 8 \
    --warmup_ratio 0.03\
    --learning_rate ${learning_rate}\
    --weight_decay 0.\
    --lr_scheduler_type cosine \
    --tf32 True \
    --apply_lora \
    --lora_alpha ${lora_alpha} \
    --lora_r ${lora_r} \
    --teacher_param_tuning_config ${teacher_param_tuning_config} \
    --report_to none | tee ${output_dir}/log.txt