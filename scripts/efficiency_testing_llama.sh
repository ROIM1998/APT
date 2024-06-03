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
    id=default
    backbone_name='roberta-base'
    model_name='roberta-base'
    lora_r=8
    lora_alpha=16
    batch_size=4
elif [ "$#" -eq 2 ]; then
    id=$1
    backbone_name='roberta-base'
    model_name=$2
    lora_r=8
    lora_alpha=16
    batch_size=4
elif [ "$#" -eq 6 ]; then
    id=$1
    backbone_name=$2
    model_name=$3
    lora_r=$4
    lora_alpha=$5
    batch_size=$6
fi

task_name=alpaca_gpt4
output_dir="output/efficiency_testing/${backbone_name}/${task_name}/${id}/bz${batch_size}/"

echo $output_dir
mkdir -p $output_dir

python efficiency_test_llama.py \
    --output_dir ${output_dir}\
    --task_name alpaca_gpt4 \
    --model_name_or_path ${model_name} \
    --bf16 True \
    --tf32 True \
    --data_path 'data/sft/alpaca_data_gpt4.json' \
    --do_eval \
    --save_strategy no \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --logging_steps 100 \
    --eval_steps 500 \
    --model_max_length 512 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --apply_lora \
    --lora_alpha ${lora_alpha} \
    --lora_r ${lora_r} \
    --report_to none | tee ${output_dir}/log.txt