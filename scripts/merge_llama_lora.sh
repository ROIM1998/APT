#!/bin/bash
#SBATCH -p ckpt
#SBATCH -A h2lab
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node (1 in this case)
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --mem=64G                 # Memory per node (total memory)
#SBATCH --gpus=a100:1               # Number of GPUs requested
#SBATCH --time=24:00:00             # Walltime (hh:mm:ss)

model_path=$1
output_dir=$2
lora_r=$3
lora_alpha=$4

python merge_llama_model_lora.py \
    --output_dir ${output_dir}\
    --model_name_or_path ${model_path} \
    --task_name alpaca_gpt4 \
    --do_train \
    --do_eval \
    --bf16 True \
    --data_path 'data/sft/alpaca_data_gpt4.json' \
    --model_max_length 512 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --tf32 True \
    --apply_lora \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha}