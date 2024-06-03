#!/bin/bash
#SBATCH -p ckpt
#SBATCH -A h2lab
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node (1 in this case)
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --mem=64G                 # Memory per node (total memory)
#SBATCH --gpus=a100:1               # Number of GPUs requested
#SBATCH --time=24:00:00             # Walltime (hh:mm:ss)

model_name_or_path=$1
if [ -d "$model_name_or_path" ]; then
    output_dir="${model_name_or_path}/alpaca_eval"
else
    output_dir="output/${model_name_or_path}/alpaca_eval"
fi
echo $output_dir
mkdir -p $output_dir

training_batch_size=4

python run_alpaca_eval.py \
    --output_dir ${output_dir}\
    --task_name alpaca_eval \
    --model_name_or_path ${model_name_or_path} \
    --bf16 True \
    --data_path 'data/eval/alpaca/alpaca_eval.json' \
    --do_train \
    --do_eval \
    --model_max_length 512 \
    --per_device_train_batch_size ${training_batch_size} \
    --per_device_eval_batch_size ${training_batch_size} \
    --tf32 True