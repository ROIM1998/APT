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
mac_constraint=$2
num_batches=$3
if [ "$#" -eq 5 ]; then
    lora_alpha=$5
else
    lora_alpha=16
fi

if [ -d $model_path ]; then
    echo "Model path exists"
    output_dir="${model_path}/pruned/constraint_${mac_constraint}/batches_${num_batches}"
else
    echo "Model path does not exist"
    output_dir="llama_output/${model_path}/${task_name}/mt_pruned/constraint_${mac_constraint}/batches_${num_batches}"
fi

echo $output_dir
mkdir -p $output_dir

python post_training_sft_prune.py \
    --output_dir ${output_dir}\
    --model_name_or_path ${model_path} \
    --task_name alpaca_gpt4 \
    --data_path 'data/sft/alpaca_data_gpt4.json' \
    --do_train \
    --do_eval \
    --model_max_length 512 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --pruning_batch_size 1 \
    --pruning_batches ${num_batches} \
    --mac_constraint ${mac_constraint} \
    --lora_alpha ${lora_alpha} \