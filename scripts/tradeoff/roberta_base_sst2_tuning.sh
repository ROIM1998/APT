#!/bin/bash
#SBATCH -p gpu-rtx6k
#SBATCH -A h2lab
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node (1 in this case)
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --mem=64G                 # Memory per node (total memory)
#SBATCH --gres=gpu:1               # Number of GPUs requested
#SBATCH --time=400:00:00             # Walltime (hh:mm:ss)


lora_rs=(16 32 64 128 256)

for lora_r in ${lora_rs[@]}; do
    echo "lora_r: $lora_r"
    bash scripts/adaptpruning/roberta_base_sst2_momentum.sh 0.4 $lora_r -1 cubic_gradual running_fisher running_fisher self_momentum dynamic_block_teacher_dynamic_student
done