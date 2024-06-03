#!/bin/bash
#SBATCH -p gpu-a100
#SBATCH -A h2lab
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node (1 in this case)
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --mem=32G                 # Memory per node (total memory)
#SBATCH --gres=gpu:1               # Number of GPUs requested
#SBATCH --time=200:00:00             # Walltime (hh:mm:ss)

constraint=$1
task_names=(qnli qqp mnli sst2)

for task in ${task_names[@]}; do
    bash scripts/adaptpruning/bert_base_${task}_momentum.sh $constraint 8 -1 cubic_gradual running_fisher running_fisher self_momentum dynamic_block_teacher_dynamic_student
done