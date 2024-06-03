#!/bin/bash
#SBATCH -p gpu-rtx6k
#SBATCH -A h2lab
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node (1 in this case)
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --mem=64G                 # Memory per node (total memory)
#SBATCH --gres=gpu:1               # Number of GPUs requested
#SBATCH --time=300:00:00             # Walltime (hh:mm:ss)


distill_mapping_strategies=(dynamic_block_teacher_dynamic_student dynamic_block_teacher_dynamic_cofi_student static_teacher_dynamic_student none static_teacher_dynamic_cofi_student static_teacher_static_student dynamic_block_teacher_static_student)

for distill_mapping_strategy in ${distill_mapping_strategies[@]}; do
    echo "distill_mapping_strategy: $distill_mapping_strategy"
    bash scripts/adaptpruning/roberta_base_mnli_momentum.sh 0.4 8 -1 cubic_gradual running_fisher running_fisher self_momentum $distill_mapping_strategy
done