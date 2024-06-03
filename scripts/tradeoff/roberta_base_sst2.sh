#!/bin/bash
#SBATCH -p gpu-rtx6k
#SBATCH -A h2lab
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node (1 in this case)
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --mem=64G                 # Memory per node (total memory)
#SBATCH --gres=gpu:1               # Number of GPUs requested
#SBATCH --time=400:00:00             # Walltime (hh:mm:ss)


mac_constraints=(0.1 0.3 0.4 0.5)

for mac_constraint in ${mac_constraints[@]}; do
    echo "mac_constraint: $mac_constraint"
    bash scripts/adaptpruning/roberta_base_sst2_momentum.sh $mac_constraint 8 -1 cubic_gradual running_fisher running_fisher self_momentum dynamic_block_teacher_dynamic_student
done