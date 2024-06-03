#!/bin/bash
#SBATCH -p gpu-a100
#SBATCH -A h2lab
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node (1 in this case)
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --mem=32G                 # Memory per node (total memory)
#SBATCH --gres=gpu:1               # Number of GPUs requested
#SBATCH --time=24:00:00             # Walltime (hh:mm:ss)

# Execute the run.sh script
bash scripts/adaptpruning/roberta_base_sst2_selfmomentum_noffnstart.sh