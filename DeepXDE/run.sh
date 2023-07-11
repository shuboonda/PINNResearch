#!/bin/bash
#SBATCH --job-name=python         # Job name
#SBATCH --output=log_slurm.%j.out # Output file
#SBATCH --error=log_slurm.%j.err  # Error file
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=1       # Number of tasks per node
#SBATCH --gres=gpu:1              # Number of GPUs
#SBATCH --partition=bsudfq        # Partition/queue name
#SBATCH --time=1:00:00            # Wall clock time limit

# Load necessary modules
module load cuda
module load cudnn

# Activate the conda environment
source activate tf-env

# Run the Python script
python3 PINN.py