#!/bin/bash
# =============================================================================
# KERNEL ELEMENT SEARCH - H200
# =============================================================================
#SBATCH --job-name=MONSTER
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G                 # CRITICAL: Need lots of CPU RAM for checkpointing
#SBATCH --time=00:15:00            # Scavenge allows longer times
#SBATCH --requeue                  # Automatically requeue if preempted
#SBATCH --signal=B:USR1@60        # Send signal 120s before timeout
#SBATCH --array=1                 # Defines the range of tasks
#SBATCH --output=slurm_logs/MONSTER_%A_%a.out
#SBATCH --error=slurm_logs/MONSTER_%A_%a.err

set -e
mkdir -p slurm_logs checkpoints

module purge
module load miniconda

PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PYTHON_PATH python braid_search.py


echo "JOB COMPLETED!"