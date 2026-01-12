#!/bin/bash
# =============================================================================
# KERNEL ELEMENT SEARCH - H200
# =============================================================================
#SBATCH --job-name=GO7
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G                 # CRITICAL: Need lots of CPU RAM for checkpointing
#SBATCH --time=00:06:00            # Scavenge allows longer times
#SBATCH --requeue                  # Automatically requeue if preempted
#SBATCH --signal=B:USR1@60        # Send signal 120s before timeout
#SBATCH --array=12                 # Defines the range of tasks
#SBATCH --output=slurm_logs/moonshot_%A_%a.out
#SBATCH --error=slurm_logs/moonshot_%A_%a.err

set -e
mkdir -p slurm_logs checkpoints

module purge
module load miniconda

PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PYTHON_PATH find_kernel.py --n 6 --r 2 --p 3 --bucket-size 10000 --matmul-chunk 8500 --use-best 20000 --bootstrap-length 2 --max-length 127 --chunk-size 70000

echo "JOB COMPLETED!"