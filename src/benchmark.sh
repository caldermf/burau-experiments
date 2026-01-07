#!/bin/bash
# =============================================================================
# BENCHMARKING THE p=5 BRAIDS PAPER
# =============================================================================
#SBATCH --job-name=GO5
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G                 
#SBATCH --time=00:01:00            # Only need 1 minute for this
#SBATCH --requeue                  # Automatically requeue if preempted
#SBATCH --array=1                 # Defines the range of tasks
#SBATCH --output=slurm_logs/benchmark_%A_%a.out
#SBATCH --error=slurm_logs/benchmark_%A_%a.err

set -e
mkdir -p slurm_logs checkpoints

module purge
module load miniconda

PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PYTHON_PATH find_kernel.py \
    --p 5 \
    --bucket-size 30000 \
    --device cuda \
    --use-best 15000 \
    --max-length 63 \
    --matmul-chunk 8000 \
    --degree-multiplier 2


echo "JOB COMPLETED!"