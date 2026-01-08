#!/bin/bash
# =============================================================================
# KERNEL ELEMENT SEARCH - H200
# =============================================================================
#SBATCH --job-name=GO-EXPERIMENTAL
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G                 # CRITICAL: Need lots of CPU RAM for checkpointing
#SBATCH --time=00:20:00            # Scavenge allows longer times
#SBATCH --requeue                  # Automatically requeue if preempted
#SBATCH --signal=B:USR1@60        # Send signal 120s before timeout
#SBATCH --array=1                 # Defines the range of tasks
#SBATCH --output=slurm_logs/experimental_%A_%a.out
#SBATCH --error=slurm_logs/experimental_%A_%a.err

set -e
mkdir -p slurm_logs checkpoints

module purge
module load miniconda

PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PYTHON_PATH find_kernel.py \
    --p 5 \
    --bucket-size 2_000_000 \
    --use-best 100_000 \
    --max-length 127 \
    --matmul-chunk 8000 \
    --chunk-size 200_000 \
    --degree-multiplier 2


echo "JOB COMPLETED!"