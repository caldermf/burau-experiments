#!/bin/bash
# =============================================================================
# KERNEL ELEMENT SEARCH - H200
# =============================================================================
#SBATCH --job-name=hard_test
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G                 # CRITICAL: Need lots of CPU RAM for checkpointing
#SBATCH --time=00:27:00            # Scavenge allows longer times
#SBATCH --requeue                  # Automatically requeue if preempted
#SBATCH --signal=B:USR1@60        # Send signal 120s before timeout
#SBATCH --output=slurm_logs/hard_%j.out
#SBATCH --error=slurm_logs/hard_%j.err

set -e
mkdir -p slurm_logs checkpoints

module purge
module load miniconda

PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PYTHON_PATH find_kernel_ultra.py \
    --p 5 \
    --bucket-size 2000000 \
    --device cuda \
    --use-best 1000000 \
    --max-length 60 \
    --matmul-chunk 10000


echo "JOB COMPLETED!"