#!/bin/bash
# =============================================================================
# KERNEL ELEMENT SEARCH - H200
# =============================================================================
#SBATCH --job-name=jan3test
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                 # CRITICAL: Need lots of CPU RAM for checkpointing
#SBATCH --time=01:30:00            # Scavenge allows longer times
#SBATCH --requeue                  # Automatically requeue if preempted
#SBATCH --signal=B:USR1@60        # Send signal 120s before timeout
#SBATCH --array=1                 # Defines the range of tasks
#SBATCH --output=slurm_logs/jan3_%A_%a.out
#SBATCH --error=slurm_logs/jan3_%A_%a.err

set -e
mkdir -p slurm_logs checkpoints

module purge
module load miniconda

PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PYTHON_PATH find_kernel_ultra.py \
    --p 7 \
    --bucket-size 400000 \
    --device cuda \
    --use-best 200000 \
    --max-length 255 \
    --matmul-chunk 9000


echo "JOB COMPLETED!"