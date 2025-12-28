#!/bin/bash
# =============================================================================
# P=7 KERNEL ELEMENT SEARCH - H200 (140GB VRAM)
# =============================================================================
# The H200 is your best bet - massive memory allows aggressive parameters
# Use scavenge_gpu partition for longer runtime without charges
# =============================================================================
#SBATCH --job-name=p7_h200
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                 # CRITICAL: Need lots of CPU RAM for checkpointing
#SBATCH --time=24:00:00            # Scavenge allows longer times
#SBATCH --requeue                  # Automatically requeue if preempted
#SBATCH --signal=B:USR1@120        # Send signal 120s before timeout
#SBATCH --output=slurm_logs/lean_%j.out
#SBATCH --error=slurm_logs/lean_%j.err

set -e
mkdir -p slurm_logs checkpoints

module purge
module load miniconda

PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PYTHON_PATH find_kernel.py \
    --p 7 \
    --bucket-size 6000 \
    --chunk-size 3000 \
    --device cuda \
    --use-best 12000 \
    --bootstrap-length 5 \
    --max-length 6400 \
    --degree-multiplier 2 \
    --checkpoint-every 300 \
    --checkpoint-dir "checkpoints/p7_a100_fresh_${SLURM_JOB_ID}"


echo "JOB COMPLETED!"
