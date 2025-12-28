#!/bin/bash
# =============================================================================
# P=7 KERNEL ELEMENT SEARCH - RTX 5000 ADA (32GB) or A5000 (24GB)
# =============================================================================
# Bouchet has 48x RTX 5000 ADA (32GB) - good middle-ground option
# This config is conservative for 24GB A5000; works for 32GB RTX 5000 Ada too
# =============================================================================
#SBATCH --job-name=p7_rtx5000
#SBATCH --partition=scavenge_gpu  
#SBATCH --gpus=1                   # Will get whatever is available
#SBATCH --constraint="rtx5000ada|a5000"  # Either RTX 5000 Ada or A5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/p7_rtx5000_%j.out
#SBATCH --error=slurm_logs/p7_rtx5000_%j.err

set -e
mkdir -p slurm_logs checkpoints

module purge
module load miniconda

PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# RTX 5000/A5000 settings (24-32GB)
# Very conservative to avoid OOM
# D=601 (degree_mult=2, max_len=150): ~43KB per matrix
# 15k braids × 10 buckets × 43KB = ~6.5GB for buckets
# Working memory: ~10GB
# Total: ~17GB - safe for 24GB

$PYTHON_PATH find_kernel.py \
    --p 7 \
    --bucket-size 15000 \
    --chunk-size 20000 \
    --device cuda \
    --use-best 20000 \
    --bootstrap-length 5 \
    --max-length 150 \
    --degree-multiplier 2 \
    --checkpoint-every 25 \
    --checkpoint-dir "checkpoints/p7_rtx5000_${SLURM_JOB_ID}"

echo "Job completed. Check for kernel elements!"
