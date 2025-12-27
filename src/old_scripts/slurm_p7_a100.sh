#!/bin/bash
# =============================================================================
# P=7 KERNEL ELEMENT SEARCH - A100 (40GB or 80GB VRAM)
# =============================================================================
# A100 comes in 40GB and 80GB variants
# This config is conservative for 40GB; can increase for 80GB
# =============================================================================
#SBATCH --job-name=p7_a100
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/p7_a100_%j.out
#SBATCH --error=slurm_logs/p7_a100_%j.err

set -e
mkdir -p slurm_logs checkpoints

module purge
module load miniconda

PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# A100 40GB settings (conservative)
# Memory estimate: 
# D=901 (degree_mult=3, max_len=150): ~65KB per matrix
# 25k braids × 10 buckets × 65KB = ~16GB for buckets
# Plus working memory: ~15GB
# Total: ~31GB - safe for 40GB

# For A100 80GB, you can double bucket-size and use-best

$PYTHON_PATH find_kernel.py \
    --p 7 \
    --bucket-size 25000 \
    --chunk-size 25000 \
    --device cuda \
    --use-best 35000 \
    --bootstrap-length 5 \
    --max-length 150 \
    --degree-multiplier 3 \
    --checkpoint-every 25 \
    --checkpoint-dir "checkpoints/p7_a100_${SLURM_JOB_ID}"

echo "Job completed. Check for kernel elements!"
