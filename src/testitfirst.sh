#!/bin/bash
# =============================================================================
# P=7 KERNEL ELEMENT SEARCH - H200 (140GB VRAM)
# =============================================================================
# The H200 is your best bet - massive memory allows aggressive parameters
# Use scavenge_gpu partition for longer runtime without charges
# =============================================================================
#SBATCH --job-name=p7_h200
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                 # CRITICAL: Need lots of CPU RAM for checkpointing
#SBATCH --time=00:03:00            # Scavenge allows longer times
#SBATCH --output=slurm_logs/p7_h200_%j.out
#SBATCH --error=slurm_logs/p7_h200_%j.err

set -e
mkdir -p slurm_logs checkpoints

module purge
module load miniconda

PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# H200 settings: 140GB VRAM allows large buckets
# Memory estimate per braid: 3x3xDx8 bytes = 72*D bytes
# With D=1201 (degree_mult=4, max_len=150): ~86KB per matrix
# 60k braids per bucket × 10 buckets × 86KB = ~52GB for buckets
# Plus working memory for matmul: ~40GB
# Total: ~92GB - safe margin on 140GB

$PYTHON_PATH find_kernel.py \
    --p 7 \
    --bucket-size 40000 \
    --chunk-size 40000 \
    --device cuda \
    --use-best 40000 \
    --bootstrap-length 5 \
    --max-length 1600 \
    --degree-multiplier 3 \
    --checkpoint-every 400 \
    --checkpoint-dir "checkpoints/testitfirst_${SLURM_JOB_ID}" \
    --resume-from "checkpoints/p7_h200_3754994/final_state_level_600.pt"

echo "Job completed. Check for kernel elements!"
