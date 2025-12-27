#!/bin/bash
# =============================================================================
# RESUME P=7 SEARCH FROM CHECKPOINT
# =============================================================================
# Use this when a scavenge job gets preempted or you want to continue
# =============================================================================
#SBATCH --job-name=p7_resume
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=h200:1              # Match the GPU type you used before
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/p7_resume_%j.out
#SBATCH --error=slurm_logs/p7_resume_%j.err

set -e
mkdir -p slurm_logs

module purge
module load miniconda

PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================
# CONFIGURE THESE:
# ============================================
CHECKPOINT_FILE="checkpoints/p7_h200_XXXXX/final_state_level_100.pt"  # <-- Edit this!
NEW_MAX_LENGTH=200                                                      # <-- How far to continue
# ============================================

if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "ERROR: Checkpoint file not found: $CHECKPOINT_FILE"
    echo "Available checkpoints:"
    find checkpoints -name "*.pt" -type f 2>/dev/null | head -20
    exit 1
fi

echo "Resuming from: $CHECKPOINT_FILE"
echo "Will search up to length: $NEW_MAX_LENGTH"

$PYTHON_PATH find_kernel.py \
    --p 7 \
    --bucket-size 60000 \
    --chunk-size 50000 \
    --device cuda \
    --use-best 80000 \
    --bootstrap-length 5 \
    --max-length $NEW_MAX_LENGTH \
    --degree-multiplier 4 \
    --checkpoint-every 25 \
    --checkpoint-dir "checkpoints/p7_resume_${SLURM_JOB_ID}" \
    --resume-from "$CHECKPOINT_FILE"

echo "Resume completed!"
