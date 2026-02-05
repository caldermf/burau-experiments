#!/bin/bash
# =============================================================================
# Test suite for morecontext.py ring42_matmul kernel on H200
# =============================================================================
#SBATCH --job-name=test_ring42
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0:05:00
#SBATCH --output=slurm_logs/test_ring42_%j.out
#SBATCH --error=slurm_logs/test_ring42_%j.err

set -e
# Run from the directory where sbatch was submitted (repo root)
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm_logs

module purge
module load miniconda

PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "Running test_morecontext.py"
echo "=========================================="
echo ""

# Try pytest first, fall back to unittest
if $PYTHON_PATH -m pytest test_morecontext.py -v 2>/dev/null; then
    echo ""
    echo "Tests completed with pytest"
else
    echo "pytest not available, using unittest..."
    $PYTHON_PATH -m unittest test_morecontext -v
fi

echo ""
echo "=========================================="
echo "JOB COMPLETED!"
echo "=========================================="
