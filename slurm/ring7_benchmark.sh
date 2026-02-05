#!/bin/bash
# =============================================================================
# Ring MatMul (F_7[x]/(x^6-1)) benchmark on H200 - Triton vs PyTorch, SoA layout
# =============================================================================
#SBATCH --job-name=ring7_bench
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0:03:00
#SBATCH --output=slurm_logs/ring7_benchmark_%j.out
#SBATCH --error=slurm_logs/ring7_benchmark_%j.err

set -e
# Run from the directory where sbatch was submitted (repo root)
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm_logs

module purge
module load miniconda

PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PYTHON_PATH benchmark7.py --save-plot

echo "JOB COMPLETED!"
