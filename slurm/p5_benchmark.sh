#!/bin/bash
#SBATCH --job-name=geordie_benchmark
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # CPU cores for data loading/overhead
#SBATCH --mem=32G                 # RAM (increase if you hit OOM)
#SBATCH --time=0:05:00           # Max runtime (hh:mm:ss)
#SBATCH --output=slurm_logs/p5_benchmark_%j.out   # Saves standard output (print statements)
#SBATCH --error=slurm_logs/p5_benchmark_%j.err    # Saves errors

# 1. Prepare environment (FIXED)

set -e  # optional but recommended

module purge
module load miniconda

# Manually (re-)initialize conda *after* module load
# source /apps/software/2022b/software/miniconda/24.11.3/etc/profile.d/conda.sh

# Tweak chunk-size and make it smaller if you can

PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 3. Run the script
$PYTHON_PATH find_kernel.py \
    --p 5 \
    --bucket-size 15000 \
    --chunk-size 300000 \
    --device cuda \
    --use-best 30000 \
    --bootstrap-length 5 \
    --max-length 66 \
    --checkpoint-dir "checkpoints/h200_test_${SLURM_JOB_ID}"

module --force purge