#!/bin/bash
#SBATCH --job-name=burau_p5
#SBATCH --partition=gpu_devel           # Request the GPU partition
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # CPU cores for data loading/overhead
#SBATCH --mem=64G                 # RAM (increase if you hit OOM)
#SBATCH --time=1:00:00           # Max runtime (hh:mm:ss)
#SBATCH --output=slurm_logs/burau_%j.out   # Saves standard output (print statements)
#SBATCH --error=slurm_logs/burau_%j.err    # Saves errors

# 1. Prepare environment (FIXED)

set -e  # optional but recommended

module purge
module load miniconda

# Manually (re-)initialize conda *after* module load
source /apps/software/2022b/software/miniconda/24.11.3/etc/profile.d/conda.sh

# Now activate
conda activate burau_gpu

export PYTHONUNBUFFERED=1

# 3. Run the script
python find_kernel.py \
    --p 5 \
    --bucket-size 20000 \
    --chunk-size 40000 \
    --device cuda \
    --use-best 40000 \
    --bootstrap-length 5 \
    --max-length 80 \
    --checkpoint-dir "checkpoints/run_p5_m80_${SLURM_JOB_ID}"

module --force purge