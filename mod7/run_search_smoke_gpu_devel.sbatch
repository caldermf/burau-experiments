#!/usr/bin/env bash
#SBATCH --job-name=mod7_search_smoke
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --chdir=/nfs/roberts/project/pi_com36/com36/burau-experiments/mod7
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

set -euo pipefail

module --force purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
module load Triton/3.3.1-foss-2024a-CUDA-12.6.0

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-cache-${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"

echo "host: $(hostname)"
echo "python: $(command -v python)"
python - <<'PY'
import sys
import torch
import triton

print("sys.executable:", sys.executable)
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("triton:", triton.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

python search.py \
  --use-best 0.001 \
  --bucket-cap 0.001 \
  --max-steps 5 \
  --save-dir results/smoke_gpu_steps5
