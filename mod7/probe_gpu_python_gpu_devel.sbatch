#!/usr/bin/env bash
#SBATCH --job-name=mod7_python_probe
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --chdir=/nfs/roberts/project/pi_com36/com36/burau-experiments/mod7
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

set -u

echo "host: $(hostname)"
echo "PATH: $PATH"
echo
echo "[default interpreters]"
command -v python || true
command -v python3 || true
python3 --version || true
python3 - <<'PY' || true
import sys
print("sys.executable:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__)
    print("torch.cuda.is_available:", torch.cuda.is_available())
except Exception as exc:
    print("torch import failed:", repr(exc))
try:
    import triton
    print("triton:", triton.__version__)
except Exception as exc:
    print("triton import failed:", repr(exc))
PY

echo
echo "[modules matching python/conda/torch/triton/cuda]"
module --terse avail 2>&1 | grep -Ei 'python|conda|torch|triton|cuda' || true

echo
echo "[miniconda module]"
module purge || true
module load miniconda || true
command -v conda || true
conda env list || true
for env in base burau_gpu torch_env; do
  echo
  echo "conda activate $env"
  source ~/miniconda3/etc/profile.d/conda.sh || true
  conda activate "$env" || true
  command -v python || true
  python --version || true
  python - <<'PY' || true
import sys
print("sys.executable:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__)
    print("torch.cuda.is_available:", torch.cuda.is_available())
except Exception as exc:
    print("torch import failed:", repr(exc))
try:
    import triton
    print("triton:", triton.__version__)
except Exception as exc:
    print("triton import failed:", repr(exc))
PY
  conda deactivate || true
done
