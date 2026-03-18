#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$ROOT_DIR"

mkdir -p data logs

timestamp="$(date +%Y%m%d_%H%M%S)"

BUCKET_JSON="data/a3_mod_p_bucket_search_p5_len60_cuda_${timestamp}.json"
BUCKET_LOG="logs/a3_mod_p_bucket_search_p5_len60_cuda_${timestamp}.log"
NATIVE_JSON="data/a3_mod_p_gpu_native_search_p5_len60_cuda_${timestamp}.json"
NATIVE_LOG="logs/a3_mod_p_gpu_native_search_p5_len60_cuda_${timestamp}.log"
NATIVE_LARGE_JSON="data/a3_mod_p_gpu_native_search_p5_len120_bv2_cuda_${timestamp}.json"
NATIVE_LARGE_LOG="logs/a3_mod_p_gpu_native_search_p5_len120_bv2_cuda_${timestamp}.log"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate burau_gpu

write_header() {
  local log_file="$1"
  local label="$2"
  shift 2
  {
    echo "[$(date --iso-8601=seconds)] $label"
    echo "cwd: $ROOT_DIR"
    echo "hostname: $(hostname)"
    echo "python: $(command -v python)"
    echo "command: $*"
    echo
    echo "[nvidia-smi]"
    nvidia-smi
    echo
    echo "[torch cuda info]"
    python - <<'PY'
import sys
import torch
print("sys.executable:", sys.executable)
print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        print(f"device[{idx}].name:", torch.cuda.get_device_name(idx))
        print(f"device[{idx}].total_memory:", props.total_memory)
PY
    echo
    echo "[run output]"
  } >"$log_file"
}

run_benchmark() {
  local label="$1"
  local log_file="$2"
  shift 2
  write_header "$log_file" "$label" "$@"
  {
    /usr/bin/time -p "$@"
  } >>"$log_file" 2>&1
}

run_benchmark \
  "a3_mod_p_bucket_search matched benchmark" \
  "$BUCKET_LOG" \
  python a3_mod_p_bucket_search.py \
  --backend torch \
  --device cuda \
  --p 5 \
  --max-g-length 60 \
  --cap-1 500 \
  --cap-2 500 \
  --total-cap-1 50000 \
  --total-cap-2 50000 \
  --first-steps 12 \
  --output "$BUCKET_JSON"

run_benchmark \
  "a3_mod_p_gpu_native_search matched benchmark" \
  "$NATIVE_LOG" \
  python a3_mod_p_gpu_native_search.py \
  --backend torch \
  --device cuda \
  --p 5 \
  --max-g-length 60 \
  --cap-1 500 \
  --cap-2 500 \
  --total-cap-1 50000 \
  --total-cap-2 50000 \
  --first-steps 12 \
  --output "$NATIVE_JSON"

run_benchmark \
  "a3_mod_p_gpu_native_search larger sanity benchmark" \
  "$NATIVE_LARGE_LOG" \
  python a3_mod_p_gpu_native_search.py \
  --backend torch \
  --device cuda \
  --p 5 \
  --base-vertex 2 \
  --max-g-length 120 \
  --cap-1 8000 \
  --cap-2 8000 \
  --total-cap-1 1000000 \
  --total-cap-2 1000000 \
  --first-steps 24 \
  --expansion-chunk-size 262144 \
  --output "$NATIVE_LARGE_JSON"

printf 'bucket_json=%s\n' "$BUCKET_JSON"
printf 'bucket_log=%s\n' "$BUCKET_LOG"
printf 'native_json=%s\n' "$NATIVE_JSON"
printf 'native_log=%s\n' "$NATIVE_LOG"
printf 'native_large_json=%s\n' "$NATIVE_LARGE_JSON"
printf 'native_large_log=%s\n' "$NATIVE_LARGE_LOG"
