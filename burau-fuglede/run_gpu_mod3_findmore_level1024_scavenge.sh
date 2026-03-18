#!/usr/bin/env bash
#SBATCH --job-name=burau_gpu_p3_more_L1024
#SBATCH --partition=scavenge_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --chdir=/nfs/roberts/project/pi_com36/com36/burau-experiments/burau-fuglede
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUN_DIR="$ROOT/manual_runs/gpu_mod3_findmore_level1024_scavenge"
LOG_FILE="$RUN_DIR/search.log"
HITS_FILE="$RUN_DIR/hits.txt"
STATUS_FILE="$RUN_DIR/status.txt"
BUILD_LOG="$RUN_DIR/build.log"
BIN="$RUN_DIR/burau_exact_modp_gpu_findmore_p3_L1024"

PRIME=3
START_LEVEL=1
STOP_LEVEL=1024
MAX_LEVEL=1024
MAX_SURVIVORS_PER_LEVEL=33554432
MAX_HITS_PER_LEVEL=4194304

mkdir -p "$ROOT/slurm_logs" "$RUN_DIR"
touch "$LOG_FILE" "$BUILD_LOG" "$STATUS_FILE"

log_status() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$1" | tee -a "$STATUS_FILE"
}

extract_last_level() {
  python3 - "$LOG_FILE" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
last_level = 0
level_re = re.compile(r"^level=(\d+)\b")

if path.exists():
    for line in path.read_text().splitlines():
        m = level_re.match(line)
        if m:
            last_level = int(m.group(1))

print(last_level)
PY
}

persist_existing_hits() {
  python3 - "$LOG_FILE" "$HITS_FILE" <<'PY'
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
hits_path = Path(sys.argv[2])

hits = [line for line in log_path.read_text().splitlines() if line.startswith("HIT level=")]
if hits:
    hits_path.write_text("\n".join(hits) + "\n")
PY
}

last_level="$(extract_last_level)"

if [[ ! -s "$HITS_FILE" && -s "$LOG_FILE" ]]; then
  persist_existing_hits
fi

start_level=$((last_level + 1))
if (( start_level < START_LEVEL )); then
  start_level=$START_LEVEL
fi

if (( start_level > STOP_LEVEL )); then
  log_status "Search already completed through level $STOP_LEVEL."
  exit 0
fi

module load CUDA/12.6.0

if [[ ! -x "$BIN" || "$ROOT/burau_exact_modp_gpu_findmore.cu" -nt "$BIN" || "$ROOT/burau_exact_modp_gpu.cu" -nt "$BIN" ]]; then
  log_status "Compiling mod-$PRIME findmore GPU search binary."
  nvcc -O3 \
    -DPRIME="$PRIME" \
    -DMAX_LEVEL="$MAX_LEVEL" \
    -DMAX_SURVIVORS_PER_LEVEL="$MAX_SURVIVORS_PER_LEVEL" \
    -DMAX_HITS_PER_LEVEL="$MAX_HITS_PER_LEVEL" \
    "$ROOT/burau_exact_modp_gpu_findmore.cu" \
    -o "$BIN" 2>&1 | tee -a "$BUILD_LOG"
fi

log_status "Starting findmore search on host $(hostname) from level $start_level to $STOP_LEVEL."

set +e
stdbuf -oL -eL "$BIN" "$start_level" "$STOP_LEVEL" 2>&1 | \
while IFS= read -r line; do
  printf '%s\n' "$line" | tee -a "$LOG_FILE"
  if [[ "$line" == HIT\ level=* ]]; then
    printf '%s\n' "$line" >> "$HITS_FILE"
    sync "$HITS_FILE" 2>/dev/null || true
  fi
done
search_status=${PIPESTATUS[0]}
set -e

if [[ -s "$HITS_FILE" ]]; then
  hit_count="$(wc -l < "$HITS_FILE" | tr -d ' ')"
  log_status "Search has recorded $hit_count hits so far."
fi

if (( search_status == 0 )); then
  log_status "Search reached level $STOP_LEVEL."
else
  log_status "Search exited with status $search_status."
fi

exit "$search_status"
