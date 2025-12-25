#!/bin/bash
#SBATCH --job-name=burau_kernel
#SBATCH --partition=gpu           # Adjust to your cluster's GPU partition name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4       # One task per GPU
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4              # Request 4 GPUs (H200 or whatever is available)
#SBATCH --mem=256G                # H200 nodes typically have lots of RAM
#SBATCH --time=24:00:00           # 24 hour runtime
#SBATCH --output=burau_%j.out
#SBATCH --error=burau_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@yale.edu  # Change this!

# ============================================================================
# SLURM Job Script for GPU-Accelerated Burau Kernel Search
# Yale HPC with NVIDIA GPUs
# ============================================================================

echo "=============================================="
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=============================================="

# Load required modules (adjust module names for your cluster)
module purge
module load miniconda
module load cuda/12.0  # or appropriate CUDA version

# Activate your conda environment
conda activate burau_gpu  # Create this environment first (see setup_env.sh)

# Print GPU info
nvidia-smi

# Set environment variables for optimal GPU performance
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUPY_CACHE_DIR=$SCRATCH/.cupy_cache

# Change to script directory
cd $SLURM_SUBMIT_DIR

# Create output directory
OUTPUT_DIR="${SCRATCH:-/tmp}/burau_results/$SLURM_JOB_ID"
mkdir -p $OUTPUT_DIR

# Get parameters from environment or use defaults
P=${BURAU_P:-5}
BUCKET_SIZE=${BURAU_BUCKET_SIZE:-1000}
MAX_LENGTH=${BURAU_MAX_LENGTH:-80}

echo ""
echo "Search parameters:"
echo "  Prime p: $P"
echo "  Bucket size: $BUCKET_SIZE"  
echo "  Max length: $MAX_LENGTH"
echo ""

# ============================================================================
# Run 4 independent searches in parallel, one per GPU
# ============================================================================
echo "Starting 4 parallel independent searches..."

for GPU_ID in 0 1 2 3; do
    SEED=$((SLURM_JOB_ID * 1000 + GPU_ID * 137 + 42))
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python -u run_search.py \
        --n 4 \
        --r 1 \
        --p $P \
        --bucket-size $BUCKET_SIZE \
        --max-length $MAX_LENGTH \
        --seed $SEED \
        --output "$OUTPUT_DIR/search_gpu${GPU_ID}.json" \
        2>&1 | tee "$OUTPUT_DIR/search_gpu${GPU_ID}.log" &
    
    echo "Started search on GPU $GPU_ID with seed $SEED (PID: $!)"
done

echo ""
echo "All searches launched. Waiting for completion..."
wait

echo ""
echo "All searches completed!"

# ============================================================================
# Aggregate results
# ============================================================================
echo ""
echo "Aggregating results..."
python aggregate_results.py --input-dir "$OUTPUT_DIR" --output "$OUTPUT_DIR/combined_results.json"

# Print summary
echo ""
echo "=============================================="
echo "RESULTS SUMMARY"
echo "=============================================="
python -c "
import json
with open('$OUTPUT_DIR/combined_results.json') as f:
    results = json.load(f)
    
print(f'Searches run: {results.get(\"n_searches\", 0)}')
print(f'Total compute time: {results.get(\"total_compute_time_seconds\", 0)/3600:.2f} hours')
print(f'Kernel candidates found: {results.get(\"n_kernel_candidates\", 0)}')

for c in results.get('kernel_candidates', []):
    print(f'  - Length {c[\"length\"]}: {c[\"braid\"][:60]}...')
"

# Copy results back to submit directory
cp -r "$OUTPUT_DIR" "$SLURM_SUBMIT_DIR/results_$SLURM_JOB_ID"

echo ""
echo "Results copied to: $SLURM_SUBMIT_DIR/results_$SLURM_JOB_ID"
echo "Job finished at $(date)"
