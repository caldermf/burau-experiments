#!/bin/bash
# ============================================================================
# Environment Setup for GPU-Accelerated Burau Kernel Search
# Run this once to set up the conda environment
# ============================================================================

set -e  # Exit on error

echo "Setting up GPU Burau search environment..."

# Create conda environment
conda create -n burau_gpu python=3.11 -y
conda activate burau_gpu

# Install CUDA-aware packages
# Note: Adjust CUDA version to match your cluster's installation
pip install cupy-cuda12x  # For CUDA 12.x
# pip install cupy-cuda11x  # For CUDA 11.x

# Install other dependencies
pip install numpy pandas matplotlib tqdm

# For the original braid library (adjust path as needed)
# pip install -e /path/to/your/braid/library

# Optional: Install Jupyter for interactive exploration
pip install jupyter jupyterlab ipywidgets

# Verify installation
python -c "
import cupy as cp
print(f'CuPy version: {cp.__version__}')
print(f'CUDA available: {cp.cuda.is_available()}')
if cp.cuda.is_available():
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f'GPU: {props[\"name\"].decode()}')
    print(f'Memory: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB')
"

echo ""
echo "Environment setup complete!"
echo "Activate with: conda activate burau_gpu"
