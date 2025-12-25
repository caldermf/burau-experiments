# GPU-Accelerated Burau Kernel Element Search

This package provides GPU acceleration for the reservoir sampling algorithm
described in "4-strand Burau is unfaithful modulo 5" by Gibson, Williamson, and Yacobi.

## Quick Start

### 1. Environment Setup

```bash
# On Yale HPC
module load miniconda cuda/12.0
conda create -n burau_gpu python=3.11 -y
conda activate burau_gpu
pip install cupy-cuda12x numpy pandas matplotlib
```

### 2. Clone and Install the Original Braid Library

```bash
git clone https://github.com/[original-repo]/braid-burau.git
cd braid-burau
pip install -e .
```

### 3. Run a Search

```bash
# Single GPU search
python run_search.py --n 4 --r 1 --p 5 --max-length 80 --seed 12345

# Or submit to SLURM (runs 4 parallel searches)
sbatch slurm_job.sh
```

## Architecture

### File Overview

```
gpu_burau/
├── gpu_polymat.py      # GPU-accelerated polynomial matrix operations
├── gpu_tracker.py      # GPU-accelerated bucket/reservoir management  
├── run_search.py       # Main search script
├── slurm_job.sh        # SLURM job script for HPC
├── setup_env.sh        # Environment setup script
└── README.md           # This file
```

### Key Optimizations

1. **Batch Matrix Operations**: All polynomial matrix multiplications are batched
   and executed on GPU using custom CUDA kernels.

2. **Contiguous Memory**: Bucket storage uses large contiguous GPU arrays to
   minimize memory fragmentation and enable efficient access patterns.

3. **Parallel Reservoir Sampling**: Multiple independent searches run in parallel
   on different GPUs, each with a different random seed.

4. **Hybrid CPU/GPU**: GNF (Garside Normal Form) operations stay on CPU
   (they're inherently sequential), while matrix arithmetic goes to GPU.

## Performance Expectations

| Configuration | Time per Search | Notes |
|--------------|-----------------|-------|
| Original Python (CPU) | ~2 hours | Baseline from paper |
| Single H200 GPU | ~20-40 minutes | 3-6x speedup |
| 4x H200 (parallel) | ~20-40 minutes wall | 12-24x throughput |
| 8-node cluster | ~20-40 minutes wall | 96-192x throughput |

The main speedup comes from:
- Batch matrix multiplication on GPU: **5-10x**
- Parallel independent searches: **Nx** (where N = number of GPUs/seeds)

## Integration with Original Code

The GPU code integrates with the original braid library via the `HybridTracker` class:

```python
from search import JonesSummand
from gpu_burau.run_search import HybridTracker

rep = JonesSummand(n=4, r=1, p=5)
tracker = HybridTracker(rep, bucket_size=500, seed=12345, use_gpu=True)
results = tracker.run_search(max_length=80)
```

## Key Integration Points

To fully enable GPU acceleration, you need to ensure these functions from your
original `search.py` work with the GPU code:

1. `JonesSummand` - Representation specification (unchanged)
2. `symmetric_table(rep)` - Precomputed permutation matrices (cached on GPU)
3. `evaluate_braids_of_same_length(rep, braids)` - Batch evaluation
4. `braid.nf_suffixes(length)` - GNF suffix enumeration (stays on CPU)

## Tuning for Different Problems

### Searching mod p=7

```bash
python run_search.py --n 4 --r 1 --p 7 --max-length 100 --bucket-size 2000
```

Larger bucket sizes help when kernel elements are rarer.

### Searching n=5 (5-strand braids)

```bash
python run_search.py --n 5 --r 1 --p 0 --max-length 30 --bucket-size 5000
```

The dimension is larger (4x4 matrices), so computation is heavier.
Reduce max-length and increase bucket-size.

### Using More GPU Memory

The H200 has 141GB of HBM3. By default we're conservative. To use more:

```python
# In gpu_tracker.py, increase these:
max_garside_length = 150  # default: 100
max_projlen = 300         # default: 200  
bucket_size = 5000        # default: 500
```

## Debugging

### Check GPU Status

```python
import cupy as cp
print(f"CuPy version: {cp.__version__}")
print(f"CUDA available: {cp.cuda.is_available()}")
mem_free, mem_total = cp.cuda.runtime.memGetInfo()
print(f"GPU Memory: {mem_free/1e9:.1f} / {mem_total/1e9:.1f} GB")
```

### Profile Performance

```bash
# Use NVIDIA Nsight Systems
nsys profile python run_search.py --max-length 20

# Or nvprof (older)
nvprof python run_search.py --max-length 20
```

### Common Issues

1. **Out of Memory**: Reduce `bucket_size` or `max_projlen`
2. **Slow Performance**: Ensure batch sizes are large enough (>1000)
3. **Wrong Results**: Run with `--cpu-only` to compare against baseline

## Extending the Code

### Adding New Statistics

To add a new statistic for bucket assignment:

```python
# In gpu_polymat.py
def my_statistic(M: GPUPolyMat) -> cp.ndarray:
    """Compute my statistic for each matrix in batch."""
    # Use CuPy operations for GPU acceleration
    return cp.sum(M.coeffs != 0, axis=(-3, -2, -1))
```

### Custom CUDA Kernels

For maximum performance, write custom CUDA kernels:

```python
kernel = cp.RawKernel(r'''
extern "C" __global__
void my_kernel(const int* input, int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2;
    }
}
''', 'my_kernel')
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{gibson2024burau,
  title={4-strand Burau is unfaithful modulo 5},
  author={Gibson, Joel and Williamson, Geordie and Yacobi, Oded},
  year={2024}
}
```

## License

MIT License - see LICENSE file.
