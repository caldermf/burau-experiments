# GPU Performance Fixes

## Summary

The GPU implementation was extremely slow because it was calling CPU functions for the critical hot paths. This document describes the fixes applied.

## Major Issues Found

1. **CPU Functions Called from GPU Code**: The `nf_descendants` function was calling `evaluate_braids_of_same_length` from the CPU version (`peyl.braidsearch`), completely defeating GPU acceleration.

2. **Excessive CPU-GPU Transfers**: Data was being transferred back and forth unnecessarily, adding significant overhead.

3. **Suboptimal Batch Sizes**: Batch sizes were too small for GPU efficiency (5000 vs 10000+).

4. **Inefficient projlen Computation**: `projlen` was being computed on CPU even when GPU was available.

## Fixes Applied

### 1. Created GPU-Accelerated Evaluation Function

Added `evaluate_braids_of_same_length_gpu()` in `gpu_tracker_fixed.py` that:
- Uses the precomputed symmetric table (already on GPU if available)
- Performs all matrix multiplications on GPU
- Minimizes CPU-GPU transfers

### 2. Updated Tracker to Use GPU Functions

- `nf_descendants()` now calls `evaluate_braids_of_same_length_gpu()` instead of the CPU version
- `bootstrap_exhaustive()` also uses GPU evaluation
- Both functions now keep data on GPU during computation

### 3. Optimized CPU-GPU Transfers

- `add_braids_images()` now computes `projlen` on GPU when available
- Data stays on GPU longer during computation
- Only converts to CPU once at the end for storage

### 4. Increased Batch Sizes

- GPU batch size increased from 5000 to 10000 for `nf_descendants`
- Bootstrap batch size increased from 1000 to 5000 when GPU is enabled
- Larger batches better utilize GPU parallelism

### 5. Improved Memory Management

- Precomputed GPU versions of identity and anti-identity matrices
- Symmetric table entries kept on GPU when possible
- Reduced redundant conversions

## Expected Performance Improvement

For the test case (p=2, bucket_size=10), the main bottleneck was calling CPU evaluation functions. With these fixes:

- **Before**: CPU evaluation for every batch → extremely slow
- **After**: GPU evaluation for all batches → should be orders of magnitude faster

The GPU should now actually accelerate the computation instead of just adding overhead.

## Testing

To test the fixes:

```bash
cd src
python run_search_fixed2.py --p 2 --bucket-size 10 --bootstrap-length 2 --max-length 10
```

This should complete much faster than before. The test case that was "still not done" should now run in seconds rather than hanging indefinitely.

## Additional Notes

- The `mul` function still uses nested loops, but this is necessary for the convolution operation
- The loops use efficient batched `matmul` operations which are well-optimized on GPU
- For small matrices and degrees, GPU overhead may still dominate, but for larger batches it should be much faster

