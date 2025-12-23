#!/usr/bin/env python3
"""
Test script for GPU polymat operations.

This can be run standalone to verify GPU functionality without the full braid library.

Usage:
    python test_gpu.py
"""

import sys
import time
import numpy as np


def test_gpu_availability():
    """Check if GPU is available."""
    print("Testing GPU availability...")
    
    try:
        import cupy as cp
        print(f"  ✓ CuPy version: {cp.__version__}")
        
        n_devices = cp.cuda.runtime.getDeviceCount()
        print(f"  ✓ GPU devices found: {n_devices}")
        
        for i in range(n_devices):
            cp.cuda.Device(i).use()
            props = cp.cuda.runtime.getDeviceProperties(i)
            mem_free, mem_total = cp.cuda.runtime.memGetInfo()
            print(f"    [{i}] {props['name'].decode()}")
            print(f"        Memory: {mem_free/1e9:.1f} / {mem_total/1e9:.1f} GB")
            
        return True
    except ImportError:
        print("  ✗ CuPy not installed")
        return False
    except Exception as e:
        print(f"  ✗ GPU error: {e}")
        return False


def test_polymat_operations():
    """Test the GPU polymat operations."""
    print("\nTesting GPU polymat operations...")
    
    # Import our module
    try:
        from gpu_polymat import (
            mul, pack, projectivise, projlen, 
            to_gpu, to_cpu, GPU_AVAILABLE
        )
    except ImportError:
        # Try from current directory
        sys.path.insert(0, '.')
        from gpu_polymat import (
            mul, pack, projectivise, projlen,
            to_gpu, to_cpu, GPU_AVAILABLE
        )
    
    print(f"  GPU_AVAILABLE: {GPU_AVAILABLE}")
    
    # Create test matrices (simulating 3x3 Burau matrices)
    np.random.seed(42)
    batch_size = 100
    dim = 3
    max_deg = 16
    p = 5
    
    A = np.random.randint(0, p, (batch_size, dim, dim, max_deg)).astype(np.int32)
    B = np.random.randint(0, p, (batch_size, dim, dim, max_deg)).astype(np.int32)
    
    print(f"  Test matrices: {batch_size} x {dim} x {dim} x {max_deg}")
    
    # Test CPU multiplication
    print("\n  Testing multiplication...")
    start = time.time()
    C_cpu = mul(A, B, p=p)
    cpu_time = time.time() - start
    print(f"    CPU time: {cpu_time*1000:.2f}ms")
    
    if GPU_AVAILABLE:
        # Test GPU multiplication
        A_gpu = to_gpu(A)
        B_gpu = to_gpu(B)
        
        # Warmup
        _ = mul(A_gpu, B_gpu, p=p)
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
        
        start = time.time()
        C_gpu = mul(A_gpu, B_gpu, p=p)
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.time() - start
        print(f"    GPU time: {gpu_time*1000:.2f}ms")
        print(f"    Speedup: {cpu_time/gpu_time:.1f}x")
        
        # Verify correctness
        C_from_gpu = to_cpu(C_gpu)
        if np.allclose(C_cpu, C_from_gpu):
            print("    ✓ Results match!")
        else:
            print("    ✗ Results differ!")
            return False
    
    # Test projlen
    print("\n  Testing projlen...")
    pl_cpu = projlen(A)
    print(f"    CPU projlen sample: {pl_cpu[:5]}")
    
    if GPU_AVAILABLE:
        pl_gpu = to_cpu(projlen(to_gpu(A)))
        if np.allclose(pl_cpu, pl_gpu):
            print("    ✓ GPU projlen matches CPU!")
        else:
            print("    ✗ projlen differs!")
            return False
    
    # Test projectivise
    print("\n  Testing projectivise...")
    A_proj = projectivise(A)
    print(f"    Original shape: {A.shape}")
    print(f"    Projectivised shape: {A_proj.shape}")
    
    return True


def benchmark_scaling():
    """Benchmark how performance scales with batch size."""
    print("\nBenchmarking performance scaling...")
    
    try:
        from gpu_polymat import mul, to_gpu, to_cpu, GPU_AVAILABLE
        import cupy as cp
    except ImportError:
        print("  Skipping (imports failed)")
        return
    
    if not GPU_AVAILABLE:
        print("  Skipping (GPU not available)")
        return
    
    dim = 3
    max_deg = 32
    p = 5
    
    batch_sizes = [10, 100, 1000, 5000, 10000]
    
    print(f"  {'Batch':<10} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
    print(f"  {'-'*44}")
    
    for batch_size in batch_sizes:
        np.random.seed(42)
        A = np.random.randint(0, p, (batch_size, dim, dim, max_deg)).astype(np.int32)
        B = np.random.randint(0, p, (batch_size, dim, dim, max_deg)).astype(np.int32)
        
        # CPU
        n_iters = max(1, 100 // batch_size * 10)
        start = time.time()
        for _ in range(n_iters):
            _ = mul(A, B, p=p)
        cpu_time = (time.time() - start) / n_iters * 1000
        
        # GPU
        A_gpu = to_gpu(A)
        B_gpu = to_gpu(B)
        _ = mul(A_gpu, B_gpu, p=p)  # warmup
        cp.cuda.Stream.null.synchronize()
        
        start = time.time()
        for _ in range(n_iters):
            _ = mul(A_gpu, B_gpu, p=p)
        cp.cuda.Stream.null.synchronize()
        gpu_time = (time.time() - start) / n_iters * 1000
        
        speedup = cpu_time / gpu_time
        print(f"  {batch_size:<10} {cpu_time:<12.2f} {gpu_time:<12.2f} {speedup:<10.1f}x")


def main():
    print("="*60)
    print("GPU Burau Search - Test Suite")
    print("="*60)
    
    # Test 1: GPU availability
    gpu_ok = test_gpu_availability()
    
    # Test 2: Polymat operations
    ops_ok = test_polymat_operations()
    
    # Test 3: Scaling benchmark
    if gpu_ok:
        benchmark_scaling()
    
    print("\n" + "="*60)
    if ops_ok:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
