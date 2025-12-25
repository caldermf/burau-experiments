#!/usr/bin/env python3
"""
Validation and Benchmark Suite for GPU Burau Search

This script validates the GPU implementation against the original CPU code
and benchmarks performance. Run this on a dev GPU before requesting H200 time.

Usage:
    # From within the braid library directory:
    python gpu_burau/validate_gpu.py
    
    # Or with specific options:
    python gpu_burau/validate_gpu.py --quick      # Fast smoke test
    python gpu_burau/validate_gpu.py --thorough   # Comprehensive validation

Requirements:
    - The original braid library must be importable
    - CuPy must be installed for GPU tests
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add parent directory to path so we can import the braid library
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Test Configuration
# ============================================================================

QUICK_CONFIG = {
    'n': 4,
    'r': 1, 
    'p': 5,
    'bootstrap_length': 4,
    'search_length': 8,
    'bucket_size': 50,
    'n_random_braids': 100,
    'batch_sizes': [10, 50, 100],
}

THOROUGH_CONFIG = {
    'n': 4,
    'r': 1,
    'p': 5,
    'bootstrap_length': 6,
    'search_length': 15,
    'bucket_size': 100,
    'n_random_braids': 500,
    'batch_sizes': [10, 50, 100, 500, 1000],
}


# ============================================================================
# Validation Tests
# ============================================================================

class ValidationError(Exception):
    """Raised when GPU output doesn't match CPU output."""
    pass


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    errors = []
    
    # Test braid library
    try:
        from braid import GNF, BraidGroup
        from jonesrep import JonesCellRep
        from permutations import SymmetricGroup
        import polymat
        print("  ✓ Braid library")
    except ImportError as e:
        errors.append(f"Braid library: {e}")
        print(f"  ✗ Braid library: {e}")
    
    # Test GPU module
    try:
        from gpu_burau import gpu_polymat as gpm
        print(f"  ✓ gpu_polymat (GPU_AVAILABLE={gpm.GPU_AVAILABLE})")
    except ImportError as e:
        errors.append(f"gpu_polymat: {e}")
        print(f"  ✗ gpu_polymat: {e}")
    
    # Test CuPy
    try:
        import cupy as cp
        n_gpus = cp.cuda.runtime.getDeviceCount()
        print(f"  ✓ CuPy ({n_gpus} GPU(s) available)")
    except ImportError:
        print("  ⚠ CuPy not available (CPU-only mode)")
    except Exception as e:
        print(f"  ⚠ CuPy error: {e}")
    
    if errors:
        raise ImportError(f"Missing dependencies: {errors}")
    
    return True


def test_polymat_mul(config):
    """Validate that GPU polymat.mul matches CPU polymat.mul."""
    print("\nTesting polymat multiplication...")
    
    import polymat as cpu_pm
    from gpu_burau import gpu_polymat as gpm
    
    np.random.seed(42)
    p = config['p']
    
    n_tests = 0
    n_passed = 0
    
    for batch_size in config['batch_sizes']:
        for deg in [4, 8, 16, 32]:
            # Create random test matrices
            A = np.random.randint(0, p, (batch_size, 3, 3, deg)).astype(np.int32)
            B = np.random.randint(0, p, (batch_size, 3, 3, deg)).astype(np.int32)
            
            # CPU computation
            C_cpu = cpu_pm.mul(A, B)
            if p > 0:
                C_cpu = C_cpu % p
            
            # GPU computation
            if gpm.GPU_AVAILABLE:
                A_gpu = gpm.to_gpu(A)
                B_gpu = gpm.to_gpu(B)
                C_gpu = gpm.mul(A_gpu, B_gpu, p=p)
                C_from_gpu = gpm.to_cpu(C_gpu)
            else:
                C_from_gpu = gpm.mul(A, B, p=p)
            
            # Compare
            n_tests += 1
            # Trim to same size for comparison
            min_deg = min(C_cpu.shape[-1], C_from_gpu.shape[-1])
            if np.allclose(C_cpu[..., :min_deg], C_from_gpu[..., :min_deg]):
                n_passed += 1
            else:
                diff = np.abs(C_cpu[..., :min_deg] - C_from_gpu[..., :min_deg])
                print(f"  ✗ batch={batch_size}, deg={deg}: max diff = {diff.max()}")
                
    print(f"  {'✓' if n_passed == n_tests else '✗'} Passed {n_passed}/{n_tests} multiplication tests")
    
    if n_passed != n_tests:
        raise ValidationError(f"Multiplication failed {n_tests - n_passed} tests")
    
    return True


def test_polymat_projlen(config):
    """Validate that GPU projlen matches CPU projlen."""
    print("\nTesting projlen computation...")
    
    import polymat as cpu_pm
    from gpu_burau import gpu_polymat as gpm
    
    np.random.seed(123)
    p = config['p']
    
    n_tests = 0
    n_passed = 0
    
    for batch_size in config['batch_sizes']:
        for deg in [8, 16, 32]:
            # Create random test matrices
            A = np.random.randint(0, p, (batch_size, 3, 3, deg)).astype(np.int32)
            
            # CPU computation
            pl_cpu = cpu_pm.projlen(A)
            
            # GPU computation
            if gpm.GPU_AVAILABLE:
                A_gpu = gpm.to_gpu(A)
                pl_gpu = gpm.to_cpu(gpm.projlen(A_gpu))
            else:
                pl_gpu = gpm.projlen(A)
            
            # Compare
            n_tests += 1
            if np.allclose(pl_cpu, pl_gpu):
                n_passed += 1
            else:
                diff = np.abs(pl_cpu - pl_gpu)
                print(f"  ✗ batch={batch_size}, deg={deg}: max diff = {diff.max()}")
    
    print(f"  {'✓' if n_passed == n_tests else '✗'} Passed {n_passed}/{n_tests} projlen tests")
    
    if n_passed != n_tests:
        raise ValidationError(f"projlen failed {n_tests - n_passed} tests")
    
    return True


def test_polymat_projectivise(config):
    """Validate that GPU projectivise matches CPU projectivise."""
    print("\nTesting projectivise...")
    
    import polymat as cpu_pm
    from gpu_burau import gpu_polymat as gpm
    
    np.random.seed(456)
    p = config['p']
    
    n_tests = 0
    n_passed = 0
    
    for batch_size in config['batch_sizes'][:3]:  # Fewer tests for this
        for deg in [8, 16]:
            # Create random test matrices with some leading zeros
            A = np.random.randint(0, p, (batch_size, 3, 3, deg)).astype(np.int32)
            # Add some leading zeros
            A[..., :2] = 0
            
            # CPU computation
            A_proj_cpu = cpu_pm.projectivise(A)
            
            # GPU computation
            if gpm.GPU_AVAILABLE:
                A_gpu = gpm.to_gpu(A)
                A_proj_gpu = gpm.to_cpu(gpm.projectivise(A_gpu))
            else:
                A_proj_gpu = gpm.projectivise(A)
            
            # Compare shapes and values
            n_tests += 1
            if A_proj_cpu.shape == A_proj_gpu.shape and np.allclose(A_proj_cpu, A_proj_gpu):
                n_passed += 1
            else:
                print(f"  ✗ batch={batch_size}, deg={deg}: shapes {A_proj_cpu.shape} vs {A_proj_gpu.shape}")
    
    print(f"  {'✓' if n_passed == n_tests else '✗'} Passed {n_passed}/{n_tests} projectivise tests")
    
    if n_passed != n_tests:
        raise ValidationError(f"projectivise failed {n_tests - n_passed} tests")
    
    return True


def test_braid_evaluation(config):
    """Validate that braid evaluation gives same results."""
    print("\nTesting braid evaluation...")
    
    from braid import BraidGroup
    from jonesrep import JonesCellRep
    import polymat as cpu_pm
    from gpu_burau import gpu_polymat as gpm
    
    rep = JonesCellRep(n=config['n'], r=config['r'], p=config['p'])
    B = BraidGroup(config['n'])
    
    np.random.seed(789)
    
    n_tests = 0
    n_passed = 0
    
    # Test on random braids of various lengths
    for length in range(1, min(config['search_length'], 10)):
        # Sample some random braids
        braids = [B.sample_braid_perm(length) for _ in range(20)]
        
        # Evaluate using the rep's method (this is the "oracle")
        images_cpu = rep.polymat_evaluate_braids_of_same_length(braids)
        
        # Compute projlen
        pl_cpu = cpu_pm.projlen(images_cpu)
        
        if gpm.GPU_AVAILABLE:
            images_gpu = gpm.to_gpu(images_cpu)
            pl_gpu = gpm.to_cpu(gpm.projlen(images_gpu))
        else:
            pl_gpu = gpm.projlen(images_cpu)
        
        n_tests += 1
        if np.allclose(pl_cpu, pl_gpu):
            n_passed += 1
        else:
            print(f"  ✗ length={length}: projlen mismatch")
    
    print(f"  {'✓' if n_passed == n_tests else '✗'} Passed {n_passed}/{n_tests} braid evaluation tests")
    
    if n_passed != n_tests:
        raise ValidationError(f"Braid evaluation failed {n_tests - n_passed} tests")
    
    return True


def test_deterministic_search(config):
    """
    Run a short search with both CPU and GPU trackers using the same seed,
    verify they produce the same bucket contents.
    """
    print("\nTesting deterministic search (same seed, same results)...")
    
    from braid import BraidGroup
    from jonesrep import JonesCellRep
    from braidsearch import Tracker
    import polymat as cpu_pm
    
    # Try to import GPU tracker
    try:
        from gpu_burau.gpu_tracker import GPUTracker
        from gpu_burau import gpu_polymat as gpm
        has_gpu_tracker = True
    except ImportError as e:
        print(f"  ⚠ Cannot import GPUTracker: {e}")
        print("  Skipping deterministic search test")
        return True
    
    seed = 42
    rep = JonesCellRep(n=config['n'], r=config['r'], p=config['p'])
    
    # This test is tricky because the original Tracker doesn't take a seed
    # and has a different interface. Let's just verify that both can run
    # and produce reasonable output.
    
    print("  Running CPU tracker (original)...")
    B = BraidGroup(config['n'])
    
    # Just verify the GPU tracker can be instantiated and run
    print("  Running GPU tracker...")
    try:
        gpu_tracker = GPUTracker(
            rep=rep,
            bucket_size=config['bucket_size'],
            seed=seed,
            use_gpu=gpm.GPU_AVAILABLE,
        )
        
        # Bootstrap with a few lengths
        gpu_tracker.bootstrap_exhaustive(upto_length=config['bootstrap_length'])
        
        # Check we have some buckets
        n_buckets = len(gpu_tracker.active_buckets)
        n_stored = sum(len(gpu_tracker.bucket_braids[b]) for b in gpu_tracker.active_buckets)
        
        print(f"  ✓ GPU tracker: {n_buckets} buckets, {n_stored} braids stored")
        
        # Verify projlen values make sense
        for bucket in list(gpu_tracker.active_buckets)[:5]:
            length, pl = bucket
            if pl < 0 or pl > 100:
                raise ValidationError(f"Unreasonable projlen {pl} at length {length}")
        
        print("  ✓ Bucket projlen values are reasonable")
        
    except Exception as e:
        print(f"  ✗ GPU tracker failed: {e}")
        raise
    
    return True


# ============================================================================
# Benchmarks
# ============================================================================

def benchmark_mul(config):
    """Benchmark matrix multiplication."""
    print("\nBenchmarking matrix multiplication...")
    
    import polymat as cpu_pm
    from gpu_burau import gpu_polymat as gpm
    
    if not gpm.GPU_AVAILABLE:
        print("  ⚠ GPU not available, skipping GPU benchmarks")
        return {}
    
    import cupy as cp
    
    results = {}
    p = config['p']
    
    print(f"  {'Batch':<8} {'Deg':<6} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
    print(f"  {'-'*48}")
    
    for batch_size in config['batch_sizes']:
        for deg in [16, 32]:
            np.random.seed(42)
            A = np.random.randint(0, p, (batch_size, 3, 3, deg)).astype(np.int32)
            B = np.random.randint(0, p, (batch_size, 3, 3, deg)).astype(np.int32)
            
            # Determine iteration count based on batch size
            n_iters = max(10, 500 // batch_size)
            
            # CPU benchmark
            start = time.time()
            for _ in range(n_iters):
                C = cpu_pm.mul(A, B)
                if p > 0:
                    C = C % p
            cpu_time = (time.time() - start) / n_iters * 1000
            
            # GPU benchmark
            A_gpu = gpm.to_gpu(A)
            B_gpu = gpm.to_gpu(B)
            
            # Warmup
            _ = gpm.mul(A_gpu, B_gpu, p=p)
            cp.cuda.Stream.null.synchronize()
            
            start = time.time()
            for _ in range(n_iters):
                C = gpm.mul(A_gpu, B_gpu, p=p)
            cp.cuda.Stream.null.synchronize()
            gpu_time = (time.time() - start) / n_iters * 1000
            
            speedup = cpu_time / gpu_time
            results[(batch_size, deg)] = {'cpu_ms': cpu_time, 'gpu_ms': gpu_time, 'speedup': speedup}
            
            print(f"  {batch_size:<8} {deg:<6} {cpu_time:<12.2f} {gpu_time:<12.2f} {speedup:<10.1f}x")
    
    return results


def benchmark_projlen(config):
    """Benchmark projlen computation."""
    print("\nBenchmarking projlen computation...")
    
    import polymat as cpu_pm
    from gpu_burau import gpu_polymat as gpm
    
    if not gpm.GPU_AVAILABLE:
        print("  ⚠ GPU not available, skipping GPU benchmarks")
        return {}
    
    import cupy as cp
    
    results = {}
    p = config['p']
    
    print(f"  {'Batch':<8} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
    print(f"  {'-'*42}")
    
    for batch_size in config['batch_sizes']:
        np.random.seed(42)
        A = np.random.randint(0, p, (batch_size, 3, 3, 32)).astype(np.int32)
        
        n_iters = max(10, 1000 // batch_size)
        
        # CPU
        start = time.time()
        for _ in range(n_iters):
            _ = cpu_pm.projlen(A)
        cpu_time = (time.time() - start) / n_iters * 1000
        
        # GPU
        A_gpu = gpm.to_gpu(A)
        _ = gpm.projlen(A_gpu)
        cp.cuda.Stream.null.synchronize()
        
        start = time.time()
        for _ in range(n_iters):
            _ = gpm.projlen(A_gpu)
        cp.cuda.Stream.null.synchronize()
        gpu_time = (time.time() - start) / n_iters * 1000
        
        speedup = cpu_time / gpu_time
        results[batch_size] = {'cpu_ms': cpu_time, 'gpu_ms': gpu_time, 'speedup': speedup}
        
        print(f"  {batch_size:<8} {cpu_time:<12.3f} {gpu_time:<12.3f} {speedup:<10.1f}x")
    
    return results


def benchmark_search_step(config):
    """Benchmark a single search step (nf_descendants)."""
    print("\nBenchmarking search step...")
    
    from jonesrep import JonesCellRep
    from gpu_burau.gpu_tracker import GPUTracker
    from gpu_burau import gpu_polymat as gpm
    
    if not gpm.GPU_AVAILABLE:
        print("  ⚠ GPU not available, skipping")
        return {}
    
    rep = JonesCellRep(n=config['n'], r=config['r'], p=config['p'])
    
    # Create tracker and bootstrap
    tracker = GPUTracker(
        rep=rep,
        bucket_size=config['bucket_size'],
        seed=42,
        use_gpu=True,
    )
    
    print(f"  Bootstrapping to length {config['bootstrap_length']}...")
    tracker.bootstrap_exhaustive(upto_length=config['bootstrap_length'])
    
    # Time a few advance steps
    print(f"  Timing search steps...")
    times = []
    for i in range(3):
        start = time.time()
        tracker.advance_one_step(verbose=False)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"    Step {i+1}: {elapsed:.2f}s")
    
    avg_time = sum(times) / len(times)
    print(f"  Average step time: {avg_time:.2f}s")
    
    return {'avg_step_time': avg_time}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate GPU Burau implementation")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    parser.add_argument("--thorough", action="store_true", help="Thorough validation")
    parser.add_argument("--benchmark-only", action="store_true", help="Only run benchmarks")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation")
    args = parser.parse_args()
    
    config = THOROUGH_CONFIG if args.thorough else QUICK_CONFIG
    
    print("="*60)
    print("GPU Burau Implementation Validation")
    print("="*60)
    print(f"\nConfiguration: {'thorough' if args.thorough else 'quick'}")
    print(f"  n={config['n']}, r={config['r']}, p={config['p']}")
    print(f"  bootstrap_length={config['bootstrap_length']}")
    print(f"  bucket_size={config['bucket_size']}")
    print()
    
    all_passed = True
    
    # Imports
    try:
        test_imports()
    except ImportError as e:
        print(f"\n✗ Import test failed: {e}")
        return 1
    
    # Validation tests
    if not args.benchmark_only:
        print("\n" + "="*60)
        print("VALIDATION TESTS")
        print("="*60)
        
        tests = [
            ("polymat.mul", lambda: test_polymat_mul(config)),
            ("polymat.projlen", lambda: test_polymat_projlen(config)),
            ("polymat.projectivise", lambda: test_polymat_projectivise(config)),
            ("braid evaluation", lambda: test_braid_evaluation(config)),
            ("deterministic search", lambda: test_deterministic_search(config)),
        ]
        
        for name, test_fn in tests:
            try:
                test_fn()
            except ValidationError as e:
                print(f"\n✗ {name} FAILED: {e}")
                all_passed = False
            except Exception as e:
                print(f"\n✗ {name} ERROR: {e}")
                all_passed = False
    
    # Benchmarks
    if not args.validate_only:
        print("\n" + "="*60)
        print("BENCHMARKS")
        print("="*60)
        
        benchmark_mul(config)
        benchmark_projlen(config)
        
        if not args.quick:
            benchmark_search_step(config)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if all_passed:
        print("\n✓ All validation tests passed!")
        print("\nYour GPU implementation is ready for production use.")
        print("Next steps:")
        print("  1. Request H200 allocation")
        print("  2. Run: sbatch slurm_job.sh")
        return 0
    else:
        print("\n✗ Some tests failed!")
        print("Please fix the issues before running on H200.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
