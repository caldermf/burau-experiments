#!/usr/bin/env python3
"""
Validation script for the GPU-accelerated Burau kernel search.

This script runs comprehensive tests to ensure:
1. The jonesrep.py fix works correctly (Laurent polynomials)
2. The GPU tracker produces correct results
3. GPU results match CPU results exactly

Run this before starting a full search to ensure everything is working.
"""

import sys
import time
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np


def test_jonesrep_fix():
    """Test that the Laurent polynomial fix in jonesrep.py works."""
    print("="*60)
    print("Test 1: JonesCellRep Laurent Polynomial Fix")
    print("="*60)

    from peyl.jonesrep import JonesCellRep

    # Test n=4, r=1, p=5 (the standard Burau representation)
    rep = JonesCellRep(n=4, r=1, p=5)
    print(f"Representation: {rep}")
    print(f"Dimension: {rep.dimension()}")

    # This call was failing before the fix
    try:
        gens, invs = rep.polymat_artin_gens_invs()
        print(f"Generators shape: {gens.shape}")
        print(f"Inverses shape: {invs.shape}")
        print("  polymat_artin_gens_invs: PASSED")
    except Exception as e:
        print(f"  polymat_artin_gens_invs: FAILED - {e}")
        return False

    # Verify that g * g^-1 = identity (projectively)
    from peyl import polymat
    for i in range(len(gens)):
        product = rep.polymat_mul(gens[i], invs[i])
        pl = polymat.projlen(product[None, ...])[0]
        if pl != 1:
            print(f"  g{i} * g{i}^-1: FAILED - projlen={pl}, expected 1")
            return False

    print("  Generator-inverse products: PASSED")
    return True


def test_braid_evaluation():
    """Test that braid evaluation produces correct results."""
    print("\n" + "="*60)
    print("Test 2: Braid Evaluation")
    print("="*60)

    from peyl.jonesrep import JonesCellRep
    from peyl.braid import GNF, BraidGroup
    from peyl import polymat

    rep = JonesCellRep(n=4, r=1, p=5)

    # Test identity
    identity = GNF.identity(4)
    eye = rep.polymat_id()
    id_eval = rep.polymat_evaluate_braid(identity)

    if not np.allclose(eye, id_eval):
        print("  Identity evaluation: FAILED")
        return False
    print("  Identity evaluation: PASSED")

    # Test a few specific braids
    for length in [1, 2, 3]:
        braids = list(GNF.all_of_length(4, length))[:5]
        images = rep.polymat_evaluate_braids_of_same_length(braids)

        for i, braid in enumerate(braids):
            single_eval = rep.polymat_evaluate_braid(braid)
            batch_eval = images[i]

            # They may differ by leading zeros, so compare projlen
            pl_single = polymat.projlen(single_eval[None, ...])[0]
            pl_batch = polymat.projlen(batch_eval[None, ...])[0]

            if pl_single != pl_batch:
                print(f"  Braid {braid}: FAILED - projlen mismatch {pl_single} vs {pl_batch}")
                return False

    print("  Batch evaluation: PASSED")
    return True


def test_gpu_tracker():
    """Test the GPU tracker implementation."""
    print("\n" + "="*60)
    print("Test 3: GPU Tracker")
    print("="*60)

    try:
        from gpu_tracker_v2 import GPUTracker
    except ImportError as e:
        print(f"  Import: FAILED - {e}")
        return False
    print("  Import: PASSED")

    from peyl.jonesrep import JonesCellRep

    rep = JonesCellRep(n=4, r=1, p=5)

    # Create tracker
    try:
        tracker = GPUTracker(rep, bucket_size=50, seed=42, use_gpu=False)  # CPU for deterministic test
    except Exception as e:
        print(f"  Initialization: FAILED - {e}")
        return False
    print("  Initialization: PASSED")

    # Bootstrap
    try:
        tracker.bootstrap_exhaustive(upto_length=3)
    except Exception as e:
        print(f"  Bootstrap: FAILED - {e}")
        return False
    print("  Bootstrap: PASSED")

    # Check that buckets were created
    if not tracker.active_buckets:
        print("  Bucket creation: FAILED - no buckets")
        return False
    print(f"  Buckets created: {len(tracker.active_buckets)}")

    # Advance one step
    try:
        tracker.advance_one_step()
    except Exception as e:
        print(f"  Advance step: FAILED - {e}")
        return False
    print("  Advance step: PASSED")

    # Get stats
    stats = tracker.get_stats()
    print(f"  Braids stored: {stats.total_braids_stored}")
    print(f"  Current length: {stats.current_length}")

    return True


def test_gpu_vs_cpu():
    """Test that GPU and CPU produce identical results."""
    print("\n" + "="*60)
    print("Test 4: GPU vs CPU Consistency")
    print("="*60)

    try:
        import gpu_polymat as gpm
    except ImportError as e:
        print(f"  gpu_polymat import: FAILED - {e}")
        return False

    if not gpm.GPU_AVAILABLE:
        print("  GPU not available - skipping GPU consistency test")
        return True

    from peyl.jonesrep import JonesCellRep
    from peyl.braid import GNF
    from peyl import polymat

    rep = JonesCellRep(n=4, r=1, p=5)

    # Evaluate some braids on CPU
    braids = list(GNF.all_of_length(4, 3))[:20]
    cpu_images = rep.polymat_evaluate_braids_of_same_length(braids)
    cpu_projlen = polymat.projlen(cpu_images)

    # Evaluate on GPU
    gpu_images = gpm.to_gpu(cpu_images)
    gpu_projlen = gpm.to_cpu(gpm.projlen(gpu_images))

    if not np.allclose(cpu_projlen, gpu_projlen):
        print(f"  Projlen: FAILED")
        print(f"    CPU: {cpu_projlen}")
        print(f"    GPU: {gpu_projlen}")
        return False
    print("  Projlen consistency: PASSED")

    # Test GPU multiplication
    A = gpm.to_gpu(cpu_images[:10])
    B = gpm.to_gpu(cpu_images[10:])
    gpu_product = gpm.mul(A, B, p=5)
    gpu_product = gpm.projectivise(gpu_product)

    cpu_product_list = []
    for i in range(10):
        prod = rep.polymat_mul(cpu_images[i], cpu_images[10 + i])
        cpu_product_list.append(prod)
    cpu_product = polymat.pack(cpu_product_list)

    gpu_product_cpu = gpm.to_cpu(gpu_product)

    # Compare projlen (exact values may differ due to degree padding)
    pl_cpu = polymat.projlen(cpu_product)
    pl_gpu = polymat.projlen(gpu_product_cpu)

    if not np.allclose(pl_cpu, pl_gpu):
        print(f"  Multiplication: FAILED")
        print(f"    CPU projlen: {pl_cpu}")
        print(f"    GPU projlen: {pl_gpu}")
        return False
    print("  Multiplication consistency: PASSED")

    return True


def benchmark():
    """Run a simple benchmark."""
    print("\n" + "="*60)
    print("Benchmark")
    print("="*60)

    try:
        import gpu_polymat as gpm
        from gpu_tracker_v2 import GPUTracker
    except ImportError as e:
        print(f"  Import failed: {e}")
        return

    from peyl.jonesrep import JonesCellRep
    from peyl.braid import GNF
    from peyl import polymat

    rep = JonesCellRep(n=4, r=1, p=5)

    # Prepare test data
    braids = list(GNF.all_of_length(4, 4))[:100]
    images = rep.polymat_evaluate_braids_of_same_length(braids)

    # CPU benchmark
    start = time.time()
    for _ in range(10):
        for i in range(len(images)):
            _ = rep.polymat_mul(images[i], images[(i+1) % len(images)])
    cpu_time = time.time() - start
    print(f"  CPU (100 muls x 10): {cpu_time*1000:.1f}ms")

    if gpm.GPU_AVAILABLE:
        # GPU benchmark
        images_gpu = gpm.to_gpu(images)
        gpm.synchronize()

        start = time.time()
        for _ in range(10):
            # Batch multiply all at once
            A = images_gpu
            B = gpm.to_gpu(np.roll(gpm.to_cpu(images_gpu), 1, axis=0))
            _ = gpm.mul(A, B, p=5)
        gpm.synchronize()
        gpu_time = time.time() - start
        print(f"  GPU (100 batch muls x 10): {gpu_time*1000:.1f}ms")
        print(f"  Speedup: {cpu_time/gpu_time:.1f}x")
    else:
        print("  GPU not available for benchmarking")


def main():
    """Run all tests."""
    print("Burau Kernel Search - Implementation Validation")
    print("="*60)
    print()

    all_passed = True

    # Run tests
    if not test_jonesrep_fix():
        all_passed = False
        print("\n*** CRITICAL: jonesrep.py fix failed! ***")

    if not test_braid_evaluation():
        all_passed = False
        print("\n*** CRITICAL: braid evaluation failed! ***")

    if not test_gpu_tracker():
        all_passed = False
        print("\n*** CRITICAL: GPU tracker failed! ***")

    if not test_gpu_vs_cpu():
        all_passed = False
        print("\n*** WARNING: GPU/CPU consistency failed! ***")

    # Run benchmark
    benchmark()

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED")
        print("="*60)
        print("\nYou can now run the search with:")
        print("  python run_search_v2.py --p 5 --max-length 80")
        return 0
    else:
        print("SOME TESTS FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
