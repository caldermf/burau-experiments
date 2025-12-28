#!/usr/bin/env python3
"""
Test script to validate and benchmark the batched FFT matmul optimization.

Run this BEFORE using v3.3 in production to ensure correctness.
"""

import torch
import time

# =============================================================================
# REFERENCE IMPLEMENTATION (known correct)
# =============================================================================

def poly_multiply_batch(a: torch.Tensor, b: torch.Tensor, p: int) -> torch.Tensor:
    """Batch polynomial multiplication using FFT convolution."""
    N, D = a.shape
    fft_size = 1 << (2 * D - 1).bit_length()
    
    a_fft = torch.fft.rfft(a.float(), n=fft_size, dim=-1)
    b_fft = torch.fft.rfft(b.float(), n=fft_size, dim=-1)
    c_fft = a_fft * b_fft
    c = torch.fft.irfft(c_fft, n=fft_size, dim=-1)
    
    c = torch.round(c).long() % p
    return c[:, :2*D-1]


def poly_matmul_batch_reference(A: torch.Tensor, B: torch.Tensor, p: int) -> torch.Tensor:
    """Reference: 27 separate FFT calls."""
    N, _, _, D = A.shape
    out_D = 2 * D - 1
    C = torch.zeros(N, 3, 3, out_D, dtype=torch.long, device=A.device)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a_ik = A[:, i, k, :]
                b_kj = B[:, k, j, :]
                conv = poly_multiply_batch(a_ik, b_kj, p)
                C[:, i, j, :] = (C[:, i, j, :] + conv) % p
    
    return C


# =============================================================================
# OPTIMIZED IMPLEMENTATION (batched FFT)
# =============================================================================

# Precompute index mapping
_MATMUL_A_IDX = []
_MATMUL_B_IDX = []
_MATMUL_OUT_IDX = []

for i in range(3):
    for j in range(3):
        for k in range(3):
            _MATMUL_A_IDX.append(i * 3 + k)
            _MATMUL_B_IDX.append(k * 3 + j)
            _MATMUL_OUT_IDX.append(i * 3 + j)

_MATMUL_A_IDX = torch.tensor(_MATMUL_A_IDX)
_MATMUL_B_IDX = torch.tensor(_MATMUL_B_IDX)
_MATMUL_OUT_IDX = torch.tensor(_MATMUL_OUT_IDX)


def poly_matmul_batch_fast(A: torch.Tensor, B: torch.Tensor, p: int) -> torch.Tensor:
    """Optimized: Single batched FFT for all 27 convolutions."""
    N, _, _, D = A.shape
    device = A.device
    out_D = 2 * D - 1
    fft_size = 1 << (out_D).bit_length()
    
    global _MATMUL_A_IDX, _MATMUL_B_IDX, _MATMUL_OUT_IDX
    if _MATMUL_A_IDX.device != device:
        _MATMUL_A_IDX = _MATMUL_A_IDX.to(device)
        _MATMUL_B_IDX = _MATMUL_B_IDX.to(device)
        _MATMUL_OUT_IDX = _MATMUL_OUT_IDX.to(device)
    
    A_flat = A.reshape(N, 9, D)
    B_flat = B.reshape(N, 9, D)
    
    A_pairs = A_flat[:, _MATMUL_A_IDX, :]
    B_pairs = B_flat[:, _MATMUL_B_IDX, :]
    
    A_fft = torch.fft.rfft(A_pairs.float(), n=fft_size, dim=-1)
    B_fft = torch.fft.rfft(B_pairs.float(), n=fft_size, dim=-1)
    
    C_fft = A_fft * B_fft
    C_conv = torch.fft.irfft(C_fft, n=fft_size, dim=-1)
    C_conv = torch.round(C_conv[:, :, :out_D]).long() % p
    
    C_flat = torch.zeros(N, 9, out_D, dtype=torch.long, device=device)
    out_idx_expanded = _MATMUL_OUT_IDX.view(1, 27, 1).expand(N, 27, out_D)
    C_flat.scatter_add_(1, out_idx_expanded, C_conv)
    C_flat = C_flat % p
    
    return C_flat.reshape(N, 3, 3, out_D)


# =============================================================================
# TESTS
# =============================================================================

def test_correctness():
    """Test that fast implementation matches reference for various inputs."""
    print("="*60)
    print("CORRECTNESS TESTS")
    print("="*60)
    
    test_cases = [
        # (N, D, p, device)
        (1, 17, 2, 'cpu'),
        (1, 17, 5, 'cpu'),
        (10, 33, 3, 'cpu'),
        (100, 65, 5, 'cpu'),
        (100, 129, 7, 'cpu'),
        (1000, 65, 5, 'cpu'),
    ]
    
    # Add CUDA tests if available
    if torch.cuda.is_available():
        test_cases += [
            (100, 65, 5, 'cuda'),
            (1000, 129, 7, 'cuda'),
            (10000, 65, 5, 'cuda'),
        ]
    
    all_passed = True
    
    for N, D, p, device in test_cases:
        torch.manual_seed(42)  # Reproducibility
        
        A = torch.randint(0, p, (N, 3, 3, D), dtype=torch.long, device=device)
        B = torch.randint(0, p, (N, 3, 3, D), dtype=torch.long, device=device)
        
        C_ref = poly_matmul_batch_reference(A, B, p)
        C_fast = poly_matmul_batch_fast(A, B, p)
        
        if torch.equal(C_ref, C_fast):
            print(f"  ✓ N={N:5d}, D={D:3d}, p={p}, device={device}")
        else:
            print(f"  ✗ N={N:5d}, D={D:3d}, p={p}, device={device} - MISMATCH!")
            diff_count = (C_ref != C_fast).sum().item()
            total = C_ref.numel()
            print(f"    {diff_count}/{total} elements differ")
            
            # Show first difference
            diff_mask = (C_ref != C_fast)
            idx = diff_mask.nonzero()[0]
            print(f"    First diff at {idx.tolist()}: ref={C_ref[tuple(idx)]}, fast={C_fast[tuple(idx)]}")
            
            all_passed = False
    
    print()
    if all_passed:
        print("✓ ALL CORRECTNESS TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED - DO NOT USE FAST IMPLEMENTATION")
    
    return all_passed


def test_edge_cases():
    """Test edge cases: zeros, identity, specific patterns."""
    print("\n" + "="*60)
    print("EDGE CASE TESTS")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_passed = True
    
    # Test 1: Zero matrices
    print("  Testing zero matrices...")
    A = torch.zeros(10, 3, 3, 50, dtype=torch.long, device=device)
    B = torch.zeros(10, 3, 3, 50, dtype=torch.long, device=device)
    C_ref = poly_matmul_batch_reference(A, B, 5)
    C_fast = poly_matmul_batch_fast(A, B, 5)
    if torch.equal(C_ref, C_fast) and (C_ref == 0).all():
        print("    ✓ Zero matrices")
    else:
        print("    ✗ Zero matrices FAILED")
        all_passed = False
    
    # Test 2: Identity-like (diagonal with 1s at center)
    print("  Testing identity-like matrices...")
    D = 50
    A = torch.zeros(10, 3, 3, D, dtype=torch.long, device=device)
    B = torch.zeros(10, 3, 3, D, dtype=torch.long, device=device)
    center = D // 2  # 25
    for i in range(3):
        A[:, i, i, center] = 1
        B[:, i, i, center] = 1
    C_ref = poly_matmul_batch_reference(A, B, 5)
    C_fast = poly_matmul_batch_fast(A, B, 5)
    if torch.equal(C_ref, C_fast):
        # Result should be identity, but at index 2*center (convolution adds degrees)
        # Output shape is (N, 3, 3, 2*D-1), and convolving index center with center gives 2*center
        out_center = 2 * center
        expected = torch.zeros_like(C_ref)
        for i in range(3):
            expected[:, i, i, out_center] = 1
        if torch.equal(C_ref, expected):
            print("    ✓ Identity-like matrices")
        else:
            print("    ✗ Identity result wrong")
            # Debug info
            print(f"      Expected nonzero at index {out_center}, C_ref shape: {C_ref.shape}")
            all_passed = False
    else:
        print("    ✗ Identity-like matrices MISMATCH")
        all_passed = False
    
    # Test 3: Single nonzero entry
    print("  Testing single nonzero entries...")
    for p in [2, 3, 5, 7]:
        A = torch.zeros(5, 3, 3, 30, dtype=torch.long, device=device)
        B = torch.zeros(5, 3, 3, 30, dtype=torch.long, device=device)
        A[:, 0, 1, 10] = 2
        B[:, 1, 2, 15] = 3
        C_ref = poly_matmul_batch_reference(A, B, p)
        C_fast = poly_matmul_batch_fast(A, B, p)
        if not torch.equal(C_ref, C_fast):
            print(f"    ✗ Single nonzero p={p} MISMATCH")
            all_passed = False
    print("    ✓ Single nonzero entries")
    
    # Test 4: Full random with different primes
    print("  Testing various primes...")
    for p in [2, 3, 5, 7, 11, 13]:
        torch.manual_seed(p)
        A = torch.randint(0, p, (50, 3, 3, 40), dtype=torch.long, device=device)
        B = torch.randint(0, p, (50, 3, 3, 40), dtype=torch.long, device=device)
        C_ref = poly_matmul_batch_reference(A, B, p)
        C_fast = poly_matmul_batch_fast(A, B, p)
        if not torch.equal(C_ref, C_fast):
            print(f"    ✗ Prime p={p} MISMATCH")
            all_passed = False
    print("    ✓ Various primes")
    
    print()
    if all_passed:
        print("✓ ALL EDGE CASE TESTS PASSED")
    else:
        print("✗ SOME EDGE CASES FAILED")
    
    return all_passed


def benchmark():
    """Benchmark reference vs fast implementation."""
    print("\n" + "="*60)
    print("BENCHMARKS")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    test_cases = [
        (1000, 65, 5),
        (5000, 65, 5),
        (10000, 65, 5),
        (10000, 129, 5),
        (50000, 65, 5),
    ]
    
    if device == 'cpu':
        # Smaller tests for CPU
        test_cases = [
            (100, 65, 5),
            (500, 65, 5),
            (1000, 65, 5),
        ]
    
    print(f"\n{'N':>8} {'D':>6} {'p':>4} {'Reference':>12} {'Fast':>12} {'Speedup':>10}")
    print("-" * 60)
    
    for N, D, p in test_cases:
        torch.manual_seed(42)
        A = torch.randint(0, p, (N, 3, 3, D), dtype=torch.long, device=device)
        B = torch.randint(0, p, (N, 3, 3, D), dtype=torch.long, device=device)
        
        # Warmup
        _ = poly_matmul_batch_reference(A[:10], B[:10], p)
        _ = poly_matmul_batch_fast(A[:10], B[:10], p)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Time reference
        t0 = time.time()
        for _ in range(3):
            _ = poly_matmul_batch_reference(A, B, p)
            if device == 'cuda':
                torch.cuda.synchronize()
        t_ref = (time.time() - t0) / 3
        
        # Time fast
        t0 = time.time()
        for _ in range(3):
            _ = poly_matmul_batch_fast(A, B, p)
            if device == 'cuda':
                torch.cuda.synchronize()
        t_fast = (time.time() - t0) / 3
        
        speedup = t_ref / t_fast
        print(f"{N:>8} {D:>6} {p:>4} {t_ref:>10.4f}s {t_fast:>10.4f}s {speedup:>9.2f}x")
    
    print()


def main():
    print("Testing batched FFT optimization for poly_matmul_batch")
    print()
    
    correct = test_correctness()
    if not correct:
        print("\n⚠️  CORRECTNESS TESTS FAILED - DO NOT USE v3.3!")
        return False
    
    edge_ok = test_edge_cases()
    if not edge_ok:
        print("\n⚠️  EDGE CASE TESTS FAILED - DO NOT USE v3.3!")
        return False
    
    benchmark()
    
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print("✓ All correctness tests passed")
    print("✓ All edge case tests passed")
    print("✓ Safe to use v3.3 with batched FFT optimization")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
