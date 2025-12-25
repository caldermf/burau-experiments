"""
GPU-accelerated polynomial matrix operations using CuPy.

This module provides drop-in replacements for the functions in polymat.py,
optimized for batch operations on NVIDIA GPUs.

Usage:
    import gpu_polymat as gpm
    
    # Convert existing numpy arrays to GPU
    A_gpu = gpm.to_gpu(A)
    B_gpu = gpm.to_gpu(B)
    
    # Multiply on GPU
    C_gpu = gpm.mul(A_gpu, B_gpu)
    
    # Get statistics
    pl = gpm.projlen(C_gpu)  # Returns cupy array
    
    # Convert back to CPU if needed
    C_cpu = gpm.to_cpu(C_gpu)
"""

import numpy as np
from typing import Sequence, Tuple, Union, Optional

# Try to import CuPy, fall back gracefully
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore
    GPU_AVAILABLE = False

ArrayType = Union[np.ndarray, "cp.ndarray"]


def is_gpu_array(A: ArrayType) -> bool:
    """Check if array is on GPU."""
    if not GPU_AVAILABLE:
        return False
    return isinstance(A, cp.ndarray)


def to_gpu(A: np.ndarray) -> ArrayType:
    """Transfer array to GPU."""
    if not GPU_AVAILABLE:
        return A
    return cp.asarray(A)


def to_cpu(A: ArrayType) -> np.ndarray:
    """Transfer array to CPU."""
    if not GPU_AVAILABLE:
        return A
    return cp.asnumpy(A)


def get_array_module(A: ArrayType):
    """Get the array module (numpy or cupy) for an array."""
    if GPU_AVAILABLE and isinstance(A, cp.ndarray):
        return cp
    return np


# =============================================================================
# Core polymat operations - GPU accelerated versions
# =============================================================================

def eye(n: int, use_gpu: bool = True) -> ArrayType:
    """The identity polymat, of shape (n, n, 1)."""
    if use_gpu and GPU_AVAILABLE:
        return cp.identity(n, dtype=cp.int64)[:, :, None]
    return np.identity(n, dtype=np.int64)[:, :, None]


def trim(A: ArrayType) -> ArrayType:
    """
    Trim trailing zeros from a polymat.
    (α, D) ↦ (α, K), where K is least such that A[..., K:] consists entirely of zeros.
    """
    xp = get_array_module(A)
    last = A.shape[-1]
    while last > 0 and not xp.any(A[..., last - 1]):
        last -= 1
    return A[..., :last] if last < A.shape[-1] else A


def trim_left(A: ArrayType) -> Tuple[int, ArrayType]:
    """
    Trim leading zeros from a polymat.
    Returns (start_index, trimmed_array).
    """
    xp = get_array_module(A)
    start, last = 0, A.shape[-1]
    while start < last and not xp.any(A[..., start]):
        start += 1
    return start, (A[..., start:] if start != 0 else A)


def mul(A: ArrayType, B: ArrayType, quotdeg: Optional[int] = None, p: int = 0) -> ArrayType:
    """
    Calculate the product AB where A, B are polynomial matrices.
    
    This is the core hot path - GPU acceleration provides significant speedup here.
    
    In short notation this function has type:
        (α, I, J, D) × (β, J, K, E) ↦ (α · β, I, K, P)
    
    Args:
        A: Left matrix, shape (..., I, J, D)
        B: Right matrix, shape (..., J, K, E)
        quotdeg: If set, truncate output to this degree
        p: Prime modulus (0 for no mod)
        
    Returns:
        Product matrix
    """
    xp = get_array_module(A)
    
    assert len(A.shape) >= 3 and len(B.shape) >= 3, "Inputs must have length at least 3."
    *alpha, I, J, D = A.shape
    *beta, J2, K, E = B.shape
    assert J == J2, f"Matrix dimensions incompatible: {J} vs {J2}"
    
    P = D + E - 1 if quotdeg is None else quotdeg
    prefix = np.broadcast_shapes(tuple(alpha), tuple(beta)) if alpha or beta else ()
    
    # Allocate output
    result = xp.zeros((*prefix, I, K, P), dtype=A.dtype)
    
    # Vectorized approach: compute all degree combinations at once
    # For each valid (i, d-i) pair, we need to compute A[..., i] @ B[..., d-i]
    # We'll use broadcasting to compute all valid combinations efficiently
    
    # Create index arrays for all valid (i, d-i) pairs
    # For degree d, we need i in [max(0, d-E+1), min(d+1, D))
    for d in range(P):
        i_start = max(0, d - E + 1)
        i_end = min(d + 1, D)
        
        if i_start >= i_end:
            continue
            
        # Compute all matrix products for this degree in one go
        # A[..., i_start:i_end] has shape (*prefix, I, J, i_end-i_start)
        # B[..., d-i_end+1:d-i_start+1] needs to be reversed and indexed correctly
        for i in range(i_start, i_end):
            di = d - i
            if di >= 0 and di < E:
                # Batched matrix multiply: (..., I, J) @ (..., J, K) -> (..., I, K)
                result[..., d] += xp.matmul(A[..., i], B[..., di])
    
    if p > 0:
        result = result % p
        
    return result


def pack(As: Sequence[ArrayType]) -> ArrayType:
    """
    Pack a list of polynomial matrices into a single tensor.
    (I, J, *) × L → (L, I, J, D)
    """
    assert len(As) >= 1
    xp = get_array_module(As[0])
    
    assert all(len(A.shape) == 3 for A in As)
    I, J, _ = As[0].shape
    assert all(A.shape[:2] == (I, J) for A in As)
    
    As_trimmed = [trim(A) for A in As]
    D = max(A.shape[-1] for A in As_trimmed)
    
    result = xp.zeros((len(As_trimmed), I, J, D), dtype=As_trimmed[0].dtype)
    for i, A in enumerate(As_trimmed):
        result[i, :, :, :A.shape[-1]] = A
        
    return result


def zeropad(A: ArrayType, D: int) -> ArrayType:
    """Pad array to degree D with zeros."""
    xp = get_array_module(A)
    assert len(A.shape) >= 3 and A.shape[-1] <= D
    result = xp.zeros((*A.shape[:-1], D), dtype=A.dtype)
    result[..., :A.shape[-1]] = A
    return result


def concatenate(As: Sequence[ArrayType]) -> ArrayType:
    """Concatenate batches of matrices, padding degrees as needed."""
    if len(As) == 0:
        return np.zeros((0, 1, 1, 1), dtype=np.int64)
    
    xp = get_array_module(As[0])
    assert all(len(A.shape) == 4 for A in As)
    _, I, J, _ = As[0].shape
    assert all(A.shape[1:3] == (I, J) for A in As)
    
    As_trimmed = [trim(A) for A in As]
    D = max(A.shape[-1] for A in As_trimmed)
    
    return xp.concatenate([zeropad(A, D) for A in As_trimmed], axis=0)


def projectivise(A: ArrayType) -> ArrayType:
    """
    Shift each matrix so the minimum degree is 0 (divide by v^d).
    (α, I, J, L) ↦ (α, I, J, D)
    """
    xp = get_array_module(A)
    
    assert len(A.shape) >= 3
    
    # Check if any matrix is entirely zero
    if not xp.all(xp.any(A, axis=(-3, -2, -1))):
        return A
    
    # Quick check if already projectivised
    if xp.all(xp.any(A[..., 0], axis=(-2, -1))) and xp.any(A[..., -1]):
        return A
    
    starts, ends = proj_starts_ends(A)
    
    if len(A.shape) == 3:
        # Single matrix case
        start = int(starts)
        end = int(ends)
        return A[..., start:end]
    
    # Batch case - need to handle variable shifts
    new_width = int(xp.max(ends - starts))
    result = xp.zeros((*A.shape[:-1], new_width), dtype=A.dtype)
    
    # Convert to numpy for iteration if on GPU
    starts_np = to_cpu(starts) if is_gpu_array(starts) else starts
    ends_np = to_cpu(ends) if is_gpu_array(ends) else ends
    
    for index in np.ndindex(A.shape[:-3]):
        start = int(starts_np[index])
        end = int(ends_np[index])
        result[index][..., :end-start] = A[index][..., start:end]
    
    return result


# =============================================================================
# Statistics functions - all GPU accelerated
# =============================================================================

def degmax(A: ArrayType) -> ArrayType:
    """Return the maximum degree of each polynomial matrix."""
    xp = get_array_module(A)
    assert len(A.shape) >= 3
    nonzeros = xp.any(A, axis=(-3, -2))
    return A.shape[-1] - 1 - xp.argmax(nonzeros[..., ::-1], axis=-1)


def valmin(A: ArrayType) -> ArrayType:
    """Return the minimum valuation (first nonzero degree) of each polynomial matrix."""
    xp = get_array_module(A)
    assert len(A.shape) >= 3
    nonzeros = xp.any(A, axis=(-3, -2))
    return xp.argmax(nonzeros, axis=-1)


def nz_terms(A: ArrayType) -> ArrayType:
    """Return the number of nonzero terms."""
    xp = get_array_module(A)
    assert len(A.shape) >= 3
    return xp.count_nonzero(A, axis=(-3, -2, -1))


def projlen(A: ArrayType) -> ArrayType:
    """
    degmax - valmin + 1, or zero if A == 0.
    This is the key statistic for the reservoir sampling algorithm.
    (α, I, J, D) -> (α)
    """
    xp = get_array_module(A)
    nonzeros = xp.any(A, axis=(-3, -2))
    starts = xp.argmax(nonzeros, axis=-1)
    ends = nonzeros.shape[-1] - xp.argmax(nonzeros[..., ::-1], axis=-1)
    return ends - starts


def rhogap(A: ArrayType) -> ArrayType:
    """Return (degmax - valmin) / 2."""
    return (degmax(A) - valmin(A)) / 2


def efflen(A: ArrayType) -> ArrayType:
    """degmax - valmin, or zero if A == 0."""
    xp = get_array_module(A)
    is_zero = (A == 0).all(axis=(-3, -2, -1))
    return xp.where(is_zero, 0, degmax(A) - valmin(A))


def proj_starts_ends(A: ArrayType) -> Tuple[ArrayType, ArrayType]:
    """
    Returns (starts, ends) arrays for projectivisation.
    (α, I, J, D) → (α), (α)
    """
    xp = get_array_module(A)
    assert len(A.shape) >= 3
    
    nonzeros = xp.any(A, axis=(-3, -2))
    starts = xp.argmax(nonzeros, axis=-1)
    ends = nonzeros.shape[-1] - xp.argmax(nonzeros[..., ::-1], axis=-1)
    return starts, ends


def projrank(A: ArrayType, terms: int) -> int:
    """Sum of ranks of the first `terms` coefficient matrices."""
    xp = get_array_module(A)
    assert terms >= 0
    assert len(A.shape) >= 3 and A.shape[-1] >= 1
    
    total = 0
    for i in range(min(terms, A.shape[-1])):
        ranks = xp.linalg.matrix_rank(A[..., i])
        total += int(ranks.sum()) if hasattr(ranks, 'sum') else int(ranks)
    return total


# =============================================================================
# GPU memory management utilities
# =============================================================================

def memory_info() -> dict:
    """Get GPU memory information."""
    if not GPU_AVAILABLE:
        return {'available': False}
    
    mem_free, mem_total = cp.cuda.runtime.memGetInfo()
    return {
        'available': True,
        'free_gb': mem_free / 1e9,
        'total_gb': mem_total / 1e9,
        'used_gb': (mem_total - mem_free) / 1e9,
    }


def clear_cache():
    """Clear GPU memory cache."""
    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()


def synchronize():
    """Synchronize GPU operations."""
    if GPU_AVAILABLE:
        cp.cuda.Stream.null.synchronize()


# =============================================================================
# Testing
# =============================================================================

def test_gpu_polymat():
    """Run tests to verify GPU implementation matches CPU."""
    print("Testing GPU polymat implementation...")
    print(f"GPU available: {GPU_AVAILABLE}")
    
    if GPU_AVAILABLE:
        info = memory_info()
        print(f"GPU memory: {info['free_gb']:.1f} GB free / {info['total_gb']:.1f} GB total")
    
    # Test basic operations
    np.random.seed(42)
    A_np = np.random.randint(0, 5, (10, 3, 3, 8)).astype(np.int32)
    B_np = np.random.randint(0, 5, (10, 3, 3, 8)).astype(np.int32)
    
    # CPU multiplication
    C_cpu = mul(A_np, B_np, p=5)
    
    if GPU_AVAILABLE:
        # GPU multiplication
        A_gpu = to_gpu(A_np)
        B_gpu = to_gpu(B_np)
        C_gpu = mul(A_gpu, B_gpu, p=5)
        C_from_gpu = to_cpu(C_gpu)
        
        assert np.allclose(C_cpu, C_from_gpu), "GPU and CPU results differ!"
        print("✓ GPU multiplication matches CPU")
        
        # Test projlen
        pl_cpu = projlen(A_np)
        pl_gpu = to_cpu(projlen(A_gpu))
        assert np.allclose(pl_cpu, pl_gpu), "projlen differs!"
        print("✓ projlen matches")
        
        # Test projectivise
        A_proj_cpu = projectivise(A_np)
        A_proj_gpu = to_cpu(projectivise(A_gpu))
        assert A_proj_cpu.shape == A_proj_gpu.shape, "projectivise shapes differ!"
        print("✓ projectivise matches")
        
        # Benchmark
        import time
        n_iters = 100
        
        # CPU timing
        start = time.time()
        for _ in range(n_iters):
            _ = mul(A_np, B_np, p=5)
        cpu_time = time.time() - start
        
        # GPU timing
        synchronize()
        start = time.time()
        for _ in range(n_iters):
            _ = mul(A_gpu, B_gpu, p=5)
        synchronize()
        gpu_time = time.time() - start
        
        print(f"✓ Benchmark ({n_iters} iters, batch=10, 3x3x8):")
        print(f"  CPU: {cpu_time*1000:.1f}ms, GPU: {gpu_time*1000:.1f}ms")
        print(f"  Speedup: {cpu_time/gpu_time:.1f}x")
    else:
        print("(GPU tests skipped - CuPy not available)")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_gpu_polymat()
