#!/usr/bin/env python3
"""
Minimal generator for triton7.py. Run: python gen_triton7.py -o triton7.py
Edit this file; regenerate to update triton7.py.
"""
from __future__ import annotations

import argparse

N_ENTRIES = 9
N_FREQ = 6


def matmul_operands(c: int) -> list[tuple[int, int]]:
    """C[c] = sum over k of A[row(c),k]*B[k,col(c)]. Returns [(a_idx, b_idx), ...]."""
    row, col = c // 3, c % 3
    return [(row * 3 + k, k * 3 + col) for k in range(3)]


def generate() -> str:
    lines = []

    # --- Preamble ---
    lines.append("""import torch
import triton
import triton.language as tl

# --- Constants for F_7 ---
# Generator g=3 is a primitive root for modulo 7. 
# 3^1=3, 3^2=2, 3^3=6, 3^4=4, 3^5=5, 3^6=1.
# But for N=6 (length of poly), we need a primitive 6th root of unity in F_7.
# Fermat's Little Theorem: a^6 = 1 mod 7.
# So literally *any* non-zero element is a 6th root.
# We need a *primitive* one. 3 is a generator of F_7*, so it has order 6.
# Powers of 3: [1, 3, 2, 6, 4, 5]

@triton.jit
def fast_mod7(x):
    \"\"\"
    Computes x % 7 using magic number multiplication (Barrett Reduction).
    We want to avoid the hardware div/rem instructions.
    Magic constant M = ceil(2^19 / 7) = 74899.
    This works for small x (sufficient for our intermediate dot products).
    \"\"\"
    # 1. Approximate division: q = x // 7
    # We cast to int64 to avoid overflow during multiply, then shift back
    q = (x.to(tl.int64) * 74899) >> 19
    q = q.to(tl.int32)
    
    # 2. Compute remainder: r = x - q * 7
    r = x - q * 7
    
    # 3. Correction if remainder >= 7 (rare, but mathematically possible with approx)
    # In Triton/CUDA this is a predicate add/sub, very cheap.
    r = tl.where(r >= 7, r - 7, r)
    return r

@triton.jit
def unpack_poly(packed_val):
    \"\"\"
    Unpacks a single int32 into 6 coefficients (each 3 bits).
    Returns a tuple of 6 tensors.
    \"\"\"
    mask = 7 # 0b111
    # We unroll this manually for the compiler
    c0 = packed_val & mask
    c1 = (packed_val >> 3) & mask
    c2 = (packed_val >> 6) & mask
    c3 = (packed_val >> 9) & mask
    c4 = (packed_val >> 12) & mask
    c5 = (packed_val >> 15) & mask
    return c0, c1, c2, c3, c4, c5

@triton.jit
def pack_poly(c0, c1, c2, c3, c4, c5):
    \"\"\"
    Repacks 6 coefficients into a single int32.
    \"\"\"
    # Shift and OR
    val = c0
    val = val | (c1 << 3)
    val = val | (c2 << 6)
    val = val | (c3 << 9)
    val = val | (c4 << 12)
    val = val | (c5 << 15)
    return val

@triton.jit
def dft_6_point(c0, c1, c2, c3, c4, c5):
    \"\"\"
    Applies DFT over F_7. 
    Input: 6 coefficients (time domain).
    Output: 6 values (frequency domain).
    W = 3 (primitive 6th root).
    \"\"\"
    # Naive DFT unroll is actually best for N=6 in registers.
    # F[k] = sum(c[n] * 3^(nk)) mod 7
    
    # Precomputed powers of 3 mod 7: 1, 3, 2, 6, 4, 5
    # Since we are lazy, let's just write the linear combos. 
    # Because inputs are small (0-6), sums won't overflow int32.
    # We can delay mod 7 until the end of the sum.
    
    # F0 = Sum(c_n * 1)
    f0 = fast_mod7(c0 + c1 + c2 + c3 + c4 + c5)
    
    # F1 = Sum(c_n * 3^n) -> 1, 3, 2, 6, 4, 5
    f1 = fast_mod7(c0*1 + c1*3 + c2*2 + c3*6 + c4*4 + c5*5)
    
    # F2 = Sum(c_n * 3^2n) -> 1, 2, 4, 1, 2, 4
    f2 = fast_mod7(c0*1 + c1*2 + c2*4 + c3*1 + c4*2 + c5*4)
    
    # F3 = Sum(c_n * 3^3n) -> 1, 6, 1, 6, 1, 6 ( Alternating 1, -1 )
    f3 = fast_mod7(c0*1 + c1*6 + c2*1 + c3*6 + c4*1 + c5*6)
    
    # F4 = Sum(c_n * 3^4n) -> 1, 4, 2, 1, 4, 2
    f4 = fast_mod7(c0*1 + c1*4 + c2*2 + c3*1 + c4*4 + c5*2)
    
    # F5 = Sum(c_n * 3^5n) -> 1, 5, 4, 6, 2, 3
    f5 = fast_mod7(c0*1 + c1*5 + c2*4 + c3*6 + c4*2 + c5*3)
    
    return f0, f1, f2, f3, f4, f5

@triton.jit
def idft_6_point(f0, f1, f2, f3, f4, f5):
    \"\"\"
    Inverse DFT. Same as DFT but with W^-1 and scaling by N^-1.
    W = 3, so W^-1 = 5 (since 3*5=15=1 mod 7).
    N = 6, so N^-1 = 6 (since 6*6=36=1 mod 7).
    \"\"\"
    # Powers of 5: 1, 5, 4, 6, 2, 3
    
    # Unscaled results
    # c0_u = Sum(f_k * 5^0) -> 1, 1, 1, 1, 1, 1
    t0 = fast_mod7(f0 + f1 + f2 + f3 + f4 + f5)
    
    # c1_u = Sum(f_k * 5^k) -> 1, 5, 4, 6, 2, 3
    t1 = fast_mod7(f0*1 + f1*5 + f2*4 + f3*6 + f4*2 + f5*3)
    
    # c2_u = Sum(f_k * 5^2k) -> 1, 4, 2, 1, 4, 2
    t2 = fast_mod7(f0*1 + f1*4 + f2*2 + f3*1 + f4*4 + f5*2)
    
    # c3_u = Sum(f_k * 5^3k) -> 1, 6, 1, 6, 1, 6
    t3 = fast_mod7(f0*1 + f1*6 + f2*1 + f3*6 + f4*1 + f5*6)
    
    # c4_u = Sum(f_k * 5^4k) -> 1, 2, 4, 1, 2, 4
    t4 = fast_mod7(f0*1 + f1*2 + f2*4 + f3*1 + f4*2 + f5*4)
    
    # c5_u = Sum(f_k * 5^5k) -> 1, 3, 2, 6, 4, 5
    t5 = fast_mod7(f0*1 + f1*3 + f2*2 + f3*6 + f4*4 + f5*5)
    
    # Apply scaling factor 6 (which is -1 mod 7, convenient!)
    # We multiply by 6 and mod 7
    return fast_mod7(t0*6), fast_mod7(t1*6), fast_mod7(t2*6), fast_mod7(t3*6), fast_mod7(t4*6), fast_mod7(t5*6)

@triton.jit
def ring_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    n_matrices,  # batch size
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Each lane handles ONE matrix multiplication
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_matrices

    # --- 1. LOAD DATA ---
    # SoA layout: A is (9, Batch), so component i is at A_ptr + i * n_matrices + batch_idx
    # This gives perfect memory coalescing: adjacent threads access adjacent addresses.
    
    # Load A (9 packed polys) - coalesced access pattern
""".rstrip())

    for i in range(N_ENTRIES):
        lines.append(f"    a{i} = tl.load(A_ptr + {i} * n_matrices + offs, mask=mask, other=0)")
    lines.append("")
    lines.append("    # Load B (9 packed polys) - coalesced access pattern")
    for i in range(N_ENTRIES):
        lines.append(f"    b{i} = tl.load(B_ptr + {i} * n_matrices + offs, mask=mask, other=0)")
    lines.append("")
    lines.append("    # --- 2. UNPACK & DFT ---")
    lines.append("    # This is the heavy lifting. We convert 9 packed ints into 9 frequency vectors (tuples of 6).")
    lines.append("    # We define a macro-like helper (since triton functions inline)")
    lines.append("    ")  # 4-space blank per triton7
    lines.append("    # A Matrix in frequency domain")
    lines.append("    # Unpack and DFT, storing frequency components explicitly")
    for i in range(N_ENTRIES):
        c_vars = ", ".join(f"a{i}_c{j}" for j in range(N_FREQ))
        lines.append(f"    {c_vars} = unpack_poly(a{i})")
        af_vars = ", ".join(f"af{i}_{j}" for j in range(N_FREQ))
        cf_vars = ", ".join(f"a{i}_c{j}" for j in range(N_FREQ))
        lines.append(f"    {af_vars} = dft_6_point({cf_vars})")
    lines.append("")
    lines.append("    # B Matrix in frequency domain")
    for i in range(N_ENTRIES):
        c_vars = ", ".join(f"b{i}_c{j}" for j in range(N_FREQ))
        lines.append(f"    {c_vars} = unpack_poly(b{i})")
        bf_vars = ", ".join(f"bf{i}_{j}" for j in range(N_FREQ))
        cf_vars = ", ".join(f"b{i}_c{j}" for j in range(N_FREQ))
        lines.append(f"    {bf_vars} = dft_6_point({cf_vars})")
    lines.append("")
    lines.append("    # --- 3. FREQUENCY DOMAIN MATMUL ---")
    lines.append("    # We now have 6 independent 3x3 matmuls.")
    lines.append("    # We unroll k=0..5 explicitly since Triton doesn't support tuple indexing or Python lists.")
    lines.append("    ")  # 4-space blank per triton7
    for k in range(N_FREQ):
        lines.append(f"    # k={k}: frequency component {k}")
        for c in range(N_ENTRIES):
            terms = " + ".join(f"af{a}_{k}*bf{b}_{k}" for a, b in matmul_operands(c))
            lines.append(f"    acc{c}_k{k} = {terms}")
        lines.append("    ")  # 4-space blank per triton7
        for c in range(N_ENTRIES):
            lines.append(f"    c{c}_f{k} = fast_mod7(acc{c}_k{k})")
        # Between freq blocks: 4-space blank; after last block, empty blank before "# --- 4"
        lines.append("    " if k < N_FREQ - 1 else "")
    lines.append("    # --- 4. IDFT & REPACK ---")
    lines.append("    # We now have the spectrum for C. We must inverse transform and repack.")
    lines.append("    # Each cell c0..c8 has 6 frequency components, we IDFT each and pack.")
    lines.append("    ")  # 4-space blank per triton7
    for c in range(N_ENTRIES):
        cf_vars = ", ".join(f"c{c}_f{j}" for j in range(N_FREQ))
        idft_out = ", ".join(f"c{c}_c{j}" for j in range(N_FREQ))
        lines.append(f"    {idft_out} = idft_6_point({cf_vars})")
        pack_args = ", ".join(f"c{c}_c{j}" for j in range(N_FREQ))
        lines.append(f"    res{c} = pack_poly({pack_args})")
        if c < N_ENTRIES - 1:
            lines.append("    ")  # 4-space blank between res blocks; after res8 use empty line
        else:
            lines.append("")  # empty blank before "# --- 5"
    lines.append("    # --- 5. STORE --- (SoA layout: coalesced writes)")
    for i in range(N_ENTRIES):
        lines.append(f"    tl.store(C_ptr + {i} * n_matrices + offs, res{i}, mask=mask)")
    lines.append("")  # blank line before def (per triton7)
    lines.append("""def ring_matmul(A, B):
    \"\"\"
    Batched ring matrix multiplication with SoA layout.
    
    Args:
        A, B: int32 tensors of shape (9, Batch) - SoA layout for coalesced memory access.
              Each row i contains component i of all matrices in the batch.
    Returns:
        C: int32 tensor of shape (9, Batch), same SoA layout.
    \"\"\"
    # A and B are shape (9, Batch) int32 - SoA layout
    assert A.shape[0] == 9 and B.shape[0] == 9, f"Expected (9, Batch), got A={A.shape}, B={B.shape}"
    batch_size = A.shape[1]
    C = torch.empty_like(A)
    
    grid = lambda meta: (triton.cdiv(batch_size, meta['BLOCK_SIZE']),)
    
    ring_matmul_kernel[grid](
        A, B, C,
        n_matrices=batch_size,
        BLOCK_SIZE=128
    )
    return C


# --- Python reference (for testing) ---
def _mod7(x):
    return int(x) % 7


def _pack_poly(coeffs):
    \"\"\"Pack 6 coefficients (0-7) into one int32.\"\"\"
    c0, c1, c2, c3, c4, c5 = coeffs
    return c0 | (c1 << 3) | (c2 << 6) | (c3 << 9) | (c4 << 12) | (c5 << 15)


def _unpack_poly(packed):
    \"\"\"Unpack one int32 into 6 coefficients.\"\"\"
    mask = 7
    return [
        packed & mask,
        (packed >> 3) & mask,
        (packed >> 6) & mask,
        (packed >> 9) & mask,
        (packed >> 12) & mask,
        (packed >> 15) & mask,
    ]


def _dft_6_point(c):
    \"\"\"DFT over F_7, W=3. c is list of 6 coeffs.\"\"\"
    c0, c1, c2, c3, c4, c5 = c
    f0 = _mod7(c0 + c1 + c2 + c3 + c4 + c5)
    f1 = _mod7(c0*1 + c1*3 + c2*2 + c3*6 + c4*4 + c5*5)
    f2 = _mod7(c0*1 + c1*2 + c2*4 + c3*1 + c4*2 + c5*4)
    f3 = _mod7(c0*1 + c1*6 + c2*1 + c3*6 + c4*1 + c5*6)
    f4 = _mod7(c0*1 + c1*4 + c2*2 + c3*1 + c4*4 + c5*2)
    f5 = _mod7(c0*1 + c1*5 + c2*4 + c3*6 + c4*2 + c5*3)
    return [f0, f1, f2, f3, f4, f5]


def _idft_6_point(f):
    \"\"\"Inverse DFT over F_7.\"\"\"
    f0, f1, f2, f3, f4, f5 = f
    t0 = _mod7(f0 + f1 + f2 + f3 + f4 + f5)
    t1 = _mod7(f0*1 + f1*5 + f2*4 + f3*6 + f4*2 + f5*3)
    t2 = _mod7(f0*1 + f1*4 + f2*2 + f3*1 + f4*4 + f5*2)
    t3 = _mod7(f0*1 + f1*6 + f2*1 + f3*6 + f4*1 + f5*6)
    t4 = _mod7(f0*1 + f1*2 + f2*4 + f3*1 + f4*2 + f5*4)
    t5 = _mod7(f0*1 + f1*3 + f2*2 + f3*6 + f4*4 + f5*5)
    return [_mod7(t0*6), _mod7(t1*6), _mod7(t2*6), _mod7(t3*6), _mod7(t4*6), _mod7(t5*6)]


def _poly_mul_conv(a, b):
    \"\"\"Multiply two polynomials (6 coeffs each) in F_7 via DFT convolution.\"\"\"
    fa, fb = _dft_6_point(a), _dft_6_point(b)
    fc = [_mod7(fa[k] * fb[k]) for k in range(6)]
    return _idft_6_point(fc)


def ring_matmul_reference(A, B):
    \"\"\"
    Reference implementation: batched 3x3 matrix multiply in the polynomial ring.
    A, B: (9, batch) int32 tensors (numpy or torch) - SoA layout.
    Returns (9, batch) int32.
    \"\"\"
    import numpy as np
    is_torch = torch.is_tensor(A)
    if is_torch:
        A, B = A.cpu().numpy(), B.cpu().numpy()
    batch = A.shape[1]
    C = np.empty((9, batch), dtype=np.int32)
    for b_idx in range(batch):
        # Extract the 9 components for this batch element
        a_flat = A[:, b_idx]  # (9,)
        b_flat = B[:, b_idx]  # (9,)
        # Reshape to 3x3 (row-major)
        a_mat = [[a_flat[i*3 + j] for j in range(3)] for i in range(3)]
        b_mat = [[b_flat[i*3 + j] for j in range(3)] for i in range(3)]
        c_mat = [[None]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                # C[i,j] = sum_k A[i,k] * B[k,j] (convolution product)
                acc = [0]*6
                for k in range(3):
                    pa = _unpack_poly(int(a_mat[i][k]))
                    pb = _unpack_poly(int(b_mat[k][j]))
                    prod = _poly_mul_conv(pa, pb)
                    for idx in range(6):
                        acc[idx] = _mod7(acc[idx] + prod[idx])
                c_mat[i][j] = _pack_poly(acc)
        for i in range(3):
            for j in range(3):
                C[i*3 + j, b_idx] = c_mat[i][j]
    return torch.from_numpy(C) if is_torch else C""")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Generate triton7.py")
    ap.add_argument("-o", "--output", help="Output file (default: stdout)")
    args = ap.parse_args()
    out = generate()
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)
        print(f"Wrote {args.output}", file=__import__("sys").stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()
