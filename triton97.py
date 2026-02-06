import torch
import triton
import triton.language as tl

# --- Constants for F_7 ---
# Generator g=5 is a primitive root for modulo 97
# 5^1 = 5, 5^2 = 25, 5^3 = 28, 5^4 = 43, 5^5 = 21, 5^6 = 8, 5^7 = 40, 5^8 = 6, 5^9 = 30, ... 
# Powers of 3: [1, 3, 2, 6, 4, 5]

@triton.jit
def fast_mod7(x):
    """
    Computes x % 7 using magic number multiplication (Barrett Reduction).
    We want to avoid the hardware div/rem instructions.
    Magic constant M = ceil(2^19 / 7) = 74899.
    This works for small x (sufficient for our intermediate dot products).
    """
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
    """
    Unpacks a single int32 into 6 coefficients (each 3 bits).
    Returns a tuple of 6 tensors.
    """
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
    """
    Repacks 6 coefficients into a single int32.
    """
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
    """
    Applies DFT over F_7. 
    Input: 6 coefficients (time domain).
    Output: 6 values (frequency domain).
    W = 3 (primitive 6th root).
    """
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
    """
    Inverse DFT. Same as DFT but with W^-1 and scaling by N^-1.
    W = 3, so W^-1 = 5 (since 3*5=15=1 mod 7).
    N = 6, so N^-1 = 6 (since 6*6=36=1 mod 7).
    """
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
    a0 = tl.load(A_ptr + 0 * n_matrices + offs, mask=mask, other=0)
    a1 = tl.load(A_ptr + 1 * n_matrices + offs, mask=mask, other=0)
    a2 = tl.load(A_ptr + 2 * n_matrices + offs, mask=mask, other=0)
    a3 = tl.load(A_ptr + 3 * n_matrices + offs, mask=mask, other=0)
    a4 = tl.load(A_ptr + 4 * n_matrices + offs, mask=mask, other=0)
    a5 = tl.load(A_ptr + 5 * n_matrices + offs, mask=mask, other=0)
    a6 = tl.load(A_ptr + 6 * n_matrices + offs, mask=mask, other=0)
    a7 = tl.load(A_ptr + 7 * n_matrices + offs, mask=mask, other=0)
    a8 = tl.load(A_ptr + 8 * n_matrices + offs, mask=mask, other=0)

    # Load B (9 packed polys) - coalesced access pattern
    b0 = tl.load(B_ptr + 0 * n_matrices + offs, mask=mask, other=0)
    b1 = tl.load(B_ptr + 1 * n_matrices + offs, mask=mask, other=0)
    b2 = tl.load(B_ptr + 2 * n_matrices + offs, mask=mask, other=0)
    b3 = tl.load(B_ptr + 3 * n_matrices + offs, mask=mask, other=0)
    b4 = tl.load(B_ptr + 4 * n_matrices + offs, mask=mask, other=0)
    b5 = tl.load(B_ptr + 5 * n_matrices + offs, mask=mask, other=0)
    b6 = tl.load(B_ptr + 6 * n_matrices + offs, mask=mask, other=0)
    b7 = tl.load(B_ptr + 7 * n_matrices + offs, mask=mask, other=0)
    b8 = tl.load(B_ptr + 8 * n_matrices + offs, mask=mask, other=0)

    # --- 2. UNPACK & DFT ---
    # This is the heavy lifting. We convert 9 packed ints into 9 frequency vectors (tuples of 6).
    # We define a macro-like helper (since triton functions inline)
    
    # A Matrix in frequency domain
    # Unpack and DFT, storing frequency components explicitly
    a0_c0, a0_c1, a0_c2, a0_c3, a0_c4, a0_c5 = unpack_poly(a0)
    af0_0, af0_1, af0_2, af0_3, af0_4, af0_5 = dft_6_point(a0_c0, a0_c1, a0_c2, a0_c3, a0_c4, a0_c5)
    a1_c0, a1_c1, a1_c2, a1_c3, a1_c4, a1_c5 = unpack_poly(a1)
    af1_0, af1_1, af1_2, af1_3, af1_4, af1_5 = dft_6_point(a1_c0, a1_c1, a1_c2, a1_c3, a1_c4, a1_c5)
    a2_c0, a2_c1, a2_c2, a2_c3, a2_c4, a2_c5 = unpack_poly(a2)
    af2_0, af2_1, af2_2, af2_3, af2_4, af2_5 = dft_6_point(a2_c0, a2_c1, a2_c2, a2_c3, a2_c4, a2_c5)
    a3_c0, a3_c1, a3_c2, a3_c3, a3_c4, a3_c5 = unpack_poly(a3)
    af3_0, af3_1, af3_2, af3_3, af3_4, af3_5 = dft_6_point(a3_c0, a3_c1, a3_c2, a3_c3, a3_c4, a3_c5)
    a4_c0, a4_c1, a4_c2, a4_c3, a4_c4, a4_c5 = unpack_poly(a4)
    af4_0, af4_1, af4_2, af4_3, af4_4, af4_5 = dft_6_point(a4_c0, a4_c1, a4_c2, a4_c3, a4_c4, a4_c5)
    a5_c0, a5_c1, a5_c2, a5_c3, a5_c4, a5_c5 = unpack_poly(a5)
    af5_0, af5_1, af5_2, af5_3, af5_4, af5_5 = dft_6_point(a5_c0, a5_c1, a5_c2, a5_c3, a5_c4, a5_c5)
    a6_c0, a6_c1, a6_c2, a6_c3, a6_c4, a6_c5 = unpack_poly(a6)
    af6_0, af6_1, af6_2, af6_3, af6_4, af6_5 = dft_6_point(a6_c0, a6_c1, a6_c2, a6_c3, a6_c4, a6_c5)
    a7_c0, a7_c1, a7_c2, a7_c3, a7_c4, a7_c5 = unpack_poly(a7)
    af7_0, af7_1, af7_2, af7_3, af7_4, af7_5 = dft_6_point(a7_c0, a7_c1, a7_c2, a7_c3, a7_c4, a7_c5)
    a8_c0, a8_c1, a8_c2, a8_c3, a8_c4, a8_c5 = unpack_poly(a8)
    af8_0, af8_1, af8_2, af8_3, af8_4, af8_5 = dft_6_point(a8_c0, a8_c1, a8_c2, a8_c3, a8_c4, a8_c5)

    # B Matrix in frequency domain
    b0_c0, b0_c1, b0_c2, b0_c3, b0_c4, b0_c5 = unpack_poly(b0)
    bf0_0, bf0_1, bf0_2, bf0_3, bf0_4, bf0_5 = dft_6_point(b0_c0, b0_c1, b0_c2, b0_c3, b0_c4, b0_c5)
    b1_c0, b1_c1, b1_c2, b1_c3, b1_c4, b1_c5 = unpack_poly(b1)
    bf1_0, bf1_1, bf1_2, bf1_3, bf1_4, bf1_5 = dft_6_point(b1_c0, b1_c1, b1_c2, b1_c3, b1_c4, b1_c5)
    b2_c0, b2_c1, b2_c2, b2_c3, b2_c4, b2_c5 = unpack_poly(b2)
    bf2_0, bf2_1, bf2_2, bf2_3, bf2_4, bf2_5 = dft_6_point(b2_c0, b2_c1, b2_c2, b2_c3, b2_c4, b2_c5)
    b3_c0, b3_c1, b3_c2, b3_c3, b3_c4, b3_c5 = unpack_poly(b3)
    bf3_0, bf3_1, bf3_2, bf3_3, bf3_4, bf3_5 = dft_6_point(b3_c0, b3_c1, b3_c2, b3_c3, b3_c4, b3_c5)
    b4_c0, b4_c1, b4_c2, b4_c3, b4_c4, b4_c5 = unpack_poly(b4)
    bf4_0, bf4_1, bf4_2, bf4_3, bf4_4, bf4_5 = dft_6_point(b4_c0, b4_c1, b4_c2, b4_c3, b4_c4, b4_c5)
    b5_c0, b5_c1, b5_c2, b5_c3, b5_c4, b5_c5 = unpack_poly(b5)
    bf5_0, bf5_1, bf5_2, bf5_3, bf5_4, bf5_5 = dft_6_point(b5_c0, b5_c1, b5_c2, b5_c3, b5_c4, b5_c5)
    b6_c0, b6_c1, b6_c2, b6_c3, b6_c4, b6_c5 = unpack_poly(b6)
    bf6_0, bf6_1, bf6_2, bf6_3, bf6_4, bf6_5 = dft_6_point(b6_c0, b6_c1, b6_c2, b6_c3, b6_c4, b6_c5)
    b7_c0, b7_c1, b7_c2, b7_c3, b7_c4, b7_c5 = unpack_poly(b7)
    bf7_0, bf7_1, bf7_2, bf7_3, bf7_4, bf7_5 = dft_6_point(b7_c0, b7_c1, b7_c2, b7_c3, b7_c4, b7_c5)
    b8_c0, b8_c1, b8_c2, b8_c3, b8_c4, b8_c5 = unpack_poly(b8)
    bf8_0, bf8_1, bf8_2, bf8_3, bf8_4, bf8_5 = dft_6_point(b8_c0, b8_c1, b8_c2, b8_c3, b8_c4, b8_c5)

    # --- 3. FREQUENCY DOMAIN MATMUL ---
    # We now have 6 independent 3x3 matmuls.
    # We unroll k=0..5 explicitly since Triton doesn't support tuple indexing or Python lists.
    
    # k=0: frequency component 0
    acc0_k0 = af0_0*bf0_0 + af1_0*bf3_0 + af2_0*bf6_0
    acc1_k0 = af0_0*bf1_0 + af1_0*bf4_0 + af2_0*bf7_0
    acc2_k0 = af0_0*bf2_0 + af1_0*bf5_0 + af2_0*bf8_0
    acc3_k0 = af3_0*bf0_0 + af4_0*bf3_0 + af5_0*bf6_0
    acc4_k0 = af3_0*bf1_0 + af4_0*bf4_0 + af5_0*bf7_0
    acc5_k0 = af3_0*bf2_0 + af4_0*bf5_0 + af5_0*bf8_0
    acc6_k0 = af6_0*bf0_0 + af7_0*bf3_0 + af8_0*bf6_0
    acc7_k0 = af6_0*bf1_0 + af7_0*bf4_0 + af8_0*bf7_0
    acc8_k0 = af6_0*bf2_0 + af7_0*bf5_0 + af8_0*bf8_0
    
    c0_f0 = fast_mod7(acc0_k0)
    c1_f0 = fast_mod7(acc1_k0)
    c2_f0 = fast_mod7(acc2_k0)
    c3_f0 = fast_mod7(acc3_k0)
    c4_f0 = fast_mod7(acc4_k0)
    c5_f0 = fast_mod7(acc5_k0)
    c6_f0 = fast_mod7(acc6_k0)
    c7_f0 = fast_mod7(acc7_k0)
    c8_f0 = fast_mod7(acc8_k0)
    
    # k=1: frequency component 1
    acc0_k1 = af0_1*bf0_1 + af1_1*bf3_1 + af2_1*bf6_1
    acc1_k1 = af0_1*bf1_1 + af1_1*bf4_1 + af2_1*bf7_1
    acc2_k1 = af0_1*bf2_1 + af1_1*bf5_1 + af2_1*bf8_1
    acc3_k1 = af3_1*bf0_1 + af4_1*bf3_1 + af5_1*bf6_1
    acc4_k1 = af3_1*bf1_1 + af4_1*bf4_1 + af5_1*bf7_1
    acc5_k1 = af3_1*bf2_1 + af4_1*bf5_1 + af5_1*bf8_1
    acc6_k1 = af6_1*bf0_1 + af7_1*bf3_1 + af8_1*bf6_1
    acc7_k1 = af6_1*bf1_1 + af7_1*bf4_1 + af8_1*bf7_1
    acc8_k1 = af6_1*bf2_1 + af7_1*bf5_1 + af8_1*bf8_1
    
    c0_f1 = fast_mod7(acc0_k1)
    c1_f1 = fast_mod7(acc1_k1)
    c2_f1 = fast_mod7(acc2_k1)
    c3_f1 = fast_mod7(acc3_k1)
    c4_f1 = fast_mod7(acc4_k1)
    c5_f1 = fast_mod7(acc5_k1)
    c6_f1 = fast_mod7(acc6_k1)
    c7_f1 = fast_mod7(acc7_k1)
    c8_f1 = fast_mod7(acc8_k1)
    
    # k=2: frequency component 2
    acc0_k2 = af0_2*bf0_2 + af1_2*bf3_2 + af2_2*bf6_2
    acc1_k2 = af0_2*bf1_2 + af1_2*bf4_2 + af2_2*bf7_2
    acc2_k2 = af0_2*bf2_2 + af1_2*bf5_2 + af2_2*bf8_2
    acc3_k2 = af3_2*bf0_2 + af4_2*bf3_2 + af5_2*bf6_2
    acc4_k2 = af3_2*bf1_2 + af4_2*bf4_2 + af5_2*bf7_2
    acc5_k2 = af3_2*bf2_2 + af4_2*bf5_2 + af5_2*bf8_2
    acc6_k2 = af6_2*bf0_2 + af7_2*bf3_2 + af8_2*bf6_2
    acc7_k2 = af6_2*bf1_2 + af7_2*bf4_2 + af8_2*bf7_2
    acc8_k2 = af6_2*bf2_2 + af7_2*bf5_2 + af8_2*bf8_2
    
    c0_f2 = fast_mod7(acc0_k2)
    c1_f2 = fast_mod7(acc1_k2)
    c2_f2 = fast_mod7(acc2_k2)
    c3_f2 = fast_mod7(acc3_k2)
    c4_f2 = fast_mod7(acc4_k2)
    c5_f2 = fast_mod7(acc5_k2)
    c6_f2 = fast_mod7(acc6_k2)
    c7_f2 = fast_mod7(acc7_k2)
    c8_f2 = fast_mod7(acc8_k2)
    
    # k=3: frequency component 3
    acc0_k3 = af0_3*bf0_3 + af1_3*bf3_3 + af2_3*bf6_3
    acc1_k3 = af0_3*bf1_3 + af1_3*bf4_3 + af2_3*bf7_3
    acc2_k3 = af0_3*bf2_3 + af1_3*bf5_3 + af2_3*bf8_3
    acc3_k3 = af3_3*bf0_3 + af4_3*bf3_3 + af5_3*bf6_3
    acc4_k3 = af3_3*bf1_3 + af4_3*bf4_3 + af5_3*bf7_3
    acc5_k3 = af3_3*bf2_3 + af4_3*bf5_3 + af5_3*bf8_3
    acc6_k3 = af6_3*bf0_3 + af7_3*bf3_3 + af8_3*bf6_3
    acc7_k3 = af6_3*bf1_3 + af7_3*bf4_3 + af8_3*bf7_3
    acc8_k3 = af6_3*bf2_3 + af7_3*bf5_3 + af8_3*bf8_3
    
    c0_f3 = fast_mod7(acc0_k3)
    c1_f3 = fast_mod7(acc1_k3)
    c2_f3 = fast_mod7(acc2_k3)
    c3_f3 = fast_mod7(acc3_k3)
    c4_f3 = fast_mod7(acc4_k3)
    c5_f3 = fast_mod7(acc5_k3)
    c6_f3 = fast_mod7(acc6_k3)
    c7_f3 = fast_mod7(acc7_k3)
    c8_f3 = fast_mod7(acc8_k3)
    
    # k=4: frequency component 4
    acc0_k4 = af0_4*bf0_4 + af1_4*bf3_4 + af2_4*bf6_4
    acc1_k4 = af0_4*bf1_4 + af1_4*bf4_4 + af2_4*bf7_4
    acc2_k4 = af0_4*bf2_4 + af1_4*bf5_4 + af2_4*bf8_4
    acc3_k4 = af3_4*bf0_4 + af4_4*bf3_4 + af5_4*bf6_4
    acc4_k4 = af3_4*bf1_4 + af4_4*bf4_4 + af5_4*bf7_4
    acc5_k4 = af3_4*bf2_4 + af4_4*bf5_4 + af5_4*bf8_4
    acc6_k4 = af6_4*bf0_4 + af7_4*bf3_4 + af8_4*bf6_4
    acc7_k4 = af6_4*bf1_4 + af7_4*bf4_4 + af8_4*bf7_4
    acc8_k4 = af6_4*bf2_4 + af7_4*bf5_4 + af8_4*bf8_4
    
    c0_f4 = fast_mod7(acc0_k4)
    c1_f4 = fast_mod7(acc1_k4)
    c2_f4 = fast_mod7(acc2_k4)
    c3_f4 = fast_mod7(acc3_k4)
    c4_f4 = fast_mod7(acc4_k4)
    c5_f4 = fast_mod7(acc5_k4)
    c6_f4 = fast_mod7(acc6_k4)
    c7_f4 = fast_mod7(acc7_k4)
    c8_f4 = fast_mod7(acc8_k4)
    
    # k=5: frequency component 5
    acc0_k5 = af0_5*bf0_5 + af1_5*bf3_5 + af2_5*bf6_5
    acc1_k5 = af0_5*bf1_5 + af1_5*bf4_5 + af2_5*bf7_5
    acc2_k5 = af0_5*bf2_5 + af1_5*bf5_5 + af2_5*bf8_5
    acc3_k5 = af3_5*bf0_5 + af4_5*bf3_5 + af5_5*bf6_5
    acc4_k5 = af3_5*bf1_5 + af4_5*bf4_5 + af5_5*bf7_5
    acc5_k5 = af3_5*bf2_5 + af4_5*bf5_5 + af5_5*bf8_5
    acc6_k5 = af6_5*bf0_5 + af7_5*bf3_5 + af8_5*bf6_5
    acc7_k5 = af6_5*bf1_5 + af7_5*bf4_5 + af8_5*bf7_5
    acc8_k5 = af6_5*bf2_5 + af7_5*bf5_5 + af8_5*bf8_5
    
    c0_f5 = fast_mod7(acc0_k5)
    c1_f5 = fast_mod7(acc1_k5)
    c2_f5 = fast_mod7(acc2_k5)
    c3_f5 = fast_mod7(acc3_k5)
    c4_f5 = fast_mod7(acc4_k5)
    c5_f5 = fast_mod7(acc5_k5)
    c6_f5 = fast_mod7(acc6_k5)
    c7_f5 = fast_mod7(acc7_k5)
    c8_f5 = fast_mod7(acc8_k5)

    # --- 4. IDFT & REPACK ---
    # We now have the spectrum for C. We must inverse transform and repack.
    # Each cell c0..c8 has 6 frequency components, we IDFT each and pack.
    
    c0_c0, c0_c1, c0_c2, c0_c3, c0_c4, c0_c5 = idft_6_point(c0_f0, c0_f1, c0_f2, c0_f3, c0_f4, c0_f5)
    res0 = pack_poly(c0_c0, c0_c1, c0_c2, c0_c3, c0_c4, c0_c5)
    
    c1_c0, c1_c1, c1_c2, c1_c3, c1_c4, c1_c5 = idft_6_point(c1_f0, c1_f1, c1_f2, c1_f3, c1_f4, c1_f5)
    res1 = pack_poly(c1_c0, c1_c1, c1_c2, c1_c3, c1_c4, c1_c5)
    
    c2_c0, c2_c1, c2_c2, c2_c3, c2_c4, c2_c5 = idft_6_point(c2_f0, c2_f1, c2_f2, c2_f3, c2_f4, c2_f5)
    res2 = pack_poly(c2_c0, c2_c1, c2_c2, c2_c3, c2_c4, c2_c5)
    
    c3_c0, c3_c1, c3_c2, c3_c3, c3_c4, c3_c5 = idft_6_point(c3_f0, c3_f1, c3_f2, c3_f3, c3_f4, c3_f5)
    res3 = pack_poly(c3_c0, c3_c1, c3_c2, c3_c3, c3_c4, c3_c5)
    
    c4_c0, c4_c1, c4_c2, c4_c3, c4_c4, c4_c5 = idft_6_point(c4_f0, c4_f1, c4_f2, c4_f3, c4_f4, c4_f5)
    res4 = pack_poly(c4_c0, c4_c1, c4_c2, c4_c3, c4_c4, c4_c5)
    
    c5_c0, c5_c1, c5_c2, c5_c3, c5_c4, c5_c5 = idft_6_point(c5_f0, c5_f1, c5_f2, c5_f3, c5_f4, c5_f5)
    res5 = pack_poly(c5_c0, c5_c1, c5_c2, c5_c3, c5_c4, c5_c5)
    
    c6_c0, c6_c1, c6_c2, c6_c3, c6_c4, c6_c5 = idft_6_point(c6_f0, c6_f1, c6_f2, c6_f3, c6_f4, c6_f5)
    res6 = pack_poly(c6_c0, c6_c1, c6_c2, c6_c3, c6_c4, c6_c5)
    
    c7_c0, c7_c1, c7_c2, c7_c3, c7_c4, c7_c5 = idft_6_point(c7_f0, c7_f1, c7_f2, c7_f3, c7_f4, c7_f5)
    res7 = pack_poly(c7_c0, c7_c1, c7_c2, c7_c3, c7_c4, c7_c5)
    
    c8_c0, c8_c1, c8_c2, c8_c3, c8_c4, c8_c5 = idft_6_point(c8_f0, c8_f1, c8_f2, c8_f3, c8_f4, c8_f5)
    res8 = pack_poly(c8_c0, c8_c1, c8_c2, c8_c3, c8_c4, c8_c5)

    # --- 5. STORE --- (SoA layout: coalesced writes)
    tl.store(C_ptr + 0 * n_matrices + offs, res0, mask=mask)
    tl.store(C_ptr + 1 * n_matrices + offs, res1, mask=mask)
    tl.store(C_ptr + 2 * n_matrices + offs, res2, mask=mask)
    tl.store(C_ptr + 3 * n_matrices + offs, res3, mask=mask)
    tl.store(C_ptr + 4 * n_matrices + offs, res4, mask=mask)
    tl.store(C_ptr + 5 * n_matrices + offs, res5, mask=mask)
    tl.store(C_ptr + 6 * n_matrices + offs, res6, mask=mask)
    tl.store(C_ptr + 7 * n_matrices + offs, res7, mask=mask)
    tl.store(C_ptr + 8 * n_matrices + offs, res8, mask=mask)

def ring_matmul(A, B):
    """
    Batched ring matrix multiplication with SoA layout.
    
    Args:
        A, B: int32 tensors of shape (9, Batch) - SoA layout for coalesced memory access.
              Each row i contains component i of all matrices in the batch.
    Returns:
        C: int32 tensor of shape (9, Batch), same SoA layout.
    """
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
    """Pack 6 coefficients (0-7) into one int32."""
    c0, c1, c2, c3, c4, c5 = coeffs
    return c0 | (c1 << 3) | (c2 << 6) | (c3 << 9) | (c4 << 12) | (c5 << 15)


def _unpack_poly(packed):
    """Unpack one int32 into 6 coefficients."""
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
    """DFT over F_7, W=3. c is list of 6 coeffs."""
    c0, c1, c2, c3, c4, c5 = c
    f0 = _mod7(c0 + c1 + c2 + c3 + c4 + c5)
    f1 = _mod7(c0*1 + c1*3 + c2*2 + c3*6 + c4*4 + c5*5)
    f2 = _mod7(c0*1 + c1*2 + c2*4 + c3*1 + c4*2 + c5*4)
    f3 = _mod7(c0*1 + c1*6 + c2*1 + c3*6 + c4*1 + c5*6)
    f4 = _mod7(c0*1 + c1*4 + c2*2 + c3*1 + c4*4 + c5*2)
    f5 = _mod7(c0*1 + c1*5 + c2*4 + c3*6 + c4*2 + c5*3)
    return [f0, f1, f2, f3, f4, f5]


def _idft_6_point(f):
    """Inverse DFT over F_7."""
    f0, f1, f2, f3, f4, f5 = f
    t0 = _mod7(f0 + f1 + f2 + f3 + f4 + f5)
    t1 = _mod7(f0*1 + f1*5 + f2*4 + f3*6 + f4*2 + f5*3)
    t2 = _mod7(f0*1 + f1*4 + f2*2 + f3*1 + f4*4 + f5*2)
    t3 = _mod7(f0*1 + f1*6 + f2*1 + f3*6 + f4*1 + f5*6)
    t4 = _mod7(f0*1 + f1*2 + f2*4 + f3*1 + f4*2 + f5*4)
    t5 = _mod7(f0*1 + f1*3 + f2*2 + f3*6 + f4*4 + f5*5)
    return [_mod7(t0*6), _mod7(t1*6), _mod7(t2*6), _mod7(t3*6), _mod7(t4*6), _mod7(t5*6)]


def _poly_mul_conv(a, b):
    """Multiply two polynomials (6 coeffs each) in F_7 via DFT convolution."""
    fa, fb = _dft_6_point(a), _dft_6_point(b)
    fc = [_mod7(fa[k] * fb[k]) for k in range(6)]
    return _idft_6_point(fc)


def ring_matmul_reference(A, B):
    """
    Reference implementation: batched 3x3 matrix multiply in the polynomial ring.
    A, B: (9, batch) int32 tensors (numpy or torch) - SoA layout.
    Returns (9, batch) int32.
    """
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
    return torch.from_numpy(C) if is_torch else C