import torch
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
    M_stride, # Stride between matrices (9 ints)
    n_matrices,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Each lane handles ONE matrix multiplication
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_matrices

    # --- 1. LOAD DATA ---
    # We load 3x3 matrices. Flattened to 9 elements.
    # We need 9 pointers for A and 9 for B.
    # This looks verbose, but we want these in registers, not a tensor loop.
    
    # Offsets for each cell in the 3x3 matrix
    # A is (Batch, 9)
    base_A = A_ptr + offs * 9
    base_B = B_ptr + offs * 9
    
    # Load A (9 packed polys)
    a0 = tl.load(base_A + 0, mask=mask, other=0); a1 = tl.load(base_A + 1, mask=mask, other=0); a2 = tl.load(base_A + 2, mask=mask, other=0)
    a3 = tl.load(base_A + 3, mask=mask, other=0); a4 = tl.load(base_A + 4, mask=mask, other=0); a5 = tl.load(base_A + 5, mask=mask, other=0)
    a6 = tl.load(base_A + 6, mask=mask, other=0); a7 = tl.load(base_A + 7, mask=mask, other=0); a8 = tl.load(base_A + 8, mask=mask, other=0)

    # Load B (9 packed polys)
    b0 = tl.load(base_B + 0, mask=mask, other=0); b1 = tl.load(base_B + 1, mask=mask, other=0); b2 = tl.load(base_B + 2, mask=mask, other=0)
    b3 = tl.load(base_B + 3, mask=mask, other=0); b4 = tl.load(base_B + 4, mask=mask, other=0); b5 = tl.load(base_B + 5, mask=mask, other=0)
    b6 = tl.load(base_B + 6, mask=mask, other=0); b7 = tl.load(base_B + 7, mask=mask, other=0); b8 = tl.load(base_B + 8, mask=mask, other=0)

    # --- 2. UNPACK & DFT ---
    # This is the heavy lifting. We convert 9 packed ints into 9 frequency vectors (tuples of 6).
    # We define a macro-like helper (since triton functions inline)
    
    # A Matrix in frequency domain
    # Each 'af_x' is a tuple of 6 values (the spectrum at that cell)
    af0 = dft_6_point(*unpack_poly(a0))
    af1 = dft_6_point(*unpack_poly(a1))
    af2 = dft_6_point(*unpack_poly(a2))
    af3 = dft_6_point(*unpack_poly(a3))
    af4 = dft_6_point(*unpack_poly(a4))
    af5 = dft_6_point(*unpack_poly(a5))
    af6 = dft_6_point(*unpack_poly(a6))
    af7 = dft_6_point(*unpack_poly(a7))
    af8 = dft_6_point(*unpack_poly(a8))

    # B Matrix in frequency domain
    bf0 = dft_6_point(*unpack_poly(b0))
    bf1 = dft_6_point(*unpack_poly(b1))
    bf2 = dft_6_point(*unpack_poly(b2))
    bf3 = dft_6_point(*unpack_poly(b3))
    bf4 = dft_6_point(*unpack_poly(b4))
    bf5 = dft_6_point(*unpack_poly(b5))
    bf6 = dft_6_point(*unpack_poly(b6))
    bf7 = dft_6_point(*unpack_poly(b7))
    bf8 = dft_6_point(*unpack_poly(b8))

    # --- 3. FREQUENCY DOMAIN MATMUL ---
    # We now have 6 independent 3x3 matmuls.
    # A = [[af0, af1, af2], [af3, af4, af5], [af6, af7, af8]]
    # B = [[bf0, bf1, bf2], ... ]
    # C = A @ B
    
    # We need to perform this for EACH of the 6 frequencies k=0..5.
    # To do this cleanly, we iterate k in 0..5.
    
    # We need storage for result C (9 cells, each 6 frequencies)
    # We'll build lists of frequencies for c0..c8
    c0_freqs = []; c1_freqs = []; c2_freqs = []
    c3_freqs = []; c4_freqs = []; c5_freqs = []
    c6_freqs = []; c7_freqs = []; c8_freqs = []

    for k in range(6):
        # Extract the k-th frequency component for the entire 3x3 matrix
        # A matrix at freq k
        Ak_00 = af0[k]; Ak_01 = af1[k]; Ak_02 = af2[k]
        Ak_10 = af3[k]; Ak_11 = af4[k]; Ak_12 = af5[k]
        Ak_20 = af6[k]; Ak_21 = af7[k]; Ak_22 = af8[k]
        
        # B matrix at freq k
        Bk_00 = bf0[k]; Bk_01 = bf1[k]; Bk_02 = bf2[k]
        Bk_10 = bf3[k]; Bk_11 = bf4[k]; Bk_12 = bf5[k]
        Bk_20 = bf6[k]; Bk_21 = bf7[k]; Bk_22 = bf8[k]

        # Standard 3x3 MatMul: Row 0
        # C00 = A00*B00 + A01*B10 + A02*B20
        # We accumulate and THEN mod 7.
        # Max val is 6*6 + 6*6 + 6*6 = 108. Safe for int32.
        
        acc0 = Ak_00*Bk_00 + Ak_01*Bk_10 + Ak_02*Bk_20
        c0_freqs.append(fast_mod7(acc0))
        
        acc1 = Ak_00*Bk_01 + Ak_01*Bk_11 + Ak_02*Bk_21
        c1_freqs.append(fast_mod7(acc1))
        
        acc2 = Ak_00*Bk_02 + Ak_01*Bk_12 + Ak_02*Bk_22
        c2_freqs.append(fast_mod7(acc2))

        # Row 1
        acc3 = Ak_10*Bk_00 + Ak_11*Bk_10 + Ak_12*Bk_20
        c3_freqs.append(fast_mod7(acc3))
        
        acc4 = Ak_10*Bk_01 + Ak_11*Bk_11 + Ak_12*Bk_21
        c4_freqs.append(fast_mod7(acc4))
        
        acc5 = Ak_10*Bk_02 + Ak_11*Bk_12 + Ak_12*Bk_22
        c5_freqs.append(fast_mod7(acc5))
        
        # Row 2
        acc6 = Ak_20*Bk_00 + Ak_21*Bk_10 + Ak_22*Bk_20
        c6_freqs.append(fast_mod7(acc6))
        
        acc7 = Ak_20*Bk_01 + Ak_21*Bk_11 + Ak_22*Bk_21
        c7_freqs.append(fast_mod7(acc7))
        
        acc8 = Ak_20*Bk_02 + Ak_21*Bk_12 + Ak_22*Bk_22
        c8_freqs.append(fast_mod7(acc8))

    # --- 4. IDFT & REPACK ---
    # We now have the spectrum for C. We must inverse transform and repack.
    
    # Helper lambda to pack from list of freqs
    def finish_cell(freq_list):
        # Unpack list to args
        f0, f1, f2, f3, f4, f5 = freq_list[0], freq_list[1], freq_list[2], freq_list[3], freq_list[4], freq_list[5]
        # IDFT
        coeffs = idft_6_point(f0, f1, f2, f3, f4, f5)
        # Pack
        return pack_poly(*coeffs)

    res0 = finish_cell(c0_freqs)
    res1 = finish_cell(c1_freqs)
    res2 = finish_cell(c2_freqs)
    res3 = finish_cell(c3_freqs)
    res4 = finish_cell(c4_freqs)
    res5 = finish_cell(c5_freqs)
    res6 = finish_cell(c6_freqs)
    res7 = finish_cell(c7_freqs)
    res8 = finish_cell(c8_freqs)

    # --- 5. STORE ---
    base_C = C_ptr + offs * 9
    tl.store(base_C + 0, res0, mask=mask)
    tl.store(base_C + 1, res1, mask=mask)
    tl.store(base_C + 2, res2, mask=mask)
    tl.store(base_C + 3, res3, mask=mask)
    tl.store(base_C + 4, res4, mask=mask)
    tl.store(base_C + 5, res5, mask=mask)
    tl.store(base_C + 6, res6, mask=mask)
    tl.store(base_C + 7, res7, mask=mask)
    tl.store(base_C + 8, res8, mask=mask)

def ring_matmul(A, B):
    # A and B are shape (Batch, 9) int32
    batch_size = A.shape[0]
    C = torch.empty_like(A)
    
    grid = lambda meta: (triton.cdiv(batch_size, meta['BLOCK_SIZE']),)
    
    ring_matmul_kernel[grid](
        A, B, C,
        M_stride=9,
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
    A, B: (batch, 9) int32 tensors (numpy or torch).
    Returns (batch, 9) int32.
    """
    import numpy as np
    is_torch = torch.is_tensor(A)
    if is_torch:
        A, B = A.cpu().numpy(), B.cpu().numpy()
    batch = A.shape[0]
    C = np.empty((batch, 9), dtype=np.int32)
    for b in range(batch):
        # 3x3 of packed polys
        a_flat = A[b]
        b_flat = B[b]
        a_mat = [a_flat[i:i+3] for i in (0, 3, 6)]
        b_mat = [b_flat[i:i+3] for i in (0, 3, 6)]
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
                C[b, i*3 + j] = c_mat[i][j]
    return torch.from_numpy(C) if is_torch else C