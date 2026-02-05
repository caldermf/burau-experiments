import torch
import triton
import triton.language as tl

# Same math as triton7: F_7[x]/(x^6-1), SoA layout (9, Batch).
# No list comprehensions or list indexing; everything explicit for Triton.


@triton.jit
def fast_mod7(x):
    """Barrett reduction x % 7. Same as triton7."""
    q = (x.to(tl.int64) * 74899) >> 19
    q = q.to(tl.int32)
    r = x - q * 7
    r = tl.where(r >= 7, r - 7, r)
    return r


@triton.jit
def unpack_poly(packed):
    """Unpack int32 into 6 coefficients. Returns tuple (c0..c5). Same as triton7."""
    mask = 7
    c0 = packed & mask
    c1 = (packed >> 3) & mask
    c2 = (packed >> 6) & mask
    c3 = (packed >> 9) & mask
    c4 = (packed >> 12) & mask
    c5 = (packed >> 15) & mask
    return c0, c1, c2, c3, c4, c5


@triton.jit
def pack_poly(c0, c1, c2, c3, c4, c5):
    """Pack 6 coefficients into int32. Same as triton7."""
    val = c0
    val = val | (c1 << 3)
    val = val | (c2 << 6)
    val = val | (c3 << 9)
    val = val | (c4 << 12)
    val = val | (c5 << 15)
    return val


@triton.jit
def dft_6_point(c0, c1, c2, c3, c4, c5):
    """DFT over F_7, W=3. Same formulas as triton7."""
    f0 = fast_mod7(c0 + c1 + c2 + c3 + c4 + c5)
    f1 = fast_mod7(c0*1 + c1*3 + c2*2 + c3*6 + c4*4 + c5*5)
    f2 = fast_mod7(c0*1 + c1*2 + c2*4 + c3*1 + c4*2 + c5*4)
    f3 = fast_mod7(c0*1 + c1*6 + c2*1 + c3*6 + c4*1 + c5*6)
    f4 = fast_mod7(c0*1 + c1*4 + c2*2 + c3*1 + c4*4 + c5*2)
    f5 = fast_mod7(c0*1 + c1*5 + c2*4 + c3*6 + c4*2 + c5*3)
    return f0, f1, f2, f3, f4, f5


@triton.jit
def idft_6_point(f0, f1, f2, f3, f4, f5):
    """Inverse DFT, W^-1=5, scale 6. Same as triton7."""
    t0 = fast_mod7(f0 + f1 + f2 + f3 + f4 + f5)
    t1 = fast_mod7(f0*1 + f1*5 + f2*4 + f3*6 + f4*2 + f5*3)
    t2 = fast_mod7(f0*1 + f1*4 + f2*2 + f3*1 + f4*4 + f5*2)
    t3 = fast_mod7(f0*1 + f1*6 + f2*1 + f3*6 + f4*1 + f5*6)
    t4 = fast_mod7(f0*1 + f1*2 + f2*4 + f3*1 + f4*2 + f5*4)
    t5 = fast_mod7(f0*1 + f1*3 + f2*2 + f3*6 + f4*4 + f5*5)
    return fast_mod7(t0*6), fast_mod7(t1*6), fast_mod7(t2*6), fast_mod7(t3*6), fast_mod7(t4*6), fast_mod7(t5*6)


@triton.jit
def ring_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    n_matrices, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_matrices

    # --- 1. LOAD (SoA) ---
    a0 = tl.load(A_ptr + 0 * n_matrices + offs, mask=mask, other=0)
    a1 = tl.load(A_ptr + 1 * n_matrices + offs, mask=mask, other=0)
    a2 = tl.load(A_ptr + 2 * n_matrices + offs, mask=mask, other=0)
    a3 = tl.load(A_ptr + 3 * n_matrices + offs, mask=mask, other=0)
    a4 = tl.load(A_ptr + 4 * n_matrices + offs, mask=mask, other=0)
    a5 = tl.load(A_ptr + 5 * n_matrices + offs, mask=mask, other=0)
    a6 = tl.load(A_ptr + 6 * n_matrices + offs, mask=mask, other=0)
    a7 = tl.load(A_ptr + 7 * n_matrices + offs, mask=mask, other=0)
    a8 = tl.load(A_ptr + 8 * n_matrices + offs, mask=mask, other=0)
    b0 = tl.load(B_ptr + 0 * n_matrices + offs, mask=mask, other=0)
    b1 = tl.load(B_ptr + 1 * n_matrices + offs, mask=mask, other=0)
    b2 = tl.load(B_ptr + 2 * n_matrices + offs, mask=mask, other=0)
    b3 = tl.load(B_ptr + 3 * n_matrices + offs, mask=mask, other=0)
    b4 = tl.load(B_ptr + 4 * n_matrices + offs, mask=mask, other=0)
    b5 = tl.load(B_ptr + 5 * n_matrices + offs, mask=mask, other=0)
    b6 = tl.load(B_ptr + 6 * n_matrices + offs, mask=mask, other=0)
    b7 = tl.load(B_ptr + 7 * n_matrices + offs, mask=mask, other=0)
    b8 = tl.load(B_ptr + 8 * n_matrices + offs, mask=mask, other=0)

    # --- 2. UNPACK & DFT for each of 9 entries ---
    # A entries 0..8 -> af0_0..af0_5, af1_0..af1_5, ..., af8_0..af8_5
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

    # --- 3. FREQUENCY-DOMAIN 3x3 MATMUL (6 freqs) ---
    # C00 = A00*B00 + A01*B10 + A02*B20, etc. Entry index: row*3+col -> 0..8
    # freq 0
    c0_f0 = fast_mod7(af0_0*bf0_0 + af1_0*bf3_0 + af2_0*bf6_0)
    c1_f0 = fast_mod7(af0_0*bf1_0 + af1_0*bf4_0 + af2_0*bf7_0)
    c2_f0 = fast_mod7(af0_0*bf2_0 + af1_0*bf5_0 + af2_0*bf8_0)
    c3_f0 = fast_mod7(af3_0*bf0_0 + af4_0*bf3_0 + af5_0*bf6_0)
    c4_f0 = fast_mod7(af3_0*bf1_0 + af4_0*bf4_0 + af5_0*bf7_0)
    c5_f0 = fast_mod7(af3_0*bf2_0 + af4_0*bf5_0 + af5_0*bf8_0)
    c6_f0 = fast_mod7(af6_0*bf0_0 + af7_0*bf3_0 + af8_0*bf6_0)
    c7_f0 = fast_mod7(af6_0*bf1_0 + af7_0*bf4_0 + af8_0*bf7_0)
    c8_f0 = fast_mod7(af6_0*bf2_0 + af7_0*bf5_0 + af8_0*bf8_0)
    # freq 1
    c0_f1 = fast_mod7(af0_1*bf0_1 + af1_1*bf3_1 + af2_1*bf6_1)
    c1_f1 = fast_mod7(af0_1*bf1_1 + af1_1*bf4_1 + af2_1*bf7_1)
    c2_f1 = fast_mod7(af0_1*bf2_1 + af1_1*bf5_1 + af2_1*bf8_1)
    c3_f1 = fast_mod7(af3_1*bf0_1 + af4_1*bf3_1 + af5_1*bf6_1)
    c4_f1 = fast_mod7(af3_1*bf1_1 + af4_1*bf4_1 + af5_1*bf7_1)
    c5_f1 = fast_mod7(af3_1*bf2_1 + af4_1*bf5_1 + af5_1*bf8_1)
    c6_f1 = fast_mod7(af6_1*bf0_1 + af7_1*bf3_1 + af8_1*bf6_1)
    c7_f1 = fast_mod7(af6_1*bf1_1 + af7_1*bf4_1 + af8_1*bf7_1)
    c8_f1 = fast_mod7(af6_1*bf2_1 + af7_1*bf5_1 + af8_1*bf8_1)
    # freq 2
    c0_f2 = fast_mod7(af0_2*bf0_2 + af1_2*bf3_2 + af2_2*bf6_2)
    c1_f2 = fast_mod7(af0_2*bf1_2 + af1_2*bf4_2 + af2_2*bf7_2)
    c2_f2 = fast_mod7(af0_2*bf2_2 + af1_2*bf5_2 + af2_2*bf8_2)
    c3_f2 = fast_mod7(af3_2*bf0_2 + af4_2*bf3_2 + af5_2*bf6_2)
    c4_f2 = fast_mod7(af3_2*bf1_2 + af4_2*bf4_2 + af5_2*bf7_2)
    c5_f2 = fast_mod7(af3_2*bf2_2 + af4_2*bf5_2 + af5_2*bf8_2)
    c6_f2 = fast_mod7(af6_2*bf0_2 + af7_2*bf3_2 + af8_2*bf6_2)
    c7_f2 = fast_mod7(af6_2*bf1_2 + af7_2*bf4_2 + af8_2*bf7_2)
    c8_f2 = fast_mod7(af6_2*bf2_2 + af7_2*bf5_2 + af8_2*bf8_2)
    # freq 3
    c0_f3 = fast_mod7(af0_3*bf0_3 + af1_3*bf3_3 + af2_3*bf6_3)
    c1_f3 = fast_mod7(af0_3*bf1_3 + af1_3*bf4_3 + af2_3*bf7_3)
    c2_f3 = fast_mod7(af0_3*bf2_3 + af1_3*bf5_3 + af2_3*bf8_3)
    c3_f3 = fast_mod7(af3_3*bf0_3 + af4_3*bf3_3 + af5_3*bf6_3)
    c4_f3 = fast_mod7(af3_3*bf1_3 + af4_3*bf4_3 + af5_3*bf7_3)
    c5_f3 = fast_mod7(af3_3*bf2_3 + af4_3*bf5_3 + af5_3*bf8_3)
    c6_f3 = fast_mod7(af6_3*bf0_3 + af7_3*bf3_3 + af8_3*bf6_3)
    c7_f3 = fast_mod7(af6_3*bf1_3 + af7_3*bf4_3 + af8_3*bf7_3)
    c8_f3 = fast_mod7(af6_3*bf2_3 + af7_3*bf5_3 + af8_3*bf8_3)
    # freq 4
    c0_f4 = fast_mod7(af0_4*bf0_4 + af1_4*bf3_4 + af2_4*bf6_4)
    c1_f4 = fast_mod7(af0_4*bf1_4 + af1_4*bf4_4 + af2_4*bf7_4)
    c2_f4 = fast_mod7(af0_4*bf2_4 + af1_4*bf5_4 + af2_4*bf8_4)
    c3_f4 = fast_mod7(af3_4*bf0_4 + af4_4*bf3_4 + af5_4*bf6_4)
    c4_f4 = fast_mod7(af3_4*bf1_4 + af4_4*bf4_4 + af5_4*bf7_4)
    c5_f4 = fast_mod7(af3_4*bf2_4 + af4_4*bf5_4 + af5_4*bf8_4)
    c6_f4 = fast_mod7(af6_4*bf0_4 + af7_4*bf3_4 + af8_4*bf6_4)
    c7_f4 = fast_mod7(af6_4*bf1_4 + af7_4*bf4_4 + af8_4*bf7_4)
    c8_f4 = fast_mod7(af6_4*bf2_4 + af7_4*bf5_4 + af8_4*bf8_4)
    # freq 5
    c0_f5 = fast_mod7(af0_5*bf0_5 + af1_5*bf3_5 + af2_5*bf6_5)
    c1_f5 = fast_mod7(af0_5*bf1_5 + af1_5*bf4_5 + af2_5*bf7_5)
    c2_f5 = fast_mod7(af0_5*bf2_5 + af1_5*bf5_5 + af2_5*bf8_5)
    c3_f5 = fast_mod7(af3_5*bf0_5 + af4_5*bf3_5 + af5_5*bf6_5)
    c4_f5 = fast_mod7(af3_5*bf1_5 + af4_5*bf4_5 + af5_5*bf7_5)
    c5_f5 = fast_mod7(af3_5*bf2_5 + af4_5*bf5_5 + af5_5*bf8_5)
    c6_f5 = fast_mod7(af6_5*bf0_5 + af7_5*bf3_5 + af8_5*bf6_5)
    c7_f5 = fast_mod7(af6_5*bf1_5 + af7_5*bf4_5 + af8_5*bf7_5)
    c8_f5 = fast_mod7(af6_5*bf2_5 + af7_5*bf5_5 + af8_5*bf8_5)

    # --- 4. IDFT & PACK & STORE ---
    r0_c0, r0_c1, r0_c2, r0_c3, r0_c4, r0_c5 = idft_6_point(c0_f0, c0_f1, c0_f2, c0_f3, c0_f4, c0_f5)
    res0 = pack_poly(r0_c0, r0_c1, r0_c2, r0_c3, r0_c4, r0_c5)
    r1_c0, r1_c1, r1_c2, r1_c3, r1_c4, r1_c5 = idft_6_point(c1_f0, c1_f1, c1_f2, c1_f3, c1_f4, c1_f5)
    res1 = pack_poly(r1_c0, r1_c1, r1_c2, r1_c3, r1_c4, r1_c5)
    r2_c0, r2_c1, r2_c2, r2_c3, r2_c4, r2_c5 = idft_6_point(c2_f0, c2_f1, c2_f2, c2_f3, c2_f4, c2_f5)
    res2 = pack_poly(r2_c0, r2_c1, r2_c2, r2_c3, r2_c4, r2_c5)
    r3_c0, r3_c1, r3_c2, r3_c3, r3_c4, r3_c5 = idft_6_point(c3_f0, c3_f1, c3_f2, c3_f3, c3_f4, c3_f5)
    res3 = pack_poly(r3_c0, r3_c1, r3_c2, r3_c3, r3_c4, r3_c5)
    r4_c0, r4_c1, r4_c2, r4_c3, r4_c4, r4_c5 = idft_6_point(c4_f0, c4_f1, c4_f2, c4_f3, c4_f4, c4_f5)
    res4 = pack_poly(r4_c0, r4_c1, r4_c2, r4_c3, r4_c4, r4_c5)
    r5_c0, r5_c1, r5_c2, r5_c3, r5_c4, r5_c5 = idft_6_point(c5_f0, c5_f1, c5_f2, c5_f3, c5_f4, c5_f5)
    res5 = pack_poly(r5_c0, r5_c1, r5_c2, r5_c3, r5_c4, r5_c5)
    r6_c0, r6_c1, r6_c2, r6_c3, r6_c4, r6_c5 = idft_6_point(c6_f0, c6_f1, c6_f2, c6_f3, c6_f4, c6_f5)
    res6 = pack_poly(r6_c0, r6_c1, r6_c2, r6_c3, r6_c4, r6_c5)
    r7_c0, r7_c1, r7_c2, r7_c3, r7_c4, r7_c5 = idft_6_point(c7_f0, c7_f1, c7_f2, c7_f3, c7_f4, c7_f5)
    res7 = pack_poly(r7_c0, r7_c1, r7_c2, r7_c3, r7_c4, r7_c5)
    r8_c0, r8_c1, r8_c2, r8_c3, r8_c4, r8_c5 = idft_6_point(c8_f0, c8_f1, c8_f2, c8_f3, c8_f4, c8_f5)
    res8 = pack_poly(r8_c0, r8_c1, r8_c2, r8_c3, r8_c4, r8_c5)

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
    Batched ring matrix multiplication over F_7[x]/(x^6-1). Same API as triton7.ring_matmul.

    Args:
        A, B: int32 tensors of shape (9, Batch) - SoA layout.
    Returns:
        C: int32 tensor of shape (9, Batch).
    """
    assert A.shape[0] == 9 and B.shape[0] == 9, f"Expected (9, Batch), got A={A.shape}, B={B.shape}"
    batch_size = A.shape[1]
    C = torch.empty_like(A)
    grid = lambda meta: (triton.cdiv(batch_size, meta['BLOCK_SIZE']),)
    ring_matmul_kernel[grid](A, B, C, n_matrices=batch_size, BLOCK_SIZE=128)
    return C
