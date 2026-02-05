import torch
import triton
import triton.language as tl

@triton.jit
def fast_mod7(x):
    """Barrett reduction x % 7. Same as triton7."""
    q = (x.to(tl.int64) * 74899) >> 19
    q = q.to(tl.int32)
    r = x - q * 7
    r = tl.where(r >= 7, r - 7, r)
    return r

@triton.jit
def unpack_poly_tensor(packed):
    """
    Unpacks a packed int32 tensor into a list of 6 coefficient tensors.
    """
    mask = 7
    # Use a static list comprehension. 
    # Triton unrolls this into 6 scalar operations in the IR.
    return [(packed >> (i * 3)) & mask for i in range(6)]

@triton.jit
def pack_poly_tensor(coeffs):
    """
    Packs a list of 6 coefficient tensors into one int32 tensor.
    """
    val = coeffs[0]
    for i in range(1, 6):
        val = val | (coeffs[i] << (i * 3))
    return val

@triton.jit
def dft_6_point_tensor(c):
    """
    Input: list of 6 tensors (coefficients).
    Output: list of 6 tensors (frequency domain).
    """
    # Powers of 3 mod 7 for n=0..5, k=0..5
    # We precompute the table in Python so it's hardcoded constants in PTX.
    # W[k][n] = 3^(n*k) mod 7
    W_pow = [
        [1, 1, 1, 1, 1, 1], # k=0
        [1, 3, 2, 6, 4, 5], # k=1
        [1, 2, 4, 1, 2, 4], # k=2
        [1, 6, 1, 6, 1, 6], # k=3
        [1, 4, 2, 1, 4, 2], # k=4
        [1, 5, 4, 6, 2, 3], # k=5
    ]
    
    freqs = []
    for k in range(6):
        # Accumulate in a temporary variable
        # logic: sum(c[n] * W[k][n])
        acc = c[0] # W[k][0] is always 1
        for n in range(1, 6):
            w = W_pow[k][n]
            # The compiler will fold `c[n] * 1` to just `c[n]` automatically.
            acc = acc + c[n] * w
        freqs.append(fast_mod7(acc))
    return freqs

@triton.jit
def idft_6_point_tensor(f):
    """
    Inverse DFT. W^-1 = 5. Scaling factor N^-1 = 6.
    """
    # Powers of 5 mod 7
    # W_inv[k][n] = 5^(n*k) mod 7
    W_inv = [
        [1, 1, 1, 1, 1, 1],
        [1, 5, 4, 6, 2, 3],
        [1, 4, 2, 1, 4, 2],
        [1, 6, 1, 6, 1, 6],
        [1, 2, 4, 1, 2, 4],
        [1, 3, 2, 6, 4, 5],
    ]
    
    coeffs = []
    for k in range(6):
        acc = f[0]
        for n in range(1, 6):
            w = W_inv[k][n]
            acc = acc + f[n] * w
        # Apply scaling factor 6 at the end
        coeffs.append(fast_mod7(acc * 6))
    return coeffs

@triton.jit
def ring_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    n_matrices, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_matrices

    # --- 1. LOAD & UNPACK & DFT ---
    # We use list comprehensions to handle the 9 matrix entries.
    
    # Structure: A_freq[entry_index][freq_index]
    A_freq = []
    B_freq = []
    
    for i in range(9):
        # Load A[i]
        a_val = tl.load(A_ptr + i * n_matrices + offs, mask=mask, other=0)
        a_coeffs = unpack_poly_tensor(a_val)
        A_freq.append(dft_6_point_tensor(a_coeffs))

        # Load B[i]
        b_val = tl.load(B_ptr + i * n_matrices + offs, mask=mask, other=0)
        b_coeffs = unpack_poly_tensor(b_val)
        B_freq.append(dft_6_point_tensor(b_coeffs))

    # --- 2. MATMUL IN FREQUENCY DOMAIN ---
    # C[row, col] = sum(A[row, k] * B[k, col])
    # We compute this for each frequency component f=0..5 independently.
    
    # Initialize C_freq structure: list of 9 entries, each a list of 6 freq tensors
    C_freq = [[None for _ in range(6)] for _ in range(9)]
    
    for f in range(6): # Iterate over frequency components
        for row in range(3):
            for col in range(3):
                # Dot product for C[row, col] at frequency f
                acc = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
                for k in range(3):
                    a_idx = row * 3 + k
                    b_idx = k * 3 + col
                    
                    # Pointwise multiply: A_freq[a_idx][f] * B_freq[b_idx][f]
                    prod = A_freq[a_idx][f] * B_freq[b_idx][f]
                    acc = acc + prod
                
                # Store result for this freq
                c_idx = row * 3 + col
                C_freq[c_idx][f] = fast_mod7(acc)

    # --- 3. IDFT & REPACK & STORE ---
    for i in range(9):
        # Transpose C_freq to get list of [f0, f1... f5] for this entry
        freqs = C_freq[i] 
        
        # Inverse Transform
        coeffs = idft_6_point_tensor(freqs)
        
        # Pack
        res = pack_poly_tensor(coeffs)
        
        # Store
        tl.store(C_ptr + i * n_matrices + offs, res, mask=mask)

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