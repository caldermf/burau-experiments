import torch

# Implements the same ring matmul as `triton7.py`, but using vectorized torch ops.
#
# - Each matrix entry is a polynomial in F_7[x]/(x^6 - 1).
# - Each polynomial is stored packed in one int32: 6 coefficients, 3 bits each.
# - Multiplication uses a 6-point DFT over F_7 with W=3 (a primitive 6th root),
#   then pointwise multiply, then IDFT (W^-1 = 5) and scaling by 6 (= 6^-1 mod 7).


_SHIFTS = torch.tensor([0, 3, 6, 9, 12, 15], dtype=torch.int64)
_MASK3 = 7


def _mod7(x: torch.Tensor) -> torch.Tensor:
    # torch.remainder matches Python % for non-negative values; we only operate on non-negative ints here.
    return torch.remainder(x, 7)


def unpack_poly(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack packed int32 polynomials into coefficients.

    Args:
        packed: int tensor (...,) with packed 6x3-bit coefficients.
    Returns:
        coeffs: int64 tensor (..., 6) with coefficients in [0, 7].
    """
    if packed.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"packed must be int32/int64, got {packed.dtype}")
    shifts = _SHIFTS.to(device=packed.device)
    x = packed.to(torch.int64)[..., None]  # (..., 1)
    coeffs = (x >> shifts) & _MASK3
    return coeffs


def pack_poly(coeffs: torch.Tensor) -> torch.Tensor:
    """
    Pack coefficients (..., 6) into one int32.

    Args:
        coeffs: int tensor (..., 6). Only low 3 bits of each coeff are kept.
    Returns:
        packed: int32 tensor (...,)
    """
    if coeffs.shape[-1] != 6:
        raise ValueError(f"coeffs last dim must be 6, got {coeffs.shape}")
    shifts = _SHIFTS.to(device=coeffs.device)
    c = coeffs.to(torch.int64) & _MASK3
    packed = torch.sum(c << shifts, dim=-1)
    return packed.to(torch.int32)


def _dft_matrix_f7(device, dtype=torch.int64) -> torch.Tensor:
    """
    6x6 DFT matrix over F_7 with W=3: M[k,n] = W^(n*k) mod 7.
    """
    W = 3
    ks = torch.arange(6, device=device, dtype=torch.int64)[:, None]  # (6,1)
    ns = torch.arange(6, device=device, dtype=torch.int64)[None, :]  # (1,6)
    exps = ks * ns  # (6,6)
    # pow in python ints then tensorize is fine (tiny constant), but do it on-device for simplicity
    M = torch.empty((6, 6), device=device, dtype=torch.int64)
    for k in range(6):
        for n in range(6):
            M[k, n] = pow(W, int(exps[k, n].item()), 7)
    return M.to(dtype=dtype)


def _idft_matrix_f7(device, dtype=torch.int64) -> torch.Tensor:
    """
    6x6 IDFT matrix over F_7 with W^-1=5 and scaling by 6:
      c[n] = 6 * sum_k f[k] * (W^-1)^(n*k) mod 7
    So matrix N[n,k] = 6 * (5^(n*k)) mod 7.
    """
    Winv = 5
    scale = 6
    ns = torch.arange(6, device=device, dtype=torch.int64)[:, None]  # (6,1)
    ks = torch.arange(6, device=device, dtype=torch.int64)[None, :]  # (1,6)
    exps = ns * ks  # (6,6)
    N = torch.empty((6, 6), device=device, dtype=torch.int64)
    for n in range(6):
        for k in range(6):
            N[n, k] = (scale * pow(Winv, int(exps[n, k].item()), 7)) % 7
    return N.to(dtype=dtype)


def dft_6_point(coeffs: torch.Tensor) -> torch.Tensor:
    """
    DFT over F_7 for last dim size 6.

    Args:
        coeffs: (..., 6) int tensor
    Returns:
        freqs: (..., 6) int64 tensor (values 0..6)
    """
    if coeffs.shape[-1] != 6:
        raise ValueError(f"coeffs last dim must be 6, got {coeffs.shape}")
    M = _dft_matrix_f7(device=coeffs.device, dtype=torch.int64)  # (6,6)
    x = coeffs.to(torch.int64)
    # (...,6) = (...,6) @ (6,6)^T where rows are k and cols are n
    freqs = x @ M.T
    return _mod7(freqs)


def idft_6_point(freqs: torch.Tensor) -> torch.Tensor:
    """
    IDFT over F_7 for last dim size 6.

    Args:
        freqs: (..., 6) int tensor
    Returns:
        coeffs: (..., 6) int64 tensor (values 0..6)
    """
    if freqs.shape[-1] != 6:
        raise ValueError(f"freqs last dim must be 6, got {freqs.shape}")
    N = _idft_matrix_f7(device=freqs.device, dtype=torch.int64)  # (6,6)
    f = freqs.to(torch.int64)
    coeffs = f @ N.T  # (...,6)
    return _mod7(coeffs)


def ring_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Torch implementation of the same batched ring matmul as `triton7.ring_matmul`.

    Args:
        A, B: int32 tensors of shape (batch, 9). Each row is a flattened 3x3 matrix
              of packed polynomials (6 coeffs, 3 bits each).
    Returns:
        C: int32 tensor of shape (batch, 9), same packed format.
    """
    if A.dtype != torch.int32 or B.dtype != torch.int32:
        raise TypeError(f"A,B must be int32, got {A.dtype}, {B.dtype}")
    if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape or A.shape[1] != 9:
        raise ValueError(f"A,B must be shape (batch, 9) and equal; got {A.shape}, {B.shape}")

    batch = A.shape[0]

    # Unpack: (batch, 9) -> (batch, 9, 6)
    A_c = unpack_poly(A)  # int64
    B_c = unpack_poly(B)

    # DFT along polynomial axis: (batch, 9, 6)
    A_f = dft_6_point(A_c)
    B_f = dft_6_point(B_c)

    # Reshape to matrices per frequency:
    # (batch, 9, 6) -> (batch, 3, 3, 6)
    A_f = A_f.view(batch, 3, 3, 6)
    B_f = B_f.view(batch, 3, 3, 6)

    # Do 6 independent 3x3 matmuls in F_7.
    # Move frequency to batch dimension to use torch.matmul efficiently:
    # (batch, 3, 3, 6) -> (6, batch, 3, 3)
    A_fk = A_f.permute(3, 0, 1, 2).contiguous()
    B_fk = B_f.permute(3, 0, 1, 2).contiguous()

    C_fk = torch.matmul(A_fk.to(torch.int64), B_fk.to(torch.int64))
    C_fk = _mod7(C_fk)  # (6, batch, 3, 3)

    # Back to (batch, 3, 3, 6)
    C_f = C_fk.permute(1, 2, 3, 0).contiguous()
    C_f = C_f.view(batch, 9, 6)

    # IDFT and pack back to int32
    C_c = idft_6_point(C_f)
    C = pack_poly(C_c)
    return C

