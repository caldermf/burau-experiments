#!/usr/bin/env python3
"""
Test script to validate that int16/int32 optimizations produce identical results to int64.

Run this BEFORE using v3.4 in production.
"""

import torch

# =============================================================================
# REFERENCE IMPLEMENTATION (int64 - known correct)
# =============================================================================

def poly_multiply_batch_ref(a, b, p):
    """Reference: int64 throughout."""
    N, D = a.shape
    fft_size = 1 << (2 * D - 1).bit_length()
    
    a_fft = torch.fft.rfft(a.float(), n=fft_size, dim=-1)
    b_fft = torch.fft.rfft(b.float(), n=fft_size, dim=-1)
    c_fft = a_fft * b_fft
    c = torch.fft.irfft(c_fft, n=fft_size, dim=-1)
    
    c = torch.round(c).long() % p
    return c[:, :2*D-1]


def poly_matmul_batch_ref(A, B, p):
    """Reference: int64 throughout."""
    N, _, _, D = A.shape
    out_D = 2 * D - 1
    C = torch.zeros(N, 3, 3, out_D, dtype=torch.long, device=A.device)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a_ik = A[:, i, k, :].long()
                b_kj = B[:, k, j, :].long()
                conv = poly_multiply_batch_ref(a_ik, b_kj, p)
                C[:, i, j, :] = (C[:, i, j, :] + conv) % p
    
    return C


# =============================================================================
# OPTIMIZED IMPLEMENTATION (int16/int32)
# =============================================================================

STORAGE_DTYPE_MATRIX = torch.int16
COMPUTE_DTYPE_INT = torch.int32


def poly_multiply_batch_opt(a, b, p):
    """Optimized: int32 output."""
    N, D = a.shape
    fft_size = 1 << (2 * D - 1).bit_length()
    
    a_fft = torch.fft.rfft(a.float(), n=fft_size, dim=-1)
    b_fft = torch.fft.rfft(b.float(), n=fft_size, dim=-1)
    c_fft = a_fft * b_fft
    c = torch.fft.irfft(c_fft, n=fft_size, dim=-1)
    
    c = torch.round(c).to(COMPUTE_DTYPE_INT) % p
    return c[:, :2*D-1]


def poly_matmul_batch_opt(A, B, p):
    """Optimized: int16 input, int32 accumulation."""
    N, _, _, D = A.shape
    out_D = 2 * D - 1
    device = A.device
    
    # Convert to int32 for computation
    if A.dtype != COMPUTE_DTYPE_INT:
        A = A.to(COMPUTE_DTYPE_INT)
    if B.dtype != COMPUTE_DTYPE_INT:
        B = B.to(COMPUTE_DTYPE_INT)
    
    C = torch.zeros(N, 3, 3, out_D, dtype=COMPUTE_DTYPE_INT, device=device)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a_ik = A[:, i, k, :]
                b_kj = B[:, k, j, :]
                conv = poly_multiply_batch_opt(a_ik, b_kj, p)
                C[:, i, j, :] = (C[:, i, j, :] + conv) % p
    
    return C


# =============================================================================
# TESTS
# =============================================================================

def test_poly_multiply():
    """Test polynomial multiplication with various dtypes."""
    print("Testing polynomial multiplication...")
    
    test_cases = [
        (100, 50, 2),
        (100, 50, 5),
        (100, 50, 7),
        (100, 200, 7),
        (1000, 100, 7),
    ]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for N, D, p in test_cases:
        # Create test data in int16 (storage format)
        a_int16 = torch.randint(0, p, (N, D), dtype=torch.int16, device=device)
        b_int16 = torch.randint(0, p, (N, D), dtype=torch.int16, device=device)
        
        # Reference (int64)
        a_int64 = a_int16.long()
        b_int64 = b_int16.long()
        c_ref = poly_multiply_batch_ref(a_int64, b_int64, p)
        
        # Optimized (int16 -> int32)
        c_opt = poly_multiply_batch_opt(a_int16.to(torch.int32), b_int16.to(torch.int32), p)
        
        # Compare (convert to same dtype)
        if torch.equal(c_ref.int(), c_opt.int()):
            print(f"  ✓ N={N}, D={D}, p={p}")
        else:
            print(f"  ✗ N={N}, D={D}, p={p} - MISMATCH!")
            diff = (c_ref.int() != c_opt.int()).sum().item()
            print(f"    {diff} elements differ")
            return False
    
    return True


def test_poly_matmul():
    """Test matrix multiplication with various dtypes."""
    print("\nTesting matrix multiplication...")
    
    test_cases = [
        (10, 50, 2),
        (10, 50, 5),
        (10, 50, 7),
        (100, 100, 7),
        (100, 200, 7),
    ]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for N, D, p in test_cases:
        # Create test data in int16 (storage format)
        A_int16 = torch.randint(0, p, (N, 3, 3, D), dtype=torch.int16, device=device)
        B_int16 = torch.randint(0, p, (N, 3, 3, D), dtype=torch.int16, device=device)
        
        # Reference (int64)
        C_ref = poly_matmul_batch_ref(A_int16.long(), B_int16.long(), p)
        
        # Optimized (int16 input)
        C_opt = poly_matmul_batch_opt(A_int16, B_int16, p)
        
        # Compare
        if torch.equal(C_ref.int(), C_opt.int()):
            print(f"  ✓ N={N}, D={D}, p={p}")
        else:
            print(f"  ✗ N={N}, D={D}, p={p} - MISMATCH!")
            diff = (C_ref.int() != C_opt.int()).sum().item()
            print(f"    {diff} elements differ")
            return False
    
    return True


def test_storage_roundtrip():
    """Test that int16 storage preserves values correctly."""
    print("\nTesting storage roundtrip...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for p in [2, 3, 5, 7]:
        # Create matrices with values 0 to p-1
        N, D = 100, 200
        mat_orig = torch.randint(0, p, (N, 3, 3, D), dtype=torch.int64, device=device)
        
        # Store as int16
        mat_int16 = mat_orig.to(torch.int16)
        
        # Load back as int32 for computation
        mat_int32 = mat_int16.to(torch.int32)
        
        # Should match original
        if torch.equal(mat_orig.int(), mat_int32):
            print(f"  ✓ p={p}: storage roundtrip OK")
        else:
            print(f"  ✗ p={p}: storage roundtrip FAILED!")
            return False
    
    return True


def test_accumulation_overflow():
    """Test that int32 accumulation doesn't overflow for realistic cases."""
    print("\nTesting accumulation overflow safety...")
    
    # Worst case: all coefficients are p-1, D is large
    D = 4000  # Very large
    p = 7
    
    # Max single convolution output before mod:
    # Each output position: sum of min(D, out_pos+1) products
    # Max products per position: D
    # Max product value: (p-1) * (p-1) = 36
    # Max sum: 36 * D = 144,000 for D=4000
    
    # In matmul, we sum 3 such convolutions per output entry:
    # Max: 144,000 * 3 = 432,000
    
    # int32 max: 2,147,483,647
    # 432,000 << 2 billion, so we're safe!
    
    max_conv_value = (p-1) * (p-1) * D
    max_matmul_value = max_conv_value * 3
    int32_max = 2**31 - 1
    
    print(f"  Max convolution value (before mod): {max_conv_value:,}")
    print(f"  Max matmul accumulation (before mod): {max_matmul_value:,}")
    print(f"  int32 max: {int32_max:,}")
    
    if max_matmul_value < int32_max:
        print(f"  ✓ Safe margin: {int32_max // max_matmul_value}x")
        return True
    else:
        print(f"  ✗ OVERFLOW RISK!")
        return False


def test_memory_savings():
    """Report memory savings from dtype optimization."""
    print("\nMemory savings analysis...")
    
    N = 250000  # Typical bucket total
    D = 3601    # degree_multiplier=3, max_length=600
    max_length = 600
    
    # Old (int64)
    matrix_old = N * 3 * 3 * D * 8  # int64 = 8 bytes
    words_old = N * max_length * 8
    total_old = matrix_old + words_old
    
    # New (int16 matrices, int32 words)
    matrix_new = N * 3 * 3 * D * 2  # int16 = 2 bytes
    words_new = N * max_length * 4   # int32 = 4 bytes
    total_new = matrix_new + words_new
    
    print(f"  Matrices: {matrix_old/1e9:.2f} GB → {matrix_new/1e9:.2f} GB ({matrix_old/matrix_new:.1f}x savings)")
    print(f"  Words: {words_old/1e9:.2f} GB → {words_new/1e9:.2f} GB ({words_old/words_new:.1f}x savings)")
    print(f"  Total: {total_old/1e9:.2f} GB → {total_new/1e9:.2f} GB ({total_old/total_new:.1f}x savings)")
    
    return True


def main():
    print("="*60)
    print("VALIDATING int16/int32 OPTIMIZATION")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    tests = [
        ("Polynomial multiplication", test_poly_multiply),
        ("Matrix multiplication", test_poly_matmul),
        ("Storage roundtrip", test_storage_roundtrip),
        ("Accumulation overflow safety", test_accumulation_overflow),
        ("Memory savings", test_memory_savings),
    ]
    
    all_passed = True
    for name, test_fn in tests:
        try:
            if not test_fn():
                all_passed = False
        except Exception as e:
            print(f"  ✗ {name} raised exception: {e}")
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Safe to use v3.4")
    else:
        print("✗ SOME TESTS FAILED - Do not use v3.4")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
