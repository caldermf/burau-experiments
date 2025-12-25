import torch
import triton
import triton.language as tl


@triton.jit
def laurent_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    A_min_deg, A_max_deg,
    B_min_deg, B_max_deg,
    C_min_deg, C_max_deg,
    batch_size,
    A_deg_stride, B_deg_stride, C_deg_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Multiply two batches of matrices over Z[v, v^{-1}].
    
    A has coefficients for degrees [A_min_deg, A_max_deg]
    B has coefficients for degrees [B_min_deg, B_max_deg]
    C will have coefficients for degrees [C_min_deg, C_max_deg]
    where C_min_deg = A_min_deg + B_min_deg and C_max_deg = A_max_deg + B_max_deg
    
    Each is stored as a tensor of shape (batch_size, num_degrees, 3, 3)
    Uses float32 for computation (coefficients should be integers stored as floats)
    """
    # 2D grid: pid_batch for batch index, pid_deg for degree index
    pid_batch = tl.program_id(0)
    pid_deg = tl.program_id(1)
    k = C_min_deg + pid_deg  # The degree we're computing
    
    # Initialize accumulator for the 3x3 matrix
    acc = tl.zeros((3, 3), dtype=tl.float32)
    
    # Compute the range of i values: i + j = k, where
    # A_min_deg <= i <= A_max_deg and B_min_deg <= j <= B_max_deg
    # So: i >= k - B_max_deg and i <= k - B_min_deg and A_min_deg <= i <= A_max_deg
    i_start = tl.maximum(A_min_deg, k - B_max_deg)
    i_end = tl.minimum(A_max_deg, k - B_min_deg)
    
    # Base offsets for this batch
    batch_offset_A = pid_batch * A_deg_stride
    batch_offset_B = pid_batch * B_deg_stride
    batch_offset_C = pid_batch * C_deg_stride
    
    # Loop over all valid i values
    for i in range(i_start, i_end + 1):
        j = k - i
        
        # Index into A and B tensors (offset from their minimum degree)
        a_idx = i - A_min_deg
        b_idx = j - B_min_deg
        
        # Load 3x3 matrices A_i and B_j
        # Each matrix is stored contiguously as 9 elements
        a_offset = batch_offset_A + a_idx * 9
        b_offset = batch_offset_B + b_idx * 9
        
        # Load A_i as 3x3
        a00 = tl.load(A_ptr + a_offset + 0)
        a01 = tl.load(A_ptr + a_offset + 1)
        a02 = tl.load(A_ptr + a_offset + 2)
        a10 = tl.load(A_ptr + a_offset + 3)
        a11 = tl.load(A_ptr + a_offset + 4)
        a12 = tl.load(A_ptr + a_offset + 5)
        a20 = tl.load(A_ptr + a_offset + 6)
        a21 = tl.load(A_ptr + a_offset + 7)
        a22 = tl.load(A_ptr + a_offset + 8)
        
        # Load B_j as 3x3
        b00 = tl.load(B_ptr + b_offset + 0)
        b01 = tl.load(B_ptr + b_offset + 1)
        b02 = tl.load(B_ptr + b_offset + 2)
        b10 = tl.load(B_ptr + b_offset + 3)
        b11 = tl.load(B_ptr + b_offset + 4)
        b12 = tl.load(B_ptr + b_offset + 5)
        b20 = tl.load(B_ptr + b_offset + 6)
        b21 = tl.load(B_ptr + b_offset + 7)
        b22 = tl.load(B_ptr + b_offset + 8)
        
        # Compute A_i @ B_j and accumulate into acc
        # First row of result
        acc[0, 0] += a00 * b00 + a01 * b10 + a02 * b20
        acc[0, 1] += a00 * b01 + a01 * b11 + a02 * b21
        acc[0, 2] += a00 * b02 + a01 * b12 + a02 * b22
        
        # Second row of result
        acc[1, 0] += a10 * b00 + a11 * b10 + a12 * b20
        acc[1, 1] += a10 * b01 + a11 * b11 + a12 * b21
        acc[1, 2] += a10 * b02 + a11 * b12 + a12 * b22
        
        # Third row of result
        acc[2, 0] += a20 * b00 + a21 * b10 + a22 * b20
        acc[2, 1] += a20 * b01 + a21 * b11 + a22 * b21
        acc[2, 2] += a20 * b02 + a21 * b12 + a22 * b22
    
    # Store the result C_k
    c_offset = batch_offset_C + pid_deg * 9
    tl.store(C_ptr + c_offset + 0, acc[0, 0])
    tl.store(C_ptr + c_offset + 1, acc[0, 1])
    tl.store(C_ptr + c_offset + 2, acc[0, 2])
    tl.store(C_ptr + c_offset + 3, acc[1, 0])
    tl.store(C_ptr + c_offset + 4, acc[1, 1])
    tl.store(C_ptr + c_offset + 5, acc[1, 2])
    tl.store(C_ptr + c_offset + 6, acc[2, 0])
    tl.store(C_ptr + c_offset + 7, acc[2, 1])
    tl.store(C_ptr + c_offset + 8, acc[2, 2])


@triton.jit
def laurent_matmul_modp_kernel(
    A_ptr, B_ptr, C_ptr,
    A_min_deg, A_max_deg,
    B_min_deg, B_max_deg,
    C_min_deg, C_max_deg,
    batch_size,
    A_deg_stride, B_deg_stride, C_deg_stride,
    p,  # Prime modulus
    BLOCK_SIZE: tl.constexpr,
):
    """
    Multiply two batches of matrices over Z/pZ[v, v^{-1}].
    
    Same as laurent_matmul_kernel but performs all operations modulo p.
    Uses int64 to handle intermediate products without overflow.
    """
    # 2D grid: pid_batch for batch index, pid_deg for degree index
    pid_batch = tl.program_id(0)
    pid_deg = tl.program_id(1)
    k = C_min_deg + pid_deg  # The degree we're computing
    
    # Initialize accumulator for the 3x3 matrix (using int64 to avoid overflow)
    acc = tl.zeros((3, 3), dtype=tl.int64)
    
    # Compute the range of i values
    i_start = tl.maximum(A_min_deg, k - B_max_deg)
    i_end = tl.minimum(A_max_deg, k - B_min_deg)
    
    # Base offsets for this batch
    batch_offset_A = pid_batch * A_deg_stride
    batch_offset_B = pid_batch * B_deg_stride
    batch_offset_C = pid_batch * C_deg_stride
    
    # Loop over all valid i values
    for i in range(i_start, i_end + 1):
        j = k - i
        
        # Index into A and B tensors
        a_idx = i - A_min_deg
        b_idx = j - B_min_deg
        
        a_offset = batch_offset_A + a_idx * 9
        b_offset = batch_offset_B + b_idx * 9
        
        # Load A_i as 3x3 (convert to int64)
        a00 = tl.load(A_ptr + a_offset + 0).to(tl.int64)
        a01 = tl.load(A_ptr + a_offset + 1).to(tl.int64)
        a02 = tl.load(A_ptr + a_offset + 2).to(tl.int64)
        a10 = tl.load(A_ptr + a_offset + 3).to(tl.int64)
        a11 = tl.load(A_ptr + a_offset + 4).to(tl.int64)
        a12 = tl.load(A_ptr + a_offset + 5).to(tl.int64)
        a20 = tl.load(A_ptr + a_offset + 6).to(tl.int64)
        a21 = tl.load(A_ptr + a_offset + 7).to(tl.int64)
        a22 = tl.load(A_ptr + a_offset + 8).to(tl.int64)
        
        # Load B_j as 3x3 (convert to int64)
        b00 = tl.load(B_ptr + b_offset + 0).to(tl.int64)
        b01 = tl.load(B_ptr + b_offset + 1).to(tl.int64)
        b02 = tl.load(B_ptr + b_offset + 2).to(tl.int64)
        b10 = tl.load(B_ptr + b_offset + 3).to(tl.int64)
        b11 = tl.load(B_ptr + b_offset + 4).to(tl.int64)
        b12 = tl.load(B_ptr + b_offset + 5).to(tl.int64)
        b20 = tl.load(B_ptr + b_offset + 6).to(tl.int64)
        b21 = tl.load(B_ptr + b_offset + 7).to(tl.int64)
        b22 = tl.load(B_ptr + b_offset + 8).to(tl.int64)
        
        # Compute A_i @ B_j and accumulate (with modulo operations)
        # First row
        acc[0, 0] = (acc[0, 0] + (a00 * b00 + a01 * b10 + a02 * b20) % p) % p
        acc[0, 1] = (acc[0, 1] + (a00 * b01 + a01 * b11 + a02 * b21) % p) % p
        acc[0, 2] = (acc[0, 2] + (a00 * b02 + a01 * b12 + a02 * b22) % p) % p
        
        # Second row
        acc[1, 0] = (acc[1, 0] + (a10 * b00 + a11 * b10 + a12 * b20) % p) % p
        acc[1, 1] = (acc[1, 1] + (a10 * b01 + a11 * b11 + a12 * b21) % p) % p
        acc[1, 2] = (acc[1, 2] + (a10 * b02 + a11 * b12 + a12 * b22) % p) % p
        
        # Third row
        acc[2, 0] = (acc[2, 0] + (a20 * b00 + a21 * b10 + a22 * b20) % p) % p
        acc[2, 1] = (acc[2, 1] + (a20 * b01 + a21 * b11 + a22 * b21) % p) % p
        acc[2, 2] = (acc[2, 2] + (a20 * b02 + a21 * b12 + a22 * b22) % p) % p
    
    # Store the result C_k (convert back to int32)
    c_offset = batch_offset_C + pid_deg * 9
    tl.store(C_ptr + c_offset + 0, acc[0, 0].to(tl.int32))
    tl.store(C_ptr + c_offset + 1, acc[0, 1].to(tl.int32))
    tl.store(C_ptr + c_offset + 2, acc[0, 2].to(tl.int32))
    tl.store(C_ptr + c_offset + 3, acc[1, 0].to(tl.int32))
    tl.store(C_ptr + c_offset + 4, acc[1, 1].to(tl.int32))
    tl.store(C_ptr + c_offset + 5, acc[1, 2].to(tl.int32))
    tl.store(C_ptr + c_offset + 6, acc[2, 0].to(tl.int32))
    tl.store(C_ptr + c_offset + 7, acc[2, 1].to(tl.int32))
    tl.store(C_ptr + c_offset + 8, acc[2, 2].to(tl.int32))


def laurent_matmul(A, A_min_deg, B, B_min_deg, p=None):
    """
    Multiply two batches of Laurent polynomial matrices.
    
    Args:
        A: Tensor of shape (batch_size, A_num_deg, 3, 3) containing coefficients
        A_min_deg: Minimum degree of v in A (can be negative)
        B: Tensor of shape (batch_size, B_num_deg, 3, 3) containing coefficients
        B_min_deg: Minimum degree of v in B (can be negative)
        p: Optional prime modulus. If provided, all operations are done mod p.
           A and B should contain int32 values in [0, p).
    
    Returns:
        C: Tensor of shape (batch_size, C_num_deg, 3, 3) containing result coefficients
        C_min_deg: Minimum degree of v in C
    """
    batch_size = A.shape[0]
    assert B.shape[0] == batch_size, "Batch sizes must match"
    
    A_num_deg = A.shape[1]
    B_num_deg = B.shape[1]
    
    A_max_deg = A_min_deg + A_num_deg - 1
    B_max_deg = B_min_deg + B_num_deg - 1
    
    C_min_deg = A_min_deg + B_min_deg
    C_max_deg = A_max_deg + B_max_deg
    C_num_deg = C_max_deg - C_min_deg + 1
    
    if p is not None:
        # Modular arithmetic version
        assert A.dtype == torch.int32, "For modular arithmetic, input must be int32"
        assert B.dtype == torch.int32, "For modular arithmetic, input must be int32"
        
        # Allocate output
        C = torch.zeros((batch_size, C_num_deg, 3, 3), device=A.device, dtype=torch.int32)
        
        # Flatten for kernel access
        A_flat = A.reshape(-1)
        B_flat = B.reshape(-1)
        C_flat = C.reshape(-1)
        
        # Compute strides
        A_deg_stride = A_num_deg * 9
        B_deg_stride = B_num_deg * 9
        C_deg_stride = C_num_deg * 9
        
        # Launch kernel
        grid = (batch_size, C_num_deg)
        laurent_matmul_modp_kernel[grid](
            A_flat, B_flat, C_flat,
            A_min_deg, A_max_deg,
            B_min_deg, B_max_deg,
            C_min_deg, C_max_deg,
            batch_size,
            A_deg_stride, B_deg_stride, C_deg_stride,
            p,
            BLOCK_SIZE=1,
        )
    else:
        # Standard floating-point version
        # Allocate output
        C = torch.zeros((batch_size, C_num_deg, 3, 3), device=A.device, dtype=A.dtype)
        
        # Flatten for kernel access
        A_flat = A.reshape(-1)
        B_flat = B.reshape(-1)
        C_flat = C.reshape(-1)
        
        # Compute strides
        A_deg_stride = A_num_deg * 9
        B_deg_stride = B_num_deg * 9
        C_deg_stride = C_num_deg * 9
        
        # Launch kernel
        grid = (batch_size, C_num_deg)
        laurent_matmul_kernel[grid](
            A_flat, B_flat, C_flat,
            A_min_deg, A_max_deg,
            B_min_deg, B_max_deg,
            C_min_deg, C_max_deg,
            batch_size,
            A_deg_stride, B_deg_stride, C_deg_stride,
            BLOCK_SIZE=1,
        )
    
    return C, C_min_deg


# Example usage and testing
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("Testing Standard (Float) Version")
    print("="*70)
    
    # Batch size for testing
    batch_size = 4
    
    # Example: A = v^{-1} * I + v^0 * [[1,2,3],[4,5,6],[7,8,9]]
    A = torch.zeros((batch_size, 2, 3, 3), device=device, dtype=torch.float32)
    for b in range(batch_size):
        A[b, 0] = torch.eye(3) * (b + 1)
        A[b, 1] = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32) * (b + 1)
    A_min_deg = -1
    
    # Example: B = v^0 * I + v^1 * [[1,0,0],[0,1,0],[0,0,1]]
    B = torch.zeros((batch_size, 2, 3, 3), device=device, dtype=torch.float32)
    for b in range(batch_size):
        B[b, 0] = torch.tensor([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=torch.float32)
        B[b, 1] = torch.eye(3) * (b + 1)
    B_min_deg = 0
    
    # Compute A * B
    C, C_min_deg = laurent_matmul(A, A_min_deg, B, B_min_deg)
    
    print(f"Batch size: {batch_size}")
    print(f"A has degrees {A_min_deg} to {A_min_deg + A.shape[1] - 1}")
    print(f"B has degrees {B_min_deg} to {B_min_deg + B.shape[1] - 1}")
    print(f"C has degrees {C_min_deg} to {C_min_deg + C.shape[1] - 1}")
    
    # Verification
    C_verify = torch.zeros((batch_size, C.shape[1], 3, 3), device=device, dtype=torch.float32)
    for b in range(batch_size):
        for i in range(A.shape[1]):
            for j in range(B.shape[1]):
                k = (A_min_deg + i) + (B_min_deg + j) - C_min_deg
                C_verify[b, k] += A[b, i] @ B[b, j]
    
    print(f"Max difference: {(C - C_verify).abs().max().item()}")
    print(f"Results match: {torch.allclose(C, C_verify, atol=1e-5)}")
    
    print("\n" + "="*70)
    print("Testing Modular Arithmetic Version (mod p)")
    print("="*70)
    
    # Test with modular arithmetic
    p = 97  # Small prime for testing
    
    # Create integer matrices (values in [0, p))
    A_int = torch.randint(0, p, (batch_size, 3, 3, 3), device=device, dtype=torch.int32)
    B_int = torch.randint(0, p, (batch_size, 3, 3, 3), device=device, dtype=torch.int32)
    
    A_min_deg_int = -1
    B_min_deg_int = 0
    
    # Compute A * B mod p
    C_int, C_min_deg_int = laurent_matmul(A_int, A_min_deg_int, B_int, B_min_deg_int, p=p)
    
    print(f"Prime p = {p}")
    print(f"A has degrees {A_min_deg_int} to {A_min_deg_int + A_int.shape[1] - 1}")
    print(f"B has degrees {B_min_deg_int} to {B_min_deg_int + B_int.shape[1] - 1}")
    print(f"C has degrees {C_min_deg_int} to {C_min_deg_int + C_int.shape[1] - 1}")
    
    # Verification with PyTorch (using int64 to avoid overflow)
    C_verify_int = torch.zeros((batch_size, C_int.shape[1], 3, 3), device=device, dtype=torch.int64)
    for b in range(batch_size):
        for i in range(A_int.shape[1]):
            for j in range(B_int.shape[1]):
                k = (A_min_deg_int + i) + (B_min_deg_int + j) - C_min_deg_int
                prod = (A_int[b, i].to(torch.int64) @ B_int[b, j].to(torch.int64)) % p
                C_verify_int[b, k] = (C_verify_int[b, k] + prod) % p
    
    print(f"Max difference: {(C_int.to(torch.int64) - C_verify_int).abs().max().item()}")
    print(f"Results match: {torch.equal(C_int.to(torch.int64), C_verify_int)}")
    
    # Show an example result
    print(f"\nExample result (batch 0, degree {C_min_deg_int}):")
    print(C_int[0, 0])
    
    # Benchmark modular version
    if torch.cuda.is_available():
        print("\n" + "="*70)
        print("Benchmarking Modular Version")
        print("="*70)
        
        large_batch = 1000
        p_large = 1000000007  # Large prime (10^9 + 7)
        A_bench = torch.randint(0, p_large, (large_batch, 10, 3, 3), device=device, dtype=torch.int32)
        B_bench = torch.randint(0, p_large, (large_batch, 10, 3, 3), device=device, dtype=torch.int32)
        
        # Warmup
        for _ in range(10):
            C_bench, _ = laurent_matmul(A_bench, -5, B_bench, -5, p=p_large)
        
        torch.cuda.synchronize()
        import time
        start = time.time()
        for _ in range(100):
            C_bench, _ = laurent_matmul(A_bench, -5, B_bench, -5, p=p_large)
        torch.cuda.synchronize()
        end = time.time()
        
        print(f"Time for {large_batch} mod {p_large} multiplications (avg over 100 runs): {(end - start) / 100 * 1000:.3f} ms")
        print(f"Throughput: {large_batch / ((end - start) / 100):.0f} multiplications/second")