"""
Tests for morecontext.py: ring42_matmul kernel for F_7[x]/(x^42-1).
Run with: python -m pytest test_morecontext.py -v
   or:    python -m unittest test_morecontext -v
"""
import unittest
import numpy as np

try:
    import torch
    import triton
    from morecontext import (
        ring42_matmul,
        ring42_matmul_reference,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE and torch.cuda.is_available(), "CUDA required")
class TestRing42Matmul(unittest.TestCase):
    """Test ring42_matmul kernel against reference implementation."""

    def test_kernel_vs_reference_small_batch(self):
        """Test correctness on small batch."""
        np.random.seed(42)
        batch = 32
        A = torch.randint(0, 2**18, (63, batch), dtype=torch.int32, device="cuda")
        B = torch.randint(0, 2**18, (63, batch), dtype=torch.int32, device="cuda")
        # Mask to valid packed polys (only low 18 bits)
        A = A & ((1 << 18) - 1)
        B = B & ((1 << 18) - 1)
        
        C_kernel = ring42_matmul(A, B)
        C_ref = ring42_matmul_reference(A, B)
        
        np.testing.assert_array_equal(
            C_kernel.cpu().numpy(),
            np.array(C_ref),
            err_msg="Kernel output should match reference for small batch",
        )

    def test_kernel_vs_reference_larger_batch(self):
        """Test correctness on larger batch."""
        np.random.seed(123)
        batch = 256
        A = torch.randint(0, 2**18, (63, batch), dtype=torch.int32, device="cuda")
        B = torch.randint(0, 2**18, (63, batch), dtype=torch.int32, device="cuda")
        A = A & ((1 << 18) - 1)
        B = B & ((1 << 18) - 1)
        
        C_kernel = ring42_matmul(A, B)
        C_ref = ring42_matmul_reference(A, B)
        
        np.testing.assert_array_equal(
            C_kernel.cpu().numpy(),
            np.array(C_ref),
            err_msg="Kernel output should match reference for larger batch",
        )

    def test_kernel_vs_reference_batch_not_multiple_of_block(self):
        """Test correctness when batch size is not a multiple of BLOCK_SIZE (128)."""
        np.random.seed(456)
        batch = 7
        A = torch.randint(0, 2**18, (63, batch), dtype=torch.int32, device="cuda")
        B = torch.randint(0, 2**18, (63, batch), dtype=torch.int32, device="cuda")
        A = A & ((1 << 18) - 1)
        B = B & ((1 << 18) - 1)
        
        C_kernel = ring42_matmul(A, B)
        C_ref = ring42_matmul_reference(A, B)
        
        np.testing.assert_array_equal(
            C_kernel.cpu().numpy(),
            np.array(C_ref),
            err_msg="Kernel output should match reference when batch is not multiple of block",
        )

    def test_kernel_vs_reference_single_matrix(self):
        """Test correctness for batch=1."""
        np.random.seed(789)
        A = torch.randint(0, 2**18, (63, 1), dtype=torch.int32, device="cuda")
        B = torch.randint(0, 2**18, (63, 1), dtype=torch.int32, device="cuda")
        A = A & ((1 << 18) - 1)
        B = B & ((1 << 18) - 1)
        
        C_kernel = ring42_matmul(A, B)
        C_ref = ring42_matmul_reference(A, B)
        
        np.testing.assert_array_equal(
            C_kernel.cpu().numpy(),
            np.array(C_ref),
            err_msg="Kernel output should match reference for batch=1",
        )

    def test_kernel_vs_reference_zero_inputs(self):
        """Test correctness when A is all zeros."""
        batch = 16
        # Create zero polynomial: all coefficients are 0, packed = 0
        zero = 0
        A = torch.full((63, batch), zero, dtype=torch.int32, device="cuda")
        B = torch.randint(0, 2**18, (63, batch), dtype=torch.int32, device="cuda")
        B = B & ((1 << 18) - 1)
        
        C_kernel = ring42_matmul(A, B)
        C_ref = ring42_matmul_reference(A, B)
        
        np.testing.assert_array_equal(
            C_kernel.cpu().numpy(),
            np.array(C_ref),
            err_msg="Kernel output should match reference for zero A",
        )
        
        # Result should also be zero
        expected = np.zeros((63, batch), dtype=np.int32)
        np.testing.assert_array_equal(C_kernel.cpu().numpy(), expected)

    def test_kernel_vs_reference_identity_like(self):
        """Test with identity-like matrix (1 in first entry, 0 elsewhere)."""
        batch = 8
        # Pack polynomial [1, 0, 0, 0, 0, 0] = 1
        one = 1
        zero = 0
        
        # Create identity-like: I[0,0] = 1, rest = 0
        # Entry 0 (row 0, col 0) = entry 0, components 0-6
        A = torch.zeros((63, batch), dtype=torch.int32, device="cuda")
        A[0, :] = one  # First component of first entry = 1
        
        B = torch.randint(0, 2**18, (63, batch), dtype=torch.int32, device="cuda")
        B = B & ((1 << 18) - 1)
        
        C_kernel = ring42_matmul(A, B)
        C_ref = ring42_matmul_reference(A, B)
        
        np.testing.assert_array_equal(
            C_kernel.cpu().numpy(),
            np.array(C_ref),
            err_msg="Kernel output should match reference for identity-like matrix",
        )

    def test_kernel_vs_reference_large_batch(self):
        """Test correctness on large batch."""
        np.random.seed(999)
        batch = 1024
        A = torch.randint(0, 2**18, (63, batch), dtype=torch.int32, device="cuda")
        B = torch.randint(0, 2**18, (63, batch), dtype=torch.int32, device="cuda")
        A = A & ((1 << 18) - 1)
        B = B & ((1 << 18) - 1)
        
        C_kernel = ring42_matmul(A, B)
        C_ref = ring42_matmul_reference(A, B)
        
        np.testing.assert_array_equal(
            C_kernel.cpu().numpy(),
            np.array(C_ref),
            err_msg="Kernel output should match reference for large batch",
        )

    def test_kernel_output_shape(self):
        """Test that output shape is correct."""
        batch = 64
        A = torch.randint(0, 2**18, (63, batch), dtype=torch.int32, device="cuda")
        B = torch.randint(0, 2**18, (63, batch), dtype=torch.int32, device="cuda")
        A = A & ((1 << 18) - 1)
        B = B & ((1 << 18) - 1)
        
        C = ring42_matmul(A, B)
        
        self.assertEqual(C.shape, (63, batch))
        self.assertEqual(C.dtype, torch.int32)
        self.assertEqual(C.device.type, "cuda")

    def test_kernel_input_validation(self):
        """Test that input validation works (wrong shape raises before kernel launch)."""
        batch = 32
        B = torch.randint(0, 2**18, (63, batch), dtype=torch.int32, device="cuda")
        # Test wrong shape first - hits Python assert, no kernel compile/launch
        A_wrong = torch.randint(0, 2**18, (64, batch), dtype=torch.int32, device="cuda")
        with self.assertRaises(AssertionError):
            ring42_matmul(A_wrong, B)
        # Correct shape is tested in test_kernel_output_shape


if __name__ == "__main__":
    unittest.main()
