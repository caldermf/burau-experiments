"""
Tests for triton7.py: pack/unpack, DFT/IDFT, and ring_matmul kernel.
Run with: python -m pytest test_triton7.py -v
   or:    python -m unittest test_triton7 -v
"""
import unittest
import numpy as np

try:
    import torch
    import triton
    from triton7 import (
        _pack_poly,
        _unpack_poly,
        _dft_6_point,
        _idft_6_point,
        _poly_mul_conv,
        ring_matmul_reference,
        ring_matmul,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import torch7
    TORCH7_AVAILABLE = True
except ImportError:
    TORCH7_AVAILABLE = False


class TestPackUnpack(unittest.TestCase):
    """Pack/unpack roundtrip and edge cases."""

    def test_pack_unpack_roundtrip(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch/triton not available")
        for _ in range(100):
            coeffs = [np.random.randint(0, 8) for _ in range(6)]
            packed = _pack_poly(coeffs)
            unpacked = _unpack_poly(packed)
            self.assertEqual(unpacked, coeffs, f"coeffs={coeffs} -> packed={packed} -> {unpacked}")

    def test_unpack_pack_roundtrip(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch/triton not available")
        for _ in range(100):
            packed = np.random.randint(0, 2**18, dtype=np.int32)
            unpacked = _unpack_poly(int(packed))
            # Only lower 3 bits per coeff matter
            repacked = _pack_poly(unpacked)
            self.assertEqual(_unpack_poly(repacked), unpacked)

    def test_zero_poly(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch/triton not available")
        zero = [0] * 6
        self.assertEqual(_unpack_poly(_pack_poly(zero)), zero)

    def test_max_bits(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch/triton not available")
        # Each coeff 0..7
        coeffs = [7] * 6
        packed = _pack_poly(coeffs)
        self.assertEqual(_unpack_poly(packed), coeffs)


class TestDFT(unittest.TestCase):
    """DFT and IDFT over F_7."""

    def test_dft_idft_roundtrip(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch/triton not available")
        for _ in range(50):
            coeffs = [np.random.randint(0, 7) for _ in range(6)]
            freq = _dft_6_point(coeffs)
            back = _idft_6_point(freq)
            self.assertEqual(back, coeffs, f"coeffs={coeffs} -> freq={freq} -> back={back}")

    def test_dft_constant_poly(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch/triton not available")
        # c = [a,0,0,0,0,0] -> F[0]=a, F[k]=a for all k (since 3^0=1)
        c = [3, 0, 0, 0, 0, 0]
        f = _dft_6_point(c)
        self.assertEqual(f[0], 3)
        for k in range(1, 6):
            self.assertEqual(f[k], 3, f"F[{k}] should be 3 for constant poly")

    def test_idft_scale(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch/triton not available")
        # IDFT of [1,0,0,0,0,0] should give (1/6)*[1,1,1,1,1,1] = 6*[1,1,1,1,1,1] mod 7 = [6,6,6,6,6,6]
        f = [1, 0, 0, 0, 0, 0]
        c = _idft_6_point(f)
        self.assertEqual(c, [6] * 6)


class TestPolyMul(unittest.TestCase):
    """Polynomial multiplication via DFT convolution."""

    def test_poly_mul_vs_naive_small(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch/triton not available")
        # Small coeffs: convolution length 6, result degree < 6 so no wrap.
        a = [1, 0, 0, 0, 0, 0]  # 1
        b = [0, 1, 0, 0, 0, 0]  # x
        prod = _poly_mul_conv(a, b)
        # 1 * x = x
        expected = [0, 1, 0, 0, 0, 0]
        self.assertEqual(prod, expected)

    def test_poly_mul_conv_identity(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch/triton not available")
        one = [1, 0, 0, 0, 0, 0]
        for _ in range(20):
            b = [np.random.randint(0, 7) for _ in range(6)]
            prod = _poly_mul_conv(one, b)
            self.assertEqual(prod, b)

    def test_poly_mul_commutative(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch/triton not available")
        for _ in range(30):
            a = [np.random.randint(0, 7) for _ in range(6)]
            b = [np.random.randint(0, 7) for _ in range(6)]
            self.assertEqual(_poly_mul_conv(a, b), _poly_mul_conv(b, a))


class TestRingMatmulReference(unittest.TestCase):
    """Reference ring_matmul only (no GPU)."""

    def test_reference_identity(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch/triton not available")
        # Identity 3x3: I @ B = B. Pack identity as 1 in (0,0), 0 elsewhere.
        # Packed 1 = 0b1 = 1. So row0: [1,0,0], row1: [0,0,0], row2: [0,0,0]
        one = _pack_poly([1, 0, 0, 0, 0, 0])
        zero = _pack_poly([0] * 6)
        I_flat = np.array([
            one, zero, zero,
            zero, zero, zero,
            zero, zero, zero,
        ], dtype=np.int32)
        B_flat = np.random.randint(0, 2**18, size=9, dtype=np.int32)
        # I @ B: only first row of I is non-zero, so result row0 = B row0, row1/2 = 0
        # Actually I is [[1,0,0],[0,0,0],[0,0,0]] so (I@B)[i,j] = B[0,j] if i==0 else 0
        C = ring_matmul_reference(
            I_flat.reshape(1, 9),
            B_flat.reshape(1, 9),
        )
        C = np.array(C).reshape(9)
        self.assertEqual(C[0], B_flat[0])
        self.assertEqual(C[1], B_flat[1])
        self.assertEqual(C[2], B_flat[2])
        self.assertEqual(C[3], 0)
        self.assertEqual(C[4], 0)
        self.assertEqual(C[5], 0)
        self.assertEqual(C[6], 0)
        self.assertEqual(C[7], 0)
        self.assertEqual(C[8], 0)

    def test_reference_batch_consistency(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch/triton not available")
        np.random.seed(42)
        A = np.random.randint(0, 2**18, size=(4, 9), dtype=np.int32)
        B = np.random.randint(0, 2**18, size=(4, 9), dtype=np.int32)
        C = ring_matmul_reference(A, B)
        for b in range(4):
            Cb = ring_matmul_reference(A[b:b+1], B[b:b+1])
            np.testing.assert_array_equal(C[b], np.array(Cb).reshape(9))


@unittest.skipUnless(TORCH_AVAILABLE and torch.cuda.is_available(), "CUDA required")
class TestRingMatmulKernel(unittest.TestCase):
    """Compare Triton ring_matmul kernel to reference on GPU."""

    def test_kernel_vs_reference_small_batch(self):
        np.random.seed(123)
        batch = 32
        A = torch.from_numpy(
            np.random.randint(0, 2**18, size=(batch, 9), dtype=np.int32)
        ).cuda()
        B = torch.from_numpy(
            np.random.randint(0, 2**18, size=(batch, 9), dtype=np.int32)
        ).cuda()
        C_kernel = ring_matmul(A, B)
        C_ref = ring_matmul_reference(A.cpu().numpy(), B.cpu().numpy())
        np.testing.assert_array_equal(
            C_kernel.cpu().numpy(),
            np.array(C_ref),
            err_msg="Kernel output should match reference",
        )

    def test_kernel_vs_reference_larger_batch(self):
        np.random.seed(456)
        batch = 300
        A = torch.from_numpy(
            np.random.randint(0, 2**18, size=(batch, 9), dtype=np.int32)
        ).cuda()
        B = torch.from_numpy(
            np.random.randint(0, 2**18, size=(batch, 9), dtype=np.int32)
        ).cuda()
        C_kernel = ring_matmul(A, B)
        C_ref = ring_matmul_reference(A.cpu().numpy(), B.cpu().numpy())
        np.testing.assert_array_equal(
            C_kernel.cpu().numpy(),
            np.array(C_ref),
            err_msg="Kernel output should match reference for batch=300",
        )

    def test_kernel_batch_not_multiple_of_block(self):
        batch = 7
        A = torch.randint(0, 2**18, (batch, 9), dtype=torch.int32, device="cuda")
        B = torch.randint(0, 2**18, (batch, 9), dtype=torch.int32, device="cuda")
        C_kernel = ring_matmul(A, B)
        C_ref = ring_matmul_reference(A.cpu().numpy(), B.cpu().numpy())
        np.testing.assert_array_equal(C_kernel.cpu().numpy(), np.array(C_ref))

    def test_kernel_zero_inputs(self):
        batch = 16
        zero = _pack_poly([0] * 6)
        A = torch.full((batch, 9), zero, dtype=torch.int32, device="cuda")
        B = torch.randint(0, 2**18, (batch, 9), dtype=torch.int32, device="cuda")
        C = ring_matmul(A, B)
        expected = np.full((batch, 9), zero, dtype=np.int32)
        np.testing.assert_array_equal(C.cpu().numpy(), expected)


@unittest.skipUnless(
    TORCH_AVAILABLE and TORCH7_AVAILABLE and torch.cuda.is_available(),
    "torch, torch7, and CUDA required",
)
class TestTorch7VsTriton7(unittest.TestCase):
    """Ensure torch7.ring_matmul and triton7.ring_matmul give identical results."""

    def test_torch7_vs_triton7_small_batch(self):
        np.random.seed(789)
        batch = 32
        A = torch.from_numpy(
            np.random.randint(0, 2**18, size=(batch, 9), dtype=np.int32)
        ).cuda()
        B = torch.from_numpy(
            np.random.randint(0, 2**18, size=(batch, 9), dtype=np.int32)
        ).cuda()
        C_triton = ring_matmul(A, B)
        C_torch = torch7.ring_matmul(A, B)
        np.testing.assert_array_equal(
            C_triton.cpu().numpy(),
            C_torch.cpu().numpy(),
            err_msg="torch7 and triton7 should match for small batch",
        )

    def test_torch7_vs_triton7_larger_batch(self):
        np.random.seed(101)
        batch = 256
        A = torch.from_numpy(
            np.random.randint(0, 2**18, size=(batch, 9), dtype=np.int32)
        ).cuda()
        B = torch.from_numpy(
            np.random.randint(0, 2**18, size=(batch, 9), dtype=np.int32)
        ).cuda()
        C_triton = ring_matmul(A, B)
        C_torch = torch7.ring_matmul(A, B)
        np.testing.assert_array_equal(
            C_triton.cpu().numpy(),
            C_torch.cpu().numpy(),
            err_msg="torch7 and triton7 should match for larger batch",
        )

    def test_torch7_vs_triton7_batch_not_multiple_of_block(self):
        np.random.seed(202)
        batch = 7
        A = torch.randint(0, 2**18, (batch, 9), dtype=torch.int32, device="cuda")
        B = torch.randint(0, 2**18, (batch, 9), dtype=torch.int32, device="cuda")
        C_triton = ring_matmul(A, B)
        C_torch = torch7.ring_matmul(A, B)
        np.testing.assert_array_equal(
            C_triton.cpu().numpy(),
            C_torch.cpu().numpy(),
            err_msg="torch7 and triton7 should match when batch is not multiple of block",
        )

    def test_torch7_vs_triton7_zero_inputs(self):
        batch = 16
        zero = _pack_poly([0] * 6)
        A = torch.full((batch, 9), zero, dtype=torch.int32, device="cuda")
        B = torch.randint(0, 2**18, (batch, 9), dtype=torch.int32, device="cuda")
        C_triton = ring_matmul(A, B)
        C_torch = torch7.ring_matmul(A, B)
        np.testing.assert_array_equal(
            C_triton.cpu().numpy(),
            C_torch.cpu().numpy(),
            err_msg="torch7 and triton7 should match for zero A",
        )

    def test_torch7_vs_triton7_single_matrix(self):
        np.random.seed(303)
        A = torch.randint(0, 2**18, (1, 9), dtype=torch.int32, device="cuda")
        B = torch.randint(0, 2**18, (1, 9), dtype=torch.int32, device="cuda")
        C_triton = ring_matmul(A, B)
        C_torch = torch7.ring_matmul(A, B)
        np.testing.assert_array_equal(
            C_triton.cpu().numpy(),
            C_torch.cpu().numpy(),
            err_msg="torch7 and triton7 should match for batch=1",
        )


if __name__ == "__main__":
    unittest.main()
