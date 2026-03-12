import unittest

from bigelow.algebra import FiniteFieldPrime, LaurentPoly, Matrix, ZZ


class LaurentPolynomialTests(unittest.TestCase):
    def test_integer_arithmetic(self) -> None:
        q = LaurentPoly.q(ZZ)
        poly = q * q - LaurentPoly.constant(ZZ, 3) * q + LaurentPoly.constant(ZZ, 2)
        self.assertEqual(poly.evaluate(2), 0)
        shifted = poly.shift(-1)
        self.assertEqual(shifted.evaluate(2), 0)

    def test_mod_p_conversion(self) -> None:
        q = LaurentPoly.q(ZZ)
        poly = LaurentPoly.constant(ZZ, 5) * q + LaurentPoly.constant(ZZ, -3)
        mod5 = poly.convert(FiniteFieldPrime(5))
        self.assertEqual(mod5.to_dict(), {0: 2})

    def test_matrix_multiplication(self) -> None:
        ring = ZZ
        q = LaurentPoly.q(ring)
        one = LaurentPoly.one(ring)
        zero = LaurentPoly.zero(ring)
        left = Matrix([[one, q], [zero, one]])
        right = Matrix([[one, zero], [q, one]])
        product = left * right
        self.assertEqual(product.rows[0][0], one + (q * q))


if __name__ == "__main__":
    unittest.main()
