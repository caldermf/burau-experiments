import unittest

from bigelow.algebra import FiniteFieldPrime, ZZ
from bigelow.braid import burau_word_matrix, first_noncommuting_generator_image
from bigelow.search import WitnessType, orbit_search_states
from bigelow.witnesses import (
    bigelow_b4_q2_polynomial,
    bigelow_b5_left_twist_word,
    bigelow_b5_right_twist_word,
    bigelow_b6_left_twist_word,
    bigelow_b6_right_twist_word,
    bigelow_b6_left_conjugator,
)


class WitnessRegressionTests(unittest.TestCase):
    def test_b5_kernel_word_is_burau_trivial_over_integers(self) -> None:
        left = burau_word_matrix(5, bigelow_b5_left_twist_word(), ZZ)
        right = burau_word_matrix(5, bigelow_b5_right_twist_word(), ZZ)
        self.assertEqual(left * right, right * left)
        self.assertIsNotNone(first_noncommuting_generator_image(5, bigelow_b5_left_twist_word(), bigelow_b5_right_twist_word()))

    def test_b6_kernel_word_is_burau_trivial_over_integers(self) -> None:
        left = burau_word_matrix(6, bigelow_b6_left_twist_word(), ZZ)
        right = burau_word_matrix(6, bigelow_b6_right_twist_word(), ZZ)
        self.assertEqual(left * right, right * left)
        self.assertIsNotNone(first_noncommuting_generator_image(6, bigelow_b6_left_twist_word(), bigelow_b6_right_twist_word()))

    def test_b5_kernel_word_remains_trivial_mod_p(self) -> None:
        ring = FiniteFieldPrime(7)
        left = burau_word_matrix(5, bigelow_b5_left_twist_word(), ring)
        right = burau_word_matrix(5, bigelow_b5_right_twist_word(), ring)
        self.assertEqual(left * right, right * left)

    def test_b4_q2_false_alarm_polynomial(self) -> None:
        polynomial = bigelow_b4_q2_polynomial()
        self.assertEqual(polynomial.evaluate(2), 0)
        self.assertEqual(polynomial.evaluate(1), 0)

    def test_n6_search_orbit_smoke(self) -> None:
        left_orbit = orbit_search_states(
            6,
            ZZ,
            WitnessType.PUNCTURE_PUNCTURE,
            3,
            len(bigelow_b6_left_conjugator()),
        )
        left_matrix = burau_word_matrix(6, bigelow_b6_left_twist_word(), ZZ)
        self.assertIn(left_matrix, left_orbit)
        self.assertGreater(len(left_orbit), 1)


if __name__ == "__main__":
    unittest.main()
