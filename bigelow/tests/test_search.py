import unittest

from bigelow.algebra import FiniteFieldPrime, ZZ
from bigelow.braid import burau_generator_matrix, burau_word_matrix
from bigelow.search import conjugate_matrix_by_generator


class SearchConjugationTests(unittest.TestCase):
    def test_exact_generator_conjugation_matches_generic_integer_path(self) -> None:
        matrix = burau_word_matrix(6, (3, -2, 4, -5, 1, -3), ZZ)
        for letter in (1, -1, 2, -2, 3, -3, 4, -4, 5, -5):
            generic = burau_generator_matrix(6, -letter, ZZ) * matrix * burau_generator_matrix(6, letter, ZZ)
            fast = conjugate_matrix_by_generator(matrix, letter)
            self.assertEqual(fast, generic)

    def test_exact_generator_conjugation_matches_generic_mod_p_path(self) -> None:
        ring = FiniteFieldPrime(7)
        matrix = burau_word_matrix(5, (2, -1, 3, -2, 1), ring)
        for letter in (1, -1, 2, -2, 3, -3, 4, -4):
            generic = burau_generator_matrix(5, -letter, ring) * matrix * burau_generator_matrix(5, letter, ring)
            fast = conjugate_matrix_by_generator(matrix, letter)
            self.assertEqual(fast, generic)


if __name__ == "__main__":
    unittest.main()
