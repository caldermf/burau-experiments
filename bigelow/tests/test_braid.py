import unittest

from bigelow.algebra import FiniteFieldPrime, ZZ
from bigelow.braid import artin_automorphism, artin_image_of_word, burau_word_matrix


class BraidRelationTests(unittest.TestCase):
    def test_burau_braid_relation_over_integers(self) -> None:
        left = burau_word_matrix(5, (1, 2, 1), ZZ)
        right = burau_word_matrix(5, (2, 1, 2), ZZ)
        self.assertEqual(left, right)

    def test_burau_braid_relation_mod_p(self) -> None:
        ring = FiniteFieldPrime(5)
        left = burau_word_matrix(6, (3, 4, 3), ring)
        right = burau_word_matrix(6, (4, 3, 4), ring)
        self.assertEqual(left, right)

    def test_artin_braid_relation(self) -> None:
        left = artin_automorphism(5, (2, 3, 2))
        right = artin_automorphism(5, (3, 2, 3))
        self.assertEqual(left, right)

    def test_artin_automorphism_matches_word_action(self) -> None:
        word = (2, -1, 3, -2)
        automorphism = artin_automorphism(5, word)
        for index in range(1, 6):
            self.assertEqual(automorphism.apply((index,)), artin_image_of_word(5, word, (index,)))


if __name__ == "__main__":
    unittest.main()
