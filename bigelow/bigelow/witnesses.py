from __future__ import annotations

from .algebra import LaurentPoly, ZZ
from .braid import BraidWord, boundary_arc_full_twist_word, commutator_word, conjugate_word


def bigelow_b5_left_conjugator() -> BraidWord:
    return (-3, 2, 1, 1, 2, 4, 4, 4, 3, 2)


def bigelow_b5_right_conjugator() -> BraidWord:
    return (-4, 3, 2, -1, -1, 2, 1, 1, 2, 2, 1, 4, 4, 4, 4, 4)


def bigelow_b5_kernel_word() -> BraidWord:
    return commutator_word(bigelow_b5_left_twist_word(), bigelow_b5_right_twist_word())


def bigelow_b5_left_twist_word() -> BraidWord:
    return conjugate_word((4,), bigelow_b5_left_conjugator())


def bigelow_b5_right_twist_word() -> BraidWord:
    return conjugate_word(boundary_arc_full_twist_word(4), bigelow_b5_right_conjugator())


def bigelow_b6_left_conjugator() -> BraidWord:
    return (4, -5, -2, 1)


def bigelow_b6_right_conjugator() -> BraidWord:
    return (-4, 5, 5, 2, -1, -1)


def bigelow_b6_kernel_word() -> BraidWord:
    return commutator_word(bigelow_b6_left_twist_word(), bigelow_b6_right_twist_word())


def bigelow_b6_left_twist_word() -> BraidWord:
    return conjugate_word((3,), bigelow_b6_left_conjugator())


def bigelow_b6_right_twist_word() -> BraidWord:
    return conjugate_word((3,), bigelow_b6_right_conjugator())


def bigelow_b4_q2_polynomial() -> LaurentPoly:
    q = LaurentPoly.q(ZZ)
    one = LaurentPoly.one(ZZ)
    return (
        LaurentPoly.constant(ZZ, -1)
        * (q - one)
        * (q - LaurentPoly.constant(ZZ, 2))
        * (LaurentPoly.constant(ZZ, 2) * q - one)
        * (q * q - q + one)
        * (q * q + one)
    )
