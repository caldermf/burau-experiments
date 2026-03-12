from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Tuple

from .algebra import CoefficientRing, LaurentPoly, Matrix
from .freegroup import FreeGroupAutomorphism, Word, generator, invert_word as invert_free_word, multiply_words


BraidWord = Tuple[int, ...]


def invert_word(word: BraidWord) -> BraidWord:
    return tuple(-letter for letter in reversed(word))


def conjugate_word(word: BraidWord, conjugator: BraidWord) -> BraidWord:
    return invert_word(conjugator) + word + conjugator


def commutator_word(left: BraidWord, right: BraidWord) -> BraidWord:
    return invert_word(left) + invert_word(right) + left + right


def word_length(word: BraidWord) -> int:
    return len(word)


def boundary_arc_full_twist_word(endpoint_index: int) -> BraidWord:
    if endpoint_index < 1:
        raise ValueError("endpoint_index must be at least 1")
    left = tuple(range(endpoint_index, 1, -1))
    right = tuple(range(2, endpoint_index + 1))
    return left + (1, 1) + right


@lru_cache(maxsize=None)
def burau_generator_matrix(n: int, generator_index: int, ring: CoefficientRing) -> Matrix:
    if n < 2:
        raise ValueError("n must be at least 2")
    if not 1 <= abs(generator_index) <= n - 1:
        raise ValueError("generator index out of range")
    positive = generator_index > 0
    size = n - 1
    q = LaurentPoly.q(ring)
    one = LaurentPoly.one(ring)
    zero = LaurentPoly.zero(ring)
    minus_q = LaurentPoly.constant(ring, -1) * q
    if positive:
        rows = [[zero for _ in range(size)] for _ in range(size)]
        for index in range(size):
            rows[index][index] = one
        i = generator_index - 1
        if generator_index == 1:
            rows[0][0] = minus_q
            rows[0][1] = q if size > 1 else zero
        elif generator_index == n - 1:
            rows[i][i - 1] = one
            rows[i][i] = minus_q
        else:
            rows[i - 1][i - 1] = one
            rows[i][i - 1] = one
            rows[i][i] = minus_q
            rows[i][i + 1] = q
        return Matrix(rows)

    positive_matrix = burau_generator_matrix(n, -generator_index, ring)
    return burau_inverse_of_generator(positive_matrix, -generator_index, n, ring)


def burau_inverse_of_generator(matrix: Matrix, generator_index: int, n: int, ring: CoefficientRing) -> Matrix:
    size = n - 1
    q = LaurentPoly.q(ring)
    one = LaurentPoly.one(ring)
    zero = LaurentPoly.zero(ring)
    minus_q_inv = LaurentPoly.constant(ring, -1).shift(-1)
    q_inv = LaurentPoly.one(ring).shift(-1)
    rows = [[zero for _ in range(size)] for _ in range(size)]
    for index in range(size):
        rows[index][index] = one
    i = generator_index - 1
    if generator_index == 1:
        rows[0][0] = minus_q_inv
        if size > 1:
            rows[0][1] = one
    elif generator_index == n - 1:
        rows[i][i - 1] = q_inv
        rows[i][i] = minus_q_inv
    else:
        rows[i][i - 1] = q_inv
        rows[i][i] = minus_q_inv
        rows[i][i + 1] = one
    inverse = Matrix(rows)
    if matrix * inverse != Matrix.identity(ring, size):
        raise ValueError("generator inverse formula is inconsistent")
    return inverse


def burau_identity(n: int, ring: CoefficientRing) -> Matrix:
    return Matrix.identity(ring, n - 1)


def burau_word_matrix(n: int, word: BraidWord, ring: CoefficientRing) -> Matrix:
    current = burau_identity(n, ring)
    for letter in word:
        current = current * burau_generator_matrix(n, letter, ring)
    return current


@lru_cache(maxsize=None)
def artin_generator_automorphism(n: int, generator_index: int) -> FreeGroupAutomorphism:
    if not 1 <= abs(generator_index) <= n - 1:
        raise ValueError("generator index out of range")
    images = [generator(index) for index in range(1, n + 1)]
    i = abs(generator_index)
    if generator_index > 0:
        images[i - 1] = multiply_words(generator(i), generator(i + 1), invert_free_word(generator(i)))
        images[i] = generator(i)
    else:
        images[i - 1] = generator(i + 1)
        images[i] = multiply_words(invert_free_word(generator(i + 1)), generator(i), generator(i + 1))
    return FreeGroupAutomorphism(tuple(images))


def artin_automorphism(n: int, word: BraidWord) -> FreeGroupAutomorphism:
    current = FreeGroupAutomorphism.identity(n)
    for letter in word:
        current = artin_generator_automorphism(n, letter).compose(current)
    return current


def artin_image_of_word(n: int, braid_word: BraidWord, free_word: Word) -> Word:
    current = free_word
    for letter in braid_word:
        current = artin_generator_automorphism(n, letter).apply(current)
    return current


def first_changed_generator_image(n: int, braid_word: BraidWord) -> tuple[int, Word] | None:
    for index in range(1, n + 1):
        image = artin_image_of_word(n, braid_word, (index,))
        if image != (index,):
            return index, image
    return None


def first_noncommuting_generator_image(n: int, left_word: BraidWord, right_word: BraidWord) -> tuple[int, Word, Word] | None:
    for index in range(1, n + 1):
        left_then_right = artin_image_of_word(n, left_word + right_word, (index,))
        right_then_left = artin_image_of_word(n, right_word + left_word, (index,))
        if left_then_right != right_then_left:
            return index, left_then_right, right_then_left
    return None


def first_noncommuting_image_for_automorphisms(
    left: FreeGroupAutomorphism,
    right: FreeGroupAutomorphism,
) -> tuple[int, Word, Word] | None:
    left_then_right = right.compose(left)
    right_then_left = left.compose(right)
    for index, (left_image, right_image) in enumerate(zip(left_then_right.images, right_then_left.images), start=1):
        if left_image != right_image:
            return index, left_image, right_image
    return None
