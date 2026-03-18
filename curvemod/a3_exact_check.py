from __future__ import annotations

from typing import List

import setup_a3 as ct


Vector = List[dict[int, int]]
Blocks = List[List[int]]


def _reduce_polynomial(pol: dict[int, int], modulus: int) -> dict[int, int]:
    out: dict[int, int] = {}
    for degree, coeff in pol.items():
        value = coeff % modulus
        if value != 0:
            out[degree] = value
    return out


def _reduce_vector(vec: Vector, modulus: int) -> Vector:
    return [_reduce_polynomial(pol, modulus) for pol in vec]


def apply_letter_word(word: List[int], vec: Vector, modulus: int) -> Vector:
    out = vec
    for letter in reversed(word):
        out = ct.oburau_fns[letter](out)
        out = _reduce_vector(out, modulus)
    return out


def apply_blocks(blocks: Blocks, vec: Vector, modulus: int) -> Vector:
    out = vec
    for block in blocks:
        out = apply_letter_word(block, out, modulus)
    return out


def invert_blocks(blocks: Blocks) -> Blocks:
    return [[-letter for letter in reversed(block)] for block in reversed(blocks)]


def commutator_blocks(blocks: Blocks, generator: int) -> Blocks:
    return blocks + [[generator]] + invert_blocks(blocks) + [[-generator]]


def matrix_columns_from_blocks(blocks: Blocks, modulus: int) -> List[Vector]:
    basis = [ct.dim_vectors[1], ct.dim_vectors[2], ct.dim_vectors[3]]
    return [apply_blocks(blocks, vec, modulus) for vec in basis]


def commutator_is_identity(blocks: Blocks, modulus: int, generator: int = 1) -> bool:
    columns = matrix_columns_from_blocks(commutator_blocks(blocks, generator), modulus)
    return columns == [ct.dim_vectors[1], ct.dim_vectors[2], ct.dim_vectors[3]]
