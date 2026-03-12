from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Dict, Iterable, Iterator, Tuple

from .algebra import CoefficientRing, FiniteFieldPrime, LaurentPoly, Matrix
from .braid import (
    BraidWord,
    artin_automorphism,
    artin_generator_automorphism,
    boundary_arc_full_twist_word,
    commutator_word,
    conjugate_word,
    first_noncommuting_image_for_automorphisms,
)
from .freegroup import FreeGroupAutomorphism


class WitnessType(str, Enum):
    PUNCTURE_PUNCTURE = "puncture_puncture"
    BOUNDARY_PUNCTURE = "boundary_puncture"


@dataclass(frozen=True)
class TwistOrbitState:
    n: int
    witness_type: WitnessType
    conjugator: BraidWord
    twist_word: BraidWord
    twist_matrix: Matrix

    @property
    def depth(self) -> int:
        return len(self.conjugator)


@dataclass(frozen=True)
class SearchConfig:
    n: int
    ring: CoefficientRing
    left_witness_type: WitnessType
    left_base_index: int
    left_max_depth: int
    right_witness_type: WitnessType
    right_base_index: int
    right_max_depth: int
    generator_order: Tuple[int, ...] | None = None
    specialization_filters: Tuple[Tuple[int, int], ...] = ((5, 2), (7, 3))
    relative_max_depth: int | None = None


@dataclass(frozen=True)
class SearchResult:
    config: SearchConfig
    left_state: TwistOrbitState
    right_state: TwistOrbitState
    kernel_word: BraidWord
    kernel_matrix: Matrix
    nontrivial_generator: int
    left_then_right_image: tuple[int, ...]
    right_then_left_image: tuple[int, ...]


@dataclass(frozen=True)
class RelativeSearchNode:
    state: TwistOrbitState
    twist_automorphism: FreeGroupAutomorphism


def base_twist_word(witness_type: WitnessType, base_index: int) -> BraidWord:
    if witness_type == WitnessType.PUNCTURE_PUNCTURE:
        return (base_index,)
    if witness_type == WitnessType.BOUNDARY_PUNCTURE:
        return boundary_arc_full_twist_word(base_index)
    raise ValueError(f"unsupported witness type: {witness_type}")


def orbit_search_states(
    n: int,
    ring: CoefficientRing,
    witness_type: WitnessType,
    base_index: int,
    max_depth: int,
    generator_order: Tuple[int, ...] | None = None,
) -> Dict[Tuple[Tuple[int, ...], ...], TwistOrbitState]:
    base_word = base_twist_word(witness_type, base_index)
    base_matrix = _evaluate_word_from_generators(n, base_word, ring)
    base_state = TwistOrbitState(
        n=n,
        witness_type=witness_type,
        conjugator=(),
        twist_word=base_word,
        twist_matrix=base_matrix,
    )
    seen: Dict[Matrix, TwistOrbitState] = {base_state.twist_matrix: base_state}
    queue = deque([base_state])
    if generator_order is None:
        generator_order = tuple(range(1, n)) + tuple(range(-1, -n, -1))
    while queue:
        state = queue.popleft()
        if state.depth >= max_depth:
            continue
        last = state.conjugator[-1] if state.conjugator else None
        for letter in generator_order:
            if last is not None and letter == -last:
                continue
            next_conjugator = state.conjugator + (letter,)
            next_matrix = conjugate_matrix_by_generator(state.twist_matrix, letter)
            if next_matrix in seen:
                continue
            next_twist_word = conjugate_word(base_word, next_conjugator)
            next_state = TwistOrbitState(
                n=n,
                witness_type=witness_type,
                conjugator=next_conjugator,
                twist_word=next_twist_word,
                twist_matrix=next_matrix,
            )
            seen[next_matrix] = next_state
            queue.append(next_state)
    return seen


def find_commuting_kernel_pair(config: SearchConfig) -> SearchResult | None:
    left_base_word = base_twist_word(config.left_witness_type, config.left_base_index)
    left_base_matrix = _evaluate_word_from_generators(config.n, left_base_word, config.ring)
    left_base_automorphism = artin_automorphism(config.n, left_base_word)
    left_state = TwistOrbitState(
        n=config.n,
        witness_type=config.left_witness_type,
        conjugator=(),
        twist_word=left_base_word,
        twist_matrix=left_base_matrix,
    )
    base_right_word = base_twist_word(config.right_witness_type, config.right_base_index)
    base_right_matrix = _evaluate_word_from_generators(config.n, base_right_word, config.ring)
    base_right_automorphism = artin_automorphism(config.n, base_right_word)
    base_right_state = TwistOrbitState(
        n=config.n,
        witness_type=config.right_witness_type,
        conjugator=(),
        twist_word=base_right_word,
        twist_matrix=base_right_matrix,
    )
    identity_matrix = Matrix.identity(config.ring, config.n - 1)
    max_depth = config.relative_max_depth
    if max_depth is None:
        max_depth = config.left_max_depth + config.right_max_depth
    if config.generator_order is None:
        generator_order = tuple(range(1, config.n)) + tuple(range(-1, -config.n, -1))
    else:
        generator_order = config.generator_order
    generator_automorphisms = {letter: artin_generator_automorphism(config.n, letter) for letter in generator_order}
    left_filters = tuple(
        _specialize_matrix(left_base_matrix, modulus, q_value) for modulus, q_value in config.specialization_filters
    )
    base_node = RelativeSearchNode(state=base_right_state, twist_automorphism=base_right_automorphism)
    seen: Dict[Matrix, RelativeSearchNode] = {base_right_state.twist_matrix: base_node}
    queue = deque([base_node])
    while queue:
        node = queue.popleft()
        right_state = node.state
        if right_state.twist_matrix != left_base_matrix and _matrices_commute_under_filters(
            left_base_matrix,
            left_filters,
            right_state.twist_matrix,
            config.specialization_filters,
        ):
            kernel_word = commutator_word(left_state.twist_word, right_state.twist_word)
            witness = first_noncommuting_image_for_automorphisms(left_base_automorphism, node.twist_automorphism)
            if witness is not None:
                nontrivial_generator, left_then_right_image, right_then_left_image = witness
                return SearchResult(
                    config=config,
                    left_state=left_state,
                    right_state=right_state,
                    kernel_word=kernel_word,
                    kernel_matrix=identity_matrix,
                    nontrivial_generator=nontrivial_generator,
                    left_then_right_image=left_then_right_image,
                    right_then_left_image=right_then_left_image,
                )
        if right_state.depth >= max_depth:
            continue
        last = right_state.conjugator[-1] if right_state.conjugator else None
        for letter in generator_order:
            if last is not None and letter == -last:
                continue
            next_conjugator = right_state.conjugator + (letter,)
            next_matrix = conjugate_matrix_by_generator(right_state.twist_matrix, letter)
            if next_matrix in seen:
                continue
            next_state = TwistOrbitState(
                n=config.n,
                witness_type=config.right_witness_type,
                conjugator=next_conjugator,
                twist_word=conjugate_word(base_right_word, next_conjugator),
                twist_matrix=next_matrix,
            )
            generator_automorphism = generator_automorphisms[letter]
            inverse_automorphism = generator_automorphisms[-letter]
            next_automorphism = generator_automorphism.compose(node.twist_automorphism).compose(inverse_automorphism)
            next_node = RelativeSearchNode(state=next_state, twist_automorphism=next_automorphism)
            seen[next_matrix] = next_node
            queue.append(next_node)
    return None


def _evaluate_word_from_generators(n: int, word: BraidWord, ring: CoefficientRing) -> Matrix:
    current = Matrix.identity(ring, n - 1)
    for letter in word:
        current = current * _generator_matrix_for_word(n, letter, ring)
    return current


def _specialize_matrix(matrix: Matrix, modulus: int, q_value: int) -> Tuple[Tuple[int, ...], ...]:
    ring = FiniteFieldPrime(modulus)
    converted = matrix.convert(ring)
    return tuple(tuple(int(entry.evaluate(q_value)) for entry in row) for row in converted.rows)


def _multiply_specialized(left: Tuple[Tuple[int, ...], ...], right: Tuple[Tuple[int, ...], ...], modulus: int) -> Tuple[Tuple[int, ...], ...]:
    size = len(left)
    rows = []
    for row_index in range(size):
        row = []
        for col_index in range(size):
            total = 0
            for inner in range(size):
                total += left[row_index][inner] * right[inner][col_index]
            row.append(total % modulus)
        rows.append(tuple(row))
    return tuple(rows)


def _matrices_commute_under_filters(
    left_matrix: Matrix,
    left_filters: Tuple[Tuple[Tuple[int, ...], ...], ...],
    right_matrix: Matrix,
    specialization_filters: Tuple[Tuple[int, int], ...],
) -> bool:
    if specialization_filters:
        right_filters = tuple(
            _specialize_matrix(right_matrix, modulus, q_value) for modulus, q_value in specialization_filters
        )
        for (modulus, _), left_specialization, right_specialization in zip(
            specialization_filters,
            left_filters,
            right_filters,
        ):
            if _multiply_specialized(left_specialization, right_specialization, modulus) != _multiply_specialized(
                right_specialization,
                left_specialization,
                modulus,
            ):
                return False
    return left_matrix * right_matrix == right_matrix * left_matrix


def conjugate_matrix_by_generator(matrix: Matrix, letter: int) -> Matrix:
    if letter == 0:
        raise ValueError("braid generators are 1-indexed and nonzero")
    if abs(letter) > matrix.width:
        raise ValueError("generator index out of range for matrix size")
    if letter > 0:
        return _conjugate_positive(matrix, letter)
    return _conjugate_negative(matrix, -letter)


def _conjugate_positive(matrix: Matrix, generator_index: int) -> Matrix:
    size = matrix.width
    pivot = generator_index - 1
    original = matrix.rows
    temp = [list(row) for row in original]
    if generator_index == 1:
        for row in range(size):
            column0 = original[row][0]
            column1 = original[row][1]
            temp[row][0] = _combine_polys((column0, -1, 1))
            temp[row][1] = _combine_polys((column0, 1, 1), (column1, 1, 0))
    elif generator_index == size:
        for row in range(size):
            left = original[row][pivot - 1]
            center = original[row][pivot]
            temp[row][pivot - 1] = _combine_polys((left, 1, 0), (center, 1, 0))
            temp[row][pivot] = _combine_polys((center, -1, 1))
    else:
        for row in range(size):
            left = original[row][pivot - 1]
            center = original[row][pivot]
            right = original[row][pivot + 1]
            temp[row][pivot - 1] = _combine_polys((left, 1, 0), (center, 1, 0))
            temp[row][pivot] = _combine_polys((center, -1, 1))
            temp[row][pivot + 1] = _combine_polys((center, 1, 1), (right, 1, 0))

    updated = [list(row) for row in temp]
    if generator_index == 1:
        for col in range(size):
            updated[0][col] = _combine_polys((temp[0][col], -1, -1), (temp[1][col], 1, 0))
    elif generator_index == size:
        for col in range(size):
            updated[pivot][col] = _combine_polys((temp[pivot - 1][col], 1, -1), (temp[pivot][col], -1, -1))
    else:
        for col in range(size):
            updated[pivot][col] = _combine_polys(
                (temp[pivot - 1][col], 1, -1),
                (temp[pivot][col], -1, -1),
                (temp[pivot + 1][col], 1, 0),
            )
    return Matrix(updated)


def _conjugate_negative(matrix: Matrix, generator_index: int) -> Matrix:
    size = matrix.width
    pivot = generator_index - 1
    original = matrix.rows
    temp = [list(row) for row in original]
    if generator_index == 1:
        for row in range(size):
            column0 = original[row][0]
            column1 = original[row][1]
            temp[row][0] = _combine_polys((column0, -1, -1))
            temp[row][1] = _combine_polys((column0, 1, 0), (column1, 1, 0))
    elif generator_index == size:
        for row in range(size):
            left = original[row][pivot - 1]
            center = original[row][pivot]
            temp[row][pivot - 1] = _combine_polys((left, 1, 0), (center, 1, -1))
            temp[row][pivot] = _combine_polys((center, -1, -1))
    else:
        for row in range(size):
            left = original[row][pivot - 1]
            center = original[row][pivot]
            right = original[row][pivot + 1]
            temp[row][pivot - 1] = _combine_polys((left, 1, 0), (center, 1, -1))
            temp[row][pivot] = _combine_polys((center, -1, -1))
            temp[row][pivot + 1] = _combine_polys((center, 1, 0), (right, 1, 0))

    updated = [list(row) for row in temp]
    if generator_index == 1:
        for col in range(size):
            updated[0][col] = _combine_polys((temp[0][col], -1, 1), (temp[1][col], 1, 1))
    elif generator_index == size:
        for col in range(size):
            updated[pivot][col] = _combine_polys((temp[pivot - 1][col], 1, 0), (temp[pivot][col], -1, 1))
    else:
        for col in range(size):
            updated[pivot][col] = _combine_polys(
                (temp[pivot - 1][col], 1, 0),
                (temp[pivot][col], -1, 1),
                (temp[pivot + 1][col], 1, 1),
            )
    return Matrix(updated)


@lru_cache(maxsize=None)
def _generator_matrix_for_word(n: int, letter: int, ring: CoefficientRing) -> Matrix:
    from .braid import burau_generator_matrix

    return burau_generator_matrix(n, letter, ring)


def _combine_polys(*pieces: tuple[LaurentPoly, int, int]) -> LaurentPoly:
    ring = pieces[0][0].ring
    result: dict[int, int] = {}
    for poly, coefficient, shift in pieces:
        scaled = ring.normalize(coefficient)
        if poly.is_zero() or ring.is_zero(scaled):
            continue
        for exponent, poly_coefficient in poly.terms:
            new_exponent = exponent + shift
            new_coefficient = ring.mul(poly_coefficient, scaled)
            accumulated = ring.add(result.get(new_exponent, ring.zero), new_coefficient)
            if ring.is_zero(accumulated):
                result.pop(new_exponent, None)
            else:
                result[new_exponent] = accumulated
    if not result:
        return LaurentPoly.zero(ring)
    return LaurentPoly._from_terms(ring, tuple(sorted(result.items())))
