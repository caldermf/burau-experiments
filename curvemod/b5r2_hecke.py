from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import laurent


Laurent = Dict[int, int]
PolyMatrix = List[List[Laurent]]
Diagram = Tuple[Tuple[str, str], ...]


DIMENSION = 5
N_STRANDS = 5
GENERATOR_RANGE = (1, 2, 3, 4)

# Temperley-Lieb cell basis for TL_5 on the one-through-line cell module,
# i.e. the Hecke representation labelled by the partition (3,2).
#
# The basis diagrams are the five monic noncrossing (5,1)-diagrams, encoded by
# pairings on the boundary points t1,...,t5,b1 in cyclic order. This is exactly
# the 5-dimensional irreducible two-row representation for B_5.
BASIS_DIAGRAMS: tuple[Diagram, ...] = (
    (("b1", "t5"), ("t1", "t2"), ("t3", "t4")),
    (("b1", "t3"), ("t1", "t2"), ("t4", "t5")),
    (("b1", "t5"), ("t1", "t4"), ("t2", "t3")),
    (("b1", "t1"), ("t2", "t3"), ("t4", "t5")),
    (("b1", "t1"), ("t2", "t5"), ("t3", "t4")),
)

BASIS_LOOKUP = {diagram: index for index, diagram in enumerate(BASIS_DIAGRAMS)}
DELTA: Laurent = {-1: 1, 1: 1}
Q_POS: Laurent = {1: 1}
Q_NEG: Laurent = {-1: 1}


def _trim(pol: Laurent) -> Laurent:
    return laurent.trim(pol)


def _add(a: Laurent, b: Laurent) -> Laurent:
    return _trim(laurent.addition(a, b))


def _mul(a: Laurent, b: Laurent) -> Laurent:
    return _trim(laurent.product(a, b))


def _scale(pol: Laurent, scalar: int) -> Laurent:
    return {degree: scalar * coeff for degree, coeff in pol.items() if scalar * coeff != 0}


def _zero_matrix() -> PolyMatrix:
    return [[{} for _ in range(DIMENSION)] for _ in range(DIMENSION)]


def identity_matrix() -> PolyMatrix:
    matrix = _zero_matrix()
    for index in range(DIMENSION):
        matrix[index][index] = {0: 1}
    return matrix


def add_matrices(left: Sequence[Sequence[Laurent]], right: Sequence[Sequence[Laurent]]) -> PolyMatrix:
    out = _zero_matrix()
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            out[row][col] = _add(left[row][col], right[row][col])
    return out


def subtract_matrices(left: Sequence[Sequence[Laurent]], right: Sequence[Sequence[Laurent]]) -> PolyMatrix:
    out = _zero_matrix()
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            out[row][col] = _add(left[row][col], _scale(right[row][col], -1))
    return out


def multiply_matrices(left: Sequence[Sequence[Laurent]], right: Sequence[Sequence[Laurent]]) -> PolyMatrix:
    out = _zero_matrix()
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            accum: Laurent = {}
            for mid in range(DIMENSION):
                accum = _add(accum, _mul(left[row][mid], right[mid][col]))
            out[row][col] = accum
    return out


def reduce_matrix(matrix: Sequence[Sequence[Laurent]], modulus: int) -> PolyMatrix:
    out = _zero_matrix()
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            reduced: Laurent = {}
            for degree, coeff in matrix[row][col].items():
                value = coeff % modulus
                if value != 0:
                    reduced[degree] = value
            out[row][col] = reduced
    return out


def matrix_is_zero(matrix: Sequence[Sequence[Laurent]]) -> bool:
    return all(len(entry) == 0 for row in matrix for entry in row)


def matrix_is_identity(matrix: Sequence[Sequence[Laurent]]) -> bool:
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            expected = {0: 1} if row == col else {}
            if matrix[row][col] != expected:
                return False
    return True


def matrix_support_bounds(matrix: Sequence[Sequence[Laurent]]) -> tuple[int, int, int]:
    degrees = [degree for row in matrix for entry in row for degree in entry]
    if not degrees:
        return 0, -1, -1
    bottom = min(degrees)
    top = max(degrees)
    return bottom, top, top - bottom


def matrix_projlen(matrix: Sequence[Sequence[Laurent]]) -> int:
    _, _, spread = matrix_support_bounds(matrix)
    return spread


def normalize_matrix(matrix: Sequence[Sequence[Laurent]]) -> PolyMatrix:
    valuation, _, _ = matrix_support_bounds(matrix)
    if valuation <= 0:
        shift = -valuation
    else:
        shift = -valuation
    if shift == 0:
        return [[dict(entry) for entry in row] for row in matrix]
    out = _zero_matrix()
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            out[row][col] = {degree + shift: coeff for degree, coeff in matrix[row][col].items()}
    return out


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _temperley_lieb_generator_diagram(index: int) -> Diagram:
    pairs: list[tuple[str, str]] = []
    for strand in range(1, N_STRANDS + 1):
        if strand in (index, index + 1):
            continue
        pairs.append((f"t{strand}", f"b{strand}"))
    pairs.append((f"t{index}", f"t{index + 1}"))
    pairs.append((f"b{index}", f"b{index + 1}"))
    return tuple(sorted(_canonical_pair(a, b) for a, b in pairs))


def _compose_diagrams(top: Diagram, bottom: Diagram) -> tuple[int, Diagram]:
    """
    Compose a TL(5,5) diagram with a TL(5,1) basis diagram.

    Returns (number_of_loops, resulting_basis_diagram).
    """
    adjacency: dict[str, set[str]] = defaultdict(set)

    def add_edge(left: str, right: str) -> None:
        adjacency[left].add(right)
        adjacency[right].add(left)

    for left, right in top:
        add_edge(left, right)

    for left, right in bottom:
        renamed_left = "x" + left[1:] if left.startswith("t") else "k" + left[1:]
        renamed_right = "x" + right[1:] if right.startswith("t") else "k" + right[1:]
        add_edge(renamed_left, renamed_right)

    for strand in range(1, N_STRANDS + 1):
        add_edge(f"b{strand}", f"x{strand}")

    external = {f"t{strand}" for strand in range(1, N_STRANDS + 1)}
    external.add("k1")

    loops = 0
    result_pairs: list[tuple[str, str]] = []
    seen: set[str] = set()

    for start in list(adjacency):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[str] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)

        boundary = [node for node in component if node in external]
        if len(boundary) == 0:
            loops += 1
            continue
        if len(boundary) != 2:
            raise RuntimeError(f"Unexpected TL component {component} with boundary {boundary}")

        left, right = sorted(boundary)
        mapped_left = left if left.startswith("t") else "b" + left[1:]
        mapped_right = right if right.startswith("t") else "b" + right[1:]
        result_pairs.append(_canonical_pair(mapped_left, mapped_right))

    result = tuple(sorted(result_pairs))
    if result not in BASIS_LOOKUP:
        raise RuntimeError(f"Composed diagram {result} is not in the (3,2) cell basis")
    return loops, result


def e_matrix(index: int) -> PolyMatrix:
    if index not in GENERATOR_RANGE:
        raise ValueError("index must be one of 1,2,3,4")

    matrix = _zero_matrix()
    generator = _temperley_lieb_generator_diagram(index)
    for column, basis_diagram in enumerate(BASIS_DIAGRAMS):
        loops, output = _compose_diagrams(generator, basis_diagram)
        row = BASIS_LOOKUP[output]
        coeff = DELTA if loops else {0: 1}
        matrix[row][column] = coeff
    return matrix


def artin_generator_matrix(letter: int) -> PolyMatrix:
    """
    Exact Laurent-polynomial matrices for the Hecke/Braid group generators.

    We use the Temperley-Lieb quotient normalization
      sigma_i   = I - q e_i
      sigma_i^-1 = I - q^-1 e_i
    with delta = q + q^-1, so the inverse relation is exact because e_i^2 = delta e_i.
    """
    if letter == 0 or abs(letter) > 4:
        raise ValueError("letter must be one of +/-1, +/-2, +/-3, +/-4")

    matrix = identity_matrix()
    e = e_matrix(abs(letter))
    q_factor = Q_POS if letter > 0 else Q_NEG
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            correction = _mul(q_factor, e[row][col])
            matrix[row][col] = _add(matrix[row][col], _scale(correction, -1))
    return matrix


def sigma_matrix(index: int = 1) -> PolyMatrix:
    return artin_generator_matrix(index)


def sigma_inverse_matrix(index: int = 1) -> PolyMatrix:
    return artin_generator_matrix(-index)


def shift_matrix(matrix: Sequence[Sequence[Laurent]], degree_shift: int) -> PolyMatrix:
    out = _zero_matrix()
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            out[row][col] = {degree + degree_shift: coeff for degree, coeff in matrix[row][col].items()}
    return out


def evaluate_word(word: Sequence[int], modulus: int | None = None) -> PolyMatrix:
    matrix = identity_matrix()
    for letter in word:
        matrix = multiply_matrices(matrix, artin_generator_matrix(letter))
        if modulus is not None:
            matrix = reduce_matrix(matrix, modulus)
    return matrix


def evaluate_inverse_word(word: Sequence[int], modulus: int | None = None) -> PolyMatrix:
    matrix = identity_matrix()
    for letter in reversed(word):
        matrix = multiply_matrices(matrix, artin_generator_matrix(-letter))
        if modulus is not None:
            matrix = reduce_matrix(matrix, modulus)
    return matrix


def commutator_matrix(word: Sequence[int], modulus: int | None = None, generator: int = 1) -> PolyMatrix:
    """
    Return rho([sigma_generator, w]) = sigma_generator * w * sigma_generator^-1 * w^-1.
    """
    matrix = multiply_matrices(
        multiply_matrices(
            multiply_matrices(
                sigma_matrix(generator),
                evaluate_word(word, modulus=modulus),
            ),
            sigma_inverse_matrix(generator),
        ),
        evaluate_inverse_word(word, modulus=modulus),
    )
    if modulus is not None:
        matrix = reduce_matrix(matrix, modulus)
    return matrix


def commutator_is_identity(word: Sequence[int], modulus: int, generator: int = 1) -> bool:
    return matrix_is_identity(commutator_matrix(word, modulus=modulus, generator=generator))
