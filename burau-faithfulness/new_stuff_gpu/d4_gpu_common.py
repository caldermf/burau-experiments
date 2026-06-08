from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
D4_DIR = ROOT / "Bucket_D4"
if str(D4_DIR) not in sys.path:
    sys.path.insert(0, str(D4_DIR))

import setup_d4 as ct  # noqa: E402


DUAL_ATOMS = [
    [-1],
    [-2],
    [-3],
    [-4],
    [1, -2, -1],
    [2, -3, -2],
    [2, -4, -2],
    [1, 2, -3, -2, -1],
    [1, 2, -4, -2, -1],
    [-3, -4, -2, 4, 3],
    [-4, -3, 1, -2, -1, 3, 4],
    [2, -4, -3, 1, -2, -1, 3, 4, -2],
]

GAMMA_WORD = [-4, -3, -2, -1]
RANK = len(ct.positive_letters)

_VECTOR_CACHE: dict[tuple[tuple[int, ...], int], list[dict[int, int]]] = {}


def compute_oburau_vector(braid_word, base_vertex: int):
    key = (tuple(int(x) for x in braid_word), int(base_vertex))
    cached = _VECTOR_CACHE.get(key)
    if cached is not None:
        return [poly.copy() for poly in cached]

    vector = ct.dim_vectors[base_vertex]
    for letter in reversed(braid_word):
        vector = ct.oburau_fns[letter](vector)

    _VECTOR_CACHE[key] = [poly.copy() for poly in vector]
    return [poly.copy() for poly in vector]


def compute_oburau_deg(braid_word) -> int:
    return max(ct.topdeg_vector(compute_oburau_vector(braid_word, base_vertex)) for base_vertex in ct.positive_letters)


def equal_braids(braid1, braid2) -> bool:
    return all(
        compute_oburau_vector(braid1, base_vertex) == compute_oburau_vector(braid2, base_vertex)
        for base_vertex in ct.positive_letters
    )


def build_garside_gens(verbose: bool = False):
    garside_gens = [[]] + [atom[:] for atom in DUAL_ATOMS]
    additions = 1
    round_idx = 0
    while additions != 0:
        round_idx += 1
        to_add = []
        for left in garside_gens:
            for right in garside_gens:
                candidate = left + right
                if compute_oburau_deg(candidate) > 1:
                    continue

                is_new = True
                for existing in to_add:
                    if equal_braids(existing, candidate):
                        if len(candidate) < len(existing):
                            to_add.remove(existing)
                            to_add.append(candidate)
                        is_new = False
                        break

                if not is_new:
                    continue

                for existing in garside_gens:
                    if equal_braids(existing, candidate):
                        if len(candidate) < len(existing):
                            garside_gens.remove(existing)
                            garside_gens.append(candidate)
                        is_new = False
                        break

                if is_new:
                    to_add.append(candidate)

        garside_gens += to_add
        additions = len(to_add)
        if verbose:
            print(f"Added {additions} D4 Garside simples in round {round_idx}.")

    return garside_gens


def find_representative(braid_word, garside_gens):
    for candidate in garside_gens:
        if equal_braids(candidate, braid_word):
            return candidate
    raise ValueError(f"No D4 Garside representative found for {braid_word}")


def build_descents(garside_gens):
    left_descents = {str(gen): [] for gen in garside_gens}
    right_descents = {str(gen): [] for gen in garside_gens}

    for descent in DUAL_ATOMS:
        for gen in garside_gens:
            if compute_oburau_deg(gen + descent) <= 1:
                rep = find_representative(gen + descent, garside_gens)
                right_descents[str(rep)].append(descent)

            if compute_oburau_deg(descent + gen) <= 1:
                rep = find_representative(descent + gen, garside_gens)
                left_descents[str(rep)].append(descent)

    return left_descents, right_descents


def build_automaton(garside_gens, right_descents, gamma_rep):
    automaton = {}
    for target in garside_gens:
        automaton[str(target)] = []
        if target == [] or target == gamma_rep:
            continue

        for source in garside_gens:
            if source == [] or source == gamma_rep:
                continue

            admissible = True
            for descent in right_descents[str(source)]:
                if descent != [] and descent != gamma_rep and compute_oburau_deg(descent + target) <= 1:
                    admissible = False
                    break

            if admissible:
                automaton[str(target)].append(source)

    return automaton


def build_automaton_indices(simple_words, automaton):
    simple_to_index = {tuple(word): idx for idx, word in enumerate(simple_words)}
    automaton_idx = {}
    for source_word in simple_words:
        if source_word == []:
            continue
        source_idx = simple_to_index[tuple(source_word)]
        targets = automaton.get(str(source_word), [])
        automaton_idx[source_idx] = [simple_to_index[tuple(target)] for target in targets]
    return automaton_idx


def make_fp_vec(vec, p: int):
    if p == 0:
        return [poly.copy() for poly in vec]

    out = []
    for poly in vec:
        new_poly = {}
        for degree, coeff in poly.items():
            new_coeff = coeff % p
            if new_coeff != 0:
                new_poly[degree] = new_coeff
        out.append(new_poly)
    return out


def dense_from_dim_vec(vec, state_width: int, p: int, dtype):
    dense = np.zeros((RANK, state_width), dtype=dtype)
    for coord, poly in enumerate(vec):
        for degree, coeff in poly.items():
            if degree < 0 or degree >= state_width:
                raise ValueError(f"Degree {degree} exceeds configured state width {state_width}")
            dense[coord, degree] = coeff % p if p else coeff
    return dense


def flatten_factors(simple_indices, simple_words):
    return [letter for idx in simple_indices for letter in simple_words[int(idx)]]


def invert_word(word):
    return [-int(letter) for letter in reversed(word)]


def commutator_word(beta_word, base_vertex: int):
    return list(beta_word) + [base_vertex] + invert_word(beta_word) + [-base_vertex]


def build_letter_rules():
    rules = {}
    for letter in ct.all_letters:
        index = abs(letter) - 1
        i_sgn = -1 if letter < 0 else 1
        terms = [(index, -i_sgn, -1)]

        for source_coord in range(RANK):
            if source_coord == index:
                continue
            if ct.coxeter_matrix[index, source_coord] <= 2:
                continue

            if ct.exists_dynkin_ograph_edge(index + 1, source_coord + 1):
                shift = 0 if letter > 0 else -i_sgn
            elif ct.exists_dynkin_ograph_edge(source_coord + 1, index + 1):
                shift = -i_sgn if letter > 0 else 0
            else:
                continue
            terms.append((source_coord, shift, -1))

        rules[letter] = {"row": index, "terms": terms}
    return rules


def build_simple_action_tensors(simple_words, p: int, dtype):
    actions = np.zeros((len(simple_words), RANK, RANK, 2), dtype=dtype)

    for simple_idx, simple_word in enumerate(simple_words):
        for basis_vertex in ct.positive_letters:
            image = compute_oburau_vector(simple_word, basis_vertex)
            for out_coord, poly in enumerate(image):
                for degree, coeff in poly.items():
                    if degree < 0 or degree > 1:
                        raise ValueError(
                            f"Simple {simple_word} has degree {degree}; expected only degrees 0 and 1"
                        )
                    actions[simple_idx, out_coord, basis_vertex - 1, degree] = coeff

    return actions
