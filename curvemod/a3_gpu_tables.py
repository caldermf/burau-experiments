from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

import setup_a3 as ct


DUAL_WORDS: List[List[int]] = [
    [-1],
    [-2],
    [-3],
    [1, -2, -1],
    [2, -3, -2],
    [1, 2, -3, -2, -1],
]


def _compute_oburau_vector(word: List[int], base_point: int):
    vec = ct.dim_vectors[base_point]
    for letter in reversed(word):
        vec = ct.oburau_fns[letter](vec)
    return vec


def _compute_oburau_degree(word: List[int]) -> float:
    return max(ct.topdeg_vector(_compute_oburau_vector(word, base_point)) for base_point in ct.positive_letters)


def _equal_braids(word1: List[int], word2: List[int]) -> bool:
    for base_point in ct.positive_letters:
        if _compute_oburau_vector(word1, base_point) != _compute_oburau_vector(word2, base_point):
            return False
    return True


def _make_fp(pol: Dict[int, int], p: int) -> Dict[int, int]:
    if p == 0:
        return dict(pol)
    out = {}
    for degree, coeff in pol.items():
        value = coeff % p
        if value != 0:
            out[degree] = value
    return out


def _make_fp_vec(vec, p: int):
    return [_make_fp(pol, p) for pol in vec]


@dataclass
class A3TableData:
    simple_words: List[List[int]]
    dual_words: List[List[int]]
    delta_id: int
    allowed_successors: List[List[int]]
    right_descents: List[List[List[int]]]
    start_simple_ids: List[int]
    allowed_suffix_padded: torch.Tensor
    allowed_count: torch.Tensor
    start_mask: torch.Tensor


def build_a3_table_data(modulus: int, device: torch.device) -> A3TableData:
    garside_gens = [[]] + [word[:] for word in DUAL_WORDS]
    number = 1
    while number != 0:
        to_add: List[List[int]] = []
        for left in garside_gens:
            for right in garside_gens:
                candidate = left + right
                if _compute_oburau_degree(candidate) <= 1:
                    fresh = True
                    for existing in list(to_add):
                        if _equal_braids(existing, candidate):
                            if len(candidate) < len(existing):
                                to_add.remove(existing)
                                to_add.append(candidate)
                            fresh = False
                            break
                    if fresh:
                        for existing in list(garside_gens):
                            if _equal_braids(existing, candidate):
                                if len(candidate) < len(existing):
                                    garside_gens.remove(existing)
                                    garside_gens.append(candidate)
                                fresh = False
                                break
                    if fresh:
                        to_add.append(candidate)
        garside_gens.extend(to_add)
        number = len(to_add)

    def find_representative(word: List[int]) -> List[int]:
        for candidate in garside_gens:
            if _equal_braids(candidate, word):
                return candidate
        raise RuntimeError(f"No representative found for {word}")

    delta_word = find_representative([-3, -2, -1])
    delta_id = next(index for index, word in enumerate(garside_gens) if word == delta_word)

    right_descents_by_word: dict[tuple[int, ...], list[list[int]]] = {}
    left_descents_by_word: dict[tuple[int, ...], list[list[int]]] = {}
    for descent in DUAL_WORDS:
        for gen in garside_gens:
            if _compute_oburau_degree(gen + descent) <= 1:
                rep = find_representative(gen + descent)
                right_descents_by_word.setdefault(tuple(rep), []).append(descent)
            if _compute_oburau_degree(descent + gen) <= 1:
                rep = find_representative(descent + gen)
                left_descents_by_word.setdefault(tuple(rep), []).append(descent)

    allowed_successors: List[List[int]] = []
    right_descents_by_id: List[List[List[int]]] = []
    for current_word in garside_gens:
        key = tuple(current_word)
        right_descents = right_descents_by_word.get(key, [])
        right_descents_by_id.append([descent[:] for descent in right_descents])
        successors: List[int] = []
        if current_word != [] and current_word != delta_word:
            for next_word in garside_gens:
                if next_word == [] or current_word == delta_word:
                    continue
                admissible = True
                for descent in right_descents_by_word.get(tuple(next_word), []):
                    if descent != [] and descent != delta_word and _compute_oburau_degree(descent + current_word) <= 1:
                        admissible = False
                        break
                if admissible:
                    successors.append(garside_gens.index(next_word))
        allowed_successors.append(successors)

    start_simple_ids: List[int] = []
    for simple_id, word in enumerate(garside_gens):
        if word == [] or word == delta_word:
            continue
        valid_start = True
        for descent in right_descents_by_word.get(tuple(word), []):
            descent_vec = _compute_oburau_vector(descent, 1)
            if ct.poly_normalize_vector(descent_vec) == ct.dim_vectors[1]:
                valid_start = False
                break
            if ct.topdeg_vector(descent_vec) - ct.botdeg_vector(descent_vec) != 1:
                valid_start = False
                break
        if not valid_start:
            continue

        image = _make_fp_vec(_compute_oburau_vector(word, 1), modulus)
        spread = ct.topdeg_vector(image) - ct.botdeg_vector(image)
        if spread == 1:
            start_simple_ids.append(simple_id)

    max_allowed = max((len(successors) for successors in allowed_successors), default=0)
    if max_allowed == 0:
        allowed_suffix_padded = torch.empty((len(garside_gens), 0), dtype=torch.long, device=device)
    else:
        allowed_suffix_padded = torch.full((len(garside_gens), max_allowed), -1, dtype=torch.long, device=device)
        for row, successors in enumerate(allowed_successors):
            if successors:
                allowed_suffix_padded[row, : len(successors)] = torch.tensor(successors, dtype=torch.long, device=device)

    allowed_count = torch.tensor([len(successors) for successors in allowed_successors], dtype=torch.long, device=device)
    start_mask = torch.zeros(len(garside_gens), dtype=torch.bool, device=device)
    if start_simple_ids:
        start_mask[torch.tensor(start_simple_ids, dtype=torch.long, device=device)] = True

    return A3TableData(
        simple_words=[word[:] for word in garside_gens],
        dual_words=[word[:] for word in DUAL_WORDS],
        delta_id=delta_id,
        allowed_successors=allowed_successors,
        right_descents=right_descents_by_id,
        start_simple_ids=start_simple_ids,
        allowed_suffix_padded=allowed_suffix_padded,
        allowed_count=allowed_count,
        start_mask=start_mask,
    )
