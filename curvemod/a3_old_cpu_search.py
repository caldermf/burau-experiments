from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import setup_a3 as ct


@dataclass
class OldSearchConfig:
    cap_1: int = 500
    cap_2: int = 500
    total_cap_1: int = 50000
    total_cap_2: int = 50000
    first_steps: int = 12
    modulus: int = 5
    max_g_length: int = 50
    seed: int = 0


@dataclass
class OldSearchResult:
    witness: Optional[List[List[int]]]
    depth: Optional[int]
    runtime_seconds: float
    found: bool


DUALS: List[List[int]] = [
    [-1],
    [-2],
    [-3],
    [1, -2, -1],
    [2, -3, -2],
    [1, 2, -3, -2, -1],
]


def compute_oburau_vector(word: List[int], base_point: int):
    vec = ct.dim_vectors[base_point]
    for letter in reversed(word):
        vec = ct.oburau_fns[letter](vec)
    return vec


def compute_oburau_degree(word: List[int]) -> float:
    return max(ct.topdeg_vector(compute_oburau_vector(word, base_point)) for base_point in ct.positive_letters)


def equal_braids(word1: List[int], word2: List[int]) -> bool:
    for base_point in ct.positive_letters:
        if compute_oburau_vector(word1, base_point) != compute_oburau_vector(word2, base_point):
            return False
    return True


def make_fp(pol: Dict[int, int], modulus: int) -> Dict[int, int]:
    if modulus == 0:
        return dict(pol)
    out = {}
    for degree, coeff in pol.items():
        value = coeff % modulus
        if value != 0:
            out[degree] = value
    return out


def make_fp_vec(vec, modulus: int):
    return [make_fp(pol, modulus) for pol in vec]


def build_garside_data():
    garside_gens = [[]] + [word[:] for word in DUALS]
    number = 1
    while number != 0:
        to_add: List[List[int]] = []
        for left in garside_gens:
            for right in garside_gens:
                candidate = left + right
                if compute_oburau_degree(candidate) <= 1:
                    fresh = True
                    for existing in list(to_add):
                        if equal_braids(existing, candidate):
                            if len(candidate) < len(existing):
                                to_add.remove(existing)
                                to_add.append(candidate)
                            fresh = False
                            break
                    if fresh:
                        for existing in list(garside_gens):
                            if equal_braids(existing, candidate):
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
            if equal_braids(candidate, word):
                return candidate
        raise RuntimeError(f"No representative found for {word}")

    delta = find_representative([-3, -2, -1])
    right_descents: Dict[str, List[List[int]]] = {}
    left_descents: Dict[str, List[List[int]]] = {}
    for descent in DUALS:
        for gen in garside_gens:
            if compute_oburau_degree(gen + descent) <= 1:
                rep = find_representative(gen + descent)
                right_descents.setdefault(str(rep), []).append(descent)
            if compute_oburau_degree(descent + gen) <= 1:
                rep = find_representative(descent + gen)
                left_descents.setdefault(str(rep), []).append(descent)

    automaton: Dict[str, List[List[int]]] = {}
    for y in garside_gens:
        automaton[str(y)] = []
        if y != [] and y != delta:
            for x in garside_gens:
                if x != [] and y != delta:
                    admissible = True
                    for descent in right_descents[str(x)]:
                        if descent != [] and descent != delta and compute_oburau_degree(descent + y) <= 1:
                            admissible = False
                            break
                    if admissible:
                        automaton[str(y)].append(x)

    return garside_gens, delta, right_descents, automaton


def run_old_search(config: OldSearchConfig) -> OldSearchResult:
    if config.modulus <= 0:
        raise ValueError("Old benchmark search currently supports only positive modulus.")

    random.seed(config.seed)
    start = time.time()

    garside_gens, delta, right_descents, automaton = build_garside_data()

    buckets = {1: {}}
    for word in garside_gens:
        if word == [] or word == delta:
            continue
        valid_start = True
        for descent in right_descents[str(word)]:
            descent_vec = compute_oburau_vector(descent, 1)
            if ct.poly_normalize_vector(descent_vec) == ct.dim_vectors[1]:
                valid_start = False
                break
            if ct.topdeg_vector(descent_vec) - ct.botdeg_vector(descent_vec) != 1:
                valid_start = False
                break
        if not valid_start:
            continue

        locburau = make_fp_vec(compute_oburau_vector(word, 1), config.modulus)
        locspread = ct.topdeg_vector(locburau) - ct.botdeg_vector(locburau)
        if locspread == 1:
            buckets[1].setdefault(locspread, []).append([[word], locburau])

    witness = None
    witness_depth = None
    cur = 1
    stop = False
    while cur < config.max_g_length and not stop:
        cur += 1
        next_bucket: Dict[int, List[List[object]]] = {}
        if cur < config.first_steps:
            total_cap = config.total_cap_1
            cap = config.cap_1
        else:
            total_cap = config.total_cap_2
            cap = config.cap_2

        prev = buckets[cur - 1]
        if not prev:
            break

        stop_key = min(prev.keys())
        counter = len(prev[stop_key])
        while counter < total_cap and stop_key + 1 in prev:
            stop_key += 1
            counter += len(prev[stop_key])
        keylist = [key for key in prev.keys() if key <= stop_key]
        keylist.sort()

        for prevdeg in keylist:
            for prevelt in prev[prevdeg]:
                for word in automaton[str(prevelt[0][0])]:
                    locburau = prevelt[1].copy()
                    for letter in reversed(word):
                        locburau = make_fp_vec(ct.oburau_fns[letter](locburau), config.modulus)
                    locspread = ct.topdeg_vector(locburau) - ct.botdeg_vector(locburau)
                    if cur != 2 or (cur == 2 and locspread != 0):
                        if locspread == 0:
                            witness = [word] + prevelt[0]
                            witness_depth = cur
                            stop = True
                        if locspread <= config.max_g_length - cur + 1:
                            bucket = next_bucket.setdefault(locspread, [])
                            candidate = [[word] + prevelt[0], locburau]
                            if len(bucket) < cap:
                                bucket.append(candidate)
                            else:
                                position = random.choice(range(cap + 1))
                                if position < cap:
                                    bucket[position] = candidate
                if stop:
                    break
            if stop:
                break

        buckets[cur] = next_bucket
        buckets[cur - 1] = {}
        if 0 in next_bucket:
            stop = True

    return OldSearchResult(
        witness=witness,
        depth=witness_depth,
        runtime_seconds=time.time() - start,
        found=witness is not None,
    )
