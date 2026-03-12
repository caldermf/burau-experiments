"""Search-space generation for the D4 Burau mod-p search described in the paper.

This module implements the paper's Section 4 weight arithmetic, the corrected
parity convention from the prose/examples, the three admissible zero-weight
cases used to exclude proper multicurves, and exhaustive candidate generation
up to a target geometric intersection bound.

The Burau-polynomial evaluators are exposed with the public interfaces from the
plan, but the paper does not give the per-path monomial update table in textual
form. Those evaluators therefore remain explicit TODOs instead of embedding
unverified guessed formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


WeightKey = Tuple[int, int, int, int, int]


def _gcd4(a: int, b: int, c: int, d: int) -> int:
    return gcd(gcd(abs(a), abs(b)), gcd(abs(c), abs(d)))


def _ceil_even(n: int) -> int:
    return n if n % 2 == 0 else n + 1


def _floor_even(n: int) -> int:
    return n if n % 2 == 0 else n - 1


def _ceil_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def _iter_even(start: int, stop: int) -> Iterator[int]:
    current = _ceil_even(start)
    while current <= stop:
        yield current
        current += 2


def _iter_odd(start: int, stop: int) -> Iterator[int]:
    current = _ceil_odd(start)
    while current <= stop:
        yield current
        current += 2


@dataclass(frozen=True)
class WeightTuple:
    """Primary free weights from the paper."""

    w0: int
    w1: int
    w2: int
    w3: int
    w14: int

    @property
    def h(self) -> int:
        if self.w14 % 2 != 0:
            raise ValueError("w14 must be even.")
        return self.w14 // 2

    @property
    def w4(self) -> int:
        return 2 * self.w0

    @property
    def w5(self) -> int:
        return 2 * self.w1

    @property
    def w6(self) -> int:
        return 2 * self.w2

    @property
    def w7(self) -> int:
        return 2 * self.w3

    @property
    def w8(self) -> int:
        return self.w0 + self.w1 - self.h

    @property
    def w9(self) -> int:
        return -self.w0 + self.w1 + self.h

    @property
    def w10(self) -> int:
        return self.w0 - self.w1 + self.h

    @property
    def w11(self) -> int:
        return self.w2 - self.w3 + self.h

    @property
    def w12(self) -> int:
        return self.w2 + self.w3 - self.h

    @property
    def w13(self) -> int:
        return -self.w2 + self.w3 + self.h

    def key(self) -> WeightKey:
        return (self.w0, self.w1, self.w2, self.w3, self.w14)

    def gcd4(self) -> int:
        return _gcd4(self.w0, self.w1, self.w2, self.w3)

    def expected_intersections(self) -> int:
        return self.w0 // 2 + self.w1 // 2 - min(self.w1, self.w8)

    def zero_cases(self) -> Tuple[bool, bool, bool]:
        return (self.w8 == 0, self.w9 == 0, self.w12 == 0)

    def has_required_parity(self) -> bool:
        # Corrected from the paper's inconsistent displayed D2/D3 line:
        # use the prose and Example 5.3.
        return (
            self.w0 % 2 == 0
            and self.w1 % 2 == 0
            and self.w2 % 2 == 1
            and self.w3 % 2 == 1
            and self.w14 % 2 == 0
        )

    def satisfies_nonnegativity(self) -> bool:
        h = self.h
        return (
            h <= self.w0 + self.w1
            and self.w0 <= self.w1 + h
            and self.w1 <= self.w0 + h
            and self.w3 <= self.w2 + h
            and h <= self.w2 + self.w3
            and self.w2 <= self.w3 + h
        )

    def satisfies_initial_terminal_conditions(self) -> bool:
        h = self.h
        i2 = self.w1 + h < self.w0 + self.w2 or self.w1 < self.w2
        i3 = (
            self.w0 + self.w2 < self.w1 + h
            or self.w0 + self.w2 < self.w14
            or self.w0 + self.w1 + self.w2 < 3 * h
            or self.w1 + self.w2 < self.w14
        )
        t4 = self.w0 + self.w1 < self.w3 + h or self.w1 < self.w3
        return (
            self.w3 < h
            and i2
            and i3
            and self.w2 < h
            and self.w1 + self.w3 < self.w0 + h
            and self.w3 < self.w0
            and t4
        )

    def satisfies_general_conditions(self) -> bool:
        return (
            self.has_required_parity()
            and self.satisfies_nonnegativity()
            and self.satisfies_initial_terminal_conditions()
        )

    def is_admissible_arc_candidate(self) -> bool:
        has_zero_weight_case = any(self.zero_cases())
        return (
            self.satisfies_general_conditions()
            and self.gcd4() == 1
            and has_zero_weight_case
        )


@dataclass(frozen=True)
class Candidate:
    case: str
    intersections: int
    weights: WeightTuple


def left_thresholds(weights: WeightTuple) -> Tuple[int, ...]:
    return (
        min(weights.w1, weights.w8, weights.w9),
        min(weights.w1, weights.w9),
        weights.w9,
        weights.w0 + weights.w9 - weights.w8,
        weights.w10 + weights.w9 - weights.w8,
        weights.w1 + weights.w10 - weights.w8,
        weights.w14 - weights.w1,
        min(weights.w10, weights.w0 + weights.w10 - weights.w8, 2 * weights.w10 - weights.w8),
    )


def right_thresholds(weights: WeightTuple) -> Tuple[int, ...]:
    return (
        min(weights.w3, weights.w12, weights.w13),
        min(weights.w3, weights.w13),
        weights.w13,
        weights.w2 + weights.w13 - weights.w12,
        weights.w11 + weights.w13 - weights.w12,
    )


def left_path_index(weights: WeightTuple, ell: int) -> int:
    for index, threshold in enumerate(left_thresholds(weights), start=1):
        if ell < threshold:
            return index
    raise ValueError("Left path index undefined for the given ell.")


def right_path_index(weights: WeightTuple, ell: int) -> int:
    for index, threshold in enumerate(right_thresholds(weights), start=1):
        if ell < threshold:
            return index
    return 6


def is_right_terminal_state(weights: WeightTuple, ell: int) -> bool:
    return (
        (ell == weights.w3 and ell < weights.w13)
        or (ell > right_thresholds(weights)[-1] and ell == 3 * weights.w3 + weights.w14)
    )


def _validate_case_i(weights: WeightTuple, target_c: int) -> bool:
    return (
        weights.w8 == 0
        and weights.w0 + weights.w1 <= weights.w2 + weights.w3
        and weights.w1 < weights.w2
        and weights.w2 < weights.w0 + weights.w1
        and weights.w3 < weights.w0
        and weights.expected_intersections() == target_c
        and weights.is_admissible_arc_candidate()
    )


def _validate_case_ii(weights: WeightTuple, target_c: int) -> bool:
    i1 = weights.w0 < weights.w2 + 2 * weights.w1 or 4 * weights.w1 + weights.w2 < 2 * weights.w0
    return (
        weights.w9 == 0
        and weights.w0 <= weights.w1 + weights.w2 + weights.w3
        and i1
        and weights.w1 < weights.w2
        and weights.w1 + weights.w3 < weights.w0
        and weights.w1 + weights.w2 < weights.w0
        and weights.w1 < weights.w3
        and weights.expected_intersections() == target_c
        and weights.is_admissible_arc_candidate()
    )


def _validate_case_iii(weights: WeightTuple, target_c: int) -> bool:
    i2 = weights.w1 + weights.w3 < weights.w0 or weights.w1 < weights.w2
    i3 = (
        weights.w0 < weights.w1 + weights.w3
        or weights.w2 + 2 * weights.w3 < weights.w0
        or weights.w0 + weights.w1 < 2 * weights.w2 + 3 * weights.w3
    )
    t4 = weights.w0 + weights.w1 < weights.w2 + 2 * weights.w3 or weights.w1 < weights.w3
    return (
        weights.w12 == 0
        and weights.w2 + weights.w3 <= weights.w0 + weights.w1
        and weights.w0 <= weights.w1 + weights.w2 + weights.w3
        and i2
        and i3
        and weights.w1 < weights.w2 + weights.w0
        and weights.w3 < weights.w0
        and t4
        and weights.expected_intersections() == target_c
        and weights.is_admissible_arc_candidate()
    )


def _generate_case_i(target_c: int) -> Iterator[Candidate]:
    w14 = 4 * target_c
    for w0 in _iter_even(2, 2 * target_c):
        w1 = 2 * target_c - w0
        max_w3 = min(w0 - 1, 2 * target_c - 1)
        for w3 in _iter_odd(1, max_w3):
            lower_w2 = max(w1 + 1, 2 * target_c - w3)
            for w2 in _iter_odd(lower_w2, 2 * target_c - 1):
                weights = WeightTuple(w0=w0, w1=w1, w2=w2, w3=w3, w14=w14)
                if _validate_case_i(weights, target_c):
                    yield Candidate(case="I", intersections=target_c, weights=weights)


def _generate_case_ii(target_c: int) -> Iterator[Candidate]:
    w14 = 4 * target_c
    for w2 in _iter_odd(1, 2 * target_c - 1):
        for w3 in _iter_odd(max(1, 2 * target_c - w2), 2 * target_c - 1):
            if w2 + w3 < 2 * target_c:
                continue
            max_w1 = min(w2, w3) - 1
            for w1 in _iter_even(0, max_w1):
                w0 = w1 + 2 * target_c
                weights = WeightTuple(w0=w0, w1=w1, w2=w2, w3=w3, w14=w14)
                if _validate_case_ii(weights, target_c):
                    yield Candidate(case="II", intersections=target_c, weights=weights)


def _generate_case_iii(target_c: int) -> Iterator[Candidate]:
    for w2 in _iter_odd(1, 2 * target_c - 1):
        for w3 in _iter_odd(1, 2 * target_c - 1):
            h = w2 + w3
            w14 = 2 * h

            # Subcase IIIa: h <= w0 and w1 = w0 - 2c.
            lower_a = max(2 * target_c, h, w3 + 1)
            upper_a = w3 + 2 * target_c - 1
            for w0 in _iter_even(lower_a, upper_a):
                w1 = w0 - 2 * target_c
                weights = WeightTuple(w0=w0, w1=w1, w2=w2, w3=w3, w14=w14)
                if _validate_case_iii(weights, target_c):
                    yield Candidate(case="IIIa", intersections=target_c, weights=weights)

            # Subcase IIIb: h > w0 and w1 = 2(h-c) - w0.
            lower_b = w3 + 1
            upper_b = min(h - 1, w3 + 2 * target_c - 1)
            for w0 in _iter_even(lower_b, upper_b):
                w1 = 2 * (h - target_c) - w0
                if w1 < 0:
                    continue
                weights = WeightTuple(w0=w0, w1=w1, w2=w2, w3=w3, w14=w14)
                if _validate_case_iii(weights, target_c):
                    yield Candidate(case="IIIb", intersections=target_c, weights=weights)


def iter_candidates_by_case(p: int, max_intersections: int) -> Iterator[Candidate]:
    if p <= 1:
        raise ValueError("p must be a prime > 1.")
    for target_c in range(1, max_intersections + 1):
        if p == 2 and target_c % 2 == 1:
            continue
        seen: set[WeightKey] = set()
        for generator in (_generate_case_i, _generate_case_ii, _generate_case_iii):
            for candidate in generator(target_c):
                key = candidate.weights.key()
                if key in seen:
                    continue
                seen.add(key)
                yield candidate


def generate_candidates(p: int, max_intersections: int) -> Iterator[WeightTuple]:
    for candidate in iter_candidates_by_case(p, max_intersections):
        yield candidate.weights


def precheck_candidate(weights: WeightTuple, p: int, fold_bits: int = 7) -> bool:
    """One-sided safe placeholder.

    Returning ``True`` on every candidate is correctness-preserving: the contract
    only requires that ``False`` means "provably not a mod-p zero." The paper's
    folded-array rejection filter belongs here once the path-transition table has
    been transcribed and validated.
    """

    if p <= 1:
        raise ValueError("p must be a prime > 1.")
    if fold_bits <= 0:
        raise ValueError("fold_bits must be positive.")
    if not weights.is_admissible_arc_candidate():
        return False
    return True


def evaluate_burau_exact(weights: WeightTuple) -> Dict[int, int]:
    raise NotImplementedError(
        "The exact Burau evaluator still needs the Figure 9 per-path return-map "
        "and crossing table. The page-13 threshold logic is implemented, but the "
        "paper does not spell out the corresponding delta/crossing data in text."
    )


def reduce_poly_mod_p(poly: Dict[int, int], p: int) -> Dict[int, int]:
    if p <= 1:
        raise ValueError("p must be a prime > 1.")
    reduced: Dict[int, int] = {}
    for exponent, coeff in poly.items():
        coeff_mod_p = coeff % p
        if coeff_mod_p:
            reduced[exponent] = coeff_mod_p
    return reduced


def search_mod_p(
    p: int,
    max_intersections: int,
    workers: int,
    fold_bits: int = 7,
) -> List[Tuple[WeightTuple, Dict[int, int]]]:
    del workers  # Parallel evaluation is deferred until the evaluator exists.

    hits: List[Tuple[WeightTuple, Dict[int, int]]] = []
    for weights in generate_candidates(p, max_intersections):
        if not precheck_candidate(weights, p=p, fold_bits=fold_bits):
            continue
        exact = evaluate_burau_exact(weights)
        reduced = reduce_poly_mod_p(exact, p=p)
        if not reduced:
            hits.append((weights, exact))
    return hits


def find_example_53(max_intersections: int = 48) -> Optional[WeightTuple]:
    target = WeightTuple(w0=22, w1=74, w2=89, w3=21, w14=192)
    for weights in generate_candidates(p=3, max_intersections=max_intersections):
        if weights == target:
            return weights
    return None
