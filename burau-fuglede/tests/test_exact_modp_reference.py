import random

import pytest

from tests.run_exact_modp_validation import (
    exact_zero,
    pairing_eval_mod,
    pairing_poly,
    passes_field_filter,
    single_whisker,
)


def eval_poly_mod(poly, q, p):
    total = 0
    for exp, coeff in poly.items():
        total = (total + coeff * pow(q, exp, p)) % p
    return total


@pytest.mark.parametrize(
    ("tuple5", "expected"),
    [
        ((0, 3, 0, 1, 3), {6: -1, 7: 1, 9: -1, 10: 1}),
        ((0, 7, 0, 3, 5), {16: -1, 17: 1, 18: -1, 20: 1, 21: -1, 22: 1}),
        ((0, 11, 0, 5, 7), {26: -1, 27: 1, 28: -1, 32: 1, 33: -1, 34: 1}),
        ((3, 0, 2, 0, 6), {2: 1, 3: -1, 4: 1, 5: -1, 6: 1, 7: -1}),
        (
            (5, 0, 4, 0, 10),
            {4: 1, 5: -1, 6: 1, 7: -1, 8: 1, 9: -1, 10: 1, 11: -1, 12: 1, 13: -1},
        ),
    ],
)
def test_known_pairing_polynomials(tuple5, expected):
    assert pairing_poly(*tuple5) == expected


@pytest.mark.parametrize("n", [2, 4, 6, 10, 12])
def test_family_closed_form_for_n(tuple5=None, n=None):
    tuple5 = (0, 2 * n - 1, 0, n - 1, n + 1)
    if n == 2:
        expected = {6: -1, 7: 1, 9: -1, 10: 1}
    else:
        expected = {
            5 * n - 4: -1,
            5 * n - 3: 1,
            5 * n - 2: -1,
            6 * n - 4: 1,
            6 * n - 3: -1,
            6 * n - 2: 1,
        }
    assert pairing_poly(*tuple5) == expected


@pytest.mark.parametrize(
    ("tuple5", "is_single"),
    [
        ((0, 3, 0, 1, 3), True),
        ((1, 0, 0, 1, 1), False),
        ((0, 0, 0, 0, 1), True),
    ],
)
def test_single_whisker_examples(tuple5, is_single):
    assert single_whisker(*tuple5) is is_single


def test_pairing_eval_matches_symbolic_evaluation():
    rng = random.Random(20260312)
    primes = [2, 3, 5, 7, 11]
    for _ in range(200):
        level = rng.randint(1, 25)
        a = rng.randint(0, level - 1)
        b = rng.randint(0, level - 1 - a)
        c = level - 1 - a - b
        d = rng.randint(0, level)
        e = level - d
        poly = pairing_poly(a, b, c, d, e)
        for p in primes:
            for q in range(p):
                assert pairing_eval_mod(a, b, c, d, e, q, p) == eval_poly_mod(poly, q, p)


@pytest.mark.parametrize("p", [2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
def test_uniform_fp_family_vanishes_on_all_field_values(p):
    tuple5 = (0, 2 * p - 3, 0, p - 2, p)
    assert single_whisker(*tuple5)
    assert passes_field_filter(*tuple5, p)
    assert not exact_zero(*tuple5, p)


@pytest.mark.parametrize(
    ("p", "tuple5", "field_zero", "exact"),
    [
        (3, (0, 3, 0, 1, 3), True, False),
        (5, (0, 7, 0, 3, 5), True, False),
        (7, (0, 11, 0, 5, 7), True, False),
        (7, (3, 0, 2, 0, 6), False, False),
        (11, (5, 0, 4, 0, 10), False, False),
    ],
)
def test_known_field_filter_and_exact_zero_status(p, tuple5, field_zero, exact):
    assert passes_field_filter(*tuple5, p) is field_zero
    assert exact_zero(*tuple5, p) is exact
