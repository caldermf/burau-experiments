#!/usr/bin/env python3

"""Exact projlen score for a Burau pairing 5-tuple modulo p."""

from __future__ import annotations

import ast
import sys


def _poly_add(poly, mon, sign=1):
    out = dict(poly)
    for exp, coeff in mon.items():
        out[exp] = out.get(exp, 0) + sign * coeff
        if out[exp] == 0:
            del out[exp]
    return out


def _poly_shift(poly, shift):
    return {exp + shift: coeff for exp, coeff in poly.items()}


def pairing_poly(a, b, c, d, e):
    """Return the exact Laurent polynomial as {exponent: coefficient}."""
    bl = 2 * a
    start = bl + b
    cl = start + b + 1
    end = cl + c
    el = 2 * d
    er = el + e
    suma = bl - 1
    sumb = 2 * start
    sumc = 2 * end
    sumd = el - 1
    sume = 2 * er - 1
    x = start
    poly = {}
    mon = {0: 1}

    while True:
        if x < el:
            if x < d:
                mon = _poly_shift(mon, 1)
                poly = _poly_add(poly, mon)
            else:
                poly = _poly_add(poly, mon, sign=-1)
                poly = _poly_shift(poly, 1)
            x = sumd - x
        else:
            if x < er:
                poly = _poly_add(poly, mon, sign=-1)
                mon = _poly_shift(mon, 1)
            else:
                poly = _poly_shift(poly, 1)
                poly = _poly_add(poly, mon)
            x = sume - x

        if x < cl:
            if x < bl:
                if x < a:
                    poly = _poly_shift(poly, 1)
                else:
                    mon = _poly_shift(mon, 1)
                x = suma - x
            else:
                if x < start:
                    mon = _poly_shift(mon, 4)
                else:
                    poly = _poly_shift(poly, 4)
                x = sumb - x
        else:
            if x < end:
                poly = _poly_shift(poly, 1)
            elif x > end:
                mon = _poly_shift(mon, 1)
            else:
                return poly
            x = sumc - x


def score(tuple5, p):
    """Return the exponent spread in characteristic p.

    Use p = 0 for characteristic 0. If no terms survive, return 0.
    """
    poly = pairing_poly(*tuple5)
    if p == 0:
        surviving = [exp for exp, coeff in poly.items() if coeff != 0]
    else:
        surviving = [exp for exp, coeff in poly.items() if coeff % p != 0]
    if not surviving:
        return 0
    return max(surviving) - min(surviving)


projlen = score


def _parse_args(argv):
    if len(argv) == 3:
        p = int(argv[1])
        tuple5 = ast.literal_eval(argv[2])
    elif len(argv) == 7:
        p = int(argv[1])
        tuple5 = tuple(int(x) for x in argv[2:7])
    else:
        raise SystemExit(
            "usage: python score.py P '(a, b, c, d, e)'\n"
            "   or: python score.py P a b c d e"
        )

    if not isinstance(tuple5, tuple) or len(tuple5) != 5:
        raise SystemExit("tuple input must be a 5-tuple")

    return tuple5, p


if __name__ == "__main__":
    args = _parse_args(sys.argv)
    print(score(*args))
