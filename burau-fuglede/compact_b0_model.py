#!/usr/bin/env python3

import argparse
from collections import Counter, defaultdict

from tests.run_exact_modp_validation import single_whisker


def tuple_from_split(x: int, y: int, w: int) -> tuple[int, int, int, int, int]:
    return (x + y, 0, w, x, y + w + 1)


def compact_phase(x: int, y: int, w: int) -> bool:
    return x >= 0 and y > x and w > 2 * y


def reduced_word_histogram(x: int, y: int, w: int) -> tuple[Counter[str], tuple[int, list[int]]]:
    if not compact_phase(x, y, w):
        raise ValueError(f"split=({x},{y},{w}) is outside the exact compact b=0 phase")

    u = -(w + 1)
    h = 0
    words: Counter[str] = Counter()
    coeffs: dict[int, int] = defaultdict(int)

    while True:
        if u == -(2 * y + 1):
            words["eLE"] += 1
            coeffs[h] -= 1
            break

        if u < -(2 * y + 1):
            words["eLcR"] += 1
            coeffs[h] -= 1
            h += 2
            u += 2 * y + 1
            continue

        if u < -y:
            words["eLcL"] += 1
            coeffs[h] -= 1
            u += 2 * y + 1
            continue

        if u <= w - 2 * y - 1:
            words["eRcL"] += 1
            coeffs[h - 1] += 1
            h -= 2
            u += 2 * y + 1
            continue

        p3_cut = x + w - 2 * y + 1
        p2_cut = 2 * x + w - 2 * y + 1
        p2_flip = x + w - y

        if u < p3_cut:
            words["eRaR_dLaL_eLcR"] += 1
            coeffs[h - 1] += 1
            coeffs[h] -= 1
            coeffs[h + 1] += 1
            h += 2
            u -= 2 * x + 2 * w - 4 * y + 1
            continue

        if u < p2_cut:
            words["eRaR_dRaL_eLcR"] += 1
            coeffs[h - 2] -= 1
            coeffs[h - 1] += 1
            coeffs[h] -= 1
            u -= 2 * x + 2 * w - 4 * y + 1
            continue

        if u <= p2_flip:
            words["eRaR_eLcR"] += 1
            coeffs[h - 1] += 1
            coeffs[h] -= 1
            h += 2
            u -= 2 * x + 2 * w - 2 * y + 1
            continue

        if u <= w:
            words["eRaL_eLcR"] += 1
            coeffs[h - 2] -= 1
            coeffs[h - 1] += 1
            u -= 2 * x + 2 * w - 2 * y + 1
            continue

        raise RuntimeError(f"orbit escaped compact phase at u={u}")

    lo = min(coeffs)
    hi = max(coeffs)
    return words, (lo, [coeffs[i] for i in range(lo, hi + 1)])


def bad_count(x: int, y: int, w: int, p: int) -> int:
    _, (_, coeffs) = reduced_word_histogram(x, y, w)
    return sum(c % p != 0 for c in coeffs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("x", type=int)
    parser.add_argument("y", type=int)
    parser.add_argument("w", type=int)
    parser.add_argument("--p", type=int, default=5)
    args = parser.parse_args()

    tup = tuple_from_split(args.x, args.y, args.w)
    words, (lo, coeffs) = reduced_word_histogram(args.x, args.y, args.w)
    print(f"split=({args.x},{args.y},{args.w}) tuple={tup}")
    print(f"compact_phase={compact_phase(args.x, args.y, args.w)} single_whisker={single_whisker(*tup)}")
    print(f"word_counts={dict(sorted(words.items()))}")
    print(f"trimmed_relative_shift={lo} trimmed_relative_hist={coeffs}")
    print(f"bad_mod_{args.p}={sum(c % args.p != 0 for c in coeffs)}")


if __name__ == "__main__":
    main()
