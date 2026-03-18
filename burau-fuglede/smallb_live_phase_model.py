#!/usr/bin/env python3

import argparse
from collections import Counter, defaultdict

from mod3_reciprocity_search import _step_type_data, block_decomposition, trimmed_relative_histogram
from tests.run_exact_modp_validation import single_whisker


# Exact on the tested local {2,3,5}-packet a-split small-b phase near
# the defect-6 seed (1908,29,601,867,1672) and defect-5 seed
# (1889,26,592,860,1648). This is intentionally scoped: outside that
# phase the orbit can use different block words.
WORDS = {
    "W1": ("eRaR", "dLaL", "eLcL"),
    "W2": ("eRaR", "dLaL", "eLbR"),
    "W3": ("eRaR", "dRaL", "eLbR"),
    "W4": ("eRaR", "dRaL", "eLbL"),
    "W5": ("eRaR", "dRaL", "eLaR", "dLaL", "eLcR"),
    "W6": ("eRaR", "dRaL", "eLaR", "dLaL", "eLE"),
    "W7": ("eRaR", "dRaL", "eLaR", "dLaL", "eLcL"),
    "W8": ("eRaR", "dRaL", "eRaR", "dLaL", "eLcL"),
    "W9": ("eRaR", "eLcR"),
    "W10": ("eRaL", "eLcR"),
}


def tuple_from_split(x: int, y: int, z: int, w: int) -> tuple[int, int, int, int, int]:
    return (x + y, z, w, x, y + z + w + 1)


def _word_data() -> dict[str, tuple[int, int, list[int]]]:
    out = {}
    for name, syms in WORDS.items():
        h = 0
        coeffs: dict[int, int] = defaultdict(int)
        for sym in syms:
            delta, step_coeffs = _step_type_data(sym)
            for level, value in step_coeffs.items():
                coeffs[h + level] += value
            h += delta
        lo = min(coeffs)
        hi = max(coeffs)
        out[name] = (h, lo, [coeffs[i] for i in range(lo, hi + 1)])
    return out


WORD_DATA = _word_data()


def reduced_word_histogram(
    x: int, y: int, z: int, w: int
) -> tuple[Counter[str], tuple[int, list[int]], list[tuple[int, int, str]]]:
    a = x + y
    b = z
    c = w
    if min(x, y, z, w) < 0:
        raise ValueError("split coordinates must be nonnegative")

    # Thresholds for the local small-b {2,3,5} return map.
    t_a = x - 2 * y + w + 1
    t_b = 2 * x - 2 * y + w + 1
    t_c = x + w - y
    t_d = 2 * x - 4 * y + z + 3 * w + 2
    t_e = 2 * x - 4 * y + 2 * z + 3 * w + 2
    t_f = 2 * x - 3 * y + z + 2 * w + 2
    t_g = 4 * x - 6 * y + 2 * z + 4 * w + 3
    t_h = 2 * x - 4 * y + 3 * w + 1

    delta_left_c = -2 * x + 4 * y - 2 * w - 1
    delta_b = -2 * x + 4 * y - 2 * z - 4 * w - 3
    delta_mid = -4 * x + 6 * y - 2 * z - 4 * w - 3
    delta_right = -(2 * x - 2 * y + 2 * w + 1)

    u = -(c + b + 1)
    h = 0
    coeffs: dict[int, int] = defaultdict(int)
    words: Counter[str] = Counter()
    trace: list[tuple[int, int, str]] = []

    for _ in range(100000):
        if u > t_c:
            name = "W10"
            du = delta_right
        elif u >= t_b:
            name = "W9"
            du = delta_right
        elif u >= t_f:
            name = "W8"
            du = delta_mid
        elif u == t_g:
            name = "W6"
            du = None
        elif u > t_g:
            name = "W7"
            du = delta_mid
        elif u > t_e:
            name = "W5"
            du = delta_mid
        elif u > t_d:
            name = "W4"
            du = delta_b
        elif u >= t_a:
            name = "W3"
            du = delta_b
        elif u > t_h:
            name = "W2"
            du = delta_b
        else:
            name = "W1"
            du = delta_left_c

        dh, lo, arr = WORD_DATA[name]
        for offset, value in enumerate(arr):
            coeffs[h + lo + offset] += value
        words[name] += 1
        trace.append((u, h, name))
        h += dh
        if du is None:
            break
        u += du
    else:
        raise RuntimeError("model did not terminate; likely outside the intended phase")

    lo = min(coeffs)
    hi = max(coeffs)
    arr = [coeffs[i] for i in range(lo, hi + 1)]
    while arr and arr[-1] == 0:
        arr.pop()
    return words, (lo, arr), trace


def verify_against_tuple(x: int, y: int, z: int, w: int) -> tuple[bool, bool]:
    tup = tuple_from_split(x, y, z, w)
    model_words, model_hist, _ = reduced_word_histogram(x, y, z, w)
    actual_hist = trimmed_relative_histogram(tup)

    actual_words: Counter[str] = Counter()
    words_ok = True
    for _, sym, _ in block_decomposition(tup):
        matched = False
        for name, word in WORDS.items():
            if sym == word:
                actual_words[name] += 1
                matched = True
                break
        if not matched:
            words_ok = False
            break
    return words_ok and actual_words == model_words, model_hist == actual_hist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("x", type=int)
    parser.add_argument("y", type=int)
    parser.add_argument("z", type=int)
    parser.add_argument("w", type=int)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    tup = tuple_from_split(args.x, args.y, args.z, args.w)
    words, (lo, coeffs), trace = reduced_word_histogram(args.x, args.y, args.z, args.w)
    print(f"split=({args.x},{args.y},{args.z},{args.w}) tuple={tup}")
    print(f"single_whisker={single_whisker(*tup)}")
    print(f"word_counts={dict(sorted(words.items()))}")
    print(f"trimmed_relative_shift={lo} trimmed_relative_hist={coeffs}")
    print(f"trace_prefix={trace[:20]}")
    if args.verify:
        words_ok, hist_ok = verify_against_tuple(args.x, args.y, args.z, args.w)
        print(f"verify_words={words_ok} verify_hist={hist_ok}")


if __name__ == "__main__":
    main()
