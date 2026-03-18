#!/usr/bin/env python3

import argparse
from collections import Counter, defaultdict
from typing import Optional, Tuple

from mod3_reciprocity_search import _step_type_data, symbolic_walk, trimmed_relative_histogram
from tests.run_exact_modp_validation import single_whisker


def _zero_drift_symbols() -> tuple[str, ...]:
    out = []
    for down in ("dL", "dR", "eL", "eR"):
        for over in ("aL", "aR", "bL", "bR", "cL", "cR", "E"):
            symbol = down + over
            try:
                delta, _ = _step_type_data(symbol)
            except ValueError:
                continue
            if delta == 0:
                out.append(symbol)
    return tuple(sorted(out))


DH0_SYMBOLS = _zero_drift_symbols()


def tuple_from_a_split(x: int, y: int, z: int, w: int) -> tuple[int, int, int, int, int]:
    return (x + y, z, w, x, y + z + w + 1)


def dh0_symbol_height_counts(
    tuple5: tuple[int, int, int, int, int]
) -> dict[str, dict[int, int]]:
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for symbol, h in symbolic_walk(tuple5):
        if symbol in DH0_SYMBOLS:
            counts[symbol][h] += 1
    return {name: dict(sorted(counter.items())) for name, counter in sorted(counts.items())}


def symbol_height_counts(
    tuple5: tuple[int, int, int, int, int],
    *,
    symbols: Optional[Tuple[str, ...]] = None,
) -> dict[str, dict[int, int]]:
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    keep = None if symbols is None else set(symbols)
    for symbol, h in symbolic_walk(tuple5):
        if keep is None or symbol in keep:
            counts[symbol][h] += 1
    return {name: dict(sorted(counter.items())) for name, counter in sorted(counts.items())}


def dh0_bad_entries(
    tuple5: tuple[int, int, int, int, int], prime: int
) -> list[tuple[str, int, int, int]]:
    bad = []
    for name, counter in dh0_symbol_height_counts(tuple5).items():
        for h, count in counter.items():
            residue = count % prime
            if residue:
                bad.append((name, h, residue, count))
    return sorted(bad)


def symbol_bad_entries(
    tuple5: tuple[int, int, int, int, int],
    prime: int,
    *,
    symbols: Optional[Tuple[str, ...]] = None,
) -> list[tuple[str, int, int, int]]:
    bad = []
    for name, counter in symbol_height_counts(tuple5, symbols=symbols).items():
        for h, count in counter.items():
            residue = count % prime
            if residue:
                bad.append((name, h, residue, count))
    return sorted(bad)


def summarize_tuple(tuple5: tuple[int, int, int, int, int], prime: int) -> str:
    lo, hist = trimmed_relative_histogram(tuple5)
    bad = sum(value % prime != 0 for value in hist)
    return (
        f"tuple={tuple5} level={sum(tuple5[:3]) + 1} single={single_whisker(*tuple5)} "
        f"bad_mod_{prime}={bad} span={len(hist)} edges=({hist[0]},{hist[-1]}) "
        f"dh0_bad={len(dh0_bad_entries(tuple5, prime))} "
        f"all_step_bad={len(symbol_bad_entries(tuple5, prime))}"
    )


def scan_normalized_family(
    *,
    prime: int,
    n_from: int,
    n_to: int,
    a_add: int,
    b_sub: int,
    z: int,
) -> list[tuple[int, int, int, int, tuple[int, int], tuple[int, int, int, int, int]]]:
    rows = []
    for n in range(n_from, n_to + 1):
        x = n
        y = n + a_add
        w = n - b_sub
        if w < 0:
            continue
        tup = tuple_from_a_split(x, y, z, w)
        if not single_whisker(*tup):
            continue
        lo, hist = trimmed_relative_histogram(tup)
        rows.append(
            (
                len(dh0_bad_entries(tup, prime)),
                sum(value % prime != 0 for value in hist),
                n,
                len(hist),
                (hist[0], hist[-1]),
                tup,
            )
        )
    rows.sort()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, default=5)
    parser.add_argument("--tuple", dest="tuple_text", type=str)
    parser.add_argument("--scan-normalized", action="store_true")
    parser.add_argument("--n-from", type=int)
    parser.add_argument("--n-to", type=int)
    parser.add_argument("--a-add", type=int)
    parser.add_argument("--b-sub", type=int)
    parser.add_argument("--z", type=int)
    parser.add_argument("--show", type=int, default=20)
    args = parser.parse_args()

    if args.tuple_text:
        tuple5 = tuple(int(x.strip()) for x in args.tuple_text.strip("()").split(","))
        if len(tuple5) != 5:
            raise SystemExit("tuple must have 5 entries")
        print(summarize_tuple(tuple5, args.prime))
        print("dh0_height_counts=", dh0_symbol_height_counts(tuple5))
        print("dh0_bad_entries=", dh0_bad_entries(tuple5, args.prime))
        print("all_symbol_bad_entries=", symbol_bad_entries(tuple5, args.prime))
        return

    if args.scan_normalized:
        missing = [name for name in ("n_from", "n_to", "a_add", "b_sub", "z") if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"missing arguments for --scan-normalized: {', '.join(missing)}")
        rows = scan_normalized_family(
            prime=args.prime,
            n_from=args.n_from,
            n_to=args.n_to,
            a_add=args.a_add,
            b_sub=args.b_sub,
            z=args.z,
        )
        print(
            f"normalized family: x=N, y=N+{args.a_add}, z={args.z}, w=N-{args.b_sub} "
            f"prime={args.prime} rows={len(rows)}"
        )
        for row in rows[: args.show]:
            print(row)
        return

    raise SystemExit("provide --tuple or --scan-normalized")


if __name__ == "__main__":
    main()
