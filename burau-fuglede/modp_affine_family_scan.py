#!/usr/bin/env python3

import argparse
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mod3_reciprocity_search import block_decomposition, raw_packet, reciprocity_type
from tests.run_exact_modp_validation import pairing_poly, single_whisker


ROOT = Path(__file__).resolve().parent
CHECKER = ROOT / ".exact_modp_check"


@dataclass(frozen=True)
class AffineFamily:
    a_mul: int
    a_add: int
    b: int
    c_mul: int
    c_add: int
    e: int

    def tuple_at(self, n: int) -> Optional[Tuple[int, int, int, int, int]]:
        a = self.a_mul * n + self.a_add
        c = self.c_mul * n + self.c_add
        if a < 0 or c < 0 or self.b < 0 or self.e < 0:
            return None
        level = a + self.b + c + 1
        d = level - self.e
        if d < 0:
            return None
        return (a, self.b, c, d, self.e)

    def describe(self) -> str:
        return (
            f"a={self.a_mul}*N+{self.a_add}, b={self.b}, "
            f"c={self.c_mul}*N+{self.c_add}, e={self.e}, d=level-e"
        )


def ensure_checker() -> None:
    if CHECKER.exists():
        return
    subprocess.run(
        ["cc", "-O3", "burau_exact_modp_check.c", "-o", str(CHECKER)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def checker_batch_modp(
    rows: List[Tuple[int, int, int, int, int]], prime: int
) -> List[Tuple[int, int, int]]:
    ensure_checker()
    input_text = "".join(f"{prime} {a} {b} {c} {d} {e}\n" for a, b, c, d, e in rows)
    proc = subprocess.run(
        [str(CHECKER)],
        cwd=ROOT,
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
    )
    return [tuple(int(x) for x in line.split()) for line in proc.stdout.strip().splitlines()]


def dense_coeffs(tuple5: Tuple[int, int, int, int, int]) -> List[int]:
    shift, coeffs = raw_packet(tuple5)
    return coeffs


def bad_coeff_count(tuple5: Tuple[int, int, int, int, int], prime: int) -> int:
    return sum(value % prime != 0 for value in dense_coeffs(tuple5))


def block_packet_distribution(
    tuple5: Tuple[int, int, int, int, int]
) -> Dict[Tuple[int, Tuple[int, ...]], Dict[int, int]]:
    out: Dict[Tuple[int, Tuple[int, ...]], Dict[int, int]] = {}
    counters: Dict[Tuple[int, Tuple[int, ...]], Counter] = defaultdict(Counter)
    for start, _symbols, packet in block_decomposition(tuple5):
        key = (packet[0], tuple(packet[1]))
        counters[key][start] += 1
    for key, counter in counters.items():
        out[key] = dict(sorted(counter.items()))
    return dict(sorted(out.items(), key=lambda item: (len(item[0][1]), item[0][0], item[0][1])))


def summarize_tuple(tuple5: Tuple[int, int, int, int, int], prime: int) -> str:
    coeffs = dense_coeffs(tuple5)
    blocks = block_decomposition(tuple5)
    starts = [start for start, _, _ in blocks]
    packet_counts = Counter((packet[0], tuple(packet[1])) for _, _, packet in blocks)
    return (
        f"tuple={tuple5} level={sum(tuple5[:3]) + 1} "
        f"bad={bad_coeff_count(tuple5, prime)} reciprocity={reciprocity_type(coeffs)} "
        f"span={len(coeffs) - 1} maxabs={max(abs(x) for x in coeffs)} "
        f"edges=({coeffs[0]},{coeffs[-1]}) "
        f"block_range=[{min(starts)},{max(starts)}] unique_packets={len(packet_counts)} "
        f"top_packets={packet_counts.most_common(6)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, required=True)
    parser.add_argument("--a-mul", type=int, required=True)
    parser.add_argument("--a-add", type=int, required=True)
    parser.add_argument("--b", type=int, required=True)
    parser.add_argument("--c-mul", type=int, required=True)
    parser.add_argument("--c-add", type=int, required=True)
    parser.add_argument("--e", type=int, required=True)
    parser.add_argument("--n-from", type=int, required=True)
    parser.add_argument("--n-to", type=int, required=True)
    parser.add_argument("--show", type=int, default=12)
    parser.add_argument("--show-distribution", type=int, default=None)
    args = parser.parse_args()

    family = AffineFamily(
        a_mul=args.a_mul,
        a_add=args.a_add,
        b=args.b,
        c_mul=args.c_mul,
        c_add=args.c_add,
        e=args.e,
    )

    rows = []
    ns = []
    for n in range(args.n_from, args.n_to + 1):
        tup = family.tuple_at(n)
        if tup is None:
            continue
        rows.append(tup)
        ns.append(n)

    triples = checker_batch_modp(rows, args.prime)
    exact = []
    field = []
    for n, tup, triple in zip(ns, rows, triples):
        single, field_zero, exact_zero = triple
        if not single:
            continue
        if exact_zero:
            exact.append((n, tup))
        elif field_zero:
            field.append((bad_coeff_count(tup, args.prime), n, tup))

    field.sort()

    print(f"family: {family.describe()}")
    print(f"prime={args.prime} N-range=[{args.n_from},{args.n_to}]")
    print(f"field_survivors={len(field)} exact_hits={len(exact)}")
    for n, tup in exact[: args.show]:
        print(f"EXACT N={n} {summarize_tuple(tup, args.prime)}")
    for bad, n, tup in field[: args.show]:
        print(f"FIELD N={n} {summarize_tuple(tup, args.prime)}")

    if args.show_distribution is not None:
        tup = family.tuple_at(args.show_distribution)
        if tup is None:
            raise SystemExit("show-distribution N is invalid for this family")
        if not single_whisker(*tup):
            raise SystemExit("show-distribution N does not give a single whisker")
        print(f"\npacket distributions for N={args.show_distribution}")
        for packet, starts in block_packet_distribution(tup).items():
            print(f"{packet}: {starts}")


if __name__ == "__main__":
    main()
