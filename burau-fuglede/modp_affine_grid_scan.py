#!/usr/bin/env python3

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from mod3_reciprocity_search import raw_packet, reciprocity_type


ROOT = Path(__file__).resolve().parent
CHECKER = ROOT / ".exact_modp_check"


@dataclass(frozen=True)
class AffineFamily:
    a_mul: int
    a_add: int
    b_mul: int
    b_add: int
    c_mul: int
    c_add: int
    e_mul: int
    e_add: int

    def tuple_at(self, n: int) -> Optional[Tuple[int, int, int, int, int]]:
        a = self.a_mul * n + self.a_add
        b = self.b_mul * n + self.b_add
        c = self.c_mul * n + self.c_add
        e = self.e_mul * n + self.e_add
        if min(a, b, c, e) < 0:
            return None
        level = a + b + c + 1
        d = level - e
        if d < 0:
            return None
        return (a, b, c, d, e)

    def describe(self) -> str:
        return (
            f"a={self.a_mul}*N+{self.a_add}, "
            f"b={self.b_mul}*N+{self.b_add}, "
            f"c={self.c_mul}*N+{self.c_add}, "
            f"e={self.e_mul}*N+{self.e_add}, "
            f"d=level-e"
        )


@dataclass(frozen=True)
class FamilySummary:
    family: AffineFamily
    sample_count: int
    single_count: int
    field_count: int
    edge_field_count: int
    exact_count: int
    best_field: Optional[Tuple[int, int, Tuple[int, int], Tuple[int, int, int, int, int], str]]
    best_edge_field: Optional[Tuple[int, int, Tuple[int, int], Tuple[int, int, int, int, int], str]]


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


def bad_coeff_count(tuple5: Tuple[int, int, int, int, int], prime: int) -> int:
    _, coeffs = raw_packet(tuple5)
    return sum(value % prime != 0 for value in coeffs)


def packet_edge_pair(tuple5: Tuple[int, int, int, int, int]) -> Tuple[int, int]:
    _, coeffs = raw_packet(tuple5)
    return coeffs[0], coeffs[-1]


def family_range(lo: int, hi: int) -> Iterator[int]:
    if lo > hi:
        return iter(())
    return iter(range(lo, hi + 1))


def analyze_family(
    family: AffineFamily, prime: int, n_from: int, n_to: int
) -> Optional[FamilySummary]:
    rows: List[Tuple[int, int, int, int, int]] = []
    ns: List[int] = []
    for n in range(n_from, n_to + 1):
        tup = family.tuple_at(n)
        if tup is None:
            continue
        rows.append(tup)
        ns.append(n)
    if not rows:
        return None

    triples = checker_batch_modp(rows, prime)
    single_count = 0
    field_count = 0
    edge_field_count = 0
    exact_count = 0
    best_field: Optional[Tuple[int, int, Tuple[int, int], Tuple[int, int, int, int, int], str]] = None
    best_edge_field: Optional[Tuple[int, int, Tuple[int, int], Tuple[int, int, int, int, int], str]] = None

    for n, tup, triple in zip(ns, rows, triples):
        single, field_zero, exact_zero = triple
        if not single:
            continue
        single_count += 1
        if exact_zero:
            exact_count += 1
            field_zero = 1
        if not field_zero:
            continue

        field_count += 1
        edges = packet_edge_pair(tup)
        bad = bad_coeff_count(tup, prime)
        rec = reciprocity_type(raw_packet(tup)[1])
        field_item = (bad, n, edges, tup, rec)
        if best_field is None or field_item < best_field:
            best_field = field_item

        if edges[0] % prime == 0 and edges[1] % prime == 0:
            edge_field_count += 1
            if best_edge_field is None or field_item < best_edge_field:
                best_edge_field = field_item

    return FamilySummary(
        family=family,
        sample_count=len(rows),
        single_count=single_count,
        field_count=field_count,
        edge_field_count=edge_field_count,
        exact_count=exact_count,
        best_field=best_field,
        best_edge_field=best_edge_field,
    )


def format_item(
    item: Optional[Tuple[int, int, Tuple[int, int], Tuple[int, int, int, int, int], str]]
) -> str:
    if item is None:
        return "none"
    bad, n, edges, tup, rec = item
    level = sum(tup[:3]) + 1
    return f"N={n} level={level} bad={bad} edges={edges} reciprocity={rec} tuple={tup}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, required=True)
    parser.add_argument("--n-from", type=int, required=True)
    parser.add_argument("--n-to", type=int, required=True)
    parser.add_argument("--a-mul-from", type=int, required=True)
    parser.add_argument("--a-mul-to", type=int, required=True)
    parser.add_argument("--a-add-from", type=int, required=True)
    parser.add_argument("--a-add-to", type=int, required=True)
    parser.add_argument("--b-mul-from", type=int, default=0)
    parser.add_argument("--b-mul-to", type=int, default=0)
    parser.add_argument("--b-add-from", type=int, required=True)
    parser.add_argument("--b-add-to", type=int, required=True)
    parser.add_argument("--c-mul-from", type=int, default=1)
    parser.add_argument("--c-mul-to", type=int, default=1)
    parser.add_argument("--c-add-from", type=int, required=True)
    parser.add_argument("--c-add-to", type=int, required=True)
    parser.add_argument("--e-mul-from", type=int, default=0)
    parser.add_argument("--e-mul-to", type=int, default=0)
    parser.add_argument("--e-add-from", type=int, required=True)
    parser.add_argument("--e-add-to", type=int, required=True)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--require-field", action="store_true")
    args = parser.parse_args()

    summaries: List[FamilySummary] = []
    family_count = 0
    for a_mul in family_range(args.a_mul_from, args.a_mul_to):
        for a_add in family_range(args.a_add_from, args.a_add_to):
            for b_mul in family_range(args.b_mul_from, args.b_mul_to):
                for b_add in family_range(args.b_add_from, args.b_add_to):
                    for c_mul in family_range(args.c_mul_from, args.c_mul_to):
                        for c_add in family_range(args.c_add_from, args.c_add_to):
                            for e_mul in family_range(args.e_mul_from, args.e_mul_to):
                                for e_add in family_range(args.e_add_from, args.e_add_to):
                                    family = AffineFamily(
                                        a_mul=a_mul,
                                        a_add=a_add,
                                        b_mul=b_mul,
                                        b_add=b_add,
                                        c_mul=c_mul,
                                        c_add=c_add,
                                        e_mul=e_mul,
                                        e_add=e_add,
                                    )
                                    family_count += 1
                                    summary = analyze_family(
                                        family=family,
                                        prime=args.prime,
                                        n_from=args.n_from,
                                        n_to=args.n_to,
                                    )
                                    if summary is None:
                                        continue
                                    if args.require_field and summary.field_count == 0:
                                        continue
                                    summaries.append(summary)

    summaries.sort(
        key=lambda s: (
            -s.exact_count,
            -s.edge_field_count,
            10**9 if s.best_edge_field is None else s.best_edge_field[0],
            -s.field_count,
            10**9 if s.best_field is None else s.best_field[0],
            -s.single_count,
        )
    )

    print(
        f"families_scanned={family_count} retained={len(summaries)} "
        f"prime={args.prime} N-range=[{args.n_from},{args.n_to}]"
    )
    for summary in summaries[: args.top]:
        print(
            f"\nfamily: {summary.family.describe()}\n"
            f"  samples={summary.sample_count} single={summary.single_count} "
            f"field={summary.field_count} edge_field={summary.edge_field_count} exact={summary.exact_count}\n"
            f"  best_field: {format_item(summary.best_field)}\n"
            f"  best_edge_field: {format_item(summary.best_edge_field)}"
        )


if __name__ == "__main__":
    main()
