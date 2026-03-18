#!/usr/bin/env python3

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent
METRICS = ROOT / ".modp_metrics"


@dataclass(frozen=True)
class SplitAffineFamily:
    x_mul: int
    x_add: int
    y_mul: int
    y_add: int
    z_mul: int
    z_add: int
    w_mul: int
    w_add: int

    def split_at(self, n: int) -> Optional[Tuple[int, int, int, int]]:
        x = self.x_mul * n + self.x_add
        y = self.y_mul * n + self.y_add
        z = self.z_mul * n + self.z_add
        w = self.w_mul * n + self.w_add
        if min(x, y, z, w) < 0:
            return None
        return (x, y, z, w)

    def tuple_at(self, n: int) -> Optional[Tuple[int, int, int, int, int]]:
        split = self.split_at(n)
        if split is None:
            return None
        x, y, z, w = split
        return (x + y, z, w, x, y + z + w + 1)

    def describe(self) -> str:
        return (
            f"x={self.x_mul}*N+{self.x_add}, "
            f"y={self.y_mul}*N+{self.y_add}, "
            f"z={self.z_mul}*N+{self.z_add}, "
            f"w={self.w_mul}*N+{self.w_add}"
        )


@dataclass(frozen=True)
class FamilySummary:
    family: SplitAffineFamily
    sample_count: int
    single_count: int
    field_count: int
    edge_count: int
    exact_count: int
    best_single: Optional[Tuple[int, int, int, int, int, int, int, Tuple[int, int, int, int, int]]]
    best_field: Optional[Tuple[int, int, int, int, int, int, int, Tuple[int, int, int, int, int]]]
    best_edge: Optional[Tuple[int, int, int, int, int, int, int, Tuple[int, int, int, int, int]]]


def ensure_metrics() -> None:
    if METRICS.exists():
        return
    subprocess.run(
        ["cc", "-O3", "burau_modp_metrics.c", "-o", str(METRICS)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def metrics_batch(
    rows: List[Tuple[int, int, int, int, int]], prime: int
) -> List[Tuple[int, int, int, int, int]]:
    ensure_metrics()
    input_text = "".join(f"{prime} {a} {b} {c} {d} {e}\n" for a, b, c, d, e in rows)
    proc = subprocess.run(
        [str(METRICS)],
        cwd=ROOT,
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
    )
    return [tuple(int(x) for x in line.split()) for line in proc.stdout.strip().splitlines()]


def family_range(lo: int, hi: int) -> Iterator[int]:
    if lo > hi:
        return iter(())
    return iter(range(lo, hi + 1))


def analyze_family(
    family: SplitAffineFamily,
    prime: int,
    n_from: int,
    n_to: int,
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

    data = metrics_batch(rows, prime)
    single_count = 0
    field_count = 0
    edge_count = 0
    exact_count = 0
    best_single = None
    best_field = None
    best_edge = None

    for n, tup, metric in zip(ns, rows, data):
        single, field, bad, left, right = metric
        if not single:
            continue
        single_count += 1
        item = (bad, (left != 0) + (right != 0), abs(left) + abs(right), left, right, field, n, tup)
        if best_single is None or item < best_single:
            best_single = item
        if field:
            field_count += 1
            if best_field is None or item < best_field:
                best_field = item
        if left == 0 and right == 0:
            edge_count += 1
            if best_edge is None or item < best_edge:
                best_edge = item
        if bad == 0 and left == 0 and right == 0:
            exact_count += 1

    return FamilySummary(
        family=family,
        sample_count=len(rows),
        single_count=single_count,
        field_count=field_count,
        edge_count=edge_count,
        exact_count=exact_count,
        best_single=best_single,
        best_field=best_field,
        best_edge=best_edge,
    )


def format_item(
    item: Optional[Tuple[int, int, int, int, int, int, int, Tuple[int, int, int, int, int]]]
) -> str:
    if item is None:
        return "none"
    bad, edge_pen, edge_l1, left, right, field, n, tup = item
    level = tup[0] + tup[1] + tup[2] + 1
    return (
        f"N={n} level={level} bad={bad} edge_pen={edge_pen} edge_l1={edge_l1} "
        f"field={field} edges=({left},{right}) tuple={tup}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, required=True)
    parser.add_argument("--n-from", type=int, required=True)
    parser.add_argument("--n-to", type=int, required=True)
    parser.add_argument("--x-mul-from", type=int, required=True)
    parser.add_argument("--x-mul-to", type=int, required=True)
    parser.add_argument("--x-add-from", type=int, required=True)
    parser.add_argument("--x-add-to", type=int, required=True)
    parser.add_argument("--y-mul-from", type=int, required=True)
    parser.add_argument("--y-mul-to", type=int, required=True)
    parser.add_argument("--y-add-from", type=int, required=True)
    parser.add_argument("--y-add-to", type=int, required=True)
    parser.add_argument("--z-mul-from", type=int, default=0)
    parser.add_argument("--z-mul-to", type=int, default=0)
    parser.add_argument("--z-add-from", type=int, required=True)
    parser.add_argument("--z-add-to", type=int, required=True)
    parser.add_argument("--w-mul-from", type=int, required=True)
    parser.add_argument("--w-mul-to", type=int, required=True)
    parser.add_argument("--w-add-from", type=int, required=True)
    parser.add_argument("--w-add-to", type=int, required=True)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    summaries: List[FamilySummary] = []
    family_count = 0
    for x_mul in family_range(args.x_mul_from, args.x_mul_to):
        for x_add in family_range(args.x_add_from, args.x_add_to):
            for y_mul in family_range(args.y_mul_from, args.y_mul_to):
                for y_add in family_range(args.y_add_from, args.y_add_to):
                    for z_mul in family_range(args.z_mul_from, args.z_mul_to):
                        for z_add in family_range(args.z_add_from, args.z_add_to):
                            for w_mul in family_range(args.w_mul_from, args.w_mul_to):
                                for w_add in family_range(args.w_add_from, args.w_add_to):
                                    family = SplitAffineFamily(
                                        x_mul=x_mul,
                                        x_add=x_add,
                                        y_mul=y_mul,
                                        y_add=y_add,
                                        z_mul=z_mul,
                                        z_add=z_add,
                                        w_mul=w_mul,
                                        w_add=w_add,
                                    )
                                    family_count += 1
                                    summary = analyze_family(
                                        family=family,
                                        prime=args.prime,
                                        n_from=args.n_from,
                                        n_to=args.n_to,
                                    )
                                    if summary is not None and summary.single_count:
                                        summaries.append(summary)

    summaries.sort(
        key=lambda s: (
            -s.exact_count,
            -s.edge_count,
            10**9 if s.best_single is None else s.best_single[0],
            10**9 if s.best_single is None else s.best_single[1],
            10**9 if s.best_single is None else s.best_single[2],
            -(s.field_count),
            -(s.single_count),
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
            f"field={summary.field_count} edge={summary.edge_count} exact={summary.exact_count}\n"
            f"  best_single: {format_item(summary.best_single)}\n"
            f"  best_field:  {format_item(summary.best_field)}\n"
            f"  best_edge:   {format_item(summary.best_edge)}"
        )


if __name__ == "__main__":
    main()
