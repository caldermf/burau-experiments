#!/usr/bin/env python3

import argparse
from collections import Counter
from typing import Optional

from mod3_reciprocity_search import block_decomposition, reduced_regime, trimmed_relative_histogram_edges
from modp_split_family_scan import metrics_batch
from step_neutrality_scan import dh0_bad_entries, symbol_bad_entries


def tuple_at(a: int, b: int, c: int, e: int) -> Optional[tuple[int, int, int, int, int]]:
    level = a + b + c + 1
    d = level - e
    if min(a, b, c, d, e) < 0:
        return None
    return (a, b, c, d, e)


def block_summary(tuple5: tuple[int, int, int, int, int]) -> tuple[int, str]:
    lengths = Counter(len(block_symbols) for _, block_symbols, _ in block_decomposition(tuple5))
    text = ",".join(f"{length}:{count}" for length, count in sorted(lengths.items()))
    return len(lengths), text


def summarize_best(
    tuple5: tuple[int, int, int, int, int],
    prime: int,
    n: int,
    bad: int,
    left: int,
    right: int,
    field: int,
) -> str:
    regime, endpoint, split = reduced_regime(tuple5)
    uniq_lengths, length_text = block_summary(tuple5)
    lo, left_edge, hi, right_edge = trimmed_relative_histogram_edges(tuple5)
    return (
        f"N={n} tuple={tuple5} level={sum(tuple5[:3]) + 1} bad={bad} field={field} "
        f"edges=({left},{right}) trimmed=({lo},{left_edge})..({hi},{right_edge}) "
        f"regime={regime}/{endpoint} split={split} "
        f"dh0_bad={len(dh0_bad_entries(tuple5, prime))} "
        f"all_step_bad={len(symbol_bad_entries(tuple5, prime))} "
        f"block_length_kinds={uniq_lengths} block_lengths={{{length_text}}}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, required=True)
    parser.add_argument("--a-from", type=int, required=True)
    parser.add_argument("--a-to", type=int, required=True)
    parser.add_argument("--b-from", type=int, required=True)
    parser.add_argument("--b-to", type=int, required=True)
    parser.add_argument("--e-from", type=int, required=True)
    parser.add_argument("--e-to", type=int, required=True)
    parser.add_argument("--n-from", type=int, required=True)
    parser.add_argument("--n-to", type=int, required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    families = []
    total = 0
    for a in range(args.a_from, args.a_to + 1):
        for b in range(args.b_from, args.b_to + 1):
            for e in range(args.e_from, args.e_to + 1):
                rows = []
                ns = []
                for n in range(args.n_from, args.n_to + 1):
                    tup = tuple_at(a, b, n, e)
                    if tup is None:
                        continue
                    rows.append(tup)
                    ns.append(n)
                if not rows:
                    continue
                total += 1
                data = metrics_batch(rows, args.prime)
                best = None
                exact = 0
                single_count = 0
                for n, tup, metric in zip(ns, rows, data):
                    single, field, bad, left, right = metric
                    if not single:
                        continue
                    single_count += 1
                    if bad == 0 and left == 0 and right == 0:
                        exact += 1
                    item = (
                        bad,
                        (left != 0) + (right != 0),
                        abs(left) + abs(right),
                        n,
                        tup,
                        field,
                        left,
                        right,
                    )
                    if best is None or item < best:
                        best = item
                if best is None:
                    continue
                bad, edge_pen, edge_l1, n, tup, field, left, right = best
                families.append(
                    (
                        exact,
                        single_count,
                        a,
                        b,
                        e,
                        (
                            bad,
                            edge_pen,
                            edge_l1,
                            len(dh0_bad_entries(tup, args.prime)),
                            len(symbol_bad_entries(tup, args.prime)),
                            n,
                            tup,
                            field,
                            left,
                            right,
                        ),
                    )
                )

    families.sort(
        key=lambda row: (
            -row[0],
            row[5][0],
            row[5][1],
            row[5][2],
            row[5][3],
            row[5][4],
            -row[1],
            row[2],
            row[3],
            row[4],
        )
    )

    print(
        f"sparse-line census prime={args.prime} families={total} "
        f"ranges: a=[{args.a_from},{args.a_to}] b=[{args.b_from},{args.b_to}] "
        f"e=[{args.e_from},{args.e_to}] N=[{args.n_from},{args.n_to}]"
    )
    for exact, single_count, a, b, e, best in families[: args.top]:
        bad, edge_pen, edge_l1, dh0_bad, all_bad, n, tup, field, left, right = best
        print(
            f"\nfamily (a,b,e)=({a},{b},{e}) exact={exact} singles={single_count} "
            f"best_bad={bad} edge_pen={edge_pen} edge_l1={edge_l1} "
            f"dh0_bad={dh0_bad} all_step_bad={all_bad}"
        )
        print(summarize_best(tup, args.prime, n, bad, left, right, field))


if __name__ == "__main__":
    main()
