#!/usr/bin/env python3

import argparse
from dataclasses import dataclass

from smallb_live_phase_model import reduced_word_histogram, tuple_from_split, verify_against_tuple
from tests.run_exact_modp_validation import single_whisker


@dataclass(frozen=True)
class NormalizedFamily:
    a_add: int
    b_sub: int
    z: int
    x_add: int = 0

    def split_at(self, n: int) -> tuple[int, int, int, int]:
        x = n + self.x_add
        y = n + self.a_add
        w = n - self.b_sub
        return (x, y, self.z, w)

    def tuple_at(self, n: int) -> tuple[int, int, int, int, int]:
        return tuple_from_split(*self.split_at(n))

    def describe(self) -> str:
        return f"x=N+{self.x_add}, y=N+{self.a_add}, z={self.z}, w=N-{self.b_sub}"


def point_key(hist: list[int], prime: int) -> tuple[int, int, int, int]:
    bad = sum(c % prime != 0 for c in hist)
    edge_pen = (hist[0] % prime != 0) + (hist[-1] % prime != 0)
    edge_l1 = (hist[0] % prime) + (hist[-1] % prime)
    edge_abs = abs(hist[0]) + abs(hist[-1])
    return (bad, edge_pen, edge_l1, edge_abs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, required=True)
    parser.add_argument("--n-from", type=int, required=True)
    parser.add_argument("--n-to", type=int, required=True)
    parser.add_argument("--a-from", type=int, required=True)
    parser.add_argument("--a-to", type=int, required=True)
    parser.add_argument("--b-from", type=int, required=True)
    parser.add_argument("--b-to", type=int, required=True)
    parser.add_argument("--z-from", type=int, required=True)
    parser.add_argument("--z-to", type=int, required=True)
    parser.add_argument("--x-add", type=int, default=0)
    parser.add_argument("--top-families", type=int, default=20)
    parser.add_argument("--top-points", type=int, default=6)
    parser.add_argument("--verify-top", type=int, default=0)
    parser.add_argument("--require-single", action="store_true")
    args = parser.parse_args()

    summaries = []
    for a_add in range(args.a_from, args.a_to + 1):
        for b_sub in range(args.b_from, args.b_to + 1):
            for z in range(args.z_from, args.z_to + 1):
                fam = NormalizedFamily(a_add=a_add, b_sub=b_sub, z=z, x_add=args.x_add)
                single_count = 0
                best = None
                best_rows = []
                for n in range(args.n_from, args.n_to + 1):
                    x, y, z0, w = fam.split_at(n)
                    if min(x, y, z0, w) < 0:
                        continue
                    tup = fam.tuple_at(n)
                    if args.require_single and not single_whisker(*tup):
                        continue
                    single_count += 1
                    try:
                        _, (lo, hist), _ = reduced_word_histogram(x, y, z0, w)
                    except Exception:
                        continue
                    if not hist:
                        continue
                    key = point_key(hist, args.prime)
                    row = (key, n, (hist[0], hist[-1]), len(hist), tup, (x, y, z0, w))
                    best_rows.append(row)
                    if best is None or row < best:
                        best = row
                if best is None:
                    continue
                best_rows.sort()
                summaries.append((best[0], -single_count, fam, best_rows[: args.top_points]))

    summaries.sort()
    print(
        f"prime={args.prime} families={len(summaries)} "
        f"N-range=[{args.n_from},{args.n_to}] "
        f"A-range=[{args.a_from},{args.a_to}] "
        f"B-range=[{args.b_from},{args.b_to}] "
        f"Z-range=[{args.z_from},{args.z_to}]"
    )
    for _, neg_single, fam, best_rows in summaries[: args.top_families]:
        count_label = "single" if args.require_single else "model_points"
        print(f"\nfamily: {fam.describe()} {count_label}={-neg_single}")
        for row in best_rows:
            key, n, edges, span, tup, split = row
            bad, edge_pen, edge_l1, edge_abs = key
            print(
                f"  N={n} bad={bad} edge_pen={edge_pen} edge_l1={edge_l1} "
                f"edge_abs={edge_abs} edges={edges} span={span} split={split} tuple={tup}"
            )
        if args.verify_top > 0:
                print("  verify:")
                for row in best_rows[: args.verify_top]:
                    _, n, _, _, tup, split = row
                    try:
                        verdict = verify_against_tuple(*split)
                    except Exception as exc:
                        verdict = f"ERR {type(exc).__name__}: {exc}"
                    print(f"    N={n} split={split} verify={verdict} single={single_whisker(*tup)}")


if __name__ == "__main__":
    main()
