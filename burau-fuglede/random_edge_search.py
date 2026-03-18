#!/usr/bin/env python3

import argparse
import random
from typing import List, Tuple

from mod3_reciprocity_search import reduced_regime, tuple_from_split
from modp_split_family_scan import metrics_batch


KINDS = ("a", "b", "ce", "cd")


def random_split(total: int, rng: random.Random) -> tuple[int, int, int, int]:
    cuts = sorted(rng.sample(range(total + 3), 3))
    parts = []
    prev = -1
    for cut in cuts + [total + 2]:
        parts.append(cut - prev - 1)
        prev = cut
    return tuple(parts)


def sample_tuples(
    *,
    samples: int,
    level_min: int,
    level_max: int,
    seed: int,
) -> list[tuple[str, tuple[int, int, int, int], tuple[int, int, int, int, int]]]:
    rng = random.Random(seed)
    out = []
    for _ in range(samples):
        kind = rng.choice(KINDS)
        level = rng.randint(level_min, level_max)
        split = random_split(level - 1, rng)
        out.append((kind, split, tuple_from_split(kind, split)))
    return out


def summarize(
    item: tuple[int, int, int, int, str, tuple[int, int, int, int], tuple[int, int, int, int, int]]
) -> str:
    bad, edge_l1, left, right, kind, split, tup = item
    regime, endpoint, reduced = reduced_regime(tup)
    level = sum(tup[:3]) + 1
    return (
        f"bad={bad} edge_l1={edge_l1} edges=({left},{right}) "
        f"kind={kind} split={split} regime={regime}/{endpoint} reduced={reduced} "
        f"level={level} tuple={tup}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--level-min", type=int, default=1851)
    parser.add_argument("--level-max", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", type=int, default=20)
    args = parser.parse_args()

    sampled = sample_tuples(
        samples=args.samples,
        level_min=args.level_min,
        level_max=args.level_max,
        seed=args.seed,
    )
    rows = [tup for _, _, tup in sampled]
    data = metrics_batch(rows, args.prime)

    singles: List[
        tuple[int, int, int, int, str, tuple[int, int, int, int], tuple[int, int, int, int, int]]
    ] = []
    edge_valid: List[
        tuple[int, int, int, int, str, tuple[int, int, int, int], tuple[int, int, int, int, int]]
    ] = []

    for (kind, split, tup), metric in zip(sampled, data):
        single, _field, bad, left, right = metric
        if not single:
            continue
        item = (bad, abs(left) + abs(right), left, right, kind, split, tup)
        singles.append(item)
        if left == 0 and right == 0:
            edge_valid.append(item)

    singles.sort()
    edge_valid.sort()

    print(
        f"prime={args.prime} samples={args.samples} "
        f"levels=[{args.level_min},{args.level_max}] seed={args.seed}"
    )
    print(f"single={len(singles)} edge_valid={len(edge_valid)}")

    print("\nbest_edge_valid:")
    for item in edge_valid[: args.show]:
        print(summarize(item))

    print("\nbest_single:")
    for item in singles[: args.show]:
        print(summarize(item))


if __name__ == "__main__":
    main()
