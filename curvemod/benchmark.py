from __future__ import annotations

import argparse
import statistics
from dataclasses import asdict

from a3_gpu_search import SearchConfig, run_search
from a3_old_cpu_search import OldSearchConfig, run_old_search


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark old CPU A3 search against the new torch implementation.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of timing runs for each implementation.")
    parser.add_argument("--p", type=int, default=5, help="Modulus p.")
    parser.add_argument("--max-g-length", type=int, default=50, help="Maximum Garside length.")
    parser.add_argument("--cap-1", type=int, default=500)
    parser.add_argument("--cap-2", type=int, default=500)
    parser.add_argument("--total-cap-1", type=int, default=50000)
    parser.add_argument("--total-cap-2", type=int, default=50000)
    parser.add_argument("--first-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--new-device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device for the new torch implementation.",
    )
    parser.add_argument(
        "--which",
        default="both",
        choices=["both", "old", "new"],
        help="Which implementation(s) to benchmark.",
    )
    return parser.parse_args()


def summarize(label: str, times: list[float]):
    print(f"{label}:")
    print(f"  runs   = {[round(x, 6) for x in times]}")
    print(f"  mean   = {statistics.mean(times):.6f}s")
    if len(times) > 1:
        print(f"  stdev  = {statistics.stdev(times):.6f}s")
    print(f"  best   = {min(times):.6f}s")
    print(f"  worst  = {max(times):.6f}s")


def main():
    args = parse_args()

    old_config = OldSearchConfig(
        cap_1=args.cap_1,
        cap_2=args.cap_2,
        total_cap_1=args.total_cap_1,
        total_cap_2=args.total_cap_2,
        first_steps=args.first_steps,
        modulus=args.p,
        max_g_length=args.max_g_length,
        seed=args.seed,
    )
    new_config = SearchConfig(
        cap_1=args.cap_1,
        cap_2=args.cap_2,
        total_cap_1=args.total_cap_1,
        total_cap_2=args.total_cap_2,
        first_steps=args.first_steps,
        modulus=args.p,
        max_g_length=args.max_g_length,
        device=args.new_device,
        seed=args.seed,
    )

    print("Benchmark config:")
    print(asdict(new_config))

    old_times: list[float] = []
    new_times: list[float] = []

    for _ in range(args.repeats):
        if args.which in ("both", "old"):
            old_result = run_old_search(old_config)
            old_times.append(old_result.runtime_seconds)
        if args.which in ("both", "new"):
            new_result = run_search(new_config)
            new_times.append(new_result.runtime_seconds)

    if old_times:
        summarize("old", old_times)
    if new_times:
        summarize("new", new_times)

    if old_times and new_times:
        old_best = min(old_times)
        new_best = min(new_times)
        ratio = old_best / new_best if new_best > 0 else float("inf")
        if ratio > 1:
            print(f"Best-run speedup (new over old): {ratio:.3f}x")
        else:
            print(f"Best-run slowdown (new vs old): {(1/ratio):.3f}x")


if __name__ == "__main__":
    main()
