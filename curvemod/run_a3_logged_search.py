from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from a3_gpu_search import SearchConfig, run_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the A3 GPU search and print a compact summary.")
    parser.add_argument("--p", type=int, required=True, help="Modulus.")
    parser.add_argument("--cap-1", type=int, required=True)
    parser.add_argument("--cap-2", type=int, required=True)
    parser.add_argument("--total-cap-1", type=int, required=True)
    parser.add_argument("--total-cap-2", type=int, required=True)
    parser.add_argument("--first-steps", type=int, default=12)
    parser.add_argument("--max-g-length", type=int, required=True)
    parser.add_argument("--base-point", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument(
        "--exact-commutator-generator",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Require [w, sigma_g] to be identity mod p before accepting a witness. Use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--max-witnesses",
        type=int,
        default=1,
        help="Maximum number of unique witnesses to collect. Use 0 for no cap.",
    )
    parser.add_argument("--print-witnesses", type=int, default=10, help="How many witnesses to print at the end.")
    parser.add_argument(
        "--print-witness-limit",
        type=int,
        default=20,
        help="How many witnesses may be dumped inline during the search itself.",
    )
    parser.add_argument(
        "--witness-output",
        help="Optional path to write all collected witnesses as a formatted text file.",
    )
    return parser.parse_args()


def write_witness_report(path: Path, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    depth_hist = dict(sorted(Counter(result.witness_depths).items()))
    with path.open("w", encoding="ascii") as handle:
        handle.write("A3 witness report\n")
        handle.write(f"found = {result.found}\n")
        handle.write(f"first_depth = {result.depth}\n")
        handle.write(f"unique_witnesses = {len(result.witnesses)}\n")
        handle.write(f"total_hits = {result.total_hits}\n")
        handle.write(f"accepted_hits = {result.accepted_hits}\n")
        handle.write(f"rejected_hits = {result.rejected_hits}\n")
        handle.write(f"witness_depth_histogram = {depth_hist}\n")
        handle.write("\n")
        for index, (depth, witness) in enumerate(zip(result.witness_depths, result.witnesses), start=1):
            handle.write(f"Witness #{index}\n")
            handle.write(f"depth = {depth}\n")
            handle.write(f"blocks = {witness}\n")
            flat = [letter for block in witness for letter in block]
            handle.write(f"flat = {flat}\n")
            handle.write("\n")


def main() -> None:
    args = parse_args()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    result = run_search(
        SearchConfig(
            cap_1=args.cap_1,
            cap_2=args.cap_2,
            total_cap_1=args.total_cap_1,
            total_cap_2=args.total_cap_2,
            first_steps=args.first_steps,
            modulus=args.p,
            max_g_length=args.max_g_length,
            base_point=args.base_point,
            exact_commutator_generator=None if args.exact_commutator_generator == 0 else args.exact_commutator_generator,
            device=args.device,
            seed=args.seed,
            max_witnesses=None if args.max_witnesses == 0 else args.max_witnesses,
            print_witness_limit=args.print_witness_limit,
        )
    )

    if args.witness_output:
        write_witness_report(Path(args.witness_output), result)
        print("SUMMARY witness_output=", args.witness_output)

    print("SUMMARY found=", result.found)
    print("SUMMARY first_depth=", result.depth)
    print("SUMMARY unique_witnesses=", len(result.witnesses))
    print("SUMMARY total_hits=", result.total_hits)
    print("SUMMARY accepted_hits=", result.accepted_hits)
    print("SUMMARY rejected_hits=", result.rejected_hits)
    print("SUMMARY witness_depth_histogram=", dict(sorted(Counter(result.witness_depths).items())))

    for index, (depth, witness) in enumerate(
        zip(result.witness_depths[: args.print_witnesses], result.witnesses[: args.print_witnesses]),
        start=1,
    ):
        print(f"SUMMARY witness_{index}_depth=", depth)
        print(f"SUMMARY witness_{index}=", witness)

    if torch.cuda.is_available():
        print("CUDA max_memory_allocated_bytes:", torch.cuda.max_memory_allocated())
        print("CUDA max_memory_reserved_bytes:", torch.cuda.max_memory_reserved())


if __name__ == "__main__":
    main()
