#!/usr/bin/env python3
"""
Benchmark: unrolled (Triton) vs PyTorch implementation of 3x3 matrix multiplication
over the ring F_7[x]/(x^6 - 1). Same math as triton7; different kernel structure.

Usage:
    python benchmark_unrolled.py [--save-plot] [--warmup N] [--rep N]
"""

import argparse
import torch
import numpy as np
import triton

import unrolled
import torch7


OPS_PER_BATCH_ELEMENT = 2052
BYTES_PER_BATCH_ELEMENT = 108


def check_correctness(batch_size: int = 256):
    """Assert unrolled and torch7 produce identical results."""
    torch.manual_seed(42)
    A = torch.randint(0, 2**18, (9, batch_size), dtype=torch.int32, device="cuda")
    B = torch.randint(0, 2**18, (9, batch_size), dtype=torch.int32, device="cuda")

    C_unrolled = unrolled.ring_matmul(A, B)
    C_torch = torch7.ring_matmul(A, B)

    if not torch.equal(C_unrolled, C_torch):
        num_diff = (C_unrolled != C_torch).sum().item()
        raise AssertionError(
            f"Correctness check FAILED: {num_diff}/{batch_size * 9} elements differ"
        )
    print(f"[OK] Correctness check passed (batch_size={batch_size})")


def benchmark_single(batch_size: int, warmup: int = 25, rep: int = 100) -> dict:
    torch.manual_seed(0)
    A = torch.randint(0, 2**18, (9, batch_size), dtype=torch.int32, device="cuda")
    B = torch.randint(0, 2**18, (9, batch_size), dtype=torch.int32, device="cuda")

    ms_unrolled = triton.testing.do_bench(
        lambda: unrolled.ring_matmul(A, B),
        warmup=warmup,
        rep=rep,
    )
    ms_torch = triton.testing.do_bench(
        lambda: torch7.ring_matmul(A, B),
        warmup=warmup,
        rep=rep,
    )

    total_ops = batch_size * OPS_PER_BATCH_ELEMENT
    total_bytes = batch_size * BYTES_PER_BATCH_ELEMENT
    speedup = ms_torch / ms_unrolled
    tops_unrolled = (total_ops / (ms_unrolled * 1e-3)) / 1e12
    tops_torch = (total_ops / (ms_torch * 1e-3)) / 1e12
    gbps_unrolled = (total_bytes / (ms_unrolled * 1e-3)) / 1e9
    gbps_torch = (total_bytes / (ms_torch * 1e-3)) / 1e9

    return {
        "batch_size": batch_size,
        "ms_unrolled": ms_unrolled,
        "ms_torch": ms_torch,
        "speedup": speedup,
        "tops_unrolled": tops_unrolled,
        "tops_torch": tops_torch,
        "gbps_unrolled": gbps_unrolled,
        "gbps_torch": gbps_torch,
    }


def run_sweep(batch_sizes: list[int], warmup: int = 25, rep: int = 100) -> list[dict]:
    results = []
    for bs in batch_sizes:
        print(f"  batch_size={bs:>8} ...", end=" ", flush=True)
        r = benchmark_single(bs, warmup=warmup, rep=rep)
        print(
            f"Unrolled: {r['ms_unrolled']:>8.3f} ms | "
            f"PyTorch: {r['ms_torch']:>8.3f} ms | "
            f"Speedup: {r['speedup']:>6.2f}x"
        )
        results.append(r)
    return results


def print_table(results: list[dict]):
    print()
    print("=" * 100)
    print("              Ring MatMul Benchmark: Unrolled (Triton) vs PyTorch")
    print("              F_7[x]/(x^6 - 1), Batched 3x3, SoA layout (9, Batch)")
    print("=" * 100)
    print()
    header = (
        f"{'Batch':>10} | "
        f"{'Unrolled (ms)':>14} | "
        f"{'PyTorch (ms)':>12} | "
        f"{'Speedup':>8} | "
        f"{'TOPS (Unr)':>10} | "
        f"{'TOPS (Torch)':>12} | "
        f"{'GB/s (Unr)':>10} | "
        f"{'GB/s (Torch)':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['batch_size']:>10} | "
            f"{r['ms_unrolled']:>14.4f} | "
            f"{r['ms_torch']:>12.4f} | "
            f"{r['speedup']:>7.2f}x | "
            f"{r['tops_unrolled']:>10.4f} | "
            f"{r['tops_torch']:>12.4f} | "
            f"{r['gbps_unrolled']:>10.2f} | "
            f"{r['gbps_torch']:>12.2f}"
        )
    print("-" * len(header))
    speedups = [r["speedup"] for r in results]
    print(f"Speedup: min={min(speedups):.2f}x, max={max(speedups):.2f}x")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark unrolled vs PyTorch ring matmul")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--save-plot", action="store_true")
    parser.add_argument("--max-exp", type=int, default=20)
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  Ring MatMul: Unrolled vs PyTorch")
    print("=" * 60)
    print()
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    print("[1/3] Correctness check...")
    check_correctness(256)
    print()

    print("[2/3] Benchmark sweep...")
    batch_sizes = [2**i for i in range(7, args.max_exp + 1)]
    results = run_sweep(batch_sizes, warmup=args.warmup, rep=args.rep)

    print("[3/3] Results:")
    print_table(results)

    if args.save_plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            bs = [r["batch_size"] for r in results]
            speedups = [r["speedup"] for r in results]
            ax.bar(range(len(bs)), speedups, color="#2ca02c", edgecolor="black", alpha=0.8)
            ax.axhline(y=1, color="red", linestyle="--", linewidth=1.5)
            ax.set_xticks(range(len(bs)))
            ax.set_xticklabels([f"2^{int(np.log2(b))}" for b in bs], fontsize=9)
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Speedup (PyTorch / Unrolled)")
            ax.set_title(r"Ring MatMul: Unrolled vs PyTorch ($\mathbb{F}_7[x]/(x^6-1)$)")
            plt.tight_layout()
            plt.savefig("benchmark_unrolled.png", dpi=150, bbox_inches="tight")
            print("[OK] Plot saved to benchmark_unrolled.png")
            plt.close()
        except ImportError:
            print("[WARN] matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
