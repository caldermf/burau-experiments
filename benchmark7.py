#!/usr/bin/env python3
"""
Benchmark: Triton vs PyTorch implementation of 3x3 matrix multiplication
over the ring F_7[x]/(x^6 - 1).

Usage:
    python benchmark7.py [--save-plot] [--warmup N] [--rep N]
"""

import argparse
import torch
import numpy as np
import triton

# Import both implementations
import triton7
import torch7


# ==============================================================================
# Constants for FLOPS/Bandwidth Calculations
# ==============================================================================

# Operations per batch element (theoretical):
# - DFT for 18 polynomials (9 from A, 9 from B): each is 6x6 matmul = 36 muls + 30 adds
# - 6 frequency-domain 3x3 matmuls: each is 27 muls + 18 adds
# - IDFT for 9 output polynomials: each is 6x6 matmul + 6 muls (scaling)
#
# DFT/IDFT per poly: 6*6 muls + 6*5 adds = 36 + 30 = 66 ops
# 18 DFTs + 9 IDFTs = 27 * 66 = 1782 ops
# 6 freq-domain 3x3 matmuls: 6 * (27 + 18) = 270 ops
# Total per batch element: ~2052 integer ops (approximate)
OPS_PER_BATCH_ELEMENT = 2052

# Memory per batch element:
# Read A: 9 * 4 bytes, Read B: 9 * 4 bytes, Write C: 9 * 4 bytes
# Total: 108 bytes per batch element
BYTES_PER_BATCH_ELEMENT = 108


def check_correctness(batch_size: int = 256):
    """Assert that both implementations produce identical results."""
    torch.manual_seed(42)
    # SoA layout: (9, batch) for coalesced memory access
    A = torch.randint(0, 2**18, (9, batch_size), dtype=torch.int32, device="cuda")
    B = torch.randint(0, 2**18, (9, batch_size), dtype=torch.int32, device="cuda")

    C_triton = triton7.ring_matmul(A, B)
    C_torch = torch7.ring_matmul(A, B)

    if not torch.equal(C_triton, C_torch):
        diff_mask = C_triton != C_torch
        num_diff = diff_mask.sum().item()
        raise AssertionError(
            f"Correctness check FAILED: {num_diff}/{batch_size * 9} elements differ"
        )
    print(f"[OK] Correctness check passed (batch_size={batch_size})")


def benchmark_single(
    batch_size: int,
    warmup: int = 25,
    rep: int = 100,
) -> dict:
    """
    Benchmark both implementations for a given batch size.
    Returns dict with timing and derived metrics.
    """
    torch.manual_seed(0)
    # SoA layout: (9, batch) for coalesced memory access
    A = torch.randint(0, 2**18, (9, batch_size), dtype=torch.int32, device="cuda")
    B = torch.randint(0, 2**18, (9, batch_size), dtype=torch.int32, device="cuda")

    # Triton kernel
    ms_triton = triton.testing.do_bench(
        lambda: triton7.ring_matmul(A, B),
        warmup=warmup,
        rep=rep,
    )

    # PyTorch (torch7)
    ms_torch = triton.testing.do_bench(
        lambda: torch7.ring_matmul(A, B),
        warmup=warmup,
        rep=rep,
    )

    # Derived metrics
    total_ops = batch_size * OPS_PER_BATCH_ELEMENT
    total_bytes = batch_size * BYTES_PER_BATCH_ELEMENT

    # TOPS = Tera Operations Per Second
    tops_triton = (total_ops / (ms_triton * 1e-3)) / 1e12
    tops_torch = (total_ops / (ms_torch * 1e-3)) / 1e12

    # GB/s = Gigabytes per second
    gbps_triton = (total_bytes / (ms_triton * 1e-3)) / 1e9
    gbps_torch = (total_bytes / (ms_torch * 1e-3)) / 1e9

    speedup = ms_torch / ms_triton

    return {
        "batch_size": batch_size,
        "ms_triton": ms_triton,
        "ms_torch": ms_torch,
        "speedup": speedup,
        "tops_triton": tops_triton,
        "tops_torch": tops_torch,
        "gbps_triton": gbps_triton,
        "gbps_torch": gbps_torch,
    }


def run_benchmark_sweep(
    batch_sizes: list[int],
    warmup: int = 25,
    rep: int = 100,
) -> list[dict]:
    """Run benchmark over a range of batch sizes."""
    results = []
    for bs in batch_sizes:
        print(f"  Benchmarking batch_size={bs:>8} ...", end=" ", flush=True)
        r = benchmark_single(bs, warmup=warmup, rep=rep)
        print(
            f"Triton: {r['ms_triton']:>8.3f} ms | "
            f"PyTorch: {r['ms_torch']:>8.3f} ms | "
            f"Speedup: {r['speedup']:>6.2f}x"
        )
        results.append(r)
    return results


def print_results_table(results: list[dict]):
    """Print results in a publication-quality ASCII table."""
    print()
    print("=" * 100)
    print("                        Ring MatMul Benchmark: Triton vs PyTorch")
    print("                        F_7[x]/(x^6 - 1), Batched 3x3 Matrices")
    print("=" * 100)
    print()

    # Header
    header = (
        f"{'Batch':>10} | "
        f"{'Triton (ms)':>12} | "
        f"{'PyTorch (ms)':>12} | "
        f"{'Speedup':>8} | "
        f"{'TOPS (Tri)':>10} | "
        f"{'TOPS (Torch)':>12} | "
        f"{'GB/s (Tri)':>10} | "
        f"{'GB/s (Torch)':>12}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['batch_size']:>10} | "
            f"{r['ms_triton']:>12.4f} | "
            f"{r['ms_torch']:>12.4f} | "
            f"{r['speedup']:>7.2f}x | "
            f"{r['tops_triton']:>10.4f} | "
            f"{r['tops_torch']:>12.4f} | "
            f"{r['gbps_triton']:>10.2f} | "
            f"{r['gbps_torch']:>12.2f}"
        )

    print("-" * len(header))
    print()

    # Summary statistics
    speedups = [r["speedup"] for r in results]
    print(f"Speedup: min={min(speedups):.2f}x, max={max(speedups):.2f}x, "
          f"mean={np.mean(speedups):.2f}x, median={np.median(speedups):.2f}x")
    print()


def generate_plot(results: list[dict], save_path: str = "benchmark7.png"):
    """Generate publication-quality plot."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[WARN] matplotlib not available, skipping plot generation")
        return

    batch_sizes = [r["batch_size"] for r in results]
    ms_triton = [r["ms_triton"] for r in results]
    ms_torch = [r["ms_torch"] for r in results]
    speedups = [r["speedup"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        r"Ring MatMul: $\mathbb{F}_7[x]/(x^6-1)$, Batched 3×3 Matrices",
        fontsize=14,
        fontweight="bold",
    )

    # --- Plot 1: Runtime ---
    ax1 = axes[0]
    ax1.plot(batch_sizes, ms_triton, "o-", label="Triton", color="#1f77b4", linewidth=2, markersize=6)
    ax1.plot(batch_sizes, ms_torch, "s-", label="PyTorch", color="#ff7f0e", linewidth=2, markersize=6)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Batch Size", fontsize=11)
    ax1.set_ylabel("Runtime (ms)", fontsize=11)
    ax1.set_title("Runtime vs Batch Size", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"$2^{{{int(np.log2(x))}}}$"))

    # --- Plot 2: Speedup ---
    ax2 = axes[1]
    ax2.bar(range(len(batch_sizes)), speedups, color="#2ca02c", edgecolor="black", alpha=0.8)
    ax2.axhline(y=1, color="red", linestyle="--", linewidth=1.5, label="Parity (1x)")
    ax2.set_xticks(range(len(batch_sizes)))
    ax2.set_xticklabels([f"$2^{{{int(np.log2(bs))}}}$" for bs in batch_sizes], fontsize=9)
    ax2.set_xlabel("Batch Size", fontsize=11)
    ax2.set_ylabel("Speedup (PyTorch / Triton)", fontsize=11)
    ax2.set_title("Speedup Factor", fontsize=12)
    ax2.legend(loc="upper right")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    # --- Plot 3: Effective TOPS ---
    ax3 = axes[2]
    tops_triton = [r["tops_triton"] for r in results]
    tops_torch = [r["tops_torch"] for r in results]
    ax3.plot(batch_sizes, tops_triton, "o-", label="Triton", color="#1f77b4", linewidth=2, markersize=6)
    ax3.plot(batch_sizes, tops_torch, "s-", label="PyTorch", color="#ff7f0e", linewidth=2, markersize=6)
    ax3.set_xscale("log", base=2)
    ax3.set_xlabel("Batch Size", fontsize=11)
    ax3.set_ylabel("Effective TOPS", fontsize=11)
    ax3.set_title("Throughput (Tera Ops/Sec)", fontsize=12)
    ax3.legend(loc="upper left")
    ax3.grid(True, which="both", linestyle="--", alpha=0.5)
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"$2^{{{int(np.log2(x))}}}$"))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton vs PyTorch ring matmul")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="Benchmark repetitions")
    parser.add_argument("--save-plot", action="store_true", help="Save plot to benchmark7.png")
    parser.add_argument("--max-exp", type=int, default=20, help="Max exponent for batch size (2^max_exp)")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  Ring MatMul Benchmark: Triton vs PyTorch")
    print("  Ring: F_7[x]/(x^6 - 1), Matrices: Batched 3x3")
    print("=" * 60)
    print()

    # Device info
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")
    print()

    # Correctness check
    print("[1/3] Running correctness check...")
    check_correctness(batch_size=256)
    print()

    # Benchmark sweep
    print("[2/3] Running benchmark sweep...")
    batch_sizes = [2**i for i in range(7, args.max_exp + 1)]  # 128 to 2^max_exp
    results = run_benchmark_sweep(batch_sizes, warmup=args.warmup, rep=args.rep)

    # Print table
    print("[3/3] Results:")
    print_results_table(results)

    # Generate plot
    if args.save_plot:
        generate_plot(results, save_path="benchmark7.png")


if __name__ == "__main__":
    main()
