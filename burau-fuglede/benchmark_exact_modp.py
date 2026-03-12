#!/usr/bin/env python3

import argparse
import re
import statistics
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(cmd, *, cwd=ROOT):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout


def build_cpu(cc, out_path):
    run([cc, "-O3", "burau_exact_modp_search.c", "-o", str(out_path)])


def build_gpu(nvcc, prime, max_level, out_path):
    run(
        [
            nvcc,
            "-O3",
            f"-DPRIME={prime}",
            f"-DMAX_LEVEL={max_level}",
            "burau_exact_modp_gpu.cu",
            "-o",
            str(out_path),
        ]
    )


def parse_cpu_output(text):
    no_hit = re.search(r"NO_HIT p=(\d+) stop=(\d+) checked=(\d+) final_level=(\d+)", text)
    if no_hit:
        return {
            "prime": int(no_hit.group(1)),
            "checked": int(no_hit.group(3)),
            "hit": False,
        }
    hit = re.search(r"HIT p=(\d+) stop=(\d+) tuple=\(([^)]+)\) checked=(\d+)", text)
    if hit:
        return {
            "prime": int(hit.group(1)),
            "checked": int(hit.group(4)) + 1,
            "hit": True,
            "tuple": tuple(int(x.strip()) for x in hit.group(3).split(",")),
        }
    raise RuntimeError(f"unexpected CPU output:\n{text}")


def parse_gpu_output(text):
    level_re = re.compile(r"level=(\d+) candidates=(\d+) survivors=(\d+) found=(\d+)")
    hit_re = re.compile(r"HIT level=(\d+) tuple=\(([^)]+)\)")
    levels = []
    hit = None
    for line in text.splitlines():
        match = level_re.search(line)
        if match:
            levels.append(
                {
                    "level": int(match.group(1)),
                    "candidates": int(match.group(2)),
                    "survivors": int(match.group(3)),
                    "found": int(match.group(4)),
                }
            )
            continue
        match = hit_re.search(line)
        if match:
            hit = (
                int(match.group(1)),
                tuple(int(x.strip()) for x in match.group(2).split(",")),
            )
    if not levels:
        raise RuntimeError(f"unexpected GPU output:\n{text}")
    return {
        "checked": sum(level["candidates"] for level in levels),
        "hit": hit is not None,
        "tuple": None if hit is None else hit[1],
        "levels": len(levels),
    }


def time_command(cmd, parser, repeats, warmup):
    parsed = None
    timings = []
    for idx in range(repeats + warmup):
        start = time.perf_counter()
        stdout = run(cmd)
        elapsed = time.perf_counter() - start
        current = parser(stdout)
        if parsed is None:
            parsed = current
        elif current != parsed:
            raise RuntimeError(
                "benchmark command produced inconsistent outputs across runs\n"
                f"first={parsed}\ncurrent={current}"
            )
        if idx >= warmup:
            timings.append(elapsed)
    return parsed, timings


def format_rate(checked, seconds):
    return checked / seconds if seconds > 0 else float("inf")


def summarize(label, checked, timings):
    best = min(timings)
    median = statistics.median(timings)
    return {
        "label": label,
        "checked": checked,
        "best_s": best,
        "median_s": median,
        "best_rate": format_rate(checked, best),
        "median_rate": format_rate(checked, median),
    }


def print_summary(summary):
    print(
        f"{summary['label']}: checked={summary['checked']} "
        f"best={summary['best_s']:.3f}s "
        f"median={summary['median_s']:.3f}s "
        f"best_rate={summary['best_rate']:.0f} cand/s "
        f"median_rate={summary['median_rate']:.0f} cand/s"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--prime", type=int, default=3)
    parser.add_argument("--stop-level", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    cpu_bin = ROOT / f".bench_cpu_p{args.prime}"
    gpu_bin = ROOT / f".bench_gpu_p{args.prime}"

    build_cpu(args.cc, cpu_bin)
    build_gpu(args.nvcc, args.prime, args.stop_level, gpu_bin)

    cpu_cmd = [str(cpu_bin), str(args.stop_level), "0", str(args.prime)]
    gpu_cmd = [str(gpu_bin), "1", str(args.stop_level)]

    cpu_result, cpu_timings = time_command(cpu_cmd, parse_cpu_output, args.repeats, args.warmup)
    gpu_result, gpu_timings = time_command(gpu_cmd, parse_gpu_output, args.repeats, args.warmup)

    if cpu_result["hit"] != gpu_result["hit"]:
        raise RuntimeError(
            f"CPU/GPU hit disagreement in benchmark window: cpu={cpu_result}, gpu={gpu_result}"
        )
    if cpu_result["hit"] and cpu_result.get("tuple") != gpu_result.get("tuple"):
        raise RuntimeError(
            f"CPU/GPU tuple disagreement in benchmark window: cpu={cpu_result}, gpu={gpu_result}"
        )

    cpu_summary = summarize("cpu_exact", cpu_result["checked"], cpu_timings)
    gpu_summary = summarize("gpu_exact", gpu_result["checked"], gpu_timings)

    print_summary(cpu_summary)
    print_summary(gpu_summary)

    best_speedup = gpu_summary["best_rate"] / cpu_summary["best_rate"]
    median_speedup = gpu_summary["median_rate"] / cpu_summary["median_rate"]
    print(
        f"speedup: best={best_speedup:.2f}x median={median_speedup:.2f}x "
        f"(prime={args.prime}, stop_level={args.stop_level}, repeats={args.repeats})"
    )
    if cpu_result["hit"]:
        print(
            "warning: a hit occurred inside the benchmark window, so CPU and GPU "
            "did not stop at exactly the same point inside the final level"
        )


if __name__ == "__main__":
    main()
