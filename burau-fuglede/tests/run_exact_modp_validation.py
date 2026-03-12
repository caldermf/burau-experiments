#!/usr/bin/env python3

import argparse
import math
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd, *, cwd=ROOT, input_text=None):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        input=input_text,
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


def poly_add(a, b, sign=1):
    out = defaultdict(int)
    for k, v in a.items():
        out[k] += v
    for k, v in b.items():
        out[k] += sign * v
    return {k: v for k, v in out.items() if v}


def poly_shift(a, n):
    return {k + n: v for k, v in a.items()}


def pairing_poly(a, b, c, d, e):
    bl = 2 * a
    start = bl + b
    cl = start + b + 1
    end = cl + c
    el = 2 * d
    er = el + e
    suma = bl - 1
    sumb = 2 * start
    sumc = 2 * end
    sumd = el - 1
    sume = er + er - 1
    x = start
    poly = {}
    mon = {0: 1}

    while True:
        if x < el:
            if x < d:
                mon = poly_shift(mon, 1)
                poly = poly_add(poly, mon)
            else:
                poly = poly_add(poly, mon, sign=-1)
                poly = poly_shift(poly, 1)
            x = sumd - x
        else:
            if x < er:
                poly = poly_add(poly, mon, sign=-1)
                mon = poly_shift(mon, 1)
            else:
                poly = poly_shift(poly, 1)
                poly = poly_add(poly, mon)
            x = sume - x

        if x < cl:
            if x < bl:
                if x < a:
                    poly = poly_shift(poly, 1)
                else:
                    mon = poly_shift(mon, 1)
                x = suma - x
            else:
                if x < start:
                    mon = poly_shift(mon, 4)
                else:
                    poly = poly_shift(poly, 4)
                x = sumb - x
        else:
            if x < end:
                poly = poly_shift(poly, 1)
            elif x > end:
                mon = poly_shift(mon, 1)
            else:
                return poly
            x = sumc - x


def pairing_eval_mod(a, b, c, d, e, q, p):
    bl = 2 * a
    start = bl + b
    cl = start + b + 1
    end = cl + c
    el = 2 * d
    er = el + e
    suma = bl - 1
    sumb = 2 * start
    sumc = 2 * end
    sumd = el - 1
    sume = er + er - 1
    x = start
    poly = 0
    mon = 1
    q4 = pow(q, 4, p)

    while True:
        if x < el:
            if x < d:
                mon = (mon * q) % p
                poly = (poly + mon) % p
            else:
                poly = (poly - mon) % p
                poly = (poly * q) % p
            x = sumd - x
        else:
            if x < er:
                poly = (poly - mon) % p
                mon = (mon * q) % p
            else:
                poly = (poly * q) % p
                poly = (poly + mon) % p
            x = sume - x

        if x < cl:
            if x < bl:
                if x < a:
                    poly = (poly * q) % p
                else:
                    mon = (mon * q) % p
                x = suma - x
            else:
                if x < start:
                    mon = (mon * q4) % p
                else:
                    poly = (poly * q4) % p
                x = sumb - x
        else:
            if x < end:
                poly = (poly * q) % p
            elif x > end:
                mon = (mon * q) % p
            else:
                return poly % p
            x = sumc - x


def single_whisker(a, b, c, d, e):
    bl = 2 * a
    start = bl + b
    cl = start + b + 1
    end = cl + c
    el = 2 * d
    er = el + e
    suma = bl - 1
    sumb = 2 * start
    sumc = 2 * end
    sumd = el - 1
    sume = er + er - 1
    x = start
    togo = d + e

    while True:
        x = (sumd - x) if x < el else (sume - x)
        togo -= 1
        if x < cl:
            x = (suma - x) if x < bl else (sumb - x)
        else:
            if x == end:
                return togo == 0
            x = sumc - x


def passes_field_filter(a, b, c, d, e, p):
    return all(pairing_eval_mod(a, b, c, d, e, q, p) == 0 for q in range(p))


def exact_zero(a, b, c, d, e, p):
    poly = pairing_poly(a, b, c, d, e)
    return all(v % p == 0 for v in poly.values())


def tuples_up_to_level(max_level):
    for level in range(1, max_level + 1):
        yield from tuples_at_level(level)


def tuples_at_level(level):
    for a in range(level):
        for b in range(level - a):
            c = level - 1 - a - b
            for d in range(level + 1):
                e = level - d
                yield level, (a, b, c, d, e)


def reference_level_summary(p, max_level):
    summary = {}
    first_hit = None
    for level in range(1, max_level + 1):
        survivors = 0
        exact_hits = []
        for _, tup in tuples_at_level(level):
            a, b, c, d, e = tup
            if not single_whisker(a, b, c, d, e):
                continue
            if passes_field_filter(a, b, c, d, e, p):
                survivors += 1
                if exact_zero(a, b, c, d, e, p):
                    exact_hits.append(tup)
        summary[level] = {
            "candidates": level * (level + 1) * (level + 1) // 2,
            "survivors": survivors,
            "exact_hits": exact_hits,
        }
        if first_hit is None and exact_hits:
            first_hit = (level, exact_hits[0])
    return summary, first_hit


def build_cpu_checker(cc, out_path):
    run([cc, "-O2", "burau_exact_modp_check.c", "-o", str(out_path)])


def cpu_check_batch(cpu_bin, rows):
    input_text = "".join(
        f"{p} {a} {b} {c} {d} {e}\n" for p, (a, b, c, d, e) in rows
    )
    out = run([str(cpu_bin)], input_text=input_text)
    parsed = []
    for line in out.strip().splitlines():
        single, field, exact = [int(x) for x in line.split()]
        parsed.append((single, field, exact))
    if len(parsed) != len(rows):
        raise AssertionError("CPU checker output length mismatch")
    return parsed


def compile_gpu_binary(nvcc, prime, max_level, out_path):
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


def parse_gpu_output(text):
    level_re = re.compile(
        r"level=(\d+) candidates=(\d+) survivors=(\d+) found=(\d+)"
    )
    hit_re = re.compile(r"HIT level=(\d+) tuple=\(([^)]+)\)")
    levels = []
    hit = None
    for line in text.splitlines():
        m = level_re.search(line)
        if m:
            levels.append(
                {
                    "level": int(m.group(1)),
                    "candidates": int(m.group(2)),
                    "survivors": int(m.group(3)),
                    "found": int(m.group(4)),
                }
            )
        m = hit_re.search(line)
        if m:
            hit = (
                int(m.group(1)),
                tuple(int(x.strip()) for x in m.group(2).split(",")),
            )
    return levels, hit


def run_gpu_validation(nvcc, gpu_primes, gpu_level):
    print(f"[gpu] compiling and validating GPU binaries up to level {gpu_level}")
    for p in gpu_primes:
        out_path = ROOT / f".gpu_validate_p{p}"
        compile_gpu_binary(nvcc, p, gpu_level, out_path)
        stdout = run([str(out_path), "1", str(gpu_level)])
        gpu_levels, gpu_hit = parse_gpu_output(stdout)
        ref_summary, ref_hit = reference_level_summary(p, gpu_level)

        expected_len = ref_hit[0] if ref_hit is not None else gpu_level
        if len(gpu_levels) != expected_len:
            raise AssertionError(
                f"GPU level count mismatch for p={p}: "
                f"expected {expected_len}, got {len(gpu_levels)}"
            )

        for rec in gpu_levels:
            level = rec["level"]
            ref = ref_summary[level]
            if rec["candidates"] != ref["candidates"]:
                raise AssertionError(
                    f"GPU candidate count mismatch for p={p}, level={level}"
                )
            if rec["survivors"] != ref["survivors"]:
                raise AssertionError(
                    f"GPU survivor count mismatch for p={p}, level={level}: "
                    f"expected {ref['survivors']}, got {rec['survivors']}"
                )
            expected_found = 1 if ref_hit is not None and level == ref_hit[0] else 0
            if rec["found"] != expected_found:
                raise AssertionError(
                    f"GPU found flag mismatch for p={p}, level={level}: "
                    f"expected {expected_found}, got {rec['found']}"
                )

        if ref_hit is None:
            if gpu_hit is not None:
                raise AssertionError(f"GPU reported unexpected hit for p={p}")
        else:
            if gpu_hit != ref_hit:
                raise AssertionError(
                    f"GPU hit mismatch for p={p}: expected {ref_hit}, got {gpu_hit}"
                )

        print(f"[gpu] p={p} passed")


def run_cpu_validation(cc, exhaustive_level, random_tuples, random_level_cap):
    cpu_bin = ROOT / ".exact_modp_check"
    build_cpu_checker(cc, cpu_bin)

    print(f"[cpu] exhaustive cross-check up to level {exhaustive_level}")
    rows = []
    expected = []
    for p in (2, 3, 5):
        for _, tup in tuples_up_to_level(exhaustive_level):
            a, b, c, d, e = tup
            single = int(single_whisker(a, b, c, d, e))
            field = int(single and passes_field_filter(a, b, c, d, e, p))
            exact = int(single and exact_zero(a, b, c, d, e, p))
            rows.append((p, tup))
            expected.append((single, field, exact))
    got = cpu_check_batch(cpu_bin, rows)
    if got != expected:
        for idx, (lhs, rhs) in enumerate(zip(got, expected)):
            if lhs != rhs:
                raise AssertionError(
                    f"CPU exhaustive mismatch at index {idx}: expected {rhs}, got {lhs}"
                )

    print(f"[cpu] randomized cross-check with {random_tuples} tuples")
    rng = random.Random(12345)
    rows = []
    expected = []
    primes = [2, 3, 5, 7, 11]
    for _ in range(random_tuples):
        p = rng.choice(primes)
        level = rng.randint(1, random_level_cap)
        a = rng.randint(0, level - 1)
        b = rng.randint(0, level - 1 - a)
        c = level - 1 - a - b
        d = rng.randint(0, level)
        e = level - d
        tup = (a, b, c, d, e)
        single = int(single_whisker(a, b, c, d, e))
        field = int(single and passes_field_filter(a, b, c, d, e, p))
        exact = int(single and exact_zero(a, b, c, d, e, p))
        rows.append((p, tup))
        expected.append((single, field, exact))
    got = cpu_check_batch(cpu_bin, rows)
    if got != expected:
        for idx, (lhs, rhs) in enumerate(zip(got, expected)):
            if lhs != rhs:
                raise AssertionError(
                    f"CPU randomized mismatch at index {idx}: expected {rhs}, got {lhs}"
                )

    print("[cpu] deterministic sanity cases")
    sanity_cases = [
        (3, (0, 3, 0, 1, 3)),
        (5, (0, 7, 0, 3, 5)),
        (7, (0, 11, 0, 5, 7)),
        (5, (1, 1, 3, 5, 1)),
        (11, (5, 0, 4, 0, 10)),
    ]
    got = cpu_check_batch(cpu_bin, sanity_cases)
    for (p, tup), triple in zip(sanity_cases, got):
        a, b, c, d, e = tup
        ref = (
            int(single_whisker(a, b, c, d, e)),
            int(single_whisker(a, b, c, d, e) and passes_field_filter(a, b, c, d, e, p)),
            int(single_whisker(a, b, c, d, e) and exact_zero(a, b, c, d, e, p)),
        )
        if triple != ref:
            raise AssertionError(
                f"CPU sanity mismatch for p={p}, tuple={tup}: expected {ref}, got {triple}"
            )

    print("[cpu] passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--nvcc", default=None)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--gpu-primes", nargs="*", type=int, default=[3, 5])
    parser.add_argument("--exhaustive-level", type=int, default=18)
    parser.add_argument("--gpu-level", type=int, default=18)
    parser.add_argument("--random-tuples", type=int, default=2000)
    parser.add_argument("--random-level-cap", type=int, default=80)
    args = parser.parse_args()

    run_cpu_validation(
        args.cc,
        args.exhaustive_level,
        args.random_tuples,
        args.random_level_cap,
    )

    if not args.skip_gpu:
        if args.nvcc is None:
            raise SystemExit("--nvcc is required unless --skip-gpu is set")
        run_gpu_validation(args.nvcc, args.gpu_primes, args.gpu_level)

    print("ALL VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    main()
