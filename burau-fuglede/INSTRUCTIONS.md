# Exact Mod-p GPU Search Instructions

This repository contains a CUDA-oriented search implementation in
`burau_exact_modp_gpu.cu` and an independent validation harness in
`tests/run_exact_modp_validation.py`.

Do not trust GPU search results until the validation harness passes.

## What the validator checks

The validator is deliberately redundant. It compares three independent views of
the problem:

1. A pure Python reference implementation in
   `tests/run_exact_modp_validation.py`.
2. A dense exact CPU checker in `burau_exact_modp_check.c`.
3. The CUDA search code in `burau_exact_modp_gpu.cu`.

The checks include:

- exhaustive agreement on all tuples up to a chosen level for small primes,
- randomized tuple checks at larger levels,
- GPU per-level survivor counts compared against the independent Python
  reference,
- GPU first-hit behavior compared against the independent Python reference.

If any mismatch appears, stop immediately and debug before running large GPU
searches.

## Prerequisites

You need:

- `python3`
- a C compiler available as `cc` or `gcc`
- `nvcc` on `PATH`

If `nvcc` is not on `PATH`, load the appropriate CUDA module first on the
cluster.

## Recommended validation command

Paste this from the repository root:

```bash
python3 tests/run_exact_modp_unit_checks.py && python3 tests/run_exact_modp_validation.py --cc cc --nvcc nvcc --gpu-primes 3 5 7 --exhaustive-level 18 --gpu-level 18 --random-tuples 2000 --random-level-cap 80
```

That is the recommended baseline validation run.

## Stronger validation run

If the baseline passes and you want a stricter check before long production
searches, use:

```bash
python3 tests/run_exact_modp_unit_checks.py && python3 tests/run_exact_modp_validation.py --cc cc --nvcc nvcc --gpu-primes 3 5 7 11 --exhaustive-level 20 --gpu-level 20 --random-tuples 5000 --random-level-cap 120
```

This is slower, but gives more confidence.

## CPU-only fallback

If you temporarily do not have CUDA available, you can still run the full
CPU/reference validation:

```bash
python3 tests/run_exact_modp_unit_checks.py && python3 tests/run_exact_modp_validation.py --skip-gpu
```

## Expected success signal

On success, the script ends with:

```text
ALL VALIDATION CHECKS PASSED
```

If that line does not appear, do not trust the GPU search.

## Benchmarking

After validation passes, benchmark the exact CPU baseline against the exact GPU
search with:

```bash
python3 benchmark_exact_modp.py --cc cc --nvcc nvcc --prime 3 --stop-level 80 --repeats 3 --warmup 1
```

This reports:

- exact candidates checked,
- best and median wall time,
- best and median candidate throughput,
- CPU-to-GPU speedup.

Benchmark only after validation passes.
This benchmark is apples-to-apples for the exact mod-`p` search; it is not a
comparison against the older heuristic code in `original/original_burau.c`.

## After validation

Once validation passes, you can compile and run the GPU search itself. For
example, for `p=3`:

```bash
nvcc -O3 -DPRIME=3 -DMAX_LEVEL=512 burau_exact_modp_gpu.cu -o burau_exact_modp_gpu_p3
./burau_exact_modp_gpu_p3 1 512
```

If you change any logic in `burau_exact_modp_gpu.cu`, rerun the full validation
before using new search results.
