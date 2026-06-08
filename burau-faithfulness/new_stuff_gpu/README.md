# CUDA-friendly mod p bucket searches

This folder contains tensorized versions of the mod-p bucket searches.

What changed:

- live states are stored as dense tensors, not Laurent-dictionary objects
- transitions are applied in batches grouped by leftmost Garside simple
- search states avoid Laurent dictionaries in the hot loop; drivers either use compact factor rows or parent-style reconstruction
- the search has a `torch` backend for CUDA devices, with a `numpy` fallback for local development

Files:

- `setup_a3.py`: the same type `A_3` oriented Burau setup as before
- `tensor_backend.py`: backend abstraction for `numpy` or `torch`
- `a3_mod_p_bucket_search.py`: optimized search driver
- `d4_gpu_common.py`: shared type `D_4` Garside/Burau helpers
- `d4_mod_p_gpu_native_search.py`: GPU-native type `D_4` bucket search
- `verify_d4_burau_kernel.py`: verifier for type `D_4` search hits

## New Codex GPU handoff

Start from the repository root and enter this directory:

```bash
cd new_stuff_gpu
```

First confirm the GPU Python environment:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Then run a tiny CUDA smoke test. This should finish quickly, build 50 D4 Garside simples, and write a JSON file. It is not expected to find a hit.

```bash
python d4_mod_p_gpu_native_search.py \
  --backend torch \
  --device cuda \
  --p 7 \
  --max-g-length 4 \
  --cap-1 50 \
  --cap-2 50 \
  --total-cap-1 200 \
  --total-cap-2 200 \
  --first-steps 3 \
  --expansion-chunk-size 1024 \
  --output data/d4_cuda_smoke.json
```

If the smoke test succeeds, run the first real target: reproduce/search the published-style `p=7` D4 case using the old CPU script's effective caps.

```bash
python d4_mod_p_gpu_native_search.py \
  --backend torch \
  --device cuda \
  --p 7 \
  --max-g-length 1000 \
  --cap-1 4000 \
  --cap-2 4000 \
  --total-cap-1 50000 \
  --total-cap-2 50000 \
  --first-steps 12 \
  --expansion-chunk-size 65536 \
  --output data/d4_p7_cuda_seed0.json
```

Verify whatever candidates were written:

```bash
python verify_d4_burau_kernel.py \
  --backend torch \
  --device cuda \
  --input data/d4_p7_cuda_seed0.json \
  --output data/d4_p7_cuda_seed0_verified.json
```

Report back:

- the search JSON path and verifier JSON path
- `num_hits`, `runtime_seconds`, and whether `hit_limit_reached` is true
- verifier counts: `verified_commutator_identity_count`, `verified_same_curve_count`, and `total_candidates`
- the last few progress lines from the search, especially the first step that produced hits

If `p=7` seed 0 has no verified hits, rerun the same command with `--seed 1`, then `--seed 2`. If memory is comfortable, next try `--cap-1 8000 --cap-2 8000`; keep `--total-cap-*` at `50000` for the first comparison.

CUDA has not been run locally yet. CPU-only checks already passed, including verification of the paper's displayed `p=7` witness.

## Type D4 GPU search

Recommended cluster usage:

```bash
python d4_mod_p_gpu_native_search.py \
  --backend torch \
  --device cuda \
  --p 7 \
  --max-g-length 1000 \
  --cap-1 4000 \
  --cap-2 4000 \
  --total-cap-1 50000 \
  --total-cap-2 50000 \
  --first-steps 12 \
  --expansion-chunk-size 65536 \
  --output data/d4_p7_cuda.json
```

Then verify candidates:

```bash
python verify_d4_burau_kernel.py \
  --backend torch \
  --device cuda \
  --input data/d4_p7_cuda.json \
  --output data/d4_p7_cuda_verified.json
```

Notes:

- `--hit-condition spread-zero` is the default and matches the original D4 script's broad hit condition. The verifier then checks the actual same-curve condition and the commutator Burau matrix.
- Use `--hit-condition same-curve` to serialize only candidates where the searched vector is already `q^l alpha_i`.
- The old CPU script used `CAP=500` on each of 8 workers, so `--cap-1 4000 --cap-2 4000` matches its effective per-spread bucket cap.
- `--storage-dtype auto` uses `int16` for small primes and `int32` for larger primes. The action tensors keep small signed coefficients, and states are reduced modulo `p`.

CPU smoke test:

```bash
python d4_mod_p_gpu_native_search.py \
  --backend numpy \
  --device cpu \
  --p 7 \
  --max-g-length 4 \
  --cap-1 50 \
  --cap-2 50 \
  --total-cap-1 200 \
  --total-cap-2 200 \
  --first-steps 3 \
  --output /tmp/d4_smoke.json
```

## Type A3 legacy GPU search

Recommended cluster usage on an RTX Ada node:

```bash
python a3_mod_p_bucket_search.py \
  --backend torch \
  --device cuda \
  --p 3 \
  --max-g-length 40 \
  --cap-1 500 \
  --cap-2 500 \
  --total-cap-1 50000 \
  --total-cap-2 50000 \
  --first-steps 12 \
  --output data/a3_p3_cuda.json
```

Local CPU fallback:

```bash
python3 new_stuff_gpu/a3_mod_p_bucket_search.py \
  --backend numpy \
  --device cpu \
  --p 3 \
  --max-g-length 20 \
  --stop-at-first
```

Local validation performed here:

- original `new_stuff` first-hit search (`p=3`, `max-g-length=20`) took about `39.5s`
- optimized `new_stuff_gpu` `numpy` backend on the same task took about `3.47s`

The `torch` backend was also validated locally on CPU. On the Yale cluster the intended path is `--backend torch --device cuda`.
