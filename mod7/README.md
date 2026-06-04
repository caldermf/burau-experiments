# Standalone Mod-7 Triton Search

This folder packages the mod-7 reservoir/frontier search as a small, runnable GPU
pipeline. It is meant to be easy to read and hand to someone who wants to see the
Triton kernel path without also unpacking the rest of the repository.

## What It Searches

The search works in the reduced Burau representation of `B_4` over `F_7`. A
frontier element is a normalized `3x3` matrix of Laurent polynomials over
`F_7[t, t^-1]`, stored in a fixed 128-coefficient window after shifting away the
minimum degree.

The score being minimized is `projlen`:

```text
projlen = max nonzero degree - min nonzero degree
```

When a candidate has `projlen == 0`, all nonzero coefficients live at one degree
after normalization. Those candidates are saved for later decoding and checking.

## Files

- `search.py`: the actual PyTorch + Triton GPU search.
- `find_kernel.py`: compatibility entrypoint that runs `search.py`.
- `decode_words.py`: decodes saved `.pt` result files back into Garside suffix words.
- `cpu_reference_tests.py`: CPU reference tests copied from the original Triton work.
- `requirements.txt`: minimal runtime dependencies.

## Pipeline

1. `search.py` hardcodes the 22 proper nontrivial Garside suffixes of `B_4`.
   Each suffix is compiled into descriptor tuples describing sparse matrix terms:
   parent entry, degree shift, sign, and term count.

2. `build_seed_braids()` constructs the length-1 frontier. Each seed matrix is
   normalized, bit-sliced, and stored in structure-of-arrays layout:

   ```text
   data shape = [54, N]
   9 matrix entries * 3 coefficient bitplanes * 2 uint64 words = 54 lanes
   ```

3. At each generation, the host shuffles the current parents. This matters
   because the kernel uses first-come-first-served bucket admission; shuffling
   avoids always privileging the same parent order.

4. The host launches the same Triton kernel once per suffix, in random suffix
   order. For each parent/suffix pair, the kernel:

   - checks the Garside adjacency rule,
   - multiplies the parent matrix by the suffix using sparse descriptors,
   - performs bit-sliced mod-7 arithmetic on `uint64` registers,
   - computes the output matrix degree range,
   - normalizes by shifting out the minimum degree,
   - atomically admits the child into a capped projlen bucket,
   - writes accepted children into a flat output buffer.

5. Back on the host, children are ranked by low `projlen` with a small random
   tie-breaker. The best `USE_BEST` children become the next frontier.

6. Any `projlen == 0` children are written to `results/projlen0/` by default.
   Saved files include the bit-sliced matrix, metadata, braid length, and the full
   Garside suffix word used to reach the candidate.

## Running On A CUDA/Triton Machine

From the repository root:

```bash
python -m mod7.find_kernel \
  --use-best 7 \
  --bucket-cap 12 \
  --max-steps 170 \
  --save-dir mod7/results/projlen0
```

For a small GPU smoke test, use much smaller buffers:

```bash
python -m mod7.find_kernel \
  --use-best 0.05 \
  --bucket-cap 0.05 \
  --max-steps 5 \
  --save-dir mod7/results/smoke
```

Decode saved candidates:

```bash
python -m mod7.decode_words mod7/results/projlen0
python -m mod7.decode_words mod7/results/projlen0 --verify
```

The `--verify` mode decodes the saved bit-sliced matrix and recomputes the
stored `projlen`. It is a format/kernel-output sanity check, not a full independent
mathematical proof.

## Why Triton Here

This search is a good Triton fit because the hot loop is not a standard dense
GEMM. It is a custom expansion kernel with small sparse polynomial matrices,
bit-sliced arithmetic over `F_7`, degree scans, normalization, pruning, and
atomic bucket admission. Triton lets the kernel keep that whole per-candidate
pipeline close to the GPU registers and memory layout, while still being written
as readable Python-side code instead of CUDA C++.

The main optimization choices are:

- structure-of-arrays layout for coalesced reads across many candidate braids,
- hardcoded suffix descriptors to avoid runtime table loads,
- bit-sliced mod-7 arithmetic over packed `uint64` coefficient masks,
- one program per parent/suffix candidate with cheap adjacency rejection,
- capped projlen buckets to avoid materializing the full branching factor,
- host-side frontier selection only after the GPU has compacted accepted children.

## Caveats

- This package requires CUDA, PyTorch, and Triton. It was not executed on this
  machine because this machine does not have that GPU/Triton environment.
- The coefficient window is fixed at 128 degrees after normalization.
- This is the standalone mod-7 path. It intentionally does not depend on the
  broader `src/find_kernel.py` table-driven search.
- The separate `triton7.py` experiment is not included here; this package uses
  the reservoir-search kernel in `search.py`, which is the Triton path for the
  mod-7 search pipeline.
