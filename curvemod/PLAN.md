# GPU Rewrite Plan for `A3_curve_Fp_parallel.py`

## Goal

Rewrite the current A3 search as a PyTorch/CUDA pipeline that keeps the full search state on the GPU, uses precomputed tensorized Burau actions and Garside transition tables, and replaces Python list/dict bucket logic with batched tensor operations and GPU-friendly sampling. The target is a large constant-factor speedup, not a mild cleanup.

## Current Bottlenecks

The current implementation is slow for structural reasons:

- Laurent polynomials are represented as Python dictionaries, so every Burau application is many tiny Python allocations and hash lookups. We should instead use a tensor of 3x3 matrices and efficient convolution; this should be easy.
- Curves are expanded one at a time in Python loops.
- Dual Garside admissibility is computed by chasing Python lists and dictionaries.
- Reservoir sampling is done by mutating Python lists.
- The code repeatedly recomputes Burau actions for the same generators instead of treating them as reusable linear operators.
- Multiprocessing only hides some Python overhead; it does not fix the data model.

## Rewrite Principles

1. Keep all per-curve state on the GPU once the search starts.
2. Precompute every small combinatorial object once:
   - oriented Artin Burau generators
   - dual atom words
   - dual simple representatives
   - simple-to-simple admissibility / `allowed_suffix`
   - initial simple mask
3. Replace Laurent-dict arithmetic by dense coefficient tensors over a fixed degree window.
4. Replace per-state Python loops by batched tensor kernels.
5. Replace sequential reservoir logic by random-priority selection, which is equivalent for our use and composable across GPU batches.

## Scope

Phase 1 should target the existing `Fp` search only, meaning `PP > 0`.

Reason:

- modular arithmetic is natural on GPU
- integer coefficient growth over `Z` is much harder to control
- correctness is easier to validate against the current code

If we later want `PP = 0`, that should be a separate extension.

## Proposed Data Model

### 1. Curve state tensor

Each curve currently stores a dimension vector with 3 Laurent polynomials. Replace that with a dense tensor

`state: [batch, n_vertices, q_width]`

where:

- `n_vertices = 3` for A3
- `q_width` is a fixed degree window
- the middle index corresponds to degree 0
- entries are coefficients mod `PP`

For A3 over `Fp`, `torch.int16` or `torch.int32` is enough depending on how PyTorch handles modular arithmetic efficiently on the target GPU.

### 2. Degree window

Do not hard-code the width blindly from the current Python script.

Instead:

1. Precompute the min/max degree shift of every dual simple action.
2. Derive a safe global bound for all search depths up to `MAX_G_LENGTH`.
3. Allocate one fixed `q_width` for the entire run.

This keeps the GPU path simple and avoids reallocation during search.

### 3. Word reconstruction

Do not store full braid words per state on the GPU.

Instead store:

- `last_simple_id`
- `parent_index`
- `depth`

for each retained state. Reconstruct the witness only when a spread-0 state is found.

This removes a major memory and transfer cost.

## Burau Tensorization

### 1. Artin generators

Precompute the oriented Burau action for each Artin letter in tensor form.

For A3 there are only 6 letters:

- `-3, -2, -1, 1, 2, 3`

Each letter acts by a very sparse shifted linear map. Represent each generator as a small stencil, not as Python code.

Two viable implementations:

- sparse-shift form: for each output row, store `(input_row, q_shift, coeff)` triples
- dense band form: a tensor `[n_letters, n_vertices, n_vertices, n_shifts]`

The sparse-shift form is probably better because the matrices are tiny and highly structured.

### 2. Dual simples

Precompute each dual simple as a composed tensor operator once at startup.

For the search, we should never apply Artin letters one by one. We should apply a precompiled dual simple operator directly.

Represent:

- `simple_word_table`: CPU metadata for reconstruction
- `simple_operator_table`: GPU tensor operator for batched application

### 3. Basis-vector convention

The current search evolves only the image of basis vector `1`, not the full 3x3 Burau matrix.

Keep that in the first GPU rewrite unless we discover a correctness reason not to. It is much cheaper and matches the current algorithm exactly.

So the primary search state remains:

`[batch, 3, q_width]`

not

`[batch, 3, 3, q_width]`.

## Garside / Automaton Precomputation

The small A3 combinatorics should be precomputed once on CPU and then transferred to GPU as integer tables.

### 1. Enumerate simples once

Use the existing logic only as a bootstrap/reference implementation, then freeze the resulting simple list.

For A3, the set is small enough that this should be deterministic and cheap.

### 2. Replace dictionary-based automaton with tables

Build:

- `simple_ids`: canonical indexing of dual simples
- `allowed_mask: [n_simples, n_simples]` boolean
- `allowed_suffix_padded: [n_simples, max_out_degree]` with `-1` padding
- `allowed_count: [n_simples]`
- `start_mask: [n_simples]` for valid initial simples

These tables replace:

- `RightDescents`
- `LeftDescents`
- `GarsideAutomaton`
- repeated `findRepresentative(...)`

### 3. Prefer id-based state

Each state should carry a `last_simple_id`, not a Python braid word.

The expansion step then becomes a gather from `allowed_suffix_padded[last_simple_id]`.

## Batched Search Kernel

## Step structure

At each Garside depth:

1. Select the previous spread buckets to expand, respecting `TOTAL_CAP`.
2. Materialize a batch of retained states on GPU.
3. Expand all admissible suffixes using the `allowed_suffix` table.
4. Apply the corresponding precomputed dual-simple operators.
5. Compute spread, drop count, and zero-spread hits.
6. Insert survivors into next-step buckets using random-priority top-k selection per spread.

### 1. Expansion strategy

Do not expand one state at a time.

Better pattern:

- partition current states by `last_simple_id`
- for each simple id `s`, gather all states ending in `s`
- broadcast over allowed successors of `s`
- apply those successor operators to the whole sub-batch

This keeps the number of GPU launches tied to the number of simple ids, not the number of states.

Since A3 has very few simples, a small outer Python loop over simple ids is acceptable. The expensive work must be tensorized.

### 2. Burau application

For each candidate successor simple `t`, apply its precomputed operator to a batch of vectors.

This should be implemented as a custom batched shifted linear combination, not as general dense matrix multiplication.

The operation is structurally:

- shift selected coefficient slices in the q-direction
- multiply by small integer coefficients
- sum across source rows
- reduce mod `PP`

### 3. Spread computation

Compute spread directly from tensor support:

1. build a boolean support mask `state != 0`
2. reduce over vertex axis to get support per q-degree
3. find first and last occupied degree
4. `spread = top - bottom`

This avoids Python `topdeg_vector` / `botdeg_vector` entirely.

### 4. Initial state generation

Also move initialization to GPU:

- apply each valid start simple to `e_1`
- compute spread
- keep only spread-1 states
- initialize parent pointers with sentinel parent `-1`

No Python bucket-building should survive.

## GPU-Friendly Reservoir Sampling

Classic sequential reservoir sampling is a bad fit for GPU kernels. Use random-priority sampling instead.

### Principle

For each candidate state, generate a random priority key on GPU.

For each spread bucket, retain the `CAP` states with best priorities.

This is equivalent to a reservoir sample of fixed size when priorities are i.i.d. continuous random variables, and it composes cleanly across chunks.

### Implementation

For each spread bucket maintain:

- `state_tensor`
- `last_simple_id`
- `parent_index`
- `priority`

When new candidates arrive:

1. concatenate existing retained items and new candidates
2. use `topk` on priority to keep the best `CAP`
3. drop the rest

Do the same at the global step-selection level for `TOTAL_CAP`, but there the rule is different:

- compute bucket counts by spread
- prefix-sum from smallest spread upward
- only expand buckets up to the cutoff spread

That logic can stay on CPU if needed because the number of spread buckets is tiny.

## Correctness Plan

We should not trust the GPU version until it matches the current code on small runs.

### Oracle-based validation

Use the existing Python implementation as the reference on tiny parameters:

- small `MAX_G_LENGTH`
- small caps
- several primes `PP`

For every step, compare:

- set of reachable spreads
- bucket sizes per spread after sampling is disabled
- exact evolved vectors for deterministic micro-tests
- existence and depth of first spread-0 witness

### Disable sampling during validation

Add a debug mode where:

- `CAP = infinity`
- `TOTAL_CAP = infinity`
- ordering is deterministic

This makes exact comparison possible before introducing sampling.

## Implementation Phases

### Phase 0: Freeze reference behavior

- record the current A3 simple list
- record the current automaton / admissibility table
- build small deterministic regression tests

Deliverable:

- a CPU test harness that asserts current outputs on small examples

### Phase 1: Tensorize Laurent vectors and Artin actions

- design dense coefficient encoding
- implement GPU/CPU tensor versions of Artin-letter action
- verify equality with `setup_a3.oburau_fns`

Deliverable:

- a tested module that applies Artin words to batches of vectors

### Phase 2: Precompute dual simples and admissibility tables

- compile dual simple operators
- compile `allowed_suffix` and `start_mask`
- replace `findRepresentative` and string-key automaton logic

Deliverable:

- a small static data package for A3 search

### Phase 3: GPU batch search without sampling

- implement batched initialization
- implement batched step expansion
- implement spread computation on GPU
- implement witness parent-pointer storage

Deliverable:

- exact-search GPU engine matching CPU results on small runs

### Phase 4: GPU reservoir / priority sampling

- implement per-spread capped retention with random priorities
- match CPU behavior statistically, not bit-for-bit
- benchmark throughput and memory

Deliverable:

- scalable GPU search with caps

### Phase 5: Performance tuning

- tune batch sizes
- choose integer dtype
- reduce host-device synchronization
- fuse operator application and spread computation where worthwhile
- benchmark single-GPU throughput against current CPU code

Deliverable:

- stable fast path with timing tables

## Likely File Structure

Suggested split:

- `a3_gpu_tables.py`
  - dual simples
  - simple ids
  - admissibility tables
  - compiled operators
- `a3_gpu_burau.py`
  - tensorized Burau actions
  - q-degree window helpers
  - spread computation
- `a3_gpu_search.py`
  - batched search engine
  - parent-pointer reconstruction
  - priority sampling
- `tests/test_a3_gpu_reference.py`
  - CPU/GPU equivalence on small cases

This keeps the math tables, kernels, and search logic separate.

## Key Risks

### 1. Degree-window mistakes

If the q-degree window is too small, coefficients silently fall off the edge and the search becomes wrong.

Mitigation:

- derive bounds from compiled operators
- add runtime assertions in debug mode

### 2. Integer dtype / modulo behavior

PyTorch integer ops on GPU are reliable, but dtype choice matters.

Mitigation:

- start with `int32`
- reduce mod `PP` after every operator application
- benchmark later

### 3. Memory blow-up from parent storage

If we store all retained states from all depths naively, witness reconstruction metadata may dominate memory.

Mitigation:

- store only retained states
- store compact parent indices and simple ids
- optionally checkpoint only every few depths if needed later

### 4. Overengineering tiny outer loops

There is no need to eliminate every Python loop. The real problem is per-state Python work.

Mitigation:

- allow small loops over simple ids or spread buckets
- move only the large data-parallel work to GPU

## Success Criteria

The rewrite is successful if it satisfies all of the following:

- matches the current CPU search on small deterministic runs
- keeps the main search state resident on GPU
- does not use Laurent dicts or per-state Python loops in the hot path
- uses precomputed simple operators and admissibility tables
- uses GPU-side capped retention via random-priority sampling
- materially outperforms the current multiprocessing CPU script

## Recommended First Implementation Move

Do not start with the whole search.

Start by implementing one tested building block:

`apply_simple_batch(states, simple_id, p) -> new_states`

for states shaped `[batch, 3, q_width]`.

If this kernel is right and fast, the rest of the GPU rewrite becomes mostly search bookkeeping. If this kernel is wrong, everything above it is built on sand.
