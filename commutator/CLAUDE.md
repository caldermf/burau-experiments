# Commutator Braid Kernel Search

GPU-accelerated search for kernel elements of the Burau representation of B_4 (4-strand braid group), specifically elements of the form [σ_i, g^{-1}].

## Project context

This is a research project by Calder (Yale math professor) in collaboration with Geordie Williamson. We are searching for kernel elements of the Jones/Burau representation of B_4 reduced mod a prime p. The original algorithm (adapted from Williamson's arXiv:2310.02403) does a breadth-first search through Garside normal forms with GPU-accelerated polynomial matrix multiplication and reservoir sampling by "projlen" (projective length = max degree minus min degree of the matrix entries). This variant restricts the search to commutators [σ_i, g^{-1}], which Calder and Geordie believe is a productive subspace to search.

## Architecture

### Files

- **`braid_search.py`** — Original GPU search engine. Provides `Config`, `BraidSearchUltra`, `FastPolyMatmul`, `GPUBuckets`, `compute_projlen_batch`, `is_scalar_identity_batch`, `build_expansion_indices_vectorized`, `load_tables_from_file`. Do not modify without understanding downstream effects.
- **`commutator_braid_search.py`** — Commutator variant. Core module. Provides `CommutatorConfig`, `CommutatorBraidSearch`, `CommutatorFastPolyMatmul`, `load_tables_for_commutator`, plus permutation utilities and polynomial matrix arithmetic for precomputing twisted matrices. Imports from `braid_search.py`.
- **`find_commutator_kernel.py`** — Entry point for commutator search. Handles CLI args, table loading, search execution, and verification via peyl.
- **`find_kernel.py`** — Entry point for the original (non-commutator) search. Keep for comparison.
- **`kernel_database.py`** — Persistent storage of found kernel elements (imported by `find_kernel.py`).
- **`precomputed_tables/tables_B4_r1_p{p}.pt`** — PyTorch tensor files with Burau matrices and Garside suffix tables for each prime p. Generated separately via peyl.

### Key mathematical structures

**Braid group B_4**: 4-strand braids with generators σ_1, σ_2, σ_3. Garside normal form writes braids as products of "simple" elements (one per permutation in S_4, so 24 simples indexed 0–23). Index 0 = identity, index 23 = Δ (half-twist).

**Permutation indexing**: Lexicographic order on S_4. Key indices:
- 0 = (0,1,2,3) = identity
- 1 = (0,1,3,2) = σ_3
- 2 = (0,2,1,3) = σ_2
- 6 = (1,0,2,3) = σ_1
- 13 = (2,0,3,1) = σ_2σ_3σ_1 (order 4 mod p=5, appears with huge frequency in known kernel elements)
- 23 = (3,2,1,0) = Δ

**Garside normal form validity**: Simple b can follow simple a iff left_descent_set(b) ⊆ right_descent_set(a). The `valid_suffixes` and `num_valid_suffixes` tables encode this. Identity gets Delta's suffix table (since after a Delta power, anything can follow).

**Burau representation**: Each simple maps to a 3×3 matrix over Z[v] mod p. Stored as tensors of shape (24, 3, 3, D) where D is the degree window size.

**Projlen (projective length)**: For a polynomial matrix, projlen = (max nonzero degree) - (min nonzero degree) + 1 across all entries. Kernel elements have projlen ≤ 1 (scalar multiple of identity or anti-diagonal).

### Commutator search specifics

**Goal**: Find g such that [σ_i, g^{-1}] = σ_i · g^{-1} · σ_i^{-1} · g is in the kernel.

**Update rule**: Define T_b = M_{σ_i} · M_b^{-1} · M_{σ_i}^{-1} (precomputed for all 24 simples). Then:
```
C_{g·b} = T_b · C_g · M_b
```
where C_g = Burau([σ_i, g^{-1}]). So we track ONE matrix per braid with TWO matmuls per expansion step.

**Avoidance condition**: The first Garside factor b_1 must satisfy left_descent_set(b_1) ∩ J = ∅ where J = centralizer generators of σ_i. This eliminates redundancy since [σ_i, (c·g)^{-1}] = [σ_i, g^{-1}] for c in the centralizer.

| Generator | Centralizer gens (0-indexed) | Allowed first factors |
|-----------|------------------------------|----------------------|
| σ_1       | {0, 2} (= {s_1, s_3})       | 5 of 23              |
| σ_2       | {1} (= {s_2})               | 11 of 23             |
| σ_3       | {0, 2} (= {s_1, s_3})       | 5 of 23              |

**Centered degree window**: Unlike the original search (non-negative degrees only), the commutator search uses a centered window [-center, center] because T_b involves matrix inverses with negative v-degrees. Degree 0 sits at array index `center`. This costs 2× memory vs the original but is necessary.

**FFT aliasing prevention**: The double convolution (T_b · C_g · M_b) is done as two separate matmuls with a truncation/recentering step in between to prevent the intermediate polynomial from exceeding the FFT window.

**Kernel detection**: Checks both scalar multiples of identity (even Δ powers) and scalar anti-diagonal matrices (odd Δ powers).

## Usage

```bash
# Quick test (CPU, small)
python find_commutator_kernel.py --p 5 --gen 1 --bucket-size 4000 --max-length 30 --device cpu

# Production run on GPU
python find_commutator_kernel.py --p 5 --gen 1 --bucket-size 100000 --use-best 50000 --max-length 60 --device cuda

# Try σ_2 (more allowed first factors)
python find_commutator_kernel.py --p 5 --gen 2 --bucket-size 100000 --use-best 50000 --max-length 60

# Original (non-commutator) search for comparison
python find_kernel.py --p 5 --bucket-size 100000 --use-best 100000 --max-length 60 --device cuda
```

### Key parameters

- `--gen {1,2,3}`: Which generator σ_i to use. σ_2 gives 11 allowed first factors; σ_1 and σ_3 give 5 each.
- `--bucket-size`: Reservoir size per projlen bucket. Larger = more diverse sample, more memory.
- `--use-best`: How many braids (prioritizing low projlen) to expand each level. 0 = expand all.
- `--max-length`: Maximum Garside length of g.
- `--degree-multiplier`: Controls degree window size. Window = 2 × multiplier × max_length + 1. Reduce if memory is tight.
- `--matmul-chunk`: Chunk size for FFT matmul batches. Reduce for less GPU memory.
- `--bootstrap-length`: How many initial levels to keep ALL braids (no reservoir sampling).

### Memory guidelines

The commutator search uses ~2× the degree window of the original search. For H200 (80GB): `--bucket-size 100000 --matmul-chunk 30000`. For RTX 5000 (32GB): `--bucket-size 50000 --matmul-chunk 15000`.

## Important invariants

1. **The word stored for each braid is the Garside factor sequence of g, NOT of the commutator [σ_i, g^{-1}]**. The commutator is reconstructed during verification.
2. **The matrix stored is the Burau matrix of the commutator [σ_i, g^{-1}]**, not of g itself.
3. **All polynomial arithmetic is mod p**. Coefficients are in {0, 1, ..., p-1}.
4. **The `is_scalar_identity_batch` check uses projlen=1 as a necessary condition** (projlen 0 = zero matrix, projlen 1 = monomial entries). But projlen 1 is not sufficient — the matrix must also be a scalar multiple of identity (or anti-diagonal for odd Δ powers).
5. **Precomputed twisted matrices T_b are computed at startup**, not loaded from the table file. They depend on the choice of generator σ_i.

## Common pitfalls

- **Don't confuse `center` (index of degree 0 in the centered array) with the loaded table's center**. The loaded tables have their own `center` field. The commutator search remaps into its own centered window.
- **The intermediate truncation between the two matmuls is critical**. Without it, the second convolution aliases in the FFT. See `CommutatorFastPolyMatmul.commutator_expand_batch`.
- **poly_mul_mod uses float conv1d** — this is fast but can accumulate rounding errors for very large degree windows and large primes. For p ≤ 7 and typical degree windows this is fine. For larger primes, consider switching to exact integer convolution (e.g. via NTT).
- **The `itertools` import in `commutator_braid_search.py` is unused** — leftover, harmless.

## Verification

When peyl is available, `find_commutator_kernel.py` verifies found elements by:
1. Building g from the Garside factor word
2. Computing the commutator [σ_i, g^{-1}] symbolically
3. Evaluating via peyl's `JonesCellRep` and checking the result is a scalar matrix

If peyl is not installed, verification is skipped and only the GPU-side check (projlen=1 + scalar identity/anti-diagonal) is used.

## Dependencies

- PyTorch (with CUDA for GPU runs)
- `braid_search.py` (must be importable)
- Precomputed table files (`tables_B4_r1_p{p}.pt`)
- Optional: peyl (for verification), kernel_database (for persistence)
