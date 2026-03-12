# PLAN.md: Correct Mod-p Search for Zero Burau Polynomials in D4

## Summary
- Base the implementation only on the paper sources in this repo: fix `\alpha` as in Reduction II, keep the paper’s standard-form restriction on `\beta` from Reduction III, and enumerate weighted train-track data exactly as in Section 4.
- Modify the paper’s integer-only shortcut for the mod-`p` setting: Reduction I (“only even intersection numbers”) is valid over `Z`, but for odd primes it is not correctness-preserving. Search all positive intersection counts for odd `p`; keep the even-only restriction only for `p = 2`.
- Use a CPU-first exhaustive search up to a configurable intersection bound `N`, with case-by-case candidate generation, a one-sided folded-polynomial rejection filter, and exact final verification.

## Public Interfaces / Core Types
- `WeightTuple = (w0, w1, w2, w3, w14)` with derived `h = w14 / 2` and derived rails `w8..w13`.
- `generate_candidates(p: int, max_intersections: int) -> Iterator[WeightTuple]`
- `precheck_candidate(weights: WeightTuple, p: int, fold_bits: int = 7) -> bool`
  Returns `False` only when the candidate is provably not a mod-`p` zero.
- `evaluate_burau_exact(weights: WeightTuple) -> dict[int, int]`
  Exact sparse Laurent polynomial over `Z`, keyed by exponent.
- `reduce_poly_mod_p(poly: dict[int, int], p: int) -> dict[int, int]`
- `search_mod_p(p: int, max_intersections: int, workers: int, fold_bits: int = 7) -> list[WeightTuple, dict[int, int]]`

## Key Implementation Rules
- Correct the divisibility typo in Section 4 before coding: use the prose and Example 5.3, not the displayed `D2/D3` line. The consistent parity is `w0, w1` even and `w2, w3` odd.
- Keep the paper’s proper-multicurve restrictions exactly: require `gcd(w0, w1, w2, w3) = 1` and require at least one of `w8 = 0`, `w9 = 0`, `w12 = 0`. Enumerate Cases I/II/III exactly as in Section 4, deduping overlaps by the full 5-tuple.
- Do not use the family heuristics from the analysis section for pruning; they are observational, not correctness-preserving.

## Candidate Enumeration
- For every target intersection count `c`:
- If `p = 2`, skip odd `c`. If `p` is odd, search every `c >= 1`.
- Case I (`w8 = 0`): set `h = w0 + w1 = 2c`. Enumerate even `w0`, set `w1 = 2c - w0`, then enumerate odd `w2, w3` satisfying the paper’s Case I inequalities.
- Case II (`w9 = 0`): set `h = w0 - w1 = 2c`. Enumerate odd `w2, w3 < 2c` with `w2 + w3 >= 2c`, enumerate even `w1`, set `w0 = w1 + 2c`, then enforce the paper’s Case II inequalities.
- Case III (`w12 = 0`): set `h = w2 + w3`.
- Subcase IIIa (`h <= w0`): `w1 = w0 - 2c`. Use the Case III inequalities plus `w3 < w0 < w3 + 2c`; they imply `w2, w3 < 2c`.
- Subcase IIIb (`h > w0`): `w1 = 2(h - c) - w0`. Use the Case III inequalities plus `w3 < w0 < min(h, w3 + 2c)`; they also imply `w2, w3 < 2c`.
- For every generated tuple, recompute `w8..w13`, re-check the original general inequalities `N1..N6`, `I1..I3`, `T1..T4`, and verify the expected intersection count using Corollary 4.3:
  `c_expected = w0/2 + w1/2 - min(w1, w8)`.

## Polynomial Evaluation
- Implement two evaluators:
- A direct reference evaluator derived from Figures 5–8 that walks the arc switch-by-switch and records crossings with `\alpha`, deck-level changes, and the exact integer Laurent polynomial.
- A fast evaluator following Figure 9 and the two threshold lists on page 13, jumping from one return to rail 14 to the next.
- Use the fast evaluator only after it is validated against the direct evaluator on exhaustive small bounds; the direct evaluator is the oracle.
- For throughput, do the paper’s preliminary folded computation over `F_p[t^{\pm1}] / (t^{2^m} - 1)` with `m = 7` by default. If the folded array is nonzero, reject immediately; this is one-sided safe because zero in `F_p[t^{\pm1}]` maps to zero in the quotient.
- After the preliminary pass, reject any tuple whose realized crossing count is smaller than `c_expected`; by Corollary 4.3 this identifies a proper multicurve rather than an arc.
- Every surviving candidate gets an exact integer evaluation, then coefficient reduction mod `p`. Report a witness only if the reduced exact polynomial is identically zero.

## Parallel Search / Output
- Parallelize only across independent candidate batches, preferably grouped by `(c, case)`; keep each worker pure and deterministic.
- Search in increasing `c` so the first hit is automatically minimal in intersection number.
- Checkpoint after each completed `c` with counts of generated tuples, tuples rejected by inequalities, tuples rejected by the folded filter, tuples rejected as proper multicurves, and confirmed mod-`p` zeros.

## Test Plan
- Unit-test the paper formulas for `w8..w13`, the three cases, and `c_expected`.
- Regression-test Example 5.3 exactly: its weights must produce the printed integer polynomial and the printed intersection count.
- Exhaustively compare direct vs fast evaluator on all candidates up to a small bound such as `c <= 20`, for `p in {2,3,5}`.
- Verify the mod-`p` deviation from the paper: for odd `p`, include odd-intersection candidates in the exhaustive test set and confirm they are not pruned.
- Acceptance criterion: no candidate is reported unless it survives exact integer recomputation, exact reduction mod `p`, and replay through both evaluators with identical output.

## Assumptions
- The user’s stated equivalence is assumed: kernel search for the four-strand Burau representation mod `p` is equivalent to searching for arc pairs with Burau polynomial zero after coefficient reduction mod `p`.
- The search is exhaustive only up to a user-chosen bound `N`; no heuristic pruning beyond the paper’s proven reductions is allowed.
- CPU-first means multicore process parallelism is in scope; GPU-specific design is not.
