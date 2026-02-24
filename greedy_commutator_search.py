#!/usr/bin/env python3
"""
Diversity-preserving commutator kernel search for [σ_i, g^{-1}] in B_4.

Uses peyl for all braid operations and Burau evaluation.
Maintains projlen-bucketed reservoir to prevent beam collapse.

The key insight: pure greedy beam search collapses because projlen/garside_len
converges to 4/3 for ALL braids. By bucketing by projlen value and using
reservoir sampling within buckets, we maintain diversity across different
regions of the search space.
"""

import sys
import os
import time
import argparse
import random
import torch
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from peyl.braid import GNF
from peyl.jonesrep import JonesCellRep


def compute_projlen(matrix):
    """Projective length: max_degree - min_degree + 1 across all entries."""
    nonzero = np.any(matrix != 0, axis=(0, 1))
    if not np.any(nonzero):
        return 0
    indices = np.where(nonzero)[0]
    return int(indices[-1] - indices[0] + 1)


def compute_offdiag_weight(matrix):
    """Number of nonzero coefficients in off-diagonal entries."""
    weight = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j:
                weight += int(np.count_nonzero(matrix[i, j, :]))
    return weight


def garside_len(braid):
    """Garside complexity = |Δ power| + canonical length."""
    return abs(braid.power) + len(braid.factors)


def make_simple(idx, n=4):
    """Create a braid consisting of a single simple element."""
    return GNF(n=n, power=0, factors=(idx,))


class ProjlenReservoir:
    """Projlen-bucketed reservoir with frozen-commutator filtering.

    Maintains separate reservoirs for each projlen value, but only keeps
    braids with "growing" commutators (comm_gl >= min_comm_gl). This
    filters out braids where g = centralizer * short_piece, whose
    commutator stays fixed as g grows (dead ends for kernel search).

    Within each bucket, braids are kept sorted by off-diagonal weight
    (lower = better, closer to kernel).
    """

    def __init__(self, max_per_bucket=200, min_comm_gl=0):
        self.buckets = defaultdict(list)
        self.max_per_bucket = max_per_bucket
        self.min_comm_gl = min_comm_gl
        self.frozen_count = 0  # braids rejected for low comm_gl

    def add(self, g, projlen, comm_gl, offdiag_weight):
        if comm_gl < self.min_comm_gl:
            self.frozen_count += 1
            return
        bucket = self.buckets[projlen]
        entry = (offdiag_weight, g, comm_gl)
        if len(bucket) < self.max_per_bucket:
            bucket.append(entry)
        else:
            # Replace worst (highest offdiag_weight) if new is better
            worst_idx = max(range(len(bucket)), key=lambda i: bucket[i][0])
            if offdiag_weight < bucket[worst_idx][0]:
                bucket[worst_idx] = entry
            else:
                # Reservoir sampling for equal-weight entries
                idx = random.randint(0, len(bucket))
                if idx < len(bucket) and bucket[idx][0] == offdiag_weight:
                    bucket[idx] = entry

    def get_braids_to_expand(self, use_best=0):
        """Return braids to expand, prioritizing low projlen.

        Returns list of (g, projlen, comm_gl, offdiag_weight).
        If use_best > 0, returns at most that many.
        """
        result = []
        for pl in sorted(self.buckets.keys()):
            for odw, g, gl in self.buckets[pl]:
                result.append((g, pl, gl, odw))
                if 0 < use_best <= len(result):
                    return result
        return result

    def total_count(self):
        return sum(len(v) for v in self.buckets.values())

    def distribution(self):
        return {pl: len(braids) for pl, braids in sorted(self.buckets.items())}

    def min_projlen(self):
        return min(self.buckets.keys()) if self.buckets else None

    def bucket_stats(self, projlen):
        """Return (count, min_odw, max_odw, min_gl, max_gl) for a bucket."""
        bucket = self.buckets.get(projlen, [])
        if not bucket:
            return 0, 0, 0, 0, 0
        odws = [e[0] for e in bucket]
        gls = [e[2] for e in bucket]
        return len(bucket), min(odws), max(odws), min(gls), max(gls)


def main():
    parser = argparse.ArgumentParser(
        description="Diversity-preserving commutator kernel search [σ_i, g^{-1}]")
    parser.add_argument('--p', type=int, default=5, help='Prime')
    parser.add_argument('--gen', type=int, default=1, choices=[1, 2, 3],
                        help='Generator index (1=σ_1, 2=σ_2, 3=σ_3)')
    parser.add_argument('--max-per-bucket', type=int, default=200,
                        help='Max braids per projlen bucket')
    parser.add_argument('--use-best', type=int, default=0,
                        help='Max braids to expand per level (0=all)')
    parser.add_argument('--min-comm-gl-frac', type=float, default=0.5,
                        help='Min commutator Garside len as fraction of 3*level (filters frozen commutators)')
    parser.add_argument('--max-level', type=int, default=65,
                        help='Maximum number of extension steps')
    parser.add_argument('--n', type=int, default=4, help='Number of strands')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    n = args.n
    p = args.p
    gen_idx = args.gen
    max_per_bucket = args.max_per_bucket
    use_best = args.use_best

    min_comm_gl_frac = args.min_comm_gl_frac

    print(f"Commutator kernel search [σ_{gen_idx}, g^{{-1}}] in B_{n}")
    print(f"  p={p}, max_per_bucket={max_per_bucket}, use_best={use_best or 'all'}")
    print(f"  max_level={args.max_level}, seed={args.seed}")
    print(f"  min_comm_gl_frac={min_comm_gl_frac} (filters frozen commutators)")
    print()

    rep = JonesCellRep(n=n, r=1, p=p)
    sigma = GNF.artin_gens(n)[gen_idx - 1]
    sigma_inv = sigma.inv()

    # Load valid suffixes
    table_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'precomputed_tables', f'tables_B4_r1_p{p}.pt')
    tables = torch.load(table_path, weights_only=True)
    valid_suffixes = tables['valid_suffixes']
    num_valid_suffixes = tables['num_valid_suffixes']
    id_idx = tables['id_index']
    delta_idx = tables['delta_index']
    valid_suffixes[id_idx] = valid_suffixes[delta_idx]
    num_valid_suffixes[id_idx] = num_valid_suffixes[delta_idx]
    num_simples = 24

    def get_suffixes(last_factor_idx):
        ns = num_valid_suffixes[last_factor_idx].item()
        return [valid_suffixes[last_factor_idx, i].item()
                for i in range(ns)
                if 0 < valid_suffixes[last_factor_idx, i].item() < num_simples - 1]

    def evaluate_braid(g):
        """Returns (projlen, comm_gl, offdiag_weight, commutator) or None if trivial."""
        commutator = sigma * g.inv() * sigma_inv * g
        gl = garside_len(commutator)
        if gl == 0:
            return None
        matrix = rep.polymat_evaluate_braid(commutator) % p
        pl = compute_projlen(matrix)
        odw = compute_offdiag_weight(matrix)
        return pl, gl, odw, commutator

    # =====================================================================
    # Level 1: initialize
    # =====================================================================
    print("Initializing level 1...")
    reservoir = ProjlenReservoir(max_per_bucket=max_per_bucket)

    for idx in range(1, num_simples - 1):
        g = make_simple(idx, n)
        result = evaluate_braid(g)
        if result is not None:
            pl, gl, odw, comm = result
            reservoir.add(g, pl, gl, odw)

    dist = reservoir.distribution()
    total = reservoir.total_count()
    print(f"Level 1: {total} braids across {len(dist)} projlen values")
    print(f"  Distribution: {dist}")

    kernel_elements = []

    # =====================================================================
    # BFS with projlen-bucketed reservoir
    # =====================================================================
    for level in range(2, args.max_level + 1):
        t0 = time.time()
        # Require comm_gl to grow with level to filter frozen commutators
        min_gl = max(3, int(min_comm_gl_frac * 3 * level))
        new_reservoir = ProjlenReservoir(max_per_bucket=max_per_bucket, min_comm_gl=min_gl)
        seen = set()
        num_trivial = 0
        num_expanded = 0

        braids_to_expand = reservoir.get_braids_to_expand(use_best=use_best)

        for g, pl, gl, odw in braids_to_expand:
            last = g.factors[-1] if g.factors else id_idx
            suffixes = get_suffixes(last)
            for suffix_idx in suffixes:
                g_new = g * make_simple(suffix_idx, n)
                key = (g_new.power, g_new.factors)
                if key in seen:
                    continue
                seen.add(key)
                num_expanded += 1

                result = evaluate_braid(g_new)
                if result is None:
                    num_trivial += 1
                    continue

                new_pl, new_gl, new_odw, comm = result
                new_reservoir.add(g_new, new_pl, new_gl, new_odw)

                # Kernel check
                if new_pl <= 1 and new_gl > 0:
                    artin = comm.artin_word()
                    if len(artin) > 0:
                        kernel_elements.append((g_new, comm, new_pl, new_gl))
                        print(f"\n  *** KERNEL ELEMENT at level {level}! ***")
                        print(f"    g factors ({len(g_new.factors)}): {list(g_new.factors)}")
                        print(f"    g power: {g_new.power}")
                        print(f"    commutator Garside len: {new_gl}")
                        aw = artin[:20]
                        print(f"    artin word ({len(artin)} gens): {aw}{'...' if len(artin) > 20 else ''}")

        reservoir = new_reservoir
        elapsed = time.time() - t0

        dist = reservoir.distribution()
        total = reservoir.total_count()
        n_buckets = len(dist)

        frozen = reservoir.frozen_count
        print(f"\nLevel {level}: {num_expanded} expanded -> {total} kept in {n_buckets} buckets  ({elapsed:.1f}s)  [min_comm_gl={min_gl}]")
        if num_trivial > 0 or frozen > 0:
            print(f"  Filtered: {num_trivial} trivial, {frozen} frozen (comm_gl<{min_gl})")

        if dist:
            sorted_pls = sorted(dist.keys())
            min_pl = sorted_pls[0]
            max_pl = sorted_pls[-1]

            # Show lowest buckets with detail
            for pl in sorted_pls[:3]:
                cnt, min_odw, max_odw, min_gl, max_gl = reservoir.bucket_stats(pl)
                print(f"  pl={pl}: {cnt} braids, offdiag_wt=[{min_odw},{max_odw}], comm_gl=[{min_gl},{max_gl}]")

            if n_buckets > 6:
                print(f"  ... ({n_buckets - 6} more buckets) ...")
                for pl in sorted_pls[-3:]:
                    cnt = dist[pl]
                    print(f"  pl={pl}: {cnt} braids")
            elif n_buckets > 3:
                for pl in sorted_pls[3:]:
                    cnt = dist[pl]
                    print(f"  pl={pl}: {cnt} braids")

            # Show best braid's factor sequence
            best_pl = sorted_pls[0]
            best_braid = reservoir.buckets[best_pl][0]  # (odw, g, comm_gl)
            best_g = best_braid[1]
            factors = list(best_g.factors)
            factor_counts = defaultdict(int)
            for f in factors:
                factor_counts[f] += 1
            top_factors = sorted(factor_counts.items(), key=lambda x: -x[1])[:5]
            print(f"  Best braid: pl={best_pl}, factors({len(factors)})={factors[:10]}{'...' if len(factors)>10 else ''}")
            print(f"    Top factors: {top_factors}")

            # Summary line
            print(f"  Range: [{min_pl}, {max_pl}], expected 4L+1={4*level+1}")
        else:
            print("  Reservoir empty!")
            break

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Search complete. {len(kernel_elements)} kernel elements found.")
    for i, (g, comm, pl, gl) in enumerate(kernel_elements):
        print(f"\n  #{i+1}: g factors={list(g.factors)}, g power={g.power}")
        print(f"       commutator garside_len={gl}, projlen={pl}")
        print(f"       artin word: {comm.artin_word()[:30]}")


if __name__ == '__main__':
    main()
