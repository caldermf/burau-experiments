#!/usr/bin/env python3
"""
Greedy search for nontrivial kernel elements [σ_i, g^{-1}] in B_4.

Uses peyl for all braid operations and Burau evaluation.
Scores braids by: projlen(Burau(commutator)) / garside_length(commutator).
Lower score = closer to the kernel.

Trivial commutators (g commutes with σ_i) have garside_length = 0
and get infinite score, so they're naturally filtered out.
"""

import sys
import os
import time
import argparse
import torch
import numpy as np

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


def garside_len(braid):
    """Garside complexity = |Δ power| + canonical length."""
    return abs(braid.power) + len(braid.factors)


def make_simple(idx, n=4):
    """Create a braid consisting of a single simple element."""
    return GNF(n=n, power=0, factors=(idx,))


def main():
    parser = argparse.ArgumentParser(
        description="Greedy search for kernel elements [σ_i, g^{-1}]")
    parser.add_argument('--p', type=int, default=5, help='Prime')
    parser.add_argument('--gen', type=int, default=1, choices=[1, 2, 3],
                        help='Generator index (1=σ_1, 2=σ_2, 3=σ_3)')
    parser.add_argument('--beam-size', type=int, default=500,
                        help='Number of braids to keep at each level')
    parser.add_argument('--max-level', type=int, default=60,
                        help='Maximum number of extension steps')
    parser.add_argument('--n', type=int, default=4, help='Number of strands')
    args = parser.parse_args()

    n = args.n
    p = args.p
    gen_idx = args.gen
    beam_size = args.beam_size

    print(f"Greedy commutator search for [σ_{gen_idx}, g^{{-1}}] in B_{n}")
    print(f"  p={p}, beam_size={beam_size}, max_level={args.max_level}")
    print()

    # Setup Burau representation
    rep = JonesCellRep(n=n, r=1, p=p)

    # Build σ_i and σ_i^{-1}
    sigma = GNF.artin_gens(n)[gen_idx - 1]
    sigma_inv = sigma.inv()

    # Load valid suffixes from precomputed tables
    table_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'precomputed_tables', f'tables_B4_r1_p{p}.pt')
    tables = torch.load(table_path, weights_only=True)
    valid_suffixes = tables['valid_suffixes']
    num_valid_suffixes = tables['num_valid_suffixes']
    id_idx = tables['id_index']
    delta_idx = tables['delta_index']

    # Identity gets Delta's suffix table
    valid_suffixes[id_idx] = valid_suffixes[delta_idx]
    num_valid_suffixes[id_idx] = num_valid_suffixes[delta_idx]

    num_simples = 24  # |S_4| for n=4

    def get_suffixes(last_factor_idx):
        """Get valid Garside suffixes for a given last factor."""
        ns = num_valid_suffixes[last_factor_idx].item()
        return [valid_suffixes[last_factor_idx, i].item()
                for i in range(ns)
                if 0 < valid_suffixes[last_factor_idx, i].item() < num_simples - 1]

    def score_braid(g):
        """
        Compute commutator [σ_i, g^{-1}] and return
        (score, commutator, projlen, comm_garside_len).
        """
        commutator = sigma * g.inv() * sigma_inv * g
        gl = garside_len(commutator)
        if gl == 0:
            return float('inf'), commutator, 0, 0

        matrix = rep.polymat_evaluate_braid(commutator) % p
        pl = compute_projlen(matrix)
        return pl / gl, commutator, pl, gl

    # =====================================================================
    # Level 1: initialize with all non-identity simples
    # =====================================================================
    print("Initializing level 1...")
    beam = []  # list of (score, g, projlen, comm_garside_len)
    seen = set()

    for idx in range(1, num_simples - 1):  # skip identity (0) and Δ (23)
        g = make_simple(idx, n)
        key = (g.power, g.factors)
        if key in seen:
            continue
        seen.add(key)

        score, comm, pl, gl = score_braid(g)
        if score < float('inf'):
            beam.append((score, g, pl, gl))

    beam.sort(key=lambda x: x[0])
    beam = beam[:beam_size]

    print(f"Level 1: {len(beam)} braids (filtered trivial commutators)")
    if beam:
        print(f"  Best:  score={beam[0][0]:.4f} (projlen={beam[0][2]}, comm_gl={beam[0][3]})")
        print(f"  Worst: score={beam[-1][0]:.4f} (projlen={beam[-1][2]}, comm_gl={beam[-1][3]})")

    kernel_elements = []

    # =====================================================================
    # Greedy BFS: expand best braids level by level
    # =====================================================================
    for level in range(2, args.max_level + 1):
        t0 = time.time()
        children = []
        seen_this_level = set()
        num_trivial = 0
        num_expanded = 0

        for score, g, pl, gl in beam:
            # Determine last factor for suffix lookup
            if len(g.factors) > 0:
                last = g.factors[-1]
            else:
                last = id_idx  # identity → use Delta's suffix table

            suffixes = get_suffixes(last)
            for suffix_idx in suffixes:
                g_new = g * make_simple(suffix_idx, n)

                key = (g_new.power, g_new.factors)
                if key in seen_this_level:
                    continue
                seen_this_level.add(key)
                num_expanded += 1

                new_score, comm, new_pl, new_gl = score_braid(g_new)

                if new_score == float('inf'):
                    num_trivial += 1
                    continue

                children.append((new_score, g_new, new_pl, new_gl))

                # Kernel element: projlen ≤ 1 with nontrivial commutator
                if new_pl <= 1 and new_gl > 0:
                    artin = comm.artin_word()
                    if len(artin) > 0:  # nontrivial commutator
                        kernel_elements.append((g_new, comm, new_pl, new_gl))
                        print(f"\n  *** KERNEL ELEMENT at level {level}! ***")
                        print(f"    g factors ({len(g_new.factors)}): {list(g_new.factors)}")
                        print(f"    g power: {g_new.power}")
                        print(f"    commutator length: {new_gl}")
                        print(f"    commutator artin word ({len(artin)} gens): {artin[:20]}{'...' if len(artin) > 20 else ''}")
                        print(f"    projlen: {new_pl}")

        # Sort and keep beam
        children.sort(key=lambda x: x[0])
        beam = children[:beam_size]

        elapsed = time.time() - t0

        print(f"\nLevel {level}: {num_expanded} expanded → {len(children)} nontrivial → {len(beam)} kept  ({elapsed:.1f}s)")
        if num_trivial > 0:
            print(f"  Trivial commutators skipped: {num_trivial}")
        if beam:
            scores = [b[0] for b in beam]
            print(f"  Best:   score={scores[0]:.4f} (pl={beam[0][2]}, comm_gl={beam[0][3]}, g_len={len(beam[0][1].factors)})")
            mid = len(scores) // 2
            print(f"  Median: score={scores[mid]:.4f} (pl={beam[mid][2]}, comm_gl={beam[mid][3]})")
            print(f"  Worst:  score={scores[-1]:.4f}")

            # Show score histogram
            brackets = [0, 0.5, 1.0, 2.0, 5.0, float('inf')]
            for i in range(len(brackets) - 1):
                count = sum(1 for s in scores if brackets[i] <= s < brackets[i + 1])
                if count > 0:
                    print(f"    score [{brackets[i]:.1f}, {brackets[i+1]:.1f}): {count}")
        else:
            print("  Beam is empty — no nontrivial braids left!")
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
