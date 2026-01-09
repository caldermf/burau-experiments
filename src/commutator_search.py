#!/usr/bin/env python3
"""
Commutator search for kernel elements using Garside generators.
Optimized to check matrix commutation (AB = BA) rather than constructing the full
commutator braid, avoiding the need for an inverse method and expensive reduction.
"""

import argparse
import numpy as np
import time
from typing import Generator

from peyl.braid import GNF, PermTable
from peyl.jonesrep import JonesCellRep

def enumerate_garside_words(n: int, max_length: int) -> Generator[tuple[int, ...], None, None]:
    """
    Enumerate all valid Garside normal form factor sequences up to given length.
    """
    table = PermTable.create(n)
    valid_simples = [i for i in range(table.order) if i != table.id and i != table.D]
    
    for s in valid_simples:
        yield (s,)
    
    for length in range(2, max_length + 1):
        for word in _extend_words(table, valid_simples, length):
            yield word

def _extend_words(table: PermTable, valid_simples: list[int], length: int) -> Generator[tuple[int, ...], None, None]:
    if length == 1:
        for s in valid_simples:
            yield (s,)
        return
    
    for prefix in _extend_words(table, valid_simples, length - 1):
        last = prefix[-1]
        for next_simple in table.follows[last]:
            if next_simple != table.id and next_simple != table.D:
                yield prefix + (next_simple,)

def garside_word_to_gnf(n: int, factors: tuple[int, ...]) -> GNF:
    return GNF(n=n, power=0, factors=factors)

def search_commutators(
    n: int = 4,
    r: int = 1, 
    p: int = 5,
    max_x_length: int = 4,
    max_y_length: int = 4,
    max_total_length: int = None,
    verbose: bool = True
):
    """
    Search for kernel elements among commutators [x, y].
    Optimized: Checks if rho(x) * rho(y) == rho(y) * rho(x).
    """
    rep = JonesCellRep(n=n, r=r, p=p)
    
    # 1. Generate Braids
    print("Generating search space...")
    x_braids_factors = list(enumerate_garside_words(n, max_x_length))
    y_braids_factors = list(enumerate_garside_words(n, max_y_length))
    
    if verbose:
        print(f"  x-braids (len 1-{max_x_length}): {len(x_braids_factors)}")
        print(f"  y-braids (len 1-{max_y_length}): {len(y_braids_factors)}")
        print(f"  Max possible pairs: {len(x_braids_factors) * len(y_braids_factors)}")
        print("-" * 60)

    # 2. Pre-compute Matrices for Y (Massive optimization)
    # This trades memory (storing ~10^4 matrices) for CPU (avoiding 10^8 evaluations)
    print("Pre-computing matrices for Y...")
    y_data = [] # List of (factors, matrix)
    
    start_pre = time.time()
    for y_factors in y_braids_factors:
        y = garside_word_to_gnf(n, y_factors)
        My = rep.polymat_evaluate_braid(y)
        if p > 0:
            My = My % p
        y_data.append((y_factors, My))
    
    if verbose:
        print(f"Pre-computation finished in {time.time() - start_pre:.2f}s")
        print("Starting Main Search Loop...")
        print("-" * 60)

    kernel_elements = []
    checked = 0
    start_search = time.time()
    
    # 3. Main Loop
    for i, x_factors in enumerate(x_braids_factors):
        x = garside_word_to_gnf(n, x_factors)
        Mx = rep.polymat_evaluate_braid(x)
        if p > 0:
            Mx = Mx % p
            
        for y_factors, My in y_data:
            # Optional total length cut
            if max_total_length and (len(x_factors) + len(y_factors) > max_total_length):
                continue

            # Commutation Check: Mx * My == My * Mx
            # If they commute, [x, y] maps to Identity (or scalar)
            
            # Note: We compute LHS and RHS. In standard Burau, scalar matrices are central,
            # so checking commutation is sufficient to find candidates where the 
            # commutator [x,y] becomes trivial (I) or scalar.
            
            LHS = np.matmul(Mx, My)
            RHS = np.matmul(My, Mx)
            if p > 0:
                LHS %= p
                RHS %= p
                
            if np.array_equal(LHS, RHS):
                # Found a commuting pair!
                # We do NOT construct the full commutator GNF here to avoid 'inverse' error.
                # We just record the components.
                
                # Check if it's trivial (x and y commute in the braid group)
                # This is hard to check cheaply without GNF reduction, but we can usually
                # spot trivial commutators if x=y or powers. We record everything.
                
                # Try to get Artin words for display
                try:
                    x_artin = x.magma_artin_word()
                    # Reconstruct y just for the print (My is already cached)
                    y_temp = garside_word_to_gnf(n, y_factors)
                    y_artin = y_temp.magma_artin_word()
                except:
                    x_artin = "n/a"
                    y_artin = "n/a"

                kernel_elements.append({
                    "x_factors": x_factors,
                    "y_factors": y_factors,
                    "x_artin": x_artin,
                    "y_artin": y_artin
                })

                if verbose:
                    print(f"\n🎉 COMMUTING PAIR FOUND! (Likely Kernel Element)")
                    print(f"  x factors: {x_factors}")
                    print(f"  y factors: {y_factors}")
                    print(f"  x artin: {x_artin}")
                    print(f"  y artin: {y_artin}")
                    print(f"  (This implies rho([x,y]) = I)")
            
            checked += 1
            if verbose and checked % 100000 == 0:
                elapsed = time.time() - start_search
                rate = checked / elapsed
                print(f"  Checked {checked} pairs... ({rate:.0f} pairs/sec)")

    return kernel_elements

def main():
    parser = argparse.ArgumentParser(
        description="Search for kernel elements among commutators [x, y]"
    )
    parser.add_argument("--n", type=int, default=4, help="Number of strands")
    parser.add_argument("--r", type=int, default=1, help="Representation parameter")
    parser.add_argument("--p", type=int, default=5, help="Prime modulus")
    # CHANGED DEFAULTS: 8 is too large for O(N^2). 4 is reasonable for quick checks.
    parser.add_argument("--max-x", type=int, default=4, help="Max Garside length for x")
    parser.add_argument("--max-y", type=int, default=4, help="Max Garside length for y")
    parser.add_argument("--max-total", type=int, default=None, help="Max combined length")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    print(f"Commutator search for B_{args.n} kernel elements mod {args.p}")
    print(f"Strategy: Matrix Commutation Check (Mx*My == My*Mx)")
    print(f"x length: 1-{args.max_x}, y length: 1-{args.max_y}")
    if args.max_total:
        print(f"Max total length: {args.max_total}")
    print()
    
    results = search_commutators(
        n=args.n,
        r=args.r,
        p=args.p,
        max_x_length=args.max_x,
        max_y_length=args.max_y,
        max_total_length=args.max_total,
        verbose=not args.quiet
    )
    
    if results:
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Found {len(results)} pairs whose matrices commute.")
        print("To verify kernel element kappa = [x, y]:")
        print("  kappa = x * y * x^-1 * y^-1")
        for i, res in enumerate(results[:10], 1):
            print(f"\n#{i}:")
            print(f"  x: {res['x_artin']}")
            print(f"  y: {res['y_artin']}")
        if len(results) > 10:
            print(f"\n... and {len(results)-10} more.")

if __name__ == "__main__":
    main()