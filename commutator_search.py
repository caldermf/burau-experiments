#!/usr/bin/env python3
"""
Commutator search for kernel elements using Garside generators.

FIXES:
1. Implements proper Polynomial Matrix Multiplication (Convolution).
2. Uses a randomized scalar check (eval at t=2) to filter pairs instantly.
3. Pre-computes y-matrices to avoid redundant work.
"""

import argparse
import numpy as np
import time
from typing import Generator

# Try importing from your library
try:
    from peyl.braid import GNF, PermTable
    from peyl.jonesrep import JonesCellRep
except ImportError:
    print("Error: 'peyl' library not found. Make sure you are in the correct environment.")
    exit(1)

def enumerate_garside_words(n: int, max_length: int) -> Generator[tuple[int, ...], None, None]:
    """Enumerate all valid Garside normal form factor sequences."""
    table = PermTable.create(n)
    valid_simples = [i for i in range(table.order) if i != table.id and i != table.D]
    
    for s in valid_simples:
        yield (s,)
    
    for length in range(2, max_length + 1):
        for word in _extend_words(table, valid_simples, length):
            yield word

def _extend_words(table: PermTable, valid_simples: list[int], length: int):
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

# =============================================================================
#  MATH HELPERS
# =============================================================================

def eval_polymat_at_t(matrix: np.ndarray, t: int, p: int) -> np.ndarray:
    """
    Evaluate a polynomial matrix M(v) at v = t over F_p.
    Input: (dim, dim, degree) array.
    Output: (dim, dim) scalar array.
    """
    dim = matrix.shape[0]
    degree = matrix.shape[2]
    
    # Efficient Horner's method or dot product
    # res[i,j] = sum( coeff[k] * t^k )
    
    # Precompute powers of t: [1, t, t^2, ...]
    powers = np.ones(degree, dtype=int)
    curr = 1
    for k in range(1, degree):
        curr = (curr * t) % p
        powers[k] = curr
        
    # Broadcast multiplication: Sum over last axis
    # (dim, dim, deg) * (deg,) -> (dim, dim)
    res = np.sum(matrix * powers, axis=2)
    return res % p

def polymat_mul(A: np.ndarray, B: np.ndarray, p: int) -> np.ndarray:
    """
    Multiply two polynomial matrices A(v) and B(v) via convolution.
    Result C(v) = A(v) * B(v).
    """
    dim = A.shape[0]
    degA = A.shape[2]
    degB = B.shape[2]
    new_deg = degA + degB - 1
    
    C = np.zeros((dim, dim, new_deg), dtype=int)
    
    for i in range(dim):
        for j in range(dim):
            poly_sum = np.zeros(new_deg, dtype=int)
            for k in range(dim):
                # Convolve coefficient vectors
                term = np.convolve(A[i, k, :], B[k, j, :])
                # Convolve size is exactly new_deg
                poly_sum += term
            C[i, j, :] = poly_sum
            
    if p > 0:
        C %= p
    return C

def search_commutators(
    n: int = 4,
    r: int = 1, 
    p: int = 5,
    max_x_length: int = 4,
    max_y_length: int = 4,
    max_total_length: int = None,
    verbose: bool = True
):
    rep = JonesCellRep(n=n, r=r, p=p)
    
    print("Generating search space...")
    x_braids_factors = list(enumerate_garside_words(n, max_x_length))
    y_braids_factors = list(enumerate_garside_words(n, max_y_length))
    
    if verbose:
        print(f"  x-braids (len 1-{max_x_length}): {len(x_braids_factors)}")
        print(f"  y-braids (len 1-{max_y_length}): {len(y_braids_factors)}")
        print(f"  Total pairs: {len(x_braids_factors) * len(y_braids_factors)}")
        print("-" * 60)

    # 1. PRE-COMPUTATION
    # Store both the Full Polynomial Matrix and the Scalar Evaluation at t=2
    # The scalar eval is a "Hash" that lets us reject non-commuting pairs in O(1).
    print("Pre-computing matrices for Y...")
    y_data = [] 
    test_t = 2 # A generator or random unit in F_p
    
    start_pre = time.time()
    for y_factors in y_braids_factors:
        y = garside_word_to_gnf(n, y_factors)
        My = rep.polymat_evaluate_braid(y)
        if p > 0: My %= p
        
        # Cache scalar version for fast filtering
        My_scalar = eval_polymat_at_t(My, test_t, p)
        y_data.append((y_factors, My, My_scalar))
    
    if verbose:
        print(f"Pre-computation finished in {time.time() - start_pre:.2f}s")
        print("Starting Main Search Loop...")
        print("-" * 60)

    kernel_elements = []
    checked = 0
    start_search = time.time()
    
    # 2. MAIN LOOP
    for x_factors in x_braids_factors:
        x = garside_word_to_gnf(n, x_factors)
        Mx = rep.polymat_evaluate_braid(x)
        if p > 0: Mx %= p
        
        # Scalar version of X for fast filtering
        Mx_scalar = eval_polymat_at_t(Mx, test_t, p)
            
        for y_factors, My, My_scalar in y_data:
            if max_total_length and (len(x_factors) + len(y_factors) > max_total_length):
                continue

            # --- FILTER 1: Scalar Matrix Check (Fastest) ---
            # If matrices don't commute at t=2, they fail.
            LHS_scalar = (Mx_scalar @ My_scalar) % p
            RHS_scalar = (My_scalar @ Mx_scalar) % p
            
            if not np.array_equal(LHS_scalar, RHS_scalar):
                checked += 1
                continue
                
            # --- FILTER 2: Full Polynomial Check (Slower) ---
            LHS = polymat_mul(Mx, My, p)
            RHS = polymat_mul(My, Mx, p)
            
            if np.array_equal(LHS, RHS):
                # Matrices commute! Now checks if it's a "real" discovery or just trivial.
                
                # --- FILTER 3: Braid Group Check (Crucial) ---
                # We need to ensure xy != yx in the Braid Group.
                # Construct the GNF products.
                y = garside_word_to_gnf(n, y_factors)
                xy = x * y
                yx = y * x
                
                # Compare their normal forms. 
                # If they are equal, it's a trivial commutator (boring).
                if xy.factors == yx.factors and xy.power == yx.power:
                    # It's a trivial commutator. Skip it silently.
                    checked += 1
                    continue

                # --- SUCCESS: NON-TRIVIAL KERNEL ELEMENT ---
                try:
                    x_artin = x.magma_artin_word()
                    y_artin = y.magma_artin_word()
                except:
                    x_artin = str(x_factors)
                    y_artin = str(y_factors)

                res_obj = {
                    "x_factors": x_factors, "y_factors": y_factors,
                    "x_artin": x_artin, "y_artin": y_artin
                }
                kernel_elements.append(res_obj)

                if verbose:
                    print(f"\n🚨 GENUINE KERNEL ELEMENT FOUND! 🚨")
                    print(f"  The matrices commute, but the braids DO NOT.")
                    print(f"  x: {x_artin}")
                    print(f"  y: {y_artin}")
            
            checked += 1
            
            checked += 1
            if verbose and checked % 500000 == 0:
                elapsed = time.time() - start_search
                rate = checked / elapsed
                print(f"  Checked {checked} pairs... ({rate:.0f} pairs/sec)")

    return kernel_elements

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--max-x", type=int, default=4)
    parser.add_argument("--max-y", type=int, default=4)
    parser.add_argument("--max-total", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    
    print(f"Commutator search for B_{args.n} kernel elements mod {args.p}")
    print(f"Strategy: Randomized Scalar Filter (t=2) -> Full Polynomial Convolution")
    print(f"x length: 1-{args.max_x}, y length: 1-{args.max_y}")
    
    results = search_commutators(
        n=args.n, r=args.r, p=args.p,
        max_x_length=args.max_x, max_y_length=args.max_y,
        max_total_length=args.max_total, verbose=not args.quiet
    )

if __name__ == "__main__":
    main()