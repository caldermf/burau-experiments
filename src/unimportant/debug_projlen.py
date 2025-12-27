#!/usr/bin/env python3
"""
Debug: What do the projlen=1 braids actually evaluate to?
"""

import sys
sys.path.insert(0, '/Users/com36/burau-experiments')

from peyl.braid import GNF, PermTable
from peyl.jonesrep import JonesCellRep
from peyl import polymat
import numpy as np

# The braids that were found with projlen=1
test_braids = [
    [10, 13, 10, 13, 10, 13, 10, 13],
    [13, 10, 13, 10, 13, 10, 13, 10],
    [2, 13, 10, 13, 21, 10, 13, 10],
]

n, r, p = 4, 1, 2

perm_table = PermTable.create(n)
rep = JonesCellRep(n=n, r=r, p=p)

print("="*60)
print(f"Checking braids in representation ({n-r}, {r}) mod {p}")
print("="*60)

for word in test_braids:
    print(f"\nBraid factors: {word}")
    
    # Create the braid
    braid = GNF(n=n, power=0, factors=tuple(word))
    print(f"  GNF: {braid}")
    
    # Evaluate as Matrix (symbolic)
    result_matrix = rep.evaluate(braid)
    print(f"  Symbolic evaluation:")
    for i in range(3):
        row = [result_matrix[i, j] for j in range(3)]
        print(f"    {row}")
    
    # Evaluate as polymat (numerical)
    result_polymat = rep.polymat_evaluate_braid(braid)
    if p > 0:
        result_polymat = result_polymat % p
    
    # Compute projlen using peyl
    result_proj = polymat.projectivise(result_polymat[None, ...])[0]
    peyl_projlen = polymat.projlen(result_proj[None, ...])[0]
    print(f"  Peyl projlen: {peyl_projlen}")
    
    # Show the actual polymat
    print(f"  Polymat shape: {result_polymat.shape}")
    print(f"  Nonzero entries:")
    for i in range(3):
        for j in range(3):
            coeffs = result_polymat[i, j, :]
            nonzero = np.where(coeffs != 0)[0]
            if len(nonzero) > 0:
                print(f"    [{i},{j}]: degrees {nonzero.tolist()}, coeffs {coeffs[nonzero].tolist()}")

    # Check if it's a scalar matrix
    # Scalar means: diagonal entries all equal, off-diagonal all zero
    is_scalar = True
    diag_poly = result_polymat[0, 0, :]
    
    for i in range(3):
        for j in range(3):
            if i == j:
                if not np.array_equal(result_polymat[i, j, :], diag_poly):
                    is_scalar = False
                    print(f"    Diagonal mismatch at [{i},{j}]")
            else:
                if np.any(result_polymat[i, j, :] != 0):
                    is_scalar = False
                    print(f"    Off-diagonal nonzero at [{i},{j}]")
    
    print(f"  Is scalar matrix? {is_scalar}")
    
    # What's the actual projlen of the diagonal?
    if is_scalar:
        diag_nonzero = np.where(diag_poly != 0)[0]
        print(f"  Diagonal polynomial: degrees {diag_nonzero.tolist()}, coeffs {diag_poly[diag_nonzero].tolist()}")
        print(f"  Diagonal projlen: {diag_nonzero[-1] - diag_nonzero[0] + 1 if len(diag_nonzero) > 0 else 0}")
