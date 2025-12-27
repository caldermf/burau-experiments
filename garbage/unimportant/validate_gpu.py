#!/usr/bin/env python3
"""
Validation Suite: Compare GPU implementation against peyl (the oracle)

This script verifies that:
1. The precomputed Burau matrices match peyl's evaluation
2. Matrix multiplication gives the same results as peyl
3. Projlen computation matches peyl's polymat.projlen
4. Multi-step braid products match peyl end-to-end

Run this BEFORE trusting any results from the GPU search.
"""

import sys
import torch
import numpy as np

# Add peyl to path if needed
sys.path.insert(0, '/Users/com36/burau-experiments')

from peyl.braid import GNF, PermTable
from peyl.jonesrep import JonesCellRep
from peyl import polymat

# Import GPU code
from new_braid_search import (
    Config, 
    load_tables_from_file, 
    poly_matmul_batch,
    compute_projlen_batch
)


def test_1_simple_matrices():
    """
    Test 1: Verify that each simple's Burau matrix matches peyl.
    """
    print("="*60)
    print("TEST 1: Simple Burau Matrices")
    print("="*60)
    
    n, r, p = 4, 1, 5
    
    # Load GPU tables
    config = Config(max_length=10, prime=p)
    table_path = "/Users/com36/burau-experiments/precomputed_tables/tables_B4_r1_p5.pt"
    gpu_burau, _, _ = load_tables_from_file(config, table_path)
    
    # Get peyl's version
    perm_table = PermTable.create(n)
    rep = JonesCellRep(n=n, r=r, p=p)
    
    center = config.degree_window // 2
    errors = []
    
    for s in range(24):
        # GPU version: extract nonzero part
        gpu_mat = gpu_burau[s].numpy()  # (3, 3, D)
        
        # Peyl version
        peyl_mat = rep._polymat_braid_factor(GNF, s)
        peyl_mat = polymat.trim(peyl_mat)
        
        # Find the nonzero range in GPU matrix
        nonzero_mask = np.abs(gpu_mat).sum(axis=(0, 1)) > 0
        if not np.any(nonzero_mask):
            # Zero matrix - check peyl is also zero
            if peyl_mat.size > 0 and np.any(peyl_mat != 0):
                errors.append(f"Simple {s}: GPU is zero but peyl is not")
            continue
        
        gpu_start = np.argmax(nonzero_mask)
        gpu_end = len(nonzero_mask) - np.argmax(nonzero_mask[::-1])
        gpu_coeffs = gpu_mat[:, :, gpu_start:gpu_end]
        
        # Peyl version - trim and shift
        val, peyl_shifted = polymat.trim_left(peyl_mat)
        peyl_coeffs = peyl_shifted % p  # Apply mod p
        
        # Compare
        if gpu_coeffs.shape != peyl_coeffs.shape:
            errors.append(f"Simple {s}: shape mismatch GPU {gpu_coeffs.shape} vs peyl {peyl_coeffs.shape}")
        elif not np.allclose(gpu_coeffs, peyl_coeffs):
            errors.append(f"Simple {s}: coefficient mismatch")
            print(f"  GPU:\n{gpu_coeffs}")
            print(f"  Peyl:\n{peyl_coeffs}")
        else:
            print(f"  Simple {s}: ✓")
    
    if errors:
        print(f"\n❌ FAILED: {len(errors)} errors")
        for e in errors:
            print(f"  {e}")
        return False
    else:
        print(f"\n✓ PASSED: All 24 simple matrices match peyl")
        return True


def test_2_single_multiplication():
    """
    Test 2: Verify that A * B computed by GPU matches peyl.
    """
    print("\n" + "="*60)
    print("TEST 2: Single Matrix Multiplication")
    print("="*60)
    
    n, r, p = 4, 1, 5
    
    config = Config(max_length=10, prime=p)
    table_path = "/Users/com36/burau-experiments/precomputed_tables/tables_B4_r1_p5.pt"
    gpu_burau, _, _ = load_tables_from_file(config, table_path)
    
    perm_table = PermTable.create(n)
    rep = JonesCellRep(n=n, r=r, p=p)
    
    # Test several pairs of simples
    test_pairs = [(1, 2), (3, 5), (7, 11), (0, 1), (1, 0), (10, 15), (22, 21)]
    errors = []
    
    for s1, s2 in test_pairs:
        # GPU multiplication
        A = gpu_burau[s1:s1+1]  # (1, 3, 3, D)
        B = gpu_burau[s2:s2+1]  # (1, 3, 3, D)
        gpu_product = poly_matmul_batch(A, B, p)  # (1, 3, 3, 2D-1)
        gpu_product = gpu_product[0].numpy()  # (3, 3, 2D-1)
        
        # Peyl multiplication
        # Create a braid with factors [s1, s2] and evaluate
        # But we need to be careful - s1 and s2 might not form a valid normal form
        # So let's just manually multiply the matrices
        
        mat1 = rep._polymat_braid_factor(GNF, s1)
        mat2 = rep._polymat_braid_factor(GNF, s2)
        peyl_product = polymat.mul(mat1, mat2)
        if p > 0:
            peyl_product = peyl_product % p
        peyl_product = polymat.projectivise(peyl_product)
        
        # Extract nonzero parts and compare
        # GPU
        gpu_nonzero = np.abs(gpu_product).sum(axis=(0, 1)) > 0
        if np.any(gpu_nonzero):
            gpu_start = np.argmax(gpu_nonzero)
            gpu_end = len(gpu_nonzero) - np.argmax(gpu_nonzero[::-1])
            gpu_coeffs = gpu_product[:, :, gpu_start:gpu_end]
        else:
            gpu_coeffs = np.zeros((3, 3, 0))
        
        # Peyl
        peyl_coeffs = polymat.trim(peyl_product)
        val, peyl_coeffs = polymat.trim_left(peyl_coeffs)
        
        # Compare (mod p again to be safe)
        gpu_coeffs = gpu_coeffs % p
        peyl_coeffs = peyl_coeffs % p
        
        if gpu_coeffs.shape != peyl_coeffs.shape:
            errors.append(f"({s1}, {s2}): shape mismatch GPU {gpu_coeffs.shape} vs peyl {peyl_coeffs.shape}")
        elif not np.allclose(gpu_coeffs, peyl_coeffs):
            errors.append(f"({s1}, {s2}): coefficient mismatch")
            print(f"  GPU:\n{gpu_coeffs[:,:,0] if gpu_coeffs.shape[-1] > 0 else 'empty'}")
            print(f"  Peyl:\n{peyl_coeffs[:,:,0] if peyl_coeffs.shape[-1] > 0 else 'empty'}")
        else:
            print(f"  ({s1}, {s2}): ✓")
    
    if errors:
        print(f"\n❌ FAILED: {len(errors)} errors")
        for e in errors:
            print(f"  {e}")
        return False
    else:
        print(f"\n✓ PASSED: All multiplications match peyl")
        return True


def test_3_projlen():
    """
    Test 3: Verify projlen computation matches peyl.
    """
    print("\n" + "="*60)
    print("TEST 3: Projlen Computation")
    print("="*60)
    
    n, r, p = 4, 1, 5
    
    config = Config(max_length=10, prime=p)
    table_path = "/Users/com36/burau-experiments/precomputed_tables/tables_B4_r1_p5.pt"
    gpu_burau, valid_suffixes, num_valid_suffixes = load_tables_from_file(config, table_path)
    
    rep = JonesCellRep(n=n, r=r, p=p)
    perm_table = PermTable.create(n)
    
    errors = []
    
    # Test projlen of individual simples
    print("  Testing individual simples...")
    for s in range(24):
        gpu_mat = gpu_burau[s:s+1]  # (1, 3, 3, D)
        gpu_projlen = compute_projlen_batch(gpu_mat)[0].item()
        
        peyl_mat = rep._polymat_braid_factor(GNF, s)
        peyl_mat = polymat.trim(peyl_mat)
        if peyl_mat.shape[-1] == 0:
            peyl_projlen = 0
        else:
            peyl_projlen = polymat.projlen(peyl_mat[None, ...])[0]
        
        if gpu_projlen != peyl_projlen:
            errors.append(f"Simple {s}: GPU projlen={gpu_projlen}, peyl projlen={peyl_projlen}")
        
    # Test projlen of some products
    print("  Testing products...")
    test_braids = [
        (1, 2),
        (3, 5, 7),
        (1, 4, 18),  # This is a valid normal form sequence for simple 1
        (10, 11, 12),
    ]
    
    for factors in test_braids:
        # GPU: multiply step by step
        gpu_result = gpu_burau[factors[0]:factors[0]+1]
        for f in factors[1:]:
            next_mat = gpu_burau[f:f+1]
            gpu_result = poly_matmul_batch(gpu_result, next_mat, p)
            # Trim to reasonable size
            D = config.degree_window
            if gpu_result.shape[-1] > D:
                center = gpu_result.shape[-1] // 2
                start = center - D // 2
                gpu_result = gpu_result[..., start:start+D]
        
        gpu_projlen = compute_projlen_batch(gpu_result)[0].item()
        
        # Peyl: multiply the matrices
        peyl_result = rep._polymat_braid_factor(GNF, factors[0])
        for f in factors[1:]:
            next_mat = rep._polymat_braid_factor(GNF, f)
            peyl_result = polymat.mul(peyl_result, next_mat)
            if p > 0:
                peyl_result = peyl_result % p
        
        peyl_result = polymat.trim(peyl_result)
        if peyl_result.shape[-1] == 0:
            peyl_projlen = 0
        else:
            peyl_projlen = polymat.projlen(peyl_result[None, ...])[0]
        
        if gpu_projlen != peyl_projlen:
            errors.append(f"Braid {factors}: GPU projlen={gpu_projlen}, peyl projlen={peyl_projlen}")
        else:
            print(f"    {factors}: projlen={gpu_projlen} ✓")
    
    if errors:
        print(f"\n❌ FAILED: {len(errors)} errors")
        for e in errors:
            print(f"  {e}")
        return False
    else:
        print(f"\n✓ PASSED: All projlen computations match peyl")
        return True


def test_4_full_braid_evaluation():
    """
    Test 4: Evaluate actual GNF braids and compare end-to-end.
    """
    print("\n" + "="*60)
    print("TEST 4: Full Braid Evaluation (End-to-End)")
    print("="*60)
    
    n, r, p = 4, 1, 5
    
    config = Config(max_length=20, prime=p)
    table_path = "/Users/com36/burau-experiments/precomputed_tables/tables_B4_r1_p5.pt"
    gpu_burau, _, _ = load_tables_from_file(config, table_path)
    
    rep = JonesCellRep(n=n, r=r, p=p)
    perm_table = PermTable.create(n)
    
    # Generate some actual valid braids using peyl
    print("  Generating test braids from peyl...")
    test_braids = []
    
    # Length 1-5 braids
    for length in range(1, 6):
        for braid in list(GNF.all_of_length(n, length))[:3]:  # Take first 3 of each length
            test_braids.append(braid)
    
    print(f"  Testing {len(test_braids)} braids...")
    errors = []
    
    for braid in test_braids:
        factors = braid.factors
        
        # GPU evaluation
        center = config.degree_window // 2
        gpu_result = torch.zeros(1, 3, 3, config.degree_window, dtype=torch.long)
        for i in range(3):
            gpu_result[0, i, i, center] = 1  # Start with identity
        
        for f in factors:
            next_mat = gpu_burau[f:f+1]
            gpu_result = poly_matmul_batch(gpu_result, next_mat, p)
            # Recenter
            D = config.degree_window
            if gpu_result.shape[-1] > D:
                current_center = gpu_result.shape[-1] // 2
                start = current_center - D // 2
                gpu_result = gpu_result[..., start:start+D]
        
        gpu_projlen = compute_projlen_batch(gpu_result)[0].item()
        
        # Peyl evaluation
        peyl_result = rep.polymat_evaluate_braid(braid)
        if p > 0:
            peyl_result = peyl_result % p
        peyl_result = polymat.projectivise(peyl_result[None, ...])[0]
        peyl_projlen = polymat.projlen(peyl_result[None, ...])[0]
        
        if gpu_projlen != peyl_projlen:
            errors.append(f"Braid {factors}: GPU={gpu_projlen}, peyl={peyl_projlen}")
        
    print(f"  Tested {len(test_braids)} braids")
    
    if errors:
        print(f"\n❌ FAILED: {len(errors)} errors")
        for e in errors[:10]:  # Show first 10
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        return False
    else:
        print(f"\n✓ PASSED: All {len(test_braids)} braid evaluations match peyl")
        return True


def test_5_suffix_table():
    """
    Test 5: Verify the suffix/follows table matches peyl.
    """
    print("\n" + "="*60)
    print("TEST 5: Suffix Table (Garside Normal Form)")
    print("="*60)
    
    n = 4
    
    config = Config(max_length=10, prime=5)
    table_path = "/Users/com36/burau-experiments/precomputed_tables/tables_B4_r1_p5.pt"
    _, gpu_suffixes, gpu_num_suffixes = load_tables_from_file(config, table_path)
    
    perm_table = PermTable.create(n)
    
    errors = []
    
    for s in range(24):
        # GPU version
        n_gpu = gpu_num_suffixes[s].item()
        gpu_set = set(gpu_suffixes[s, :n_gpu].tolist())
        
        # Peyl version (note: identity's suffixes were copied from Delta)
        if s == perm_table.id:
            peyl_set = set(perm_table.follows[perm_table.D])
        else:
            peyl_set = set(perm_table.follows[s])
        
        if gpu_set != peyl_set:
            errors.append(f"Simple {s}: GPU {gpu_set} vs peyl {peyl_set}")
        else:
            print(f"  Simple {s}: {n_gpu} suffixes ✓")
    
    if errors:
        print(f"\n❌ FAILED: {len(errors)} errors")
        for e in errors:
            print(f"  {e}")
        return False
    else:
        print(f"\n✓ PASSED: All suffix tables match peyl")
        return True


def run_all_tests():
    """Run the complete validation suite."""
    print("\n" + "="*60)
    print("VALIDATION SUITE: GPU Code vs Peyl Oracle")
    print("="*60)
    print("This will verify that the GPU implementation gives")
    print("identical results to the original peyl library.")
    print("="*60 + "\n")
    
    results = {}
    
    results["simple_matrices"] = test_1_simple_matrices()
    results["multiplication"] = test_2_single_multiplication()
    results["projlen"] = test_3_projlen()
    results["full_braid"] = test_4_full_braid_evaluation()
    results["suffix_table"] = test_5_suffix_table()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED! The GPU code matches peyl exactly. 🎉")
        print("   You can trust the GPU search results.")
    else:
        print("⚠️  SOME TESTS FAILED! Do not trust GPU results until fixed.")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)