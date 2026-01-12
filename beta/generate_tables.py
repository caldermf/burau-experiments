#!/usr/bin/env python3
"""
Script to generate precomputed multiplication tables and Garside suffix tables
from the peyl library, for use in GPU-accelerated braid group computations.

This generates:
1. simple_burau: (n!, dim, dim, degree_window) - Representation matrices for all simples
2. valid_suffixes: (n!, max_suffixes) - Which simples can follow each simple
3. num_valid_suffixes: (n!,) - Count of valid suffixes for each simple

Usage:
    python generate_tables.py --n 4 --r 1 --p 5
    python generate_tables.py --n 5 --r 1 --p 7 --degree-window 128
    python generate_tables.py --n 6 --r 2 --p 0  # char 0

The tables will be saved as .pt files (PyTorch tensors).
"""

import argparse
import math
import numpy as np
import torch
from pathlib import Path

# Import peyl modules
from peyl.braid import GNF, PermTable
from peyl.jonesrep import JonesCellRep
from peyl import polymat


def generate_tables(
    n: int = 4,           # Number of strands (B_n has n! simples)
    r: int = 1,           # Partition parameter: (n-r, r)
    p: int = 2,           # Prime for mod p reduction (0 for no reduction)
    degree_window: int = 64,  # Size of the degree window
    output_dir: str = ".",
):
    """
    Generate precomputed tables for GPU acceleration.
    
    Parameters:
        n: Number of strands in braid group B_n
        r: Partition parameter for the two-row representation (n-r, r)
        p: Prime for modular reduction (0 for integer coefficients)
        degree_window: Size of degree window for polynomial coefficients
        output_dir: Directory to save output files
    """
    # Validate parameters
    assert n >= 2, f"Need n >= 2, got {n}"
    assert r >= 1, f"Need r >= 1, got {r}"
    assert n - 2 * r >= 0, f"Need n >= 2r for partition (n-r, r), got n={n}, r={r}"
    
    print(f"{'='*60}")
    print(f"Generating tables for B_{n}")
    print(f"  Representation: ({n-r}, {r})")
    print(f"  Coefficients: {'mod ' + str(p) if p > 0 else 'integers (char 0)'}")
    print(f"  Degree window: {degree_window}")
    print(f"{'='*60}")
    
    # Create the permutation table and representation
    perm_table = PermTable.create(n)
    rep = JonesCellRep(n=n, r=r, p=p)
    
    # Basic info
    num_simples = perm_table.order  # n!
    dim = rep.dimension()           # Dimension of representation
    center = degree_window // 2     # Where degree 0 sits in the window
    
    print(f"\nParameters:")
    print(f"  Number of simples (n!): {num_simples}")
    print(f"  Representation dimension: {dim}")
    print(f"  Identity index: {perm_table.id}")
    print(f"  Delta index: {perm_table.D}")
    print(f"  Center offset: {center}")
    
    # Estimate memory
    matrix_mem = num_simples * dim * dim * degree_window * 8 / 1e9  # int64
    print(f"  Estimated table size: {matrix_mem:.2f} GB")
    
    # =========================================================================
    # 1. Generate simple_burau: Representation matrices for all simples
    # =========================================================================
    print(f"\n1. Generating representation matrices for all {num_simples} simples...")
    
    simple_burau = torch.zeros(num_simples, dim, dim, degree_window, dtype=torch.long)
    
    max_degree_seen = 0
    min_degree_seen = 0
    
    for s in range(num_simples):
        if s % max(1, num_simples // 10) == 0:
            print(f"   Processing simple {s}/{num_simples}...")
        
        # Get the polymat (numpy array) for this simple
        # Shape: (dim, dim, num_degrees)
        mat = rep._polymat_braid_factor(GNF, s)
        
        # mat has shape (dim, dim, L) where L is the number of nonzero degrees
        # The valuation (lowest degree) can be extracted
        mat_trimmed = polymat.trim(mat)
        val, mat_shifted = polymat.trim_left(mat_trimmed)
        
        # Track degree range
        num_coeffs = mat_shifted.shape[-1]
        if num_coeffs > 0:
            min_degree_seen = min(min_degree_seen, val)
            max_degree_seen = max(max_degree_seen, val + num_coeffs - 1)
        
        # Place coefficients starting at (center + val)
        start_idx = center + val
        end_idx = start_idx + num_coeffs
        
        if start_idx < 0 or end_idx > degree_window:
            print(f"  WARNING: Simple {s} has degrees [{val}, {val + num_coeffs - 1}] "
                  f"which doesn't fit in window with center={center}")
            # Clip to window bounds
            src_start = max(0, -start_idx)
            src_end = min(num_coeffs, degree_window - start_idx)
            dst_start = max(0, start_idx)
            dst_end = min(degree_window, end_idx)
            simple_burau[s, :, :, dst_start:dst_end] = torch.from_numpy(
                mat_shifted[:, :, src_start:src_end].astype(np.int64)
            )
        else:
            simple_burau[s, :, :, start_idx:end_idx] = torch.from_numpy(
                mat_shifted.astype(np.int64)
            )
        
        if p > 0:
            simple_burau[s] = simple_burau[s] % p
    
    print(f"  Done! Shape: {tuple(simple_burau.shape)}")
    print(f"  Degree range seen: [{min_degree_seen}, {max_degree_seen}]")
    
    # Verify identity is correct
    id_mat = simple_burau[perm_table.id]
    expected_id = torch.zeros(dim, dim, degree_window, dtype=torch.long)
    for i in range(dim):
        expected_id[i, i, center] = 1
    assert torch.equal(id_mat, expected_id), "Identity matrix verification failed!"
    print("  ✓ Identity matrix verified")
    
    # =========================================================================
    # 2. Generate valid_suffixes and num_valid_suffixes
    # =========================================================================
    print(f"\n2. Generating Garside suffix tables...")
    
    # The 'follows' table in perm_table tells us which simples can follow each simple
    # Note: follows[s] gives indices of simples that can follow s in normal form
    # It excludes identity and Delta (they're not valid canonical factors)
    
    max_suffixes = max(len(perm_table.follows[s]) for s in range(num_simples))
    print(f"  Maximum number of valid suffixes: {max_suffixes}")
    
    valid_suffixes = torch.full((num_simples, max_suffixes), -1, dtype=torch.int32)
    num_valid_suffixes = torch.zeros(num_simples, dtype=torch.int32)
    
    for s in range(num_simples):
        follows_s = perm_table.follows[s]
        num_valid_suffixes[s] = len(follows_s)
        for j, suffix_idx in enumerate(follows_s):
            valid_suffixes[s, j] = suffix_idx
    
    print(f"  valid_suffixes shape: {tuple(valid_suffixes.shape)}")
    print(f"  num_valid_suffixes shape: {tuple(num_valid_suffixes.shape)}")
    
    # Show some statistics
    print(f"\n  Suffix counts per simple:")
    print(f"    Identity (idx {perm_table.id}): {num_valid_suffixes[perm_table.id].item()} valid suffixes")
    print(f"    Delta (idx {perm_table.D}): {num_valid_suffixes[perm_table.D].item()} valid suffixes")
    print(f"    Min: {num_valid_suffixes.min().item()}, Max: {num_valid_suffixes.max().item()}, "
          f"Mean: {num_valid_suffixes.float().mean().item():.1f}")
    
    # =========================================================================
    # 3. Save tables
    # =========================================================================
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"tables_B{n}_r{r}_p{p}.pt"
    save_path = output_path / filename
    
    tables = {
        'n': n,
        'r': r,
        'p': p,
        'dim': dim,
        'num_simples': num_simples,
        'degree_window': degree_window,
        'center': center,
        'id_index': perm_table.id,
        'delta_index': perm_table.D,
        'simple_burau': simple_burau,
        'valid_suffixes': valid_suffixes,
        'num_valid_suffixes': num_valid_suffixes,
    }
    
    torch.save(tables, save_path)
    file_size = save_path.stat().st_size / 1e6
    print(f"\n3. Saved tables to: {save_path}")
    print(f"   File size: {file_size:.1f} MB")
    
    return tables


def print_simple_info(tables: dict, simple_idx: int):
    """Debug helper: print info about a specific simple."""
    dim = tables['dim']
    center = tables['center']
    
    print(f"\nSimple {simple_idx}:")
    print(f"  Matrix (nonzero degrees):")
    mat = tables['simple_burau'][simple_idx]
    nonzero_degs = torch.where(mat.abs().sum(dim=(0, 1)) > 0)[0]
    
    if len(nonzero_degs) > 0:
        for d in nonzero_degs[:5]:  # Show first 5 nonzero degrees
            actual_deg = d.item() - center
            print(f"    Degree {actual_deg}:")
            print(mat[:, :, d])
        if len(nonzero_degs) > 5:
            print(f"    ... and {len(nonzero_degs) - 5} more nonzero degrees")
    else:
        print("    (zero matrix)")
    
    print(f"  Valid suffixes ({tables['num_valid_suffixes'][simple_idx].item()}):")
    n_valid = tables['num_valid_suffixes'][simple_idx].item()
    suffixes = tables['valid_suffixes'][simple_idx, :min(n_valid, 10)].tolist()
    if n_valid > 10:
        print(f"    {suffixes} ... ({n_valid - 10} more)")
    else:
        print(f"    {suffixes}")


def verify_multiplication(tables: dict, n: int, r: int, p: int, num_tests: int = 10):
    """Verify that the table entries are correct by comparing with peyl."""
    from peyl.braid import GNF, PermTable
    from peyl.jonesrep import JonesCellRep
    from peyl import polymat
    
    perm_table = PermTable.create(n)
    rep = JonesCellRep(n=n, r=r, p=p)
    
    print(f"\nVerification: comparing {num_tests} table entries with peyl evaluation...")
    
    # Test specific indices plus some random ones
    num_simples = tables['num_simples']
    test_indices = [0, 1, perm_table.id, perm_table.D]
    
    # Add random samples
    import random
    random_samples = random.sample(range(num_simples), min(num_tests - len(test_indices), num_simples))
    test_indices.extend(random_samples)
    test_indices = list(set(test_indices))[:num_tests]
    
    center = tables['center']
    
    passed = 0
    failed = 0
    
    for s in test_indices:
        # Get from table
        table_mat = tables['simple_burau'][s].numpy()
        
        # Get from peyl
        peyl_mat = rep._polymat_braid_factor(GNF, s)
        peyl_mat = polymat.trim(peyl_mat)
        
        # Compare by reconstructing
        # Find nonzero range in table
        nonzero_mask = np.abs(table_mat).sum(axis=(0, 1)) > 0
        if not np.any(nonzero_mask):
            # Zero matrix
            if peyl_mat.shape[-1] == 0 or np.all(peyl_mat == 0):
                print(f"  Simple {s}: ✓ (zero matrix)")
                passed += 1
            else:
                print(f"  Simple {s}: ✗ (table is zero, peyl is not)")
                failed += 1
            continue
        
        table_start = np.argmax(nonzero_mask)
        table_end = len(nonzero_mask) - np.argmax(nonzero_mask[::-1])
        
        # The table stores degree d at index center + d
        table_coeffs = table_mat[:, :, table_start:table_end]
        
        # peyl_mat starts at some valuation
        val, peyl_shifted = polymat.trim_left(peyl_mat)
        
        # They should match
        if table_coeffs.shape == peyl_shifted.shape and np.allclose(table_coeffs, peyl_shifted):
            print(f"  Simple {s}: ✓")
            passed += 1
        else:
            print(f"  Simple {s}: ✗")
            print(f"    Table shape: {table_coeffs.shape}, Peyl shape: {peyl_shifted.shape}")
            failed += 1
    
    print(f"\nVerification complete: {passed} passed, {failed} failed")
    return failed == 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate precomputed tables for GPU braid search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --n 4 --r 1 --p 5           # B_4, Burau (3,1), mod 5
  %(prog)s --n 5 --r 1 --p 7           # B_5, Burau (4,1), mod 7  
  %(prog)s --n 6 --r 2 --p 5           # B_6, (4,2) rep, mod 5
  %(prog)s --n 10 --r 2 --p 0          # B_10, (8,2) rep, char 0

Notes:
  - Representation (n-r, r) has dimension C(n,r) - C(n,r-1)
  - B_n has n! simples, so B_10 has 3,628,800 simples!
  - p=0 means integer coefficients (characteristic 0)
  - Larger degree windows are needed for longer braids
        """
    )
    
    parser.add_argument("--n", type=int, required=True,
                        help="Number of strands in braid group B_n")
    
    parser.add_argument("--r", type=int, required=True,
                        help="Partition parameter for (n-r, r) representation")
    
    parser.add_argument("--p", type=int, required=True,
                        help="Prime for mod p reduction (0 for integers)")
    
    parser.add_argument("--degree-window", "-d", type=int, default=64,
                        help="Size of degree window (default: 64)")
    
    parser.add_argument("--output-dir", "-o", type=str, default="precomputed_tables",
                        help="Output directory (default: precomputed_tables)")
    
    parser.add_argument("--verify", action="store_true",
                        help="Verify tables against peyl after generation")
    
    parser.add_argument("--info", type=int, nargs="*", default=None,
                        help="Print info about specific simple indices")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Warn about large computations
    num_simples = math.factorial(args.n)
    if num_simples > 100000:
        print(f"WARNING: B_{args.n} has {num_simples:,} simples. This will take a while!")
        print("Press Ctrl+C within 5 seconds to cancel...")
        import time
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nCancelled.")
            exit(0)
    
    tables = generate_tables(
        n=args.n,
        r=args.r,
        p=args.p,
        degree_window=args.degree_window,
        output_dir=args.output_dir
    )
    
    # Print info about specific simples
    if args.info is not None:
        if len(args.info) == 0:
            # Default: show identity and one other
            args.info = [tables['id_index'], 1]
        for idx in args.info:
            if 0 <= idx < tables['num_simples']:
                print_simple_info(tables, idx)
            else:
                print(f"Index {idx} out of range [0, {tables['num_simples']})")
    
    # Verify if requested
    if args.verify:
        verify_multiplication(tables, n=args.n, r=args.r, p=args.p)
    
    print(f"\n{'='*60}")
    print("DONE! Tables are ready for GPU use.")
    print(f"{'='*60}")
    
    # Compute rep dimension for usage example
    dim = tables['dim']
    
    print(f"""
To use these tables in your GPU code:

    from braid_search import Config, BraidSearchUltra, load_tables_from_file
    
    config = Config(
        n={args.n},
        r={args.r},
        prime={args.p},
        bucket_size=100000,
        max_length=50,
        device="cuda"
    )
    
    simple_matrices, valid_suffixes, num_valid_suffixes = load_tables_from_file(
        config, 
        table_path="{args.output_dir}/tables_B{args.n}_r{args.r}_p{args.p}.pt"
    )
    
    search = BraidSearchUltra(simple_matrices, valid_suffixes, num_valid_suffixes, config)
    kernel_braids = search.run()
""")
