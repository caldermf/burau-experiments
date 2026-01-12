#!/usr/bin/env python3
"""
Find kernel elements for various primes p and partitions (n-r, r).
Generalized to support arbitrary n (number of strands) and r (partition parameter).

Usage examples:
    # Burau representation for B_4
    python find_kernel.py --n 4 --r 1 --p 5 --bucket-size 100000 --max-length 60
    
    # (4,2) representation for B_6
    python find_kernel.py --n 6 --r 2 --p 5 --bucket-size 50000 --max-length 40
    
    # (8,2) representation for B_10 (Geordie's suggestion!)
    python find_kernel.py --n 10 --r 2 --p 0 --bucket-size 10000 --max-length 30
"""

import argparse
import math
import torch
import os

from braid_search import Config, BraidSearchUltra as BraidSearch, load_tables_from_file

# For verification (optional)
try:
    from peyl.braid import GNF
    from peyl.jonesrep import JonesCellRep
    import numpy as np
    PEYL_AVAILABLE = True
except ImportError:
    PEYL_AVAILABLE = False
    print("WARNING: peyl not available, skipping verification")


def verify_kernel_element(word_list, n=4, r=1, p=2):
    """
    Verify that a braid word is actually in the kernel using peyl.
    
    A braid is in the kernel (up to central elements) if it evaluates to a scalar
    multiple of identity (for Delta^even) or anti-diagonal permutation (for Delta^odd).
    
    Generalized for arbitrary dimension.
    """
    if not PEYL_AVAILABLE:
        return True, "Verification skipped (peyl not available)"
    
    if not word_list:
        return False, "Empty word"
    
    try:
        braid = GNF(n=n, power=0, factors=tuple(word_list))
    except AssertionError as e:
        return False, f"Invalid normal form: {e}"
    
    rep = JonesCellRep(n=n, r=r, p=p)
    dim = rep.dimension()
    result = rep.polymat_evaluate_braid(braid)
    if p > 0:
        result = result % p
    
    # Check if it's a scalar multiple of I (Delta^even case)
    is_scalar_identity = True
    diag_poly = result[0, 0, :]
    
    for i in range(dim):
        for j in range(dim):
            if i == j:
                if not np.array_equal(result[i, j, :], diag_poly):
                    is_scalar_identity = False
                    break
            else:
                if np.any(result[i, j, :] != 0):
                    is_scalar_identity = False
                    break
        if not is_scalar_identity:
            break
    
    if is_scalar_identity:
        nonzero_degs = np.where(diag_poly != 0)[0]
        if len(nonzero_degs) == 0:
            return True, "Kernel element! Evaluates to 0 (trivial)"
        if len(nonzero_degs) == 1:
            deg = nonzero_degs[0]
            coeff = diag_poly[deg]
            scalar_str = f"{coeff}*v^{deg}" if coeff != 1 else f"v^{deg}"
        else:
            terms = [f"{diag_poly[d]}*v^{d}" for d in nonzero_degs]
            scalar_str = " + ".join(terms)
        return True, f"Kernel element! Evaluates to ({scalar_str}) * I"
    
    # Check if it's a scalar multiple of the anti-diagonal permutation (Delta^odd case)
    # Anti-diagonal: (i, dim-1-i) for i in range(dim) should all be equal; everything else zero
    is_scalar_antidiag = True
    antidiag_poly = result[0, dim - 1, :]  # First anti-diagonal entry
    
    # Check anti-diagonal entries are all equal
    for i in range(dim):
        j = dim - 1 - i
        if not np.array_equal(result[i, j, :], antidiag_poly):
            is_scalar_antidiag = False
            break
    
    # Check all other entries are zero
    if is_scalar_antidiag:
        for i in range(dim):
            for j in range(dim):
                if i + j != dim - 1:  # Not on anti-diagonal
                    if np.any(result[i, j, :] != 0):
                        is_scalar_antidiag = False
                        break
            if not is_scalar_antidiag:
                break
    
    if is_scalar_antidiag:
        nonzero_degs = np.where(antidiag_poly != 0)[0]
        if len(nonzero_degs) == 0:
            return True, "Kernel element! Evaluates to 0 (trivial)"
        if len(nonzero_degs) == 1:
            deg = nonzero_degs[0]
            coeff = antidiag_poly[deg]
            scalar_str = f"{coeff}*v^{deg}" if coeff != 1 else f"v^{deg}"
        else:
            terms = [f"{antidiag_poly[d]}*v^{d}" for d in nonzero_degs]
            scalar_str = " + ".join(terms)
        return True, f"Kernel element! Evaluates to ({scalar_str}) * Delta"
    
    # Not a scalar multiple of any power of Delta
    # Provide diagnostic info about which entry failed
    for i in range(dim):
        for j in range(dim):
            if np.any(result[i, j, :] != 0):
                if (i == j) or (i + j == dim - 1):  # diagonal or anti-diagonal
                    continue
                return False, f"Off-diagonal nonzero at [{i},{j}]"
    
    return False, "Not a scalar multiple of I or Delta"


def find_kernel(
    n=4,
    r=1,
    p=5, 
    bucket_size=4000, 
    bootstrap_length=4, 
    max_length=127, 
    device="cuda", 
    chunk_size=50000, 
    use_best=0, 
    degree_multiplier=2,
    matmul_chunk_size=8000,
    table_dir="precomputed_tables"
):
    """Search for kernel elements using ultra-optimized algorithm."""
    
    # Compute representation info
    if r == 0:
        dim = 1
    else:
        dim = math.comb(n, r) - math.comb(n, r - 1)
    num_simples = math.factorial(n)
    
    if max_length is None:
        max_length = 10 if p == 2 else 25
    
    config = Config(
        n=n,
        r=r,
        bucket_size=bucket_size,
        max_length=max_length,
        bootstrap_length=bootstrap_length,
        prime=p,
        degree_multiplier=degree_multiplier,
        device=device,
        expansion_chunk_size=chunk_size,
        use_best=use_best,
        matmul_chunk_size=matmul_chunk_size
    )
    
    print("="*60)
    print(f"SEARCHING FOR KERNEL ELEMENTS")
    print("="*60)
    print(f"Braid group: B_{n}")
    print(f"Representation: ({n-r}, {r})")
    print(f"Dimension: {dim}")
    print(f"Number of simples: {num_simples:,}")
    print(f"Prime: {p if p > 0 else 'char 0'}")
    print(f"Device: {config.device}")
    print(f"Bucket size: {config.bucket_size}")
    print(f"Max length: {config.max_length}")
    print(f"Bootstrap length: {config.bootstrap_length}")
    print(f"Degree multiplier: {config.degree_multiplier}")
    print(f"Degree window: [0, {config.degree_window - 1}] ({config.degree_window} coeffs)")
    print(f"Use best: {config.use_best if config.use_best > 0 else 'unlimited'}")
    print(f"Expansion chunk size: {config.expansion_chunk_size}")
    print(f"Matmul chunk size: {config.matmul_chunk_size}")
    print()
    
    # Find table path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    table_filename = f"tables_B{n}_r{r}_p{p}.pt"
    
    possible_paths = [
        os.path.join(table_dir, table_filename),
        os.path.join(script_dir, table_dir, table_filename),
        os.path.join(script_dir, "precomputed_tables", table_filename),
        os.path.join(os.path.dirname(script_dir), "precomputed_tables", table_filename),
        os.path.join(script_dir, table_filename),
        table_filename,
    ]
    
    table_path = None
    for path in possible_paths:
        if os.path.exists(path):
            table_path = path
            break
    
    if table_path is None:
        print(f"ERROR: Could not find table file {table_filename}")
        print(f"Searched in: {possible_paths[:4]}")
        print(f"\nGenerate tables first with:")
        print(f"  python generate_tables.py --n {n} --r {r} --p {p}")
        return None

    try:
        simple_matrices, valid_suffixes, num_valid_suffixes = load_tables_from_file(
            config, 
            table_path=table_path
        )
    except FileNotFoundError:
        print(f"ERROR: Table file not found at {table_path}")
        return None
    except AssertionError as e:
        print(f"ERROR: {e}")
        return None
    
    # Identity matrix check at index 0
    assert simple_matrices[0, 0, 0, 0] == 1, "Identity matrix check failed"
    if dim > 1:
        assert simple_matrices[0, 1, 1, 0] == 1, "Identity matrix check failed"
    print("✓ Identity matrix verified (at degree 0)\n")
    
    search = BraidSearch(simple_matrices, valid_suffixes, num_valid_suffixes, config)
    kernel_braids = search.run()
    
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    if not kernel_braids:
        print("No projlen=1 braids found.")
        return None
    
    verified = []
    
    for batch_idx, batch in enumerate(kernel_braids):
        print(f"\nBatch {batch_idx}: {len(batch)} candidates")
        
        for i, word_tensor in enumerate(batch):
            word_list = [w.item() for w in word_tensor]
            while word_list and word_list[-1] == 0:
                word_list.pop()
            
            if not word_list:
                continue
            
            is_kernel, msg = verify_kernel_element(word_list, n=n, r=r, p=p)
            
            if is_kernel:
                verified.append(word_list)
                print(f"\n  🎉 KERNEL ELEMENT #{len(verified)} 🎉")
                print(f"    Factors: {word_list}")
                print(f"    Length: {len(word_list)}")
                print(f"    {msg}")
                
                if PEYL_AVAILABLE:
                    braid = GNF(n=n, power=0, factors=tuple(word_list))
                    print(f"    Artin word: {braid.magma_artin_word()}")
            elif i < 20:
                print(f"  Braid {i}: {word_list[:8]}{'...' if len(word_list) > 8 else ''} - {msg}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print("="*60)
    print(f"Total candidates with projlen=1: {sum(len(b) for b in kernel_braids)}")
    print(f"Verified kernel elements: {len(verified)}")
    
    if verified:
        print(f"\n✓ SUCCESS! Found {len(verified)} kernel elements")
        print(f"  Representation: ({n-r}, {r}) mod {p if p > 0 else 'char 0'}")
    
    return verified


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search for kernel elements (generalized for any n and partition)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Burau representation for B_4 (original)
  %(prog)s --n 4 --r 1 --p 5 --bucket-size 100000 --max-length 60
  
  # Burau representation for B_5
  %(prog)s --n 5 --r 1 --p 7 --bucket-size 100000 --max-length 80
  
  # (4,2) representation for B_6 (Geordie's suggestion!)
  %(prog)s --n 6 --r 2 --p 5 --bucket-size 50000 --max-length 40
  
  # (8,2) representation for B_10 - the holy grail!
  %(prog)s --n 10 --r 2 --p 0 --bucket-size 10000 --max-length 30

Representation dimensions:
  (n-1, 1): n-1 dimensional (Burau)
  (n-2, 2): C(n,2) - C(n,1) = n(n-1)/2 - 1 dimensional
  
  Examples:
    B_4, (3,1): dim=3
    B_5, (4,1): dim=4  
    B_6, (4,2): dim=9
    B_10, (8,2): dim=44

Recommended settings for H200 (80GB):
  --bucket-size 200000 --use-best 200000 --matmul-chunk 50000

Recommended settings for RTX 5000 (32GB):
  --bucket-size 100000 --use-best 100000 --matmul-chunk 20000
        """
    )
    
    parser.add_argument("--n", type=int, required=True,
                        help="Number of strands in braid group B_n")
    
    parser.add_argument("--r", type=int, required=True,
                        help="Partition parameter for (n-r, r) representation")
    
    parser.add_argument("--p", type=int, default=5,
                        help="Prime for the representation (0 for char 0)")
    
    parser.add_argument("--bucket-size", "-b", type=int, default=4000,
                        help="Number of braids to keep per projlen bucket")
    
    parser.add_argument("--bootstrap-length", "-l", type=int, default=5,
                        help="Length of initial exhaustive search")
    
    parser.add_argument("--max-length", "-m", type=int, default=None,
                        help="Maximum braid length to search")
    
    parser.add_argument("--device", "-d", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda"],
                        help="Device to use")
    
    parser.add_argument("--chunk-size", "-c", type=int, default=50000,
                        help="Max candidates per expansion chunk")
    
    parser.add_argument("--use-best", "-u", type=int, default=0,
                        help="Max braids to expand per level")
    
    parser.add_argument("--degree-multiplier", type=int, default=2,
                        help="Degree window = multiplier * max_length + 1")
    
    parser.add_argument("--matmul-chunk", type=int, default=8000,
                        help="Chunk size for batched FFT matmul")
    
    parser.add_argument("--table-dir", type=str, default="precomputed_tables",
                        help="Directory containing precomputed tables")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Validate parameters
    assert args.n >= 2, f"Need n >= 2, got {args.n}"
    assert args.r >= 1, f"Need r >= 1, got {args.r}"
    assert args.n - 2 * args.r >= 0, f"Need n >= 2r for partition ({args.n - args.r}, {args.r})"
    
    # Warn about large searches
    num_simples = math.factorial(args.n)
    if num_simples > 1000000:
        print(f"WARNING: B_{args.n} has {num_simples:,} simples. This will be slow!")
    
    find_kernel(
        n=args.n,
        r=args.r,
        p=args.p,
        bucket_size=args.bucket_size,
        bootstrap_length=args.bootstrap_length,
        max_length=args.max_length,
        device=args.device,
        chunk_size=args.chunk_size,
        use_best=args.use_best,
        degree_multiplier=args.degree_multiplier,
        matmul_chunk_size=args.matmul_chunk,
        table_dir=args.table_dir
    )
