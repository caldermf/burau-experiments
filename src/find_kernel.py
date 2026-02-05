#!/usr/bin/env python3
"""
Find kernel elements for various primes p.
ULTRA-OPTIMIZED VERSION with precomputed FFTs and non-negative degrees only.

Usage examples:
    python find_kernel_ultra.py --p 7 --bucket-size 100000 --use-best 100000 --max-length 150 --device cuda
    python find_kernel_ultra.py --p 5 --bucket-size 1500000 --use-best 750000 --max-length 60
"""

import argparse
import torch
import os

from braid_search import Config, BraidSearchUltra as BraidSearch, load_tables_from_file
import kernel_database

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
    """Verify that a braid word is actually in the kernel using peyl.
    
    A braid is in the kernel (up to central elements) if it evaluates to a scalar
    multiple of some power of Delta. Since Delta is central, if the braid evaluates
    to c(v) * Delta^k, then multiplying by Delta^{-k} gives c(v) * I, which is in
    the kernel.
    
    Delta's Burau representation is the anti-diagonal matrix:
    [[0, 0, -v^4], [0, -v^4, 0], [-v^4, 0, 0]]
    
    So Delta^(even) is a scalar multiple of I, and Delta^(odd) is a scalar multiple
    of the anti-diagonal permutation matrix.
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
    result = rep.polymat_evaluate_braid(braid)
    if p > 0:
        result = result % p
    
    # Check if it's a scalar multiple of I (Delta^even case)
    is_scalar_identity = True
    diag_poly = result[0, 0, :]
    
    for i in range(3):
        for j in range(3):
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
        # Only accept if diagonal is v^{8k} for some k (i.e. a power of Delta^2)
        nonzero_degs = np.where(diag_poly != 0)[0]
        if len(nonzero_degs) == 0:
            return False, "Evaluates to 0 (not a power of Delta)"
        if len(nonzero_degs) != 1:
            return False, f"Diagonal is not a single power of v: {nonzero_degs}"
        deg = int(nonzero_degs[0])
        coeff = int(diag_poly[deg])
        if deg % 8 != 0:
            return False, f"Diagonal degree {deg} is not 8k (expected v^{{8k}})"
        if coeff != 1:
            return False, f"Diagonal coefficient {coeff} is not 1 (expected v^{deg})"
        k = deg // 8
        return True, f"Kernel element! Evaluates to v^{deg} * I = Delta^{2*k}"
    
    # Check if it's a scalar multiple of the anti-diagonal (Delta^odd case)
    # Delta = [[0,0,-v^4],[0,-v^4,0],[-v^4,0,0]]; odd powers are -v^{8k+4} on antidiag
    is_scalar_antidiag = True
    antidiag_poly = result[0, 2, :]
    
    if not np.array_equal(result[1, 1, :], antidiag_poly):
        is_scalar_antidiag = False
    if not np.array_equal(result[2, 0, :], antidiag_poly):
        is_scalar_antidiag = False
    
    if is_scalar_antidiag:
        for i, j in [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2)]:
            if np.any(result[i, j, :] != 0):
                is_scalar_antidiag = False
                break
    
    if is_scalar_antidiag:
        # Only accept if antidiagonal is -v^{8k+4} for some k (coefficient -1 mod p)
        nonzero_degs = np.where(antidiag_poly != 0)[0]
        if len(nonzero_degs) == 0:
            return False, "Evaluates to 0 (not a power of Delta)"
        if len(nonzero_degs) != 1:
            return False, f"Antidiagonal is not a single power of v: {nonzero_degs}"
        deg = int(nonzero_degs[0])
        coeff = int(antidiag_poly[deg])
        minus_one = (p - 1) % p
        if deg % 8 != 4:
            return False, f"Antidiagonal degree {deg} is not 8k+4 (expected -v^{{8k+4}})"
        if coeff != minus_one:
            return False, f"Antidiagonal coefficient {coeff} is not -1 mod p (expected -v^{deg})"
        k = (deg - 4) // 8
        return True, f"Kernel element! Evaluates to -v^{deg} * Delta = Delta^{2*k+1}"
    
    # Not a power of Delta
    # Provide diagnostic info about which entry failed
    for i in range(3):
        for j in range(3):
            if np.any(result[i, j, :] != 0):
                if (i == j) or (i + j == 2):  # diagonal or anti-diagonal
                    continue
                return False, f"Off-diagonal nonzero at [{i},{j}]"
    
    return False, "Not a power of Delta (wrong scalar form)"
    
    rep = JonesCellRep(n=n, r=r, p=p)
    result = rep.polymat_evaluate_braid(braid)
    if p > 0:
        result = result % p
    
    diag_poly = result[0, 0, :]
    
    for i in range(3):
        for j in range(3):
            if i == j:
                if not np.array_equal(result[i, j, :], diag_poly):
                    return False, f"Diagonal mismatch at [{i},{j}]"
            else:
                if np.any(result[i, j, :] != 0):
                    return False, f"Off-diagonal nonzero at [{i},{j}]"
    
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


def find_kernel(
    p=5, 
    bucket_size=4000, 
    bootstrap_length=4, 
    max_length=127, 
    device="cuda", 
    chunk_size=50000, 
    use_best=0, 
    degree_multiplier=2,
    matmul_chunk_size=8000
):
    """Search for kernel elements using ultra-optimized algorithm."""
    
    if max_length is None:
        max_length = 10 if p == 2 else 25
    
    config = Config(
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
    print(f"SEARCHING FOR p={p} KERNEL ELEMENTS (ULTRA-OPTIMIZED)")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Bucket size: {config.bucket_size}")
    print(f"Max length: {config.max_length}")
    print(f"Bootstrap length: {config.bootstrap_length}")
    print(f"Prime: {config.prime}")
    print(f"Degree multiplier: {config.degree_multiplier}")
    print(f"Degree window: [0, {config.degree_window - 1}] ({config.degree_window} coeffs) - NON-NEGATIVE ONLY")
    print(f"Use best: {config.use_best if config.use_best > 0 else 'unlimited'}")
    print(f"Expansion chunk size: {config.expansion_chunk_size}")
    print(f"Matmul chunk size: {config.matmul_chunk_size}")
    print(f"⚡ Precomputed FFTs: ENABLED")
    print(f"⚡ Non-negative degrees only: ENABLED (2x memory savings!)")
    print()
    
    # Find table path
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    
    possible_paths = [
        os.path.join(script_dir, "precomputed_tables", f"tables_B4_r1_p{p}.pt"),
        os.path.join(os.path.dirname(script_dir), "precomputed_tables", f"tables_B4_r1_p{p}.pt"),
        os.path.join(script_dir, f"tables_B4_r1_p{p}.pt"),
        f"tables_B4_r1_p{p}.pt",
    ]
    
    table_path = None
    for path in possible_paths:
        if os.path.exists(path):
            table_path = path
            break
    
    if table_path is None:
        print(f"ERROR: Could not find table file for p={p}")
        print(f"Searched in: {possible_paths}")
        return None

    try:
        simple_burau, valid_suffixes, num_valid_suffixes = load_tables_from_file(
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
    assert simple_burau[0, 0, 0, 0] == 1, "Identity matrix check failed"
    assert simple_burau[0, 1, 1, 0] == 1, "Identity matrix check failed"
    assert simple_burau[0, 2, 2, 0] == 1, "Identity matrix check failed"
    print("✓ Identity matrix verified (at degree 0)\n")
    
    search = BraidSearch(simple_burau, valid_suffixes, num_valid_suffixes, config)
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
            
            is_kernel, msg = verify_kernel_element(word_list, p=p)
            
            if is_kernel:
                verified.append(word_list)
                print(f"\n  🎉 KERNEL ELEMENT #{len(verified)} 🎉")
                print(f"    Factors: {word_list}")
                print(f"    Length: {len(word_list)}")
                print(f"    {msg}")
            elif i < 20:
                print(f"  Braid {i}: {word_list[:8]}{'...' if len(word_list) > 8 else ''} - {msg}")
                
            if PEYL_AVAILABLE:
                braid = GNF(n=4, power=0, factors=tuple(word_list))
                print(f"    Artin word: {braid.magma_artin_word()}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print("="*60)
    print(f"Total candidates with projlen=1: {sum(len(b) for b in kernel_braids)}")
    print(f"Verified kernel elements: {len(verified)}")
    
    # Save verified kernel elements to database
    if verified:
        num_new, num_total = kernel_database.add_kernel_elements(p, verified)
        print(f"\n✓ SUCCESS! Found {len(verified)} kernel elements for p={p}")
        print(f"  Database updated: {num_new} new, {num_total} total for p={p}")
    else:
        # Still show database stats even if no new elements
        stats = kernel_database.get_statistics()
        if str(p) in [str(k) for k in stats["primes"].keys()]:
            print(f"\n  Database has {stats['primes'][p]['count']} kernel elements for p={p}")
    
    return verified


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search for kernel elements (ULTRA-OPTIMIZED)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --p 7 --bucket-size 100000 --use-best 100000 --max-length 150 --device cuda
  %(prog)s --p 5 --bucket-size 1500000 --use-best 750000 --max-length 60

Recommended settings for H200 (80GB):
  --bucket-size 200000 --use-best 200000 --matmul-chunk 50000

Recommended settings for RTX 5000 (32GB):
  --bucket-size 100000 --use-best 100000 --matmul-chunk 20000
        """
    )
    
    parser.add_argument("--p", "-p", type=int, default=2,
                        help="Prime for the representation (default: 2)")
    
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
    
    parser.add_argument("--matmul-chunk", type=int, default=20000,
                        help="Chunk size for batched FFT matmul")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    find_kernel(
        p=args.p,
        bucket_size=args.bucket_size,
        bootstrap_length=args.bootstrap_length,
        max_length=args.max_length,
        device=args.device,
        chunk_size=args.chunk_size,
        use_best=args.use_best,
        degree_multiplier=args.degree_multiplier,
        matmul_chunk_size=args.matmul_chunk
    )
