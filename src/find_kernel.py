#!/usr/bin/env python3
"""
Find kernel elements for various primes p.

FIXED: The verification function now correctly checks for scalar matrices.

Usage examples:
    python find_kernel.py --p 2
    python find_kernel.py --p 3 --bucket-size 8000 --bootstrap-length 5
    python find_kernel.py --p 5 --bucket-size 10000 --max-length 30 --device cuda
"""

import sys
import argparse
import torch

# Add paths
sys.path.insert(0, '/Users/com36/burau-experiments')
sys.path.insert(0, '/Users/com36/burau-experiments/src')

from new_braid_search_v3 import Config, BraidSearch, load_tables_from_file

# For verification
from peyl.braid import GNF, PermTable, BraidGroup
from peyl.jonesrep import JonesCellRep
from peyl import polymat
import numpy as np


def verify_kernel_element(word_list, n=4, r=1, p=2):
    """
    Verify that a braid word is actually in the kernel using peyl.
    
    A braid is in the kernel if it evaluates to a scalar matrix c * v^k * I.
    
    Args:
        word_list: list of simple indices (not tensor)
        
    Returns:
        (is_kernel, message)
    """
    if not word_list:
        return False, "Empty word"
    
    # Create the braid using peyl
    try:
        braid = GNF(n=n, power=0, factors=tuple(word_list))
    except AssertionError as e:
        return False, f"Invalid normal form: {e}"
    
    # Evaluate in the representation using polymat (numerical)
    rep = JonesCellRep(n=n, r=r, p=p)
    result = rep.polymat_evaluate_braid(braid)
    if p > 0:
        result = result % p
    
    # Check if it's a scalar matrix:
    # - All diagonal entries must be equal
    # - All off-diagonal entries must be zero
    diag_poly = result[0, 0, :]
    
    for i in range(3):
        for j in range(3):
            if i == j:
                # Diagonal: must equal [0,0] entry
                if not np.array_equal(result[i, j, :], diag_poly):
                    return False, f"Diagonal mismatch at [{i},{j}]"
            else:
                # Off-diagonal: must be all zeros
                if np.any(result[i, j, :] != 0):
                    return False, f"Off-diagonal nonzero at [{i},{j}]"
    
    # It's a scalar matrix! Find what scalar
    nonzero_degs = np.where(diag_poly != 0)[0]
    if len(nonzero_degs) == 0:
        return True, "Kernel element! Evaluates to 0 (trivial)"
    
    # Format the scalar nicely
    if len(nonzero_degs) == 1:
        deg = nonzero_degs[0]
        coeff = diag_poly[deg]
        scalar_str = f"{coeff}*v^{deg}" if coeff != 1 else f"v^{deg}"
    else:
        terms = [f"{diag_poly[d]}*v^{d}" for d in nonzero_degs]
        scalar_str = " + ".join(terms)
    
    return True, f"Kernel element! Evaluates to ({scalar_str}) * I"


def find_kernel(p=2, bucket_size=4000, bootstrap_length=4, max_length=None, device="cpu", chunk_size=50000, use_best=0, checkpoint_dir=None):
    """Search for kernel elements.
    
    Args:
        p: Prime for the representation
        bucket_size: Number of braids to keep per projlen bucket
        bootstrap_length: Length of initial exhaustive search
        max_length: Maximum braid length to search (default: 10 for p=2, 25 otherwise)
        device: "cpu" or "cuda"
        chunk_size: Max candidates to process at once (lower = less memory, slower)
        use_best: Max braids to expand per level, prioritizing low projlen (0 = no limit)
    """
    
    if max_length is None:
        max_length = 10 if p == 2 else 25
    
    # Configuration 
    config = Config(
        bucket_size=bucket_size,
        max_length=max_length,
        bootstrap_length=bootstrap_length,
        prime=p,
        degree_multiplier=3,
        checkpoint_every=max_length + 1,  # Don't checkpoint for short runs
        device=device,
        expansion_chunk_size=chunk_size,
        use_best=use_best
    )
    
    print("="*60)
    print(f"SEARCHING FOR p={p} KERNEL ELEMENTS")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Bucket size: {config.bucket_size}")
    print(f"Max length: {config.max_length}")
    print(f"Bootstrap length: {config.bootstrap_length}")
    print(f"Prime: {config.prime}")
    print(f"Degree window: {config.degree_window}")
    print(f"Use best: {config.use_best if config.use_best > 0 else 'unlimited'}")
    print()
    
    # Load tables
    import os

    # Get the directory where the script is located (src)
    script_dir = os.path.dirname(os.path.abspath(__file__)) 

    # Move up one level to the project root and then into precomputed_tables
    project_root = os.path.dirname(script_dir)
    table_path_end = f"tables_B4_r1_p{p}.pt"
    table_path = os.path.join(project_root, "precomputed_tables", table_path_end)

    
    try:
        simple_burau, valid_suffixes, num_valid_suffixes = load_tables_from_file(
            config, 
            table_path=table_path
        )
    except FileNotFoundError:
        print(f"ERROR: Table file not found at {table_path}")
        print(f"Please generate tables with p={p}")
        return None
    except AssertionError as e:
        print(f"ERROR: {e}")
        return None
    
    # Verify identity matrix
    center = config.degree_window // 2
    assert simple_burau[0, 0, 0, center] == 1, "Identity matrix check failed"
    print("✓ Identity matrix verified\n")
    
    # Run the search
    search = BraidSearch(simple_burau, valid_suffixes, num_valid_suffixes, config)
    kernel_braids = search.run(checkpoint_dir=checkpoint_dir)
    
    # Verify found braids
    print("\n" + "="*60)
    print("VERIFICATION USING PEYL")
    print("="*60)
    
    if not kernel_braids:
        print("No projlen=1 braids found.")
        return None
    
    verified = []
    
    for batch_idx, batch in enumerate(kernel_braids):
        print(f"\nBatch {batch_idx}: {len(batch)} candidates")
        
        for i, word_tensor in enumerate(batch):
            # Convert tensor to list, remove padding
            word_list = [w.item() for w in word_tensor]
            while word_list and word_list[-1] == 0:
                word_list.pop()
            
            # Skip if somehow empty
            if not word_list:
                continue
            
            # Verify with peyl
            is_kernel, msg = verify_kernel_element(word_list, p=p)
            
            if is_kernel:
                verified.append(word_list)
                print(f"\n  🎉 KERNEL ELEMENT #{len(verified)} 🎉")
                print(f"    Factors: {word_list}")
                print(f"    Length: {len(word_list)}")
                print(f"    {msg}")
                
                # Get Artin word for Magma verification
                braid = GNF(n=4, power=0, factors=tuple(word_list))
                print(f"    Artin word: {braid.magma_artin_word()}")
            elif i < 5:  # Only show first few failures
                print(f"  Braid {i}: {word_list[:8]}{'...' if len(word_list) > 8 else ''} - {msg}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print("="*60)
    print(f"Total candidates with projlen=1: {sum(len(b) for b in kernel_braids)}")
    print(f"Verified kernel elements: {len(verified)}")
    
    if verified:
        print(f"\n✓ SUCCESS! Found {len(verified)} kernel elements for p={p}")
    else:
        print(f"\n✗ No kernel elements verified (this shouldn't happen)")
    
    return verified


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search for kernel elements in Burau representations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --p 2
  %(prog)s --p 3 --bucket-size 8000
  %(prog)s --p 5 --bucket-size 10000 --bootstrap-length 5 --max-length 30
  %(prog)s --p 7 --device cuda
  %(prog)s --p 5 --use-best 50000 --bucket-size 15000 -d cuda  # Like peyl's use-best
        """
    )
    
    parser.add_argument(
        "--p", "-p",
        type=int,
        default=2,
        help="Prime for the representation (default: 2)"
    )
    
    parser.add_argument(
        "--bucket-size", "-b",
        type=int,
        default=4000,
        help="Number of braids to keep per projlen bucket (default: 4000)"
    )
    
    parser.add_argument(
        "--bootstrap-length", "-l",
        type=int,
        default=4,
        help="Length of initial exhaustive search (default: 4)"
    )
    
    parser.add_argument(
        "--max-length", "-m",
        type=int,
        default=None,
        help="Maximum braid length to search (default: 10 for p=2, 25 otherwise)"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for computation (default: cpu)"
    )
    
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=50000,
        help="Max candidates to process at once in expansion step (default: 50000). Lower = less memory usage."
    )
    
    parser.add_argument(
        "--use-best", "-u",
        type=int,
        default=0,
        help="Max braids to expand per level, prioritizing low projlen (default: 0 = no limit). Like peyl's --use-best."
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save JSON checkpoints"
    )
    
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
        checkpoint_dir=args.checkpoint_dir
    )