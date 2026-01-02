#!/usr/bin/env python3
"""
Find kernel elements for various primes p.
ULTRA-OPTIMIZED VERSION v2 with reduced GC overhead and fused projlen.

Usage examples:
    python find_kernel_ultra_v2.py --p 7 --bucket-size 100000 --use-best 100000 --max-length 150 --device cuda
    python find_kernel_ultra_v2.py --p 5 --bucket-size 1500000 --use-best 750000 --max-length 60 --gc-every 20
"""

import sys
import argparse
import torch
import os

# Import the ultra-optimized v2 search
from braid_search_ultra_v2 import Config, BraidSearchUltra as BraidSearch, load_tables_from_file

# For verification (optional - comment out if peyl not available)
try:
    from peyl.braid import GNF, PermTable, BraidGroup
    from peyl.jonesrep import JonesCellRep
    from peyl import polymat
    import numpy as np
    PEYL_AVAILABLE = True
except ImportError:
    PEYL_AVAILABLE = False
    print("WARNING: peyl not available, skipping verification")


def verify_kernel_element(word_list, n=4, r=1, p=2):
    """Verify that a braid word is actually in the kernel using peyl."""
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
    p=2, 
    bucket_size=4000, 
    bootstrap_length=4, 
    max_length=None, 
    device="cpu", 
    chunk_size=50000, 
    use_best=0, 
    checkpoint_dir=None,
    checkpoint_every=9999,
    degree_multiplier=4,
    matmul_chunk_size=20000,
    gc_every_n_chunks=10,
    resume_from=None
):
    """Search for kernel elements using ultra-optimized v2 algorithm."""
    
    if max_length is None:
        max_length = 10 if p == 2 else 25
    
    config = Config(
        bucket_size=bucket_size,
        max_length=max_length,
        bootstrap_length=bootstrap_length,
        prime=p,
        degree_multiplier=degree_multiplier,
        checkpoint_every=checkpoint_every,
        device=device,
        expansion_chunk_size=chunk_size,
        use_best=use_best,
        matmul_chunk_size=matmul_chunk_size,
        gc_every_n_chunks=gc_every_n_chunks
    )
    
    print("="*60)
    print(f"SEARCHING FOR p={p} KERNEL ELEMENTS (ULTRA-OPTIMIZED v2)")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Bucket size: {config.bucket_size}")
    print(f"Max length: {config.max_length}")
    print(f"Bootstrap length: {config.bootstrap_length}")
    print(f"Prime: {config.prime}")
    print(f"Degree multiplier: {config.degree_multiplier}")
    print(f"Degree window: {config.degree_window}")
    print(f"Use best: {config.use_best if config.use_best > 0 else 'unlimited'}")
    print(f"Expansion chunk size: {config.expansion_chunk_size}")
    print(f"Matmul chunk size: {config.matmul_chunk_size}")
    print(f"GC every N chunks: {config.gc_every_n_chunks}")
    print(f"⚡ v2 optimizations: reduced GC, fused projlen")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print()
    
    # Find table path
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    
    # Try multiple possible locations
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
    
    center = config.degree_window // 2
    assert simple_burau[0, 0, 0, center] == 1, "Identity matrix check failed"
    print("✓ Identity matrix verified\n")
    
    search = BraidSearch(simple_burau, valid_suffixes, num_valid_suffixes, config)
    kernel_braids = search.run(checkpoint_dir=checkpoint_dir, resume_from=resume_from)
    
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
                
                if PEYL_AVAILABLE:
                    braid = GNF(n=4, power=0, factors=tuple(word_list))
                    print(f"    Artin word: {braid.magma_artin_word()}")
            elif i < 5:
                print(f"  Braid {i}: {word_list[:8]}{'...' if len(word_list) > 8 else ''} - {msg}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print("="*60)
    print(f"Total candidates with projlen=1: {sum(len(b) for b in kernel_braids)}")
    print(f"Verified kernel elements: {len(verified)}")
    
    if verified:
        print(f"\n✓ SUCCESS! Found {len(verified)} kernel elements for p={p}")
    
    return verified


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search for kernel elements (ULTRA-OPTIMIZED v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --p 7 --bucket-size 100000 --use-best 100000 --max-length 150 --device cuda
  %(prog)s --p 5 --bucket-size 1500000 --use-best 750000 --max-length 60 --gc-every 20
  %(prog)s --p 7 --resume-from checkpoints/final_state_level_50.pt --max-length 200

Recommended settings for H200 (80GB):
  --bucket-size 200000 --use-best 200000 --matmul-chunk 50000 --gc-every 20

Recommended settings for RTX 5000 (32GB):
  --bucket-size 100000 --use-best 100000 --matmul-chunk 20000 --gc-every 10
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

    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory to save checkpoints")
    
    parser.add_argument("--checkpoint-every", type=int, default=9999,
                        help="Save checkpoint every N levels")
    
    parser.add_argument("--degree-multiplier", type=int, default=4,
                        help="Degree window = 2 * multiplier * max_length + 1")
    
    parser.add_argument("--matmul-chunk", type=int, default=20000,
                        help="Chunk size for batched FFT matmul (tune for your GPU memory)")
    
    parser.add_argument("--gc-every", type=int, default=10,
                        help="Run garbage collection every N chunks (higher = faster but more memory)")
    
    parser.add_argument("--resume-from", "-r", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
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
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        degree_multiplier=args.degree_multiplier,
        matmul_chunk_size=args.matmul_chunk,
        gc_every_n_chunks=args.gc_every,
        resume_from=args.resume_from
    )
