#!/usr/bin/env python3
"""
Find kernel elements with ADAPTIVE FFT sizing.

This version uses smaller FFTs at early levels for massive speedup:
- Levels 1-15:   fft_size=128  (64x smaller than max!)
- Levels 16-30:  fft_size=256  (32x smaller)
- Levels 31-50:  fft_size=512  (16x smaller)
- Levels 51-80:  fft_size=1024 (8x smaller)
- Levels 81-120: fft_size=2048 (4x smaller)
- Levels 121+:   fft_size=4096 (full size)

Usage:
    python find_kernel_adaptive.py --p 7 --bucket-size 200000 --use-best 200000 --max-length 150
"""

import sys
import argparse
import torch
import os

from braid_search_adaptive import Config, BraidSearchAdaptive as BraidSearch, load_tables_from_file

# Optional verification
try:
    from peyl.braid import GNF
    from peyl.jonesrep import JonesCellRep
    import numpy as np
    PEYL_AVAILABLE = True
except ImportError:
    PEYL_AVAILABLE = False
    print("Note: peyl not available, skipping verification")


def verify_kernel_element(word_list, n=4, r=1, p=2):
    """Verify that a braid word is actually in the kernel."""
    if not PEYL_AVAILABLE:
        return True, "Verification skipped"
    
    if not word_list:
        return False, "Empty word"
    
    try:
        braid = GNF(n=n, power=0, factors=tuple(word_list))
    except AssertionError as e:
        return False, f"Invalid: {e}"
    
    rep = JonesCellRep(n=n, r=r, p=p)
    result = rep.polymat_evaluate_braid(braid)
    if p > 0:
        result = result % p
    
    diag_poly = result[0, 0, :]
    
    for i in range(3):
        for j in range(3):
            if i == j:
                if not np.array_equal(result[i, j, :], diag_poly):
                    return False, f"Diagonal mismatch"
            else:
                if np.any(result[i, j, :] != 0):
                    return False, f"Off-diagonal nonzero"
    
    return True, "Kernel element!"


def find_kernel(
    p=7, 
    bucket_size=100000, 
    bootstrap_length=5, 
    max_length=150, 
    device="cuda", 
    chunk_size=50000, 
    use_best=100000, 
    checkpoint_dir=None,
    checkpoint_every=10,
    degree_multiplier=4,
    matmul_chunk_size=10000,
    resume_from=None
):
    """Search for kernel elements using adaptive FFT algorithm."""
    
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
        matmul_chunk_size=matmul_chunk_size
    )
    
    print("="*60)
    print(f"KERNEL SEARCH - ADAPTIVE FFT")
    print("="*60)
    print(f"Prime: {p}")
    print(f"Device: {config.device}")
    print(f"Bucket size: {config.bucket_size}")
    print(f"Use best: {config.use_best}")
    print(f"Max length: {config.max_length}")
    print(f"Degree window (max): {config.degree_window}")
    print(f"Matmul chunk: {config.matmul_chunk_size}")
    print()
    print("FFT Tiers (fft_size must be >= 2*D-1):")
    print("  Levels 1-15:   D=33,   fft_size=128")
    print("  Levels 16-30:  D=65,   fft_size=256")
    print("  Levels 31-50:  D=129,  fft_size=512")
    print("  Levels 51-80:  D=257,  fft_size=1024")
    print("  Levels 81-120: D=513,  fft_size=2048")
    print("  Levels 121+:   D=1025, fft_size=4096")
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
        return None

    simple_burau, valid_suffixes, num_valid_suffixes = load_tables_from_file(config, table_path)
    
    center = config.degree_window // 2
    assert simple_burau[0, 0, 0, center] == 1, "Identity check failed"
    print("✓ Identity matrix verified\n")
    
    search = BraidSearch(simple_burau, valid_suffixes, num_valid_suffixes, config)
    kernel_braids = search.run(checkpoint_dir=checkpoint_dir, resume_from=resume_from)
    
    # Verification
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    if not kernel_braids:
        print("No projlen=1 braids found.")
        return None
    
    verified = []
    for batch_idx, batch in enumerate(kernel_braids):
        for word_tensor in batch:
            word_list = [w.item() for w in word_tensor]
            while word_list and word_list[-1] == 0:
                word_list.pop()
            if word_list:
                is_kernel, msg = verify_kernel_element(word_list, p=p)
                if is_kernel:
                    verified.append(word_list)
                    print(f"  🎉 KERNEL #{len(verified)}: length={len(word_list)}")
    
    print(f"\nTotal verified: {len(verified)}")
    return verified


def parse_args():
    parser = argparse.ArgumentParser(description="Kernel search with adaptive FFT")
    
    parser.add_argument("--p", type=int, default=7)
    parser.add_argument("--bucket-size", "-b", type=int, default=100000)
    parser.add_argument("--bootstrap-length", type=int, default=5)
    parser.add_argument("--max-length", "-m", type=int, default=150)
    parser.add_argument("--device", "-d", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", "-c", type=int, default=50000)
    parser.add_argument("--use-best", "-u", type=int, default=100000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--degree-multiplier", type=int, default=4)
    parser.add_argument("--matmul-chunk", type=int, default=10000, 
                        help="Smaller is often faster (default: 10000)")
    parser.add_argument("--resume-from", "-r", type=str, default=None)
    
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
        resume_from=args.resume_from
    )
