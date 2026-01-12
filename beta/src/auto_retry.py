#!/usr/bin/env python3
"""
Auto-retry Supervisor for Kernel Search
---------------------------------------
Repeatedly runs searches for B_6 (4,2) mod 3.
Logic:
1. Run up to max_length=63.
2. At length 30, check the minimum projective length in the buckets.
3. If min_projlen >= 40, ABORT and restart.
4. If min_projlen < 40, continue.
5. Stop only when a kernel element is found.
"""

import torch
import time
import sys
import os
import argparse

# Import necessary components from your existing files
# Assuming files are in the same directory
from braid_search import Config, BraidSearchUltra
from find_kernel import load_tables_from_file, verify_kernel_element

def run_auto_search():
    # --- CONFIGURATION ---
    # Hardcoded based on your specific request and log snippet
    N = 6
    R = 2
    P = 3
    MAX_LENGTH = 63
    
    # The "Give Up" Condition
    CHECK_LEVEL = 23
    PROJLEN_CUTOFF = 21  # Must be strictly less than this to continue
    
    # Search hyperparameters (matched to your log)
    BUCKET_SIZE = 1000    # Increased slightly from 1000 for robustness
    USE_BEST = 2000
    MATMUL_CHUNK = 8000
    EXPANSION_CHUNK = 50000
    DEGREE_MULTIPLIER = 2
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"=== AUTO-RETRY KERNEL SEARCH ===")
    print(f"Target: B_{N} ({N-R},{R}) mod {P}")
    print(f"Condition: Restart if min_projlen >= {PROJLEN_CUTOFF} at level {CHECK_LEVEL}")
    print(f"Max Length: {MAX_LENGTH}")
    print(f"Device: {device}")
    print("="*40)

    # --- 1. Load Tables (Once) ---
    # We create a dummy config just to load the tables
    config_template = Config(
        n=N, r=R, prime=P,
        bucket_size=BUCKET_SIZE,
        max_length=MAX_LENGTH,
        bootstrap_length=2,
        degree_multiplier=DEGREE_MULTIPLIER,
        device=device,
        expansion_chunk_size=EXPANSION_CHUNK,
        use_best=USE_BEST,
        matmul_chunk_size=MATMUL_CHUNK
    )

    # Locate table file
    table_filename = f"tables_B{N}_r{R}_p{P}.pt"
    possible_paths = [
        table_filename,
        os.path.join("precomputed_tables", table_filename),
        os.path.join(os.path.dirname(__file__), "precomputed_tables", table_filename),
        # Add path from your log for convenience
        "/nfs/roberts/project/pi_com36/com36/burau-experiments/beta/precomputed_tables/" + table_filename
    ]
    
    table_path = None
    for path in possible_paths:
        if os.path.exists(path):
            table_path = path
            break
            
    if not table_path:
        print(f"ERROR: Could not find table file {table_filename}")
        sys.exit(1)
        
    print(f"Loading tables from: {table_path}")
    try:
        simple_matrices, valid_suffixes, num_valid_suffixes = load_tables_from_file(
            config_template, table_path
        )
    except Exception as e:
        print(f"Error loading tables: {e}")
        sys.exit(1)

    # --- 2. The Infinite Retry Loop ---
    run_count = 0
    total_start_time = time.time()
    
    while True:
        run_count += 1
        print(f"\n\n{'#'*60}")
        print(f"STARTING RUN #{run_count}")
        print(f"{'#'*60}")
        
        # Initialize Search
        search = BraidSearchUltra(simple_matrices, valid_suffixes, num_valid_suffixes, config_template)
        search.initialize()
        
        start_time = time.time()
        aborted = False
        
        # --- Level Loop ---
        for level in range(1, MAX_LENGTH + 1):
            success = search.process_level(level)
            
            if not success:
                print("  [Run Failed] No candidates remaining.")
                aborted = True
                break
            
            # 1. Check if we found a kernel element
            if len(search.kernel_braids) > 0:
                print(f"\n\n{'!'*60}")
                print(f"SUCCESS! Found {len(search.kernel_braids)} kernel elements at level {level}")
                print(f"Run #{run_count} succeeded.")
                print(f"{'!'*60}\n")
                
                # Verify and Print
                for tensor_word in search.kernel_braids:
                    word_list = [w.item() for w in tensor_word]
                    # Strip trailing zeros/padding if necessary
                    while word_list and word_list[-1] == 0:
                        word_list.pop()
                    
                    print(f"Found word: {word_list}")
                    is_kernel, msg = verify_kernel_element(word_list, n=N, r=R, p=P)
                    print(f"Verification: {msg}")
                
                print(f"Total time elapsed: {time.time() - total_start_time:.2f}s")
                return # EXIT SCRIPT SUCCESSFULLY
            
            # 2. Check the "Give Up" Condition
            if level == CHECK_LEVEL:
                if not search.buckets:
                    aborted = True
                    break
                    
                min_projlen = min(search.buckets.keys())
                print(f"\n  [CHECKPOINT @ L{level}] Min Projlen: {min_projlen} (Cutoff: < {PROJLEN_CUTOFF})")
                
                if min_projlen >= PROJLEN_CUTOFF:
                    print(f"  ❌ ABORTING RUN: {min_projlen} >= {PROJLEN_CUTOFF}")
                    print(f"  Restarting...")
                    aborted = True
                    break
                else:
                    print(f"  ✅ CONTINUING: {min_projlen} is promising.")

        # Cleanup between runs to prevent OOM
        del search
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if not aborted and run_count > 10000:
             # Safety break just in case
             print("Hit safety limit of 10000 runs.")
             break

if __name__ == "__main__":
    try:
        run_auto_search()
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user.")