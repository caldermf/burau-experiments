#!/usr/bin/env python3
"""
Small test run for the GPU braid search.

This uses minimal parameters so it completes in under a minute,
letting you verify everything works before doing a full run.
"""

import torch
from new_braid_search import Config, BraidSearch, load_tables_from_file

def test_small():
    """Quick test run - should complete in under a minute."""
    
    config = Config(
        bucket_size=500,        # Keep only 500 braids per bucket (small for testing)
        max_length=6,           # Only go up to length 6 (quick)
        bootstrap_length=3,     # Exhaustive only for lengths 1-3
        prime=5,                # Mod 5
        degree_multiplier=4,    # Degree window = 4 * 6 * 2 + 1
        checkpoint_every=100,   # Don't bother checkpointing for this short run
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("="*60)
    print("SMALL TEST RUN")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Bucket size: {config.bucket_size}")
    print(f"Max length: {config.max_length}")
    print(f"Bootstrap length: {config.bootstrap_length}")
    print(f"Prime: {config.prime}")
    print(f"Degree window: {config.degree_window}")
    print()
    
    # Load tables - UPDATE THIS PATH to match your setup
    table_path = "/Users/com36/burau-experiments/precomputed_tables/tables_B4_r1_p5.pt"
    
    simple_burau, valid_suffixes, num_valid_suffixes = load_tables_from_file(
        config, 
        table_path=table_path
    )
    
    # Verify identity matrix
    center = config.degree_window // 2
    assert simple_burau[0, 0, 0, center] == 1, "Identity matrix check failed"
    assert simple_burau[0, 1, 1, center] == 1, "Identity matrix check failed"
    assert simple_burau[0, 2, 2, center] == 1, "Identity matrix check failed"
    print("✓ Identity matrix verified\n")
    
    # Run the search
    search = BraidSearch(simple_burau, valid_suffixes, num_valid_suffixes, config)
    zero_braids = search.run(checkpoint_dir=None)  # No checkpoints for quick test
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    
    if zero_braids:
        total = sum(len(w) for w in zero_braids)
        print(f"Found {total} braids with projlen=0")
    else:
        print("No projlen=0 braids found (expected at short lengths)")
    
    return search


def test_medium():
    """Medium test - a few minutes, might find something interesting."""
    
    config = Config(
        bucket_size=5000,       # More braids per bucket
        max_length=25,          # Go deeper
        bootstrap_length=4,     # Exhaustive for lengths 1-4
        prime=5,
        degree_multiplier=4,
        checkpoint_every=5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("="*60)
    print("MEDIUM TEST RUN")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Bucket size: {config.bucket_size}")
    print(f"Max length: {config.max_length}")
    print(f"Degree window: {config.degree_window}")
    print()
    
    table_path = "/Users/com36/burau-experiments/precomputed_tables/tables_B4_r1_p5.pt"
    
    simple_burau, valid_suffixes, num_valid_suffixes = load_tables_from_file(
        config, 
        table_path=table_path
    )
    
    search = BraidSearch(simple_burau, valid_suffixes, num_valid_suffixes, config)
    zero_braids = search.run(checkpoint_dir="checkpoints")
    
    return search


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "medium":
        test_medium()
    else:
        test_small()
        print("\nTo run a longer test: python test_run.py medium")
