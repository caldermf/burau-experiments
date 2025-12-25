    #!/usr/bin/env python3
"""
Find kernel elements for p=3 (should be quick!)

Based on the paper:
- p=2 finds kernel elements at length 8
- p=3 should find them around length 10-20
- p=5 finds them at length 65

This test should complete in 5-10 minutes on CPU and find actual kernel elements.
"""

import sys
import torch

# Add paths
sys.path.insert(0, '/Users/com36/burau-experiments')
sys.path.insert(0, '/Users/com36/burau-experiments/src')

from new_braid_search import Config, BraidSearch, load_tables_from_file

# For verification
from peyl.braid import GNF, PermTable, BraidGroup
from peyl.jonesrep import JonesCellRep


def verify_kernel_element(word_tensor, n=4, r=1, p=3):
    """
    Verify that a braid word is actually in the kernel using peyl.
    
    Args:
        word_tensor: tensor of simple indices
        
    Returns:
        True if in kernel (evaluates to scalar matrix), False otherwise
    """
    # Convert tensor to list of simple indices
    word = [w.item() for w in word_tensor if w.item() != 0 or len([w2 for w2 in word_tensor if w2.item() != 0]) == 0]
    
    # Remove trailing zeros (padding)
    while word and word[-1] == 0:
        word.pop()
    
    if not word:
        return False, "Empty word"
    
    # Create the braid using peyl
    perm_table = PermTable.create(n)
    
    # Check if this is a valid normal form
    # (In our search, we maintain normal form, so it should be)
    try:
        braid = GNF(n=n, power=0, factors=tuple(word))
    except AssertionError as e:
        return False, f"Invalid normal form: {e}"
    
    # Evaluate in the representation
    rep = JonesCellRep(n=n, r=r, p=p)
    result = rep.evaluate(braid)
    
    # Check if it's a scalar matrix (kernel element)
    # A scalar matrix has the form c * v^k * I for some c, k
    is_scalar = True
    diagonal_val = result[0, 0]
    
    for i in range(result.nrows):
        for j in range(result.ncols):
            if i == j:
                if result[i, j] != diagonal_val:
                    is_scalar = False
                    break
            else:
                if not result[i, j].is_zero():
                    is_scalar = False
                    break
        if not is_scalar:
            break
    
    if is_scalar:
        return True, f"Kernel element! Evaluates to {diagonal_val} * I"
    else:
        return False, f"Not in kernel"


def find_p3_kernel():
    """Search for p=2 kernel elements."""
    
    # Configuration tuned for p=3
    # Based on the paper, p=3 should be easier than p=5
    config = Config(
        bucket_size=20,       # Moderate bucket size
        max_length=10,          # Should find something by length 20
        bootstrap_length=3,     # Exhaustive for short lengths
        prime=2,                # p=3
        degree_multiplier=4,
        checkpoint_every=5,
        device="cpu"            # CPU for now
    )
    
    print("="*60)
    print("SEARCHING FOR p=3 KERNEL ELEMENTS")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Bucket size: {config.bucket_size}")
    print(f"Max length: {config.max_length}")
    print(f"Bootstrap length: {config.bootstrap_length}")
    print(f"Prime: {config.prime}")
    print(f"Degree window: {config.degree_window}")
    print()
    
    # Load tables for p=3
    table_path = "/Users/com36/burau-experiments/precomputed_tables/tables_B4_r1_p2.pt"
    
    try:
        simple_burau, valid_suffixes, num_valid_suffixes = load_tables_from_file(
            config, 
            table_path=table_path
        )
    except FileNotFoundError:
        print(f"ERROR: Table file not found at {table_path}")
        print("Please regenerate tables with p=3:")
        print("  1. Edit generate_tables.py to use p=3")
        print("  2. Run: python generate_tables.py")
        return None
    except AssertionError as e:
        print(f"ERROR: {e}")
        print("Please regenerate tables with p=3")
        return None
    
    # Verify identity matrix
    center = config.degree_window // 2
    assert simple_burau[0, 0, 0, center] == 1, "Identity matrix check failed"
    print("✓ Identity matrix verified\n")
    
    # Run the search
    search = BraidSearch(simple_burau, valid_suffixes, num_valid_suffixes, config)
    zero_braids = search.run(checkpoint_dir=None)
    
    # Verify any found braids
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    if not zero_braids:
        print("No projlen=0 braids found.")
        print("Try increasing max_length or bucket_size.")
        return None
    
    verified_count = 0
    for batch_idx, batch in enumerate(zero_braids):
        print(f"\nBatch {batch_idx}: {len(batch)} candidates")
        for i, word in enumerate(batch[:10]):  # Check first 10 per batch
            is_kernel, msg = verify_kernel_element(word, p=3)
            status = "✓ VERIFIED" if is_kernel else "✗"
            
            # Get the actual word (remove padding)
            word_list = [w.item() for w in word]
            while word_list and word_list[-1] == 0:
                word_list.pop()
            
            print(f"  Braid {i}: factors={word_list[:10]}{'...' if len(word_list) > 10 else ''}")
            print(f"    Length: {len(word_list)}, {status}: {msg}")
            
            if is_kernel:
                verified_count += 1
                
                # Print full details for kernel elements
                print(f"\n    🎉 KERNEL ELEMENT FOUND! 🎉")
                print(f"    Full Garside word: {word_list}")
                
                # Convert to GNF for nicer output
                perm_table = PermTable.create(4)
                braid = GNF(n=4, power=0, factors=tuple(word_list))
                print(f"    Canonical form: {braid}")
                print(f"    Artin word: {braid.magma_artin_word()}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: Verified {verified_count} kernel elements")
    print("="*60)
    
    return zero_braids


if __name__ == "__main__":
    find_p3_kernel()
