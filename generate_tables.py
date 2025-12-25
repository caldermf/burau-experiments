"""
Script to generate precomputed multiplication tables and Garside suffix tables
from the peyl library, for use in GPU-accelerated braid group computations.

This generates:
1. simple_burau: (n!, dim, dim, degree_window) - Burau matrices for all simples
2. valid_suffixes: (n!, max_suffixes) - Which simples can follow each simple
3. num_valid_suffixes: (n!,) - Count of valid suffixes for each simple

Usage:
    python generate_tables.py

The tables will be saved as .pt files (PyTorch tensors).
"""

import numpy as np
import torch
from pathlib import Path

# Import peyl modules
from peyl.braid import GNF, PermTable
from peyl.jonesrep import JonesCellRep
from peyl import polymat


def generate_tables(
    n: int = 4,           # Number of strands (B_4 has 4! = 24 simples)
    r: int = 1,           # Partition parameter: (n-r, r) = (3, 1) for n=4, r=1
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
    print(f"Generating tables for B_{n} in representation ({n-r}, {r}) mod {p if p > 0 else 'Z'}")
    print(f"Degree window: {degree_window}")
    
    # Create the permutation table and representation
    perm_table = PermTable.create(n)
    rep = JonesCellRep(n=n, r=r, p=p)
    
    # Basic info
    num_simples = perm_table.order  # n! = 24 for n=4
    dim = rep.dimension()           # Dimension of representation
    center = degree_window // 2     # Where degree 0 sits in the window
    
    print(f"Number of simples (n!): {num_simples}")
    print(f"Representation dimension: {dim}")
    print(f"Identity index: {perm_table.id}")
    print(f"Delta index: {perm_table.D}")
    
    # =========================================================================
    # 1. Generate simple_burau: Burau matrices for all simples
    # =========================================================================
    print("\n1. Generating Burau matrices for all simples...")
    
    simple_burau = torch.zeros(num_simples, dim, dim, degree_window, dtype=torch.long)
    
    for s in range(num_simples):
        # Get the polymat (numpy array) for this simple
        # Shape: (dim, dim, num_degrees)
        mat = rep._polymat_braid_factor(GNF, s)
        
        # mat has shape (dim, dim, L) where L is the number of nonzero degrees
        # The valuation (lowest degree) can be extracted
        mat_trimmed = polymat.trim(mat)
        val, mat_shifted = polymat.trim_left(mat_trimmed)
        
        # mat_shifted now has the polynomial starting from degree 0
        # We need to place it at the right offset in our window
        num_coeffs = mat_shifted.shape[-1]
        
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
    
    print(f"  simple_burau shape: {simple_burau.shape}")
    
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
    print("\n2. Generating Garside suffix tables...")
    
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
    
    print(f"  valid_suffixes shape: {valid_suffixes.shape}")
    print(f"  num_valid_suffixes shape: {num_valid_suffixes.shape}")
    
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
    print(f"\n3. Saved tables to: {save_path}")
    
    return tables


def print_simple_info(tables: dict, simple_idx: int):
    """Debug helper: print info about a specific simple."""
    print(f"\nSimple {simple_idx}:")
    print(f"  Burau matrix (nonzero degrees):")
    mat = tables['simple_burau'][simple_idx]
    nonzero_degs = torch.where(mat.abs().sum(dim=(0, 1)) > 0)[0]
    center = tables['center']
    if len(nonzero_degs) > 0:
        for d in nonzero_degs:
            actual_deg = d.item() - center
            print(f"    Degree {actual_deg}:")
            print(mat[:, :, d])
    else:
        print("    (zero matrix)")
    
    print(f"  Valid suffixes ({tables['num_valid_suffixes'][simple_idx].item()}):")
    n_valid = tables['num_valid_suffixes'][simple_idx].item()
    suffixes = tables['valid_suffixes'][simple_idx, :n_valid].tolist()
    print(f"    {suffixes}")


def verify_multiplication(tables: dict, n: int, r: int, p: int):
    """Verify that the table entries are correct by comparing with peyl."""
    from peyl.braid import GNF, PermTable
    from peyl.jonesrep import JonesCellRep
    from peyl import polymat
    
    perm_table = PermTable.create(n)
    rep = JonesCellRep(n=n, r=r, p=p)
    
    print("\nVerification: comparing table entries with peyl evaluation...")
    
    # Test a few random simples
    test_indices = [0, 1, perm_table.id, perm_table.D, 5, 10]
    test_indices = [i for i in test_indices if i < tables['num_simples']]
    
    center = tables['center']
    dim = tables['dim']
    
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
            assert peyl_mat.shape[-1] == 0 or np.all(peyl_mat == 0), f"Simple {s} mismatch"
            print(f"  Simple {s}: ✓ (zero matrix)")
            continue
        
        table_start = np.argmax(nonzero_mask)
        table_end = len(nonzero_mask) - np.argmax(nonzero_mask[::-1])
        
        # The table stores degree d at index center + d
        # So actual degrees are [table_start - center, table_end - center)
        table_coeffs = table_mat[:, :, table_start:table_end]
        
        # peyl_mat starts at some valuation
        val, peyl_shifted = polymat.trim_left(peyl_mat)
        
        # They should match
        if table_coeffs.shape == peyl_shifted.shape and np.allclose(table_coeffs, peyl_shifted):
            print(f"  Simple {s}: ✓")
        else:
            print(f"  Simple {s}: ✗")
            print(f"    Table shape: {table_coeffs.shape}, Peyl shape: {peyl_shifted.shape}")
    
    print("Verification complete.")


# =========================================================================
# Example usage
# =========================================================================
if __name__ == "__main__":
    # Generate tables for B_4 with representation mod 2
    tables = generate_tables(
        n=4,
        r=1,
        p=2,
        degree_window=64,
        output_dir="./precomputed_tables"
    )
    
    # Print info about a few simples
    print_simple_info(tables, tables['id_index'])  # Identity
    print_simple_info(tables, 1)                    # A nontrivial simple
    
    # Verify correctness
    verify_multiplication(tables, n=4, r=1, p=3)
    
    print("\n" + "="*60)
    print("DONE! Tables are ready for GPU use.")
    print("="*60)
    print(f"""
To use these tables in your GPU code:

    tables = torch.load('precomputed_tables/tables_B4_r1_p3.pt')
    
    simple_burau = tables['simple_burau']        # Shape: (24, 3, 3, 64)
    valid_suffixes = tables['valid_suffixes']    # Shape: (24, {tables['valid_suffixes'].shape[1]})
    num_valid_suffixes = tables['num_valid_suffixes']  # Shape: (24,)
    
    # Move to GPU
    simple_burau = simple_burau.to('cuda')
    valid_suffixes = valid_suffixes.to('cuda')
    num_valid_suffixes = num_valid_suffixes.to('cuda')
    
    # In your kernel, to multiply by simple s:
    #   result[i,j,d] = sum_k sum_e A[i,k,e] * simple_burau[s,k,j,d-e]
    # (with appropriate mod p handling)
""")
