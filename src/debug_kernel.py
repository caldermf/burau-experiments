#!/usr/bin/env python3
"""
Diagnostic script to investigate why certain kernel elements aren't being found.

This script will check:
1. What is delta_index and id_index?
2. What simple braid is index 13?
3. Can 13 follow 13 in the valid suffix table?
4. What is the Burau matrix for simple 13?
5. What is [13,13,13,13] in the Burau representation?
6. Trace the path from identity - can we reach [13,13,13,13]?
"""

import torch
import os
import sys

def find_table_path(p=5):
    """Find the table file for prime p."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(script_dir, "precomputed_tables", f"tables_B4_r1_p{p}.pt"),
        os.path.join(os.path.dirname(script_dir), "precomputed_tables", f"tables_B4_r1_p{p}.pt"),
        os.path.join(script_dir, f"tables_B4_r1_p{p}.pt"),
        f"tables_B4_r1_p{p}.pt",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def main(p=5):
    print("=" * 70)
    print(f"DIAGNOSTIC: Investigating kernel element discovery for p={p}")
    print("=" * 70)
    
    # Load tables
    table_path = find_table_path(p)
    if table_path is None:
        print(f"ERROR: Could not find table file for p={p}")
        return
    
    print(f"\nLoading tables from: {table_path}")
    tables = torch.load(table_path, weights_only=True)
    
    # Basic info
    print(f"\n{'='*70}")
    print("BASIC TABLE INFO")
    print("=" * 70)
    print(f"Keys in table: {list(tables.keys())}")
    print(f"n = {tables['n']}")
    print(f"p = {tables['p']}")
    print(f"delta_index = {tables['delta_index']}")
    print(f"id_index = {tables['id_index']}")
    
    delta_idx = tables['delta_index']
    id_idx = tables['id_index']
    
    # Valid suffixes info
    print(f"\n{'='*70}")
    print("VALID SUFFIX TABLE STRUCTURE")
    print("=" * 70)
    valid_suffixes = tables['valid_suffixes']
    num_valid_suffixes = tables['num_valid_suffixes']
    
    print(f"valid_suffixes shape: {valid_suffixes.shape}")
    print(f"num_valid_suffixes shape: {num_valid_suffixes.shape}")
    
    # Check what can follow identity
    print(f"\n{'='*70}")
    print("WHAT CAN FOLLOW IDENTITY (index {id_idx})?")
    print("=" * 70)
    num_after_id = num_valid_suffixes[id_idx].item()
    suffixes_after_id = valid_suffixes[id_idx, :num_after_id].tolist()
    print(f"Number of valid suffixes after identity: {num_after_id}")
    print(f"Valid suffixes: {suffixes_after_id}")
    
    # Check what can follow delta
    print(f"\n{'='*70}")
    print(f"WHAT CAN FOLLOW DELTA (index {delta_idx})?")
    print("=" * 70)
    num_after_delta = num_valid_suffixes[delta_idx].item()
    suffixes_after_delta = valid_suffixes[delta_idx, :num_after_delta].tolist()
    print(f"Number of valid suffixes after delta: {num_after_delta}")
    print(f"Valid suffixes: {suffixes_after_delta}")
    
    # Check simple 13 specifically
    print(f"\n{'='*70}")
    print("INVESTIGATING SIMPLE INDEX 13")
    print("=" * 70)
    
    num_after_13 = num_valid_suffixes[13].item()
    suffixes_after_13 = valid_suffixes[13, :num_after_13].tolist()
    print(f"Number of valid suffixes after simple 13: {num_after_13}")
    print(f"Valid suffixes after 13: {suffixes_after_13}")
    print(f"Can 13 follow 13? {13 in suffixes_after_13}")
    
    # Check if 13 can follow identity
    print(f"\nCan 13 follow identity? {13 in suffixes_after_id}")
    
    # Trace path to [13, 13, 13, 13]
    print(f"\n{'='*70}")
    print("TRACING PATH TO [13, 13, 13, 13]")
    print("=" * 70)
    
    target = [13, 13, 13, 13]
    current = id_idx  # Start from identity
    path_valid = True
    
    for i, next_simple in enumerate(target):
        num_valid = num_valid_suffixes[current].item()
        valid = valid_suffixes[current, :num_valid].tolist()
        
        can_continue = next_simple in valid
        print(f"Step {i+1}: From {current}, can we go to {next_simple}? {can_continue}")
        print(f"         Valid suffixes from {current}: {valid}")
        
        if not can_continue:
            path_valid = False
            print(f"         ❌ PATH BLOCKED HERE!")
            break
        
        current = next_simple
    
    if path_valid:
        print(f"\n✓ Path [13, 13, 13, 13] is REACHABLE from identity")
    else:
        print(f"\n❌ Path [13, 13, 13, 13] is NOT REACHABLE from identity")
    
    # Check Burau matrices
    print(f"\n{'='*70}")
    print("BURAU MATRICES")
    print("=" * 70)
    
    simple_burau = tables['simple_burau']
    center = tables['center']
    print(f"simple_burau shape: {simple_burau.shape}")
    print(f"center (degree offset): {center}")
    
    # Show Burau matrix for identity
    print(f"\nBurau matrix for identity (index {id_idx}):")
    id_mat = simple_burau[id_idx]
    print_nonzero_matrix(id_mat, center)
    
    # Show Burau matrix for delta
    print(f"\nBurau matrix for delta (index {delta_idx}):")
    delta_mat = simple_burau[delta_idx]
    print_nonzero_matrix(delta_mat, center)
    
    # Show Burau matrix for simple 13
    print(f"\nBurau matrix for simple 13:")
    s13_mat = simple_burau[13]
    print_nonzero_matrix(s13_mat, center)
    
    # Compute [13, 13, 13, 13] product
    print(f"\n{'='*70}")
    print("COMPUTING BURAU MATRIX FOR [13, 13, 13, 13]")
    print("=" * 70)
    
    # We need to do polynomial matrix multiplication
    # For simplicity, let's just show the raw tensors
    print("(This would require polynomial matrix multiplication)")
    print("Each matrix entry is a polynomial in v, stored as coefficient arrays")
    
    # Show all 24 simples and their suffix counts
    print(f"\n{'='*70}")
    print("ALL 24 SIMPLES - SUFFIX COUNTS")
    print("=" * 70)
    for s in range(24):
        num = num_valid_suffixes[s].item()
        suffixes = valid_suffixes[s, :num].tolist()
        marker = ""
        if s == id_idx:
            marker = " <-- IDENTITY"
        elif s == delta_idx:
            marker = " <-- DELTA"
        print(f"Simple {s:2d}: {num:2d} valid suffixes: {suffixes}{marker}")


def print_nonzero_matrix(mat, center):
    """Print a polynomial matrix, showing only nonzero entries."""
    for i in range(3):
        row_strs = []
        for j in range(3):
            poly = mat[i, j, :]
            nonzero = torch.where(poly != 0)[0]
            if len(nonzero) == 0:
                row_strs.append("0")
            else:
                terms = []
                for idx in nonzero:
                    deg = idx.item() - center
                    coeff = poly[idx].item()
                    if deg == 0:
                        terms.append(f"{coeff}")
                    elif deg == 1:
                        terms.append(f"{coeff}v" if coeff != 1 else "v")
                    else:
                        terms.append(f"{coeff}v^{deg}" if coeff != 1 else f"v^{deg}")
                row_strs.append(" + ".join(terms))
        print(f"  [{', '.join(row_strs)}]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=5, help="Prime")
    args = parser.parse_args()
    
    main(args.p)