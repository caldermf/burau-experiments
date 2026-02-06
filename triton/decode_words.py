#!/usr/bin/env python3
"""
Decode and display Garside words from projlen-0 .pt files.

Usage:
    python decode_words.py projlen0_results/projlen0_length042.pt
    python decode_words.py projlen0_results/            # all files in dir
    python decode_words.py projlen0_results/ --verify   # also verify matrices

Suffix indices 0-21 correspond to the 22 proper nontrivial suffixes of the
Garside element Delta in B_4 (i.e., the elements of S_4 minus {id, w_0}).
The output uses 1-indexed labels (1-22) for readability.
"""

import torch
import sys
import os
import glob


def decode_poly_bitsliced(vals):
    """Decode 6 uint64 values back to polynomial coefficients mod 7."""
    poly = [0] * 128
    for i in range(128):
        word = 0 if i < 64 else 1
        bit = i if i < 64 else i - 64
        b0 = (vals[0 * 2 + word] >> bit) & 1
        b1 = (vals[1 * 2 + word] >> bit) & 1
        b2 = (vals[2 * 2 + word] >> bit) & 1
        poly[i] = b0 + 2 * b1 + 4 * b2
    return poly


def decode_matrix_bitsliced(vals):
    """Decode 54 uint64 values back to 3x3 polynomial matrix."""
    mat = [[[0] * 128 for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            offset = (i * 3 + j) * 6
            mat[i][j] = decode_poly_bitsliced(vals[offset:offset + 6])
    return mat


def poly_to_str(poly, var="t"):
    """Pretty-print a polynomial mod 7."""
    terms = []
    for deg, coeff in enumerate(poly):
        if coeff == 0:
            continue
        if deg == 0:
            terms.append(str(coeff))
        elif deg == 1:
            if coeff == 1:
                terms.append(var)
            else:
                terms.append(f"{coeff}{var}")
        else:
            if coeff == 1:
                terms.append(f"{var}^{deg}")
            else:
                terms.append(f"{coeff}{var}^{deg}")
    return " + ".join(terms) if terms else "0"


def matrix_projlen(mat):
    """Compute projlen = max_degree - min_degree across all entries."""
    all_degrees = []
    for i in range(3):
        for j in range(3):
            for k in range(128):
                if mat[i][j][k] != 0:
                    all_degrees.append(k)
    if not all_degrees:
        return 0, True  # zero matrix
    return max(all_degrees) - min(all_degrees), False


def is_zero_matrix(mat):
    """Check if matrix is identically zero."""
    for i in range(3):
        for j in range(3):
            for k in range(128):
                if mat[i][j][k] != 0:
                    return False
    return True


def process_file(filepath, verify=False):
    """Process one .pt file and print its contents."""
    info = torch.load(filepath, weights_only=True)
    
    data = info["data"]       # [N, 54] int64
    meta = info["meta"]       # [N] int32
    braid_length = info["braid_length"]
    
    has_words = "words" in info
    words = info.get("words", None)  # [N, braid_length] uint8 or None
    
    n = data.shape[0]
    
    print(f"\n{'='*70}")
    print(f"File: {os.path.basename(filepath)}")
    print(f"Braid length: {braid_length}")
    print(f"Number of braids: {n}")
    print(f"Words available: {'yes' if has_words else 'no'}")
    print(f"{'='*70}")
    
    for idx in range(n):
        suffix_idx = meta[idx].item() & 0xFF
        projlen = (meta[idx].item() >> 8) & 0x7F
        
        print(f"\n--- Braid #{idx + 1} ---")
        print(f"  Last suffix (0-indexed): {suffix_idx}")
        print(f"  ProjLen: {projlen}")
        
        if has_words and words is not None:
            word = words[idx, :braid_length].tolist()
            # Convert to 1-indexed for display
            word_1indexed = [s + 1 for s in word]
            print(f"  Garside word (1-indexed): {word_1indexed}")
            print(f"  Garside word (0-indexed): {word}")
        
        if verify:
            # Decode and display the matrix
            vals = data[idx].tolist()
            vals_unsigned = [v if v >= 0 else v + (1 << 64) for v in vals]
            mat = decode_matrix_bitsliced(vals_unsigned)
            
            pl, is_zero = matrix_projlen(mat)
            
            print(f"  Matrix is zero: {is_zero}")
            print(f"  Computed projlen: {pl}")
            
            if not is_zero:
                print(f"  Matrix entries (nonzero only):")
                for i in range(3):
                    for j in range(3):
                        ps = poly_to_str(mat[i][j])
                        if ps != "0":
                            print(f"    [{i}][{j}] = {ps}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python decode_words.py <file.pt or directory> [--verify]")
        print()
        print("Examples:")
        print("  python decode_words.py projlen0_results/projlen0_length042.pt")
        print("  python decode_words.py projlen0_results/")
        print("  python decode_words.py projlen0_results/ --verify")
        sys.exit(1)
    
    path = sys.argv[1]
    verify = "--verify" in sys.argv
    
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "projlen0_length*.pt")))
        if not files:
            print(f"No projlen0_length*.pt files found in {path}")
            sys.exit(1)
        for f in files:
            process_file(f, verify=verify)
    elif os.path.isfile(path):
        process_file(path, verify=verify)
    else:
        print(f"Not found: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
