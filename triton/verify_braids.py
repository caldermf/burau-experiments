#!/usr/bin/env python3
"""
Verify reported projlen-0 braids by computing matrix products from Garside words.
"""

def get_suffix_matrices():
    """Return the 22 suffix matrices as polynomial matrices over F_7."""
    m = [None] * 22
    m[0] = [[[], [(3,1)], [(2,1)]], [[], [(4,-1)], []], [[(2,1)], [(3,1)], []]]
    m[1] = [[[(2,-1)], [(1,-1)], []], [[], [(0,1)], []], [[], [(1,-1)], [(2,-1)]]]
    m[2] = [[[(0,1)], [], []], [[(1,-1)], [(2,-1)], [(1,-1)]], [[(2,1)], [(3,1)], []]]
    m[3] = [[[], [(3,1)], [(2,1)]], [[(1,-1)], [(2,-1)], [(1,-1)]], [[], [], [(0,1)]]]
    m[4] = [[[], [], [(4,-1)]], [[(3,1)], [(2,1)], [(3,1)]], [[], [(1,-1)], [(2,-1)]]]
    m[5] = [[[(2,-1)], [(1,-1)], []], [[(3,1)], [(2,1)], [(3,1)]], [[(4,-1)], [], []]]
    m[6] = [[[(0,1)], [], []], [[(1,-1)], [], [(3,1)]], [[], [(1,-1)], [(2,-1)]]]
    m[7] = [[[], [(3,1)], [(2,1)]], [[(3,1)], [], [(1,-1)]], [[(4,-1)], [], []]]
    m[8] = [[[], [], [(4,-1)]], [[(1,-1)], [], [(3,1)]], [[(2,1)], [(3,1)], []]]
    m[9] = [[[(2,-1)], [(1,-1)], []], [[(3,1)], [], [(1,-1)]], [[], [], [(0,1)]]]
    m[10] = [[[(0,1)], [], []], [[], [(0,1)], []], [[], [(1,-1)], [(2,-1)]]]
    m[11] = [[[], [(3,1)], [(2,1)]], [[], [(4,-1)], []], [[(4,-1)], [], []]]
    m[12] = [[[], [], [(4,-1)]], [[], [(4,-1)], []], [[(2,1)], [(3,1)], []]]
    m[13] = [[[(2,-1)], [(1,-1)], []], [[], [(0,1)], []], [[], [], [(0,1)]]]
    m[14] = [[[(0,1)], [], []], [[(1,-1)], [(2,-1)], [(1,-1)]], [[], [], [(0,1)]]]
    m[15] = [[[], [(3,1)], [(2,1)]], [[(1,-1)], [(2,-1)], [(1,-1)]], [[(2,1)], [(3,1)], []]]
    m[16] = [[[], [], [(4,-1)]], [[(3,1)], [(2,1)], [(3,1)]], [[(4,-1)], [], []]]
    m[17] = [[[(2,-1)], [(1,-1)], []], [[(3,1)], [(2,1)], [(3,1)]], [[], [(1,-1)], [(2,-1)]]]
    m[18] = [[[(0,1)], [], []], [[(1,-1)], [], [(3,1)]], [[(2,1)], [(3,1)], []]]
    m[19] = [[[], [(3,1)], [(2,1)]], [[(3,1)], [], [(1,-1)]], [[], [], [(0,1)]]]
    m[20] = [[[], [], [(4,-1)]], [[(1,-1)], [], [(3,1)]], [[], [(1,-1)], [(2,-1)]]]
    m[21] = [[[(2,-1)], [(1,-1)], []], [[(3,1)], [], [(1,-1)]], [[(4,-1)], [], []]]
    return m


def sparse_to_dense(entries, size=256):
    """Convert [(deg, coeff), ...] to dense polynomial list mod 7."""
    poly = [0] * size
    for deg, coeff in entries:
        poly[deg] = coeff % 7
    return poly


def poly_add(a, b):
    n = max(len(a), len(b))
    r = [0] * n
    for i in range(len(a)):
        r[i] = (r[i] + a[i]) % 7
    for i in range(len(b)):
        r[i] = (r[i] + b[i]) % 7
    return r


def poly_mul(a, b):
    # Trim trailing zeros for efficiency
    a = list(a)
    b = list(b)
    while a and a[-1] == 0:
        a.pop()
    while b and b[-1] == 0:
        b.pop()
    if not a or not b:
        return [0]
    n = len(a) + len(b) - 1
    r = [0] * n
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            if bj == 0:
                continue
            r[i + j] = (r[i + j] + ai * bj) % 7
    return r


def mat_mul(A, B):
    """Multiply two 3x3 polynomial matrices mod 7."""
    C = [[[0] for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            acc = [0]
            for k in range(3):
                prod = poly_mul(A[i][k], B[k][j])
                acc = poly_add(acc, prod)
            C[i][j] = acc
    return C


def poly_degree_range(poly):
    """Return (min_deg, max_deg) of nonzero coefficients, or None if zero."""
    min_d = None
    max_d = None
    for i, c in enumerate(poly):
        if c != 0:
            if min_d is None:
                min_d = i
            max_d = i
    return min_d, max_d


def matrix_projlen(M):
    """Compute projlen = max_degree - min_degree across all entries."""
    global_min = None
    global_max = None
    for i in range(3):
        for j in range(3):
            lo, hi = poly_degree_range(M[i][j])
            if lo is not None:
                if global_min is None or lo < global_min:
                    global_min = lo
                if global_max is None or hi > global_max:
                    global_max = hi
    if global_min is None:
        return 0, True  # zero matrix
    return global_max - global_min, False


def poly_to_str(poly, var="v"):
    terms = []
    for deg, coeff in enumerate(poly):
        if coeff == 0:
            continue
        if deg == 0:
            terms.append(str(coeff))
        elif coeff == 1:
            terms.append(f"{var}^{deg}" if deg > 1 else var)
        else:
            terms.append(f"{coeff}{var}^{deg}" if deg > 1 else f"{coeff}{var}")
    return " + ".join(terms) if terms else "0"


def compute_braid_matrix(word_0indexed, suffix_matrices_dense):
    """Compute the product of suffix matrices for a Garside word."""
    # Start with the first suffix
    result = suffix_matrices_dense[word_0indexed[0]]
    for idx in word_0indexed[1:]:
        result = mat_mul(result, suffix_matrices_dense[idx])
    return result


def main():
    raw = get_suffix_matrices()
    # Convert to dense polynomial matrices
    dense = []
    for s in range(22):
        M = [[sparse_to_dense(raw[s][i][j]) for j in range(3)] for i in range(3)]
        dense.append(M)

    # The reported braids (0-indexed words)
    braids = [
        [16, 2, 17, 8, 5, 1, 2, 17, 7, 0, 9, 3, 0, 17, 16, 8, 14, 17, 8, 5, 16, 7, 14, 0, 17, 1, 8, 0, 0, 9, 3, 17, 16, 16, 7, 12, 11, 4, 16, 2, 0, 17, 16, 16, 16, 16, 8, 0, 6, 2, 0, 0, 0, 0, 0, 17, 16, 16, 8, 11, 0, 0],
        [16, 16, 8, 17, 8, 0, 6, 2, 17, 15, 17, 16, 1, 7, 4, 1, 3, 14, 17, 16, 7, 14, 0, 9, 3, 17, 16, 8, 17, 7, 12, 17, 16, 7, 17, 8, 0, 0, 9, 3, 0, 17, 16, 7, 0, 17, 15, 0, 17, 1, 16, 16, 7, 12, 17, 16, 16, 8, 11, 12, 11, 0],
    ]

    for bi, word in enumerate(braids):
        print(f"\n{'='*70}")
        print(f"Braid #{bi+1}, length {len(word)}")
        print(f"Word (0-indexed): {word}")
        print()

        # Compute incrementally, printing projlen at each step
        result = dense[word[0]]
        for step in range(1, len(word)):
            result = mat_mul(result, dense[word[step]])
            
            # Print projlen every 10 steps and at the end
            if (step + 1) % 10 == 0 or step == len(word) - 1:
                pl, is_zero = matrix_projlen(result)
                print(f"  After {step+1:3d} suffixes: projlen = {pl:3d}  (zero={is_zero})")

        print()
        pl, is_zero = matrix_projlen(result)
        print(f"  FINAL projlen: {pl}")
        print(f"  Is zero matrix: {is_zero}")
        
        # Show nonzero entries
        print(f"  Nonzero matrix entries:")
        for i in range(3):
            for j in range(3):
                s = poly_to_str(result[i][j])
                if s != "0":
                    print(f"    [{i}][{j}] = {s}")

        # Also check: what does projlen look like if we only use 128 coeffs?
        # (the kernel only has 128 coefficients / 2 uint64 per plane)
        global_min_128 = None
        global_max_128 = None
        has_overflow = False
        for i in range(3):
            for j in range(3):
                for k, c in enumerate(result[i][j]):
                    if c != 0:
                        if k >= 128:
                            has_overflow = True
                        if global_min_128 is None or k < global_min_128:
                            global_min_128 = k
                        if global_max_128 is None or k > global_max_128:
                            global_max_128 = k
        
        if has_overflow:
            print(f"\n  *** OVERFLOW: Polynomial has coefficients at degree >= 128! ***")
            print(f"  *** Max degree seen: {global_max_128} ***")
            print(f"  *** The kernel only has 128 coefficient slots (0-127)! ***")
            print(f"  *** High-degree terms are silently LOST, corrupting the matrix! ***")


if __name__ == "__main__":
    main()
