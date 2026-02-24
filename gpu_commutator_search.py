#!/usr/bin/env python3
"""
GPU-accelerated commutator kernel search for [σ_i, g^{-1}] in B_4.

Self-contained PyTorch implementation. No Garside normal form computation
needed on GPU — we track commutator Burau matrices directly using the
update rule: C_{g·b} = T_b · C_g · M_b

where T_b = M_σ · M_b^{-1} · M_σ^{-1} is precomputed for all 24 simples.
"""

import torch
import numpy as np
import os, sys, time, argparse
from itertools import permutations


# ================================================================
# S_4 permutation utilities
# ================================================================

def all_perms_S4():
    return list(permutations(range(4)))

def left_descents(perm):
    return {i for i in range(len(perm) - 1) if perm[i] > perm[i + 1]}

def centralizer_simple_indices(gen_idx):
    """Indices of simples whose permutation is in the centralizer of σ_{gen_idx}.

    These commute with σ_{gen_idx} in B_4, so extending by them just
    conjugates the commutator matrix — provably redundant for kernel detection.
    """
    sigma_perm = {1: (1, 0, 2, 3), 2: (0, 2, 1, 3), 3: (0, 1, 3, 2)}[gen_idx]
    perms = all_perms_S4()
    centralizer = set()
    for i, p in enumerate(perms):
        # Check if p commutes with sigma_perm in S_4
        if compose_perm(sigma_perm, p) == compose_perm(p, sigma_perm):
            centralizer.add(i)
    return centralizer


def allowed_first_factors(gen_idx):
    """Which simples (1-22) can be first Garside factor for σ_{gen_idx}."""
    J = {1: {0, 2}, 2: {1}, 3: {0, 2}}[gen_idx]
    perms = all_perms_S4()
    return [i for i in range(1, 23) if left_descents(perms[i]).isdisjoint(J)]


def build_perm_table():
    """Build permutation table: perm_table[i] = permutation for simple i."""
    return all_perms_S4()


def compose_perm(a, b):
    """Compose permutations: (a ∘ b)(x) = a(b(x))."""
    return tuple(a[b[i]] for i in range(len(a)))


def perm_commutator_is_trivial(word, gen_idx):
    """Check if π([σ_i, g^{-1}]) = identity.

    If False, the braid commutator is definitely nontrivial.
    If True, it MIGHT be trivial (need peyl to confirm).
    """
    perms = all_perms_S4()
    sigma_perm = {1: (1, 0, 2, 3), 2: (0, 2, 1, 3), 3: (0, 1, 3, 2)}[gen_idx]

    # Compose permutations for g
    g_perm = (0, 1, 2, 3)  # identity
    for idx in word:
        g_perm = compose_perm(g_perm, perms[idx])

    # g_inv permutation
    g_inv = [0] * 4
    for i, v in enumerate(g_perm):
        g_inv[v] = i
    g_inv = tuple(g_inv)

    # σ_inv permutation
    s_inv = [0] * 4
    for i, v in enumerate(sigma_perm):
        s_inv[v] = i
    s_inv = tuple(s_inv)

    # [σ, g^{-1}] = σ ∘ g^{-1} ∘ σ^{-1} ∘ g
    result = compose_perm(sigma_perm, compose_perm(g_inv, compose_perm(s_inv, g_perm)))
    return result == (0, 1, 2, 3)


# ================================================================
# CPU polynomial matrix arithmetic (for precomputing T_b)
# ================================================================

def cpu_poly_matmul(A, B, p, out_D, center):
    """3x3 centered polynomial matrix multiply.
    A, B: (3,3,D) numpy. Degree 0 at index `center`.
    Returns (3,3,out_D) re-centered.
    """
    D = A.shape[-1]
    conv_len = 2 * D - 1
    C = np.zeros((3, 3, conv_len), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                C[i, j] += np.convolve(A[i, k], B[k, j])
    C %= p
    # Re-center: output[n] = conv[n + center]
    result = np.zeros((3, 3, out_D), dtype=np.int64)
    for n in range(out_D):
        src = n + center
        if 0 <= src < conv_len:
            result[:, :, n] = C[:, :, src]
    return result % p


def cpu_poly_mat_inv(M, p, center):
    """Invert 3x3 centered polynomial matrix mod p.
    Assumes det(M) is a monomial. Returns (3,3,D) centered."""
    D = M.shape[-1]

    def minor_2x2(r, c):
        rows = [i for i in range(3) if i != r]
        cols = [j for j in range(3) if j != c]
        a = np.convolve(M[rows[0], cols[0]], M[rows[1], cols[1]])
        b = np.convolve(M[rows[0], cols[1]], M[rows[1], cols[0]])
        L = max(len(a), len(b))
        return (np.pad(a, (0, L - len(a))) - np.pad(b, (0, L - len(b)))) % p

    cofactors = {}
    max_len = 0
    for i in range(3):
        for j in range(3):
            m = minor_2x2(i, j)
            if (i + j) % 2 == 1:
                m = (-m) % p
            cofactors[(i, j)] = m
            max_len = max(max_len, len(m))

    # Adjugate = transpose of cofactors
    adj = np.zeros((3, 3, max_len), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            c = cofactors[(j, i)]
            adj[i, j, :len(c)] = c

    # Determinant via first row
    det = np.zeros(max_len + D, dtype=np.int64)
    for j in range(3):
        term = np.convolve(M[0, j], cofactors[(0, j)])
        det[:len(term)] += term
    det %= p

    nonzero = np.where(det != 0)[0]
    assert len(nonzero) == 1, f"Det not monomial: nonzero at {nonzero.tolist()}"
    det_idx = int(nonzero[0])
    c_val = int(det[det_idx])
    c_inv = pow(c_val, p - 2, p)
    det_degree = det_idx - 3 * center

    # result[n] = adj[n + center + det_degree] * c_inv
    result = np.zeros((3, 3, D), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            for n in range(D):
                src = n + center + det_degree
                if 0 <= src < max_len:
                    result[i, j, n] = (adj[i, j, src] * c_inv) % p
    return result


def precompute_all(simple_burau_raw, raw_center, sigma_idx, p, center, D):
    """Compute centered simple Burau matrices and twisted matrices T_b.

    Returns: simple_centered (24,3,3,D), twisted (24,3,3,D) as numpy.
    """
    D_raw = simple_burau_raw.shape[-1]

    # Map raw to our centered representation
    # Raw: index d corresponds to degree (d - raw_center)
    # Ours: index n corresponds to degree (n - center)
    # So: n = d - raw_center + center
    simple = np.zeros((24, 3, 3, D), dtype=np.int64)
    for s in range(24):
        for d in range(D_raw):
            n = d - raw_center + center
            if 0 <= n < D:
                simple[s, :, :, n] = simple_burau_raw[s, :, :, d]

    # Verify identity at center
    assert simple[0, 0, 0, center] == 1, "Identity check failed"
    assert simple[0, 1, 1, center] == 1, "Identity check failed"
    assert simple[0, 2, 2, center] == 1, "Identity check failed"

    M_sigma_inv = cpu_poly_mat_inv(simple[sigma_idx], p, center)

    twisted = np.zeros((24, 3, 3, D), dtype=np.int64)
    for b in range(24):
        M_b_inv = cpu_poly_mat_inv(simple[b], p, center)
        inter = cpu_poly_matmul(simple[sigma_idx], M_b_inv, p, D, center)
        twisted[b] = cpu_poly_matmul(inter, M_sigma_inv, p, D, center)
        if b % 6 == 0:
            print(f"  T_{b}/24 ...", end="\r")

    print(f"  Computed all 24 twisted matrices.    ")
    return simple, twisted


# ================================================================
# GPU batch operations
# ================================================================

def next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def gpu_poly_matmul(A, B, p, out_D, center, chunk_size=10000):
    """Batched 3x3 centered polynomial matrix multiply on GPU.

    A, B: (N,3,3,D) long tensors on GPU.
    Returns: (N,3,3,out_D) long tensor, re-centered and mod p.
    """
    N = A.shape[0]
    if N == 0:
        return torch.zeros(0, 3, 3, out_D, dtype=torch.long, device=A.device)

    D = A.shape[-1]
    fft_n = next_pow2(2 * D - 1)

    results = []
    for i in range(0, N, chunk_size):
        j = min(i + chunk_size, N)
        a_fft = torch.fft.rfft(A[i:j].float(), n=fft_n, dim=-1)
        b_fft = torch.fft.rfft(B[i:j].float(), n=fft_n, dim=-1)
        c_fft = torch.einsum('bikf,bkjf->bijf', a_fft, b_fft)
        c = torch.fft.irfft(c_fft, n=fft_n, dim=-1)
        # Re-center: take [center, center + out_D)
        c = c[..., center:center + out_D].round().long() % p
        results.append(c)
    return torch.cat(results, dim=0)


def compute_projlen_batch(matrices):
    """Compute projective length for batch. Returns (N,) long tensor."""
    nonzero = (matrices != 0).any(dim=1).any(dim=1)  # (N, D)
    D = matrices.shape[-1]
    idx = torch.arange(D, device=matrices.device).unsqueeze(0)
    min_deg = torch.where(nonzero, idx, D).min(dim=-1).values
    max_deg = torch.where(nonzero, idx, -1).max(dim=-1).values
    pl = max_deg - min_deg + 1
    return torch.where(max_deg >= 0, pl, torch.zeros_like(pl))


def is_scalar_matrix_batch(matrices):
    """Check if matrices are scalar * I or scalar * anti-diag. Returns (N,) bool."""
    # Scalar identity check
    diag_eq = (matrices[:, 0, 0] == matrices[:, 1, 1]).all(-1) & \
              (matrices[:, 0, 0] == matrices[:, 2, 2]).all(-1)
    offdiag_zero = True
    for i in range(3):
        for j in range(3):
            if i != j:
                offdiag_zero = offdiag_zero & (matrices[:, i, j] == 0).all(-1)
    scalar_id = diag_eq & offdiag_zero

    # Anti-diagonal check (odd Delta powers)
    anti_eq = (matrices[:, 0, 2] == matrices[:, 1, 1]).all(-1) & \
              (matrices[:, 0, 2] == matrices[:, 2, 0]).all(-1)
    anti_zero = True
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2)]:
        anti_zero = anti_zero & (matrices[:, i, j] == 0).all(-1)

    return (scalar_id | (anti_eq & anti_zero))


# ================================================================
# Main search
# ================================================================

def run_search(p, gen_idx, max_length, bucket_size, use_best,
               device, degree_multiplier, matmul_chunk):

    # --- Load tables ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    table_path = None
    for path in [
        os.path.join(script_dir, 'precomputed_tables', f'tables_B4_r1_p{p}.pt'),
        os.path.join(os.path.dirname(script_dir), 'precomputed_tables', f'tables_B4_r1_p{p}.pt'),
        os.path.join(script_dir, 'src', 'precomputed_tables', f'tables_B4_r1_p{p}.pt'),
    ]:
        if os.path.exists(path):
            table_path = path
            break
    if table_path is None:
        print(f"ERROR: No table file found for p={p}")
        return []

    print(f"Loading tables from {table_path}")
    tables = torch.load(table_path, weights_only=True)
    simple_burau_raw = tables['simple_burau'].numpy()
    rc = tables.get('center', 0)
    raw_center = rc.item() if hasattr(rc, 'item') else int(rc)
    valid_suffixes = tables['valid_suffixes']
    num_valid_suffixes = tables['num_valid_suffixes']
    id_val = tables['id_index']
    id_idx = id_val.item() if hasattr(id_val, 'item') else int(id_val)
    delta_val = tables['delta_index']
    delta_idx = delta_val.item() if hasattr(delta_val, 'item') else int(delta_val)
    valid_suffixes[id_idx] = valid_suffixes[delta_idx]
    num_valid_suffixes[id_idx] = num_valid_suffixes[delta_idx]

    # --- Degree window ---
    center = degree_multiplier * max_length + 15
    D = 2 * center + 1
    sigma_simple = {1: 6, 2: 2, 3: 1}[gen_idx]
    allowed = allowed_first_factors(gen_idx)

    print(f"Precomputing twisted matrices (center={center}, D={D}, raw_center={raw_center})...")
    simple_np, twisted_np = precompute_all(simple_burau_raw, raw_center, sigma_simple, p, center, D)

    # --- Move to GPU ---
    simple_gpu = torch.tensor(simple_np, dtype=torch.long, device=device)
    twisted_gpu = torch.tensor(twisted_np, dtype=torch.long, device=device)
    valid_suffixes = valid_suffixes.to(device)
    num_valid_suffixes = num_valid_suffixes.to(device)
    max_suf = valid_suffixes.shape[1]

    # Centralizer simples to exclude from ALL extensions (not just first factor)
    cent_set = centralizer_simple_indices(gen_idx)
    # Always exclude identity (0) and Delta (23); also exclude centralizer simples
    excluded = cent_set | {0, 23}
    excluded_tensor = torch.tensor(sorted(excluded), dtype=torch.long, device=device)
    cent_display = sorted(cent_set - {0, 23})  # non-trivial centralizer simples

    print(f"\n{'='*60}")
    print(f"COMMUTATOR SEARCH: [σ_{gen_idx}, g^{{-1}}] for p={p}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Degree window: D={D} (center={center})")
    print(f"Bucket size: {bucket_size}")
    print(f"Use best: {use_best if use_best > 0 else 'all'}")
    print(f"Max length: {max_length}")
    print(f"Allowed first factors: {len(allowed)}")
    print(f"Centralizer simples excluded from extensions: {cent_display}")
    print(f"Matmul chunk: {matmul_chunk}")
    print()

    # --- Level 1: initial commutator matrices ---
    init = torch.tensor(allowed, dtype=torch.long, device=device)
    N = len(init)
    T_init = twisted_gpu[init]
    M_init = simple_gpu[init]
    matrices = gpu_poly_matmul(T_init, M_init, p, D, center, matmul_chunk)
    del T_init, M_init

    words = torch.zeros(N, max_length, dtype=torch.long, device=device)
    words[:, 0] = init
    last_factors = init.clone()

    projlens = compute_projlen_batch(matrices)
    print(f"Level 1: {N} braids, projlen [{projlens.min().item()}, {projlens.max().item()}]")

    kernel_words = []

    # --- Main BFS ---
    for level in range(2, max_length + 1):
        t0 = time.time()
        N = matrices.shape[0]
        if N == 0:
            print("Empty! Done.")
            break

        # Select braids to expand (prioritize low projlen)
        if use_best > 0 and N > use_best:
            _, idx = projlens.sort()
            sel = idx[:use_best]
            matrices = matrices[sel]
            words = words[sel]
            last_factors = last_factors[sel]
            projlens = projlens[sel]
            N = use_best

        # Build expansion indices
        ns = num_valid_suffixes[last_factors]  # (N,)
        suf_mask = torch.arange(max_suf, device=device).unsqueeze(0) < ns.unsqueeze(1)
        suf_vals = valid_suffixes[last_factors]  # (N, max_suf)
        # Exclude identity, Delta, AND centralizer simples
        for ex in excluded_tensor:
            suf_mask &= (suf_vals != ex)

        braid_idx, suf_pos = torch.where(suf_mask)
        chosen = suf_vals[braid_idx, suf_pos]
        M_exp = len(braid_idx)

        if M_exp == 0:
            print(f"Level {level}: no expansions!")
            break

        # Gather and compute T · C · M in two steps
        parent = matrices[braid_idx]
        T = twisted_gpu[chosen]
        M = simple_gpu[chosen]

        intermediate = gpu_poly_matmul(T, parent, p, D, center, matmul_chunk)
        new_matrices = gpu_poly_matmul(intermediate, M, p, D, center, matmul_chunk)
        del intermediate, T, parent, M

        # Build new words
        new_words = words[braid_idx].clone()
        if level - 1 < max_length:
            new_words[:, level - 1] = chosen
        new_last = chosen

        # Projlens
        new_pl = compute_projlen_batch(new_matrices)

        # Check for kernel elements (projlen <= 1, nonzero, scalar form)
        # IMPORTANT: Exclude identity matrices — these are trivial commutators
        # (g centralizes σ_i, so [σ_i, g^{-1}] = 1). A genuine kernel element
        # maps to c·v^k · I with k ≠ 0 (some power of Δ²).
        nonzero = (new_matrices != 0).any(dim=-1).any(dim=-1).any(dim=-1)
        kernel_cand = (new_pl <= 1) & nonzero
        n_cand_this_level = 0
        n_trivial_this_level = 0
        if kernel_cand.any():
            cand_idx = kernel_cand.nonzero(as_tuple=True)[0]
            scalar = is_scalar_matrix_batch(new_matrices[cand_idx])
            for ci, is_s in zip(cand_idx, scalar):
                if is_s:
                    # Check if this is the identity matrix (trivial commutator)
                    mat = new_matrices[ci]
                    is_identity = (mat[0, 0, center] == 1) and \
                                  (mat[0, 0].sum() == 1)  # only one nonzero coeff at center
                    if is_identity:
                        n_trivial_this_level += 1
                        continue
                    w = new_words[ci].cpu().tolist()
                    w = [x for x in w if x != 0]
                    if len(w) > 0:
                        kernel_words.append(w)
                        n_cand_this_level += 1

        # Don't expand braids that already reached projlen <= 1
        # (they're either trivial commutators or kernel elements — dead ends either way)
        keep = new_pl > 1
        new_matrices = new_matrices[keep]
        new_words = new_words[keep]
        new_last = new_last[keep]
        new_pl = new_pl[keep]

        # Reservoir sampling by projlen
        unique_pls = new_pl.unique()
        keep_mask = torch.zeros(len(new_pl), dtype=torch.bool, device=device)
        for pl_val in unique_pls:
            pl_mask = new_pl == pl_val
            count = pl_mask.sum().item()
            if count <= bucket_size:
                keep_mask |= pl_mask
            else:
                indices = pl_mask.nonzero(as_tuple=True)[0]
                perm = torch.randperm(count, device=device)[:bucket_size]
                keep_mask[indices[perm]] = True

        matrices = new_matrices[keep_mask]
        words = new_words[keep_mask]
        last_factors = new_last[keep_mask]
        projlens = new_pl[keep_mask]
        del new_matrices, new_words, new_last, new_pl

        elapsed = time.time() - t0
        N_kept = matrices.shape[0]
        n_bk = unique_pls.numel()
        min_pl = projlens.min().item() if N_kept > 0 else -1
        max_pl = projlens.max().item() if N_kept > 0 else -1

        # Report
        cand_str = f"  +{n_cand_this_level} KERNEL" if n_cand_this_level > 0 else ""
        triv_str = f"  ({n_trivial_this_level} trivial)" if n_trivial_this_level > 0 else ""
        line = f"Level {level}: {M_exp} exp -> {N_kept} kept ({n_bk} bk)  [{elapsed:.1f}s]{cand_str}{triv_str}"
        print(line)
        if N_kept > 0:
            sorted_pls = sorted(unique_pls.tolist())
            for pl_val in sorted_pls[:3]:
                print(f"  pl={int(pl_val)}: {(projlens == pl_val).sum().item()}")
            if n_bk > 6:
                print(f"  ... ({n_bk - 6} more) ...")
            if n_bk > 3:
                for pl_val in sorted_pls[-min(3, n_bk - 3):]:
                    print(f"  pl={int(pl_val)}: {(projlens == pl_val).sum().item()}")
            print(f"  Range: [{min_pl}, {max_pl}], expected ~{4*level+1}")

        if device == "cuda":
            torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Search complete. {len(kernel_words)} kernel candidates (projlen<=1, scalar matrix).")
    print(f"Centralizer simples {cent_display} excluded from all extensions.")

    if kernel_words:
        print(f"\nKERNEL CANDIDATES (need peyl verification):")
        for i, w in enumerate(kernel_words[:20]):
            print(f"  #{i+1}: factors({len(w)})={w[:15]}{'...' if len(w) > 15 else ''}")
        if len(kernel_words) > 20:
            print(f"  ... and {len(kernel_words) - 20} more")

    return kernel_words


# ================================================================
# CLI
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GPU-accelerated commutator kernel search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --p 5 --gen 1 --bucket-size 50000 --use-best 50000 --max-length 65 --device cuda
  %(prog)s --p 5 --gen 1 --bucket-size 5000 --max-length 40 --device cpu  # Quick test
""")
    parser.add_argument('--p', type=int, default=5)
    parser.add_argument('--gen', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--bucket-size', type=int, default=50000)
    parser.add_argument('--use-best', type=int, default=0,
                        help='Expand only this many braids per level (0=all)')
    parser.add_argument('--max-length', type=int, default=65)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--degree-multiplier', type=int, default=2)
    parser.add_argument('--matmul-chunk', type=int, default=10000)
    args = parser.parse_args()

    run_search(
        p=args.p, gen_idx=args.gen,
        max_length=args.max_length,
        bucket_size=args.bucket_size,
        use_best=args.use_best,
        device=args.device,
        degree_multiplier=args.degree_multiplier,
        matmul_chunk=args.matmul_chunk,
    )
