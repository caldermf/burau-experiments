"""
Code generator: produces the full braid_search.py with hardcoded Triton kernels.
"""

def get_raw_matrix_data_pruned():
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

def get_adjacency():
    raw = {
        0: [0, 6, 9, 14, 17], 1: [1, 2, 3, 7, 8, 10, 13, 15, 16, 20, 21],
        2: [0, 6, 9, 14, 17], 3: [0, 6, 9, 14, 17],
        4: [1, 2, 3, 7, 8, 10, 13, 15, 16, 20, 21],
        5: [1, 2, 3, 7, 8, 10, 13, 15, 16, 20, 21],
        6: [2, 10, 21], 7: [0, 3, 4, 6, 9, 12, 13, 14, 17, 19, 20],
        8: [0, 2, 5, 6, 9, 10, 11, 14, 17, 18, 21],
        9: [3, 13, 20], 10: [2, 10, 21],
        11: [0, 3, 4, 6, 9, 12, 13, 14, 17, 19, 20],
        12: [0, 2, 5, 6, 9, 10, 11, 14, 17, 18, 21],
        13: [3, 13, 20], 14: [0, 6, 9, 14, 17],
        15: [0, 6, 9, 14, 17],
        16: [1, 2, 3, 7, 8, 10, 13, 15, 16, 20, 21],
        17: [1, 2, 3, 7, 8, 10, 13, 15, 16, 20, 21],
        18: [0, 2, 5, 6, 9, 10, 11, 14, 17, 18, 21],
        19: [0, 3, 4, 6, 9, 12, 13, 14, 17, 19, 20],
        20: [2, 10, 21], 21: [3, 13, 20]
    }
    return raw

def gen_shift(src, dst, deg, planes=3):
    """Generate code to shift src polynomial into dst by deg bits (128-bit shift)."""
    lines = []
    for p in range(planes):
        slo = f"{src}_p{p}_lo"
        shi = f"{src}_p{p}_hi"
        dlo = f"{dst}_p{p}_lo"
        dhi = f"{dst}_p{p}_hi"
        if deg == 0:
            lines.append(f"        {dlo} = {slo}")
            lines.append(f"        {dhi} = {shi}")
        else:
            mask = (1 << deg) - 1
            lines.append(f"        {dlo} = {slo} << {deg}")
            lines.append(f"        {dhi} = ({shi} << {deg}) | (({slo} >> {64 - deg}) & {mask})")
    return lines

def gen_negate(var):
    """Generate code to negate var in-place (mod 7)."""
    lines = []
    lines.append(f"        {var}_p0_lo, {var}_p1_lo, {var}_p2_lo = negate_mod7({var}_p0_lo, {var}_p1_lo, {var}_p2_lo)")
    lines.append(f"        {var}_p0_hi, {var}_p1_hi, {var}_p2_hi = negate_mod7({var}_p0_hi, {var}_p1_hi, {var}_p2_hi)")
    return lines

def gen_add_into(acc, src):
    """Generate code: acc = add_mod7(acc, src) for lo and hi."""
    lines = []
    lines.append(f"        {acc}_p0_lo, {acc}_p1_lo, {acc}_p2_lo = add_mod7({acc}_p0_lo, {acc}_p1_lo, {acc}_p2_lo, {src}_p0_lo, {src}_p1_lo, {src}_p2_lo)")
    lines.append(f"        {acc}_p0_hi, {acc}_p1_hi, {acc}_p2_hi = add_mod7({acc}_p0_hi, {acc}_p1_hi, {acc}_p2_hi, {src}_p0_hi, {src}_p1_hi, {src}_p2_hi)")
    return lines

def gen_assign(dst, src):
    """Assign src to dst."""
    lines = []
    for p in range(3):
        lines.append(f"        {dst}_p{p}_lo = {src}_p{p}_lo")
        lines.append(f"        {dst}_p{p}_hi = {src}_p{p}_hi")
    return lines

def gen_zero(var):
    """Set var to zero."""
    lines = []
    for p in range(3):
        lines.append(f"        {var}_p{p}_lo = ZERO")
        lines.append(f"        {var}_p{p}_hi = ZERO")
    return lines

def generate_suffix_code(suffix_id, mat):
    """
    Generate the multiplication code for one suffix.
    
    output[i][j] = sum_k parent[i][k] * suffix[k][j]
    
    Parent registers: m{i}{k}_p{plane}_{lo/hi}
    Output registers: o{i}{j}_p{plane}_{lo/hi}
    """
    lines = []
    
    # For each output row i (0..2):
    for i in range(3):
        # For each output col j (0..2):
        for j in range(3):
            # Collect terms: parent[i][k] * suffix[k][j] for k where suffix[k][j] is nonzero
            terms = []
            for k in range(3):
                entry = mat[k][j]  # suffix[k][j]
                if entry:
                    for (deg, coeff) in entry:
                        terms.append((k, deg, coeff))
            
            out_var = f"o{i}{j}"
            
            if not terms:
                # Zero output
                lines.extend(gen_zero(out_var))
            else:
                for t_idx, (k, deg, coeff) in enumerate(terms):
                    src_var = f"m{i}{k}"
                    tmp_var = f"t{i}{j}_{t_idx}"
                    
                    # Shift
                    lines.extend(gen_shift(src_var, tmp_var, deg))
                    
                    # Negate if needed
                    if coeff == -1:
                        lines.extend(gen_negate(tmp_var))
                    
                    # Accumulate
                    if t_idx == 0:
                        lines.extend(gen_assign(out_var, tmp_var))
                    else:
                        lines.extend(gen_add_into(out_var, tmp_var))
    
    return lines

def generate_full_switch():
    """Generate the complete if/elif switch for all 22 suffixes."""
    matrices = get_raw_matrix_data_pruned()
    all_lines = []
    
    for s in range(22):
        if s == 0:
            all_lines.append(f"    if SUFFIX_IDX == {s}:")
        else:
            all_lines.append(f"    elif SUFFIX_IDX == {s}:")
        
        suffix_lines = generate_suffix_code(s, matrices[s])
        all_lines.extend(suffix_lines)
    
    return "\n".join(all_lines)


def generate_full_file():
    switch_code = generate_full_switch()
    
    # Generate load code for 9 parent polynomials (each 6 uint64s)
    load_lines = []
    for i in range(3):
        for k in range(3):
            idx = (i * 3 + k) * 6
            for p in range(3):
                for w, wname in enumerate(["lo", "hi"]):
                    offset = idx + p * 2 + w
                    load_lines.append(f"    m{i}{k}_p{p}_{wname} = tl.load(Parents_Ptr + base + {offset}).to(tl.uint64)")
    load_code = "\n".join(load_lines)
    
    # Generate store code for 9 output polynomials
    store_lines = []
    for i in range(3):
        for j in range(3):
            idx = (i * 3 + j) * 6
            for p in range(3):
                for w, wname in enumerate(["lo", "hi"]):
                    offset = idx + p * 2 + w
                    store_lines.append(f"    tl.store(Output_Ptr + out_base + {offset}, o{i}{j}_p{p}_{wname}.to(tl.int64))")
    store_code = "\n".join(store_lines)
    
    # Generate projlen computation: OR all planes then find MSB/LSB
    projlen_or_lines = []
    first = True
    for i in range(3):
        for j in range(3):
            if first:
                projlen_or_lines.append(f"    all_lo = o{i}{j}_p0_lo | o{i}{j}_p1_lo | o{i}{j}_p2_lo")
                projlen_or_lines.append(f"    all_hi = o{i}{j}_p0_hi | o{i}{j}_p1_hi | o{i}{j}_p2_hi")
                first = False
            else:
                projlen_or_lines.append(f"    all_lo = all_lo | o{i}{j}_p0_lo | o{i}{j}_p1_lo | o{i}{j}_p2_lo")
                projlen_or_lines.append(f"    all_hi = all_hi | o{i}{j}_p0_hi | o{i}{j}_p1_hi | o{i}{j}_p2_hi")
    projlen_or_code = "\n".join(projlen_or_lines)
    
    # Generate adjacency table as Python literal
    adj = get_adjacency()
    adj_rows = []
    for parent in range(22):
        row = [0]*22
        for child in adj[parent]:
            row[child] = 1
        adj_rows.append(row)
    adj_literal = repr(adj_rows)
    
    # Seed computation
    matrices = get_raw_matrix_data_pruned()
    
    file_content = f'''#!/usr/bin/env python3
"""
GPU Braid Search — Mod 7 Burau Representation (n=4)
Bit-sliced Triton kernel with 22 Garside suffix specializations.

Auto-generated by codegen.py
"""

import torch
import triton
import triton.language as tl
import time
import sys

# ==============================================================================
# CONFIGURATION
# ==============================================================================

N_SUFFIXES   = 22
BUCKET_SIZE  = 5_000_000    # Max braids to keep per step
USE_BEST     = 1_000_000    # Parents to select for next step
OUTPUT_CAP   = 12_000_000   # Max children per step (flat buffer)
MAX_STEPS    = 200          # Max search depth
BUCKET_CAP   = 500_000      # Max children per projlen bucket (FCFS)
N_BUCKETS    = 128          # One per possible projlen value (0..127)

# ==============================================================================
# TRITON HELPERS
# ==============================================================================

@triton.jit
def add_mod7(a0, a1, a2, b0, b1, b2):
    """Bit-sliced mod-7 addition on uint64 registers."""
    sum0 = a0 ^ b0
    c0   = a0 & b0
    sum1 = a1 ^ b1 ^ c0
    c1   = (a1 & b1) | (c0 & (a1 ^ b1))
    sum2 = a2 ^ b2 ^ c1
    c_out = (a2 & b2) | (c1 & (a2 ^ b2))
    # End-around carry
    final_s0 = sum0 ^ c_out
    c_fix0   = sum0 & c_out
    final_s1 = sum1 ^ c_fix0
    c_fix1   = sum1 & c_fix0
    final_s2 = sum2 ^ c_fix1
    # If result == 7 (0b111), force to 0
    is_seven = final_s0 & final_s1 & final_s2
    mask     = ~is_seven
    return final_s0 & mask, final_s1 & mask, final_s2 & mask

@triton.jit
def negate_mod7(p0, p1, p2):
    """Bit-sliced mod-7 negation. -x mod 7 = ~x for nonzero x."""
    is_zero = ~(p0 | p1 | p2)
    mask = ~is_zero
    return (~p0) & mask, (~p1) & mask, (~p2) & mask

# ==============================================================================
# MSB / LSB helpers for projlen
# ==============================================================================

@triton.jit
def msb64(x):
    """Return index of highest set bit (0..63), or -1 if x==0. Branchless binary search."""
    # x is uint64
    r = tl.zeros([], dtype=tl.int32)
    is_zero = (x == 0)
    
    hi32 = (x >> 32) & 0xFFFFFFFF
    use_hi = (hi32 != 0)
    r = tl.where(use_hi, r + 32, r)
    v = tl.where(use_hi, hi32, x & 0xFFFFFFFF)
    
    hi16 = (v >> 16) & 0xFFFF
    use_hi = (hi16 != 0)
    r = tl.where(use_hi, r + 16, r)
    v = tl.where(use_hi, hi16, v & 0xFFFF)
    
    hi8 = (v >> 8) & 0xFF
    use_hi = (hi8 != 0)
    r = tl.where(use_hi, r + 8, r)
    v = tl.where(use_hi, hi8, v & 0xFF)
    
    hi4 = (v >> 4) & 0xF
    use_hi = (hi4 != 0)
    r = tl.where(use_hi, r + 4, r)
    v = tl.where(use_hi, hi4, v & 0xF)
    
    hi2 = (v >> 2) & 0x3
    use_hi = (hi2 != 0)
    r = tl.where(use_hi, r + 2, r)
    v = tl.where(use_hi, hi2, v & 0x3)
    
    hi1 = (v >> 1) & 0x1
    use_hi = (hi1 != 0)
    r = tl.where(use_hi, r + 1, r)
    
    r = tl.where(is_zero, -1, r)
    return r

@triton.jit
def lsb64(x):
    """Return index of lowest set bit (0..63), or 64 if x==0."""
    is_zero = (x == 0)
    # Isolate lowest set bit
    lowest = x & (0 - x)   # x & (-x) using subtraction to avoid Triton sign issues
    pos = msb64(lowest)  # For a power of 2, msb == lsb
    pos = tl.where(is_zero, 64, pos)
    return pos

# ==============================================================================
# MAIN BRAID STEP KERNEL
# ==============================================================================

@triton.jit
def kernel_braid_step(
    Parents_Ptr,         # int64  [N_parents, 54] flattened
    Parent_Meta_Ptr,     # int32  [N_parents]
    Output_Ptr,          # int64  [OUTPUT_CAP, 54] flattened
    Output_Meta_Ptr,     # int32  [OUTPUT_CAP]
    Global_Counter_Ptr,  # int32  [1]
    Bucket_Counters_Ptr, # int32  [N_BUCKETS]
    Adj_Ptr,             # int8   [22 * 22]
    N_PARENTS: tl.constexpr,
    OUTPUT_CAP_PARAM: tl.constexpr,
    BUCKET_CAP_PARAM: tl.constexpr,
    SUFFIX_IDX: tl.constexpr,
):
    parent_idx = tl.program_id(0)
    if parent_idx >= N_PARENTS:
        return

    # --- Adjacency check ---
    last_suffix = tl.load(Parent_Meta_Ptr + parent_idx) & 0xFF
    # Cast to int32 for the offset calculation
    adj_offset = last_suffix.to(tl.int32) * 22 + SUFFIX_IDX
    adj_val = tl.load(Adj_Ptr + adj_offset)
    if adj_val == 0:
        return

    # --- Load parent matrix (54 uint64s) ---
    base = parent_idx * 54
    ZERO = tl.zeros([], dtype=tl.uint64)
{load_code}

    # --- Matrix multiplication: output = parent @ suffix[SUFFIX_IDX] ---
{switch_code}

    # --- Compute ProjLen (max_degree - min_degree) ---
{projlen_or_code}

    # Check for zero matrix (kernel element!)
    is_zero_matrix = (all_lo == 0) & (all_hi == 0)
    
    # Max degree
    max_deg_hi = msb64(all_hi)  # -1 if all_hi==0
    max_deg_lo = msb64(all_lo)
    has_hi = (all_hi != 0)
    max_deg = tl.where(has_hi, max_deg_hi + 64, max_deg_lo)
    
    # Min degree
    min_deg_lo = lsb64(all_lo)  # 64 if all_lo==0
    min_deg_hi = lsb64(all_hi)
    has_lo = (all_lo != 0)
    min_deg = tl.where(has_lo, min_deg_lo, min_deg_hi + 64)
    
    projlen = tl.where(is_zero_matrix, tl.zeros([], dtype=tl.int32), max_deg - min_deg)

    # --- FCFS bucket check ---
    bucket_slot = tl.atomic_add(Bucket_Counters_Ptr + projlen, 1)
    if bucket_slot >= BUCKET_CAP_PARAM:
        return  # Bucket full, discard

    # --- Reserve global output slot ---
    global_slot = tl.atomic_add(Global_Counter_Ptr, 1)
    if global_slot >= OUTPUT_CAP_PARAM:
        return  # Buffer full

    # --- Write output ---
    out_base = global_slot.to(tl.int64) * 54
{store_code}

    # --- Write metadata: projlen << 8 | suffix_idx ---
    meta = (projlen << 8) | SUFFIX_IDX
    tl.store(Output_Meta_Ptr + global_slot, meta)

    # --- Flag zero matrices (kernel elements!) ---
    if is_zero_matrix:
        # Write a sentinel to a known location (slot 0 of bucket counters
        # is projlen 0, we use bucket 127 as a flag)
        tl.atomic_add(Bucket_Counters_Ptr + 127, 1000000)


# ==============================================================================
# HOST FUNCTIONS
# ==============================================================================

ADJ_TABLE = {adj_literal}

def build_adjacency_tensor():
    """Build adjacency table as int8 tensor on GPU."""
    flat = []
    for row in ADJ_TABLE:
        flat.extend(row)
    return torch.tensor(flat, dtype=torch.int8, device="cuda")

def build_seed_braids():
    """
    Build the 22 seed braids (one per suffix applied to identity).
    Each seed is the suffix matrix itself in bit-sliced form.
    Returns (data [22, 54] int64, meta [22] int32).
    """
    matrices = get_raw_matrix_data_pruned_host()
    
    data = torch.zeros((22, 54), dtype=torch.int64, device="cpu")
    meta = torch.zeros((22,), dtype=torch.int32, device="cpu")
    
    for s in range(22):
        mat = matrices[s]
        for r in range(3):
            for c in range(3):
                poly_idx = r * 3 + c
                base = poly_idx * 6  # 6 uint64s per polynomial
                
                for (deg, coeff) in mat[r][c]:
                    # coeff: +1 -> 0b001 (p0=1), -1 -> 6 mod 7 = 0b110 (p1=1,p2=1)
                    if deg < 64:
                        word = 0  # lo
                        bit_pos = deg
                    else:
                        word = 1  # hi
                        bit_pos = deg - 64
                    
                    bit_val = 1 << bit_pos
                    
                    if coeff == 1:
                        # p0 = 1
                        data[s, base + 0 * 2 + word] |= bit_val  # plane 0
                    else:  # coeff == -1 -> 6 = 0b110
                        # p1 = 1, p2 = 1
                        data[s, base + 1 * 2 + word] |= bit_val  # plane 1
                        data[s, base + 2 * 2 + word] |= bit_val  # plane 2
        
        # Compute projlen for seed
        all_bits = 0
        for idx in range(9):
            base_idx = idx * 6
            for p in range(3):
                all_bits |= data[s, base_idx + p * 2 + 0].item()
                all_bits |= (data[s, base_idx + p * 2 + 1].item() << 64)
        
        if all_bits == 0:
            projlen = 0
        else:
            max_deg = all_bits.bit_length() - 1
            # Find min degree
            min_deg = 0
            tmp = all_bits
            while tmp and not (tmp & 1):
                min_deg += 1
                tmp >>= 1
            projlen = max_deg - min_deg
        
        meta[s] = (projlen << 8) | s
    
    return data.cuda(), meta.cuda()


def get_raw_matrix_data_pruned_host():
    """Same matrix data, accessible at runtime for seeding."""
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


def run_search():
    """Main search loop."""
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    device = torch.device("cuda")
    print(f"Device: {{torch.cuda.get_device_name()}}")
    print(f"Config: BUCKET_SIZE={{BUCKET_SIZE:,}}, USE_BEST={{USE_BEST:,}}, OUTPUT_CAP={{OUTPUT_CAP:,}}")
    
    # --- Build adjacency table ---
    adj_tensor = build_adjacency_tensor()
    
    # --- Build seed braids ---
    parent_data, parent_meta = build_seed_braids()
    n_parents = parent_data.shape[0]
    print(f"Seeded with {{n_parents}} braids")
    
    # --- Allocate output buffers ---
    out_data = torch.zeros((OUTPUT_CAP, 54), dtype=torch.int64, device=device)
    out_meta = torch.zeros((OUTPUT_CAP,), dtype=torch.int32, device=device)
    global_counter = torch.zeros((1,), dtype=torch.int32, device=device)
    bucket_counters = torch.zeros((N_BUCKETS,), dtype=torch.int32, device=device)
    
    # --- Main loop ---
    for step in range(MAX_STEPS):
        t0 = time.time()
        
        # Reset counters
        global_counter.zero_()
        bucket_counters.zero_()
        
        # Shuffle parents (critical for FCFS uniformity)
        if n_parents > 1:
            perm = torch.randperm(n_parents, device=device)
            parent_data = parent_data[perm]
            parent_meta = parent_meta[perm]
        
        # Launch 22 suffix kernels
        grid = (n_parents,)
        for s in range(N_SUFFIXES):
            kernel_braid_step[grid](
                parent_data.data_ptr(),
                parent_meta.data_ptr(),
                out_data.data_ptr(),
                out_meta.data_ptr(),
                global_counter.data_ptr(),
                bucket_counters.data_ptr(),
                adj_tensor.data_ptr(),
                n_parents,
                OUTPUT_CAP,
                BUCKET_CAP,
                s,  # SUFFIX_IDX constexpr
            )
        
        torch.cuda.synchronize()
        
        # Read results
        n_children = min(global_counter.item(), OUTPUT_CAP)
        bucket_counts = bucket_counters.cpu().tolist()
        
        # Check for kernel elements (zero matrices flagged via bucket 127)
        if bucket_counts[127] >= 1000000:
            print(f"\\n*** KERNEL ELEMENT FOUND at step {{step}}! ***")
            # Find which children are zero
            child_projlens = (out_meta[:n_children] >> 8) & 0xFF
            zero_mask = (child_projlens == 0)
            n_zeros = zero_mask.sum().item()
            print(f"Found {{n_zeros}} zero-matrix children. Investigate further!")
            # Could save and break here
        
        # Extract projlens for selection
        child_projlens = (out_meta[:n_children] >> 8).to(torch.int32) & 0x7F
        
        # Select best braids: lowest projlen, up to USE_BEST
        if n_children <= USE_BEST:
            # Keep all
            parent_data = out_data[:n_children].clone()
            parent_meta = out_meta[:n_children].clone()
            n_parents = n_children
        else:
            # Sort by projlen (ascending) and take top USE_BEST
            sorted_indices = torch.argsort(child_projlens[:n_children])
            keep = sorted_indices[:USE_BEST]
            parent_data = out_data[keep].clone()
            parent_meta = out_meta[keep].clone()
            n_parents = USE_BEST
        
        t1 = time.time()
        dt = t1 - t0
        
        # Stats
        if n_parents > 0:
            kept_projlens = (parent_meta[:n_parents] >> 8).to(torch.int32) & 0x7F
            min_pl = kept_projlens.min().item()
            max_pl = kept_projlens.max().item()
            mean_pl = kept_projlens.float().mean().item()
        else:
            min_pl = max_pl = mean_pl = 0
        
        # Print bucket histogram (nonzero buckets only)
        nonzero_buckets = [(i, c) for i, c in enumerate(bucket_counts[:127]) if c > 0]
        bucket_str = " ".join(f"[{{b}}]:{{c}}" for b, c in nonzero_buckets[:10])
        
        print(f"Step {{step:3d}} | {{dt:.2f}}s | children={{n_children:>10,}} | kept={{n_parents:>10,}} | "
              f"projlen min/mean/max={{min_pl}}/{{mean_pl:.1f}}/{{max_pl}} | buckets: {{bucket_str}}")
        
        if n_parents == 0:
            print("No surviving braids. Search exhausted.")
            break
    
    print("\\nSearch complete.")


if __name__ == "__main__":
    run_search()
'''
    return file_content


if __name__ == "__main__":
    content = generate_full_file()
    with open("/home/claude/braid_search.py", "w") as f:
        f.write(content)
    print(f"Generated braid_search.py ({len(content)} bytes, {content.count(chr(10))} lines)")
