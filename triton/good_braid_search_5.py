#!/usr/bin/env python3
"""
GPU Braid Search — Mod 5 Burau Representation (n=4)
Fused SoA Triton kernel with data-driven suffix computation.

Rewritten from auto-generated codegen.py output for:
  - SoA memory layout [54, N] for coalesced GPU reads
  - Single fused kernel (all 22 suffixes in one launch)
  - Compile-time suffix descriptors (no runtime descriptor loads)
"""

import torch
import triton
import triton.language as tl
import time
import sys
import argparse

# ==============================================================================
# COMMAND LINE ARGUMENTS
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="GPU Braid Search — Mod 5 Burau Representation")
    parser.add_argument("-u", "--use-best", type=float, default=5.0,
                        help="USE_BEST in millions (default: 5.0, i.e., 5_000_000)")
    parser.add_argument("-b", "--bucket-cap", type=float, default=2.5,
                        help="BUCKET_CAP in millions (default: 2.5, i.e., 2_500_000)")
    parser.add_argument("-m", "--max-steps", type=int, default=127,
                        help="MAX_STEPS (raw value, default: 127)")
    return parser.parse_args()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

args = parse_args()

N_SUFFIXES   = 22
N_BUCKETS    = 128          # One per possible projlen value (0..127)
MAX_STEPS    = args.max_steps  # Run from length 1 (seeds) through length MAX_STEPS

# --- GPU Presets ---
# Uncomment the preset matching your GPU, or set manually.

# H200 (80 GB):
#   USE_BEST    = 40_000_000
#   OUTPUT_CAP  = 120_000_000
#   BUCKET_CAP  = 20_000_000

# RTX 5000 Ada (32 GB):
#   USE_BEST    = 7_000_000
#   OUTPUT_CAP  = 56_000_000
#   BUCKET_CAP  = 3_500_000

# Conservative default (works on ~16 GB):
USE_BEST     = int(args.use_best * 1_000_000)    # Parents to select for next step
OUTPUT_CAP   = 8*USE_BEST   # Max children per step (flat buffer, >= USE_BEST * 8)
BUCKET_CAP   = int(args.bucket_cap * 1_000_000)    # Max children per projlen bucket (FCFS)

# ==============================================================================
# SUFFIX DESCRIPTORS (compile-time constants for the Triton kernel)
# ==============================================================================
#
# For matrix multiplication output[i][j] = sum_k parent[i][k] * suffix[k][j],
# each output entry has 1-3 terms. Each term is described by:
#   - PE: parent matrix entry index (0-8), where parent_entry = i*3+k
#   - SHIFT: monomial degree shift (0-4)
#   - NEG: 1 if coefficient is -1 (negate), 0 if +1
#   - NTERMS: number of active terms for this output entry
#
# Indexing: PE/SHIFT/NEG at [s*27 + entry*3 + t], NTERMS at [s*9 + entry]
# where s=suffix(0-21), entry=i*3+j(0-8), t=term(0-2)

DESC_PE = (2, 0, 0, 0, 1, 2, 0, 0, 0, 5, 0, 0, 3, 4, 5, 3, 0, 0, 8, 0, 0, 6, 7, 8, 6, 0, 0, 0, 0, 0, 0, 1, 2, 2, 0, 0, 3, 0, 0, 3, 4, 5, 5, 0, 0, 6, 0, 0, 6, 7, 8, 8, 0, 0, 0, 1, 2, 1, 2, 0, 1, 0, 0, 3, 4, 5, 4, 5, 0, 4, 0, 0, 6, 7, 8, 7, 8, 0, 7, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 2, 4, 0, 0, 3, 4, 0, 3, 4, 5, 7, 0, 0, 6, 7, 0, 6, 7, 8, 1, 0, 0, 1, 2, 0, 0, 1, 2, 4, 0, 0, 4, 5, 0, 3, 4, 5, 7, 0, 0, 7, 8, 0, 6, 7, 8, 0, 1, 2, 0, 1, 0, 1, 0, 0, 3, 4, 5, 3, 4, 0, 4, 0, 0, 6, 7, 8, 6, 7, 0, 7, 0, 0, 0, 1, 0, 2, 0, 0, 1, 2, 0, 3, 4, 0, 5, 0, 0, 4, 5, 0, 6, 7, 0, 8, 0, 0, 7, 8, 0, 1, 2, 0, 0, 0, 0, 0, 1, 0, 4, 5, 0, 3, 0, 0, 3, 4, 0, 7, 8, 0, 6, 0, 0, 6, 7, 0, 1, 2, 0, 2, 0, 0, 0, 1, 0, 4, 5, 0, 5, 0, 0, 3, 4, 0, 7, 8, 0, 8, 0, 0, 6, 7, 0, 0, 1, 0, 0, 0, 0, 1, 2, 0, 3, 4, 0, 3, 0, 0, 4, 5, 0, 6, 7, 0, 6, 0, 0, 7, 8, 0, 0, 0, 0, 1, 2, 0, 2, 0, 0, 3, 0, 0, 4, 5, 0, 5, 0, 0, 6, 0, 0, 7, 8, 0, 8, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 5, 0, 0, 3, 4, 0, 3, 0, 0, 8, 0, 0, 6, 7, 0, 6, 0, 0, 2, 0, 0, 1, 2, 0, 0, 0, 0, 5, 0, 0, 4, 5, 0, 3, 0, 0, 8, 0, 0, 7, 8, 0, 6, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 3, 0, 0, 3, 4, 0, 5, 0, 0, 6, 0, 0, 6, 7, 0, 8, 0, 0, 0, 1, 0, 1, 0, 0, 1, 2, 0, 3, 4, 0, 4, 0, 0, 4, 5, 0, 6, 7, 0, 7, 0, 0, 7, 8, 0, 1, 2, 0, 0, 1, 2, 0, 1, 0, 4, 5, 0, 3, 4, 5, 3, 4, 0, 7, 8, 0, 6, 7, 8, 6, 7, 0, 1, 2, 0, 1, 0, 0, 0, 1, 0, 4, 5, 0, 4, 0, 0, 3, 4, 0, 7, 8, 0, 7, 0, 0, 6, 7, 0, 0, 1, 0, 0, 1, 2, 1, 2, 0, 3, 4, 0, 3, 4, 5, 4, 5, 0, 6, 7, 0, 6, 7, 8, 7, 8, 0, 0, 1, 2, 2, 0, 0, 1, 0, 0, 3, 4, 5, 5, 0, 0, 4, 0, 0, 6, 7, 8, 8, 0, 0, 7, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 4, 0, 0, 3, 0, 0, 3, 4, 5, 7, 0, 0, 6, 0, 0, 6, 7, 8, 1, 0, 0, 2, 0, 0, 0, 1, 2, 4, 0, 0, 5, 0, 0, 3, 4, 5, 7, 0, 0, 8, 0, 0, 6, 7, 8, 0, 1, 2, 0, 0, 0, 1, 0, 0, 3, 4, 5, 3, 0, 0, 4, 0, 0, 6, 7, 8, 6, 0, 0, 7, 0, 0)

DESC_SHIFT = (2, 0, 0, 3, 4, 3, 2, 0, 0, 2, 0, 0, 3, 4, 3, 2, 0, 0, 2, 0, 0, 3, 4, 3, 2, 0, 0, 2, 0, 0, 1, 0, 1, 2, 0, 0, 2, 0, 0, 1, 0, 1, 2, 0, 0, 2, 0, 0, 1, 0, 1, 2, 0, 0, 0, 1, 2, 2, 3, 0, 1, 0, 0, 0, 1, 2, 2, 3, 0, 1, 0, 0, 0, 1, 2, 2, 3, 0, 1, 0, 0, 1, 0, 0, 3, 2, 0, 2, 1, 0, 1, 0, 0, 3, 2, 0, 2, 1, 0, 1, 0, 0, 3, 2, 0, 2, 1, 0, 3, 0, 0, 2, 1, 0, 4, 3, 2, 3, 0, 0, 2, 1, 0, 4, 3, 2, 3, 0, 0, 2, 1, 0, 4, 3, 2, 2, 3, 4, 1, 2, 0, 3, 0, 0, 2, 3, 4, 1, 2, 0, 3, 0, 0, 2, 3, 4, 1, 2, 0, 3, 0, 0, 0, 1, 0, 1, 0, 0, 3, 2, 0, 0, 1, 0, 1, 0, 0, 3, 2, 0, 0, 1, 0, 1, 0, 0, 3, 2, 0, 3, 4, 0, 3, 0, 0, 2, 1, 0, 3, 4, 0, 3, 0, 0, 2, 1, 0, 3, 4, 0, 3, 0, 0, 2, 1, 0, 1, 2, 0, 3, 0, 0, 4, 3, 0, 1, 2, 0, 3, 0, 0, 4, 3, 0, 1, 2, 0, 3, 0, 0, 4, 3, 0, 2, 3, 0, 1, 0, 0, 1, 0, 0, 2, 3, 0, 1, 0, 0, 1, 0, 0, 2, 3, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 4, 0, 0, 3, 4, 0, 2, 0, 0, 4, 0, 0, 3, 4, 0, 2, 0, 0, 4, 0, 0, 3, 4, 0, 2, 0, 0, 2, 0, 0, 4, 3, 0, 4, 0, 0, 2, 0, 0, 4, 3, 0, 4, 0, 0, 2, 0, 0, 4, 3, 0, 4, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 1, 0, 0, 1, 2, 0, 3, 2, 3, 2, 1, 0, 1, 2, 0, 3, 2, 3, 2, 1, 0, 1, 2, 0, 3, 2, 3, 2, 1, 0, 3, 4, 0, 2, 0, 0, 4, 3, 0, 3, 4, 0, 2, 0, 0, 4, 3, 0, 3, 4, 0, 2, 0, 0, 4, 3, 0, 2, 3, 0, 1, 2, 1, 3, 2, 0, 2, 3, 0, 1, 2, 1, 3, 2, 0, 2, 3, 0, 1, 2, 1, 3, 2, 0, 0, 1, 2, 3, 0, 0, 3, 0, 0, 0, 1, 2, 3, 0, 0, 3, 0, 0, 0, 1, 2, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 2, 1, 0, 3, 0, 0, 3, 0, 0, 2, 1, 0, 3, 0, 0, 3, 0, 0, 2, 1, 0, 1, 0, 0, 1, 0, 0, 4, 3, 2, 1, 0, 0, 1, 0, 0, 4, 3, 2, 1, 0, 0, 1, 0, 0, 4, 3, 2, 2, 3, 4, 1, 0, 0, 1, 0, 0, 2, 3, 4, 1, 0, 0, 1, 0, 0, 2, 3, 4, 1, 0, 0, 1, 0, 0)

DESC_NEG = (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0)

DESC_NTERMS = (1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 2, 1, 3, 2, 1, 3, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 3, 2, 2, 3, 2, 2, 3, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 3, 2, 2, 3, 2, 2, 3, 2, 3, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 3, 1, 1, 3, 1, 1, 3, 1, 1)

# ==============================================================================
# TRITON HELPERS
# ==============================================================================

@triton.jit
def add_mod5(a0, a1, a2, b0, b1, b2):
    """Bit-sliced mod-5 addition on uint64 registers."""
    # Standard 3-bit binary addition
    sum0 = a0 ^ b0
    c0   = a0 & b0
    sum1 = a1 ^ b1 ^ c0
    c1   = (a1 & b1) | (c0 & (a1 ^ b1))
    sum2 = a2 ^ b2 ^ c1
    c_out = (a2 & b2) | (c1 & (a2 ^ b2))
    # Overflow: sum >= 5 iff carry_out or (bit2 and (bit1 or bit0))
    overflow = c_out | (sum2 & (sum1 | sum0))
    # Correction: add 3 (011) when overflow to map 5..8 → 0..3
    t0 = sum0 ^ overflow
    fix_c0 = sum0 & overflow
    t1 = sum1 ^ overflow ^ fix_c0
    fix_c1 = (sum1 & overflow) | (fix_c0 & (sum1 ^ overflow))
    t2 = sum2 ^ fix_c1
    return t0, t1, t2

@triton.jit
def negate_mod5(p0, p1, p2):
    """Bit-sliced mod-5 negation. -x mod 5 = 5-x for nonzero x.
    Binary subtraction 101 - (p2,p1,p0) gives:
      r0 = ~p0, r1 = p1, r2 = ~p2 ^ p1  (masked to zero for zero inputs).
    """
    is_zero = ~(p0 | p1 | p2)
    mask = ~is_zero
    return (~p0) & mask, p1 & mask, ((~p2) ^ p1) & mask

# ==============================================================================
# MSB / LSB helpers for projlen
# ==============================================================================

@triton.jit
def shr128(lo, hi, s):
    """
    Logical right-shift of a 128-bit value (lo, hi) by s bits.
    Safe for s in [0, 127]. Uses uint64 arithmetic for logical shifts.
    Returns (new_lo, new_hi) as int64.
    """
    lo_u = lo.to(tl.uint64)
    hi_u = hi.to(tl.uint64)
    small = (s < 64)
    inv_s = 63 - s
    lo_small = (lo_u >> s) | ((hi_u << inv_s) << 1)
    hi_small = hi_u >> s
    s_big = s - 64
    s_big_clamped = tl.where(s_big < 64, s_big, tl.zeros([], dtype=tl.int32) + 63)
    lo_big = hi_u >> s_big_clamped
    lo_big = tl.where(s_big < 64, lo_big, tl.zeros([], dtype=tl.uint64))
    hi_big = tl.zeros([], dtype=tl.uint64)

    new_lo = tl.where(small, lo_small, lo_big)
    new_hi = tl.where(small, hi_small, hi_big)
    return new_lo.to(tl.int64), new_hi.to(tl.int64)

@triton.jit
def shl128_0(lo, hi):
    return lo, hi

@triton.jit
def shl128_1(lo, hi):
    lo_u = lo.to(tl.uint64)
    hi_u = hi.to(tl.uint64)
    new_lo = lo_u << 1
    new_hi = (hi_u << 1) | (lo_u >> 63)
    return new_lo.to(tl.int64), new_hi.to(tl.int64)

@triton.jit
def shl128_2(lo, hi):
    lo_u = lo.to(tl.uint64)
    hi_u = hi.to(tl.uint64)
    new_lo = lo_u << 2
    new_hi = (hi_u << 2) | (lo_u >> 62)
    return new_lo.to(tl.int64), new_hi.to(tl.int64)

@triton.jit
def shl128_3(lo, hi):
    lo_u = lo.to(tl.uint64)
    hi_u = hi.to(tl.uint64)
    new_lo = lo_u << 3
    new_hi = (hi_u << 3) | (lo_u >> 61)
    return new_lo.to(tl.int64), new_hi.to(tl.int64)

@triton.jit
def shl128_4(lo, hi):
    lo_u = lo.to(tl.uint64)
    hi_u = hi.to(tl.uint64)
    new_lo = lo_u << 4
    new_hi = (hi_u << 4) | (lo_u >> 60)
    return new_lo.to(tl.int64), new_hi.to(tl.int64)

@triton.jit
def shl128(lo, hi, SHIFT: tl.constexpr):
    if SHIFT == 0:
        return shl128_0(lo, hi)
    elif SHIFT == 1:
        return shl128_1(lo, hi)
    elif SHIFT == 2:
        return shl128_2(lo, hi)
    elif SHIFT == 3:
        return shl128_3(lo, hi)
    else:  # SHIFT == 4
        return shl128_4(lo, hi)

@triton.jit
def msb64(x):
    """Return index of highest set bit (0..63), or -1 if x==0. Branchless binary search."""
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
    lowest = x & (0 - x)
    pos = msb64(lowest)
    pos = tl.where(is_zero, 64, pos)
    return pos

# ==============================================================================
# LOAD 6 PARENT FIELDS HELPER
# ==============================================================================

@triton.jit
def load_parent_poly(Parents_Ptr, PE: tl.constexpr, N_STRIDE, idx):
    """Load 6 uint64 values for parent entry PE from SoA layout.
    SoA layout: field f of braid i at Parents_Ptr + f * N_STRIDE + i
    Parent entry PE (0-8) maps to fields PE*6 through PE*6+5.
    idx is a scalar parent index.
    """
    base_field: tl.constexpr = PE * 6
    p0_lo = tl.load(Parents_Ptr + (base_field + 0) * N_STRIDE + idx)
    p0_hi = tl.load(Parents_Ptr + (base_field + 1) * N_STRIDE + idx)
    p1_lo = tl.load(Parents_Ptr + (base_field + 2) * N_STRIDE + idx)
    p1_hi = tl.load(Parents_Ptr + (base_field + 3) * N_STRIDE + idx)
    p2_lo = tl.load(Parents_Ptr + (base_field + 4) * N_STRIDE + idx)
    p2_hi = tl.load(Parents_Ptr + (base_field + 5) * N_STRIDE + idx)
    return p0_lo, p0_hi, p1_lo, p1_hi, p2_lo, p2_hi

@triton.jit
def shift_and_neg(p0_lo, p0_hi, p1_lo, p1_hi, p2_lo, p2_hi,
                  SHIFT: tl.constexpr, NEG: tl.constexpr):
    """Apply shift and optional negation to a polynomial."""
    p0_lo_u = p0_lo.to(tl.uint64)
    p0_hi_u = p0_hi.to(tl.uint64)
    p1_lo_u = p1_lo.to(tl.uint64)
    p1_hi_u = p1_hi.to(tl.uint64)
    p2_lo_u = p2_lo.to(tl.uint64)
    p2_hi_u = p2_hi.to(tl.uint64)
    p0_lo_u, p0_hi_u = shl128(p0_lo_u, p0_hi_u, SHIFT)
    p1_lo_u, p1_hi_u = shl128(p1_lo_u, p1_hi_u, SHIFT)
    p2_lo_u, p2_hi_u = shl128(p2_lo_u, p2_hi_u, SHIFT)
    if NEG == 1:
        p0_lo_u, p1_lo_u, p2_lo_u = negate_mod5(p0_lo_u, p1_lo_u, p2_lo_u)
        p0_hi_u, p1_hi_u, p2_hi_u = negate_mod5(p0_hi_u, p1_hi_u, p2_hi_u)
    return p0_lo_u, p0_hi_u, p1_lo_u, p1_hi_u, p2_lo_u, p2_hi_u

# ==============================================================================
# PER-ENTRY COMPUTATION HELPER
# ==============================================================================

@triton.jit
def compute_entry_1term(Parents_Ptr, N_STRIDE, idx,
                        PE0: tl.constexpr, SHIFT0: tl.constexpr, NEG0: tl.constexpr):
    """Compute one output matrix entry with exactly 1 term."""
    p0_lo, p0_hi, p1_lo, p1_hi, p2_lo, p2_hi = load_parent_poly(Parents_Ptr, PE0, N_STRIDE, idx)
    return shift_and_neg(p0_lo, p0_hi, p1_lo, p1_hi, p2_lo, p2_hi, SHIFT0, NEG0)

@triton.jit
def compute_entry_2terms(Parents_Ptr, N_STRIDE, idx,
                         PE0: tl.constexpr, SHIFT0: tl.constexpr, NEG0: tl.constexpr,
                         PE1: tl.constexpr, SHIFT1: tl.constexpr, NEG1: tl.constexpr):
    """Compute one output matrix entry with exactly 2 terms."""
    a0_lo, a0_hi, a1_lo, a1_hi, a2_lo, a2_hi = load_parent_poly(Parents_Ptr, PE0, N_STRIDE, idx)
    a0_lo, a0_hi, a1_lo, a1_hi, a2_lo, a2_hi = shift_and_neg(a0_lo, a0_hi, a1_lo, a1_hi, a2_lo, a2_hi, SHIFT0, NEG0)
    b0_lo, b0_hi, b1_lo, b1_hi, b2_lo, b2_hi = load_parent_poly(Parents_Ptr, PE1, N_STRIDE, idx)
    b0_lo, b0_hi, b1_lo, b1_hi, b2_lo, b2_hi = shift_and_neg(b0_lo, b0_hi, b1_lo, b1_hi, b2_lo, b2_hi, SHIFT1, NEG1)
    r0_lo, r1_lo, r2_lo = add_mod5(a0_lo, a1_lo, a2_lo, b0_lo, b1_lo, b2_lo)
    r0_hi, r1_hi, r2_hi = add_mod5(a0_hi, a1_hi, a2_hi, b0_hi, b1_hi, b2_hi)
    return r0_lo, r0_hi, r1_lo, r1_hi, r2_lo, r2_hi

@triton.jit
def compute_entry_3terms(Parents_Ptr, N_STRIDE, idx,
                         PE0: tl.constexpr, SHIFT0: tl.constexpr, NEG0: tl.constexpr,
                         PE1: tl.constexpr, SHIFT1: tl.constexpr, NEG1: tl.constexpr,
                         PE2: tl.constexpr, SHIFT2: tl.constexpr, NEG2: tl.constexpr):
    """Compute one output matrix entry with exactly 3 terms."""
    a0_lo, a0_hi, a1_lo, a1_hi, a2_lo, a2_hi = load_parent_poly(Parents_Ptr, PE0, N_STRIDE, idx)
    a0_lo, a0_hi, a1_lo, a1_hi, a2_lo, a2_hi = shift_and_neg(a0_lo, a0_hi, a1_lo, a1_hi, a2_lo, a2_hi, SHIFT0, NEG0)
    b0_lo, b0_hi, b1_lo, b1_hi, b2_lo, b2_hi = load_parent_poly(Parents_Ptr, PE1, N_STRIDE, idx)
    b0_lo, b0_hi, b1_lo, b1_hi, b2_lo, b2_hi = shift_and_neg(b0_lo, b0_hi, b1_lo, b1_hi, b2_lo, b2_hi, SHIFT1, NEG1)
    r0_lo, r1_lo, r2_lo = add_mod5(a0_lo, a1_lo, a2_lo, b0_lo, b1_lo, b2_lo)
    r0_hi, r1_hi, r2_hi = add_mod5(a0_hi, a1_hi, a2_hi, b0_hi, b1_hi, b2_hi)
    c0_lo, c0_hi, c1_lo, c1_hi, c2_lo, c2_hi = load_parent_poly(Parents_Ptr, PE2, N_STRIDE, idx)
    c0_lo, c0_hi, c1_lo, c1_hi, c2_lo, c2_hi = shift_and_neg(c0_lo, c0_hi, c1_lo, c1_hi, c2_lo, c2_hi, SHIFT2, NEG2)
    r0_lo, r1_lo, r2_lo = add_mod5(r0_lo, r1_lo, r2_lo, c0_lo, c1_lo, c2_lo)
    r0_hi, r1_hi, r2_hi = add_mod5(r0_hi, r1_hi, r2_hi, c0_hi, c1_hi, c2_hi)
    return r0_lo, r0_hi, r1_lo, r1_hi, r2_lo, r2_hi

# ==============================================================================
# MAIN FUSED BRAID STEP KERNEL
# ==============================================================================

@triton.jit
def kernel_braid_step(
    Parents_Ptr,         # int64  [54, N_STRIDE] SoA
    Parent_Meta_Ptr,     # int32  [N_STRIDE]
    Output_Ptr,          # int64  [54, OUT_STRIDE] SoA
    Output_Meta_Ptr,     # int32  [OUT_STRIDE]
    Output_Parent_Ptr,   # int32  [OUT_STRIDE] — parent index for word reconstruction
    Global_Counter_Ptr,  # int32  [1]
    Bucket_Counters_Ptr, # int32  [N_BUCKETS]
    Adj_Ptr,             # int8   [22 * 22]
    N_PARENTS,                       # int32 (runtime, avoids recompile)
    N_STRIDE,            # int64 (stride between fields in SoA)
    OUT_STRIDE,          # int64 (stride for output SoA)
    OUTPUT_CAP_PARAM: tl.constexpr,
    BUCKET_CAP_PARAM: tl.constexpr,
    SUFFIX_IDX: tl.constexpr,
    # Descriptor values for this suffix's 9 output entries (each entry has up to 3 terms)
    # Entry 0
    NT0: tl.constexpr, PE0_0: tl.constexpr, SH0_0: tl.constexpr, NG0_0: tl.constexpr,
    PE0_1: tl.constexpr, SH0_1: tl.constexpr, NG0_1: tl.constexpr,
    PE0_2: tl.constexpr, SH0_2: tl.constexpr, NG0_2: tl.constexpr,
    # Entry 1
    NT1: tl.constexpr, PE1_0: tl.constexpr, SH1_0: tl.constexpr, NG1_0: tl.constexpr,
    PE1_1: tl.constexpr, SH1_1: tl.constexpr, NG1_1: tl.constexpr,
    PE1_2: tl.constexpr, SH1_2: tl.constexpr, NG1_2: tl.constexpr,
    # Entry 2
    NT2: tl.constexpr, PE2_0: tl.constexpr, SH2_0: tl.constexpr, NG2_0: tl.constexpr,
    PE2_1: tl.constexpr, SH2_1: tl.constexpr, NG2_1: tl.constexpr,
    PE2_2: tl.constexpr, SH2_2: tl.constexpr, NG2_2: tl.constexpr,
    # Entry 3
    NT3: tl.constexpr, PE3_0: tl.constexpr, SH3_0: tl.constexpr, NG3_0: tl.constexpr,
    PE3_1: tl.constexpr, SH3_1: tl.constexpr, NG3_1: tl.constexpr,
    PE3_2: tl.constexpr, SH3_2: tl.constexpr, NG3_2: tl.constexpr,
    # Entry 4
    NT4: tl.constexpr, PE4_0: tl.constexpr, SH4_0: tl.constexpr, NG4_0: tl.constexpr,
    PE4_1: tl.constexpr, SH4_1: tl.constexpr, NG4_1: tl.constexpr,
    PE4_2: tl.constexpr, SH4_2: tl.constexpr, NG4_2: tl.constexpr,
    # Entry 5
    NT5: tl.constexpr, PE5_0: tl.constexpr, SH5_0: tl.constexpr, NG5_0: tl.constexpr,
    PE5_1: tl.constexpr, SH5_1: tl.constexpr, NG5_1: tl.constexpr,
    PE5_2: tl.constexpr, SH5_2: tl.constexpr, NG5_2: tl.constexpr,
    # Entry 6
    NT6: tl.constexpr, PE6_0: tl.constexpr, SH6_0: tl.constexpr, NG6_0: tl.constexpr,
    PE6_1: tl.constexpr, SH6_1: tl.constexpr, NG6_1: tl.constexpr,
    PE6_2: tl.constexpr, SH6_2: tl.constexpr, NG6_2: tl.constexpr,
    # Entry 7
    NT7: tl.constexpr, PE7_0: tl.constexpr, SH7_0: tl.constexpr, NG7_0: tl.constexpr,
    PE7_1: tl.constexpr, SH7_1: tl.constexpr, NG7_1: tl.constexpr,
    PE7_2: tl.constexpr, SH7_2: tl.constexpr, NG7_2: tl.constexpr,
    # Entry 8
    NT8: tl.constexpr, PE8_0: tl.constexpr, SH8_0: tl.constexpr, NG8_0: tl.constexpr,
    PE8_1: tl.constexpr, SH8_1: tl.constexpr, NG8_1: tl.constexpr,
    PE8_2: tl.constexpr, SH8_2: tl.constexpr, NG8_2: tl.constexpr,
):
    parent_idx = tl.program_id(0)
    if parent_idx >= N_PARENTS:
        return

    # --- Adjacency check ---
    last_suffix = tl.load(Parent_Meta_Ptr + parent_idx) & 0xFF
    adj_offset = last_suffix.to(tl.int32) * 22 + SUFFIX_IDX
    adj_val = tl.load(Adj_Ptr + adj_offset)
    if adj_val == 0:
        return

    # Convert to int64 for pointer arithmetic
    idx = parent_idx.to(tl.int64)

    # --- Compute 9 output entries using descriptors ---
    # Entry 0 (output[0][0])
    if NT0 == 1:
        o0_p0_lo, o0_p0_hi, o0_p1_lo, o0_p1_hi, o0_p2_lo, o0_p2_hi = compute_entry_1term(
            Parents_Ptr, N_STRIDE, idx, PE0_0, SH0_0, NG0_0)
    elif NT0 == 2:
        o0_p0_lo, o0_p0_hi, o0_p1_lo, o0_p1_hi, o0_p2_lo, o0_p2_hi = compute_entry_2terms(
            Parents_Ptr, N_STRIDE, idx, PE0_0, SH0_0, NG0_0, PE0_1, SH0_1, NG0_1)
    else:
        o0_p0_lo, o0_p0_hi, o0_p1_lo, o0_p1_hi, o0_p2_lo, o0_p2_hi = compute_entry_3terms(
            Parents_Ptr, N_STRIDE, idx, PE0_0, SH0_0, NG0_0, PE0_1, SH0_1, NG0_1, PE0_2, SH0_2, NG0_2)

    # Entry 1 (output[0][1])
    if NT1 == 1:
        o1_p0_lo, o1_p0_hi, o1_p1_lo, o1_p1_hi, o1_p2_lo, o1_p2_hi = compute_entry_1term(
            Parents_Ptr, N_STRIDE, idx, PE1_0, SH1_0, NG1_0)
    elif NT1 == 2:
        o1_p0_lo, o1_p0_hi, o1_p1_lo, o1_p1_hi, o1_p2_lo, o1_p2_hi = compute_entry_2terms(
            Parents_Ptr, N_STRIDE, idx, PE1_0, SH1_0, NG1_0, PE1_1, SH1_1, NG1_1)
    else:
        o1_p0_lo, o1_p0_hi, o1_p1_lo, o1_p1_hi, o1_p2_lo, o1_p2_hi = compute_entry_3terms(
            Parents_Ptr, N_STRIDE, idx, PE1_0, SH1_0, NG1_0, PE1_1, SH1_1, NG1_1, PE1_2, SH1_2, NG1_2)

    # Entry 2 (output[0][2])
    if NT2 == 1:
        o2_p0_lo, o2_p0_hi, o2_p1_lo, o2_p1_hi, o2_p2_lo, o2_p2_hi = compute_entry_1term(
            Parents_Ptr, N_STRIDE, idx, PE2_0, SH2_0, NG2_0)
    elif NT2 == 2:
        o2_p0_lo, o2_p0_hi, o2_p1_lo, o2_p1_hi, o2_p2_lo, o2_p2_hi = compute_entry_2terms(
            Parents_Ptr, N_STRIDE, idx, PE2_0, SH2_0, NG2_0, PE2_1, SH2_1, NG2_1)
    else:
        o2_p0_lo, o2_p0_hi, o2_p1_lo, o2_p1_hi, o2_p2_lo, o2_p2_hi = compute_entry_3terms(
            Parents_Ptr, N_STRIDE, idx, PE2_0, SH2_0, NG2_0, PE2_1, SH2_1, NG2_1, PE2_2, SH2_2, NG2_2)

    # Entry 3 (output[1][0])
    if NT3 == 1:
        o3_p0_lo, o3_p0_hi, o3_p1_lo, o3_p1_hi, o3_p2_lo, o3_p2_hi = compute_entry_1term(
            Parents_Ptr, N_STRIDE, idx, PE3_0, SH3_0, NG3_0)
    elif NT3 == 2:
        o3_p0_lo, o3_p0_hi, o3_p1_lo, o3_p1_hi, o3_p2_lo, o3_p2_hi = compute_entry_2terms(
            Parents_Ptr, N_STRIDE, idx, PE3_0, SH3_0, NG3_0, PE3_1, SH3_1, NG3_1)
    else:
        o3_p0_lo, o3_p0_hi, o3_p1_lo, o3_p1_hi, o3_p2_lo, o3_p2_hi = compute_entry_3terms(
            Parents_Ptr, N_STRIDE, idx, PE3_0, SH3_0, NG3_0, PE3_1, SH3_1, NG3_1, PE3_2, SH3_2, NG3_2)

    # Entry 4 (output[1][1])
    if NT4 == 1:
        o4_p0_lo, o4_p0_hi, o4_p1_lo, o4_p1_hi, o4_p2_lo, o4_p2_hi = compute_entry_1term(
            Parents_Ptr, N_STRIDE, idx, PE4_0, SH4_0, NG4_0)
    elif NT4 == 2:
        o4_p0_lo, o4_p0_hi, o4_p1_lo, o4_p1_hi, o4_p2_lo, o4_p2_hi = compute_entry_2terms(
            Parents_Ptr, N_STRIDE, idx, PE4_0, SH4_0, NG4_0, PE4_1, SH4_1, NG4_1)
    else:
        o4_p0_lo, o4_p0_hi, o4_p1_lo, o4_p1_hi, o4_p2_lo, o4_p2_hi = compute_entry_3terms(
            Parents_Ptr, N_STRIDE, idx, PE4_0, SH4_0, NG4_0, PE4_1, SH4_1, NG4_1, PE4_2, SH4_2, NG4_2)

    # Entry 5 (output[1][2])
    if NT5 == 1:
        o5_p0_lo, o5_p0_hi, o5_p1_lo, o5_p1_hi, o5_p2_lo, o5_p2_hi = compute_entry_1term(
            Parents_Ptr, N_STRIDE, idx, PE5_0, SH5_0, NG5_0)
    elif NT5 == 2:
        o5_p0_lo, o5_p0_hi, o5_p1_lo, o5_p1_hi, o5_p2_lo, o5_p2_hi = compute_entry_2terms(
            Parents_Ptr, N_STRIDE, idx, PE5_0, SH5_0, NG5_0, PE5_1, SH5_1, NG5_1)
    else:
        o5_p0_lo, o5_p0_hi, o5_p1_lo, o5_p1_hi, o5_p2_lo, o5_p2_hi = compute_entry_3terms(
            Parents_Ptr, N_STRIDE, idx, PE5_0, SH5_0, NG5_0, PE5_1, SH5_1, NG5_1, PE5_2, SH5_2, NG5_2)

    # Entry 6 (output[2][0])
    if NT6 == 1:
        o6_p0_lo, o6_p0_hi, o6_p1_lo, o6_p1_hi, o6_p2_lo, o6_p2_hi = compute_entry_1term(
            Parents_Ptr, N_STRIDE, idx, PE6_0, SH6_0, NG6_0)
    elif NT6 == 2:
        o6_p0_lo, o6_p0_hi, o6_p1_lo, o6_p1_hi, o6_p2_lo, o6_p2_hi = compute_entry_2terms(
            Parents_Ptr, N_STRIDE, idx, PE6_0, SH6_0, NG6_0, PE6_1, SH6_1, NG6_1)
    else:
        o6_p0_lo, o6_p0_hi, o6_p1_lo, o6_p1_hi, o6_p2_lo, o6_p2_hi = compute_entry_3terms(
            Parents_Ptr, N_STRIDE, idx, PE6_0, SH6_0, NG6_0, PE6_1, SH6_1, NG6_1, PE6_2, SH6_2, NG6_2)

    # Entry 7 (output[2][1])
    if NT7 == 1:
        o7_p0_lo, o7_p0_hi, o7_p1_lo, o7_p1_hi, o7_p2_lo, o7_p2_hi = compute_entry_1term(
            Parents_Ptr, N_STRIDE, idx, PE7_0, SH7_0, NG7_0)
    elif NT7 == 2:
        o7_p0_lo, o7_p0_hi, o7_p1_lo, o7_p1_hi, o7_p2_lo, o7_p2_hi = compute_entry_2terms(
            Parents_Ptr, N_STRIDE, idx, PE7_0, SH7_0, NG7_0, PE7_1, SH7_1, NG7_1)
    else:
        o7_p0_lo, o7_p0_hi, o7_p1_lo, o7_p1_hi, o7_p2_lo, o7_p2_hi = compute_entry_3terms(
            Parents_Ptr, N_STRIDE, idx, PE7_0, SH7_0, NG7_0, PE7_1, SH7_1, NG7_1, PE7_2, SH7_2, NG7_2)

    # Entry 8 (output[2][2])
    if NT8 == 1:
        o8_p0_lo, o8_p0_hi, o8_p1_lo, o8_p1_hi, o8_p2_lo, o8_p2_hi = compute_entry_1term(
            Parents_Ptr, N_STRIDE, idx, PE8_0, SH8_0, NG8_0)
    elif NT8 == 2:
        o8_p0_lo, o8_p0_hi, o8_p1_lo, o8_p1_hi, o8_p2_lo, o8_p2_hi = compute_entry_2terms(
            Parents_Ptr, N_STRIDE, idx, PE8_0, SH8_0, NG8_0, PE8_1, SH8_1, NG8_1)
    else:
        o8_p0_lo, o8_p0_hi, o8_p1_lo, o8_p1_hi, o8_p2_lo, o8_p2_hi = compute_entry_3terms(
            Parents_Ptr, N_STRIDE, idx, PE8_0, SH8_0, NG8_0, PE8_1, SH8_1, NG8_1, PE8_2, SH8_2, NG8_2)

    # --- Compute ProjLen (max_degree - min_degree) ---
    all_lo = o0_p0_lo.to(tl.uint64) | o0_p1_lo.to(tl.uint64) | o0_p2_lo.to(tl.uint64)
    all_hi = o0_p0_hi.to(tl.uint64) | o0_p1_hi.to(tl.uint64) | o0_p2_hi.to(tl.uint64)
    all_lo = all_lo | o1_p0_lo.to(tl.uint64) | o1_p1_lo.to(tl.uint64) | o1_p2_lo.to(tl.uint64)
    all_hi = all_hi | o1_p0_hi.to(tl.uint64) | o1_p1_hi.to(tl.uint64) | o1_p2_hi.to(tl.uint64)
    all_lo = all_lo | o2_p0_lo.to(tl.uint64) | o2_p1_lo.to(tl.uint64) | o2_p2_lo.to(tl.uint64)
    all_hi = all_hi | o2_p0_hi.to(tl.uint64) | o2_p1_hi.to(tl.uint64) | o2_p2_hi.to(tl.uint64)
    all_lo = all_lo | o3_p0_lo.to(tl.uint64) | o3_p1_lo.to(tl.uint64) | o3_p2_lo.to(tl.uint64)
    all_hi = all_hi | o3_p0_hi.to(tl.uint64) | o3_p1_hi.to(tl.uint64) | o3_p2_hi.to(tl.uint64)
    all_lo = all_lo | o4_p0_lo.to(tl.uint64) | o4_p1_lo.to(tl.uint64) | o4_p2_lo.to(tl.uint64)
    all_hi = all_hi | o4_p0_hi.to(tl.uint64) | o4_p1_hi.to(tl.uint64) | o4_p2_hi.to(tl.uint64)
    all_lo = all_lo | o5_p0_lo.to(tl.uint64) | o5_p1_lo.to(tl.uint64) | o5_p2_lo.to(tl.uint64)
    all_hi = all_hi | o5_p0_hi.to(tl.uint64) | o5_p1_hi.to(tl.uint64) | o5_p2_hi.to(tl.uint64)
    all_lo = all_lo | o6_p0_lo.to(tl.uint64) | o6_p1_lo.to(tl.uint64) | o6_p2_lo.to(tl.uint64)
    all_hi = all_hi | o6_p0_hi.to(tl.uint64) | o6_p1_hi.to(tl.uint64) | o6_p2_hi.to(tl.uint64)
    all_lo = all_lo | o7_p0_lo.to(tl.uint64) | o7_p1_lo.to(tl.uint64) | o7_p2_lo.to(tl.uint64)
    all_hi = all_hi | o7_p0_hi.to(tl.uint64) | o7_p1_hi.to(tl.uint64) | o7_p2_hi.to(tl.uint64)
    all_lo = all_lo | o8_p0_lo.to(tl.uint64) | o8_p1_lo.to(tl.uint64) | o8_p2_lo.to(tl.uint64)
    all_hi = all_hi | o8_p0_hi.to(tl.uint64) | o8_p1_hi.to(tl.uint64) | o8_p2_hi.to(tl.uint64)

    # Check for zero matrix (kernel element!)
    is_zero_matrix = (all_lo == 0) & (all_hi == 0)

    # Max degree
    max_deg_hi = msb64(all_hi)
    max_deg_lo = msb64(all_lo)
    has_hi = (all_hi != 0)
    max_deg = tl.where(has_hi, max_deg_hi + 64, max_deg_lo)

    # Min degree
    min_deg_lo = lsb64(all_lo)
    min_deg_hi = lsb64(all_hi)
    has_lo = (all_lo != 0)
    min_deg = tl.where(has_lo, min_deg_lo, min_deg_hi + 64)

    projlen = tl.where(is_zero_matrix, tl.zeros([], dtype=tl.int32), max_deg - min_deg)

    # --- Normalize: right-shift all polynomials by min_deg ---
    s_norm = tl.where(is_zero_matrix, tl.zeros([], dtype=tl.int32), min_deg)
    o0_p0_lo, o0_p0_hi = shr128(o0_p0_lo, o0_p0_hi, s_norm)
    o0_p1_lo, o0_p1_hi = shr128(o0_p1_lo, o0_p1_hi, s_norm)
    o0_p2_lo, o0_p2_hi = shr128(o0_p2_lo, o0_p2_hi, s_norm)
    o1_p0_lo, o1_p0_hi = shr128(o1_p0_lo, o1_p0_hi, s_norm)
    o1_p1_lo, o1_p1_hi = shr128(o1_p1_lo, o1_p1_hi, s_norm)
    o1_p2_lo, o1_p2_hi = shr128(o1_p2_lo, o1_p2_hi, s_norm)
    o2_p0_lo, o2_p0_hi = shr128(o2_p0_lo, o2_p0_hi, s_norm)
    o2_p1_lo, o2_p1_hi = shr128(o2_p1_lo, o2_p1_hi, s_norm)
    o2_p2_lo, o2_p2_hi = shr128(o2_p2_lo, o2_p2_hi, s_norm)
    o3_p0_lo, o3_p0_hi = shr128(o3_p0_lo, o3_p0_hi, s_norm)
    o3_p1_lo, o3_p1_hi = shr128(o3_p1_lo, o3_p1_hi, s_norm)
    o3_p2_lo, o3_p2_hi = shr128(o3_p2_lo, o3_p2_hi, s_norm)
    o4_p0_lo, o4_p0_hi = shr128(o4_p0_lo, o4_p0_hi, s_norm)
    o4_p1_lo, o4_p1_hi = shr128(o4_p1_lo, o4_p1_hi, s_norm)
    o4_p2_lo, o4_p2_hi = shr128(o4_p2_lo, o4_p2_hi, s_norm)
    o5_p0_lo, o5_p0_hi = shr128(o5_p0_lo, o5_p0_hi, s_norm)
    o5_p1_lo, o5_p1_hi = shr128(o5_p1_lo, o5_p1_hi, s_norm)
    o5_p2_lo, o5_p2_hi = shr128(o5_p2_lo, o5_p2_hi, s_norm)
    o6_p0_lo, o6_p0_hi = shr128(o6_p0_lo, o6_p0_hi, s_norm)
    o6_p1_lo, o6_p1_hi = shr128(o6_p1_lo, o6_p1_hi, s_norm)
    o6_p2_lo, o6_p2_hi = shr128(o6_p2_lo, o6_p2_hi, s_norm)
    o7_p0_lo, o7_p0_hi = shr128(o7_p0_lo, o7_p0_hi, s_norm)
    o7_p1_lo, o7_p1_hi = shr128(o7_p1_lo, o7_p1_hi, s_norm)
    o7_p2_lo, o7_p2_hi = shr128(o7_p2_lo, o7_p2_hi, s_norm)
    o8_p0_lo, o8_p0_hi = shr128(o8_p0_lo, o8_p0_hi, s_norm)
    o8_p1_lo, o8_p1_hi = shr128(o8_p1_lo, o8_p1_hi, s_norm)
    o8_p2_lo, o8_p2_hi = shr128(o8_p2_lo, o8_p2_hi, s_norm)

    # --- FCFS bucket check ---
    bucket_slot = tl.atomic_add(Bucket_Counters_Ptr + projlen, 1)
    if bucket_slot >= BUCKET_CAP_PARAM:
        return  # Bucket full, discard

    # --- Reserve global output slot ---
    global_slot = tl.atomic_add(Global_Counter_Ptr, 1)
    if global_slot >= OUTPUT_CAP_PARAM:
        return  # Buffer full

    # --- Write output in SoA layout ---
    out_base = global_slot.to(tl.int64)
    tl.store(Output_Ptr + 0  * OUT_STRIDE + out_base, o0_p0_lo.to(tl.int64))
    tl.store(Output_Ptr + 1  * OUT_STRIDE + out_base, o0_p0_hi.to(tl.int64))
    tl.store(Output_Ptr + 2  * OUT_STRIDE + out_base, o0_p1_lo.to(tl.int64))
    tl.store(Output_Ptr + 3  * OUT_STRIDE + out_base, o0_p1_hi.to(tl.int64))
    tl.store(Output_Ptr + 4  * OUT_STRIDE + out_base, o0_p2_lo.to(tl.int64))
    tl.store(Output_Ptr + 5  * OUT_STRIDE + out_base, o0_p2_hi.to(tl.int64))
    tl.store(Output_Ptr + 6  * OUT_STRIDE + out_base, o1_p0_lo.to(tl.int64))
    tl.store(Output_Ptr + 7  * OUT_STRIDE + out_base, o1_p0_hi.to(tl.int64))
    tl.store(Output_Ptr + 8  * OUT_STRIDE + out_base, o1_p1_lo.to(tl.int64))
    tl.store(Output_Ptr + 9  * OUT_STRIDE + out_base, o1_p1_hi.to(tl.int64))
    tl.store(Output_Ptr + 10 * OUT_STRIDE + out_base, o1_p2_lo.to(tl.int64))
    tl.store(Output_Ptr + 11 * OUT_STRIDE + out_base, o1_p2_hi.to(tl.int64))
    tl.store(Output_Ptr + 12 * OUT_STRIDE + out_base, o2_p0_lo.to(tl.int64))
    tl.store(Output_Ptr + 13 * OUT_STRIDE + out_base, o2_p0_hi.to(tl.int64))
    tl.store(Output_Ptr + 14 * OUT_STRIDE + out_base, o2_p1_lo.to(tl.int64))
    tl.store(Output_Ptr + 15 * OUT_STRIDE + out_base, o2_p1_hi.to(tl.int64))
    tl.store(Output_Ptr + 16 * OUT_STRIDE + out_base, o2_p2_lo.to(tl.int64))
    tl.store(Output_Ptr + 17 * OUT_STRIDE + out_base, o2_p2_hi.to(tl.int64))
    tl.store(Output_Ptr + 18 * OUT_STRIDE + out_base, o3_p0_lo.to(tl.int64))
    tl.store(Output_Ptr + 19 * OUT_STRIDE + out_base, o3_p0_hi.to(tl.int64))
    tl.store(Output_Ptr + 20 * OUT_STRIDE + out_base, o3_p1_lo.to(tl.int64))
    tl.store(Output_Ptr + 21 * OUT_STRIDE + out_base, o3_p1_hi.to(tl.int64))
    tl.store(Output_Ptr + 22 * OUT_STRIDE + out_base, o3_p2_lo.to(tl.int64))
    tl.store(Output_Ptr + 23 * OUT_STRIDE + out_base, o3_p2_hi.to(tl.int64))
    tl.store(Output_Ptr + 24 * OUT_STRIDE + out_base, o4_p0_lo.to(tl.int64))
    tl.store(Output_Ptr + 25 * OUT_STRIDE + out_base, o4_p0_hi.to(tl.int64))
    tl.store(Output_Ptr + 26 * OUT_STRIDE + out_base, o4_p1_lo.to(tl.int64))
    tl.store(Output_Ptr + 27 * OUT_STRIDE + out_base, o4_p1_hi.to(tl.int64))
    tl.store(Output_Ptr + 28 * OUT_STRIDE + out_base, o4_p2_lo.to(tl.int64))
    tl.store(Output_Ptr + 29 * OUT_STRIDE + out_base, o4_p2_hi.to(tl.int64))
    tl.store(Output_Ptr + 30 * OUT_STRIDE + out_base, o5_p0_lo.to(tl.int64))
    tl.store(Output_Ptr + 31 * OUT_STRIDE + out_base, o5_p0_hi.to(tl.int64))
    tl.store(Output_Ptr + 32 * OUT_STRIDE + out_base, o5_p1_lo.to(tl.int64))
    tl.store(Output_Ptr + 33 * OUT_STRIDE + out_base, o5_p1_hi.to(tl.int64))
    tl.store(Output_Ptr + 34 * OUT_STRIDE + out_base, o5_p2_lo.to(tl.int64))
    tl.store(Output_Ptr + 35 * OUT_STRIDE + out_base, o5_p2_hi.to(tl.int64))
    tl.store(Output_Ptr + 36 * OUT_STRIDE + out_base, o6_p0_lo.to(tl.int64))
    tl.store(Output_Ptr + 37 * OUT_STRIDE + out_base, o6_p0_hi.to(tl.int64))
    tl.store(Output_Ptr + 38 * OUT_STRIDE + out_base, o6_p1_lo.to(tl.int64))
    tl.store(Output_Ptr + 39 * OUT_STRIDE + out_base, o6_p1_hi.to(tl.int64))
    tl.store(Output_Ptr + 40 * OUT_STRIDE + out_base, o6_p2_lo.to(tl.int64))
    tl.store(Output_Ptr + 41 * OUT_STRIDE + out_base, o6_p2_hi.to(tl.int64))
    tl.store(Output_Ptr + 42 * OUT_STRIDE + out_base, o7_p0_lo.to(tl.int64))
    tl.store(Output_Ptr + 43 * OUT_STRIDE + out_base, o7_p0_hi.to(tl.int64))
    tl.store(Output_Ptr + 44 * OUT_STRIDE + out_base, o7_p1_lo.to(tl.int64))
    tl.store(Output_Ptr + 45 * OUT_STRIDE + out_base, o7_p1_hi.to(tl.int64))
    tl.store(Output_Ptr + 46 * OUT_STRIDE + out_base, o7_p2_lo.to(tl.int64))
    tl.store(Output_Ptr + 47 * OUT_STRIDE + out_base, o7_p2_hi.to(tl.int64))
    tl.store(Output_Ptr + 48 * OUT_STRIDE + out_base, o8_p0_lo.to(tl.int64))
    tl.store(Output_Ptr + 49 * OUT_STRIDE + out_base, o8_p0_hi.to(tl.int64))
    tl.store(Output_Ptr + 50 * OUT_STRIDE + out_base, o8_p1_lo.to(tl.int64))
    tl.store(Output_Ptr + 51 * OUT_STRIDE + out_base, o8_p1_hi.to(tl.int64))
    tl.store(Output_Ptr + 52 * OUT_STRIDE + out_base, o8_p2_lo.to(tl.int64))
    tl.store(Output_Ptr + 53 * OUT_STRIDE + out_base, o8_p2_hi.to(tl.int64))

    # --- Write metadata: projlen << 8 | suffix_idx ---
    meta = (projlen << 8) | SUFFIX_IDX
    tl.store(Output_Meta_Ptr + out_base, meta)

    # --- Write parent index for word reconstruction ---
    tl.store(Output_Parent_Ptr + out_base, parent_idx)

    # --- Flag zero matrices (kernel elements!) ---
    if is_zero_matrix:
        tl.atomic_add(Bucket_Counters_Ptr + 127, 1000000)


# ==============================================================================
# HOST FUNCTIONS
# ==============================================================================

def build_suffix_kwargs(s):
    """Build the compile-time descriptor kwargs for suffix s."""
    base_nt = s * 9
    base_desc = s * 27
    kwargs = {"SUFFIX_IDX": s}
    for entry in range(9):
        nt = DESC_NTERMS[base_nt + entry]
        kwargs[f"NT{entry}"] = nt
        for t in range(3):
            idx = base_desc + entry * 3 + t
            kwargs[f"PE{entry}_{t}"] = DESC_PE[idx]
            kwargs[f"SH{entry}_{t}"] = DESC_SHIFT[idx]
            kwargs[f"NG{entry}_{t}"] = DESC_NEG[idx]
    return kwargs

# Pre-compute suffix kwargs for all 22 suffixes
SUFFIX_KWARGS = [build_suffix_kwargs(s) for s in range(N_SUFFIXES)]

ADJ_TABLE = [[1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1], [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1], [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1], [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]]

def build_adjacency_tensor():
    """Build adjacency table as int8 tensor on GPU."""
    flat = []
    for row in ADJ_TABLE:
        flat.extend(row)
    return torch.tensor(flat, dtype=torch.int8, device="cuda")

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


def build_seed_braids(n_stride):
    """
    Build the 22 seed braids (one per suffix applied to identity).
    Each seed is the suffix matrix itself in bit-sliced form.
    Returns (data [54, n_stride] int64 SoA, meta [n_stride] int32) on GPU.
    """
    matrices = get_raw_matrix_data_pruned_host()

    # Build in AoS first, then transpose
    data_aos = torch.zeros((22, 54), dtype=torch.int64, device="cpu")
    meta = torch.zeros((n_stride,), dtype=torch.int32, device="cpu")

    for s in range(22):
        mat = matrices[s]
        for r in range(3):
            for c in range(3):
                poly_idx = r * 3 + c
                base = poly_idx * 6  # 6 uint64s per polynomial

                for (deg, coeff) in mat[r][c]:
                    if deg < 64:
                        word = 0  # lo
                        bit_pos = deg
                    else:
                        word = 1  # hi
                        bit_pos = deg - 64

                    bit_val = 1 << bit_pos

                    if coeff == 1:
                        data_aos[s, base + 0 * 2 + word] |= bit_val
                    else:  # coeff == -1 -> 4 = 0b100
                        data_aos[s, base + 2 * 2 + word] |= bit_val

        # Compute projlen for seed
        all_bits = 0
        for idx in range(9):
            base_idx = idx * 6
            for p in range(3):
                lo_val = data_aos[s, base_idx + p * 2 + 0].item()
                hi_val = data_aos[s, base_idx + p * 2 + 1].item()
                if lo_val < 0:
                    lo_val += (1 << 64)
                if hi_val < 0:
                    hi_val += (1 << 64)
                all_bits |= lo_val
                all_bits |= (hi_val << 64)

        if all_bits == 0:
            projlen = 0
        else:
            max_deg = all_bits.bit_length() - 1
            min_deg = 0
            tmp = all_bits
            while tmp and not (tmp & 1):
                min_deg += 1
                tmp >>= 1
            projlen = max_deg - min_deg

            # Normalize: shift all polynomials right by min_deg
            if min_deg > 0:
                for idx in range(9):
                    base_idx = idx * 6
                    for p in range(3):
                        lo_val = data_aos[s, base_idx + p * 2 + 0].item()
                        hi_val = data_aos[s, base_idx + p * 2 + 1].item()
                        if lo_val < 0:
                            lo_val += (1 << 64)
                        if hi_val < 0:
                            hi_val += (1 << 64)
                        val128 = lo_val | (hi_val << 64)
                        val128 >>= min_deg
                        new_lo = val128 & ((1 << 64) - 1)
                        new_hi = (val128 >> 64) & ((1 << 64) - 1)
                        if new_lo >= (1 << 63):
                            new_lo -= (1 << 64)
                        if new_hi >= (1 << 63):
                            new_hi -= (1 << 64)
                        data_aos[s, base_idx + p * 2 + 0] = new_lo
                        data_aos[s, base_idx + p * 2 + 1] = new_hi

        meta[s] = (projlen << 8) | s

    # Transpose to SoA: [54, n_stride]
    data_soa = torch.zeros((54, n_stride), dtype=torch.int64, device="cpu")
    data_soa[:, :22] = data_aos.t()

    return data_soa.cuda(), meta.cuda()


def save_projlen0_braids(zero_data_soa, zero_meta, zero_words, braid_length, save_dir="projlen0_results"):
    """
    Save pre-extracted projlen-0 braids to disk.
    Converts SoA back to AoS for compatibility with decode/verify scripts.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    n_zeros = zero_meta.shape[0]
    if n_zeros == 0:
        return 0

    # Convert SoA [54, n] -> AoS [n, 54] for saved format compatibility
    zero_data_aos = zero_data_soa.t().cpu()
    zero_meta = zero_meta.cpu()

    save_path = os.path.join(save_dir, f"projlen0_length{braid_length:03d}.pt")

    if os.path.exists(save_path):
        existing = torch.load(save_path, weights_only=True)
        zero_data_aos = torch.cat([existing["data"], zero_data_aos], dim=0)
        zero_meta = torch.cat([existing["meta"], zero_meta], dim=0)
        zero_words = torch.cat([existing["words"], zero_words], dim=0)

    torch.save({
        "data": zero_data_aos,
        "meta": zero_meta,
        "words": zero_words,
        "braid_length": braid_length,
    }, save_path)

    return n_zeros


def run_search():
    """
    Main search loop.
    Runs from braid length 1 (the 22 seeds) through length 128.
    Tracks Garside words via parent_idx output from kernel.
    Saves projlen-0 braids with their full Garside words.
    """
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    device = torch.device("cuda")

    # Seed RNG from system entropy so each run explores different braids
    import os
    seed = int.from_bytes(os.urandom(4), 'little')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    gpu_name = torch.cuda.get_device_name()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Device: {gpu_name} ({vram_gb:.1f} GB)")
    print(f"Random seed: {seed}")
    print(f"Config: USE_BEST={USE_BEST:,}, OUTPUT_CAP={OUTPUT_CAP:,}, BUCKET_CAP={BUCKET_CAP:,}")
    bytes_per_braid = 54 * 8 + 4 + 4  # data + meta + parent_idx
    print(f"GPU memory per braid: {bytes_per_braid} bytes")
    print(f"Parent buffer:  {USE_BEST * bytes_per_braid / (1024**3):.1f} GB")
    print(f"Output buffer:  {OUTPUT_CAP * bytes_per_braid / (1024**3):.1f} GB")
    print(f"Word buffer:    ~{USE_BEST * (MAX_STEPS+2) / (1024**3):.2f} GB CPU")
    print(f"Running lengths 1 through {MAX_STEPS + 1}")
    print(f"Layout: SoA [54, N], fused kernel with {N_SUFFIXES} suffixes")
    print("=" * 100)

    # --- Build adjacency table ---
    adj_tensor = build_adjacency_tensor()

    # --- Stride for SoA buffers ---
    n_stride = USE_BEST   # Parent buffer stride
    out_stride = OUTPUT_CAP  # Output buffer stride

    # --- Build seed braids (length 1) in SoA format ---
    parent_data, parent_meta = build_seed_braids(n_stride)
    n_parents = 22

    # Word tracking on CPU: parent_words[i, :word_len] = Garside word for parent i
    parent_words = torch.zeros((n_parents, MAX_STEPS + 2), dtype=torch.uint8)
    for s in range(22):
        parent_words[s, 0] = s
    word_len = 1

    # Check seeds for projlen 0
    seed_projlens = (parent_meta[:n_parents] >> 8) & 0x7F
    n_seed_zeros = (seed_projlens == 0).sum().item()
    print(f"Length  1: {n_parents} seed braids, {n_seed_zeros} with projlen 0")

    total_projlen0 = 0
    if n_seed_zeros > 0:
        os.makedirs("projlen0_results", exist_ok=True)
        zero_indices = torch.where(seed_projlens == 0)[0]
        zi_cpu = zero_indices.cpu()
        # Convert SoA back to AoS for saving
        seed_data_aos = parent_data[:, zero_indices].t().cpu()
        torch.save({
            "data": seed_data_aos,
            "meta": parent_meta[zero_indices].cpu(),
            "words": parent_words[zi_cpu, :word_len],
            "braid_length": 1,
        }, "projlen0_results/projlen0_length001.pt")
        total_projlen0 += n_seed_zeros

    # --- Allocate output buffers (SoA) ---
    out_data = torch.zeros((54, out_stride), dtype=torch.int64, device=device)
    out_meta = torch.zeros((out_stride,), dtype=torch.int32, device=device)
    out_parent_idx = torch.zeros((out_stride,), dtype=torch.int32, device=device)
    global_counter = torch.zeros((1,), dtype=torch.int32, device=device)
    bucket_counters = torch.zeros((N_BUCKETS,), dtype=torch.int32, device=device)

    # --- Main loop ---
    for step in range(MAX_STEPS):
        braid_length = step + 2
        t0 = time.time()

        # Reset counters
        global_counter.zero_()
        bucket_counters.zero_()

        # Shuffle parents (critical for FCFS uniformity)
        if n_parents > 1:
            perm = torch.randperm(n_parents, device=device)
            parent_data[:, :n_parents] = parent_data[:, perm].clone()
            parent_meta[:n_parents] = parent_meta[perm].clone()
            parent_words = parent_words[perm.cpu()]

        # Launch 22 suffix kernels in random order
        grid = (n_parents,)
        # SoA stride = second dimension of the [54, N] buffer
        n_stride_val = parent_data.shape[1]
        out_stride_val = out_data.shape[1]

        suffix_order = torch.randperm(N_SUFFIXES).tolist()
        for s in suffix_order:
            kw = SUFFIX_KWARGS[s]
            kernel_braid_step[grid](
                parent_data,
                parent_meta,
                out_data,
                out_meta,
                out_parent_idx,
                global_counter,
                bucket_counters,
                adj_tensor,
                n_parents,
                n_stride_val,
                out_stride_val,
                OUTPUT_CAP,
                BUCKET_CAP,
                num_warps=1,
                **kw,
            )

        torch.cuda.synchronize()

        n_children = min(global_counter.item(), OUTPUT_CAP)
        bucket_counts = bucket_counters.cpu().tolist()

        if n_children == 0:
            t1 = time.time()
            print(f"Length {braid_length:3d} | {t1-t0:.2f}s | NO CHILDREN. Search exhausted.")
            break

        # --- Select survivors FIRST, then build words only for kept + projlen-0 ---
        child_projlens = (out_meta[:n_children] >> 8).to(torch.int32) & 0x7F

        if n_children <= USE_BEST:
            keep_indices = torch.arange(n_children, device=device)
            n_keep = n_children
        else:
            jittered = child_projlens[:n_children] * 2 + torch.randint(0, 2, (n_children,), device=device)
            sorted_indices = torch.argsort(jittered)
            keep_indices = sorted_indices[:USE_BEST]
            n_keep = USE_BEST

        # Find projlen-0 children
        child_projlens_cpu = child_projlens[:n_children].cpu()
        zero_mask = (child_projlens_cpu == 0)
        n_zeros_this_step = zero_mask.sum().item()

        # Save projlen-0 braids
        if n_zeros_this_step > 0:
            zero_indices_cpu = torch.where(zero_mask)[0]
            zero_indices_gpu = zero_indices_cpu.to(device)

            zero_parent_indices = out_parent_idx[zero_indices_gpu].cpu()
            zero_suffixes = (out_meta[zero_indices_gpu].cpu() & 0xFF).to(torch.uint8)
            zero_words = torch.zeros((n_zeros_this_step, braid_length), dtype=torch.uint8)
            zero_words[:, :word_len] = parent_words[zero_parent_indices.long(), :word_len]
            zero_words[:, word_len] = zero_suffixes

            zero_data_soa = out_data[:, zero_indices_gpu]
            zero_meta_save = out_meta[zero_indices_gpu]

            save_projlen0_braids(zero_data_soa, zero_meta_save, zero_words, braid_length)

        total_projlen0 += n_zeros_this_step

        # Build words for kept braids
        kept_parent_indices = out_parent_idx[keep_indices].cpu()
        kept_suffixes = (out_meta[keep_indices].cpu() & 0xFF).to(torch.uint8)
        kept_words = torch.zeros((n_keep, braid_length), dtype=torch.uint8)
        kept_words[:, :word_len] = parent_words[kept_parent_indices.long(), :word_len]
        kept_words[:, word_len] = kept_suffixes

        # Copy kept children to parent buffer (SoA)
        # Resize parent_data if needed
        if n_keep > parent_data.shape[1]:
            parent_data = torch.zeros((54, n_keep), dtype=torch.int64, device=device)
            parent_meta = torch.zeros((n_keep,), dtype=torch.int32, device=device)
        parent_data[:, :n_keep] = out_data[:, keep_indices]
        parent_meta[:n_keep] = out_meta[keep_indices]
        parent_words = kept_words
        n_parents = n_keep
        word_len = braid_length

        t1 = time.time()
        dt = t1 - t0

        # Stats
        kept_projlens = (parent_meta[:n_parents] >> 8).to(torch.int32) & 0x7F
        min_pl = kept_projlens.min().item()
        max_pl = kept_projlens.max().item()
        mean_pl = kept_projlens.float().mean().item()

        nonzero_buckets = [(i, c) for i, c in enumerate(bucket_counts[:127]) if c > 0]
        bucket_str = " ".join(f"[{b}]:{c}" for b, c in nonzero_buckets[:8])
        if len(nonzero_buckets) > 8:
            bucket_str += " ..."

        zeros_str = f" *** {n_zeros_this_step} PROJLEN-0 ***" if n_zeros_this_step > 0 else ""

        print(f"Length {braid_length:3d} | {dt:5.2f}s | children={n_children:>10,} | "
              f"kept={n_parents:>10,} | projlen {min_pl}/{mean_pl:.1f}/{max_pl} | "
              f"buckets: {bucket_str}{zeros_str}")

    # --- Summary ---
    print("=" * 100)
    print(f"Search complete. Total projlen-0 braids found: {total_projlen0}")
    if total_projlen0 > 0:
        import os, glob
        files = sorted(glob.glob("projlen0_results/projlen0_length*.pt"))
        for f in files:
            info = torch.load(f, weights_only=True)
            n = info["data"].shape[0]
            print(f"  {os.path.basename(f)}: {n} braids")
    print("Done.")


if __name__ == "__main__":
    run_search()
