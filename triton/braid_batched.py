#!/usr/bin/env python3
"""
Batched Braid Search — scales beyond single-GPU VRAM limits.

Parents live on CPU; GPU processes them in batches. Children accumulate on CPU,
then one big selection picks the best for the next step.

Usage:
    python braid_batched.py --total-use-best 100 --gpu-batch-size 5
    python braid_batched.py --total-use-best 0.1 --gpu-batch-size 0.05  # small test
"""

import torch
import time
import os
import sys
import argparse
import math

# ==============================================================================
# COMMAND LINE ARGUMENTS
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Batched GPU Braid Search")
    parser.add_argument("--total-use-best", type=float, default=100.0,
                        help="Total parents to keep per step, in millions (default: 100)")
    parser.add_argument("--gpu-batch-size", type=float, default=5.0,
                        help="Parents per GPU batch, in millions (default: 5)")
    parser.add_argument("--gpu-output-cap", type=float, default=None,
                        help="Children buffer per GPU batch, in millions (default: 8x gpu-batch-size)")
    parser.add_argument("--bucket-cap", type=float, default=2.5,
                        help="Per-projlen FCFS cap per batch, in millions (default: 2.5)")
    parser.add_argument("--max-steps", type=int, default=127,
                        help="MAX_STEPS (default: 127)")
    return parser.parse_args()


# ==============================================================================
# IMPORT FROM braid_search.py (suppress its arg parsing)
# ==============================================================================

_saved_argv = sys.argv
sys.argv = ['braid_search']
from braid_search import (
    kernel_braid_step, SUFFIX_KWARGS, N_SUFFIXES, N_BUCKETS,
    build_adjacency_tensor, build_seed_braids, save_projlen0_braids,
)
sys.argv = _saved_argv

args = parse_args()

TOTAL_USE_BEST = int(args.total_use_best * 1_000_000)
GPU_BATCH_SIZE = int(args.gpu_batch_size * 1_000_000)
GPU_OUTPUT_CAP = int((args.gpu_output_cap if args.gpu_output_cap is not None
                      else args.gpu_batch_size * 8) * 1_000_000)
BUCKET_CAP     = int(args.bucket_cap * 1_000_000)
MAX_STEPS      = args.max_steps

N_BATCHES = math.ceil(TOTAL_USE_BEST / GPU_BATCH_SIZE)
TOTAL_ACC_CAP = N_BATCHES * GPU_OUTPUT_CAP


def print_memory_budget():
    """Print estimated memory usage and warn if tight."""
    bytes_data = 54 * 8  # per braid SoA data
    bytes_meta = 4       # per braid meta

    cpu_parents_data = bytes_data * TOTAL_USE_BEST
    cpu_parents_meta = bytes_meta * TOTAL_USE_BEST
    cpu_parents_words = TOTAL_USE_BEST * (MAX_STEPS + 2)
    cpu_acc_data = bytes_data * TOTAL_ACC_CAP
    cpu_acc_meta = 4 * TOTAL_ACC_CAP      # meta
    cpu_acc_pidx = 4 * TOTAL_ACC_CAP      # parent_idx
    cpu_total = (cpu_parents_data + cpu_parents_meta + cpu_parents_words +
                 cpu_acc_data + cpu_acc_meta + cpu_acc_pidx)

    gpu_parent = (bytes_data + bytes_meta) * GPU_BATCH_SIZE
    gpu_output = (bytes_data + bytes_meta + 4) * GPU_OUTPUT_CAP
    gpu_total = gpu_parent + gpu_output

    print(f"Config: TOTAL_USE_BEST={TOTAL_USE_BEST:,}, GPU_BATCH_SIZE={GPU_BATCH_SIZE:,}, "
          f"GPU_OUTPUT_CAP={GPU_OUTPUT_CAP:,}, BUCKET_CAP={BUCKET_CAP:,}")
    print(f"Batches per step: {N_BATCHES}, accumulator capacity: {TOTAL_ACC_CAP:,}")
    print(f"Estimated CPU memory: {cpu_total / (1024**3):.1f} GB")
    print(f"  Parents data:  {cpu_parents_data / (1024**3):.1f} GB")
    print(f"  Parents meta:  {cpu_parents_meta / (1024**3):.2f} GB")
    print(f"  Parents words: {cpu_parents_words / (1024**3):.1f} GB")
    print(f"  Accumulator:   {(cpu_acc_data + cpu_acc_meta + cpu_acc_pidx) / (1024**3):.1f} GB")
    print(f"Estimated GPU memory: {gpu_total / (1024**3):.1f} GB")

    try:
        import psutil
        available_ram = psutil.virtual_memory().available
        if cpu_total > available_ram * 0.9:
            print(f"*** WARNING: estimated CPU memory ({cpu_total / (1024**3):.1f} GB) "
                  f"exceeds 90% of available RAM ({available_ram / (1024**3):.1f} GB) ***")
    except ImportError:
        pass


def run_batched_search():
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    device = torch.device("cuda")

    seed = int.from_bytes(os.urandom(4), 'little')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    gpu_name = torch.cuda.get_device_name()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Device: {gpu_name} ({vram_gb:.1f} GB)")
    print(f"Random seed: {seed}")
    print_memory_budget()
    print(f"Running lengths 1 through {MAX_STEPS + 1}")
    print("=" * 100)

    # --- Build adjacency table on GPU ---
    adj_tensor = build_adjacency_tensor()

    # --- Seed braids (length 1) ---
    # build_seed_braids returns GPU tensors; move to CPU for our flow
    seed_data_gpu, seed_meta_gpu = build_seed_braids(22)
    n_parents = 22

    # CPU parent buffers (no pin_memory — avoids page-locking on shared nodes)
    cpu_data = torch.zeros((54, TOTAL_USE_BEST), dtype=torch.int64)
    cpu_meta = torch.zeros((TOTAL_USE_BEST,), dtype=torch.int32)
    cpu_data[:, :n_parents] = seed_data_gpu.cpu()
    cpu_meta[:n_parents] = seed_meta_gpu.cpu()

    # Word tracking on CPU
    parent_words = torch.zeros((n_parents, MAX_STEPS + 2), dtype=torch.uint8)
    for s in range(22):
        parent_words[s, 0] = s
    word_len = 1

    # Check seeds for projlen 0
    seed_projlens = (cpu_meta[:n_parents] >> 8) & 0x7F
    n_seed_zeros = (seed_projlens == 0).sum().item()
    print(f"Length  1: {n_parents} seed braids, {n_seed_zeros} with projlen 0")

    total_projlen0 = 0
    if n_seed_zeros > 0:
        os.makedirs("projlen0_results", exist_ok=True)
        zero_indices = torch.where(seed_projlens == 0)[0]
        zero_data_soa = cpu_data[:, zero_indices].cuda()
        zero_meta_save = cpu_meta[zero_indices].cuda()
        zero_words = parent_words[zero_indices, :word_len]
        save_projlen0_braids(zero_data_soa, zero_meta_save, zero_words, 1)
        total_projlen0 += n_seed_zeros

    # --- Allocate GPU buffers (reused across batches) ---
    gpu_parent_data = torch.zeros((54, GPU_BATCH_SIZE), dtype=torch.int64, device=device)
    gpu_parent_meta = torch.zeros((GPU_BATCH_SIZE,), dtype=torch.int32, device=device)
    gpu_out_data = torch.zeros((54, GPU_OUTPUT_CAP), dtype=torch.int64, device=device)
    gpu_out_meta = torch.zeros((GPU_OUTPUT_CAP,), dtype=torch.int32, device=device)
    gpu_out_parent_idx = torch.zeros((GPU_OUTPUT_CAP,), dtype=torch.int32, device=device)
    global_counter = torch.zeros((1,), dtype=torch.int32, device=device)
    bucket_counters = torch.zeros((N_BUCKETS,), dtype=torch.int32, device=device)

    del seed_data_gpu, seed_meta_gpu

    # --- Pre-allocate CPU accumulator (reused across steps) ---
    acc_data = torch.zeros((54, TOTAL_ACC_CAP), dtype=torch.int64)
    acc_meta = torch.zeros((TOTAL_ACC_CAP,), dtype=torch.int32)
    acc_parent_idx = torch.zeros((TOTAL_ACC_CAP,), dtype=torch.int32)

    # --- Main loop ---
    for step in range(MAX_STEPS):
        braid_length = step + 2
        t0 = time.time()

        # Virtual shuffle: generate permutation indices instead of physically
        # moving data. Avoids a full-size .clone() temporary.
        if n_parents > 1:
            perm = torch.randperm(n_parents)
        else:
            perm = torch.arange(1)

        # Number of batches for this step (may be fewer if n_parents < TOTAL_USE_BEST)
        n_batches_this_step = math.ceil(n_parents / GPU_BATCH_SIZE)
        total_acc_cap = n_batches_this_step * GPU_OUTPUT_CAP
        acc_count = 0

        n_zeros_this_step = 0

        for b in range(n_batches_this_step):
            batch_start = b * GPU_BATCH_SIZE
            batch_end = min(batch_start + GPU_BATCH_SIZE, n_parents)
            batch_n = batch_end - batch_start

            # Use permutation to select which parents go in this batch
            batch_indices = perm[batch_start:batch_end]

            # Copy batch parents to GPU
            gpu_parent_data[:, :batch_n] = cpu_data[:, batch_indices].to(device, non_blocking=True)
            gpu_parent_meta[:batch_n] = cpu_meta[batch_indices].to(device, non_blocking=True)

            # Reset counters
            global_counter.zero_()
            bucket_counters.zero_()

            # Launch 22 suffix kernels in random order
            grid = (batch_n,)
            n_stride_val = gpu_parent_data.shape[1]
            out_stride_val = gpu_out_data.shape[1]

            suffix_order = torch.randperm(N_SUFFIXES).tolist()
            for s in suffix_order:
                kw = SUFFIX_KWARGS[s]
                kernel_braid_step[grid](
                    gpu_parent_data,
                    gpu_parent_meta,
                    gpu_out_data,
                    gpu_out_meta,
                    gpu_out_parent_idx,
                    global_counter,
                    bucket_counters,
                    adj_tensor,
                    batch_n,
                    n_stride_val,
                    out_stride_val,
                    GPU_OUTPUT_CAP,
                    BUCKET_CAP,
                    num_warps=1,
                    **kw,
                )

            torch.cuda.synchronize()

            n_children = min(global_counter.item(), GPU_OUTPUT_CAP)

            if n_children == 0:
                continue

            # Extract projlen-0 from this batch immediately
            child_projlens_gpu = (gpu_out_meta[:n_children] >> 8).to(torch.int32) & 0x7F
            zero_mask_gpu = (child_projlens_gpu == 0)
            n_zeros_batch = zero_mask_gpu.sum().item()

            if n_zeros_batch > 0:
                zero_indices_gpu = torch.where(zero_mask_gpu)[0]

                zero_parent_indices_local = gpu_out_parent_idx[zero_indices_gpu].cpu()
                # Remap: batch-local idx -> perm index -> global parent index
                zero_parent_indices_global = batch_indices[zero_parent_indices_local.long()]

                zero_suffixes = (gpu_out_meta[zero_indices_gpu].cpu() & 0xFF).to(torch.uint8)
                zero_words = torch.zeros((n_zeros_batch, braid_length), dtype=torch.uint8)
                zero_words[:, :word_len] = parent_words[zero_parent_indices_global, :word_len]
                zero_words[:, word_len] = zero_suffixes

                zero_data_soa = gpu_out_data[:, zero_indices_gpu]
                zero_meta_save = gpu_out_meta[zero_indices_gpu]

                save_projlen0_braids(zero_data_soa, zero_meta_save, zero_words, braid_length)
                n_zeros_this_step += n_zeros_batch

            # Copy children to CPU accumulator (remap parent_idx)
            space_left = total_acc_cap - acc_count
            n_copy = min(n_children, space_left)
            if n_copy < n_children:
                print(f"  WARNING: accumulator overflow at batch {b}, "
                      f"dropping {n_children - n_copy} children")

            if n_copy > 0:
                acc_data[:, acc_count:acc_count + n_copy] = gpu_out_data[:, :n_copy].cpu()
                acc_meta[acc_count:acc_count + n_copy] = gpu_out_meta[:n_copy].cpu()
                # Remap parent_idx: batch-local -> global via permutation
                batch_pidx = gpu_out_parent_idx[:n_copy].cpu()
                acc_parent_idx[acc_count:acc_count + n_copy] = batch_indices[batch_pidx.long()]
                acc_count += n_copy

        total_projlen0 += n_zeros_this_step

        if acc_count == 0:
            t1 = time.time()
            print(f"Length {braid_length:3d} | {t1-t0:.2f}s | NO CHILDREN. Search exhausted.")
            break

        # --- CPU selection: sort by jittered projlen, keep top TOTAL_USE_BEST ---
        child_projlens_cpu = (acc_meta[:acc_count] >> 8).to(torch.int32) & 0x7F

        if acc_count <= TOTAL_USE_BEST:
            keep_indices = torch.arange(acc_count)
            n_keep = acc_count
        else:
            jittered = child_projlens_cpu * 2 + torch.randint(0, 2, (acc_count,))
            sorted_indices = torch.argsort(jittered)
            keep_indices = sorted_indices[:TOTAL_USE_BEST]
            n_keep = TOTAL_USE_BEST

        # Build words for kept braids
        kept_parent_indices = acc_parent_idx[keep_indices].long()
        kept_suffixes = (acc_meta[keep_indices] & 0xFF).to(torch.uint8)
        kept_words = torch.zeros((n_keep, braid_length), dtype=torch.uint8)
        kept_words[:, :word_len] = parent_words[kept_parent_indices, :word_len]
        kept_words[:, word_len] = kept_suffixes

        # Update CPU parent buffers — copy kept children into parent arrays
        # Use contiguous writes to avoid large gather temporaries
        cpu_data[:, :n_keep] = acc_data[:, keep_indices]
        cpu_meta[:n_keep] = acc_meta[keep_indices]
        parent_words = kept_words
        n_parents = n_keep
        word_len = braid_length

        t1 = time.time()
        dt = t1 - t0

        # Stats
        kept_projlens = (cpu_meta[:n_parents] >> 8).to(torch.int32) & 0x7F
        min_pl = kept_projlens.min().item()
        max_pl = kept_projlens.max().item()
        mean_pl = kept_projlens.float().mean().item()

        zeros_str = f" *** {n_zeros_this_step} PROJLEN-0 ***" if n_zeros_this_step > 0 else ""

        print(f"Length {braid_length:3d} | {dt:5.2f}s | children={acc_count:>10,} | "
              f"kept={n_parents:>10,} | batches={n_batches_this_step} | "
              f"projlen {min_pl}/{mean_pl:.1f}/{max_pl}{zeros_str}")

    # --- Summary ---
    print("=" * 100)
    print(f"Search complete. Total projlen-0 braids found: {total_projlen0}")
    if total_projlen0 > 0:
        import glob
        files = sorted(glob.glob("projlen0_results/projlen0_length*.pt"))
        for f in files:
            info = torch.load(f, weights_only=True)
            n = info["data"].shape[0]
            print(f"  {os.path.basename(f)}: {n} braids")
    print("Done.")


if __name__ == "__main__":
    run_batched_search()
