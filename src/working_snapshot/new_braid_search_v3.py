"""
GPU-accelerated reservoir sampling for braids with low projlen.

VECTORIZED VERSION v3 - Memory-efficient incremental GPU reservoir sampling.

Key improvement: Instead of concatenating all chunks then sampling,
we do incremental reservoir sampling on GPU after each chunk.
This bounds memory usage while keeping everything on GPU.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import json
from pathlib import Path
import os
import time

script_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(script_dir)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Algorithm parameters."""
    bucket_size: int = 50000
    max_length: int = 50
    bootstrap_length: int = 5
    prime: int = 5
    degree_multiplier: int = 4
    checkpoint_every: int = 5
    device: str = "cuda"
    expansion_chunk_size: int = 100000
    use_best: int = 0
    
    @property
    def degree_window(self) -> int:
        return 2 * self.degree_multiplier * self.max_length + 1
    
    @property
    def degree_offset(self) -> int:
        return self.degree_multiplier * self.max_length


# =============================================================================
# CORE POLYNOMIAL OPERATIONS
# =============================================================================

def poly_multiply_batch(a: torch.Tensor, b: torch.Tensor, p: int) -> torch.Tensor:
    """Batch polynomial multiplication using FFT convolution."""
    N, D = a.shape
    fft_size = 1 << (2 * D - 1).bit_length()
    
    a_fft = torch.fft.rfft(a.float(), n=fft_size, dim=-1)
    b_fft = torch.fft.rfft(b.float(), n=fft_size, dim=-1)
    c_fft = a_fft * b_fft
    c = torch.fft.irfft(c_fft, n=fft_size, dim=-1)
    
    c = torch.round(c).long() % p
    return c[:, :2*D-1]


def poly_matmul_batch(A: torch.Tensor, B: torch.Tensor, p: int) -> torch.Tensor:
    """Batch 3x3 matrix multiplication over polynomial ring."""
    N, _, _, D = A.shape
    out_D = 2 * D - 1
    C = torch.zeros(N, 3, 3, out_D, dtype=torch.long, device=A.device)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a_ik = A[:, i, k, :]
                b_kj = B[:, k, j, :]
                conv = poly_multiply_batch(a_ik, b_kj, p)
                C[:, i, j, :] = (C[:, i, j, :] + conv) % p
    
    return C


def compute_projlen_batch(matrices: torch.Tensor) -> torch.Tensor:
    """Compute projective length for a batch of matrices."""
    N, _, _, D = matrices.shape
    
    # More memory-efficient: don't create huge intermediate tensors
    # Process in smaller sub-batches if needed
    device = matrices.device
    
    projlens = torch.zeros(N, dtype=torch.long, device=device)
    
    # Process in sub-batches to avoid OOM on the where() operations
    sub_batch_size = 50000
    for start in range(0, N, sub_batch_size):
        end = min(start + sub_batch_size, N)
        batch = matrices[start:end]
        batch_N = end - start
        
        flat = batch.reshape(batch_N, -1)  # (batch_N, 9*D)
        
        # Find min and max degree for each matrix
        # Use a loop over degrees to avoid creating huge tensors
        has_nonzero = (flat != 0).any(dim=-1)
        
        # For matrices with nonzero entries, find degree bounds
        # Reshape to (batch_N, 9, D) to work with degrees
        by_degree = batch.reshape(batch_N, 9, D)
        
        # Check which degrees have any nonzero entry
        degree_has_nonzero = (by_degree != 0).any(dim=1)  # (batch_N, D)
        
        # Find min degree (first True in each row)
        # Use argmax on the boolean tensor
        min_degrees = degree_has_nonzero.long().argmax(dim=-1)
        
        # Find max degree (last True in each row)  
        # Flip and argmax, then convert back
        max_degrees = D - 1 - degree_has_nonzero.flip(dims=[-1]).long().argmax(dim=-1)
        
        # projlen = max - min + 1
        batch_projlen = max_degrees - min_degrees + 1
        batch_projlen = torch.where(has_nonzero, batch_projlen, torch.zeros_like(batch_projlen))
        
        projlens[start:end] = batch_projlen
    
    return projlens


# =============================================================================
# VECTORIZED SUFFIX EXPANSION
# =============================================================================

def build_expansion_indices_vectorized(
    last_simples: torch.Tensor,
    num_valid_suffixes: torch.Tensor,
    valid_suffixes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (braid_index, suffix_index) pairs. FULLY VECTORIZED."""
    device = last_simples.device
    N = len(last_simples)
    
    suffix_counts = num_valid_suffixes[last_simples]
    total_expansions = suffix_counts.sum().item()
    
    if total_expansions == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device)
        )
    
    braid_indices = torch.repeat_interleave(
        torch.arange(N, device=device),
        suffix_counts
    )
    
    cumsum = torch.cumsum(suffix_counts, dim=0)
    starts = cumsum - suffix_counts
    
    global_positions = torch.arange(total_expansions, device=device)
    local_suffix_indices = global_positions - starts[braid_indices]
    
    last_simples_expanded = last_simples[braid_indices]
    suffix_indices = valid_suffixes[last_simples_expanded, local_suffix_indices].long()
    
    return braid_indices, suffix_indices


# =============================================================================
# INCREMENTAL GPU RESERVOIR SAMPLING
# =============================================================================

class GPUBuckets:
    """
    Maintains reservoir-sampled buckets entirely on GPU.
    
    Each bucket stores (matrices, words, lengths, priorities) and maintains
    at most bucket_size items using priority-based sampling.
    """
    
    def __init__(self, bucket_size: int, device: torch.device):
        self.bucket_size = bucket_size
        self.device = device
        # projlen -> (matrices, words, lengths, priorities)
        self.data: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    
    def add_chunk(
        self,
        matrices: torch.Tensor,
        words: torch.Tensor, 
        lengths: torch.Tensor,
        projlens: torch.Tensor,
        is_bootstrap: bool
    ):
        """
        Add a chunk of candidates to the buckets with reservoir sampling.
        All operations stay on GPU.
        """
        if len(matrices) == 0:
            return
        
        # Generate random priorities for new items
        priorities = torch.rand(len(matrices), device=self.device)
        
        # Group by projlen
        unique_pls = torch.unique(projlens)
        
        for pl in unique_pls.tolist():
            mask = (projlens == pl)
            new_mat = matrices[mask]
            new_words = words[mask]
            new_lengths = lengths[mask]
            new_priorities = priorities[mask]
            
            if pl not in self.data:
                # First time seeing this projlen
                if is_bootstrap or len(new_mat) <= self.bucket_size:
                    self.data[pl] = (new_mat, new_words, new_lengths, new_priorities)
                else:
                    # Sample down immediately
                    _, topk_idx = torch.topk(new_priorities, self.bucket_size, largest=False)
                    self.data[pl] = (
                        new_mat[topk_idx],
                        new_words[topk_idx],
                        new_lengths[topk_idx],
                        new_priorities[topk_idx]
                    )
            else:
                # Merge with existing bucket
                old_mat, old_words, old_lengths, old_priorities = self.data[pl]
                
                merged_mat = torch.cat([old_mat, new_mat], dim=0)
                merged_words = torch.cat([old_words, new_words], dim=0)
                merged_lengths = torch.cat([old_lengths, new_lengths], dim=0)
                merged_priorities = torch.cat([old_priorities, new_priorities], dim=0)
                
                if is_bootstrap or len(merged_mat) <= self.bucket_size:
                    self.data[pl] = (merged_mat, merged_words, merged_lengths, merged_priorities)
                else:
                    # Sample down to bucket_size
                    _, topk_idx = torch.topk(merged_priorities, self.bucket_size, largest=False)
                    self.data[pl] = (
                        merged_mat[topk_idx],
                        merged_words[topk_idx],
                        merged_lengths[topk_idx],
                        merged_priorities[topk_idx]
                    )
                    
                    # Free merged tensors
                    del merged_mat, merged_words, merged_lengths, merged_priorities
    
    def get_buckets(self) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return final buckets without priorities."""
        return {
            pl: (mat, words, lengths)
            for pl, (mat, words, lengths, _) in self.data.items()
        }
    
    def total_count(self) -> int:
        return sum(mat.shape[0] for mat, _, _, _ in self.data.values())
    
    def clear(self):
        self.data.clear()


# =============================================================================
# MAIN ALGORITHM
# =============================================================================

class BraidSearch:
    """
    GPU-accelerated search for braids with low projlen.
    v3: Memory-efficient incremental GPU reservoir sampling.
    """
    
    def __init__(
        self,
        simple_burau: torch.Tensor,
        valid_suffixes: torch.Tensor,
        num_valid_suffixes: torch.Tensor,
        config: Config
    ):
        self.config = config
        self.device = torch.device(config.device)
        
        self.simple_burau = simple_burau.to(self.device)
        self.valid_suffixes = valid_suffixes.to(self.device)
        self.num_valid_suffixes = num_valid_suffixes.to(self.device)
        
        self.D = simple_burau.shape[-1]
        self.buckets: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.kernel_braids: list[torch.Tensor] = []
        self.stats = {
            "candidates_per_level": [], 
            "buckets_per_level": [],
            "time_per_level": [],
            "time_matmul": [],
            "time_sampling": [],
        }
    
    def initialize(self):
        """Start with the identity braid."""
        identity_matrix = torch.zeros(1, 3, 3, self.D, dtype=torch.long, device=self.device)
        center = self.D // 2
        for i in range(3):
            identity_matrix[0, i, i, center] = 1
        
        identity_word = torch.zeros(1, self.config.max_length, dtype=torch.long, device=self.device)
        identity_length = torch.zeros(1, dtype=torch.long, device=self.device)
        
        self.buckets[1] = (identity_matrix, identity_word, identity_length)
        
        print(f"Initialized with identity braid")
        print(f"Degree window: [-{self.D//2}, {self.D//2}] ({self.D} coefficients)")
        print(f"Config: bucket_size={self.config.bucket_size}, "
              f"bootstrap={self.config.bootstrap_length}, "
              f"max_length={self.config.max_length}, "
              f"chunk_size={self.config.expansion_chunk_size}, "
              f"use_best={self.config.use_best if self.config.use_best > 0 else 'all'}")
    
    def gather_level_braids(self, use_best: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather braids from current buckets, prioritizing low projlen."""
        if not self.buckets:
            raise RuntimeError("No braids to process!")
        
        sorted_projlens = sorted(self.buckets.keys())
        
        all_matrices = []
        all_words = []
        all_lengths = []
        total_selected = 0
        
        for projlen in sorted_projlens:
            matrices, words, lengths = self.buckets[projlen]
            bucket_count = len(matrices)
            
            if use_best > 0:
                remaining = use_best - total_selected
                if remaining <= 0:
                    break
                
                if bucket_count <= remaining:
                    all_matrices.append(matrices)
                    all_words.append(words)
                    all_lengths.append(lengths)
                    total_selected += bucket_count
                else:
                    idx = torch.randperm(bucket_count, device=self.device)[:remaining]
                    all_matrices.append(matrices[idx])
                    all_words.append(words[idx])
                    all_lengths.append(lengths[idx])
                    total_selected += remaining
                    break
            else:
                all_matrices.append(matrices)
                all_words.append(words)
                all_lengths.append(lengths)
        
        matrices = torch.cat(all_matrices, dim=0)
        words = torch.cat(all_words, dim=0)
        lengths = torch.cat(all_lengths, dim=0)
        
        batch_idx = torch.arange(len(lengths), device=self.device)
        last_pos = torch.clamp(lengths - 1, min=0)
        last_simples = words[batch_idx, last_pos]
        last_simples = torch.where(lengths > 0, last_simples, torch.zeros_like(last_simples))
        
        return matrices, words, lengths, last_simples
    
    def expand_and_multiply_chunk(
        self,
        matrices: torch.Tensor,
        words: torch.Tensor,
        lengths: torch.Tensor,
        braid_indices: torch.Tensor,
        suffix_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand a chunk: gather parent matrices, multiply by suffix matrices."""
        num_candidates = len(braid_indices)
        
        parent_matrices = matrices[braid_indices]
        parent_words = words[braid_indices]
        parent_lengths = lengths[braid_indices]
        
        suffix_matrices = self.simple_burau[suffix_indices]
        
        new_matrices = poly_matmul_batch(parent_matrices, suffix_matrices, self.config.prime)
        
        new_words = parent_words.clone()
        batch_idx = torch.arange(num_candidates, device=self.device)
        new_words[batch_idx, parent_lengths] = suffix_indices
        new_lengths = parent_lengths + 1
        
        return new_matrices, new_words, new_lengths
    
    def recenter_matrices(self, matrices: torch.Tensor) -> torch.Tensor:
        """Trim matrices back to target degree window."""
        current_D = matrices.shape[-1]
        target_D = self.D
        
        if current_D <= target_D:
            pad_total = target_D - current_D
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return F.pad(matrices, (pad_left, pad_right), value=0)
        
        trim_total = current_D - target_D
        trim_left = trim_total // 2
        trim_right = current_D - trim_left
        
        # Only warn once per level if trimming nonzero
        left_loss = matrices[..., :trim_left].abs().sum().item()
        right_loss = matrices[..., trim_right:].abs().sum().item()
        if left_loss > 0 or right_loss > 0:
            print(f"  WARNING: Trimming nonzero coefficients! left={left_loss}, right={right_loss}")
        
        return matrices[..., trim_left:trim_right]
    
    def process_level(self, level: int):
        """
        Process one level with incremental GPU reservoir sampling.
        Memory-efficient: processes chunks and samples incrementally.
        """
        level_start = time.time()
        
        is_bootstrap = (level <= self.config.bootstrap_length)
        mode = "BOOTSTRAP" if is_bootstrap else "SAMPLING"
        
        print(f"\n{'='*60}")
        print(f"Level {level} - {mode}")
        print(f"{'='*60}")
        
        # Gather braids
        use_best_limit = 0 if is_bootstrap else self.config.use_best
        matrices, words, lengths, last_simples = self.gather_level_braids(use_best=use_best_limit)
        num_starting = len(matrices)
        print(f"  Starting braids: {num_starting}")
        
        # Build expansion indices (vectorized)
        t0 = time.time()
        braid_indices, suffix_indices = build_expansion_indices_vectorized(
            last_simples, self.num_valid_suffixes, self.valid_suffixes
        )
        num_candidates = len(braid_indices)
        t_index = time.time() - t0
        print(f"  Candidates to generate: {num_candidates} (index build: {t_index:.3f}s)")
        
        if num_candidates == 0:
            print("  No candidates! Algorithm terminates.")
            return False
        
        # Process in chunks with incremental GPU reservoir sampling
        chunk_size = self.config.expansion_chunk_size
        num_chunks = (num_candidates + chunk_size - 1) // chunk_size
        
        if num_chunks > 1:
            print(f"  Processing in {num_chunks} chunks...")
        
        # Create GPU bucket manager
        gpu_buckets = GPUBuckets(self.config.bucket_size, self.device)
        
        t_matmul_total = 0.0
        t_sample_total = 0.0
        projlen_counts: dict[int, int] = {}
        
        for chunk_idx in range(num_chunks):
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, num_candidates)
            
            chunk_braid_idx = braid_indices[start:end]
            chunk_suffix_idx = suffix_indices[start:end]
            
            # === MATMUL (on GPU) ===
            t0 = time.time()
            chunk_matrices, chunk_words, chunk_lengths = self.expand_and_multiply_chunk(
                matrices, words, lengths,
                chunk_braid_idx, chunk_suffix_idx
            )
            chunk_matrices = self.recenter_matrices(chunk_matrices)
            t_matmul_total += time.time() - t0
            
            # === PROJLEN (on GPU) ===
            chunk_projlens = compute_projlen_batch(chunk_matrices)
            
            # Check for kernel elements
            one_mask = (chunk_projlens == 1)
            num_ones = one_mask.sum().item()
            if num_ones > 0:
                print(f"\n  🎉 FOUND {num_ones} KERNEL ELEMENTS (projlen=1)! 🎉")
                self.kernel_braids.append(chunk_words[one_mask].cpu())
            
            # Track projlen distribution
            unique_pls, counts = torch.unique(chunk_projlens, return_counts=True)
            for pl, c in zip(unique_pls.tolist(), counts.tolist()):
                projlen_counts[pl] = projlen_counts.get(pl, 0) + c
            
            # === INCREMENTAL RESERVOIR SAMPLING (on GPU) ===
            t0 = time.time()
            gpu_buckets.add_chunk(
                chunk_matrices, chunk_words, chunk_lengths, chunk_projlens,
                is_bootstrap
            )
            t_sample_total += time.time() - t0
            
            # Free chunk tensors
            del chunk_matrices, chunk_words, chunk_lengths, chunk_projlens
            
        # Free parent tensors and expansion indices
        del matrices, words, lengths, last_simples
        del braid_indices, suffix_indices
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Report projlen distribution
        print(f"  Projlen distribution:")
        for pl in sorted(projlen_counts.keys())[:10]:
            print(f"    projlen={pl}: {projlen_counts[pl]} braids")
        if len(projlen_counts) > 10:
            print(f"    ... and {len(projlen_counts) - 10} more projlen values")
        
        # Get final buckets (still on GPU)
        self.buckets = gpu_buckets.get_buckets()
        
        total_kept = sum(m.shape[0] for m, _, _ in self.buckets.values())
        print(f"  Braids kept: {total_kept} (in {len(self.buckets)} buckets)")
        
        level_time = time.time() - level_start
        print(f"  Timing: matmul={t_matmul_total:.2f}s, sampling={t_sample_total:.2f}s, total={level_time:.2f}s")
        
        self.stats["candidates_per_level"].append(num_candidates)
        self.stats["buckets_per_level"].append(len(self.buckets))
        self.stats["time_per_level"].append(level_time)
        self.stats["time_matmul"].append(t_matmul_total)
        self.stats["time_sampling"].append(t_sample_total)
        
        return True
    
    def save_checkpoint(self, level: int, path: str):
        """Save current state to disk."""
        checkpoint = {
            "level": level,
            "config": {k: v for k, v in self.config.__dict__.items()},
            "stats": self.stats,
            "buckets": {},
            "kernel_braids": [w.tolist() for w in self.kernel_braids]
        }
        
        for projlen, (matrices, words, lengths) in self.buckets.items():
            checkpoint["buckets"][projlen] = {
                "matrices": matrices.cpu().tolist(),
                "words": words.cpu().tolist(),
                "lengths": lengths.cpu().tolist()
            }
        
        with open(path, 'w') as f:
            json.dump(checkpoint, f)
        print(f"  Checkpoint saved to {path}")
    
    def run(self, checkpoint_dir: Optional[str] = None):
        """Run the full search algorithm."""
        self.initialize()
        
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(exist_ok=True)
        
        total_start = time.time()
        
        for level in range(1, self.config.max_length + 1):
            success = self.process_level(level)
            
            if not success:
                break
            
            if checkpoint_dir and (level % self.config.checkpoint_every == 0):
                self.save_checkpoint(level, f"{checkpoint_dir}/checkpoint_level_{level}.json")
        
        total_time = time.time() - total_start
        
        print(f"\n{'='*60}")
        print("SEARCH COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg time per level: {total_time / max(1, len(self.stats['time_per_level'])):.2f}s")
        if self.stats["time_matmul"]:
            print(f"Total matmul time: {sum(self.stats['time_matmul']):.2f}s")
            print(f"Total sampling time: {sum(self.stats['time_sampling']):.2f}s")
        print(f"Total kernel elements (projlen=1) found: {sum(len(w) for w in self.kernel_braids)}")
        
        return self.kernel_braids


# =============================================================================
# TABLE LOADING
# =============================================================================

def load_tables_from_file(config: Config, table_path: str):
    """Load precomputed tables from .pt file."""
    tables = torch.load(table_path, weights_only=True)
    
    assert tables['n'] == 4, f"Expected n=4, got {tables['n']}"
    assert tables['p'] == config.prime, f"Table prime {tables['p']} != config prime {config.prime}"
    
    loaded_burau = tables['simple_burau']
    loaded_center = tables['center']
    
    D = config.degree_window
    new_center = D // 2
    
    simple_burau = torch.zeros(24, 3, 3, D, dtype=torch.long)
    
    for s in range(24):
        mat = loaded_burau[s]
        nonzero_mask = mat.abs().sum(dim=(0, 1)) > 0
        if not nonzero_mask.any():
            continue
            
        nonzero_indices = torch.where(nonzero_mask)[0]
        src_start = nonzero_indices[0].item()
        src_end = nonzero_indices[-1].item() + 1
        
        min_degree = src_start - loaded_center
        max_degree = src_end - 1 - loaded_center
        
        dst_start = new_center + min_degree
        dst_end = new_center + max_degree + 1
        
        if dst_start < 0 or dst_end > D:
            raise ValueError(
                f"Simple {s} degrees [{min_degree}, {max_degree}] don't fit in window {D}"
            )
        
        simple_burau[s, :, :, dst_start:dst_end] = mat[:, :, src_start:src_end]
    
    loaded_valid_suffixes = tables['valid_suffixes']
    loaded_num_valid = tables['num_valid_suffixes']
    
    delta_idx = tables['delta_index']
    id_idx = tables['id_index']
    
    valid_suffixes = loaded_valid_suffixes.clone()
    num_valid_suffixes = loaded_num_valid.clone()
    
    valid_suffixes[id_idx] = valid_suffixes[delta_idx]
    num_valid_suffixes[id_idx] = num_valid_suffixes[delta_idx]
    
    print(f"Loaded tables from {table_path}")
    print(f"  Re-centered: degree_window={D}")
    print(f"  Identity suffixes fixed: {num_valid_suffixes[id_idx]} valid")
    
    return simple_burau, valid_suffixes, num_valid_suffixes


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = Config(
        bucket_size=50000,
        max_length=50,
        bootstrap_length=5,
        prime=5,
        degree_multiplier=4,
        checkpoint_every=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        expansion_chunk_size=100000
    )
    
    print(f"Using device: {config.device}")
    print(f"Degree window: {config.degree_window}")
    
    table_path = os.path.join(project_root, "precomputed_tables", f"tables_B4_r1_p{config.prime}.pt")
    simple_burau, valid_suffixes, num_valid_suffixes = load_tables_from_file(config, table_path)
    
    center = config.degree_window // 2
    assert simple_burau[0, 0, 0, center] == 1, "Identity check failed"
    print("✓ Identity matrix verified")
    
    search = BraidSearch(simple_burau, valid_suffixes, num_valid_suffixes, config)
    kernel_braids = search.run(checkpoint_dir="checkpoints")
    
    for i, words in enumerate(kernel_braids):
        print(f"\nBatch {i}: {len(words)} kernel elements")
        for word in words[:5]:
            print(f"  {word.tolist()}")


if __name__ == "__main__":
    main()
