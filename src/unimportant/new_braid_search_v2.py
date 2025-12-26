"""
GPU-accelerated reservoir sampling for braids with low projlen.

VECTORIZED VERSION v2 - Reservoir sampling actually on GPU now!

This implements the algorithm for finding 4-strand braids whose Burau 
representation (mod p) has low projective length.
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
    bucket_size: int = 50000       # max braids per bucket (k in the paper)
    max_length: int = 50           # maximum Garside length to explore
    bootstrap_length: int = 5      # exhaustive enumeration until this length
    prime: int = 5                 # modular arithmetic mod p
    degree_multiplier: int = 4     # degree window = [-mult*max_len, mult*max_len]
    checkpoint_every: int = 5      # save checkpoints every N levels
    device: str = "cuda"           # "cuda" or "cpu"
    expansion_chunk_size: int = 100000  # max candidates to process at once
    use_best: int = 0              # max braids to expand per level (0 = no limit)
    
    @property
    def degree_window(self) -> int:
        """Total size of degree coefficient array."""
        return 2 * self.degree_multiplier * self.max_length + 1
    
    @property
    def degree_offset(self) -> int:
        """Index offset: degree d is stored at index d + degree_offset."""
        return self.degree_multiplier * self.max_length


# =============================================================================
# CORE POLYNOMIAL OPERATIONS
# =============================================================================

def poly_multiply_batch(a: torch.Tensor, b: torch.Tensor, p: int) -> torch.Tensor:
    """
    Batch polynomial multiplication using FFT convolution.
    
    Args:
        a: (N, D) - N polynomials, each with D coefficients
        b: (N, D) - N polynomials to multiply with
        p: prime modulus
    
    Returns:
        (N, 2D-1) - convolution results, mod p
    """
    N, D = a.shape
    
    # Pad to avoid circular convolution artifacts
    fft_size = 1 << (2 * D - 1).bit_length()
    
    # FFT-based convolution
    a_fft = torch.fft.rfft(a.float(), n=fft_size, dim=-1)
    b_fft = torch.fft.rfft(b.float(), n=fft_size, dim=-1)
    c_fft = a_fft * b_fft
    c = torch.fft.irfft(c_fft, n=fft_size, dim=-1)
    
    # Round to integers and take mod p
    c = torch.round(c).long() % p
    
    return c[:, :2*D-1]


def poly_matmul_batch(A: torch.Tensor, B: torch.Tensor, p: int) -> torch.Tensor:
    """
    Batch 3x3 matrix multiplication over polynomial ring (Z/pZ)[v, v^-1].
    
    Args:
        A: (N, 3, 3, D) - N matrices with polynomial entries
        B: (N, 3, 3, D) - N matrices to multiply with
        p: prime modulus
    
    Returns:
        (N, 3, 3, 2D-1) - product matrices, mod p
    """
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
    """
    Compute projective length for a batch of matrices.
    
    projlen = (max degree with nonzero coeff) - (min degree with nonzero coeff) + 1
    """
    N, _, _, D = matrices.shape
    
    flat = matrices.reshape(N, -1)
    degree_indices = torch.arange(9 * D, device=matrices.device) % D
    nonzero_mask = (flat != 0)
    has_nonzero = nonzero_mask.any(dim=-1)
    
    degrees_for_min = torch.where(nonzero_mask, degree_indices.unsqueeze(0), D + 1)
    min_degrees = degrees_for_min.min(dim=-1).values
    
    degrees_for_max = torch.where(nonzero_mask, degree_indices.unsqueeze(0), -1)
    max_degrees = degrees_for_max.max(dim=-1).values
    
    projlen = max_degrees - min_degrees + 1
    projlen = torch.where(has_nonzero, projlen, torch.zeros_like(projlen))
    
    return projlen


# =============================================================================
# VECTORIZED SUFFIX EXPANSION
# =============================================================================

def build_expansion_indices_vectorized(
    last_simples: torch.Tensor,
    num_valid_suffixes: torch.Tensor,
    valid_suffixes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build (braid_index, suffix_index) pairs for all valid expansions.
    FULLY VECTORIZED - no Python loops!
    """
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
# GPU RESERVOIR SAMPLING
# =============================================================================

def reservoir_sample_gpu(
    matrices: torch.Tensor,
    words: torch.Tensor,
    word_lengths: torch.Tensor,
    projlens: torch.Tensor,
    bucket_size: int,
    is_bootstrap: bool
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Perform reservoir sampling grouped by projlen, ENTIRELY ON GPU.
    
    Uses random priorities + topk for O(N log bucket_size) sampling.
    """
    device = matrices.device
    N = len(matrices)
    
    if N == 0:
        return {}
    
    unique_projlens = torch.unique(projlens)
    buckets = {}
    
    if is_bootstrap:
        # Keep everything, just group by projlen
        for pl in unique_projlens.tolist():
            mask = (projlens == pl)
            buckets[pl] = (
                matrices[mask],
                words[mask],
                word_lengths[mask]
            )
    else:
        # GPU reservoir sampling with random priorities
        priorities = torch.rand(N, device=device)
        
        for pl in unique_projlens.tolist():
            mask = (projlens == pl)
            indices = torch.where(mask)[0]
            count = len(indices)
            
            if count <= bucket_size:
                selected = indices
            else:
                # Select bucket_size items with lowest random priority
                group_priorities = priorities[indices]
                _, topk_local = torch.topk(group_priorities, bucket_size, largest=False)
                selected = indices[topk_local]
            
            buckets[pl] = (
                matrices[selected],
                words[selected],
                word_lengths[selected]
            )
    
    return buckets


# =============================================================================
# MAIN ALGORITHM
# =============================================================================

class BraidSearch:
    """
    GPU-accelerated search for braids with low projlen.
    VECTORIZED VERSION v2 - with proper GPU reservoir sampling.
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
              f"bootstrap_length={self.config.bootstrap_length}, "
              f"max_length={self.config.max_length}, "
              f"chunk_size={self.config.expansion_chunk_size}, "
              f"use_best={self.config.use_best if self.config.use_best > 0 else 'unlimited'}")
    
    def gather_level_braids(self, use_best: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather braids from current buckets, prioritizing low projlen if use_best > 0."""
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
                remaining_budget = use_best - total_selected
                if remaining_budget <= 0:
                    break
                
                if bucket_count <= remaining_budget:
                    all_matrices.append(matrices)
                    all_words.append(words)
                    all_lengths.append(lengths)
                    total_selected += bucket_count
                else:
                    indices = torch.randperm(bucket_count, device=self.device)[:remaining_budget]
                    all_matrices.append(matrices[indices])
                    all_words.append(words[indices])
                    all_lengths.append(lengths[indices])
                    total_selected += remaining_budget
                    break
            else:
                all_matrices.append(matrices)
                all_words.append(words)
                all_lengths.append(lengths)
                total_selected += bucket_count
        
        matrices = torch.cat(all_matrices, dim=0)
        words = torch.cat(all_words, dim=0)
        lengths = torch.cat(all_lengths, dim=0)
        
        batch_indices = torch.arange(len(lengths), device=self.device)
        last_positions = torch.clamp(lengths - 1, min=0)
        last_simples = words[batch_indices, last_positions]
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
        
        left_loss = matrices[..., :trim_left].abs().sum()
        right_loss = matrices[..., trim_right:].abs().sum()
        if left_loss > 0 or right_loss > 0:
            print(f"  WARNING: Trimming nonzero coefficients! left={left_loss}, right={right_loss}")
        
        return matrices[..., trim_left:trim_right]
    
    def process_level(self, level: int):
        """
        Process one level: expand all braids and reservoir sample.
        ALL COMPUTATION STAYS ON GPU until final bucket storage.
        """
        level_start = time.time()
        
        is_bootstrap = (level <= self.config.bootstrap_length)
        mode = "BOOTSTRAP" if is_bootstrap else "SAMPLING"
        
        print(f"\n{'='*60}")
        print(f"Level {level} - {mode}")
        print(f"{'='*60}")
        
        # Gather braids from previous level
        use_best_limit = 0 if is_bootstrap else self.config.use_best
        matrices, words, lengths, last_simples = self.gather_level_braids(use_best=use_best_limit)
        num_starting = len(matrices)
        print(f"  Starting braids: {num_starting}")
        
        # === VECTORIZED EXPANSION INDEX GENERATION ===
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
        
        # Process in chunks, keeping everything on GPU
        chunk_size = self.config.expansion_chunk_size
        num_chunks = (num_candidates + chunk_size - 1) // chunk_size
        
        if num_chunks > 1:
            print(f"  Processing in {num_chunks} chunks...")
        
        # Accumulate ALL results on GPU
        all_new_matrices = []
        all_new_words = []
        all_new_lengths = []
        
        t_matmul_total = 0.0
        
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, num_candidates)
            
            chunk_braid_idx = braid_indices[start:end]
            chunk_suffix_idx = suffix_indices[start:end]
            
            # Expand and multiply (on GPU)
            t0 = time.time()
            chunk_matrices, chunk_words, chunk_lengths = self.expand_and_multiply_chunk(
                matrices, words, lengths,
                chunk_braid_idx, chunk_suffix_idx
            )
            
            # Recenter (on GPU)
            chunk_matrices = self.recenter_matrices(chunk_matrices)
            t_matmul_total += time.time() - t0
            
            # Keep on GPU!
            all_new_matrices.append(chunk_matrices)
            all_new_words.append(chunk_words)
            all_new_lengths.append(chunk_lengths)
        
        # Concatenate all chunks on GPU
        t0 = time.time()
        new_matrices = torch.cat(all_new_matrices, dim=0)
        new_words = torch.cat(all_new_words, dim=0)
        new_lengths = torch.cat(all_new_lengths, dim=0)
        
        # Free chunk lists
        del all_new_matrices, all_new_words, all_new_lengths
        
        # Compute projlen for ALL candidates at once (on GPU)
        projlens = compute_projlen_batch(new_matrices)
        
        # Check for kernel elements (on GPU)
        one_mask = (projlens == 1)
        num_ones = one_mask.sum().item()
        if num_ones > 0:
            print(f"\n  🎉 FOUND {num_ones} KERNEL ELEMENTS (projlen=1)! 🎉")
            self.kernel_braids.append(new_words[one_mask].cpu())
        
        # Report projlen distribution
        unique_pls, counts = torch.unique(projlens, return_counts=True)
        projlen_counts = {pl.item(): c.item() for pl, c in zip(unique_pls, counts)}
        
        print(f"  Projlen distribution:")
        for pl in sorted(projlen_counts.keys())[:10]:
            print(f"    projlen={pl}: {projlen_counts[pl]} braids")
        if len(projlen_counts) > 10:
            print(f"    ... and {len(projlen_counts) - 10} more projlen values")
        
        # === GPU RESERVOIR SAMPLING ===
        t_sample_start = time.time()
        self.buckets = reservoir_sample_gpu(
            new_matrices, new_words, new_lengths, projlens,
            self.config.bucket_size, is_bootstrap
        )
        t_sample = time.time() - t_sample_start
        
        # Free the big tensors
        del new_matrices, new_words, new_lengths, projlens
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        total_kept = sum(m.shape[0] for m, _, _ in self.buckets.values())
        print(f"  Braids kept: {total_kept} (in {len(self.buckets)} buckets)")
        
        level_time = time.time() - level_start
        print(f"  Timing: matmul={t_matmul_total:.2f}s, sampling={t_sample:.2f}s, total={level_time:.2f}s")
        
        self.stats["candidates_per_level"].append(num_candidates)
        self.stats["buckets_per_level"].append(len(self.buckets))
        self.stats["time_per_level"].append(level_time)
        self.stats["time_matmul"].append(t_matmul_total)
        self.stats["time_sampling"].append(t_sample)
        
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
        print(f"Avg time per level: {total_time / self.config.max_length:.2f}s")
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
