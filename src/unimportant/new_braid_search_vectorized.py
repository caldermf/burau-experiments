"""
GPU-accelerated reservoir sampling for braids with low projlen.

VECTORIZED VERSION - No Python loops in hot paths.

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
    # FFT size should be >= 2D-1, use next power of 2 for efficiency
    fft_size = 1 << (2 * D - 1).bit_length()
    
    # FFT-based convolution
    a_fft = torch.fft.rfft(a.float(), n=fft_size, dim=-1)
    b_fft = torch.fft.rfft(b.float(), n=fft_size, dim=-1)
    c_fft = a_fft * b_fft
    c = torch.fft.irfft(c_fft, n=fft_size, dim=-1)
    
    # Round to integers and take mod p
    c = torch.round(c).long() % p
    
    # Trim to actual convolution length
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
    
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # Loop over matrix indices (only 27 iterations, negligible overhead)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Extract the (N,D) batches of polynomials
                a_ik = A[:, i, k, :]  # (N, D)
                b_kj = B[:, k, j, :]  # (N, D)
                
                # Convolve and accumulate
                conv = poly_multiply_batch(a_ik, b_kj, p)  # (N, 2D-1)
                C[:, i, j, :] = (C[:, i, j, :] + conv) % p
    
    return C


def compute_projlen_batch(matrices: torch.Tensor) -> torch.Tensor:
    """
    Compute projective length for a batch of matrices.
    
    projlen = (max degree with nonzero coeff) - (min degree with nonzero coeff) + 1
    
    Args:
        matrices: (N, 3, 3, D) - polynomial matrices
    
    Returns:
        (N,) - projlen for each matrix
    """
    N, _, _, D = matrices.shape
    
    # Flatten to (N, 9*D) to find nonzero positions
    flat = matrices.reshape(N, -1)  # (N, 9*D)
    
    # Create degree indices
    degree_indices = torch.arange(9 * D, device=matrices.device) % D  # (9*D,)
    
    # Mask for nonzero entries
    nonzero_mask = (flat != 0)  # (N, 9*D)
    
    # Check for all-zero matrices
    has_nonzero = nonzero_mask.any(dim=-1)  # (N,)
    
    # For min: set zeros to large value, then take min
    degrees_for_min = torch.where(nonzero_mask, degree_indices.unsqueeze(0), D + 1)
    min_degrees = degrees_for_min.min(dim=-1).values  # (N,)
    
    # For max: set zeros to small value, then take max  
    degrees_for_max = torch.where(nonzero_mask, degree_indices.unsqueeze(0), -1)
    max_degrees = degrees_for_max.max(dim=-1).values  # (N,)
    
    # projlen = max - min + 1 (matching peyl's definition)
    projlen = max_degrees - min_degrees + 1
    
    # Zero matrix has projlen 0
    projlen = torch.where(has_nonzero, projlen, torch.zeros_like(projlen))
    
    return projlen


# =============================================================================
# VECTORIZED SUFFIX EXPANSION (THE KEY OPTIMIZATION)
# =============================================================================

def build_expansion_indices_vectorized(
    last_simples: torch.Tensor,
    num_valid_suffixes: torch.Tensor,
    valid_suffixes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build (braid_index, suffix_index) pairs for all valid expansions.
    
    FULLY VECTORIZED - no Python loops!
    
    Args:
        last_simples: (N,) - last simple index for each braid
        num_valid_suffixes: (24,) - count of valid suffixes per simple
        valid_suffixes: (24, max_suffixes) - valid suffix indices, padded with -1
    
    Returns:
        braid_indices: (total_expansions,) - which braid each expansion comes from
        suffix_indices: (total_expansions,) - which suffix to append
    
    Example:
        If last_simples = [5, 3] and num_valid_suffixes[5] = 11, num_valid_suffixes[3] = 3,
        then we produce 14 total expansions:
        - braid_indices = [0,0,0,0,0,0,0,0,0,0,0, 1,1,1]
        - suffix_indices = [valid_suffixes[5,0], ..., valid_suffixes[5,10], 
                           valid_suffixes[3,0], valid_suffixes[3,1], valid_suffixes[3,2]]
    """
    device = last_simples.device
    N = len(last_simples)
    
    # Get suffix counts for each braid's last simple
    suffix_counts = num_valid_suffixes[last_simples]  # (N,)
    
    # Total number of expansions
    total_expansions = suffix_counts.sum().item()
    
    if total_expansions == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device)
        )
    
    # Build braid_indices: repeat each braid index by its suffix count
    # e.g., if suffix_counts = [11, 3], we get [0,0,0,0,0,0,0,0,0,0,0, 1,1,1]
    braid_indices = torch.repeat_interleave(
        torch.arange(N, device=device),
        suffix_counts
    )
    
    # Build suffix_indices: for each braid, we need indices 0, 1, ..., count-1
    # into that braid's suffix list
    #
    # Strategy: 
    # 1. Create cumsum of suffix_counts to get starting positions
    # 2. Use arange and subtract to get local indices within each group
    # 3. Look up actual suffix values
    
    # Cumulative sum gives us the start position of each braid's suffixes
    cumsum = torch.cumsum(suffix_counts, dim=0)  # (N,)
    starts = cumsum - suffix_counts  # (N,) - start index for each braid
    
    # For each expansion, compute which local suffix index (0, 1, 2, ...) it is
    # global_position - start_of_this_braid = local_suffix_index
    global_positions = torch.arange(total_expansions, device=device)
    local_suffix_indices = global_positions - starts[braid_indices]
    
    # Now look up actual suffix values from the table
    # valid_suffixes[last_simples[braid], local_suffix_index]
    last_simples_expanded = last_simples[braid_indices]  # (total_expansions,)
    suffix_indices = valid_suffixes[last_simples_expanded, local_suffix_indices].long()
    
    return braid_indices, suffix_indices


# =============================================================================
# RESERVOIR SAMPLING ON GPU  
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
    Perform reservoir sampling grouped by projlen, entirely on GPU.
    """
    device = matrices.device
    unique_projlens = torch.unique(projlens).tolist()
    
    buckets = {}
    
    if is_bootstrap:
        # Keep everything, just group by projlen
        for m in unique_projlens:
            mask = (projlens == m)
            buckets[m] = (
                matrices[mask],
                words[mask],
                word_lengths[mask]
            )
    else:
        # Priority-based reservoir sampling
        priorities = torch.rand(len(matrices), device=device)
        
        for m in unique_projlens:
            mask = (projlens == m)
            indices = torch.where(mask)[0]
            
            if len(indices) <= bucket_size:
                selected = indices
            else:
                group_priorities = priorities[indices]
                _, topk_local = torch.topk(group_priorities, bucket_size, largest=False)
                selected = indices[topk_local]
            
            buckets[m] = (
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
    VECTORIZED VERSION.
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
        
        # Store precomputed tables on GPU
        self.simple_burau = simple_burau.to(self.device)
        self.valid_suffixes = valid_suffixes.to(self.device)
        self.num_valid_suffixes = num_valid_suffixes.to(self.device)
        
        # Degree window info
        self.D = simple_burau.shape[-1]
        
        # Current level's buckets: projlen -> (matrices, words, word_lengths)
        self.buckets: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        
        # Track kernel elements
        self.kernel_braids: list[torch.Tensor] = []
        
        # Statistics
        self.stats = {
            "candidates_per_level": [], 
            "buckets_per_level": [],
            "time_per_level": []
        }
    
    def initialize(self):
        """Start with the identity braid."""
        identity_matrix = torch.zeros(1, 3, 3, self.D, dtype=torch.long, device=self.device)
        center = self.D // 2
        for i in range(3):
            identity_matrix[0, i, i, center] = 1
        
        identity_word = torch.zeros(1, self.config.max_length, dtype=torch.long, device=self.device)
        identity_length = torch.zeros(1, dtype=torch.long, device=self.device)
        
        # projlen of identity is 1 (single nonzero degree)
        self.buckets[1] = (identity_matrix, identity_word, identity_length)
        
        print(f"Initialized with identity braid")
        print(f"Degree window: [-{self.D//2}, {self.D//2}] ({self.D} coefficients)")
        print(f"Config: bucket_size={self.config.bucket_size}, "
              f"bootstrap_length={self.config.bootstrap_length}, "
              f"max_length={self.config.max_length}, "
              f"chunk_size={self.config.expansion_chunk_size}, "
              f"use_best={self.config.use_best if self.config.use_best > 0 else 'unlimited'}")
    
    def gather_level_braids(self, use_best: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gather braids from current buckets, prioritizing low projlen if use_best > 0.
        """
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
        
        # Extract last simple from each word (vectorized)
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
        """
        Expand a chunk: gather parent matrices, multiply by suffix matrices.
        """
        num_candidates = len(braid_indices)
        
        # Gather parent data
        parent_matrices = matrices[braid_indices]
        parent_words = words[braid_indices]
        parent_lengths = lengths[braid_indices]
        
        # Get suffix matrices
        suffix_matrices = self.simple_burau[suffix_indices]
        
        # Batch matrix multiplication
        new_matrices = poly_matmul_batch(parent_matrices, suffix_matrices, self.config.prime)
        
        # Build new words
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
        
        # Warn if trimming nonzero values
        left_loss = matrices[..., :trim_left].abs().sum()
        right_loss = matrices[..., trim_right:].abs().sum()
        if left_loss > 0 or right_loss > 0:
            print(f"  WARNING: Trimming nonzero coefficients! left={left_loss}, right={right_loss}")
        
        return matrices[..., trim_left:trim_right]
    
    def process_level(self, level: int):
        """
        Process one level: expand all braids and reservoir sample.
        VECTORIZED candidate generation.
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
        
        # === VECTORIZED EXPANSION INDEX GENERATION ===
        t0 = time.time()
        braid_indices, suffix_indices = build_expansion_indices_vectorized(
            last_simples, self.num_valid_suffixes, self.valid_suffixes
        )
        num_candidates = len(braid_indices)
        t1 = time.time()
        print(f"  Candidates to generate: {num_candidates} (index build: {t1-t0:.3f}s)")
        
        if num_candidates == 0:
            print("  No candidates! Algorithm terminates.")
            return False
        
        # Process in chunks
        chunk_size = self.config.expansion_chunk_size
        num_chunks = (num_candidates + chunk_size - 1) // chunk_size
        
        if num_chunks > 1:
            print(f"  Processing in {num_chunks} chunks...")
        
        # Track projlen distribution
        projlen_counts = {}
        
        # Incremental reservoir sampling
        accumulated_buckets: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        
        t_matmul = 0.0
        t_sample = 0.0
        
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, num_candidates)
            
            chunk_braid_idx = braid_indices[start:end]
            chunk_suffix_idx = suffix_indices[start:end]
            
            # Expand and multiply
            t0 = time.time()
            chunk_matrices, chunk_words, chunk_lengths = self.expand_and_multiply_chunk(
                matrices, words, lengths,
                chunk_braid_idx, chunk_suffix_idx
            )
            
            # Recenter
            chunk_matrices = self.recenter_matrices(chunk_matrices)
            t1 = time.time()
            t_matmul += t1 - t0
            
            # Compute projlen
            chunk_projlens = compute_projlen_batch(chunk_matrices)
            
            # Check for kernel elements
            one_mask = (chunk_projlens == 1)
            num_ones = one_mask.sum().item()
            if num_ones > 0:
                print(f"\n  🎉 FOUND {num_ones} KERNEL ELEMENTS (projlen=1)! 🎉")
                self.kernel_braids.append(chunk_words[one_mask].cpu())
            
            # Update projlen distribution
            unique_pls, counts = torch.unique(chunk_projlens, return_counts=True)
            for pl, count in zip(unique_pls.tolist(), counts.tolist()):
                projlen_counts[pl] = projlen_counts.get(pl, 0) + count
            
            # Generate priorities
            chunk_priorities = torch.rand(len(chunk_matrices), device=self.device)
            
            # Move to CPU for reservoir sampling
            t0 = time.time()
            chunk_matrices_cpu = chunk_matrices.cpu()
            chunk_words_cpu = chunk_words.cpu()
            chunk_lengths_cpu = chunk_lengths.cpu()
            chunk_priorities_cpu = chunk_priorities.cpu()
            chunk_projlens_cpu = chunk_projlens.cpu()
            
            # Incremental reservoir sampling
            for pl in unique_pls.tolist():
                mask = (chunk_projlens_cpu == pl)
                new_mat = chunk_matrices_cpu[mask]
                new_words = chunk_words_cpu[mask]
                new_lengths = chunk_lengths_cpu[mask]
                new_priorities = chunk_priorities_cpu[mask]
                
                if pl not in accumulated_buckets:
                    accumulated_buckets[pl] = (new_mat, new_words, new_lengths, new_priorities)
                else:
                    old_mat, old_words, old_lengths, old_priorities = accumulated_buckets[pl]
                    
                    merged_mat = torch.cat([old_mat, new_mat], dim=0)
                    merged_words = torch.cat([old_words, new_words], dim=0)
                    merged_lengths = torch.cat([old_lengths, new_lengths], dim=0)
                    merged_priorities = torch.cat([old_priorities, new_priorities], dim=0)
                    
                    if not is_bootstrap and len(merged_mat) > self.config.bucket_size:
                        _, topk_indices = torch.topk(merged_priorities, self.config.bucket_size, largest=False)
                        merged_mat = merged_mat[topk_indices]
                        merged_words = merged_words[topk_indices]
                        merged_lengths = merged_lengths[topk_indices]
                        merged_priorities = merged_priorities[topk_indices]
                    
                    accumulated_buckets[pl] = (merged_mat, merged_words, merged_lengths, merged_priorities)
            
            t1 = time.time()
            t_sample += t1 - t0
            
            # Free GPU memory
            del chunk_matrices, chunk_words, chunk_lengths, chunk_projlens, chunk_priorities
            del chunk_matrices_cpu, chunk_words_cpu, chunk_lengths_cpu, chunk_priorities_cpu, chunk_projlens_cpu
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        print(f"  Timing: matmul={t_matmul:.2f}s, sampling={t_sample:.2f}s")
        
        # Report projlen distribution
        print(f"  Projlen distribution:")
        for pl in sorted(projlen_counts.keys())[:10]:  # Show first 10
            print(f"    projlen={pl}: {projlen_counts[pl]} braids")
        if len(projlen_counts) > 10:
            print(f"    ... and {len(projlen_counts) - 10} more projlen values")
        
        # Move final buckets to GPU
        self.buckets = {}
        for pl, (mat, wrds, lens, _) in accumulated_buckets.items():
            self.buckets[pl] = (
                mat.to(self.device),
                wrds.to(self.device),
                lens.to(self.device)
            )
        
        del accumulated_buckets
        
        total_kept = sum(m.shape[0] for m, _, _ in self.buckets.values())
        print(f"  Braids kept: {total_kept} (in {len(self.buckets)} buckets)")
        
        level_time = time.time() - level_start
        print(f"  Level time: {level_time:.2f}s")
        
        self.stats["candidates_per_level"].append(num_candidates)
        self.stats["buckets_per_level"].append(len(self.buckets))
        self.stats["time_per_level"].append(level_time)
        
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
        print(f"Total kernel elements (projlen=1) found: {sum(len(w) for w in self.kernel_braids)}")
        
        return self.kernel_braids


# =============================================================================
# TABLE LOADING
# =============================================================================

def load_tables_from_file(config: Config, table_path: str):
    """Load precomputed tables from .pt file."""
    import torch
    
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
