"""
MAXIMALLY OPTIMIZED braid search with:
1. Batched FFT matrix multiplication (3x fewer FFT ops)
2. Precomputed FFTs of simple Burau matrices (eliminates half of per-expansion FFTs)
3. Projlen decrease tracking - prioritizes braids whose projlen is decreasing

Expected speedup: 4-5x over original implementation
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import os
import time
import gc

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
    checkpoint_every: int = 9999
    device: str = "cuda"
    expansion_chunk_size: int = 100000
    use_best: int = 0
    matmul_chunk_size: int = 20000
    
    # Projlen decrease tracking options
    track_decrease: bool = True           # Enable tracking
    decrease_priority_boost: float = 0.5  # Priority multiplier for decreasing braids (lower = more likely kept)
    streak_priority_boost: float = 0.8    # Additional multiplier per streak level (compounds)
    max_streak_boost: int = 5             # Maximum streak levels to apply boost
    
    @property
    def degree_window(self) -> int:
        return 2 * self.degree_multiplier * self.max_length + 1
    
    @property
    def degree_offset(self) -> int:
        return self.degree_multiplier * self.max_length


# =============================================================================
# DTYPE CONFIGURATION
# =============================================================================

STORAGE_DTYPE_MATRIX = torch.int16
STORAGE_DTYPE_WORD = torch.int32
STORAGE_DTYPE_LENGTH = torch.int32
STORAGE_DTYPE_PROJLEN = torch.int16  # For previous projlen
STORAGE_DTYPE_STREAK = torch.int16   # For decrease streak counter
COMPUTE_DTYPE_INT = torch.int32


# =============================================================================
# ULTRA-OPTIMIZED POLYNOMIAL MATRIX MULTIPLICATION
# =============================================================================

class FastPolyMatmul:
    """
    Precomputes and caches FFTs of simple Burau matrices for maximum speed.
    
    Key insight: The 24 simple Burau matrices never change, so we can 
    precompute their FFTs once and reuse them for every expansion.
    
    This saves 50% of FFT operations during the expansion step!
    """
    
    def __init__(self, simple_burau: torch.Tensor, D: int, device: torch.device):
        """
        Args:
            simple_burau: (24, 3, 3, D) tensor of simple Burau matrices
            D: Degree window size
            device: Target device
        """
        self.D = D
        self.out_D = 2 * D - 1
        self.fft_size = 1 << (self.out_D).bit_length()
        self.device = device
        
        # Precompute FFTs of all 24 simple matrices
        print(f"Precomputing FFTs of simple Burau matrices...")
        print(f"  D={D}, fft_size={self.fft_size}")
        
        # Store both integer and FFT versions
        self.simple_burau = simple_burau.to(device)
        
        # FFT version: (24, 3, 3, fft_len) complex64
        self.simple_burau_fft = torch.fft.rfft(
            simple_burau.float().to(device),
            n=self.fft_size,
            dim=-1
        )
        
        fft_mem = self.simple_burau_fft.numel() * 8 / 1e6
        print(f"  FFT cache: {fft_mem:.1f} MB")
    
    def matmul_batch(self, A: torch.Tensor, suffix_indices: torch.Tensor, 
                     p: int, chunk_size: int = 20000) -> torch.Tensor:
        """
        Optimized batch matrix multiply: A @ simple_burau[suffix_indices]
        
        Uses precomputed FFTs of simple matrices, so we only need to FFT A.
        
        Args:
            A: (N, 3, 3, D) parent matrices
            suffix_indices: (N,) indices into simple_burau
            p: Prime modulus
            chunk_size: Chunk size for memory management
            
        Returns:
            (N, 3, 3, 2D-1) result matrices
        """
        N = A.shape[0]
        C = torch.zeros(N, 3, 3, self.out_D, dtype=COMPUTE_DTYPE_INT, device=self.device)
        
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            
            A_chunk = A[start:end].float()
            chunk_indices = suffix_indices[start:end]
            
            # FFT of A only (B's FFT is precomputed!)
            A_fft = torch.fft.rfft(A_chunk, n=self.fft_size, dim=-1)
            del A_chunk
            
            # Gather precomputed FFTs for the suffix matrices
            B_fft = self.simple_burau_fft[chunk_indices]  # (chunk_N, 3, 3, fft_len)
            
            # Matrix multiply in FFT space
            C_fft = torch.einsum('nikf,nkjf->nijf', A_fft, B_fft)
            del A_fft, B_fft
            
            # IFFT
            C_real = torch.fft.irfft(C_fft, n=self.fft_size, dim=-1)
            del C_fft
            
            # Truncate, round, mod p
            C_int = torch.round(C_real[..., :self.out_D]).to(COMPUTE_DTYPE_INT) % p
            del C_real
            
            C[start:end] = C_int
            del C_int
        
        return C
    
    def matmul_batch_general(self, A: torch.Tensor, B: torch.Tensor, 
                             p: int, chunk_size: int = 20000) -> torch.Tensor:
        """
        General batch matrix multiply (both A and B need FFT).
        Used when B is not from the simple Burau set.
        """
        N = A.shape[0]
        C = torch.zeros(N, 3, 3, self.out_D, dtype=COMPUTE_DTYPE_INT, device=self.device)
        
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            
            A_chunk = A[start:end].float()
            B_chunk = B[start:end].float()
            
            A_fft = torch.fft.rfft(A_chunk, n=self.fft_size, dim=-1)
            B_fft = torch.fft.rfft(B_chunk, n=self.fft_size, dim=-1)
            del A_chunk, B_chunk
            
            C_fft = torch.einsum('nikf,nkjf->nijf', A_fft, B_fft)
            del A_fft, B_fft
            
            C_real = torch.fft.irfft(C_fft, n=self.fft_size, dim=-1)
            del C_fft
            
            C_int = torch.round(C_real[..., :self.out_D]).to(COMPUTE_DTYPE_INT) % p
            del C_real
            
            C[start:end] = C_int
            del C_int
        
        return C


def compute_projlen_batch(matrices: torch.Tensor) -> torch.Tensor:
    """Compute projective length for a batch of matrices."""
    N, _, _, D = matrices.shape
    device = matrices.device
    
    projlens = torch.zeros(N, dtype=torch.int32, device=device)
    
    sub_batch_size = 50000
    for start in range(0, N, sub_batch_size):
        end = min(start + sub_batch_size, N)
        batch = matrices[start:end]
        batch_N = end - start
        
        flat = batch.reshape(batch_N, -1)
        has_nonzero = (flat != 0).any(dim=-1)
        
        by_degree = batch.reshape(batch_N, 9, D)
        degree_has_nonzero = (by_degree != 0).any(dim=1)
        
        min_degrees = degree_has_nonzero.int().argmax(dim=-1)
        max_degrees = D - 1 - degree_has_nonzero.flip(dims=[-1]).int().argmax(dim=-1)
        
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
    
    last_simples = last_simples.long()
    
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
# GPU BUCKETS WITH DECREASE TRACKING
# =============================================================================

class GPUBuckets:
    """
    Maintains reservoir-sampled buckets entirely on GPU.
    Now with projlen decrease tracking for priority boosting.
    """
    
    def __init__(self, bucket_size: int, device: torch.device, config: Config):
        self.bucket_size = bucket_size
        self.device = device
        self.config = config
        # Data: projlen -> (matrices, words, lengths, prev_projlens, streaks, priorities)
        self.data: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                    torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    
    def add_chunk(
        self,
        matrices: torch.Tensor,
        words: torch.Tensor, 
        lengths: torch.Tensor,
        projlens: torch.Tensor,
        prev_projlens: torch.Tensor,
        streaks: torch.Tensor,
        is_bootstrap: bool
    ):
        """Add a chunk of braids, with decrease-aware priority."""
        if len(matrices) == 0:
            return
        
        matrices = matrices.to(STORAGE_DTYPE_MATRIX)
        words = words.to(STORAGE_DTYPE_WORD)
        lengths = lengths.to(STORAGE_DTYPE_LENGTH)
        prev_projlens = prev_projlens.to(STORAGE_DTYPE_PROJLEN)
        streaks = streaks.to(STORAGE_DTYPE_STREAK)
        
        # Base random priorities
        priorities = torch.rand(len(matrices), device=self.device)
        
        # Apply priority boost for decreasing braids
        if self.config.track_decrease:
            # Boost for any decrease this level
            decreased = (projlens < prev_projlens) & (prev_projlens > 0)
            priorities = torch.where(
                decreased,
                priorities * self.config.decrease_priority_boost,
                priorities
            )
            
            # Additional boost based on streak (compounds)
            for s in range(1, self.config.max_streak_boost + 1):
                streak_mask = (streaks >= s)
                priorities = torch.where(
                    streak_mask,
                    priorities * self.config.streak_priority_boost,
                    priorities
                )
        
        unique_pls = torch.unique(projlens)
        
        for pl in unique_pls.tolist():
            mask = (projlens == pl)
            new_mat = matrices[mask]
            new_words = words[mask]
            new_lengths = lengths[mask]
            new_prev = prev_projlens[mask]
            new_streaks = streaks[mask]
            new_priorities = priorities[mask]
            
            if pl not in self.data:
                if is_bootstrap or len(new_mat) <= self.bucket_size:
                    self.data[pl] = (new_mat, new_words, new_lengths, 
                                     new_prev, new_streaks, new_priorities)
                else:
                    _, topk_idx = torch.topk(new_priorities, self.bucket_size, largest=False)
                    self.data[pl] = (
                        new_mat[topk_idx],
                        new_words[topk_idx],
                        new_lengths[topk_idx],
                        new_prev[topk_idx],
                        new_streaks[topk_idx],
                        new_priorities[topk_idx]
                    )
            else:
                old_mat, old_words, old_lengths, old_prev, old_streaks, old_priorities = self.data[pl]
                
                merged_mat = torch.cat([old_mat, new_mat], dim=0)
                merged_words = torch.cat([old_words, new_words], dim=0)
                merged_lengths = torch.cat([old_lengths, new_lengths], dim=0)
                merged_prev = torch.cat([old_prev, new_prev], dim=0)
                merged_streaks = torch.cat([old_streaks, new_streaks], dim=0)
                merged_priorities = torch.cat([old_priorities, new_priorities], dim=0)
                
                if is_bootstrap or len(merged_mat) <= self.bucket_size:
                    self.data[pl] = (merged_mat, merged_words, merged_lengths,
                                     merged_prev, merged_streaks, merged_priorities)
                else:
                    _, topk_idx = torch.topk(merged_priorities, self.bucket_size, largest=False)
                    self.data[pl] = (
                        merged_mat[topk_idx],
                        merged_words[topk_idx],
                        merged_lengths[topk_idx],
                        merged_prev[topk_idx],
                        merged_streaks[topk_idx],
                        merged_priorities[topk_idx]
                    )
                    del merged_mat, merged_words, merged_lengths
                    del merged_prev, merged_streaks, merged_priorities
    
    def get_buckets(self) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                              torch.Tensor, torch.Tensor]]:
        """Return buckets without priorities (for storage)."""
        return {
            pl: (mat, words, lengths, prev_proj, streaks)
            for pl, (mat, words, lengths, prev_proj, streaks, _) in self.data.items()
        }
    
    def total_count(self) -> int:
        return sum(mat.shape[0] for mat, _, _, _, _, _ in self.data.values())
    
    def clear(self):
        self.data.clear()


# =============================================================================
# MAIN ALGORITHM
# =============================================================================

class BraidSearchUltra:
    """
    ULTRA-OPTIMIZED GPU-accelerated search for braids with low projlen.
    
    Optimizations:
    1. Batched FFT matrix multiplication
    2. Precomputed FFTs of simple Burau matrices
    3. Projlen decrease tracking for smarter sampling
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
        
        self.simple_burau = simple_burau.to(STORAGE_DTYPE_MATRIX).to(self.device)
        self.valid_suffixes = valid_suffixes.to(self.device)
        self.num_valid_suffixes = num_valid_suffixes.to(self.device)
        
        self.D = simple_burau.shape[-1]
        
        # Initialize fast polynomial matmul with precomputed FFTs
        self.fast_matmul = FastPolyMatmul(simple_burau, self.D, self.device)
        
        # Buckets now store: (matrices, words, lengths, prev_projlens, streaks)
        self.buckets: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                       torch.Tensor, torch.Tensor]] = {}
        self.kernel_braids: list[torch.Tensor] = []
        self.stats = {
            "candidates_per_level": [], 
            "buckets_per_level": [],
            "time_per_level": [],
            "time_matmul": [],
            "time_sampling": [],
            "decreased_count": [],      # NEW: how many decreased this level
            "streak_distribution": [],  # NEW: streak counts per level
        }
        self.start_level = 1
    
    def initialize(self):
        """Start with the identity braid."""
        identity_matrix = torch.zeros(1, 3, 3, self.D, dtype=STORAGE_DTYPE_MATRIX, device=self.device)
        center = self.D // 2
        for i in range(3):
            identity_matrix[0, i, i, center] = 1
        
        identity_word = torch.zeros(1, self.config.max_length, dtype=STORAGE_DTYPE_WORD, device=self.device)
        identity_length = torch.zeros(1, dtype=STORAGE_DTYPE_LENGTH, device=self.device)
        
        # Initial prev_projlen = 1 (identity has projlen 1), streak = 0
        identity_prev_projlen = torch.ones(1, dtype=STORAGE_DTYPE_PROJLEN, device=self.device)
        identity_streak = torch.zeros(1, dtype=STORAGE_DTYPE_STREAK, device=self.device)
        
        self.buckets[1] = (identity_matrix, identity_word, identity_length,
                          identity_prev_projlen, identity_streak)
        
        print(f"Initialized with identity braid")
        print(f"Degree window: [-{self.D//2}, {self.D//2}] ({self.D} coefficients)")
        print(f"FFT size: {self.fast_matmul.fft_size}")
        print(f"Storage types: matrix={STORAGE_DTYPE_MATRIX}, word={STORAGE_DTYPE_WORD}")
        print(f"Config: bucket_size={self.config.bucket_size}, "
              f"bootstrap={self.config.bootstrap_length}, "
              f"max_length={self.config.max_length}, "
              f"chunk_size={self.config.expansion_chunk_size}, "
              f"matmul_chunk={self.config.matmul_chunk_size}, "
              f"use_best={self.config.use_best if self.config.use_best > 0 else 'all'}")
        print(f"⚡ Ultra-optimized: precomputed Burau FFTs enabled")
        if self.config.track_decrease:
            print(f"📉 Decrease tracking: enabled (boost={self.config.decrease_priority_boost}, "
                  f"streak_boost={self.config.streak_priority_boost}, max_streak={self.config.max_streak_boost})")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load state from checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        saved_level = checkpoint['level']
        self.start_level = saved_level + 1
        
        saved_config = checkpoint.get('config', {})
        saved_max_length = saved_config.get('max_length', self.config.max_length)
        saved_degree_mult = saved_config.get('degree_multiplier', self.config.degree_multiplier)
        saved_D = 2 * saved_degree_mult * saved_max_length + 1
        new_D = self.D
        new_max_length = self.config.max_length
        
        needs_matrix_resize = (saved_D != new_D)
        needs_word_resize = (saved_max_length != new_max_length)
        
        self.stats = checkpoint.get('stats', self.stats)
        # Ensure new stat fields exist
        if 'decreased_count' not in self.stats:
            self.stats['decreased_count'] = []
        if 'streak_distribution' not in self.stats:
            self.stats['streak_distribution'] = []
        
        if 'kernel_braids' in checkpoint:
            self.kernel_braids = []
            for w in checkpoint['kernel_braids']:
                if isinstance(w, list):
                    self.kernel_braids.append(torch.tensor(w))
                else:
                    self.kernel_braids.append(w.clone())
        
        self.buckets = {}
        for pl, bucket_data in checkpoint['buckets'].items():
            pl = int(pl)
            
            if isinstance(bucket_data, dict):
                mat = bucket_data['matrices']
                words = bucket_data['words']
                lengths = bucket_data['lengths']
                prev_projlens = bucket_data.get('prev_projlens', None)
                streaks = bucket_data.get('streaks', None)
            elif len(bucket_data) == 5:
                mat, words, lengths, prev_projlens, streaks = bucket_data
            else:
                # Old format without tracking
                mat, words, lengths = bucket_data[:3]
                prev_projlens = None
                streaks = None
            
            if not isinstance(mat, torch.Tensor):
                mat = torch.tensor(mat)
            else:
                mat = mat.clone()
            if not isinstance(words, torch.Tensor):
                words = torch.tensor(words)
            else:
                words = words.clone()
            if not isinstance(lengths, torch.Tensor):
                lengths = torch.tensor(lengths)
            else:
                lengths = lengths.clone()
            
            # Handle prev_projlens and streaks
            n_braids = mat.shape[0]
            if prev_projlens is None:
                prev_projlens = torch.full((n_braids,), pl, dtype=STORAGE_DTYPE_PROJLEN)
            elif not isinstance(prev_projlens, torch.Tensor):
                prev_projlens = torch.tensor(prev_projlens)
            else:
                prev_projlens = prev_projlens.clone()
            
            if streaks is None:
                streaks = torch.zeros(n_braids, dtype=STORAGE_DTYPE_STREAK)
            elif not isinstance(streaks, torch.Tensor):
                streaks = torch.tensor(streaks)
            else:
                streaks = streaks.clone()
            
            if needs_matrix_resize:
                old_D = mat.shape[-1]
                old_center = old_D // 2
                new_center = new_D // 2
                
                if new_D > old_D:
                    new_mat = torch.zeros(mat.shape[0], 3, 3, new_D, dtype=mat.dtype)
                    offset = new_center - old_center
                    new_mat[:, :, :, offset:offset + old_D] = mat
                    mat = new_mat
                else:
                    offset = old_center - new_center
                    mat = mat[:, :, :, offset:offset + new_D].clone()
            
            if needs_word_resize:
                old_len = words.shape[-1]
                if new_max_length > old_len:
                    padding = torch.zeros(words.shape[0], new_max_length - old_len, dtype=words.dtype)
                    words = torch.cat([words, padding], dim=-1)
                else:
                    words = words[:, :new_max_length].clone()
            
            self.buckets[pl] = (
                mat.to(STORAGE_DTYPE_MATRIX).to(self.device),
                words.to(STORAGE_DTYPE_WORD).to(self.device),
                lengths.to(STORAGE_DTYPE_LENGTH).to(self.device),
                prev_projlens.to(STORAGE_DTYPE_PROJLEN).to(self.device),
                streaks.to(STORAGE_DTYPE_STREAK).to(self.device)
            )
        
        total_braids = sum(m.shape[0] for m, _, _, _, _ in self.buckets.values())
        print(f"  Loaded level {saved_level}, {total_braids} braids")
        print(f"  Resuming from level {self.start_level}")
        
        return self.start_level
    
    def save_checkpoint(self, level: int, path: str):
        """Save current state to disk."""
        print(f"  Saving checkpoint to {path}...")
        
        checkpoint = {
            "level": level,
            "config": {k: v for k, v in self.config.__dict__.items()},
            "stats": self.stats,
            "kernel_braids": [w.cpu() for w in self.kernel_braids],
            "buckets": {
                pl: (mat.cpu(), words.cpu(), lengths.cpu(), 
                     prev_proj.cpu(), streaks.cpu())
                for pl, (mat, words, lengths, prev_proj, streaks) in self.buckets.items()
            }
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
    
    def gather_level_braids(self, use_best: int = 0):
        """Gather braids from current buckets, prioritizing low projlen."""
        if not self.buckets:
            raise RuntimeError("No braids to process!")
        
        sorted_projlens = sorted(self.buckets.keys())
        
        all_matrices = []
        all_words = []
        all_lengths = []
        all_prev_projlens = []
        all_streaks = []
        all_current_projlens = []  # Track current projlen for each braid
        total_selected = 0
        
        for projlen in sorted_projlens:
            matrices, words, lengths, prev_projlens, streaks = self.buckets[projlen]
            bucket_count = len(matrices)
            
            if use_best > 0:
                remaining = use_best - total_selected
                if remaining <= 0:
                    break
                
                if bucket_count <= remaining:
                    all_matrices.append(matrices)
                    all_words.append(words)
                    all_lengths.append(lengths)
                    all_prev_projlens.append(prev_projlens)
                    all_streaks.append(streaks)
                    all_current_projlens.append(torch.full((bucket_count,), projlen, 
                                                           dtype=STORAGE_DTYPE_PROJLEN, device=self.device))
                    total_selected += bucket_count
                else:
                    idx = torch.randperm(bucket_count, device=self.device)[:remaining]
                    all_matrices.append(matrices[idx])
                    all_words.append(words[idx])
                    all_lengths.append(lengths[idx])
                    all_prev_projlens.append(prev_projlens[idx])
                    all_streaks.append(streaks[idx])
                    all_current_projlens.append(torch.full((remaining,), projlen,
                                                           dtype=STORAGE_DTYPE_PROJLEN, device=self.device))
                    total_selected += remaining
                    break
            else:
                all_matrices.append(matrices)
                all_words.append(words)
                all_lengths.append(lengths)
                all_prev_projlens.append(prev_projlens)
                all_streaks.append(streaks)
                all_current_projlens.append(torch.full((bucket_count,), projlen,
                                                       dtype=STORAGE_DTYPE_PROJLEN, device=self.device))
        
        matrices = torch.cat(all_matrices, dim=0).to(COMPUTE_DTYPE_INT)
        words = torch.cat(all_words, dim=0)
        lengths = torch.cat(all_lengths, dim=0)
        prev_projlens = torch.cat(all_prev_projlens, dim=0)
        streaks = torch.cat(all_streaks, dim=0)
        current_projlens = torch.cat(all_current_projlens, dim=0)
        
        batch_idx = torch.arange(len(lengths), device=self.device)
        last_pos = torch.clamp(lengths - 1, min=0).long()
        last_simples = words[batch_idx, last_pos].long()
        last_simples = torch.where(lengths > 0, last_simples, torch.zeros_like(last_simples))
        
        return matrices, words, lengths, last_simples, prev_projlens, streaks, current_projlens
    
    def expand_and_multiply_chunk(
        self,
        matrices: torch.Tensor,
        words: torch.Tensor,
        lengths: torch.Tensor,
        prev_projlens: torch.Tensor,
        streaks: torch.Tensor,
        current_projlens: torch.Tensor,
        braid_indices: torch.Tensor,
        suffix_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand a chunk using PRECOMPUTED FFTs of simple Burau matrices."""
        num_candidates = len(braid_indices)
        
        parent_matrices = matrices[braid_indices]
        parent_words = words[braid_indices]
        parent_lengths = lengths[braid_indices]
        parent_prev_projlens = prev_projlens[braid_indices]
        parent_streaks = streaks[braid_indices]
        parent_current_projlens = current_projlens[braid_indices]
        
        # USE ULTRA-FAST MATMUL WITH PRECOMPUTED FFTS
        new_matrices = self.fast_matmul.matmul_batch(
            parent_matrices, 
            suffix_indices,
            self.config.prime,
            chunk_size=self.config.matmul_chunk_size
        )
        
        new_words = parent_words.clone()
        batch_idx = torch.arange(num_candidates, device=self.device)
        new_words[batch_idx, parent_lengths.long()] = suffix_indices.to(STORAGE_DTYPE_WORD)
        new_lengths = parent_lengths + 1
        
        # The parent's current projlen becomes the child's prev_projlen
        new_prev_projlens = parent_current_projlens.clone()
        
        # Streaks will be updated after we compute the new projlens
        new_streaks = parent_streaks.clone()
        
        return new_matrices, new_words, new_lengths, new_prev_projlens, new_streaks
    
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
        
        return matrices[..., trim_left:trim_right]
    
    def process_level(self, level: int):
        """Process one level with ultra-fast matrix multiplication."""
        level_start = time.time()
        
        is_bootstrap = (level <= self.config.bootstrap_length)
        mode = "BOOTSTRAP" if is_bootstrap else "SAMPLING"
        
        print(f"\n{'='*60}")
        print(f"Level {level} - {mode}")
        print(f"{'='*60}")
        
        use_best_limit = 0 if is_bootstrap else self.config.use_best
        matrices, words, lengths, last_simples, prev_projlens, streaks, current_projlens = \
            self.gather_level_braids(use_best=use_best_limit)
        num_starting = len(matrices)
        print(f"  Starting braids: {num_starting}")
        
        # Print streak distribution for starting braids
        if self.config.track_decrease and level > self.config.bootstrap_length:
            streak_counts = {}
            unique_streaks, counts = torch.unique(streaks, return_counts=True)
            for s, c in zip(unique_streaks.tolist(), counts.tolist()):
                streak_counts[s] = c
            if streak_counts:
                streak_str = ", ".join([f"streak={s}: {c}" for s, c in sorted(streak_counts.items())])
                print(f"  Starting streak distribution: {streak_str}")
        
        t0 = time.time()
        braid_indices, suffix_indices = build_expansion_indices_vectorized(
            last_simples, self.num_valid_suffixes, self.valid_suffixes
        )
        num_candidates = len(braid_indices)
        print(f"  Candidates to generate: {num_candidates}")
        
        if num_candidates == 0:
            print("  No candidates! Algorithm terminates.")
            return False
        
        chunk_size = self.config.expansion_chunk_size
        num_chunks = (num_candidates + chunk_size - 1) // chunk_size
        
        if num_chunks > 1:
            print(f"  Processing in {num_chunks} chunks...")
        
        gpu_buckets = GPUBuckets(self.config.bucket_size, self.device, self.config)
        
        t_matmul_total = 0.0
        t_sample_total = 0.0
        projlen_counts: dict[int, int] = {}
        
        # Track decrease statistics
        total_decreased = 0
        total_increased = 0
        total_same = 0
        streak_after_counts: dict[int, int] = {}
        
        for chunk_idx in range(num_chunks):
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, num_candidates)
            
            chunk_braid_idx = braid_indices[start:end]
            chunk_suffix_idx = suffix_indices[start:end]
            
            t0 = time.time()
            chunk_matrices, chunk_words, chunk_lengths, chunk_prev_projlens, chunk_streaks = \
                self.expand_and_multiply_chunk(
                    matrices, words, lengths, prev_projlens, streaks, current_projlens,
                    chunk_braid_idx, chunk_suffix_idx
                )
            chunk_matrices = self.recenter_matrices(chunk_matrices)
            t_matmul_total += time.time() - t0
            
            chunk_projlens = compute_projlen_batch(chunk_matrices)
            
            # Update streaks based on projlen change
            if self.config.track_decrease:
                # Compare new projlen to previous (parent's) projlen
                decreased = (chunk_projlens < chunk_prev_projlens) & (chunk_prev_projlens > 0)
                increased_or_same = ~decreased | (chunk_prev_projlens == 0)
                
                # Increment streak where decreased, reset where not
                chunk_streaks = torch.where(decreased, chunk_streaks + 1, torch.zeros_like(chunk_streaks))
                
                # Count statistics
                chunk_decreased = decreased.sum().item()
                chunk_increased = ((chunk_projlens > chunk_prev_projlens) & (chunk_prev_projlens > 0)).sum().item()
                chunk_same = ((chunk_projlens == chunk_prev_projlens) & (chunk_prev_projlens > 0)).sum().item()
                total_decreased += chunk_decreased
                total_increased += chunk_increased
                total_same += chunk_same
                
                # Track streak distribution after update
                unique_streaks, counts = torch.unique(chunk_streaks, return_counts=True)
                for s, c in zip(unique_streaks.tolist(), counts.tolist()):
                    streak_after_counts[s] = streak_after_counts.get(s, 0) + c
            
            one_mask = (chunk_projlens == 1)
            num_ones = one_mask.sum().item()
            if num_ones > 0:
                print(f"\n  🎉 FOUND {num_ones} KERNEL ELEMENTS (projlen=1)! 🎉")
                self.kernel_braids.append(chunk_words[one_mask].cpu())
            
            unique_pls, counts = torch.unique(chunk_projlens, return_counts=True)
            for pl, c in zip(unique_pls.tolist(), counts.tolist()):
                projlen_counts[pl] = projlen_counts.get(pl, 0) + c
            
            t0 = time.time()
            gpu_buckets.add_chunk(
                chunk_matrices, chunk_words, chunk_lengths, chunk_projlens,
                chunk_prev_projlens, chunk_streaks,
                is_bootstrap
            )
            t_sample_total += time.time() - t0
            
            del chunk_matrices, chunk_words, chunk_lengths, chunk_projlens
            del chunk_prev_projlens, chunk_streaks
        
        del matrices, words, lengths, last_simples
        del prev_projlens, streaks, current_projlens
        del braid_indices, suffix_indices
        
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Print decrease statistics
        if self.config.track_decrease:
            total_valid = total_decreased + total_increased + total_same
            if total_valid > 0:
                print(f"\n  📉 Projlen change stats:")
                print(f"     Decreased: {total_decreased} ({100*total_decreased/total_valid:.1f}%)")
                print(f"     Increased: {total_increased} ({100*total_increased/total_valid:.1f}%)")
                print(f"     Same:      {total_same} ({100*total_same/total_valid:.1f}%)")
                
                if streak_after_counts:
                    # Show distribution of streaks (consecutive decreases)
                    streak_str = ", ".join([f"{s}:{c}" for s, c in sorted(streak_after_counts.items()) if s <= 10])
                    max_streak = max(streak_after_counts.keys())
                    if max_streak > 0:
                        print(f"     Streak distribution: {streak_str}" + 
                              (f" (max={max_streak})" if max_streak > 10 else ""))
        
        print(f"\n  Projlen distribution:")
        for pl in sorted(projlen_counts.keys())[:10]:
            print(f"    projlen={pl}: {projlen_counts[pl]} braids")
        
        self.buckets = gpu_buckets.get_buckets()
        
        total_kept = sum(m.shape[0] for m, _, _, _, _ in self.buckets.values())
        
        total_bytes = 0
        for mat, wrds, lens, prev_p, strks in self.buckets.values():
            total_bytes += mat.numel() * mat.element_size()
            total_bytes += wrds.numel() * wrds.element_size()
            total_bytes += lens.numel() * lens.element_size()
            total_bytes += prev_p.numel() * prev_p.element_size()
            total_bytes += strks.numel() * strks.element_size()
        
        print(f"  Braids kept: {total_kept} (in {len(self.buckets)} buckets, {total_bytes/1e9:.2f} GB)")
        
        level_time = time.time() - level_start
        print(f"  ⚡ Timing: matmul={t_matmul_total:.2f}s, sampling={t_sample_total:.2f}s, total={level_time:.2f}s")
        
        self.stats["candidates_per_level"].append(num_candidates)
        self.stats["buckets_per_level"].append(len(self.buckets))
        self.stats["time_per_level"].append(level_time)
        self.stats["time_matmul"].append(t_matmul_total)
        self.stats["time_sampling"].append(t_sample_total)
        self.stats["decreased_count"].append(total_decreased)
        self.stats["streak_distribution"].append(dict(streak_after_counts))
        
        return True
    
    def run(self, checkpoint_dir: Optional[str] = None, resume_from: Optional[str] = None):
        """Run the full search algorithm."""
        if resume_from:
            self.load_checkpoint(resume_from)
        else:
            self.initialize()
        
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        total_start = time.time()
        final_level = self.start_level - 1
        
        try:
            for level in range(self.start_level, self.config.max_length + 1):
                success = self.process_level(level)
                final_level = level
                
                if not success:
                    break
                
                if checkpoint_dir and (level % self.config.checkpoint_every == 0):
                    self.save_checkpoint(level, f"{checkpoint_dir}/checkpoint_level_{level}.pt")
        
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted at level {final_level}!")
        
        finally:
            total_time = time.time() - total_start
            
            print(f"\n{'='*60}")
            print("SEARCH COMPLETE" if final_level == self.config.max_length else "SEARCH STOPPED")
            print(f"{'='*60}")
            print(f"Final level: {final_level}")
            print(f"Total time: {total_time:.2f}s")
            levels_completed = final_level - self.start_level + 1
            print(f"Avg time per level: {total_time / max(1, levels_completed):.2f}s")
            if self.stats["time_matmul"]:
                print(f"Total matmul time: {sum(self.stats['time_matmul']):.2f}s")
            print(f"Total kernel elements: {sum(len(w) for w in self.kernel_braids)}")
            
            # Summary of decrease tracking
            if self.config.track_decrease and self.stats["decreased_count"]:
                total_decreased = sum(self.stats["decreased_count"])
                print(f"Total braids with projlen decrease: {total_decreased}")
            
            save_dir = checkpoint_dir if checkpoint_dir else "."
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            final_path = f"{save_dir}/final_state_level_{final_level}.pt"
            self.save_checkpoint(final_level, final_path)
            print(f"\nFinal state saved to {final_path}")
        
        return self.kernel_braids


# =============================================================================
# TABLE LOADING
# =============================================================================

def load_tables_from_file(config: Config, table_path: str):
    """Load precomputed tables from .pt file."""
    tables = torch.load(table_path, weights_only=True)
    
    assert tables['n'] == 4
    assert tables['p'] == config.prime
    
    loaded_burau = tables['simple_burau']
    loaded_center = tables['center']
    
    D = config.degree_window
    new_center = D // 2
    
    simple_burau = torch.zeros(24, 3, 3, D, dtype=STORAGE_DTYPE_MATRIX)
    
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
        
        simple_burau[s, :, :, dst_start:dst_end] = mat[:, :, src_start:src_end].to(STORAGE_DTYPE_MATRIX)
    
    loaded_valid_suffixes = tables['valid_suffixes']
    loaded_num_valid = tables['num_valid_suffixes']
    
    delta_idx = tables['delta_index']
    id_idx = tables['id_index']
    
    valid_suffixes = loaded_valid_suffixes.clone()
    num_valid_suffixes = loaded_num_valid.clone()
    
    valid_suffixes[id_idx] = valid_suffixes[delta_idx]
    num_valid_suffixes[id_idx] = num_valid_suffixes[delta_idx]
    
    print(f"Loaded tables from {table_path}")
    print(f"  Degree window: {D}")
    
    return simple_burau, valid_suffixes, num_valid_suffixes


# Alias for drop-in replacement
BraidSearch = BraidSearchUltra
