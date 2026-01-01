"""
ADAPTIVE FFT braid search - uses smaller FFTs at early levels for massive speedup.

Key insight: At level L, the max degree span is roughly 2*L (each simple adds ~2 to degree).
Using the full degree window (based on max_length=150) at level 10 is wasteful.

This version:
1. Uses level-appropriate FFT sizes (huge speedup at early levels)
2. Expands degree window only when needed
3. Uses optimized memory access patterns
4. Precomputes suffix FFTs at each size tier
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
    max_length: int = 150
    bootstrap_length: int = 5
    prime: int = 7
    degree_multiplier: int = 4
    checkpoint_every: int = 9999
    device: str = "cuda"
    expansion_chunk_size: int = 50000
    use_best: int = 0
    matmul_chunk_size: int = 10000  # Smaller chunks often faster due to cache
    
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
COMPUTE_DTYPE_INT = torch.int32


# =============================================================================
# ADAPTIVE FFT TIERS
# =============================================================================

def get_fft_tier(level: int, degree_multiplier: int = 4) -> tuple[int, int, int]:
    """
    Determine the appropriate D and FFT size for a given level.
    
    Returns: (D, fft_size, tier_max_level)
    
    The idea: at level L, max degree span is ~2*L. We add margin and round up.
    
    IMPORTANT: fft_size must be >= 2*D - 1 for correct convolution!
    """
    # Define tiers: (max_level_for_tier, D, fft_size)
    # fft_size must be >= 2*D - 1, so for D=65, need fft_size >= 129 -> 256
    tiers = [
        (15, 33, 128),       # Levels 1-15: D=33, out_D=65, fft=128 ✓
        (30, 65, 256),       # Levels 16-30: D=65, out_D=129, fft=256 ✓
        (50, 129, 512),      # Levels 31-50: D=129, out_D=257, fft=512 ✓
        (80, 257, 1024),     # Levels 51-80: D=257, out_D=513, fft=1024 ✓
        (120, 513, 2048),    # Levels 81-120: D=513, out_D=1025, fft=2048 ✓
        (200, 1025, 4096),   # Levels 121-200: D=1025, out_D=2049, fft=4096 ✓
        (400, 2049, 8192),   # Levels 201-400: D=2049, out_D=4097, fft=8192 ✓
    ]
    
    for max_level, D, fft_size in tiers:
        if level <= max_level:
            return D, fft_size, max_level
    
    # Beyond all tiers
    return 2049, 8192, 9999


class AdaptiveFFTManager:
    """
    Manages FFT computations with level-adaptive sizing.
    
    Key optimizations:
    1. Uses smallest sufficient FFT size for each level
    2. Precomputes suffix FFTs for each tier
    3. Handles degree window transitions smoothly
    """
    
    def __init__(self, simple_burau: torch.Tensor, max_D: int, device: torch.device, prime: int):
        """
        Args:
            simple_burau: (24, 3, 3, max_D) tensor of simple Burau matrices
            max_D: Maximum degree window (for final storage)
            device: Target device
            prime: Prime modulus
        """
        self.max_D = max_D
        self.device = device
        self.prime = prime
        self.center = max_D // 2
        
        # Store original simple Burau matrices
        self.simple_burau_full = simple_burau.to(device)
        
        # Cache for precomputed FFTs at different tiers
        # tier_fft_size -> (simple_burau_fft, D_for_tier)
        self.tier_cache: dict[int, tuple[torch.Tensor, int]] = {}
        
        # Current tier info
        self.current_fft_size = 0
        self.current_D = 0
        
        print(f"AdaptiveFFTManager initialized")
        print(f"  Max D: {max_D}")
        print(f"  Device: {device}")
    
    def ensure_tier(self, level: int) -> tuple[int, int, int]:
        """
        Ensure we have precomputed FFTs for the appropriate tier.
        Returns (D, fft_size, tier_max_level) for this tier.
        """
        D, fft_size, tier_max = get_fft_tier(level)
        
        if fft_size in self.tier_cache:
            self.current_D, self.current_fft_size = D, fft_size
            return D, fft_size, tier_max
        
        print(f"  Precomputing FFTs for tier: D={D}, fft_size={fft_size}")
        
        # Extract the relevant degree range from simple_burau_full
        # The simple matrices are centered, so we need the middle D coefficients
        half_D = D // 2
        start = self.center - half_D
        end = start + D
        
        simple_subset = self.simple_burau_full[:, :, :, start:end].float()
        
        # Compute FFT of simple matrices for this tier
        simple_fft = torch.fft.rfft(simple_subset, n=fft_size, dim=-1)
        
        self.tier_cache[fft_size] = (simple_fft, D)
        self.current_D = D
        self.current_fft_size = fft_size
        
        mem_mb = simple_fft.numel() * 8 / 1e6
        print(f"  Tier cache: {mem_mb:.1f} MB")
        
        return D, fft_size, tier_max
    
    def get_suffix_fft(self, suffix_indices: torch.Tensor) -> torch.Tensor:
        """Get precomputed FFTs for the given suffix indices."""
        simple_fft, _ = self.tier_cache[self.current_fft_size]
        return simple_fft[suffix_indices]
    
    def compact_matrices(self, matrices: torch.Tensor, from_D: int) -> torch.Tensor:
        """
        Extract the central from_D coefficients from full-size matrices.
        """
        if matrices.shape[-1] == from_D:
            return matrices
        
        full_D = matrices.shape[-1]
        full_center = full_D // 2
        half_D = from_D // 2
        
        start = full_center - half_D
        end = start + from_D
        
        return matrices[..., start:end]
    
    def expand_matrices(self, matrices: torch.Tensor, to_D: int) -> torch.Tensor:
        """
        Expand matrices to larger degree window, padding with zeros.
        """
        if matrices.shape[-1] == to_D:
            return matrices
        
        from_D = matrices.shape[-1]
        N = matrices.shape[0]
        
        result = torch.zeros(N, 3, 3, to_D, dtype=matrices.dtype, device=matrices.device)
        
        from_center = from_D // 2
        to_center = to_D // 2
        
        offset = to_center - from_center
        result[..., offset:offset + from_D] = matrices
        
        return result
    
    def matmul_batch(self, A: torch.Tensor, suffix_indices: torch.Tensor,
                     chunk_size: int = 10000) -> torch.Tensor:
        """
        Batch matrix multiply using current tier's FFT size.
        
        A should already be compacted to current_D.
        Returns result with out_D = 2 * current_D - 1 coefficients.
        """
        N = A.shape[0]
        D = self.current_D
        fft_size = self.current_fft_size
        out_D = 2 * D - 1
        
        # The irfft output size is fft_size, but we only need out_D coefficients
        # Make sure fft_size >= out_D (it should be by construction)
        assert fft_size >= out_D, f"fft_size {fft_size} < out_D {out_D}"
        
        C = torch.zeros(N, 3, 3, out_D, dtype=COMPUTE_DTYPE_INT, device=self.device)
        
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_N = end - start
            
            A_chunk = A[start:end].float()
            chunk_indices = suffix_indices[start:end]
            
            # FFT of A
            A_fft = torch.fft.rfft(A_chunk, n=fft_size, dim=-1)
            del A_chunk
            
            # Get precomputed FFTs for suffix matrices
            B_fft = self.get_suffix_fft(chunk_indices)
            
            # Matrix multiply in FFT space using loop (often faster than einsum for cache)
            fft_len = A_fft.shape[-1]
            C_fft = torch.zeros(chunk_N, 3, 3, fft_len, dtype=torch.complex64, device=self.device)
            
            # Unrolled loop is often faster than einsum due to memory access patterns
            for i in range(3):
                for j in range(3):
                    C_fft[:, i, j] = (A_fft[:, i, 0] * B_fft[:, 0, j] + 
                                      A_fft[:, i, 1] * B_fft[:, 1, j] + 
                                      A_fft[:, i, 2] * B_fft[:, 2, j])
            
            del A_fft, B_fft
            
            # IFFT - output has fft_size elements
            C_real = torch.fft.irfft(C_fft, n=fft_size, dim=-1)
            del C_fft
            
            # Truncate to out_D, round, mod p
            C_int = torch.round(C_real[..., :out_D]).to(COMPUTE_DTYPE_INT) % self.prime
            del C_real
            
            C[start:end] = C_int
            del C_int
        
        return C


def compute_projlen_batch(matrices: torch.Tensor) -> torch.Tensor:
    """Compute projective length for a batch of matrices."""
    N, _, _, D = matrices.shape
    device = matrices.device
    
    # Flatten to find nonzero
    flat = matrices.reshape(N, 9, D)
    degree_has_nonzero = (flat != 0).any(dim=1)  # (N, D)
    
    has_nonzero = degree_has_nonzero.any(dim=-1)  # (N,)
    
    # Find min and max degree with nonzero
    min_degrees = degree_has_nonzero.int().argmax(dim=-1)
    max_degrees = D - 1 - degree_has_nonzero.flip(dims=[-1]).int().argmax(dim=-1)
    
    projlens = max_degrees - min_degrees + 1
    projlens = torch.where(has_nonzero, projlens, torch.zeros_like(projlens))
    
    return projlens.to(torch.int32)


# =============================================================================
# COMPACT MATRIX STORAGE
# =============================================================================

class CompactMatrixStorage:
    """
    Stores matrices in compact form with only the non-zero degree range.
    
    For each matrix, stores:
    - coefficients: (3, 3, width) where width = max_deg - min_deg + 1
    - min_degree: offset from center
    - width: number of coefficients stored
    
    This can reduce memory by 10-50x for sparse matrices!
    """
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def compact(self, matrices: torch.Tensor, center: int) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Convert full matrices to compact storage.
        
        Returns:
            coeffs_list: List of (3, 3, width) tensors (variable width)
            min_degrees: (N,) tensor of minimum degrees relative to center
            widths: (N,) tensor of widths
        """
        N, _, _, D = matrices.shape
        
        # Find degree bounds for each matrix
        flat = matrices.reshape(N, 9, D)
        degree_has_nonzero = (flat != 0).any(dim=1)
        
        has_nonzero = degree_has_nonzero.any(dim=-1)
        
        min_indices = degree_has_nonzero.int().argmax(dim=-1)
        max_indices = D - 1 - degree_has_nonzero.flip(dims=[-1]).int().argmax(dim=-1)
        
        # For all-zero matrices, set sensible defaults
        min_indices = torch.where(has_nonzero, min_indices, torch.full_like(min_indices, center))
        max_indices = torch.where(has_nonzero, max_indices, torch.full_like(max_indices, center))
        
        widths = max_indices - min_indices + 1
        min_degrees = min_indices - center
        
        # Extract compact coefficients
        coeffs_list = []
        for i in range(N):
            start = min_indices[i].item()
            end = max_indices[i].item() + 1
            coeffs_list.append(matrices[i, :, :, start:end])
        
        return coeffs_list, min_degrees, widths
    
    def expand(self, coeffs_list: list[torch.Tensor], min_degrees: torch.Tensor, 
               widths: torch.Tensor, target_D: int) -> torch.Tensor:
        """
        Expand compact storage back to full matrices.
        """
        N = len(coeffs_list)
        center = target_D // 2
        
        matrices = torch.zeros(N, 3, 3, target_D, dtype=coeffs_list[0].dtype, device=self.device)
        
        for i in range(N):
            start = center + min_degrees[i].item()
            end = start + widths[i].item()
            matrices[i, :, :, start:end] = coeffs_list[i]
        
        return matrices


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
# GPU BUCKETS
# =============================================================================

class GPUBuckets:
    """Maintains reservoir-sampled buckets entirely on GPU."""
    
    def __init__(self, bucket_size: int, device: torch.device):
        self.bucket_size = bucket_size
        self.device = device
        self.data: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    
    def add_chunk(
        self,
        matrices: torch.Tensor,
        words: torch.Tensor, 
        lengths: torch.Tensor,
        projlens: torch.Tensor,
        is_bootstrap: bool
    ):
        if len(matrices) == 0:
            return
        
        matrices = matrices.to(STORAGE_DTYPE_MATRIX)
        words = words.to(STORAGE_DTYPE_WORD)
        lengths = lengths.to(STORAGE_DTYPE_LENGTH)
        
        priorities = torch.rand(len(matrices), device=self.device)
        unique_pls = torch.unique(projlens)
        
        for pl in unique_pls.tolist():
            mask = (projlens == pl)
            new_mat = matrices[mask]
            new_words = words[mask]
            new_lengths = lengths[mask]
            new_priorities = priorities[mask]
            
            if pl not in self.data:
                if is_bootstrap or len(new_mat) <= self.bucket_size:
                    self.data[pl] = (new_mat, new_words, new_lengths, new_priorities)
                else:
                    _, topk_idx = torch.topk(new_priorities, self.bucket_size, largest=False)
                    self.data[pl] = (
                        new_mat[topk_idx],
                        new_words[topk_idx],
                        new_lengths[topk_idx],
                        new_priorities[topk_idx]
                    )
            else:
                old_mat, old_words, old_lengths, old_priorities = self.data[pl]
                
                merged_mat = torch.cat([old_mat, new_mat], dim=0)
                merged_words = torch.cat([old_words, new_words], dim=0)
                merged_lengths = torch.cat([old_lengths, new_lengths], dim=0)
                merged_priorities = torch.cat([old_priorities, new_priorities], dim=0)
                
                if is_bootstrap or len(merged_mat) <= self.bucket_size:
                    self.data[pl] = (merged_mat, merged_words, merged_lengths, merged_priorities)
                else:
                    _, topk_idx = torch.topk(merged_priorities, self.bucket_size, largest=False)
                    self.data[pl] = (
                        merged_mat[topk_idx],
                        merged_words[topk_idx],
                        merged_lengths[topk_idx],
                        merged_priorities[topk_idx]
                    )
                    del merged_mat, merged_words, merged_lengths, merged_priorities
    
    def get_buckets(self) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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

class BraidSearchAdaptive:
    """
    ADAPTIVE FFT braid search - uses level-appropriate FFT sizes.
    
    At level 20, uses fft_size=256 instead of 4096 = 16x smaller!
    This gives massive speedup at early levels and reduces memory.
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
        
        self.max_D = simple_burau.shape[-1]
        self.center = self.max_D // 2
        
        # Initialize adaptive FFT manager
        self.fft_manager = AdaptiveFFTManager(
            simple_burau, self.max_D, self.device, config.prime
        )
        
        # Current working D (updated per level)
        self.current_D = self.max_D
        
        self.buckets: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.kernel_braids: list[torch.Tensor] = []
        self.stats = {
            "candidates_per_level": [], 
            "buckets_per_level": [],
            "time_per_level": [],
            "time_matmul": [],
            "time_sampling": [],
            "fft_sizes": [],
        }
        self.start_level = 1
    
    def initialize(self):
        """Start with the identity braid."""
        # Use full D for storage (will be compacted during computation)
        identity_matrix = torch.zeros(1, 3, 3, self.max_D, dtype=STORAGE_DTYPE_MATRIX, device=self.device)
        identity_matrix[0, 0, 0, self.center] = 1
        identity_matrix[0, 1, 1, self.center] = 1
        identity_matrix[0, 2, 2, self.center] = 1
        
        identity_word = torch.zeros(1, self.config.max_length, dtype=STORAGE_DTYPE_WORD, device=self.device)
        identity_length = torch.zeros(1, dtype=STORAGE_DTYPE_LENGTH, device=self.device)
        
        self.buckets[1] = (identity_matrix, identity_word, identity_length)
        
        print(f"Initialized with identity braid")
        print(f"Max degree window: {self.max_D}")
        print(f"Storage types: matrix={STORAGE_DTYPE_MATRIX}, word={STORAGE_DTYPE_WORD}")
        print(f"Config: bucket_size={self.config.bucket_size}, "
              f"bootstrap={self.config.bootstrap_length}, "
              f"max_length={self.config.max_length}, "
              f"chunk_size={self.config.expansion_chunk_size}, "
              f"matmul_chunk={self.config.matmul_chunk_size}, "
              f"use_best={self.config.use_best if self.config.use_best > 0 else 'all'}")
        print(f"⚡ ADAPTIVE FFT: enabled (auto-sizing per level)")
    
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
        new_D = self.max_D
        new_max_length = self.config.max_length
        
        needs_matrix_resize = (saved_D != new_D)
        needs_word_resize = (saved_max_length != new_max_length)
        
        self.stats = checkpoint.get('stats', self.stats)
        
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
            else:
                mat, words, lengths = bucket_data
            
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
                lengths.to(STORAGE_DTYPE_LENGTH).to(self.device)
            )
        
        total_braids = sum(m.shape[0] for m, _, _ in self.buckets.values())
        print(f"  Loaded level {saved_level}, {total_braids} braids")
        print(f"  Resuming from level {self.start_level}")
        
        return self.start_level
    
    def save_checkpoint(self, level: int, path: str):
        """Save current state to disk."""
        checkpoint = {
            "level": level,
            "config": {k: v for k, v in self.config.__dict__.items()},
            "stats": self.stats,
            "kernel_braids": [w.cpu() for w in self.kernel_braids],
            "buckets": {
                pl: (mat.cpu(), words.cpu(), lengths.cpu())
                for pl, (mat, words, lengths) in self.buckets.items()
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
        
        matrices = torch.cat(all_matrices, dim=0).to(COMPUTE_DTYPE_INT)
        words = torch.cat(all_words, dim=0)
        lengths = torch.cat(all_lengths, dim=0)
        
        batch_idx = torch.arange(len(lengths), device=self.device)
        last_pos = torch.clamp(lengths - 1, min=0).long()
        last_simples = words[batch_idx, last_pos].long()
        last_simples = torch.where(lengths > 0, last_simples, torch.zeros_like(last_simples))
        
        return matrices, words, lengths, last_simples
    
    def expand_and_multiply_chunk(
        self,
        matrices: torch.Tensor,
        words: torch.Tensor,
        lengths: torch.Tensor,
        braid_indices: torch.Tensor,
        suffix_indices: torch.Tensor,
        working_D: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand a chunk using adaptive FFT sizing."""
        num_candidates = len(braid_indices)
        
        parent_matrices = matrices[braid_indices]
        parent_words = words[braid_indices]
        parent_lengths = lengths[braid_indices]
        
        # Compact matrices to working D
        parent_compact = self.fft_manager.compact_matrices(parent_matrices, working_D)
        
        # Matrix multiply using adaptive FFT
        new_matrices = self.fft_manager.matmul_batch(
            parent_compact, 
            suffix_indices,
            chunk_size=self.config.matmul_chunk_size
        )
        
        # Update words
        new_words = parent_words.clone()
        batch_idx = torch.arange(num_candidates, device=self.device)
        new_words[batch_idx, parent_lengths.long()] = suffix_indices.to(STORAGE_DTYPE_WORD)
        new_lengths = parent_lengths + 1
        
        return new_matrices, new_words, new_lengths
    
    def recenter_to_storage(self, matrices: torch.Tensor, from_D: int) -> torch.Tensor:
        """
        Recenter matrices from working D back to storage (max_D).
        Also trim if output is larger than max_D.
        """
        out_D = matrices.shape[-1]  # This is 2*from_D - 1
        
        if out_D <= self.max_D:
            # Expand to max_D
            return self.fft_manager.expand_matrices(matrices, self.max_D)
        else:
            # Need to trim - center the output
            trim_total = out_D - self.max_D
            trim_left = trim_total // 2
            trimmed = matrices[..., trim_left:trim_left + self.max_D]
            return trimmed
    
    def process_level(self, level: int):
        """Process one level with adaptive FFT sizing."""
        level_start = time.time()
        
        is_bootstrap = (level <= self.config.bootstrap_length)
        mode = "BOOTSTRAP" if is_bootstrap else "SAMPLING"
        
        # Get appropriate FFT size for this level
        working_D, fft_size, tier_max = self.fft_manager.ensure_tier(level)
        
        print(f"\n{'='*60}")
        print(f"Level {level} - {mode}")
        print(f"{'='*60}")
        print(f"  FFT tier: D={working_D}, fft_size={fft_size} (up to level {tier_max})")
        
        use_best_limit = 0 if is_bootstrap else self.config.use_best
        matrices, words, lengths, last_simples = self.gather_level_braids(use_best=use_best_limit)
        num_starting = len(matrices)
        print(f"  Starting braids: {num_starting}")
        
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
        
        gpu_buckets = GPUBuckets(self.config.bucket_size, self.device)
        
        t_matmul_total = 0.0
        t_sample_total = 0.0
        projlen_counts: dict[int, int] = {}
        
        for chunk_idx in range(num_chunks):
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, num_candidates)
            
            chunk_braid_idx = braid_indices[start:end]
            chunk_suffix_idx = suffix_indices[start:end]
            
            t0 = time.time()
            chunk_matrices, chunk_words, chunk_lengths = self.expand_and_multiply_chunk(
                matrices, words, lengths,
                chunk_braid_idx, chunk_suffix_idx,
                working_D
            )
            
            # Recenter to storage D
            chunk_matrices = self.recenter_to_storage(chunk_matrices, working_D)
            t_matmul_total += time.time() - t0
            
            chunk_projlens = compute_projlen_batch(chunk_matrices)
            
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
                is_bootstrap
            )
            t_sample_total += time.time() - t0
            
            del chunk_matrices, chunk_words, chunk_lengths, chunk_projlens
        
        del matrices, words, lengths, last_simples
        del braid_indices, suffix_indices
        
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"  Projlen distribution:")
        for pl in sorted(projlen_counts.keys())[:10]:
            print(f"    projlen={pl}: {projlen_counts[pl]} braids")
        
        self.buckets = gpu_buckets.get_buckets()
        
        total_kept = sum(m.shape[0] for m, _, _ in self.buckets.values())
        
        total_bytes = 0
        for mat, wrds, lens in self.buckets.values():
            total_bytes += mat.numel() * mat.element_size()
            total_bytes += wrds.numel() * wrds.element_size()
            total_bytes += lens.numel() * lens.element_size()
        
        print(f"  Braids kept: {total_kept} (in {len(self.buckets)} buckets, {total_bytes/1e9:.2f} GB)")
        
        level_time = time.time() - level_start
        print(f"  ⚡ Timing: matmul={t_matmul_total:.2f}s, sampling={t_sample_total:.2f}s, total={level_time:.2f}s")
        
        self.stats["candidates_per_level"].append(num_candidates)
        self.stats["buckets_per_level"].append(len(self.buckets))
        self.stats["time_per_level"].append(level_time)
        self.stats["time_matmul"].append(t_matmul_total)
        self.stats["time_sampling"].append(t_sample_total)
        self.stats["fft_sizes"].append(fft_size)
        
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
BraidSearch = BraidSearchAdaptive
