"""
GPU-accelerated braid search with DYNAMIC DEGREE STORAGE.

Key innovation: Polynomial matrices are stored compactly based on actual degree span,
not padded to max_length. Storage grows dynamically as needed.

This gives:
- MUCH smaller memory footprint (especially at early levels)
- MUCH faster FFTs (smaller sizes at early levels)
- Same correctness guarantees

Architecture:
- Each matrix batch has a "D" (storage width) and "offset" (where degree 0 is)
- Simple Burau matrices stored in minimal form
- Buckets resize dynamically when needed
- FFT sizes computed from actual degree spans
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
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
    checkpoint_every: int = 9999
    device: str = "cuda"
    expansion_chunk_size: int = 100000
    use_best: int = 0
    
    # Dynamic degree parameters
    degree_margin: int = 10          # Extra coefficients for safety
    min_D: int = 32                  # Minimum storage width (for FFT efficiency)


# =============================================================================
# DTYPE CONFIGURATION
# =============================================================================

STORAGE_DTYPE_MATRIX = torch.int16
STORAGE_DTYPE_WORD = torch.int32
STORAGE_DTYPE_LENGTH = torch.int32
COMPUTE_DTYPE_INT = torch.int32


# =============================================================================
# COMPACT POLYNOMIAL STORAGE
# =============================================================================

class CompactPolyMatrix:
    """
    Stores polynomial matrices compactly with offset tracking.
    
    A batch of N polynomial 3x3 matrices is stored as:
    - data: (N, 3, 3, D) tensor where D is the compact storage width
    - offset: int, the index corresponding to degree 0
    
    Degree range is [-(offset), D - 1 - offset].
    """
    
    def __init__(self, data: torch.Tensor, offset: int):
        """
        Args:
            data: (N, 3, 3, D) tensor
            offset: index of degree 0 in the storage
        """
        self.data = data
        self.offset = offset
    
    @property
    def N(self) -> int:
        return self.data.shape[0]
    
    @property
    def D(self) -> int:
        return self.data.shape[-1]
    
    @property
    def device(self):
        return self.data.device
    
    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def min_degree(self) -> int:
        return -self.offset
    
    @property
    def max_degree(self) -> int:
        return self.D - 1 - self.offset
    
    def to(self, device_or_dtype):
        """Move to device or convert dtype."""
        return CompactPolyMatrix(self.data.to(device_or_dtype), self.offset)
    
    def clone(self):
        return CompactPolyMatrix(self.data.clone(), self.offset)
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        """Index into the batch dimension."""
        return CompactPolyMatrix(self.data[idx], self.offset)
    
    @staticmethod
    def zeros(N: int, D: int, offset: int, dtype=STORAGE_DTYPE_MATRIX, device='cpu'):
        """Create zero-initialized compact storage."""
        data = torch.zeros(N, 3, 3, D, dtype=dtype, device=device)
        return CompactPolyMatrix(data, offset)
    
    @staticmethod
    def identity(N: int, dtype=STORAGE_DTYPE_MATRIX, device='cpu'):
        """Create N identity matrices (minimal storage: D=1, offset=0)."""
        data = torch.zeros(N, 3, 3, 1, dtype=dtype, device=device)
        for i in range(3):
            data[:, i, i, 0] = 1
        return CompactPolyMatrix(data, offset=0)
    
    def expand_to(self, new_D: int, new_offset: int) -> 'CompactPolyMatrix':
        """
        Expand storage to new size with new offset.
        
        Returns a new CompactPolyMatrix with the data properly aligned.
        """
        if new_D == self.D and new_offset == self.offset:
            return self
        
        if new_D < self.D:
            raise ValueError(f"Cannot shrink from D={self.D} to D={new_D}")
        
        # Compute where old data goes in new storage
        # Old degree d was at index (d + old_offset)
        # New degree d should be at index (d + new_offset)
        # So old index i (representing degree i - old_offset) 
        # goes to new index (i - old_offset + new_offset)
        shift = new_offset - self.offset
        
        new_data = torch.zeros(self.N, 3, 3, new_D, dtype=self.dtype, device=self.device)
        
        # Copy old data to new location
        src_start = 0
        src_end = self.D
        dst_start = shift
        dst_end = shift + self.D
        
        # Handle edge cases
        if dst_start < 0:
            src_start = -dst_start
            dst_start = 0
        if dst_end > new_D:
            src_end = self.D - (dst_end - new_D)
            dst_end = new_D
        
        if src_start < src_end and dst_start < dst_end:
            new_data[:, :, :, dst_start:dst_end] = self.data[:, :, :, src_start:src_end]
        
        return CompactPolyMatrix(new_data, new_offset)
    
    def trim_to_content(self, margin: int = 2) -> 'CompactPolyMatrix':
        """
        Trim storage to actual non-zero content plus margin.
        Useful after operations that may have introduced zeros at edges.
        """
        # Find actual extent across all matrices
        any_nonzero = (self.data != 0).any(dim=(0, 1, 2))  # (D,)
        
        if not any_nonzero.any():
            # All zeros - return minimal storage
            return CompactPolyMatrix.zeros(self.N, 1, 0, self.dtype, self.device)
        
        nonzero_idx = torch.where(any_nonzero)[0]
        min_idx = max(0, nonzero_idx[0].item() - margin)
        max_idx = min(self.D, nonzero_idx[-1].item() + 1 + margin)
        
        new_D = max_idx - min_idx
        new_offset = self.offset - min_idx
        new_data = self.data[:, :, :, min_idx:max_idx].contiguous()
        
        return CompactPolyMatrix(new_data, new_offset)
    
    @staticmethod
    def cat(matrices: list['CompactPolyMatrix'], dim: int = 0) -> 'CompactPolyMatrix':
        """
        Concatenate multiple CompactPolyMatrix objects along batch dimension.
        Automatically expands all to common storage size.
        """
        if len(matrices) == 0:
            raise ValueError("Cannot concatenate empty list")
        if len(matrices) == 1:
            return matrices[0]
        
        # Find required storage to hold all
        min_deg = min(m.min_degree for m in matrices)
        max_deg = max(m.max_degree for m in matrices)
        new_D = max_deg - min_deg + 1
        new_offset = -min_deg
        
        # Expand all matrices to common size
        expanded = [m.expand_to(new_D, new_offset) for m in matrices]
        
        # Concatenate
        cat_data = torch.cat([m.data for m in expanded], dim=dim)
        return CompactPolyMatrix(cat_data, new_offset)


# =============================================================================
# COMPACT SIMPLE BURAU STORAGE
# =============================================================================

class CompactSimpleBurau:
    """
    Stores the 24 simple Burau matrices compactly.
    
    Each simple braid matrix has a small degree span (typically 2-3).
    We store each one with its own minimal storage and offset.
    """
    
    def __init__(self, simple_burau_full: torch.Tensor, loaded_center: int, device: torch.device):
        """
        Args:
            simple_burau_full: (24, 3, 3, D_full) tensor from file
            loaded_center: where degree 0 is in the loaded tensor
            device: target device
        """
        self.device = device
        self.num_simples = 24
        
        # Analyze each simple braid to find its degree span
        self.matrices = []  # List of (3, 3, D_i) tensors
        self.offsets = []   # List of offsets
        self.min_degrees = []
        self.max_degrees = []
        
        self.global_min_degree = 0
        self.global_max_degree = 0
        
        for s in range(24):
            mat = simple_burau_full[s]  # (3, 3, D_full)
            
            # Find non-zero extent
            nonzero_mask = (mat != 0).any(dim=(0, 1))  # (D_full,)
            
            if not nonzero_mask.any():
                # Zero matrix (shouldn't happen for valid simples)
                self.matrices.append(torch.zeros(3, 3, 1, dtype=STORAGE_DTYPE_MATRIX, device=device))
                self.offsets.append(0)
                self.min_degrees.append(0)
                self.max_degrees.append(0)
                continue
            
            nonzero_idx = torch.where(nonzero_mask)[0]
            min_idx = nonzero_idx[0].item()
            max_idx = nonzero_idx[-1].item()
            
            # Convert to degrees
            min_deg = min_idx - loaded_center
            max_deg = max_idx - loaded_center
            
            # Extract compact storage
            compact_D = max_idx - min_idx + 1
            compact_offset = -min_deg  # offset such that index 0 corresponds to min_deg
            
            compact_mat = mat[:, :, min_idx:max_idx+1].to(STORAGE_DTYPE_MATRIX).to(device)
            
            self.matrices.append(compact_mat)
            self.offsets.append(compact_offset)
            self.min_degrees.append(min_deg)
            self.max_degrees.append(max_deg)
            
            self.global_min_degree = min(self.global_min_degree, min_deg)
            self.global_max_degree = max(self.global_max_degree, max_deg)
        
        # The maximum degree shift per multiplication
        self.max_shift = max(abs(self.global_min_degree), abs(self.global_max_degree))
        
        print(f"CompactSimpleBurau: loaded 24 simples")
        print(f"  Degree range: [{self.global_min_degree}, {self.global_max_degree}]")
        print(f"  Max shift per multiply: ±{self.max_shift}")
        print(f"  Storage sizes: {[m.shape[-1] for m in self.matrices]}")
    
    def get_expanded(self, indices: torch.Tensor, target_D: int, target_offset: int) -> torch.Tensor:
        """
        Get simple Burau matrices for given indices, expanded to target size.
        
        Args:
            indices: (N,) tensor of simple indices (0-23)
            target_D: target storage width
            target_offset: target offset
            
        Returns:
            (N, 3, 3, target_D) tensor
        """
        N = len(indices)
        result = torch.zeros(N, 3, 3, target_D, dtype=STORAGE_DTYPE_MATRIX, device=self.device)
        
        for s in range(24):
            mask = (indices == s)
            count = mask.sum().item()
            if count == 0:
                continue
            
            mat = self.matrices[s]  # (3, 3, D_s)
            src_offset = self.offsets[s]
            src_D = mat.shape[-1]
            
            # Compute where to place in target
            # Degree d is at src index (d + src_offset), should go to target index (d + target_offset)
            # So src index i goes to target index (i - src_offset + target_offset)
            shift = target_offset - src_offset
            
            dst_start = max(0, shift)
            dst_end = min(target_D, shift + src_D)
            src_start = max(0, -shift)
            src_end = src_start + (dst_end - dst_start)
            
            if src_start < src_end:
                # Broadcast the single matrix to all matching indices
                result[mask, :, :, dst_start:dst_end] = mat[:, :, src_start:src_end].unsqueeze(0)
        
        return result


# =============================================================================
# POLYNOMIAL MULTIPLICATION WITH DYNAMIC SIZING
# =============================================================================

def poly_matmul_compact(
    A: CompactPolyMatrix, 
    B_data: torch.Tensor,
    B_offset: int,
    p: int
) -> CompactPolyMatrix:
    """
    Multiply compact polynomial matrices A by simple Burau matrices B.
    
    Uses optimal FFT size based on actual degree spans.
    
    Args:
        A: CompactPolyMatrix of shape (N, 3, 3, D_A)
        B_data: (N, 3, 3, D_B) tensor of simple Burau matrices
        B_offset: offset for B matrices
        p: prime modulus
        
    Returns:
        CompactPolyMatrix with result
    """
    N = A.N
    D_A = A.D
    D_B = B_data.shape[-1]
    device = A.device
    
    # Output degree span: convolution of [A_min, A_max] with [B_min, B_max]
    # gives [A_min + B_min, A_max + B_max]
    A_min_deg, A_max_deg = A.min_degree, A.max_degree
    B_min_deg = -B_offset
    B_max_deg = D_B - 1 - B_offset
    
    out_min_deg = A_min_deg + B_min_deg
    out_max_deg = A_max_deg + B_max_deg
    out_D = out_max_deg - out_min_deg + 1
    out_offset = -out_min_deg
    
    # Convolution output size
    conv_len = D_A + D_B - 1
    
    # FFT size (next power of 2)
    fft_size = 1
    while fft_size < conv_len:
        fft_size *= 2
    fft_size = max(fft_size, 32)  # Minimum for efficiency
    
    # Perform matrix multiplication
    A_data = A.data.to(COMPUTE_DTYPE_INT)
    B_data = B_data.to(COMPUTE_DTYPE_INT)
    
    C_data = torch.zeros(N, 3, 3, conv_len, dtype=COMPUTE_DTYPE_INT, device=device)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = A_data[:, i, k, :].float()  # (N, D_A)
                b = B_data[:, k, j, :].float()  # (N, D_B)
                
                a_fft = torch.fft.rfft(a, n=fft_size)
                b_fft = torch.fft.rfft(b, n=fft_size)
                c = torch.fft.irfft(a_fft * b_fft, n=fft_size)[:, :conv_len]
                
                C_data[:, i, j, :] = (C_data[:, i, j, :] + torch.round(c).to(COMPUTE_DTYPE_INT)) % p
    
    # The convolution result has offset = A_offset + B_offset
    # (degree 0 in A was at A_offset, degree 0 in B was at B_offset,
    #  their product is degree 0 in output, at position A_offset + B_offset)
    result_offset = A.offset + B_offset
    
    return CompactPolyMatrix(C_data, result_offset)


def compute_projlen_compact(matrices: CompactPolyMatrix) -> torch.Tensor:
    """Compute projective length for compact polynomial matrices."""
    N = matrices.N
    D = matrices.D
    device = matrices.device
    
    projlens = torch.zeros(N, dtype=torch.int32, device=device)
    
    data = matrices.data
    sub_batch_size = 50000
    
    for start in range(0, N, sub_batch_size):
        end = min(start + sub_batch_size, N)
        batch = data[start:end]
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
# BUCKET STORAGE FOR COMPACT MATRICES
# =============================================================================

class CompactGPUBuckets:
    """
    Reservoir-sampled buckets storing CompactPolyMatrix objects.
    
    Each bucket maintains matrices with consistent D and offset.
    When adding matrices with different sizes, we expand as needed.
    """
    
    def __init__(self, bucket_size: int, device: torch.device):
        self.bucket_size = bucket_size
        self.device = device
        # projlen -> (CompactPolyMatrix, words, lengths, priorities)
        self.data: dict[int, tuple[CompactPolyMatrix, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    
    def add_chunk(
        self,
        matrices: CompactPolyMatrix,
        words: torch.Tensor,
        lengths: torch.Tensor,
        projlens: torch.Tensor,
        is_bootstrap: bool
    ):
        """Add a chunk of candidates with reservoir sampling."""
        if matrices.N == 0:
            return
        
        # Convert to storage types
        mat_storage = matrices.to(STORAGE_DTYPE_MATRIX)
        words = words.to(STORAGE_DTYPE_WORD)
        lengths = lengths.to(STORAGE_DTYPE_LENGTH)
        
        priorities = torch.rand(matrices.N, device=self.device)
        unique_pls = torch.unique(projlens)
        
        for pl in unique_pls.tolist():
            mask = (projlens == pl)
            new_mat = mat_storage[mask]
            new_words = words[mask]
            new_lengths = lengths[mask]
            new_priorities = priorities[mask]
            
            if pl not in self.data:
                if is_bootstrap or new_mat.N <= self.bucket_size:
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
                
                # Concatenate (automatically expands to common size)
                merged_mat = CompactPolyMatrix.cat([old_mat, new_mat])
                merged_words = torch.cat([old_words, new_words], dim=0)
                merged_lengths = torch.cat([old_lengths, new_lengths], dim=0)
                merged_priorities = torch.cat([old_priorities, new_priorities], dim=0)
                
                if is_bootstrap or merged_mat.N <= self.bucket_size:
                    self.data[pl] = (merged_mat, merged_words, merged_lengths, merged_priorities)
                else:
                    _, topk_idx = torch.topk(merged_priorities, self.bucket_size, largest=False)
                    self.data[pl] = (
                        merged_mat[topk_idx],
                        merged_words[topk_idx],
                        merged_lengths[topk_idx],
                        merged_priorities[topk_idx]
                    )
    
    def get_buckets(self) -> dict[int, tuple[CompactPolyMatrix, torch.Tensor, torch.Tensor]]:
        """Return final buckets without priorities."""
        return {
            pl: (mat, words, lengths)
            for pl, (mat, words, lengths, _) in self.data.items()
        }
    
    def total_count(self) -> int:
        return sum(mat.N for mat, _, _, _ in self.data.values())
    
    def memory_usage(self) -> int:
        """Return total bytes used by bucket storage."""
        total = 0
        for mat, words, lengths, prio in self.data.values():
            total += mat.data.numel() * mat.data.element_size()
            total += words.numel() * words.element_size()
            total += lengths.numel() * lengths.element_size()
            total += prio.numel() * prio.element_size()
        return total


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
# MAIN SEARCH ALGORITHM
# =============================================================================

class BraidSearchDynamic:
    """
    GPU-accelerated braid search with dynamic degree storage.
    
    Key features:
    - Matrices stored compactly based on actual degree span
    - FFT sizes adapt to content
    - Memory grows gradually with level, not fixed upfront
    """
    
    def __init__(
        self,
        simple_burau: CompactSimpleBurau,
        valid_suffixes: torch.Tensor,
        num_valid_suffixes: torch.Tensor,
        config: Config
    ):
        self.config = config
        self.device = torch.device(config.device)
        
        self.simple_burau = simple_burau
        self.valid_suffixes = valid_suffixes.to(self.device)
        self.num_valid_suffixes = num_valid_suffixes.to(self.device)
        
        # Buckets store CompactPolyMatrix objects
        self.buckets: dict[int, tuple[CompactPolyMatrix, torch.Tensor, torch.Tensor]] = {}
        self.kernel_braids: list[torch.Tensor] = []
        self.stats = {
            "candidates_per_level": [],
            "buckets_per_level": [],
            "time_per_level": [],
            "time_matmul": [],
            "time_sampling": [],
            "D_per_level": [],
            "fft_size_per_level": [],
        }
        self.start_level = 1
        self.current_level = 0
    
    def initialize(self):
        """Start with the identity braid."""
        # Identity matrix: minimal storage (D=1, offset=0)
        identity_matrix = CompactPolyMatrix.identity(1, STORAGE_DTYPE_MATRIX, self.device)
        identity_word = torch.zeros(1, self.config.max_length, dtype=STORAGE_DTYPE_WORD, device=self.device)
        identity_length = torch.zeros(1, dtype=STORAGE_DTYPE_LENGTH, device=self.device)
        
        self.buckets[1] = (identity_matrix, identity_word, identity_length)
        
        print(f"Initialized with identity braid")
        print(f"Simple Burau max shift: ±{self.simple_burau.max_shift}")
        print(f"Config: bucket_size={self.config.bucket_size}, "
              f"bootstrap={self.config.bootstrap_length}, "
              f"max_length={self.config.max_length}")
    
    def get_expected_D_for_level(self, level: int) -> int:
        """Estimate required D for matrices at a given level."""
        # After L multiplications, degree span is at most 2 * L * max_shift + 1
        span = 2 * level * self.simple_burau.max_shift + 1
        # Add margin and round up for FFT efficiency
        D = span + self.config.degree_margin
        D = max(D, self.config.min_D)
        # Round to multiple of 16 for memory alignment
        D = ((D + 15) // 16) * 16
        return D
    
    def gather_level_braids(self, use_best: int = 0) -> tuple[CompactPolyMatrix, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather braids from current buckets, returning compact matrices."""
        if not self.buckets:
            raise RuntimeError("No braids to process!")
        
        sorted_projlens = sorted(self.buckets.keys())
        
        all_matrices = []
        all_words = []
        all_lengths = []
        total_selected = 0
        
        for projlen in sorted_projlens:
            matrices, words, lengths = self.buckets[projlen]
            bucket_count = matrices.N
            
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
        
        # Concatenate (auto-expands to common size)
        matrices = CompactPolyMatrix.cat(all_matrices)
        words = torch.cat(all_words, dim=0)
        lengths = torch.cat(all_lengths, dim=0)
        
        # Get last simples
        batch_idx = torch.arange(len(lengths), device=self.device)
        last_pos = torch.clamp(lengths - 1, min=0).long()
        last_simples = words[batch_idx, last_pos].long()
        last_simples = torch.where(lengths > 0, last_simples, torch.zeros_like(last_simples))
        
        return matrices, words, lengths, last_simples
    
    def expand_and_multiply_chunk(
        self,
        matrices: CompactPolyMatrix,
        words: torch.Tensor,
        lengths: torch.Tensor,
        braid_indices: torch.Tensor,
        suffix_indices: torch.Tensor
    ) -> tuple[CompactPolyMatrix, torch.Tensor, torch.Tensor]:
        """Expand a chunk: gather parent matrices, multiply by suffix matrices."""
        num_candidates = len(braid_indices)
        
        # Gather parents
        parent_matrices = matrices[braid_indices]
        parent_words = words[braid_indices]
        parent_lengths = lengths[braid_indices]
        
        # Get suffix matrices expanded to match parent storage
        suffix_data = self.simple_burau.get_expanded(
            suffix_indices,
            target_D=parent_matrices.D,
            target_offset=parent_matrices.offset
        )
        
        # Actually, for multiplication, B should be in its natural form
        # Let me reconsider: we want A * B where A is parent, B is simple
        # The output offset depends on both input offsets
        
        # Get suffix in compact form - we need to handle variable sizes
        # For simplicity, expand all simples to a common size first
        max_simple_D = max(m.shape[-1] for m in self.simple_burau.matrices)
        
        # Get unique suffix indices and their compact forms
        unique_suffixes = torch.unique(suffix_indices)
        
        # Build suffix tensor with consistent sizing
        suffix_D = max_simple_D
        suffix_offset = self.simple_burau.max_shift  # Center the simples
        
        suffix_data = torch.zeros(num_candidates, 3, 3, suffix_D, 
                                   dtype=STORAGE_DTYPE_MATRIX, device=self.device)
        
        for s in unique_suffixes.tolist():
            mask = (suffix_indices == s)
            mat = self.simple_burau.matrices[s]
            src_D = mat.shape[-1]
            src_offset = self.simple_burau.offsets[s]
            
            # Place in output
            shift = suffix_offset - src_offset
            dst_start = max(0, shift)
            dst_end = min(suffix_D, shift + src_D)
            src_start = max(0, -shift)
            src_end = src_start + (dst_end - dst_start)
            
            if src_start < src_end:
                suffix_data[mask, :, :, dst_start:dst_end] = mat[:, :, src_start:src_end]
        
        # Perform multiplication
        new_matrices = poly_matmul_compact(
            parent_matrices,
            suffix_data,
            suffix_offset,
            self.config.prime
        )
        
        # Update words
        new_words = parent_words.clone()
        batch_idx = torch.arange(num_candidates, device=self.device)
        new_words[batch_idx, parent_lengths.long()] = suffix_indices.to(STORAGE_DTYPE_WORD)
        new_lengths = parent_lengths + 1
        
        return new_matrices, new_words, new_lengths
    
    def process_level(self, level: int):
        """Process one level with dynamic degree storage."""
        level_start = time.time()
        self.current_level = level
        
        is_bootstrap = (level <= self.config.bootstrap_length)
        mode = "BOOTSTRAP" if is_bootstrap else "SAMPLING"
        
        expected_D = self.get_expected_D_for_level(level)
        
        print(f"\n{'='*60}")
        print(f"Level {level} - {mode}")
        print(f"{'='*60}")
        print(f"  Expected D for this level: {expected_D}")
        
        use_best_limit = 0 if is_bootstrap else self.config.use_best
        matrices, words, lengths, last_simples = self.gather_level_braids(use_best=use_best_limit)
        num_starting = matrices.N
        print(f"  Starting braids: {num_starting} (current D={matrices.D}, offset={matrices.offset})")
        
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
        
        chunk_size = self.config.expansion_chunk_size
        num_chunks = (num_candidates + chunk_size - 1) // chunk_size
        
        if num_chunks > 1:
            print(f"  Processing in {num_chunks} chunks...")
        
        gpu_buckets = CompactGPUBuckets(self.config.bucket_size, self.device)
        
        t_matmul_total = 0.0
        t_sample_total = 0.0
        projlen_counts: dict[int, int] = {}
        max_fft_used = 0
        
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
                chunk_braid_idx, chunk_suffix_idx
            )
            
            # Trim to actual content (remove excess zeros)
            chunk_matrices = chunk_matrices.trim_to_content(margin=4)
            t_matmul_total += time.time() - t0
            
            # Track FFT size used
            conv_len = matrices.D + self.simple_burau.max_shift * 2 + 1
            fft_size = 1
            while fft_size < conv_len:
                fft_size *= 2
            max_fft_used = max(max_fft_used, fft_size)
            
            chunk_projlens = compute_projlen_compact(chunk_matrices)
            
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
        
        del matrices, words, lengths, last_simples
        del braid_indices, suffix_indices
        
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"  Projlen distribution:")
        for pl in sorted(projlen_counts.keys())[:10]:
            print(f"    projlen={pl}: {projlen_counts[pl]} braids")
        if len(projlen_counts) > 10:
            print(f"    ... and {len(projlen_counts) - 10} more projlen values")
        
        self.buckets = gpu_buckets.get_buckets()
        
        total_kept = sum(m.N for m, _, _ in self.buckets.values())
        
        # Report memory usage and D values
        total_bytes = 0
        max_D = 0
        for mat, wrds, lens in self.buckets.values():
            total_bytes += mat.data.numel() * mat.data.element_size()
            total_bytes += wrds.numel() * wrds.element_size()
            total_bytes += lens.numel() * lens.element_size()
            max_D = max(max_D, mat.D)
        
        print(f"  Braids kept: {total_kept} (in {len(self.buckets)} buckets)")
        print(f"  Storage: D={max_D}, FFT={max_fft_used}, memory={total_bytes/1e6:.1f} MB")
        
        level_time = time.time() - level_start
        print(f"  Timing: matmul={t_matmul_total:.2f}s, sampling={t_sample_total:.2f}s, total={level_time:.2f}s")
        
        self.stats["candidates_per_level"].append(num_candidates)
        self.stats["buckets_per_level"].append(len(self.buckets))
        self.stats["time_per_level"].append(level_time)
        self.stats["time_matmul"].append(t_matmul_total)
        self.stats["time_sampling"].append(t_sample_total)
        self.stats["D_per_level"].append(max_D)
        self.stats["fft_size_per_level"].append(max_fft_used)
        
        return True
    
    def save_checkpoint(self, level: int, path: str):
        """Save current state to disk."""
        print(f"  Saving checkpoint to {path}...")
        
        # Convert buckets to serializable format
        buckets_data = {}
        for pl, (mat, words, lengths) in self.buckets.items():
            buckets_data[pl] = {
                'data': mat.data.cpu(),
                'offset': mat.offset,
                'words': words.cpu(),
                'lengths': lengths.cpu()
            }
        
        checkpoint = {
            "level": level,
            "config": {k: v for k, v in self.config.__dict__.items()},
            "stats": self.stats,
            "kernel_braids": [w.cpu() for w in self.kernel_braids],
            "buckets": buckets_data,
            "version": "dynamic_v1"
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load state from checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        saved_level = checkpoint['level']
        self.start_level = saved_level + 1
        
        self.stats = checkpoint.get('stats', self.stats)
        
        if 'kernel_braids' in checkpoint:
            self.kernel_braids = [w.clone() for w in checkpoint['kernel_braids']]
        
        self.buckets = {}
        for pl, bucket_data in checkpoint['buckets'].items():
            pl = int(pl)
            
            if isinstance(bucket_data, dict) and 'offset' in bucket_data:
                # New format with offset
                data = bucket_data['data'].to(STORAGE_DTYPE_MATRIX).to(self.device)
                offset = bucket_data['offset']
                words = bucket_data['words'].to(STORAGE_DTYPE_WORD).to(self.device)
                lengths = bucket_data['lengths'].to(STORAGE_DTYPE_LENGTH).to(self.device)
                mat = CompactPolyMatrix(data, offset)
            else:
                # Old format - assume centered
                if isinstance(bucket_data, dict):
                    data = bucket_data['matrices']
                    words = bucket_data['words']
                    lengths = bucket_data['lengths']
                else:
                    data, words, lengths = bucket_data
                
                data = data.to(STORAGE_DTYPE_MATRIX).to(self.device)
                words = words.to(STORAGE_DTYPE_WORD).to(self.device)
                lengths = lengths.to(STORAGE_DTYPE_LENGTH).to(self.device)
                offset = data.shape[-1] // 2
                mat = CompactPolyMatrix(data, offset)
            
            self.buckets[pl] = (mat, words, lengths)
        
        total_braids = sum(m.N for m, _, _ in self.buckets.values())
        print(f"  Loaded level {saved_level}")
        print(f"  Buckets: {len(self.buckets)} projlen values, {total_braids} total braids")
        print(f"  Resuming from level {self.start_level}")
        
        return self.start_level
    
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
            
            if self.stats["D_per_level"]:
                print(f"D progression: {self.stats['D_per_level'][:5]} ... {self.stats['D_per_level'][-3:]}")
                print(f"FFT progression: {self.stats['fft_size_per_level'][:5]} ... {self.stats['fft_size_per_level'][-3:]}")
            
            print(f"Total kernel elements (projlen=1) found: {sum(len(w) for w in self.kernel_braids)}")
            
            save_dir = checkpoint_dir if checkpoint_dir else "."
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            final_path = f"{save_dir}/final_state_level_{final_level}.pt"
            self.save_checkpoint(final_level, final_path)
            print(f"\nFinal state saved to {final_path}")
        
        return self.kernel_braids


# =============================================================================
# TABLE LOADING
# =============================================================================

def load_tables_dynamic(config: Config, table_path: str, device: torch.device):
    """Load precomputed tables and create compact representations."""
    tables = torch.load(table_path, weights_only=True)
    
    assert tables['n'] == 4, f"Expected n=4, got {tables['n']}"
    assert tables['p'] == config.prime, f"Table prime {tables['p']} != config prime {config.prime}"
    
    loaded_burau = tables['simple_burau']
    loaded_center = tables['center']
    
    # Create compact simple Burau storage
    simple_burau = CompactSimpleBurau(loaded_burau, loaded_center, device)
    
    # Load suffix tables
    loaded_valid_suffixes = tables['valid_suffixes']
    loaded_num_valid = tables['num_valid_suffixes']
    
    delta_idx = tables['delta_index']
    id_idx = tables['id_index']
    
    valid_suffixes = loaded_valid_suffixes.clone()
    num_valid_suffixes = loaded_num_valid.clone()
    
    valid_suffixes[id_idx] = valid_suffixes[delta_idx]
    num_valid_suffixes[id_idx] = num_valid_suffixes[delta_idx]
    
    print(f"Loaded tables from {table_path}")
    print(f"  Identity suffixes fixed: {num_valid_suffixes[id_idx]} valid")
    
    return simple_burau, valid_suffixes, num_valid_suffixes


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = Config(
        bucket_size=50000,
        max_length=100,
        bootstrap_length=5,
        prime=5,
        checkpoint_every=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        expansion_chunk_size=100000,
        degree_margin=10,
        min_D=32
    )
    
    device = torch.device(config.device)
    print(f"Using device: {device}")
    print(f"Dynamic degree storage enabled")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    table_path = os.path.join(project_root, "precomputed_tables", f"tables_B4_r1_p{config.prime}.pt")
    
    simple_burau, valid_suffixes, num_valid_suffixes = load_tables_dynamic(config, table_path, device)
    
    search = BraidSearchDynamic(simple_burau, valid_suffixes, num_valid_suffixes, config)
    kernel_braids = search.run(checkpoint_dir="checkpoints_dynamic")
    
    for i, words in enumerate(kernel_braids):
        print(f"\nBatch {i}: {len(words)} kernel elements")
        for word in words[:5]:
            print(f"  {word.tolist()}")


if __name__ == "__main__":
    main()
