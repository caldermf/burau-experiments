"""
MAXIMALLY OPTIMIZED braid search with:
1. Batched FFT matrix multiplication (3x fewer FFT ops)
2. Precomputed FFTs of simple Burau matrices (eliminates half of per-expansion FFTs)
3. NON-NEGATIVE DEGREES ONLY - halves memory and FFT size!
4. DEVIATION-BASED OPTIMIZATION - targets specific power of v based on Artin length!

Key insight: Kernel elements of the abelianization have a SPECIFIC power of v determined by
their Artin length. Instead of minimizing projlen (max_deg - min_deg), we minimize
deviation from the target power: target = (2/3) * artin_length.

For B_4, the Garside element Δ has Artin length 6 and evaluates to v^4 * (scalar),
so the ratio is 4/6 = 2/3.

CRITICAL: Artin length is the NUMBER OF ARTIN GENERATORS, not the max degree!
For simple elements in B_n, Artin length = number of inversions in the permutation.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
import time
import gc
from itertools import permutations

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
    device: str = "cuda"
    expansion_chunk_size: int = 100000
    use_best: int = 0
    matmul_chunk_size: int = 20000
    
    @property
    def degree_window(self) -> int:
        # Only non-negative degrees [0, degree_multiplier * max_length]
        return self.degree_multiplier * self.max_length + 1


# =============================================================================
# DTYPE CONFIGURATION
# =============================================================================

STORAGE_DTYPE_MATRIX = torch.int16
STORAGE_DTYPE_WORD = torch.int32
STORAGE_DTYPE_LENGTH = torch.int32
STORAGE_DTYPE_ARTIN = torch.int32  # For Artin lengths
COMPUTE_DTYPE_INT = torch.int32


# =============================================================================
# ARTIN LENGTH COMPUTATION
# =============================================================================

def count_inversions(perm: tuple) -> int:
    """Count inversions in a permutation. This equals the Artin length."""
    n = len(perm)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                count += 1
    return count


def compute_B4_simple_artin_lengths() -> torch.Tensor:
    """
    Compute TRUE Artin lengths for all 24 simple elements in B_4.
    
    Simple elements in B_n correspond to permutations in S_n.
    The Artin length of a simple is the number of inversions in its permutation.
    
    For B_4: 24 simples (= 4! permutations), with Artin lengths 0 to 6.
    - Identity (1234): 0 inversions
    - Δ (4321): 6 inversions
    
    Returns tensor of shape (24,) with Artin lengths indexed by simple number.
    """
    # Generate all permutations of (0,1,2,3) in lexicographic order
    # This matches how Garside normal form indexes simple elements
    perms = list(permutations(range(4)))
    
    artin_lengths = torch.zeros(24, dtype=STORAGE_DTYPE_ARTIN)
    
    for idx, perm in enumerate(perms):
        artin_lengths[idx] = count_inversions(perm)
    
    return artin_lengths


def get_simple_artin_lengths(n: int = 4) -> torch.Tensor:
    """
    Get TRUE Artin lengths for simple elements in B_n.
    
    Currently only B_4 is supported.
    """
    if n != 4:
        raise ValueError(f"Only B_4 is currently supported, got n={n}")
    
    return compute_B4_simple_artin_lengths()


# =============================================================================
# ULTRA-OPTIMIZED POLYNOMIAL MATRIX MULTIPLICATION
# =============================================================================

class FastPolyMatmul:
    """
    Precomputes and caches FFTs of simple Burau matrices for maximum speed.
    """
    
    def __init__(self, simple_burau: torch.Tensor, D: int, device: torch.device):
        self.D = D
        self.out_D = 2 * D - 1
        self.fft_size = 1 << (self.out_D).bit_length()
        self.device = device
        
        print(f"Precomputing FFTs of simple Burau matrices...")
        print(f"  D={D}, out_D={self.out_D}, fft_size={self.fft_size}")
        
        self.simple_burau = simple_burau.to(device)
        self.simple_burau_fft = torch.fft.rfft(
            simple_burau.float().to(device),
            n=self.fft_size,
            dim=-1
        )
        
        fft_mem = self.simple_burau_fft.numel() * 8 / 1e6
        print(f"  FFT cache: {fft_mem:.1f} MB")
    
    def matmul_batch(self, A: torch.Tensor, suffix_indices: torch.Tensor, 
                     p: int, chunk_size: int = 20000) -> torch.Tensor:
        """Optimized batch matrix multiply with precomputed FFTs."""
        N = A.shape[0]
        C = torch.zeros(N, 3, 3, self.out_D, dtype=COMPUTE_DTYPE_INT, device=self.device)
        
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            
            A_chunk = A[start:end].float()
            chunk_indices = suffix_indices[start:end]
            
            A_fft = torch.fft.rfft(A_chunk, n=self.fft_size, dim=-1)
            del A_chunk
            
            B_fft = self.simple_burau_fft[chunk_indices]
            
            C_fft = torch.einsum('nikf,nkjf->nijf', A_fft, B_fft)
            del A_fft, B_fft
            
            C_real = torch.fft.irfft(C_fft, n=self.fft_size, dim=-1)
            del C_fft
            
            C_int = torch.round(C_real[..., :self.out_D]).to(COMPUTE_DTYPE_INT) % p
            del C_real
            
            C[start:end] = C_int
            del C_int
        
        return C


def compute_deviation_batch(matrices: torch.Tensor, artin_lengths: torch.Tensor) -> torch.Tensor:
    """
    Compute deviation from target power for a batch of matrices.
    
    The deviation is projlen (max_deg - min_deg + 1) plus a penalty if the target
    degree (2/3 * artin_length) is outside the [min_deg, max_deg] range.
    
    Specifically (using *3 for integer arithmetic):
        projlen_times_3 = 3 * (max_deg - min_deg + 1)
        
        If target in [min_deg, max_deg]: penalty = 0
        If target < min_deg: penalty_times_3 = 3 * (min_deg - target) 
        If target > max_deg: penalty_times_3 = 3 * (target - max_deg)
        
        deviation_times_3 = projlen_times_3 + penalty_times_3
    
    Equivalently:
        deviation_times_3 = 3*(max(max_deg, target) - min(min_deg, target) + 1)
    
    A deviation_times_3 of 3 means projlen=1 and target is within range (kernel element!).
    """
    N, _, _, D = matrices.shape
    device = matrices.device
    
    deviations = torch.zeros(N, dtype=torch.int32, device=device)
    
    # Compute target_times_3 for each braid: target = (2/3) * artin, so target*3 = 2*artin
    target_times_3 = 2 * artin_lengths  # Shape: (N,)
    
    sub_batch_size = 50000
    for start in range(0, N, sub_batch_size):
        end = min(start + sub_batch_size, N)
        batch = matrices[start:end]
        batch_N = end - start
        batch_target = target_times_3[start:end]
        
        flat = batch.reshape(batch_N, -1)
        has_nonzero = (flat != 0).any(dim=-1)
        
        by_degree = batch.reshape(batch_N, 9, D)
        degree_has_nonzero = (by_degree != 0).any(dim=1)
        
        # Find min and max degrees with nonzero entries
        min_degrees = degree_has_nonzero.int().argmax(dim=-1)
        max_degrees = D - 1 - degree_has_nonzero.flip(dims=[-1]).int().argmax(dim=-1)
        
        # Scale by 3 for integer arithmetic
        min_deg_times_3 = 3 * min_degrees
        max_deg_times_3 = 3 * max_degrees
        
        # Compute expanded range that includes target
        upper = torch.maximum(max_deg_times_3, batch_target)
        lower = torch.minimum(min_deg_times_3, batch_target)
        
        # deviation_times_3 = (upper - lower) + 3 = expanded_projlen_times_3
        # This is projlen + penalty, where penalty accounts for target being outside [min, max]
        batch_deviation = upper - lower + 3  # +3 because projlen = (max - min + 1)
        
        # Zero matrix: set deviation to just 3 (projlen=1, no penalty since target=0 is at degree 0)
        batch_deviation = torch.where(has_nonzero, batch_deviation, 
                                      torch.full_like(batch_deviation, 3))
        
        deviations[start:end] = batch_deviation
    
    return deviations


def compute_projlen_batch(matrices: torch.Tensor) -> torch.Tensor:
    """Compute projective length for a batch of matrices (legacy, for compatibility)."""
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
# GPU BUCKETS (now keyed by deviation_times_3)
# =============================================================================

class GPUBuckets:
    """Maintains reservoir-sampled buckets entirely on GPU, keyed by deviation."""
    
    def __init__(self, bucket_size: int, device: torch.device):
        self.bucket_size = bucket_size
        self.device = device
        # Data: deviation -> (matrices, words, lengths, artin_lengths, priorities)
        self.data: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    
    def add_chunk(
        self,
        matrices: torch.Tensor,
        words: torch.Tensor, 
        lengths: torch.Tensor,
        artin_lengths: torch.Tensor,
        deviations: torch.Tensor,
        is_bootstrap: bool
    ):
        if len(matrices) == 0:
            return
        
        matrices = matrices.to(STORAGE_DTYPE_MATRIX)
        words = words.to(STORAGE_DTYPE_WORD)
        lengths = lengths.to(STORAGE_DTYPE_LENGTH)
        artin_lengths = artin_lengths.to(STORAGE_DTYPE_ARTIN)
        
        priorities = torch.rand(len(matrices), device=self.device)
        unique_devs = torch.unique(deviations)
        
        for dev in unique_devs.tolist():
            mask = (deviations == dev)
            new_mat = matrices[mask]
            new_words = words[mask]
            new_lengths = lengths[mask]
            new_artin = artin_lengths[mask]
            new_priorities = priorities[mask]
            
            if dev not in self.data:
                if is_bootstrap or len(new_mat) <= self.bucket_size:
                    self.data[dev] = (new_mat, new_words, new_lengths, new_artin, new_priorities)
                else:
                    _, topk_idx = torch.topk(new_priorities, self.bucket_size, largest=False)
                    self.data[dev] = (
                        new_mat[topk_idx],
                        new_words[topk_idx],
                        new_lengths[topk_idx],
                        new_artin[topk_idx],
                        new_priorities[topk_idx]
                    )
            else:
                old_mat, old_words, old_lengths, old_artin, old_priorities = self.data[dev]
                
                merged_mat = torch.cat([old_mat, new_mat], dim=0)
                merged_words = torch.cat([old_words, new_words], dim=0)
                merged_lengths = torch.cat([old_lengths, new_lengths], dim=0)
                merged_artin = torch.cat([old_artin, new_artin], dim=0)
                merged_priorities = torch.cat([old_priorities, new_priorities], dim=0)
                
                if is_bootstrap or len(merged_mat) <= self.bucket_size:
                    self.data[dev] = (merged_mat, merged_words, merged_lengths, merged_artin, merged_priorities)
                else:
                    _, topk_idx = torch.topk(merged_priorities, self.bucket_size, largest=False)
                    self.data[dev] = (
                        merged_mat[topk_idx],
                        merged_words[topk_idx],
                        merged_lengths[topk_idx],
                        merged_artin[topk_idx],
                        merged_priorities[topk_idx]
                    )
                    del merged_mat, merged_words, merged_lengths, merged_artin, merged_priorities
    
    def get_buckets(self) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return buckets without priorities."""
        return {
            dev: (mat, words, lengths, artin)
            for dev, (mat, words, lengths, artin, _) in self.data.items()
        }
    
    def total_count(self) -> int:
        return sum(mat.shape[0] for mat, _, _, _, _ in self.data.values())
    
    def clear(self):
        self.data.clear()


# =============================================================================
# MAIN ALGORITHM
# =============================================================================

class BraidSearchUltra:
    """
    ULTRA-OPTIMIZED GPU-accelerated search for braids with low deviation.
    
    Uses deviation from target power (2/3 * artin_length) instead of projlen.
    """
    
    def __init__(
        self,
        simple_burau: torch.Tensor,
        valid_suffixes: torch.Tensor,
        num_valid_suffixes: torch.Tensor,
        config: Config,
        simple_artin_lengths: torch.Tensor = None
    ):
        self.config = config
        self.device = torch.device(config.device)
        
        self.simple_burau = simple_burau.to(STORAGE_DTYPE_MATRIX).to(self.device)
        self.valid_suffixes = valid_suffixes.to(self.device)
        self.num_valid_suffixes = num_valid_suffixes.to(self.device)
        
        self.D = simple_burau.shape[-1]
        
        # Use provided Artin lengths or compute TRUE Artin lengths from permutations
        if simple_artin_lengths is not None:
            self.simple_artin_lengths = simple_artin_lengths.to(STORAGE_DTYPE_ARTIN).to(self.device)
            print(f"Using provided Artin lengths")
        else:
            self.simple_artin_lengths = get_simple_artin_lengths(n=4).to(self.device)
            print(f"Computed TRUE Artin lengths from permutation inversions")
        
        print(f"Simple Artin lengths: {self.simple_artin_lengths.tolist()}")
        print(f"  Range: [{self.simple_artin_lengths.min().item()}, {self.simple_artin_lengths.max().item()}]")
        print(f"  Distribution: {torch.bincount(self.simple_artin_lengths.int()).tolist()}")
        
        self.fast_matmul = FastPolyMatmul(simple_burau, self.D, self.device)
        
        # Buckets now keyed by deviation_times_3
        self.buckets: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.kernel_braids: list[torch.Tensor] = []
    
    def initialize(self):
        """Start with the identity braid."""
        identity_matrix = torch.zeros(1, 3, 3, self.D, dtype=STORAGE_DTYPE_MATRIX, device=self.device)
        for i in range(3):
            identity_matrix[0, i, i, 0] = 1  # v^0 = 1 at index 0
        
        identity_word = torch.zeros(1, self.config.max_length, dtype=STORAGE_DTYPE_WORD, device=self.device)
        identity_length = torch.zeros(1, dtype=STORAGE_DTYPE_LENGTH, device=self.device)
        identity_artin = torch.zeros(1, dtype=STORAGE_DTYPE_ARTIN, device=self.device)
        
        # Identity has deviation_times_3 = 0 (it's the trivial kernel element)
        # But we want to expand from it, so put it in bucket 0
        # Actually, identity evaluates to I = v^0 * I, target = (2/3)*0 = 0, deviation = 0
        self.buckets[0] = (identity_matrix, identity_word, identity_length, identity_artin)
        
        print(f"Initialized with identity braid")
        print(f"Degree window: [0, {self.D-1}] ({self.D} coefficients) - NON-NEGATIVE ONLY")
        print(f"FFT size: {self.fast_matmul.fft_size}")
        print(f"Storage types: matrix={STORAGE_DTYPE_MATRIX}, word={STORAGE_DTYPE_WORD}")
        print(f"Config: bucket_size={self.config.bucket_size}, "
              f"bootstrap={self.config.bootstrap_length}, "
              f"max_length={self.config.max_length}, "
              f"chunk_size={self.config.expansion_chunk_size}, "
              f"matmul_chunk={self.config.matmul_chunk_size}, "
              f"use_best={self.config.use_best if self.config.use_best > 0 else 'all'}")
        print(f"⚡ Ultra-optimized: precomputed Burau FFTs + non-negative degrees only")
        print(f"🎯 Deviation-based search: target = (2/3) * artin_length")
    
    def gather_level_braids(self, use_best: int = 0):
        """Gather braids from current buckets, prioritizing low deviation."""
        if not self.buckets:
            raise RuntimeError("No braids to process!")
        
        sorted_deviations = sorted(self.buckets.keys())
        
        all_matrices = []
        all_words = []
        all_lengths = []
        all_artin = []
        total_selected = 0
        
        for deviation in sorted_deviations:
            matrices, words, lengths, artin_lengths = self.buckets[deviation]
            bucket_count = len(matrices)
            
            if use_best > 0:
                remaining = use_best - total_selected
                if remaining <= 0:
                    break
                
                if bucket_count <= remaining:
                    all_matrices.append(matrices)
                    all_words.append(words)
                    all_lengths.append(lengths)
                    all_artin.append(artin_lengths)
                    total_selected += bucket_count
                else:
                    idx = torch.randperm(bucket_count, device=self.device)[:remaining]
                    all_matrices.append(matrices[idx])
                    all_words.append(words[idx])
                    all_lengths.append(lengths[idx])
                    all_artin.append(artin_lengths[idx])
                    total_selected += remaining
                    break
            else:
                all_matrices.append(matrices)
                all_words.append(words)
                all_lengths.append(lengths)
                all_artin.append(artin_lengths)
        
        matrices = torch.cat(all_matrices, dim=0).to(COMPUTE_DTYPE_INT)
        words = torch.cat(all_words, dim=0)
        lengths = torch.cat(all_lengths, dim=0)
        artin_lengths = torch.cat(all_artin, dim=0)
        
        batch_idx = torch.arange(len(lengths), device=self.device)
        last_pos = torch.clamp(lengths - 1, min=0).long()
        last_simples = words[batch_idx, last_pos].long()
        last_simples = torch.where(lengths > 0, last_simples, torch.zeros_like(last_simples))
        
        return matrices, words, lengths, artin_lengths, last_simples
    
    def expand_and_multiply_chunk(
        self,
        matrices: torch.Tensor,
        words: torch.Tensor,
        lengths: torch.Tensor,
        artin_lengths: torch.Tensor,
        braid_indices: torch.Tensor,
        suffix_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand a chunk using PRECOMPUTED FFTs of simple Burau matrices."""
        num_candidates = len(braid_indices)
        
        parent_matrices = matrices[braid_indices]
        parent_words = words[braid_indices]
        parent_lengths = lengths[braid_indices]
        parent_artin = artin_lengths[braid_indices]
        
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
        
        # Update Artin lengths: add the TRUE Artin length of the appended simple
        suffix_artin = self.simple_artin_lengths[suffix_indices]
        new_artin = parent_artin + suffix_artin
        
        return new_matrices, new_words, new_lengths, new_artin
    
    def recenter_matrices(self, matrices: torch.Tensor) -> torch.Tensor:
        """Trim matrices back to target degree window (from right only)."""
        current_D = matrices.shape[-1]
        target_D = self.D
        
        if current_D <= target_D:
            pad_needed = target_D - current_D
            return F.pad(matrices, (0, pad_needed), value=0)
        
        return matrices[..., :target_D]
    
    def process_level(self, level: int):
        """Process one level with ultra-fast matrix multiplication."""
        level_start = time.time()
        
        is_bootstrap = (level <= self.config.bootstrap_length)
        mode = "BOOTSTRAP" if is_bootstrap else "SAMPLING"
        
        print(f"\n{'='*60}")
        print(f"Level {level} - {mode}")
        print(f"{'='*60}")
        
        use_best_limit = 0 if is_bootstrap else self.config.use_best
        matrices, words, lengths, artin_lengths, last_simples = self.gather_level_braids(use_best=use_best_limit)
        num_starting = len(matrices)
        print(f"  Starting braids: {num_starting}")
        
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
        deviation_counts: dict[int, int] = {}
        
        for chunk_idx in range(num_chunks):
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, num_candidates)
            
            chunk_braid_idx = braid_indices[start:end]
            chunk_suffix_idx = suffix_indices[start:end]
            
            t0 = time.time()
            chunk_matrices, chunk_words, chunk_lengths, chunk_artin = self.expand_and_multiply_chunk(
                matrices, words, lengths, artin_lengths,
                chunk_braid_idx, chunk_suffix_idx
            )
            chunk_matrices = self.recenter_matrices(chunk_matrices)
            t_matmul_total += time.time() - t0
            
            # Compute deviation instead of projlen
            chunk_deviations = compute_deviation_batch(chunk_matrices, chunk_artin)
            
            # Kernel elements have deviation_times_3 == 3 (projlen=1 with target in range)
            kernel_mask = (chunk_deviations == 3)
            num_kernel = kernel_mask.sum().item()
            if num_kernel > 0:
                # Filter out identity (length 0)
                nonidentity_mask = kernel_mask & (chunk_lengths > 0)
                num_kernel_nonid = nonidentity_mask.sum().item()
                if num_kernel_nonid > 0:
                    print(f"\n  🎉 FOUND {num_kernel_nonid} KERNEL ELEMENTS (deviation=1, projlen=1 with target in range)! 🎉")
                    self.kernel_braids.append(chunk_words[nonidentity_mask].cpu())
            
            unique_devs, counts = torch.unique(chunk_deviations, return_counts=True)
            for dev, c in zip(unique_devs.tolist(), counts.tolist()):
                deviation_counts[dev] = deviation_counts.get(dev, 0) + c
            
            t0 = time.time()
            gpu_buckets.add_chunk(
                chunk_matrices, chunk_words, chunk_lengths, chunk_artin, chunk_deviations,
                is_bootstrap
            )
            t_sample_total += time.time() - t0
            
            del chunk_matrices, chunk_words, chunk_lengths, chunk_artin, chunk_deviations
        
        del matrices, words, lengths, artin_lengths, last_simples
        del braid_indices, suffix_indices
        
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"  Deviation distribution (dev = projlen + penalty, shown as dev*3):")
        for dev in sorted(deviation_counts.keys())[:15]:
            # deviation_times_3 = 3*projlen + 3*penalty, so actual deviation = dev/3
            actual_dev = dev / 3.0
            print(f"    deviation={actual_dev:.2f} (x3={dev}): {deviation_counts[dev]} braids")
        
        self.buckets = gpu_buckets.get_buckets()
        
        total_kept = sum(m.shape[0] for m, _, _, _ in self.buckets.values())
        
        total_bytes = 0
        for mat, wrds, lens, artin in self.buckets.values():
            total_bytes += mat.numel() * mat.element_size()
            total_bytes += wrds.numel() * wrds.element_size()
            total_bytes += lens.numel() * lens.element_size()
            total_bytes += artin.numel() * artin.element_size()
        
        print(f"  Braids kept: {total_kept} (in {len(self.buckets)} buckets, {total_bytes/1e9:.2f} GB)")
        
        level_time = time.time() - level_start
        print(f"  ⚡ Timing: matmul={t_matmul_total:.2f}s, sampling={t_sample_total:.2f}s, total={level_time:.2f}s")
        
        return True
    
    def run(self):
        """Run the full search algorithm."""
        self.initialize()
        
        total_start = time.time()
        final_level = 0
        
        try:
            for level in range(1, self.config.max_length + 1):
                success = self.process_level(level)
                final_level = level
                
                if not success:
                    break
        
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted at level {final_level}!")
        
        finally:
            total_time = time.time() - total_start
            
            print(f"\n{'='*60}")
            print("SEARCH COMPLETE" if final_level == self.config.max_length else "SEARCH STOPPED")
            print(f"{'='*60}")
            print(f"Final level: {final_level}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Avg time per level: {total_time / max(1, final_level):.2f}s")
            print(f"Total kernel elements: {sum(len(w) for w in self.kernel_braids)}")
        
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
        
        assert min_degree >= 0, f"Simple {s} has negative degree {min_degree}!"
        
        dst_start = min_degree
        dst_end = max_degree + 1
        
        if dst_end <= D:
            simple_burau[s, :, :, dst_start:dst_end] = mat[:, :, src_start:src_end].to(STORAGE_DTYPE_MATRIX)
        else:
            usable_end = D
            src_usable_end = src_start + (usable_end - dst_start)
            simple_burau[s, :, :, dst_start:usable_end] = mat[:, :, src_start:src_usable_end].to(STORAGE_DTYPE_MATRIX)
    
    loaded_valid_suffixes = tables['valid_suffixes']
    loaded_num_valid = tables['num_valid_suffixes']
    
    delta_idx = tables['delta_index']
    id_idx = tables['id_index']
    
    valid_suffixes = loaded_valid_suffixes.clone()
    num_valid_suffixes = loaded_num_valid.clone()
    
    valid_suffixes[id_idx] = valid_suffixes[delta_idx]
    num_valid_suffixes[id_idx] = num_valid_suffixes[delta_idx]
    
    # Compute TRUE Artin lengths from permutation inversions (not from matrices!)
    simple_artin_lengths = get_simple_artin_lengths(n=4)
    
    print(f"Loaded tables from {table_path}")
    print(f"  Degree window: [0, {D-1}] ({D} coefficients) - NON-NEGATIVE ONLY")
    print(f"  TRUE Artin lengths computed from permutation inversions")
    
    return simple_burau, valid_suffixes, num_valid_suffixes, simple_artin_lengths


# Keep BraidSearch as alias for backwards compatibility
BraidSearch = BraidSearchUltra
