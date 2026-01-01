"""
GPU-accelerated reservoir sampling for braids with low projlen.
OPTIMIZED VERSION: Batched FFT matrix multiplication (3x fewer FFT ops)
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import os
import time
import gc

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
    checkpoint_every: int = 9999
    device: str = "cuda"
    expansion_chunk_size: int = 100000
    use_best: int = 0
    matmul_chunk_size: int = 20000  # NEW: chunk size for batched FFT matmul
    
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
# OPTIMIZED POLYNOMIAL OPERATIONS
# =============================================================================

def poly_matmul_batch_optimized(A: torch.Tensor, B: torch.Tensor, p: int, 
                                 chunk_size: int = 20000) -> torch.Tensor:
    """
    Optimized batch 3x3 matrix multiplication over polynomial ring.
    
    Key optimization: Pre-compute FFTs of A and B, do matrix multiply in FFT space,
    then single batched IFFT. Reduces FFT operations from 81 to 27 (3x speedup).
    
    Processes in chunks to manage memory.
    
    Input: int16 or int32 tensors of shape (N, 3, 3, D)
    Output: int32 tensor of shape (N, 3, 3, 2D-1)
    """
    N, _, _, D = A.shape
    out_D = 2 * D - 1
    fft_size = 1 << (out_D).bit_length()  # Next power of 2
    device = A.device
    
    # Output tensor
    C = torch.zeros(N, 3, 3, out_D, dtype=COMPUTE_DTYPE_INT, device=device)
    
    # Process in chunks to manage GPU memory
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_N = end - start
        
        # Get chunks
        A_chunk = A[start:end].float()  # (chunk_N, 3, 3, D)
        B_chunk = B[start:end].float()  # (chunk_N, 3, 3, D)
        
        # Batched FFT: reshape to (chunk_N * 9, D), FFT, reshape back
        A_flat = A_chunk.reshape(chunk_N * 9, D)
        B_flat = B_chunk.reshape(chunk_N * 9, D)
        
        A_fft = torch.fft.rfft(A_flat, n=fft_size, dim=-1)  # (chunk_N * 9, fft_size//2 + 1)
        B_fft = torch.fft.rfft(B_flat, n=fft_size, dim=-1)
        
        # Reshape back to matrix form
        fft_len = A_fft.shape[-1]
        A_fft = A_fft.reshape(chunk_N, 3, 3, fft_len)  # (chunk_N, 3, 3, fft_len)
        B_fft = B_fft.reshape(chunk_N, 3, 3, fft_len)  # (chunk_N, 3, 3, fft_len)
        
        # Matrix multiply in FFT space
        # C_fft[n, i, j] = sum_k A_fft[n, i, k] * B_fft[n, k, j]
        # This is element-wise multiplication along the polynomial dimension!
        C_fft = torch.zeros(chunk_N, 3, 3, fft_len, dtype=torch.complex64, device=device)
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    C_fft[:, i, j, :] += A_fft[:, i, k, :] * B_fft[:, k, j, :]
        
        # Free FFT intermediates
        del A_fft, B_fft, A_flat, B_flat, A_chunk, B_chunk
        
        # Batched IFFT
        C_fft_flat = C_fft.reshape(chunk_N * 9, fft_len)
        C_real = torch.fft.irfft(C_fft_flat, n=fft_size, dim=-1)  # (chunk_N * 9, fft_size)
        del C_fft, C_fft_flat
        
        # Truncate, round, mod p
        C_real = C_real[:, :out_D]  # (chunk_N * 9, out_D)
        C_int = torch.round(C_real).to(COMPUTE_DTYPE_INT) % p
        del C_real
        
        # Reshape and store
        C[start:end] = C_int.reshape(chunk_N, 3, 3, out_D)
        del C_int
    
    return C


def poly_matmul_batch_optimized_v2(A: torch.Tensor, B: torch.Tensor, p: int,
                                    chunk_size: int = 20000) -> torch.Tensor:
    """
    Further optimized version using torch.einsum for the FFT-space matrix multiply.
    
    This avoids the Python loop and lets PyTorch optimize the computation.
    """
    N, _, _, D = A.shape
    out_D = 2 * D - 1
    fft_size = 1 << (out_D).bit_length()
    device = A.device
    
    C = torch.zeros(N, 3, 3, out_D, dtype=COMPUTE_DTYPE_INT, device=device)
    
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_N = end - start
        
        A_chunk = A[start:end].float()
        B_chunk = B[start:end].float()
        
        # Batched FFT
        A_fft = torch.fft.rfft(A_chunk, n=fft_size, dim=-1)  # (chunk_N, 3, 3, fft_len)
        B_fft = torch.fft.rfft(B_chunk, n=fft_size, dim=-1)
        del A_chunk, B_chunk
        
        # Matrix multiply in FFT space using einsum
        # For each frequency bin, do a 3x3 matrix multiply
        # C[n,i,j,f] = sum_k A[n,i,k,f] * B[n,k,j,f]
        C_fft = torch.einsum('nikf,nkjf->nijf', A_fft, B_fft)
        del A_fft, B_fft
        
        # Batched IFFT
        C_real = torch.fft.irfft(C_fft, n=fft_size, dim=-1)
        del C_fft
        
        # Truncate, round, mod p
        C_int = torch.round(C_real[..., :out_D]).to(COMPUTE_DTYPE_INT) % p
        del C_real
        
        C[start:end] = C_int
        del C_int
    
    return C


def poly_matmul_batch_legacy(A: torch.Tensor, B: torch.Tensor, p: int) -> torch.Tensor:
    """
    Original implementation for comparison.
    """
    N, _, _, D = A.shape
    out_D = 2 * D - 1
    fft_size = 1 << (out_D).bit_length()
    device = A.device
    
    if A.dtype != COMPUTE_DTYPE_INT:
        A = A.to(COMPUTE_DTYPE_INT)
    if B.dtype != COMPUTE_DTYPE_INT:
        B = B.to(COMPUTE_DTYPE_INT)
    
    C = torch.zeros(N, 3, 3, out_D, dtype=COMPUTE_DTYPE_INT, device=device)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a_ik = A[:, i, k, :].float()
                b_kj = B[:, k, j, :].float()
                
                a_fft = torch.fft.rfft(a_ik, n=fft_size, dim=-1)
                b_fft = torch.fft.rfft(b_kj, n=fft_size, dim=-1)
                c_fft = a_fft * b_fft
                c = torch.fft.irfft(c_fft, n=fft_size, dim=-1)
                
                conv = torch.round(c[:, :out_D]).to(COMPUTE_DTYPE_INT) % p
                C[:, i, j, :] = (C[:, i, j, :] + conv) % p
    
    return C


def compute_projlen_batch(matrices: torch.Tensor) -> torch.Tensor:
    """
    Compute projective length for a batch of matrices.
    Works with any integer dtype.
    """
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
# BENCHMARK FUNCTION
# =============================================================================

def benchmark_matmul_implementations(N: int = 10000, D: int = 1201, p: int = 7, 
                                      device: str = "cuda", num_warmup: int = 3,
                                      num_trials: int = 5):
    """
    Benchmark different matmul implementations.
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARKING POLYNOMIAL MATRIX MULTIPLY")
    print(f"{'='*60}")
    print(f"N={N} matrices, D={D} coefficients, p={p}")
    print(f"Device: {device}")
    print()
    
    device = torch.device(device)
    
    # Create random test matrices
    A = torch.randint(0, p, (N, 3, 3, D), dtype=torch.int16, device=device)
    B = torch.randint(0, p, (N, 3, 3, D), dtype=torch.int16, device=device)
    
    implementations = [
        ("Legacy (27 FFT pairs)", poly_matmul_batch_legacy, {}),
        ("Optimized v1 (loop)", poly_matmul_batch_optimized, {"chunk_size": N}),
        ("Optimized v2 (einsum)", poly_matmul_batch_optimized_v2, {"chunk_size": N}),
        ("Optimized v2 (chunk=5K)", poly_matmul_batch_optimized_v2, {"chunk_size": 5000}),
        ("Optimized v2 (chunk=10K)", poly_matmul_batch_optimized_v2, {"chunk_size": 10000}),
    ]
    
    results = {}
    
    for name, func, kwargs in implementations:
        print(f"Testing: {name}")
        
        # Warmup
        for _ in range(num_warmup):
            _ = func(A, B, p, **kwargs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        
        # Timed trials
        times = []
        for trial in range(num_trials):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            t0 = time.time()
            C = func(A, B, p, **kwargs)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.time() - t0
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        results[name] = {"avg": avg_time, "min": min_time, "result": C}
        
        print(f"  Avg: {avg_time*1000:.1f}ms, Min: {min_time*1000:.1f}ms")
        
        # Clear cache
        del C
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Verify correctness
    print(f"\nVerifying correctness...")
    ref_name = "Legacy (27 FFT pairs)"
    ref_result = results[ref_name]["result"]
    
    for name, data in results.items():
        if name == ref_name:
            continue
        diff = (data["result"] != ref_result).sum().item()
        status = "✓ MATCH" if diff == 0 else f"✗ MISMATCH ({diff} elements)"
        print(f"  {name}: {status}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SPEEDUP SUMMARY")
    print(f"{'='*60}")
    baseline = results[ref_name]["min"]
    for name, data in results.items():
        speedup = baseline / data["min"]
        print(f"  {name}: {speedup:.2f}x")
    
    return results


# =============================================================================
# VECTORIZED SUFFIX EXPANSION (unchanged)
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
# GPU BUCKETS (unchanged)
# =============================================================================

class GPUBuckets:
    """
    Maintains reservoir-sampled buckets entirely on GPU.
    """
    
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
        """Add a chunk of candidates with reservoir sampling."""
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

class BraidSearch:
    """
    GPU-accelerated search for braids with low projlen.
    OPTIMIZED: Uses batched FFT for 3x faster matrix multiplication.
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
        self.buckets: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.kernel_braids: list[torch.Tensor] = []
        self.stats = {
            "candidates_per_level": [], 
            "buckets_per_level": [],
            "time_per_level": [],
            "time_matmul": [],
            "time_sampling": [],
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
        
        self.buckets[1] = (identity_matrix, identity_word, identity_length)
        
        print(f"Initialized with identity braid")
        print(f"Degree window: [-{self.D//2}, {self.D//2}] ({self.D} coefficients)")
        print(f"Storage types: matrix={STORAGE_DTYPE_MATRIX}, word={STORAGE_DTYPE_WORD}")
        print(f"Config: bucket_size={self.config.bucket_size}, "
              f"bootstrap={self.config.bootstrap_length}, "
              f"max_length={self.config.max_length}, "
              f"chunk_size={self.config.expansion_chunk_size}, "
              f"matmul_chunk={self.config.matmul_chunk_size}, "
              f"use_best={self.config.use_best if self.config.use_best > 0 else 'all'}")
    
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
        
        if needs_matrix_resize or needs_word_resize:
            print(f"  Resizing tensors:")
            if needs_matrix_resize:
                print(f"    Matrices: D={saved_D} → D={new_D}")
            if needs_word_resize:
                print(f"    Words: length={saved_max_length} → length={new_max_length}")
        
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
                    src_start = offset
                    src_end = offset + new_D
                    mat = mat[:, :, :, src_start:src_end].clone()
            
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
        
        total_bytes = 0
        for mat, words, lengths in self.buckets.values():
            total_bytes += mat.numel() * mat.element_size()
            total_bytes += words.numel() * words.element_size()
            total_bytes += lengths.numel() * lengths.element_size()
        
        print(f"  Loaded level {saved_level}")
        print(f"  Buckets: {len(self.buckets)} projlen values, {total_braids} total braids")
        print(f"  Bucket memory: {total_bytes / 1e9:.2f} GB")
        print(f"  Kernel elements found so far: {len(self.kernel_braids)}")
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
                pl: (mat.cpu(), words.cpu(), lengths.cpu())
                for pl, (mat, words, lengths) in self.buckets.items()
            }
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")
    
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
        suffix_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand a chunk: gather parent matrices, multiply by suffix matrices."""
        num_candidates = len(braid_indices)
        
        parent_matrices = matrices[braid_indices]
        parent_words = words[braid_indices]
        parent_lengths = lengths[braid_indices]
        
        suffix_matrices = self.simple_burau[suffix_indices]
        
        # USE OPTIMIZED MATRIX MULTIPLY
        new_matrices = poly_matmul_batch_optimized_v2(
            parent_matrices, suffix_matrices, self.config.prime,
            chunk_size=self.config.matmul_chunk_size
        )
        
        new_words = parent_words.clone()
        batch_idx = torch.arange(num_candidates, device=self.device)
        new_words[batch_idx, parent_lengths.long()] = suffix_indices.to(STORAGE_DTYPE_WORD)
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
        
        left_loss = matrices[..., :trim_left].abs().sum().item()
        right_loss = matrices[..., trim_right:].abs().sum().item()
        if left_loss > 0 or right_loss > 0:
            print(f"  WARNING: Trimming nonzero coefficients! left={left_loss}, right={right_loss}")
        
        return matrices[..., trim_left:trim_right]
    
    def process_level(self, level: int):
        """Process one level with incremental GPU reservoir sampling."""
        level_start = time.time()
        
        is_bootstrap = (level <= self.config.bootstrap_length)
        mode = "BOOTSTRAP" if is_bootstrap else "SAMPLING"
        
        print(f"\n{'='*60}")
        print(f"Level {level} - {mode}")
        print(f"{'='*60}")
        
        use_best_limit = 0 if is_bootstrap else self.config.use_best
        matrices, words, lengths, last_simples = self.gather_level_braids(use_best=use_best_limit)
        num_starting = len(matrices)
        print(f"  Starting braids: {num_starting}")
        
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
                chunk_braid_idx, chunk_suffix_idx
            )
            chunk_matrices = self.recenter_matrices(chunk_matrices)
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
        if len(projlen_counts) > 10:
            print(f"    ... and {len(projlen_counts) - 10} more projlen values")
        
        self.buckets = gpu_buckets.get_buckets()
        
        total_kept = sum(m.shape[0] for m, _, _ in self.buckets.values())
        
        total_bytes = 0
        for mat, wrds, lens in self.buckets.values():
            total_bytes += mat.numel() * mat.element_size()
            total_bytes += wrds.numel() * wrds.element_size()
            total_bytes += lens.numel() * lens.element_size()
        
        print(f"  Braids kept: {total_kept} (in {len(self.buckets)} buckets, {total_bytes/1e9:.2f} GB)")
        
        level_time = time.time() - level_start
        print(f"  Timing: matmul={t_matmul_total:.2f}s, sampling={t_sample_total:.2f}s, total={level_time:.2f}s")
        
        self.stats["candidates_per_level"].append(num_candidates)
        self.stats["buckets_per_level"].append(len(self.buckets))
        self.stats["time_per_level"].append(level_time)
        self.stats["time_matmul"].append(t_matmul_total)
        self.stats["time_sampling"].append(t_sample_total)
        
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
                print(f"Total sampling time: {sum(self.stats['time_sampling']):.2f}s")
            print(f"Total kernel elements (projlen=1) found: {sum(len(w) for w in self.kernel_braids)}")
            
            save_dir = checkpoint_dir if checkpoint_dir else "."
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            final_path = f"{save_dir}/final_state_level_{final_level}.pt"
            self.save_checkpoint(final_level, final_path)
            print(f"\nFinal state saved to {final_path}")
            print(f"To resume: use --resume-from {final_path}")
        
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
        
        if dst_start < 0 or dst_end > D:
            raise ValueError(
                f"Simple {s} degrees [{min_degree}, {max_degree}] don't fit in window {D}"
            )
        
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
    print(f"  Re-centered: degree_window={D}")
    print(f"  Storage dtype: {STORAGE_DTYPE_MATRIX}")
    print(f"  Identity suffixes fixed: {num_valid_suffixes[id_idx]} valid")
    
    return simple_burau, valid_suffixes, num_valid_suffixes


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run benchmark by default
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
        D = int(sys.argv[3]) if len(sys.argv) > 3 else 1201
        benchmark_matmul_implementations(N=N, D=D, device=device)
    else:
        print("Usage: python braid_search_optimized.py --benchmark [N] [D]")
        print("       or import and use the BraidSearch class")