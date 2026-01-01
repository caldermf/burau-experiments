"""
GPU-accelerated reservoir sampling for braids with low projlen.
OPTIMIZED: Tiled FFT Matrix Multiplication + Pre-computed Suffixes
FIXED: FFT Size Calculation
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
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
    
    @property
    def degree_window(self) -> int:
        return 2 * self.degree_multiplier * self.max_length + 1
    
    @property
    def degree_offset(self) -> int:
        return self.degree_multiplier * self.max_length


# =============================================================================
# DTYPE CONFIGURATION
# =============================================================================

STORAGE_DTYPE_MATRIX = torch.int16   # Values 0-6, fits in int16
STORAGE_DTYPE_WORD = torch.int32     # Indices 0-23
STORAGE_DTYPE_LENGTH = torch.int32   # Lengths 0-600

# Compute types (for arithmetic operations)
COMPUTE_DTYPE_INT = torch.int32      # For matmul accumulation before mod p

# =============================================================================
# CORE POLYNOMIAL OPERATIONS (OPTIMIZED)
# =============================================================================

def poly_matmul_fft_precomputed(
    A: torch.Tensor, 
    suffix_indices: torch.Tensor, 
    suffix_fft_table: torch.Tensor, 
    p: int, 
    out_D: int,
    sub_batch_size: int = 4096
) -> torch.Tensor:
    """
    Computes A @ Suffixes using Tiled FFT with pre-computed suffix FFTs.
    
    Args:
        A: Input batch of matrices (N, 3, 3, D_in)
        suffix_indices: Indices into the suffix table for each A (N)
        suffix_fft_table: Pre-computed FFTs of all valid suffixes (NumSuffixes, 3, 3, F)
        p: Prime modulus
        out_D: Desired output degree (usually 2*D_in - 1)
        sub_batch_size: Tile size to prevent OOM
    """
    N, _, _, D = A.shape
    device = A.device
    
    # Pre-allocate output to avoid memory spikes from concatenation
    # We use int32 for the stored result to save space
    C_final = torch.empty((N, 3, 3, out_D), dtype=torch.int32, device=device)
    
    # FIX: Correctly reconstruct the original FFT size (N) from the frequency dimension (N/2 + 1)
    stored_freq_dim = suffix_fft_table.shape[-1]
    fft_size = (stored_freq_dim - 1) * 2
    
    # Process in safe tiles
    for start in range(0, N, sub_batch_size):
        end = min(start + sub_batch_size, N)
        
        # 1. Prepare Inputs
        a_chunk = A[start:end].float()
        indices_chunk = suffix_indices[start:end]
        
        # 2. FFT of A (Calculated on the fly)
        # Shape: (sub_batch, 3, 3, F)
        a_fft = torch.fft.rfft(a_chunk, n=fft_size, dim=-1)
        
        # 3. Retrieve Pre-computed FFT of Suffixes
        # Shape: (sub_batch, 3, 3, F)
        b_fft = suffix_fft_table[indices_chunk]
        
        # 4. Matrix Multiplication in Frequency Domain
        # Permute to (sub_batch, F, 3, 3) for efficient batch matmul
        a_fft = a_fft.permute(0, 3, 1, 2)
        b_fft = b_fft.permute(0, 3, 1, 2)
        
        # Batch MatMul: (B, F, 3, 3) @ (B, F, 3, 3) -> (B, F, 3, 3)
        c_fft = torch.matmul(a_fft, b_fft)
        
        # 5. Inverse FFT
        c_fft = c_fft.permute(0, 2, 3, 1) # Back to (sub_batch, 3, 3, F)
        c_chunk = torch.fft.irfft(c_fft, n=fft_size, dim=-1)
        
        # 6. Post-process
        # Trim padding
        c_chunk = c_chunk[..., :out_D]
        # Round, mod p, cast to int32
        c_chunk = torch.round(c_chunk).to(torch.int32) % p
        
        # Store
        C_final[start:end] = c_chunk
        
        # Explicitly delete intermediates to ensure memory is freed before next loop
        del a_chunk, a_fft, b_fft, c_fft, c_chunk

    return C_final


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
# VECTORIZED SUFFIX EXPANSION
# =============================================================================

def build_expansion_indices_vectorized(
    last_simples: torch.Tensor,
    num_valid_suffixes: torch.Tensor,
    valid_suffixes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
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
# INCREMENTAL GPU RESERVOIR SAMPLING
# =============================================================================

class GPUBuckets:
    """
    Maintains reservoir-sampled buckets entirely on GPU.
    Memory-optimized: stores matrices as int16, words as int32.
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
        if len(matrices) == 0:
            return
        
        # Convert to storage types for memory efficiency
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
    v4.0: Optimized with Tiled FFT Matrix Multiplication and Pre-computed Suffixes.
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
        
        # Simple Burau matrices stored as int16
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
        
        # --- PRE-COMPUTE SUFFIX FFTS ---
        self._precompute_suffix_fft()
    
    def _precompute_suffix_fft(self):
        """Pre-compute FFTs for all simple generators to speed up matmul."""
        print("Pre-computing suffix FFTs table...")
        
        # 1. Determine sizes
        # Output will be size 2*D - 1. FFT needs next power of 2.
        out_D = 2 * self.D - 1
        fft_size = 1 << (out_D).bit_length()
        
        self.fft_size = fft_size
        self.out_D = out_D
        
        # 2. Convert simple_burau to float and compute FFT
        # Shape: (NumGenerators, 3, 3, F)
        # Note: We compute ALL generators in the table, so we can index directly
        self.suffix_fft_table = torch.fft.rfft(
            self.simple_burau.float(), 
            n=fft_size, 
            dim=-1
        )
        print(f"  FFT Size: {fft_size} (freq dim: {self.suffix_fft_table.shape[-1]})")
        print(f"  Table Memory: {self.suffix_fft_table.numel() * 8 / 1e6:.1f} MB")
    
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
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load state from checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.start_level = checkpoint['level'] + 1
        self.stats = checkpoint.get('stats', self.stats)
        
        if 'kernel_braids' in checkpoint:
            self.kernel_braids = []
            for w in checkpoint['kernel_braids']:
                if isinstance(w, list):
                    self.kernel_braids.append(torch.tensor(w).to(self.device))
                else:
                    self.kernel_braids.append(w.to(self.device))
        
        # Assume compatibility for optimized run
        self.buckets = {}
        for pl, bucket_data in checkpoint['buckets'].items():
            pl = int(pl)
            if isinstance(bucket_data, dict):
                mat, words, lengths = bucket_data['matrices'], bucket_data['words'], bucket_data['lengths']
            else:
                mat, words, lengths = bucket_data
            
            self.buckets[pl] = (
                mat.to(STORAGE_DTYPE_MATRIX).to(self.device),
                words.to(STORAGE_DTYPE_WORD).to(self.device),
                lengths.to(STORAGE_DTYPE_LENGTH).to(self.device)
            )
        
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
        print(f"  Checkpoint saved.")
    
    def gather_level_braids(self, use_best: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather braids from current buckets."""
        if not self.buckets:
            raise RuntimeError("No braids to process!")
        
        sorted_projlens = sorted(self.buckets.keys())
        
        all_matrices, all_words, all_lengths = [], [], []
        total_selected = 0
        
        for projlen in sorted_projlens:
            matrices, words, lengths = self.buckets[projlen]
            bucket_count = len(matrices)
            
            if use_best > 0:
                remaining = use_best - total_selected
                if remaining <= 0: break
                
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
        """Expand a chunk using OPTIMIZED FFT multiplication."""
        
        parent_matrices = matrices[braid_indices]
        parent_words = words[braid_indices]
        parent_lengths = lengths[braid_indices]
        
        # Call the optimized tiled FFT matmul
        # Note: We pass the pre-computed FFT table and the suffix INDICES
        new_matrices = poly_matmul_fft_precomputed(
            parent_matrices,
            suffix_indices,
            self.suffix_fft_table,
            self.config.prime,
            self.out_D,
            sub_batch_size=4096 # Tunable based on GPU VRAM
        )
        
        new_words = parent_words.clone()
        batch_idx = torch.arange(len(new_matrices), device=self.device)
        new_words[batch_idx, parent_lengths.long()] = suffix_indices.to(STORAGE_DTYPE_WORD)
        new_lengths = parent_lengths + 1
        
        return new_matrices, new_words, new_lengths
    
    def recenter_matrices(self, matrices: torch.Tensor) -> torch.Tensor:
        """Trim matrices back to target degree window."""
        current_D = matrices.shape[-1]
        target_D = self.D
        
        if current_D <= target_D:
            pad = target_D - current_D
            return F.pad(matrices, (pad//2, pad - pad//2), value=0)
        
        trim = current_D - target_D
        return matrices[..., trim//2 : current_D - (trim - trim//2)]
    
    def process_level(self, level: int):
        """Process one level."""
        level_start = time.time()
        is_bootstrap = (level <= self.config.bootstrap_length)
        print(f"\n{'='*60}\nLevel {level} - {'BOOTSTRAP' if is_bootstrap else 'SAMPLING'}\n{'='*60}")
        
        matrices, words, lengths, last_simples = self.gather_level_braids(use_best=0 if is_bootstrap else self.config.use_best)
        print(f"  Starting braids: {len(matrices)}")
        
        braid_indices, suffix_indices = build_expansion_indices_vectorized(
            last_simples, self.num_valid_suffixes, self.valid_suffixes
        )
        num_candidates = len(braid_indices)
        print(f"  Candidates: {num_candidates}")
        
        if num_candidates == 0: return False
        
        chunk_size = self.config.expansion_chunk_size
        gpu_buckets = GPUBuckets(self.config.bucket_size, self.device)
        
        t_matmul_total = 0.0
        t_sample_total = 0.0
        projlen_counts = {}
        
        for i in range(0, num_candidates, chunk_size):
            # Explicit garbage collection to prevent memory fragmentation
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            end = min(i + chunk_size, num_candidates)
            chunk_b_idx = braid_indices[i:end]
            chunk_s_idx = suffix_indices[i:end]
            
            t0 = time.time()
            chunk_matrices, chunk_words, chunk_lengths = self.expand_and_multiply_chunk(
                matrices, words, lengths, chunk_b_idx, chunk_s_idx
            )
            chunk_matrices = self.recenter_matrices(chunk_matrices)
            t_matmul_total += time.time() - t0
            
            chunk_projlens = compute_projlen_batch(chunk_matrices)
            
            # Check for kernel elements
            one_mask = (chunk_projlens == 1)
            if one_mask.any():
                hits = chunk_words[one_mask]
                print(f"  🎉 FOUND {len(hits)} KERNEL ELEMENTS! 🎉")
                self.kernel_braids.extend([h.cpu() for h in hits])
            
            # Count distribution
            unique_pls, counts = torch.unique(chunk_projlens, return_counts=True)
            for pl, c in zip(unique_pls.tolist(), counts.tolist()):
                projlen_counts[pl] = projlen_counts.get(pl, 0) + c
            
            t0 = time.time()
            gpu_buckets.add_chunk(chunk_matrices, chunk_words, chunk_lengths, chunk_projlens, is_bootstrap)
            t_sample_total += time.time() - t0
            
            del chunk_matrices, chunk_words, chunk_lengths, chunk_projlens
            
        self.buckets = gpu_buckets.get_buckets()
        
        print(f"  Projlen distribution:")
        for pl in sorted(projlen_counts.keys())[:5]:
            print(f"    projlen={pl}: {projlen_counts[pl]}")
            
        print(f"  Kept: {sum(len(m) for m,_,_ in self.buckets.values())}")
        print(f"  Timing: Matmul={t_matmul_total:.2f}s, Sample={t_sample_total:.2f}s, Total={time.time()-level_start:.2f}s")
        
        return True
    
    def run(self, checkpoint_dir: Optional[str] = None, resume_from: Optional[str] = None):
        """Run the full search."""
        if resume_from:
            self.load_checkpoint(resume_from)
        else:
            self.initialize()
        
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            for level in range(self.start_level, self.config.max_length + 1):
                if not self.process_level(level): break
                
                if checkpoint_dir and (level % self.config.checkpoint_every == 0):
                    self.save_checkpoint(level, f"{checkpoint_dir}/checkpoint_level_{level}.pt")
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted at level {level}!")
            
        return self.kernel_braids


# =============================================================================
# TABLE LOADING UTILITY
# =============================================================================

def load_tables_from_file(config: Config, table_path: str):
    """Load precomputed tables from .pt file."""
    tables = torch.load(table_path, weights_only=True)
    
    loaded_burau = tables['simple_burau']
    loaded_center = tables['center']
    
    D = config.degree_window
    new_center = D // 2
    
    simple_burau = torch.zeros(24, 3, 3, D, dtype=STORAGE_DTYPE_MATRIX)
    
    for s in range(24):
        mat = loaded_burau[s]
        nonzero_mask = mat.abs().sum(dim=(0, 1)) > 0
        if not nonzero_mask.any(): continue
            
        nonzero_indices = torch.where(nonzero_mask)[0]
        src_start = nonzero_indices[0].item()
        src_end = nonzero_indices[-1].item() + 1
        
        min_degree = src_start - loaded_center
        max_degree = src_end - 1 - loaded_center
        
        dst_start = new_center + min_degree
        dst_end = new_center + max_degree + 1
        
        if dst_start < 0 or dst_end > D:
            raise ValueError(f"Degree overflow: {min_degree} to {max_degree} wont fit in window {D}")
        
        simple_burau[s, :, :, dst_start:dst_end] = mat[:, :, src_start:src_end].to(STORAGE_DTYPE_MATRIX)
    
    valid_suffixes = tables['valid_suffixes'].clone()
    num_valid_suffixes = tables['num_valid_suffixes'].clone()
    
    # Propagate identity if needed
    if 'id_index' in tables and 'delta_index' in tables:
        id_idx = tables['id_index']
        delta_idx = tables['delta_index']
        valid_suffixes[id_idx] = valid_suffixes[delta_idx]
        num_valid_suffixes[id_idx] = num_valid_suffixes[delta_idx]
    
    print(f"Loaded tables from {table_path}")
    print(f"  Re-centered: degree_window={D}")
    
    return simple_burau, valid_suffixes, num_valid_suffixes

if __name__ == "__main__":
    pass