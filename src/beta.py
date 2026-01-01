"""
GPU-accelerated reservoir sampling for braids with low projlen.
OPTIMIZED: Tiled FFT Matrix Multiplication + Pre-computed Suffixes
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
    
    @property
    def degree_window(self) -> int:
        return 2 * self.degree_multiplier * self.max_length + 1
    
    @property
    def degree_offset(self) -> int:
        return self.degree_multiplier * self.max_length


# =============================================================================
# DTYPE CONFIGURATION
# =============================================================================

STORAGE_DTYPE_MATRIX = torch.int16   # Values 0-6
STORAGE_DTYPE_WORD = torch.int32     # Indices 0-23
STORAGE_DTYPE_LENGTH = torch.int32   # Lengths 0-600

# Compute types
COMPUTE_DTYPE_INT = torch.int32      # For intermediate accumulation

# =============================================================================
# OPTIMIZED FFT POLYNOMIAL OPERATIONS
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
    
    # Pre-allocate output
    C_final = torch.empty((N, 3, 3, out_D), dtype=torch.int32, device=device)
    
    # FFT size must match the pre-computed table
    fft_size = suffix_fft_table.shape[-1]
    
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
        
        c_fft = torch.matmul(a_fft, b_fft)
        
        # 5. Inverse FFT
        c_fft = c_fft.permute(0, 2, 3, 1) # Back to (sub_batch, 3, 3, F)
        c_chunk = torch.fft.irfft(c_fft, n=fft_size, dim=-1)
        
        # 6. Post-process
        c_chunk = c_chunk[..., :out_D]
        c_chunk = torch.round(c_chunk).to(torch.int32) % p
        
        C_final[start:end] = c_chunk
        
        # Cleanup to ensure memory is freed
        del a_chunk, a_fft, b_fft, c_fft, c_chunk

    return C_final


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
    """Build (braid_index, suffix_index) pairs."""
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
# GPU RESERVOIR SAMPLING
# =============================================================================

class GPUBuckets:
    def __init__(self, bucket_size: int, device: torch.device):
        self.bucket_size = bucket_size
        self.device = device
        self.data: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    
    def add_chunk(self, matrices, words, lengths, projlens, is_bootstrap):
        if len(matrices) == 0: return
        
        matrices = matrices.to(STORAGE_DTYPE_MATRIX)
        words = words.to(STORAGE_DTYPE_WORD)
        lengths = lengths.to(STORAGE_DTYPE_LENGTH)
        
        priorities = torch.rand(len(matrices), device=self.device)
        unique_pls = torch.unique(projlens)
        
        for pl in unique_pls.tolist():
            mask = (projlens == pl)
            new_data = (matrices[mask], words[mask], lengths[mask], priorities[mask])
            
            if pl not in self.data:
                if is_bootstrap or len(new_data[0]) <= self.bucket_size:
                    self.data[pl] = new_data
                else:
                    _, idx = torch.topk(new_data[3], self.bucket_size, largest=False)
                    self.data[pl] = tuple(x[idx] for x in new_data)
            else:
                old_data = self.data[pl]
                merged = tuple(torch.cat([o, n], dim=0) for o, n in zip(old_data, new_data))
                
                if is_bootstrap or len(merged[0]) <= self.bucket_size:
                    self.data[pl] = merged
                else:
                    _, idx = torch.topk(merged[3], self.bucket_size, largest=False)
                    self.data[pl] = tuple(x[idx] for x in merged)

    def get_buckets(self):
        return {pl: (m, w, l) for pl, (m, w, l, _) in self.data.items()}


# =============================================================================
# MAIN ALGORITHM
# =============================================================================

class BraidSearch:
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
        self.buckets = {}
        self.kernel_braids = []
        self.stats = {"candidates_per_level": [], "buckets_per_level": [], "time_per_level": [], "time_matmul": [], "time_sampling": []}
        self.start_level = 1
        
        # --- PRE-COMPUTE SUFFIX FFTS ---
        self._precompute_suffix_fft()
        
    def _precompute_suffix_fft(self):
        """Pre-compute FFTs for all simple generators to speed up matmul."""
        print("Pre-computing suffix FFTs table...")
        
        # 1. Determine sizes
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
        print(f"  FFT Size: {fft_size}")
        print(f"  Table Memory: {self.suffix_fft_table.numel() * 8 / 1e6:.1f} MB")

    def initialize(self):
        identity_matrix = torch.zeros(1, 3, 3, self.D, dtype=STORAGE_DTYPE_MATRIX, device=self.device)
        center = self.D // 2
        for i in range(3): identity_matrix[0, i, i, center] = 1
        
        self.buckets[1] = (
            identity_matrix,
            torch.zeros(1, self.config.max_length, dtype=STORAGE_DTYPE_WORD, device=self.device),
            torch.zeros(1, dtype=STORAGE_DTYPE_LENGTH, device=self.device)
        )
        print(f"Initialized with identity braid")

    def load_checkpoint(self, checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.start_level = checkpoint['level'] + 1
        
        # Restore buckets (omitted logic for resizing for brevity, assumed matching config)
        self.buckets = {}
        for pl, (mat, words, lengths) in checkpoint['buckets'].items():
             self.buckets[int(pl)] = (
                 mat.to(STORAGE_DTYPE_MATRIX).to(self.device),
                 words.to(STORAGE_DTYPE_WORD).to(self.device),
                 lengths.to(STORAGE_DTYPE_LENGTH).to(self.device)
             )
        
        if 'kernel_braids' in checkpoint:
            self.kernel_braids = [w.to(self.device) for w in checkpoint['kernel_braids']]
            
        print(f"  Resuming from level {self.start_level}")
        return self.start_level

    def save_checkpoint(self, level, path):
        print(f"  Saving checkpoint to {path}...")
        checkpoint = {
            "level": level,
            "config": self.config.__dict__,
            "stats": self.stats,
            "kernel_braids": [w.cpu() for w in self.kernel_braids],
            "buckets": {pl: (m.cpu(), w.cpu(), l.cpu()) for pl, (m, w, l) in self.buckets.items()}
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def gather_level_braids(self, use_best=0):
        if not self.buckets: raise RuntimeError("No braids!")
        sorted_projlens = sorted(self.buckets.keys())
        
        all_mat, all_words, all_len = [], [], []
        total = 0
        
        for pl in sorted_projlens:
            mat, words, lengths = self.buckets[pl]
            count = len(mat)
            
            if use_best > 0:
                rem = use_best - total
                if rem <= 0: break
                if count > rem:
                    idx = torch.randperm(count, device=self.device)[:rem]
                    mat, words, lengths = mat[idx], words[idx], lengths[idx]
                    count = rem
            
            all_mat.append(mat)
            all_words.append(words)
            all_len.append(lengths)
            total += count
            
        matrices = torch.cat(all_mat, dim=0).to(COMPUTE_DTYPE_INT)
        words = torch.cat(all_words, dim=0)
        lengths = torch.cat(all_len, dim=0)
        
        batch_idx = torch.arange(len(lengths), device=self.device)
        last_pos = torch.clamp(lengths - 1, min=0).long()
        last_simples = words[batch_idx, last_pos].long()
        last_simples = torch.where(lengths > 0, last_simples, torch.zeros_like(last_simples))
        
        return matrices, words, lengths, last_simples

    def expand_and_multiply_chunk(self, matrices, words, lengths, braid_indices, suffix_indices):
        """Modified to use pre-computed FFTs."""
        parent_matrices = matrices[braid_indices]
        parent_words = words[braid_indices]
        parent_lengths = lengths[braid_indices]
        
        # Use Optimized FFT Mul with pre-computed suffixes
        # Note: We pass the INDICES of the suffixes, not the matrices
        new_matrices = poly_matmul_fft_precomputed(
            parent_matrices,
            suffix_indices,
            self.suffix_fft_table,
            self.config.prime,
            self.out_D,
            sub_batch_size=4096 # Adjust this based on GPU VRAM
        )
        
        new_words = parent_words.clone()
        batch_idx = torch.arange(len(new_matrices), device=self.device)
        new_words[batch_idx, parent_lengths.long()] = suffix_indices.to(STORAGE_DTYPE_WORD)
        new_lengths = parent_lengths + 1
        
        return new_matrices, new_words, new_lengths

    def recenter_matrices(self, matrices):
        current_D = matrices.shape[-1]
        target_D = self.D
        
        if current_D <= target_D:
            pad = target_D - current_D
            return F.pad(matrices, (pad//2, pad - pad//2), value=0)
        
        trim = current_D - target_D
        return matrices[..., trim//2 : current_D - (trim - trim//2)]

    def process_level(self, level):
        start_time = time.time()
        is_bootstrap = (level <= self.config.bootstrap_length)
        print(f"\n{'='*60}\nLevel {level} - {'BOOTSTRAP' if is_bootstrap else 'SAMPLING'}\n{'='*60}")
        
        matrices, words, lengths, last_simples = self.gather_level_braids(0 if is_bootstrap else self.config.use_best)
        print(f"  Starting braids: {len(matrices)}")
        
        braid_indices, suffix_indices = build_expansion_indices_vectorized(
            last_simples, self.num_valid_suffixes, self.valid_suffixes
        )
        num_candidates = len(braid_indices)
        print(f"  Candidates: {num_candidates}")
        
        if num_candidates == 0: return False
        
        chunk_size = self.config.expansion_chunk_size
        gpu_buckets = GPUBuckets(self.config.bucket_size, self.device)
        
        t_matmul, t_sample = 0.0, 0.0
        
        for i in range(0, num_candidates, chunk_size):
            gc.collect(); torch.cuda.empty_cache()
            
            end = min(i + chunk_size, num_candidates)
            b_idx = braid_indices[i:end]
            s_idx = suffix_indices[i:end]
            
            t0 = time.time()
            chunk_mat, chunk_words, chunk_lens = self.expand_and_multiply_chunk(
                matrices, words, lengths, b_idx, s_idx
            )
            chunk_mat = self.recenter_matrices(chunk_mat)
            t_matmul += time.time() - t0
            
            chunk_pl = compute_projlen_batch(chunk_mat)
            
            if (chunk_pl == 1).any():
                hits = chunk_words[chunk_pl == 1]
                print(f"  🎉 FOUND {len(hits)} KERNEL ELEMENTS! 🎉")
                self.kernel_braids.extend([h.cpu() for h in hits])
            
            t0 = time.time()
            gpu_buckets.add_chunk(chunk_mat, chunk_words, chunk_lens, chunk_pl, is_bootstrap)
            t_sample += time.time() - t0
            
            del chunk_mat, chunk_words, chunk_lens, chunk_pl
            
        self.buckets = gpu_buckets.get_buckets()
        print(f"  Kept: {sum(len(m) for m,_,_ in self.buckets.values())}")
        print(f"  Timing: Matmul={t_matmul:.2f}s, Sample={t_sample:.2f}s, Total={time.time()-start_time:.2f}s")
        return True

    def run(self, checkpoint_dir=None, resume_from=None):
        if resume_from: self.load_checkpoint(resume_from)
        else: self.initialize()
        
        for level in range(self.start_level, self.config.max_length + 1):
            if not self.process_level(level): break
            if checkpoint_dir and level % self.config.checkpoint_every == 0:
                self.save_checkpoint(level, f"{checkpoint_dir}/checkpoint_level_{level}.pt")
                
        return self.kernel_braids

# =============================================================================
# LOADING & ENTRY
# =============================================================================

def load_tables_from_file(config, table_path):
    tables = torch.load(table_path, weights_only=True)
    
    # Load and resize simple_burau to match window
    loaded_burau = tables['simple_burau']
    D = config.degree_window
    center_new = D // 2
    center_old = tables['center']
    
    simple_burau = torch.zeros(24, 3, 3, D, dtype=STORAGE_DTYPE_MATRIX)
    
    for s in range(24):
        mat = loaded_burau[s]
        nz = torch.where(mat.abs().sum((0,1)) > 0)[0]
        if len(nz) == 0: continue
        
        start, end = nz[0].item(), nz[-1].item() + 1
        min_deg, max_deg = start - center_old, end - 1 - center_old
        dst_start, dst_end = center_new + min_deg, center_new + max_deg + 1
        
        simple_burau[s, :, :, dst_start:dst_end] = mat[:, :, start:end].to(STORAGE_DTYPE_MATRIX)
        
    valid_suffixes = tables['valid_suffixes'].clone()
    num_valid = tables['num_valid_suffixes'].clone()
    
    # Fix identity
    id_idx = tables['id_index']
    delta_idx = tables['delta_index']
    valid_suffixes[id_idx] = valid_suffixes[delta_idx]
    num_valid[id_idx] = num_valid[delta_idx]
    
    return simple_burau, valid_suffixes, num_valid

if __name__ == "__main__":
    # Example usage
    cfg = Config(bucket_size=50000, max_length=50, bootstrap_length=5, prime=5, device="cuda")
    # Add table loading and search.run() call here as in original...