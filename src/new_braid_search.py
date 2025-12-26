"""
GPU-accelerated reservoir sampling for braids with low projlen.

This implements the algorithm for finding 4-strand braids whose Burau 
representation (mod p) has low projective length.

PRECOMPUTED DATA YOU NEED TO PROVIDE:
=====================================

1. simple_burau: torch.Tensor of shape (24, 3, 3, D)
   - Burau matrices for each of the 24 simple braids (including identity at index 0)
   - Entry [s, i, j, d] is the coefficient of v^(d + min_degree) in the (i,j) entry
   - The identity (index 0) should have 1s on diagonal at degree 0
   - All coefficients are mod p

2. valid_suffixes: torch.Tensor of shape (24, max_suffixes), dtype=torch.int32
   - valid_suffixes[s, :] lists the simple braid indices that can follow simple s
     while maintaining Garside normal form
   - Padded with -1 for unused slots
   - Example: if simple 5 can be followed by simples 2, 7, 11, then
     valid_suffixes[5] = [2, 7, 11, -1, -1, -1, ...]

3. num_valid_suffixes: torch.Tensor of shape (24,), dtype=torch.int32
   - num_valid_suffixes[s] = number of valid suffixes for simple s
   - Example: num_valid_suffixes[5] = 3 in the above case

The degree window is [-D//2, D//2] where D = 4 * max_length by default.
Index d in the tensor corresponds to degree (d - D//2).
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import json
from pathlib import Path
import os

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
    expansion_chunk_size: int = 50000  # max candidates to process at once in expand_candidates
    
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
    
    Args:
        matrices: (N, 3, 3, D) - all candidate matrices
        words: (N, max_len) - Garside words for each candidate
        word_lengths: (N,) - length of each word
        projlens: (N,) - projlen of each candidate
        bucket_size: max items per bucket (ignored if is_bootstrap)
        is_bootstrap: if True, keep everything (no sampling)
    
    Returns:
        Dictionary mapping projlen -> (matrices, words, word_lengths) for that bucket
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
                # Bucket not full, keep all
                selected = indices
            else:
                # Select bucket_size items with lowest priority
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
    """
    
    def __init__(
        self,
        simple_burau: torch.Tensor,
        valid_suffixes: torch.Tensor,
        num_valid_suffixes: torch.Tensor,
        config: Config
    ):
        """
        Args:
            simple_burau: (24, 3, 3, D) Burau matrices for simple braids
            valid_suffixes: (24, max_suffixes) valid suffix indices, padded with -1
            num_valid_suffixes: (24,) count of valid suffixes per simple
            config: algorithm parameters
        """
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
        
        # Track exciting discoveries
        self.kernel_braids: list[torch.Tensor] = []
        
        # Statistics
        self.stats = {"candidates_per_level": [], "buckets_per_level": []}
    
    def initialize(self):
        """Start with the identity braid in bucket (0, 0)."""
        # Identity matrix: 1s on diagonal at degree = 0, which is index degree_offset
        identity_matrix = torch.zeros(1, 3, 3, self.D, dtype=torch.long, device=self.device)
        center = self.D // 2  # degree 0 is at center of window
        for i in range(3):
            identity_matrix[0, i, i, center] = 1
        
        # Identity word: length 0 (or we can use simple index 0 if that's identity)
        identity_word = torch.zeros(1, self.config.max_length, dtype=torch.long, device=self.device)
        identity_length = torch.zeros(1, dtype=torch.long, device=self.device)
        
        # projlen of identity is 0
        self.buckets[0] = (identity_matrix, identity_word, identity_length)
        
        print(f"Initialized with identity braid in bucket (0, 0)")
        print(f"Degree window: [-{self.D//2}, {self.D//2}] ({self.D} coefficients)")
        print(f"Config: bucket_size={self.config.bucket_size}, "
              f"bootstrap_length={self.config.bootstrap_length}, "
              f"max_length={self.config.max_length}, "
              f"expansion_chunk_size={self.config.expansion_chunk_size}")
    
    def gather_level_braids(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gather all braids from current buckets into flat tensors.
        
        Returns:
            matrices: (total_braids, 3, 3, D)
            words: (total_braids, max_length)
            word_lengths: (total_braids,)
            last_simples: (total_braids,) - last simple in each word (for suffix lookup)
        """
        all_matrices = []
        all_words = []
        all_lengths = []
        
        for projlen, (matrices, words, lengths) in self.buckets.items():
            all_matrices.append(matrices)
            all_words.append(words)
            all_lengths.append(lengths)
        
        if not all_matrices:
            raise RuntimeError("No braids to process!")
        
        matrices = torch.cat(all_matrices, dim=0)
        words = torch.cat(all_words, dim=0)
        lengths = torch.cat(all_lengths, dim=0)
        
        # Extract last simple from each word
        # For length-0 (identity), we need a convention. Use index 0 (identity).
        batch_indices = torch.arange(len(lengths), device=self.device)
        last_positions = torch.clamp(lengths - 1, min=0)
        last_simples = words[batch_indices, last_positions]
        # For identity (length 0), set last_simple to 0
        last_simples = torch.where(lengths > 0, last_simples, torch.zeros_like(last_simples))
        
        return matrices, words, lengths, last_simples
    
    def expand_candidates_chunk(
        self,
        matrices: torch.Tensor,
        words: torch.Tensor,
        lengths: torch.Tensor,
        last_simples: torch.Tensor,
        braid_indices: torch.Tensor,
        suffix_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expand a chunk of (braid, suffix) pairs.
        
        Args:
            matrices, words, lengths, last_simples: full arrays for the level
            braid_indices: which parent braids to expand (indices into matrices)
            suffix_indices: which suffix to append to each
            
        Returns:
            new_matrices: (chunk_size, 3, 3, 2D-1)
            new_words: (chunk_size, max_length)
            new_lengths: (chunk_size,)
        """
        device = self.device
        num_candidates = len(braid_indices)
        
        # Gather matrices for multiplication
        parent_matrices = matrices[braid_indices]          # (chunk_size, 3, 3, D)
        suffix_matrices = self.simple_burau[suffix_indices]  # (chunk_size, 3, 3, D)
        
        # Batch polynomial matrix multiplication
        new_matrices = poly_matmul_batch(parent_matrices, suffix_matrices, self.config.prime)
        
        # Build new words by appending suffix
        parent_words = words[braid_indices]                # (chunk_size, max_length)
        parent_lengths = lengths[braid_indices]            # (chunk_size,)
        
        new_words = parent_words.clone()
        # Set position [length] to the new suffix
        batch_idx = torch.arange(num_candidates, device=device)
        new_words[batch_idx, parent_lengths] = suffix_indices
        new_lengths = parent_lengths + 1
        
        return new_matrices, new_words, new_lengths
    
    def expand_candidates(
        self,
        matrices: torch.Tensor,
        words: torch.Tensor,
        lengths: torch.Tensor,
        last_simples: torch.Tensor,
        new_length: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate all valid (braid, suffix) pairs and compute products.
        Uses chunked processing to avoid OOM on large bucket sizes.
        
        Returns:
            new_matrices: (num_candidates, 3, 3, D_out)
            new_words: (num_candidates, max_length)
            new_lengths: (num_candidates,)
            suffix_indices: (num_candidates,) - which suffix was appended
        """
        device = self.device
        num_braids = len(matrices)
        chunk_size = self.config.expansion_chunk_size
        
        # Build list of (braid_idx, suffix_idx) pairs
        braid_indices_list = []
        suffix_indices_list = []
        
        for i in range(num_braids):
            last_simple = last_simples[i].item()
            n_suffixes = self.num_valid_suffixes[last_simple].item()
            for j in range(n_suffixes):
                suffix = self.valid_suffixes[last_simple, j].item()
                braid_indices_list.append(i)
                suffix_indices_list.append(suffix)
        
        if not braid_indices_list:
            # No valid expansions (shouldn't happen in practice)
            return (
                torch.empty(0, 3, 3, 2*self.D-1, dtype=torch.long, device=device),
                torch.empty(0, self.config.max_length, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device)
            )
        
        all_braid_indices = torch.tensor(braid_indices_list, dtype=torch.long, device=device)
        all_suffix_indices = torch.tensor(suffix_indices_list, dtype=torch.long, device=device)
        num_candidates = len(all_braid_indices)
        
        # Process in chunks to avoid OOM
        if num_candidates <= chunk_size:
            # Small enough to do in one shot
            new_matrices, new_words, new_lengths = self.expand_candidates_chunk(
                matrices, words, lengths, last_simples,
                all_braid_indices, all_suffix_indices
            )
        else:
            # Process in chunks and accumulate
            all_new_matrices = []
            all_new_words = []
            all_new_lengths = []
            
            num_chunks = (num_candidates + chunk_size - 1) // chunk_size
            print(f"    Processing {num_candidates} expansions in {num_chunks} chunks...")
            
            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, num_candidates)
                
                chunk_braid_idx = all_braid_indices[start:end]
                chunk_suffix_idx = all_suffix_indices[start:end]
                
                chunk_matrices, chunk_words, chunk_lengths = self.expand_candidates_chunk(
                    matrices, words, lengths, last_simples,
                    chunk_braid_idx, chunk_suffix_idx
                )
                
                all_new_matrices.append(chunk_matrices)
                all_new_words.append(chunk_words)
                all_new_lengths.append(chunk_lengths)
                
                # Free memory between chunks
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            new_matrices = torch.cat(all_new_matrices, dim=0)
            new_words = torch.cat(all_new_words, dim=0)
            new_lengths = torch.cat(all_new_lengths, dim=0)
        
        return new_matrices, new_words, new_lengths, all_suffix_indices
    
    def recenter_matrices(self, matrices: torch.Tensor) -> torch.Tensor:
        """
        Recenter polynomial matrices to keep coefficients in valid degree window.
        
        After multiplication, degree window expands. We need to either:
        1. Check that all coefficients fit in our target window
        2. Or shift to recenter
        
        For simplicity, we just truncate/check bounds here.
        In practice, you may want to shift based on actual degree range.
        """
        # Current size after multiplication
        current_D = matrices.shape[-1]
        target_D = self.D
        
        if current_D <= target_D:
            # Pad if somehow smaller (shouldn't happen)
            pad_total = target_D - current_D
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return F.pad(matrices, (pad_left, pad_right), value=0)
        
        # Trim symmetrically from both ends
        trim_total = current_D - target_D
        trim_left = trim_total // 2
        trim_right = current_D - trim_left
        
        # Check for coefficient loss (warn if trimming nonzero values)
        left_loss = matrices[..., :trim_left].abs().sum()
        right_loss = matrices[..., trim_right:].abs().sum()
        if left_loss > 0 or right_loss > 0:
            print(f"  WARNING: Trimming nonzero coefficients! Loss: left={left_loss}, right={right_loss}")
            print(f"  Consider increasing degree_multiplier in config.")
        
        return matrices[..., trim_left:trim_right]
    
    def process_level(self, level: int):
        """
        Process one level of the BFS: expand all braids and reservoir sample.
        Uses fully chunked processing to avoid OOM on large expansions.
        """
        is_bootstrap = (level <= self.config.bootstrap_length)
        mode = "BOOTSTRAP (exhaustive)" if is_bootstrap else "SAMPLING"
        
        print(f"\n{'='*60}")
        print(f"Level {level} - {mode}")
        print(f"{'='*60}")
        
        # Gather all braids from previous level
        matrices, words, lengths, last_simples = self.gather_level_braids()
        num_starting = len(matrices)
        print(f"  Starting braids: {num_starting}")
        
        # Build list of all (braid_idx, suffix_idx) pairs
        braid_indices_list = []
        suffix_indices_list = []
        
        for i in range(num_starting):
            last_simple = last_simples[i].item()
            n_suffixes = self.num_valid_suffixes[last_simple].item()
            for j in range(n_suffixes):
                suffix = self.valid_suffixes[last_simple, j].item()
                braid_indices_list.append(i)
                suffix_indices_list.append(suffix)
        
        num_candidates = len(braid_indices_list)
        print(f"  Candidates to generate: {num_candidates}")
        
        if num_candidates == 0:
            print("  No candidates! Algorithm terminates.")
            return False
        
        all_braid_indices = torch.tensor(braid_indices_list, dtype=torch.long, device=self.device)
        all_suffix_indices = torch.tensor(suffix_indices_list, dtype=torch.long, device=self.device)
        
        chunk_size = self.config.expansion_chunk_size
        num_chunks = (num_candidates + chunk_size - 1) // chunk_size
        
        if num_chunks > 1:
            print(f"    Processing in {num_chunks} chunks...")
        
        # For collecting results across chunks
        # We'll do reservoir sampling incrementally per chunk
        # to avoid ever materializing all candidates at once
        
        # Track projlen distribution
        projlen_counts = {}
        total_candidates = 0
        
        # For reservoir sampling, we need to process all chunks and sample
        # We'll accumulate into buckets, doing sampling as we go
        accumulated_buckets: dict[int, tuple[list, list, list, list]] = {}
        # Each bucket: (matrices_list, words_list, lengths_list, priorities_list)
        
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, num_candidates)
            
            chunk_braid_idx = all_braid_indices[start:end]
            chunk_suffix_idx = all_suffix_indices[start:end]
            
            # Expand this chunk
            chunk_matrices, chunk_words, chunk_lengths = self.expand_candidates_chunk(
                matrices, words, lengths, last_simples,
                chunk_braid_idx, chunk_suffix_idx
            )
            
            # Recenter
            chunk_matrices = self.recenter_matrices(chunk_matrices)
            
            # Compute projlen
            chunk_projlens = compute_projlen_batch(chunk_matrices)
            
            # Check for kernel elements (projlen = 1)
            one_mask = (chunk_projlens == 1)
            num_ones = one_mask.sum().item()
            if num_ones > 0:
                print(f"\n  🎉 FOUND {num_ones} BRAIDS WITH PROJLEN = 1 (KERNEL ELEMENTS)! 🎉")
                self.kernel_braids.append(chunk_words[one_mask].cpu())
            
            # Update projlen distribution
            unique_pls, counts = torch.unique(chunk_projlens, return_counts=True)
            for pl, count in zip(unique_pls.tolist(), counts.tolist()):
                projlen_counts[pl] = projlen_counts.get(pl, 0) + count
            
            total_candidates += len(chunk_matrices)
            
            # Generate priorities for this chunk (for reservoir sampling)
            chunk_priorities = torch.rand(len(chunk_matrices), device=self.device)
            
            # Add to accumulated buckets
            for pl in unique_pls.tolist():
                mask = (chunk_projlens == pl)
                if pl not in accumulated_buckets:
                    accumulated_buckets[pl] = ([], [], [], [])
                
                accumulated_buckets[pl][0].append(chunk_matrices[mask].cpu())
                accumulated_buckets[pl][1].append(chunk_words[mask].cpu())
                accumulated_buckets[pl][2].append(chunk_lengths[mask].cpu())
                accumulated_buckets[pl][3].append(chunk_priorities[mask].cpu())
            
            # Free GPU memory
            del chunk_matrices, chunk_words, chunk_lengths, chunk_projlens, chunk_priorities
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        print(f"  Candidates generated: {total_candidates}")
        
        # Report projlen distribution
        print(f"  Projlen distribution:")
        for pl in sorted(projlen_counts.keys()):
            print(f"    projlen={pl}: {projlen_counts[pl]} braids")
        
        # Now do final reservoir sampling from accumulated buckets
        self.buckets = {}
        
        for pl, (mat_list, word_list, len_list, pri_list) in accumulated_buckets.items():
            # Concatenate all chunks for this projlen
            all_mat = torch.cat(mat_list, dim=0)
            all_words = torch.cat(word_list, dim=0)
            all_lengths = torch.cat(len_list, dim=0)
            all_priorities = torch.cat(pri_list, dim=0)
            
            n_items = len(all_mat)
            
            if is_bootstrap or n_items <= self.config.bucket_size:
                # Keep all
                selected_indices = torch.arange(n_items)
            else:
                # Reservoir sample: keep bucket_size items with lowest priority
                _, topk_indices = torch.topk(all_priorities, self.config.bucket_size, largest=False)
                selected_indices = topk_indices
            
            # Move selected items back to GPU
            self.buckets[pl] = (
                all_mat[selected_indices].to(self.device),
                all_words[selected_indices].to(self.device),
                all_lengths[selected_indices].to(self.device)
            )
            
            # Free CPU memory
            del all_mat, all_words, all_lengths, all_priorities
        
        # Report bucket sizes
        total_kept = sum(m.shape[0] for m, _, _ in self.buckets.values())
        print(f"  Braids kept: {total_kept} (in {len(self.buckets)} buckets)")
        
        # Update stats
        self.stats["candidates_per_level"].append(total_candidates)
        self.stats["buckets_per_level"].append(len(self.buckets))
        
        return True
    
    def save_checkpoint(self, level: int, path: str):
        """Save current state to disk."""
        checkpoint = {
            "level": level,
            "config": self.config.__dict__,
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
        """
        Run the full search algorithm.
        """
        self.initialize()
        
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(exist_ok=True)
        
        for level in range(1, self.config.max_length + 1):
            success = self.process_level(level)
            
            if not success:
                break
            
            # Checkpoint periodically
            if checkpoint_dir and (level % self.config.checkpoint_every == 0):
                self.save_checkpoint(level, f"{checkpoint_dir}/checkpoint_level_{level}.json")
        
        print(f"\n{'='*60}")
        print("SEARCH COMPLETE")
        print(f"{'='*60}")
        print(f"Total kernel elements (projlen=1) found: {sum(len(w) for w in self.kernel_braids)}")
        
        return self.kernel_braids


# =============================================================================
# EXAMPLE: HOW TO CREATE THE PRECOMPUTED TABLES
# =============================================================================

"""
EXACT REPLACEMENT CODE
======================

Find this function in your GPU braid search code:

    def create_example_tables(config: Config):
        ...

And replace the ENTIRE function with the code below.
"""

def load_tables_from_file(config: Config, table_path: str = "precomputed_tables/tables_B4_r1_p3.pt"):
    """
    Load precomputed tables from the .pt file generated by peyl.
    
    IMPORTANT: This replaces the create_example_tables() function.
    
    Args:
        config: The Config object with degree_window, prime, etc.
        table_path: Path to the .pt file generated by generate_tables.py
    
    Returns:
        simple_burau: (24, 3, 3, D) tensor of Burau matrices
        valid_suffixes: (24, max_suffixes) tensor of valid suffix indices
        num_valid_suffixes: (24,) tensor of suffix counts
    """
    import torch
    
    # Load the precomputed tables
    tables = torch.load(table_path, weights_only=True)
    
    # Verify parameters match
    assert tables['n'] == 4, f"Expected n=4, got {tables['n']}"
    assert tables['p'] == config.prime, f"Table prime {tables['p']} != config prime {config.prime}"
    
    # The loaded tables use degree_window=64 with center=32
    # We need to re-center them to match config.degree_window
    loaded_burau = tables['simple_burau']  # Shape: (24, 3, 3, 64)
    loaded_center = tables['center']        # = 32
    
    D = config.degree_window
    new_center = D // 2
    
    # Create new tensor with the right degree window size
    simple_burau = torch.zeros(24, 3, 3, D, dtype=torch.long)
    
    # Copy data from loaded tables, re-centering
    # Loaded data: degree d is at index loaded_center + d (i.e., index 32 + d)
    # New data: degree d should be at index new_center + d
    
    # Find the range of degrees actually used in loaded data
    for s in range(24):
        mat = loaded_burau[s]  # Shape: (3, 3, 64)
        
        # Find nonzero degree range
        nonzero_mask = mat.abs().sum(dim=(0, 1)) > 0
        if not nonzero_mask.any():
            continue  # Zero matrix, skip
            
        # Get indices where there's data
        nonzero_indices = torch.where(nonzero_mask)[0]
        src_start = nonzero_indices[0].item()
        src_end = nonzero_indices[-1].item() + 1
        
        # Convert to actual degrees: loaded_index = loaded_center + degree
        # So degree = loaded_index - loaded_center
        min_degree = src_start - loaded_center
        max_degree = src_end - 1 - loaded_center
        
        # New indices: new_index = new_center + degree
        dst_start = new_center + min_degree
        dst_end = new_center + max_degree + 1
        
        # Bounds check
        if dst_start < 0 or dst_end > D:
            raise ValueError(
                f"Simple {s} has degrees [{min_degree}, {max_degree}] which don't fit in "
                f"degree window of size {D} (center={new_center}). "
                f"Increase config.degree_window or config.degree_multiplier."
            )
        
        # Copy the data
        simple_burau[s, :, :, dst_start:dst_end] = mat[:, :, src_start:src_end]
    
    # Load suffix tables
    loaded_valid_suffixes = tables['valid_suffixes']      # Shape: (24, 22)
    loaded_num_valid = tables['num_valid_suffixes']       # Shape: (24,)
    
    # CRITICAL FIX: In peyl, identity (idx 0) has 0 valid suffixes because
    # identity is never a canonical factor. But in the GPU code, we use
    # index 0 to mean "start of braid" (i.e., what can follow Δ^k with no factors).
    # 
    # Solution: Copy Delta's suffixes to identity's slot, since "start of braid"
    # should allow all 22 nontrivial simples (same as what follows Delta).
    
    delta_idx = tables['delta_index']  # = 23
    id_idx = tables['id_index']        # = 0
    
    # Make mutable copies
    valid_suffixes = loaded_valid_suffixes.clone()
    num_valid_suffixes = loaded_num_valid.clone()
    
    # Copy Delta's suffix info to identity's slot
    valid_suffixes[id_idx] = valid_suffixes[delta_idx]
    num_valid_suffixes[id_idx] = num_valid_suffixes[delta_idx]
    
    print(f"Loaded tables from {table_path}")
    print(f"  Re-centered from degree_window=64 to degree_window={D}")
    print(f"  Fixed identity suffixes: now has {num_valid_suffixes[id_idx]} valid suffixes")
    print(f"  simple_burau shape: {simple_burau.shape}")
    print(f"  valid_suffixes shape: {valid_suffixes.shape}")
    
    return simple_burau, valid_suffixes, num_valid_suffixes

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Example usage."""
    
    # Configure the search
    config = Config(
        bucket_size=50000,
        max_length=50,
        bootstrap_length=5,
        prime=5,
        degree_multiplier=4,
        checkpoint_every=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        expansion_chunk_size=50000
    )
    
    print(f"Using device: {config.device}")
    print(f"Degree window size: {config.degree_window}")
    
    # Create tables (REPLACE with your actual precomputed data)
    simple_burau, valid_suffixes, num_valid_suffixes = load_tables_from_file(
       config, 
       table_path = os.path.join(project_root, "precomputed_tables", "tables_B4_r1_p2.pt")
   )
    
    # Add this after load_tables_from_file() call in main():
    center = config.degree_window // 2
    assert simple_burau[0, 0, 0, center] == 1, "Identity matrix check failed"
    assert simple_burau[0, 1, 1, center] == 1, "Identity matrix check failed"
    assert simple_burau[0, 2, 2, center] == 1, "Identity matrix check failed"
    print("✓ Identity matrix verified")

    
    # Run the search
    search = BraidSearch(simple_burau, valid_suffixes, num_valid_suffixes, config)
    zero_braids = search.run(checkpoint_dir="checkpoints")
    
    # Print any projlen=0 braids found
    for i, words in enumerate(zero_braids):
        print(f"\nBatch {i}: {len(words)} braids with projlen=0")
        for word in words[:5]:  # show first 5
            print(f"  Garside word: {word.tolist()}")

if __name__ == "__main__":
    main()