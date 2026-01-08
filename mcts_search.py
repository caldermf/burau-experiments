#!/usr/bin/env python3
"""
Monte Carlo Tree Search for finding braid kernel elements.

Uses fast GPU reservoir sampling as the "playout" step, giving us
massively parallel Monte Carlo evaluation of each node.

Key idea: Instead of evaluating a braid by its current projlen,
evaluate it by the BEST projlen reachable via random playouts.

Usage:
    python mcts_search.py --p 7 --playout-depth 50 --iterations 100
    python mcts_search.py --p 5 --playout-depth 40 --db-size 5000
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
import argparse
import time
import gc
import os
import heapq

from braid_search import (
    Config, FastPolyMatmul, GPUBuckets,
    compute_projlen_batch, build_expansion_indices_vectorized,
    load_tables_from_file, 
    STORAGE_DTYPE_MATRIX, STORAGE_DTYPE_WORD, STORAGE_DTYPE_LENGTH, COMPUTE_DTYPE_INT
)


# =============================================================================
# MCTS NODE
# =============================================================================

@dataclass
class MCTSNode:
    """A node in the MCTS tree (a braid at some depth)."""
    word: tuple  # Immutable tuple of generator indices
    length: int
    matrix: torch.Tensor  # (3, 3, D) on CPU for storage
    projlen: int  # Current projlen of this braid
    
    # MCTS statistics
    score: float = float('inf')  # Best projlen seen from this node (lower = better)
    visits: int = 0
    
    def __lt__(self, other):
        # For heap: lower score = higher priority
        return self.score < other.score
    
    def key(self) -> tuple:
        return self.word


# =============================================================================
# MCTS CONFIGURATION
# =============================================================================

@dataclass 
class MCTSConfig:
    """Configuration for MCTS search."""
    # Problem parameters
    prime: int = 7
    device: str = "cuda"
    
    # MCTS parameters
    max_iterations: int = 1000
    db_max_size: int = 10000  # Max nodes in database
    db_keep_size: int = 5000  # Keep this many after pruning
    select_top_k: int = 5     # How many nodes to expand per iteration
    
    # Playout parameters (your fast GPU search)
    playout_depth: int = 50   # How many levels to run per playout
    playout_bucket_size: int = 3000
    playout_use_best: int = 1500
    playout_bootstrap: int = 3
    
    # What to add back to database from playouts
    add_from_playout: int = 50  # Add top N braids from each playout
    add_projlen_threshold: int = 100  # Only add if projlen below this
    
    # Memory management
    degree_multiplier: int = 2
    expansion_chunk_size: int = 50000
    matmul_chunk_size: int = 8000
    
    @property
    def degree_window(self) -> int:
        # Need enough space for playout_depth levels from any starting point
        # A braid at depth D expanded for playout_depth more levels needs D + playout_depth
        max_possible_length = 500  # Conservative upper bound
        return self.degree_multiplier * max_possible_length + 1


# =============================================================================
# MCTS PLAYOUT ENGINE
# =============================================================================

class PlayoutEngine:
    """
    Runs fast GPU playouts from given starting braids.
    This is basically your BraidSearchUltra but starting from arbitrary braids.
    """
    
    def __init__(
        self,
        simple_burau: torch.Tensor,
        valid_suffixes: torch.Tensor,
        num_valid_suffixes: torch.Tensor,
        config: MCTSConfig
    ):
        self.config = config
        self.device = torch.device(config.device)
        
        self.simple_burau = simple_burau.to(STORAGE_DTYPE_MATRIX).to(self.device)
        self.valid_suffixes = valid_suffixes.to(self.device)
        self.num_valid_suffixes = num_valid_suffixes.to(self.device)
        
        self.D = simple_burau.shape[-1]
        self.fast_matmul = FastPolyMatmul(simple_burau, self.D, self.device)
        
        # For word storage during playouts
        self.max_word_length = 500  # Should be enough for any playout
    
    def run_playout(
        self,
        start_matrices: torch.Tensor,  # (N, 3, 3, D)
        start_words: torch.Tensor,      # (N, max_length)
        start_lengths: torch.Tensor,    # (N,)
        depth: int,
        bucket_size: int,
        use_best: int,
        bootstrap: int = 3
    ) -> tuple[dict, int, list]:
        """
        Run reservoir sampling playout from given starting braids.
        
        Returns:
            final_buckets: {projlen: (matrices, words, lengths)}
            best_projlen: Minimum projlen seen during entire playout
            kernel_braids: List of any projlen=1 braids found
        """
        N = len(start_matrices)
        
        # Move to device and ensure correct dtypes
        matrices = start_matrices.to(COMPUTE_DTYPE_INT).to(self.device)
        
        # Pad words to max_word_length if needed
        current_word_len = start_words.shape[1]
        if current_word_len < self.max_word_length:
            padding = torch.zeros(N, self.max_word_length - current_word_len, 
                                  dtype=STORAGE_DTYPE_WORD, device=self.device)
            words = torch.cat([start_words.to(self.device), padding], dim=1)
        else:
            words = start_words.to(self.device)
        
        lengths = start_lengths.to(STORAGE_DTYPE_LENGTH).to(self.device)
        
        # Initialize buckets with starting braids
        start_projlens = compute_projlen_batch(matrices)
        
        buckets = GPUBuckets(bucket_size, self.device)
        buckets.add_chunk(
            matrices.to(STORAGE_DTYPE_MATRIX), words, lengths, start_projlens,
            is_bootstrap=True
        )
        
        best_projlen = start_projlens.min().item()
        kernel_braids = []
        
        # Check if we already have kernel elements
        if best_projlen == 1:
            one_mask = (start_projlens == 1)
            kernel_braids.append(words[one_mask].cpu())
        
        # Run playout for `depth` levels
        for level in range(depth):
            is_bootstrap = (level < bootstrap)
            
            # Gather braids from buckets
            bucket_dict = buckets.get_buckets()
            if not bucket_dict:
                break
            
            sorted_projlens = sorted(bucket_dict.keys())
            
            all_matrices = []
            all_words = []
            all_lengths = []
            total_selected = 0
            
            for projlen in sorted_projlens:
                mat, wrd, lng = bucket_dict[projlen]
                bucket_count = len(mat)
                
                if not is_bootstrap and use_best > 0:
                    remaining = use_best - total_selected
                    if remaining <= 0:
                        break
                    
                    if bucket_count <= remaining:
                        all_matrices.append(mat)
                        all_words.append(wrd)
                        all_lengths.append(lng)
                        total_selected += bucket_count
                    else:
                        idx = torch.randperm(bucket_count, device=self.device)[:remaining]
                        all_matrices.append(mat[idx])
                        all_words.append(wrd[idx])
                        all_lengths.append(lng[idx])
                        total_selected += remaining
                        break
                else:
                    all_matrices.append(mat)
                    all_words.append(wrd)
                    all_lengths.append(lng)
            
            if not all_matrices:
                break
            
            matrices = torch.cat(all_matrices, dim=0).to(COMPUTE_DTYPE_INT)
            words = torch.cat(all_words, dim=0)
            lengths = torch.cat(all_lengths, dim=0)
            
            # Get last simples for expansion
            batch_idx = torch.arange(len(lengths), device=self.device)
            last_pos = torch.clamp(lengths - 1, min=0).long()
            last_simples = words[batch_idx, last_pos].long()
            last_simples = torch.where(lengths > 0, last_simples, torch.zeros_like(last_simples))
            
            # Build expansion indices
            braid_indices, suffix_indices = build_expansion_indices_vectorized(
                last_simples, self.num_valid_suffixes, self.valid_suffixes
            )
            
            if len(braid_indices) == 0:
                break
            
            # Clear old buckets
            buckets.clear()
            gc.collect()
            torch.cuda.empty_cache()
            
            # Process in chunks
            chunk_size = self.config.expansion_chunk_size
            new_buckets = GPUBuckets(bucket_size, self.device)
            
            for chunk_start in range(0, len(braid_indices), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(braid_indices))
                
                chunk_braid_idx = braid_indices[chunk_start:chunk_end]
                chunk_suffix_idx = suffix_indices[chunk_start:chunk_end]
                
                # Expand
                parent_matrices = matrices[chunk_braid_idx]
                parent_words = words[chunk_braid_idx]
                parent_lengths = lengths[chunk_braid_idx]
                
                new_matrices = self.fast_matmul.matmul_batch(
                    parent_matrices, chunk_suffix_idx,
                    self.config.prime,
                    chunk_size=self.config.matmul_chunk_size
                )
                
                # Recenter (trim to D)
                if new_matrices.shape[-1] > self.D:
                    new_matrices = new_matrices[..., :self.D]
                elif new_matrices.shape[-1] < self.D:
                    new_matrices = F.pad(new_matrices, (0, self.D - new_matrices.shape[-1]))
                
                # Update words and lengths
                num_chunk = len(chunk_braid_idx)
                new_words = parent_words.clone()
                new_words[torch.arange(num_chunk, device=self.device), parent_lengths.long()] = \
                    chunk_suffix_idx.to(STORAGE_DTYPE_WORD)
                new_lengths = parent_lengths + 1
                
                # Compute projlens
                new_projlens = compute_projlen_batch(new_matrices)
                
                # Check for kernel elements
                one_mask = (new_projlens == 1)
                if one_mask.any():
                    kernel_braids.append(new_words[one_mask].cpu())
                
                # Track best
                level_best = new_projlens.min().item()
                best_projlen = min(best_projlen, level_best)
                
                # Add to buckets
                new_buckets.add_chunk(
                    new_matrices, new_words, new_lengths, new_projlens,
                    is_bootstrap=is_bootstrap
                )
                
                del parent_matrices, parent_words, parent_lengths
                del new_matrices, new_words, new_lengths, new_projlens
            
            buckets = new_buckets
            del matrices, words, lengths
            
            gc.collect()
            torch.cuda.empty_cache()
        
        # Return final buckets
        final_buckets = buckets.get_buckets()
        return final_buckets, best_projlen, kernel_braids


# =============================================================================
# MCTS SEARCH
# =============================================================================

class MCTSBraidSearch:
    """
    Monte Carlo Tree Search for kernel elements.
    
    Maintains a database of promising braids and uses fast GPU playouts
    to evaluate and explore them.
    """
    
    def __init__(
        self,
        simple_burau: torch.Tensor,
        valid_suffixes: torch.Tensor,
        num_valid_suffixes: torch.Tensor,
        config: MCTSConfig
    ):
        self.config = config
        self.device = torch.device(config.device)
        
        # Store tables
        self.simple_burau = simple_burau
        self.valid_suffixes = valid_suffixes
        self.num_valid_suffixes = num_valid_suffixes
        
        # Playout engine
        self.engine = PlayoutEngine(
            simple_burau, valid_suffixes, num_valid_suffixes, config
        )
        
        # Database of MCTS nodes: key -> MCTSNode
        self.database: dict[tuple, MCTSNode] = {}
        
        # Priority queue for selection (min-heap by score)
        # We'll rebuild this as needed
        
        # Track found kernel elements
        self.kernel_braids: list[torch.Tensor] = []
        
        # Statistics
        self.stats = {
            'iterations': 0,
            'total_playouts': 0,
            'best_score_history': [],
        }
    
    def initialize(self):
        """Initialize with identity braid."""
        D = self.engine.D
        
        identity_matrix = torch.zeros(3, 3, D, dtype=STORAGE_DTYPE_MATRIX)
        for i in range(3):
            identity_matrix[i, i, 0] = 1
        
        identity_node = MCTSNode(
            word=(),
            length=0,
            matrix=identity_matrix,
            projlen=1,
            score=float('inf'),
            visits=0
        )
        
        self.database[identity_node.key()] = identity_node
        
        print(f"MCTS initialized with identity braid")
        print(f"Database max size: {self.config.db_max_size}")
        print(f"Playout depth: {self.config.playout_depth}")
        print(f"Playout bucket_size: {self.config.playout_bucket_size}")
        print(f"Playout use_best: {self.config.playout_use_best}")
    
    def select_nodes(self, k: int) -> list[MCTSNode]:
        """Select top-k nodes by score for expansion."""
        # Simple strategy: pick the k nodes with best (lowest) scores
        # Could add UCB1-style exploration bonus here
        
        nodes = list(self.database.values())
        nodes.sort(key=lambda n: (n.score, -n.visits))  # Best score, then most visited
        
        return nodes[:k]
    
    def add_node(self, word: tuple, length: int, matrix: torch.Tensor, projlen: int):
        """Add a node to the database if it's promising."""
        key = word
        
        if key in self.database:
            # Update existing node if this is better
            existing = self.database[key]
            if projlen < existing.score:
                existing.score = projlen
            return
        
        if projlen > self.config.add_projlen_threshold:
            return  # Not promising enough
        
        node = MCTSNode(
            word=word,
            length=length,
            matrix=matrix.cpu() if matrix.is_cuda else matrix,
            projlen=projlen,
            score=projlen,  # Initial score is current projlen
            visits=0
        )
        
        self.database[key] = node
    
    def prune_database(self):
        """Prune database to keep_size if over max_size."""
        if len(self.database) <= self.config.db_max_size:
            return
        
        print(f"  Pruning database: {len(self.database)} -> {self.config.db_keep_size}")
        
        # Keep the best nodes by score
        nodes = list(self.database.values())
        nodes.sort(key=lambda n: (n.score, n.length))  # Best score, then shortest
        
        keep_nodes = nodes[:self.config.db_keep_size]
        self.database = {n.key(): n for n in keep_nodes}
    
    def run_iteration(self) -> bool:
        """
        Run one MCTS iteration:
        1. Select promising nodes
        2. Run playouts from each
        3. Backpropagate scores
        4. Add promising children to database
        
        Returns True if kernel element found.
        """
        # Select nodes to expand
        selected = self.select_nodes(self.config.select_top_k)
        
        if not selected:
            print("  No nodes to expand!")
            return False
        
        print(f"  Selected {len(selected)} nodes for expansion")
        for i, node in enumerate(selected[:3]):
            print(f"    Node {i}: length={node.length}, projlen={node.projlen}, "
                  f"score={node.score:.1f}, visits={node.visits}")
        
        # Run playout for each selected node
        for node in selected:
            # Prepare starting tensors
            start_matrix = node.matrix.unsqueeze(0).to(self.device)
            
            # Convert word tuple to tensor
            word_tensor = torch.zeros(1, self.engine.max_word_length, 
                                      dtype=STORAGE_DTYPE_WORD, device=self.device)
            for i, w in enumerate(node.word):
                word_tensor[0, i] = w
            
            start_length = torch.tensor([node.length], dtype=STORAGE_DTYPE_LENGTH, 
                                        device=self.device)
            
            # Run playout
            t0 = time.time()
            final_buckets, best_projlen, kernel_found = self.engine.run_playout(
                start_matrix, word_tensor, start_length,
                depth=self.config.playout_depth,
                bucket_size=self.config.playout_bucket_size,
                use_best=self.config.playout_use_best,
                bootstrap=self.config.playout_bootstrap
            )
            playout_time = time.time() - t0
            
            self.stats['total_playouts'] += 1
            
            # Check for kernel elements
            if kernel_found:
                self.kernel_braids.extend(kernel_found)
                print(f"\n  🎉 KERNEL ELEMENT FOUND! 🎉")
                return True
            
            # Backpropagate: update node's score
            old_score = node.score
            node.score = min(node.score, best_projlen)
            node.visits += 1
            
            print(f"    Playout from length={node.length}: "
                  f"best_projlen={best_projlen}, "
                  f"score: {old_score:.1f} -> {node.score:.1f}, "
                  f"time={playout_time:.1f}s")
            
            # Add promising endpoints to database
            added_count = 0
            for projlen in sorted(final_buckets.keys()):
                if projlen > self.config.add_projlen_threshold:
                    continue
                
                matrices, words, lengths = final_buckets[projlen]
                
                # Add top N from this projlen bucket
                n_to_add = min(self.config.add_from_playout, len(matrices))
                
                for i in range(n_to_add):
                    length_i = lengths[i].item()
                    word_list = words[i, :length_i].cpu().tolist()
                    word_tuple = tuple(word_list)
                    
                    if word_tuple not in self.database:
                        self.add_node(
                            word=word_tuple,
                            length=length_i,
                            matrix=matrices[i].cpu(),
                            projlen=projlen
                        )
                        added_count += 1
            
            if added_count > 0:
                print(f"    Added {added_count} new nodes to database")
            
            # Clean up
            del start_matrix, word_tensor, start_length
            del final_buckets
            gc.collect()
            torch.cuda.empty_cache()
        
        return False
    
    def run(self) -> list[torch.Tensor]:
        """Run the full MCTS search."""
        self.initialize()
        
        total_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"MCTS SEARCH FOR p={self.config.prime} KERNEL ELEMENTS")
        print(f"{'='*60}")
        
        try:
            for iteration in range(self.config.max_iterations):
                self.stats['iterations'] = iteration + 1
                
                print(f"\n--- Iteration {iteration + 1}/{self.config.max_iterations} ---")
                print(f"  Database size: {len(self.database)}")
                
                # Get best score in database
                if self.database:
                    best_score = min(n.score for n in self.database.values())
                    best_length = min((n.length for n in self.database.values() 
                                      if n.score == best_score), default=0)
                    self.stats['best_score_history'].append(best_score)
                    print(f"  Best score in DB: {best_score} (at length {best_length})")
                
                # Run iteration
                found = self.run_iteration()
                
                if found:
                    break
                
                # Prune database if needed
                self.prune_database()
                
        except KeyboardInterrupt:
            print(f"\n\n⚠️ Interrupted!")
        
        finally:
            total_time = time.time() - total_start
            
            print(f"\n{'='*60}")
            print("MCTS SEARCH COMPLETE")
            print(f"{'='*60}")
            print(f"Iterations: {self.stats['iterations']}")
            print(f"Total playouts: {self.stats['total_playouts']}")
            print(f"Total time: {total_time:.1f}s")
            print(f"Final database size: {len(self.database)}")
            
            if self.database:
                best_score = min(n.score for n in self.database.values())
                print(f"Best score achieved: {best_score}")
            
            print(f"Kernel elements found: {sum(len(k) for k in self.kernel_braids)}")
        
        return self.kernel_braids


# =============================================================================
# VERIFICATION (from find_kernel.py)
# =============================================================================

def verify_kernel_element(word_list, n=4, r=1, p=2):
    """Verify that a braid word is actually in the kernel."""
    try:
        from peyl.braid import GNF
        from peyl.jonesrep import JonesCellRep
        import numpy as np
    except ImportError:
        return True, "Verification skipped (peyl not available)"
    
    if not word_list:
        return False, "Empty word"
    
    try:
        braid = GNF(n=n, power=0, factors=tuple(word_list))
    except AssertionError as e:
        return False, f"Invalid normal form: {e}"
    
    rep = JonesCellRep(n=n, r=r, p=p)
    result = rep.polymat_evaluate_braid(braid)
    if p > 0:
        result = result % p
    
    diag_poly = result[0, 0, :]
    
    for i in range(3):
        for j in range(3):
            if i == j:
                if not np.array_equal(result[i, j, :], diag_poly):
                    return False, f"Diagonal mismatch at [{i},{j}]"
            else:
                if np.any(result[i, j, :] != 0):
                    return False, f"Off-diagonal nonzero at [{i},{j}]"
    
    return True, "Kernel element verified!"


# =============================================================================
# MAIN
# =============================================================================

def find_table_path(p: int) -> Optional[str]:
    """Find the precomputed tables file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(script_dir, "precomputed_tables", f"tables_B4_r1_p{p}.pt"),
        os.path.join(os.path.dirname(script_dir), "precomputed_tables", f"tables_B4_r1_p{p}.pt"),
        os.path.join(script_dir, f"tables_B4_r1_p{p}.pt"),
        f"tables_B4_r1_p{p}.pt",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="MCTS search for braid kernel elements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test on p=5 (should find kernel elements)
  python mcts_search.py --p 5 --playout-depth 40 --iterations 50
  
  # Long p=7 search
  python mcts_search.py --p 7 --playout-depth 50 --iterations 500 --db-size 10000
  
  # Aggressive exploration for p=7
  python mcts_search.py --p 7 --playout-depth 60 --playout-bucket 5000 --playout-use-best 2500
        """
    )
    
    # Problem parameters
    parser.add_argument("--p", type=int, default=5,
                        help="Prime for the representation (default: 5)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    
    # MCTS parameters
    parser.add_argument("--iterations", type=int, default=100,
                        help="Max MCTS iterations (default: 100)")
    parser.add_argument("--db-size", type=int, default=10000,
                        help="Max database size (default: 10000)")
    parser.add_argument("--select-k", type=int, default=5,
                        help="Nodes to expand per iteration (default: 5)")
    
    # Playout parameters
    parser.add_argument("--playout-depth", type=int, default=50,
                        help="Depth of each playout (default: 50)")
    parser.add_argument("--playout-bucket", type=int, default=3000,
                        help="Bucket size for playouts (default: 3000)")
    parser.add_argument("--playout-use-best", type=int, default=1500,
                        help="Use-best for playouts (default: 1500)")
    
    # What to keep from playouts
    parser.add_argument("--add-from-playout", type=int, default=50,
                        help="Braids to add from each playout (default: 50)")
    parser.add_argument("--add-threshold", type=int, default=100,
                        help="Max projlen to add to database (default: 100)")
    
    # Memory parameters
    parser.add_argument("--degree-mult", type=int, default=2,
                        help="Degree multiplier (default: 2)")
    parser.add_argument("--matmul-chunk", type=int, default=8000,
                        help="Matmul chunk size (default: 8000)")
    
    args = parser.parse_args()
    
    # Build config
    config = MCTSConfig(
        prime=args.p,
        device=args.device,
        max_iterations=args.iterations,
        db_max_size=args.db_size,
        db_keep_size=args.db_size // 2,
        select_top_k=args.select_k,
        playout_depth=args.playout_depth,
        playout_bucket_size=args.playout_bucket,
        playout_use_best=args.playout_use_best,
        add_from_playout=args.add_from_playout,
        add_projlen_threshold=args.add_threshold,
        degree_multiplier=args.degree_mult,
        matmul_chunk_size=args.matmul_chunk,
    )
    
    print("="*60)
    print(f"MCTS KERNEL SEARCH")
    print("="*60)
    print(f"Prime: {config.prime}")
    print(f"Device: {config.device}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Database size: {config.db_max_size}")
    print(f"Select top-k: {config.select_top_k}")
    print(f"Playout depth: {config.playout_depth}")
    print(f"Playout bucket_size: {config.playout_bucket_size}")
    print(f"Playout use_best: {config.playout_use_best}")
    print()
    
    # Load tables
    table_path = find_table_path(config.prime)
    if table_path is None:
        print(f"ERROR: Could not find table file for p={config.prime}")
        return
    
    # Create a dummy Config for loading tables
    load_config = Config(
        prime=config.prime,
        degree_multiplier=config.degree_multiplier,
        max_length=config.degree_window // config.degree_multiplier,
    )
    
    simple_burau, valid_suffixes, num_valid_suffixes = load_tables_from_file(
        load_config, table_path
    )
    
    # Run MCTS
    search = MCTSBraidSearch(simple_burau, valid_suffixes, num_valid_suffixes, config)
    kernel_braids = search.run()
    
    # Verify results
    if kernel_braids:
        print(f"\n{'='*60}")
        print("VERIFICATION")
        print("="*60)
        
        verified = []
        for batch in kernel_braids:
            for word_tensor in batch:
                word_list = [w.item() for w in word_tensor]
                while word_list and word_list[-1] == 0:
                    word_list.pop()
                
                if not word_list:
                    continue
                
                is_kernel, msg = verify_kernel_element(word_list, p=config.prime)
                
                if is_kernel:
                    verified.append(word_list)
                    print(f"\n  🎉 VERIFIED KERNEL ELEMENT 🎉")
                    print(f"    Length: {len(word_list)}")
                    print(f"    Factors: {word_list[:20]}{'...' if len(word_list) > 20 else ''}")
        
        print(f"\nVerified {len(verified)} kernel elements for p={config.prime}")


if __name__ == "__main__":
    main()
