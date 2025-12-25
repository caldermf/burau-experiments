"""
GPU-accelerated Tracker for Burau kernel element search.

This module provides a fully GPU-optimized version of the Tracker class,
designed to work with the JonesCellRep class from peyl/jonesrep.py.

Key Features:
1. All matrix operations run on GPU (multiplication, projectivization, projlen)
2. All loops are GPU-parallelized via batched operations
3. Correct API usage - works with JonesCellRep interface
4. Minimal CPU<->GPU transfers

Usage:
    from peyl.jonesrep import JonesCellRep
    from gpu_tracker_v2 import GPUTracker

    rep = JonesCellRep(n=4, r=1, p=5)
    tracker = GPUTracker(rep, bucket_size=500)
    tracker.bootstrap_exhaustive(upto_length=8)
    tracker.run_search(max_length=80)
"""

from __future__ import annotations

import functools
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

# Import GPU polymat operations
try:
    from . import gpu_polymat as gpm
except ImportError:
    import gpu_polymat as gpm

# Import from the actual braid library
try:
    from peyl.braid import GNF, DGNF, NFBase, BraidGroup, PermTable
    from peyl.jonesrep import JonesCellRep, RepBase
    from peyl.permutations import SymmetricGroup, Permutation
    from peyl import polymat
    BRAID_LIB_AVAILABLE = True
except ImportError as e:
    BRAID_LIB_AVAILABLE = False
    print(f"Warning: braid library not found: {e}")


@dataclass
class SearchStats:
    """Statistics about the search progress."""
    total_braids_stored: int
    total_braids_seen: int
    n_active_buckets: int
    current_length: int
    min_projlen_at_length: Dict[int, int]
    kernel_candidates: List[Tuple[int, int, str]]
    elapsed_time: float


class GPUTracker:
    """
    GPU-accelerated version of the reservoir sampling tracker.

    This class maintains buckets of braids organized by (Garside length, projlen),
    with efficient GPU computation of matrix products and statistics.

    All matrix operations are batched and run on GPU when available. The key
    optimization is in nf_descendants(), which computes all descendants of a
    bucket in a single batched GPU operation.
    """

    def __init__(
        self,
        rep: RepBase,
        bucket_size: int = 500,
        seed: Optional[int] = None,
        use_gpu: bool = True,
        criterion: Optional[Callable] = None,
    ):
        """
        Initialize the GPU tracker.

        Args:
            rep: Representation object (JonesCellRep from jonesrep.py)
            bucket_size: Maximum elements per bucket for reservoir sampling
            seed: Random seed for reproducibility
            use_gpu: Whether to use GPU acceleration
            criterion: Optional function to filter which braids to keep
        """
        if not BRAID_LIB_AVAILABLE:
            raise RuntimeError("Braid library not available. Please ensure peyl is installed.")

        self.rep = rep
        self.bucket_size = bucket_size
        self.use_gpu = use_gpu and gpm.GPU_AVAILABLE
        self.criterion = criterion

        # Random state
        self.rng = random.Random(seed)

        # Get representation parameters
        self.n = rep.n
        self.p = rep.p
        self.dim = rep.dimension()
        self.dtype = rep.polymat_dtype()

        # Storage: buckets are keyed by (garside_length, projlen)
        # Images are stored on CPU as numpy arrays (GPU memory is used for computation)
        self.bucket_braids: Dict[Tuple[int, int], List[GNF]] = {}
        self.bucket_images: Dict[Tuple[int, int], List[npt.NDArray]] = {}
        self.bucket_reservoir_counts: Dict[Tuple[int, int], int] = {}
        self.active_buckets: Set[Tuple[int, int]] = set()

        # Results
        self.kernel_candidates: List[Tuple[int, int, GNF]] = []
        self.start_time = time.time()

        # Precompute lookup tables for fast evaluation
        self._init_tables()

        print(f"GPUTracker initialized:")
        print(f"  Representation: {rep}")
        print(f"  Dimension: {self.dim}")
        print(f"  GPU enabled: {self.use_gpu}")
        print(f"  Bucket size: {bucket_size}")

    def _init_tables(self):
        """
        Precompute lookup tables for fast braid evaluation.

        We build two tables:
        1. _perm_table: Maps permutation index -> polymat (for GNF factors)
        2. _sym_table: Maps Permutation object -> polymat (for convenience)
        """
        # Get the NF table for this n
        self._nf_table = GNF._nf_table(self.n)

        # Build table mapping factor index -> polymat
        # This uses the internal _polymat_braid_factor method from RepBase
        print(f"  Building factor table ({self._nf_table.order} permutations)...", end=" ", flush=True)

        self._factor_table: List[npt.NDArray] = []
        for factor_idx in range(self._nf_table.order):
            mat = self.rep._polymat_braid_factor(GNF, factor_idx)
            self._factor_table.append(mat)

        # Also build delta power table
        self._delta_table: List[npt.NDArray] = []
        for power in range(self._nf_table.tau_order):
            mat = self.rep._polymat_delta_power(GNF, power)
            self._delta_table.append(mat)

        print("done")

        # Transfer tables to GPU if enabled
        if self.use_gpu:
            print("  Transferring tables to GPU...", end=" ", flush=True)
            self._factor_table_gpu = [gpm.to_gpu(m) for m in self._factor_table]
            self._delta_table_gpu = [gpm.to_gpu(m) for m in self._delta_table]
            print("done")
        else:
            self._factor_table_gpu = self._factor_table
            self._delta_table_gpu = self._delta_table

    def _get_factor_matrix(self, factor_idx: int, on_gpu: bool = False) -> npt.NDArray:
        """Get the polymat for a normal form factor by index."""
        if on_gpu and self.use_gpu:
            return self._factor_table_gpu[factor_idx]
        return self._factor_table[factor_idx]

    def _get_delta_matrix(self, power: int, on_gpu: bool = False) -> npt.NDArray:
        """Get the polymat for delta^power (reduced mod tau_order)."""
        power = power % self._nf_table.tau_order
        if on_gpu and self.use_gpu:
            return self._delta_table_gpu[power]
        return self._delta_table[power]

    def _evaluate_braid(self, braid: GNF) -> npt.NDArray:
        """
        Evaluate a single braid in the representation.
        Returns a CPU numpy array.
        """
        # Start with appropriate delta power
        result = self._get_delta_matrix(braid.power, on_gpu=False).copy()

        # Multiply by each factor
        for factor_idx in braid.factors:
            factor_mat = self._get_factor_matrix(factor_idx, on_gpu=False)
            result = self.rep.polymat_mul(result, factor_mat)

        return result

    def _evaluate_braids_batch(self, braids: List[GNF]) -> npt.NDArray:
        """
        Evaluate a batch of braids of the same Garside length.
        All computation happens on GPU if available.
        Returns a numpy array of shape (batch, dim, dim, degree).
        """
        if not braids:
            return np.zeros((0, self.dim, self.dim, 1), dtype=self.dtype)

        length = braids[0].canonical_length()
        assert all(b.canonical_length() == length for b in braids), "All braids must have same length"

        # Pack delta powers
        delta_mats = [self._get_delta_matrix(b.power, on_gpu=False) for b in braids]
        result = gpm.pack(delta_mats)

        if self.use_gpu:
            result = gpm.to_gpu(result)

        # Multiply by each factor position
        for pos in range(length):
            factor_mats = [self._get_factor_matrix(b.factors[pos], on_gpu=False) for b in braids]
            factors = gpm.pack(factor_mats)
            if self.use_gpu:
                factors = gpm.to_gpu(factors)
            result = gpm.mul(result, factors, p=self.p)
            result = gpm.projectivise(result)

        return gpm.to_cpu(result) if self.use_gpu else result

    def _evaluate_suffixes_batch(
        self,
        base_images: npt.NDArray,
        suffix_factors: List[Tuple[int, ...]],
    ) -> npt.NDArray:
        """
        Given base images and suffix factor sequences, compute base * suffix for each.

        This is the core GPU kernel - all operations are batched on GPU.

        Args:
            base_images: Array of shape (batch, dim, dim, deg) - the base braid images
            suffix_factors: List of tuples, each tuple contains factor indices for a suffix

        Returns:
            Array of shape (batch, dim, dim, deg) with the products
        """
        if len(suffix_factors) == 0:
            return base_images

        # Ensure all suffixes have the same length
        suffix_len = len(suffix_factors[0])
        assert all(len(sf) == suffix_len for sf in suffix_factors)

        # Transfer base to GPU
        if self.use_gpu:
            result = gpm.to_gpu(base_images)
        else:
            result = base_images.copy()

        # Multiply by each suffix factor position
        for pos in range(suffix_len):
            factor_mats = [self._get_factor_matrix(sf[pos], on_gpu=False) for sf in suffix_factors]
            factors = gpm.pack(factor_mats)
            if self.use_gpu:
                factors = gpm.to_gpu(factors)
            result = gpm.mul(result, factors, p=self.p)
            result = gpm.projectivise(result)

        return gpm.to_cpu(result) if self.use_gpu else result

    def add_braids_images(
        self,
        braids: List[GNF],
        images: npt.NDArray
    ) -> int:
        """
        Add braids and their images to appropriate buckets.

        Args:
            braids: List of GNF braid objects
            images: Array of shape (batch, dim, dim, degree)

        Returns:
            Number of braids actually added
        """
        if len(braids) == 0:
            return 0

        # Ensure images are on CPU for storage
        images_cpu = gpm.to_cpu(images) if gpm.is_gpu_array(images) else images

        # Compute projlens on GPU if available
        if self.use_gpu:
            images_gpu = gpm.to_gpu(images_cpu)
            projlens = gpm.to_cpu(gpm.projlen(images_gpu))
        else:
            projlens = gpm.projlen(images_cpu)

        # Get Garside lengths
        lengths = np.array([braid.canonical_length() for braid in braids])

        # Apply criterion if provided
        if self.criterion is not None:
            keep_mask = self.criterion(lengths, projlens)
        else:
            keep_mask = np.ones(len(braids), dtype=bool)

        n_added = 0
        for i in range(len(braids)):
            if not keep_mask[i]:
                continue

            braid = braids[i]
            length = int(lengths[i])
            pl = int(projlens[i])
            image = images_cpu[i] if len(images_cpu.shape) == 4 else images_cpu
            bucket = (length, pl)

            # Check for kernel element!
            if pl == 0 and length > 0:
                print(f"\n  KERNEL ELEMENT at length {length}!")
                print(f"   Braid: {braid}")
                self.kernel_candidates.append((length, pl, braid))

            # Reservoir sampling into buckets
            if bucket not in self.active_buckets:
                self.active_buckets.add(bucket)
                self.bucket_braids[bucket] = [braid]
                self.bucket_images[bucket] = [image]
                self.bucket_reservoir_counts[bucket] = 1
                n_added += 1
                continue

            self.bucket_reservoir_counts[bucket] += 1
            current_count = len(self.bucket_braids[bucket])

            if current_count < self.bucket_size:
                # Bucket not full, just add
                self.bucket_braids[bucket].append(braid)
                self.bucket_images[bucket].append(image)
                n_added += 1
            else:
                # Reservoir sampling: replace with probability bucket_size/count
                j = self.rng.randint(1, self.bucket_reservoir_counts[bucket])
                if j <= self.bucket_size:
                    self.bucket_braids[bucket][j-1] = braid
                    self.bucket_images[bucket][j-1] = image

        return n_added

    def nf_descendants(self, bucket: Tuple[int, int], suffix_length: int = 1):
        """
        Compute Garside normal form descendants of all braids in a bucket.

        This is the main computational kernel. All operations are batched on GPU.

        For each braid β in the bucket and each valid suffix σ of length suffix_length,
        we compute β * σ and add it to the appropriate bucket.
        """
        if bucket not in self.active_buckets:
            return

        braids = self.bucket_braids[bucket]
        images = self.bucket_images[bucket]

        if not braids:
            return

        # Gather all (base_idx, new_braid, suffix_factors) tuples
        all_pairs: List[Tuple[int, GNF, Tuple[int, ...]]] = []

        for i, braid in enumerate(braids):
            # Get all valid suffixes
            last_factor = braid.factors[-1] if braid.factors else self._nf_table.D

            for suffix_factors in self._nf_table.normal_forms(suffix_length, following=last_factor):
                # Create the new braid (no recomputation needed since factors are in NF)
                new_braid = GNF(self.n, braid.power, braid.factors + suffix_factors)
                all_pairs.append((i, new_braid, suffix_factors))

        if not all_pairs:
            return

        # Process in batches to avoid GPU memory issues
        batch_size = 5000

        for batch_start in range(0, len(all_pairs), batch_size):
            batch = all_pairs[batch_start:batch_start + batch_size]

            # Pack base images
            base_indices = [idx for idx, _, _ in batch]
            base_images_list = [images[idx] for idx in base_indices]
            base_images = gpm.pack(base_images_list)

            # Get suffix factors
            suffix_factors = [sf for _, _, sf in batch]

            # Compute products on GPU
            products = self._evaluate_suffixes_batch(base_images, suffix_factors)

            # Get new braids
            new_braids = [b for _, b, _ in batch]

            # Add results
            self.add_braids_images(new_braids, products)

    def bootstrap_exhaustive(self, upto_length: int):
        """
        Bootstrap buckets by exhaustively enumerating all braids up to given length.
        """
        B = BraidGroup(self.n)

        # Add identity
        eye = self.rep.polymat_id()
        identity = GNF.identity(self.n)
        self.add_braids_images([identity], eye[None, ...])

        for length in range(1, upto_length + 1):
            print(f"  Bootstrapping length {length}...", end=" ", flush=True)
            count = 0

            # Enumerate all braids of this length in batches
            batch: List[GNF] = []
            for braid in GNF.all_of_length(self.n, length):
                batch.append(braid)
                if len(batch) >= 1000:
                    images = self._evaluate_braids_batch(batch)
                    self.add_braids_images(batch, images)
                    count += len(batch)
                    batch = []

            # Process remaining
            if batch:
                images = self._evaluate_braids_batch(batch)
                self.add_braids_images(batch, images)
                count += len(batch)

            min_pl = self._min_projlen_at_length(length)
            print(f"added {count}, min_projlen={min_pl}")

    def advance_one_step(self, suffix_length: int = 1, verbose: bool = True):
        """
        Process all current buckets at max length and compute their descendants.
        """
        if not self.active_buckets:
            return

        current_max = max(b[0] for b in self.active_buckets)
        buckets_to_process = [b for b in self.active_buckets if b[0] == current_max]

        start = time.time()
        for bucket in buckets_to_process:
            self.nf_descendants(bucket, suffix_length)

        elapsed = time.time() - start
        new_length = current_max + suffix_length
        min_pl = self._min_projlen_at_length(new_length)

        if verbose:
            total_stored = sum(len(self.bucket_braids[b]) for b in self.active_buckets)
            print(f"  Length {new_length}: min_projlen={min_pl}, "
                  f"total_stored={total_stored}, time={elapsed:.1f}s")

    def run_search(
        self,
        bootstrap_length: int = 8,
        max_length: int = 80,
        suffix_length: int = 1,
        verbose: bool = True
    ) -> SearchStats:
        """
        Run the complete search algorithm.

        Args:
            bootstrap_length: Exhaustively enumerate up to this length first
            max_length: Stop searching at this Garside length
            suffix_length: How many Garside factors to extend by each step
            verbose: Print progress

        Returns:
            SearchStats object with results
        """
        print(f"\n{'='*60}")
        print(f"GPU Burau Kernel Search")
        print(f"n={self.n}, p={self.p}, bucket_size={self.bucket_size}")
        print(f"{'='*60}\n")

        # Bootstrap
        print("Phase 1: Exhaustive bootstrap")
        self.bootstrap_exhaustive(bootstrap_length)

        # Main search loop
        print(f"\nPhase 2: Reservoir sampling (up to length {max_length})")

        current_length = bootstrap_length
        while current_length < max_length:
            self.advance_one_step(suffix_length, verbose)
            current_length += suffix_length

            # Check for kernel elements
            if self.kernel_candidates:
                print(f"\n  Found {len(self.kernel_candidates)} kernel element(s)!")
                break

        # Return statistics
        return self.get_stats()

    def _min_projlen_at_length(self, length: int) -> Optional[int]:
        """Get minimum projlen among buckets at given length."""
        pls = [pl for (l, pl) in self.active_buckets if l == length]
        return min(pls) if pls else None

    def has_kernel_elements(self) -> bool:
        """Check if any kernel elements have been found."""
        return len(self.kernel_candidates) > 0

    def get_stats(self) -> SearchStats:
        """Get current search statistics."""
        total_stored = sum(len(self.bucket_braids[b]) for b in self.active_buckets)
        total_seen = sum(self.bucket_reservoir_counts.values())

        min_projlen_by_length: Dict[int, int] = {}
        for length, pl in self.active_buckets:
            if length not in min_projlen_by_length:
                min_projlen_by_length[length] = pl
            else:
                min_projlen_by_length[length] = min(min_projlen_by_length[length], pl)

        max_length = max(b[0] for b in self.active_buckets) if self.active_buckets else 0

        return SearchStats(
            total_braids_stored=total_stored,
            total_braids_seen=total_seen,
            n_active_buckets=len(self.active_buckets),
            current_length=max_length,
            min_projlen_at_length=min_projlen_by_length,
            kernel_candidates=[(l, pl, str(b)) for l, pl, b in self.kernel_candidates],
            elapsed_time=time.time() - self.start_time,
        )

    def get_bucket_braids(self, length: int, projlen: int) -> List[GNF]:
        """Get braids in a specific bucket."""
        return self.bucket_braids.get((length, projlen), [])

    def get_trajectory(self, braid: GNF) -> List[Tuple[int, int]]:
        """
        Get the trajectory of a braid through (length, projlen) space.
        Useful for visualizing kernel element paths.
        """
        trajectory = []

        # Build up prefix by prefix
        for i in range(len(braid.factors)):
            prefix = GNF(self.n, braid.power, braid.factors[:i+1])
            image = self._evaluate_braid(prefix)
            pl = int(gpm.projlen(image[None, ...])[0])
            trajectory.append((i + 1, pl))

        return trajectory


def batched(iterable, chunk_size: int):
    """Split an iterable into chunks."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# Convenience function for quick testing
def test_tracker():
    """Quick test to verify the tracker works."""
    if not BRAID_LIB_AVAILABLE:
        print("Braid library not available, skipping test")
        return

    print("Testing GPUTracker...")

    rep = JonesCellRep(n=4, r=1, p=5)
    tracker = GPUTracker(rep, bucket_size=100, seed=42)

    # Test bootstrap
    tracker.bootstrap_exhaustive(upto_length=3)

    # Test one step
    tracker.advance_one_step()

    stats = tracker.get_stats()
    print(f"\nTest complete:")
    print(f"  Braids stored: {stats.total_braids_stored}")
    print(f"  Active buckets: {stats.n_active_buckets}")
    print(f"  Current length: {stats.current_length}")

    return tracker


if __name__ == "__main__":
    test_tracker()
