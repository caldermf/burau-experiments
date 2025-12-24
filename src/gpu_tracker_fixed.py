"""
GPU-accelerated Tracker for Burau kernel element search.
"""

import functools
import random
import time
from typing import Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass

import numpy as np

# Import GPU polymat operations
import gpu_polymat as gpm

# Import from the braid library
try:
    from peyl.braid import GNF, DGNF, BraidGroup
    from peyl.jonesrep import JonesSummand
    from peyl.permutations import SymmetricGroup
    from peyl import polymat
    BRAID_LIB_AVAILABLE = True
except ImportError:
    BRAID_LIB_AVAILABLE = False
    print("Warning: braid library not found, some features unavailable")


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


def symmetric_table_gpu(rep: "JonesSummand", use_gpu: bool = True):
    """Build the symmetric table, optionally on GPU."""
    gens, invs = rep.artin_gens_invs()
    eye = rep.id()
    
    table = {}
    for perm in SymmetricGroup(rep.n).elements():
        result = functools.reduce(rep.mul, [gens[s] for s in perm.shortlex()], eye)
        result = polymat.projectivise(result)
        
        # Transfer to GPU if requested
        if use_gpu and gpm.GPU_AVAILABLE:
            result = gpm.to_gpu(result)
            
        table[perm] = result
        
    return table


class GPUTracker:
    """GPU-accelerated version of the reservoir sampling tracker."""
    
    def __init__(
        self,
        rep: "JonesSummand",
        bucket_size: int = 500,
        seed: Optional[int] = None,
        use_gpu: bool = True,
        criterion: Optional[Callable] = None,
    ):
        """
        Args:
            rep: JonesSummand representation object
            bucket_size: Maximum elements per bucket for reservoir sampling
            seed: Random seed for reproducibility
            use_gpu: Whether to use GPU acceleration
            criterion: Optional function to filter which braids to keep
        """
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
        self.dtype = rep.dtype()
        
        # Storage
        self.bucket_braids: Dict[Tuple[int, int], List] = {}
        self.bucket_images: Dict[Tuple[int, int], List[np.ndarray]] = {}
        self.bucket_reservoir_counts: Dict[Tuple[int, int], int] = {}
        self.active_buckets: Set[Tuple[int, int]] = set()
        
        # Results
        self.kernel_candidates: List[Tuple[int, int, "GNF"]] = []
        self.start_time = time.time()
        
        # Precompute symmetric group table
        print(f"GPUTracker initialized:")
        print(f"  Representation: {rep}")
        print(f"  GPU enabled: {self.use_gpu}")
        print(f"  Bucket size: {bucket_size}")
        
        if BRAID_LIB_AVAILABLE:
            print("  Building symmetric table...", end=" ")
            self._sym_table = symmetric_table_gpu(rep, use_gpu=self.use_gpu)
            print(f"done ({len(self._sym_table)} permutations)")
        else:
            self._sym_table = None
        
    def add_braids_images(
        self,
        braids: List["GNF"],
        images: np.ndarray
    ) -> int:
        """Add braids and their images to appropriate buckets."""
        if len(braids) == 0:
            return 0
            
        # Ensure images are on CPU for storage
        images_cpu = gpm.to_cpu(images) if gpm.is_gpu_array(images) else images
        
        # Compute projlens - do on GPU if available
        if self.use_gpu:
            images_gpu = gpm.to_gpu(images_cpu)
            projlens = gpm.to_cpu(gpm.projlen(images_gpu))
        else:
            projlens = gpm.projlen(images_cpu)
            
        # Get Garside lengths
        lengths = np.array([braid.garside_length() for braid in braids])
        
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
                print(f"\n🎉 KERNEL ELEMENT at length {length}!")
                print(f"   Braid: {braid}")
                self.kernel_candidates.append((length, pl, braid))
                
            # Reservoir sampling
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
                self.bucket_braids[bucket].append(braid)
                self.bucket_images[bucket].append(image)
                n_added += 1
            else:
                j = self.rng.randint(1, self.bucket_reservoir_counts[bucket])
                if j <= self.bucket_size:
                    self.bucket_braids[bucket][j-1] = braid
                    self.bucket_images[bucket][j-1] = image
                    
        return n_added
        
    def nf_descendants(self, bucket: Tuple[int, int], suffix_length: int = 1):
        """Compute Garside normal form descendants."""
        if bucket not in self.active_buckets:
            return
            
        braids = self.bucket_braids[bucket]
        images = self.bucket_images[bucket]
        
        if not braids:
            return
            
        # Gather all (index, braid, suffix) tuples
        all_pairs = []
        for i, braid in enumerate(braids):
            for suffix in braid.nf_suffixes(suffix_length):
                all_pairs.append((i, braid, suffix))
                
        if not all_pairs:
            return
            
        # Process in batches
        batch_size = 5000
        
        for batch_start in range(0, len(all_pairs), batch_size):
            batch = all_pairs[batch_start:batch_start + batch_size]
            
            # Pack base images
            base_images_list = [images[idx] for idx, _, _ in batch]
            left = gpm.pack(base_images_list)
            
            if self.use_gpu:
                left = gpm.to_gpu(left)
                
            # Evaluate suffixes at each length
            for k in range(1, suffix_length + 1):
                suffix_braids = [suffix.substring(0, k) for _, _, suffix in batch]
                
                # Use the CPU evaluation function from braidsearch
                from peyl.braidsearch import evaluate_braids_of_same_length
                suffix_images = evaluate_braids_of_same_length(self.rep, suffix_braids)
                
                if self.use_gpu:
                    suffix_images = gpm.to_gpu(suffix_images)
                    
                # Batch multiply
                products = gpm.mul(left, suffix_images, p=self.p)
                products = gpm.projectivise(products)
                
                # Add results
                new_braids = [braid * suffix_k for _, braid, suffix_k in zip(batch, batch, suffix_braids)]
                self.add_braids_images(new_braids, products)
                
    def bootstrap_exhaustive(self, upto_length: int):
        """Bootstrap by exhaustively enumerating all braids up to given length."""
        if not BRAID_LIB_AVAILABLE:
            raise RuntimeError("Braid library not available")
            
        B = BraidGroup(self.n)
        
        # Add identity
        eye = self.rep.id()
        identity = B.identity()
        self.add_braids_images([identity], eye[None, ...])
        
        for length in range(1, upto_length + 1):
            print(f"Bootstrapping length {length}...", end=" ")
            count = 0
            
            # Enumerate in batches
            from peyl.braidsearch import evaluate_braids_of_same_length, batched
            for batch in batched(B.all_of_garside_length(length), 1000):
                images = evaluate_braids_of_same_length(self.rep, batch)
                self.add_braids_images(batch, images)
                count += len(batch)
                
            min_pl = self._min_projlen_at_length(length)
            print(f"added {count}, min_projlen={min_pl}")
            
    def advance_one_step(self, suffix_length: int = 1, verbose: bool = True):
        """Process all current buckets and compute their descendants."""
        if not self.active_buckets:
            return
            
        current_max = max(b[0] for b in self.active_buckets)
        buckets_to_process = [b for b in self.active_buckets if b[0] == current_max]
        
        start = time.time()
        for bucket in buckets_to_process:
            self.nf_descendants(bucket, suffix_length)
            
        elapsed = time.time() - start
        new_length = current_max + 1
        min_pl = self._min_projlen_at_length(new_length)
        
        if verbose:
            total_stored = sum(len(self.bucket_braids[b]) for b in self.active_buckets)
            print(f"Length {new_length}: min_projlen={min_pl}, "
                  f"total_stored={total_stored}, time={elapsed:.1f}s")
                  
    def run_search(
        self,
        bootstrap_length: int = 8,
        max_length: int = 80,
        suffix_length: int = 1,
        verbose: bool = True
    ) -> SearchStats:
        """Run the complete search algorithm."""
        print(f"\n{'='*60}")
        print(f"GPU Burau Kernel Search")
        print(f"n={self.n}, p={self.p}, bucket_size={self.bucket_size}")
        print(f"{'='*60}\n")
        
        # Bootstrap
        print("Phase 1: Exhaustive bootstrap")
        self.bootstrap_exhaustive(bootstrap_length)
        
        # Main search loop
        print(f"\nPhase 2: Reservoir sampling (up to length {max_length})")
        
        for length in range(bootstrap_length + 1, max_length + 1):
            self.advance_one_step(suffix_length, verbose)
            
            if self.kernel_candidates:
                print(f"\n🎉 Found {len(self.kernel_candidates)} kernel element(s)!")
                break
                
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
        
        min_projlen_by_length = {}
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
