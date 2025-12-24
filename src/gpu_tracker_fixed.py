"""
GPU-accelerated Tracker for Burau kernel element search.
"""

import functools
import random
import sys
import time
from typing import Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass

import numpy as np

# Import GPU polymat operations
import gpu_polymat as gpm

# Import from the braid library - use lazy imports to handle path issues
def _import_braid_lib():
    """Lazy import of braid library components. Tries multiple strategies."""
    import sys
    from pathlib import Path
    
    # Strategy 0: Check if already imported
    if 'peyl' in sys.modules:
        try:
            from peyl.braid import GNF, DGNF, BraidGroup
            from peyl.jonesrep import JonesSummand, JonesCellRep
            from peyl.permutations import SymmetricGroup
            from peyl import polymat
            return {
                'GNF': GNF,
                'DGNF': DGNF,
                'BraidGroup': BraidGroup,
                'JonesSummand': JonesSummand,
                'JonesCellRep': JonesCellRep,
                'SymmetricGroup': SymmetricGroup,
                'polymat': polymat,
                'available': True
            }
        except (ImportError, AttributeError):
            pass
    
    # Strategy 1: Try direct import
    try:
        from peyl.braid import GNF, DGNF, BraidGroup
        from peyl.jonesrep import JonesCellRep
        from peyl.braidsearch import JonesSummand
        from peyl.permutations import SymmetricGroup
        from peyl import polymat
        return {
            'GNF': GNF,
            'DGNF': DGNF,
            'BraidGroup': BraidGroup,
            'JonesSummand': JonesSummand,
            'JonesCellRep': JonesCellRep,
            'SymmetricGroup': SymmetricGroup,
            'polymat': polymat,
            'available': True
        }
    except ImportError:
        pass
    
    # Strategy 2: Add parent directory (where peyl/ lives)
    current_file = Path(__file__).resolve()
    parent_dir = current_file.parent.parent
    paths_to_try = [
        parent_dir,
        parent_dir / 'peyl',
        current_file.parent,
    ]
    
    for path_to_add in paths_to_try:
        path_str = str(path_to_add)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        
        try:
            from peyl.braid import GNF, DGNF, BraidGroup
            from peyl.jonesrep import JonesCellRep
            from peyl.braidsearch import JonesSummand
            from peyl.permutations import SymmetricGroup
            from peyl import polymat
            return {
                'GNF': GNF,
                'DGNF': DGNF,
                'BraidGroup': BraidGroup,
                'JonesSummand': JonesSummand,
                'JonesCellRep': JonesCellRep,
                'SymmetricGroup': SymmetricGroup,
                'polymat': polymat,
                'available': True
            }
        except ImportError:
            continue
    
    # Strategy 3: Try importing from peyl package if it's installed
    try:
        import peyl
        from peyl.braid import GNF, DGNF, BraidGroup
        from peyl.jonesrep import JonesCellRep
        from peyl.braidsearch import JonesSummand
        from peyl.permutations import SymmetricGroup
        from peyl import polymat
        return {
            'GNF': GNF,
            'DGNF': DGNF,
            'BraidGroup': BraidGroup,
            'JonesSummand': JonesSummand,
            'JonesCellRep': JonesCellRep,
            'SymmetricGroup': SymmetricGroup,
            'polymat': polymat,
            'available': True
        }
    except ImportError:
        pass
    
    # All strategies failed
    return {'available': False}

# Cache the imports
_braid_lib = None

def _get_braid_lib():
    """Get braid library imports, caching the result."""
    global _braid_lib
    if _braid_lib is None:
        _braid_lib = _import_braid_lib()
    return _braid_lib


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


def symmetric_table_gpu(rep, use_gpu: bool = True):
    """Build the symmetric table, optionally on GPU."""
    braid_lib = _get_braid_lib()
    if not braid_lib['available']:
        raise RuntimeError("Braid library not available - cannot build symmetric table")
    
    SymmetricGroup = braid_lib['SymmetricGroup']
    polymat = braid_lib['polymat']
    
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


def evaluate_braids_of_same_length_gpu(rep, braids: List, sym_table: Dict, use_gpu: bool = True, return_gpu: bool = False) -> np.ndarray:
    """
    GPU-accelerated version of evaluate_braids_of_same_length.
    This is a critical hot path that was previously calling CPU functions.
    
    Args:
        rep: Representation object
        braids: List of braids to evaluate
        sym_table: Precomputed symmetric table
        use_gpu: Whether to use GPU acceleration
        return_gpu: If True, return GPU array (for further GPU operations)
    
    Returns:
        Array of evaluated braid images (on CPU unless return_gpu=True)
    """
    braid_lib = _get_braid_lib()
    if not braid_lib['available']:
        raise RuntimeError("Braid library not available")
    
    SymmetricGroup = braid_lib['SymmetricGroup']
    
    if not braids:
        result = np.zeros((0, rep.dimension(), rep.dimension(), 1), dtype=rep.polymat_dtype())
        return gpm.to_gpu(result) if (use_gpu and gpm.GPU_AVAILABLE and return_gpu) else result
    
    length = braids[0].garside_length()
    assert all(braid.garside_length() == length for braid in braids)
    
    # Get factors for all braids
    factors = [braid.canonical_factors() for braid in braids]
    w0 = SymmetricGroup(rep.n).longest_element()
    eye = rep.id()
    anti = sym_table[w0]
    
    # Start with identity or anti based on inf
    # Build initial images list
    initial_images = []
    for braid in braids:
        img = eye if braid.inf() % 2 == 0 else anti
        # Convert to CPU numpy if on GPU, for packing
        if gpm.is_gpu_array(img):
            img = gpm.to_cpu(img)
        initial_images.append(img)
    
    images = gpm.pack(initial_images)
    if use_gpu and gpm.GPU_AVAILABLE:
        images = gpm.to_gpu(images)
    
    # Multiply by each factor
    for l in range(length):
        # Get permutation images for this level
        perm_images_list = [sym_table[factors[i][l]] for i in range(len(braids))]
        
        # Convert all to CPU for packing, then back to GPU if needed
        perm_images_cpu = [gpm.to_cpu(img) if gpm.is_gpu_array(img) else img for img in perm_images_list]
        perm_images = gpm.pack(perm_images_cpu)
        
        if use_gpu and gpm.GPU_AVAILABLE:
            perm_images = gpm.to_gpu(perm_images)
        
        # Multiply: images = images * perm_images
        images = gpm.mul(images, perm_images, p=rep.p)
        images = gpm.projectivise(images)
    
    # Return on GPU if requested, otherwise CPU
    if use_gpu and gpm.GPU_AVAILABLE and return_gpu:
        return images
    return gpm.to_cpu(images) if gpm.is_gpu_array(images) else images


class GPUTracker:
    """GPU-accelerated version of the reservoir sampling tracker."""
    
    def __init__(
        self,
        rep,
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
        self.dtype = rep.polymat_dtype()
        
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
        
        # Get braid library - try the lazy import first
        braid_lib = _get_braid_lib()
        
        # If not available, but we have a rep object from peyl, import directly
        # Since rep is a peyl.jonesrep.JonesCellRep, peyl must be importable
        if not braid_lib['available']:
            rep_module = rep.__class__.__module__
            if 'peyl' in rep_module:
                # Import directly - we know peyl works since rep came from there
                try:
                    import importlib
                    # Get the module that contains the rep class
                    rep_class_module = importlib.import_module(rep_module)
                    # Now import what we need
                    peyl_braid = importlib.import_module('peyl.braid')
                    peyl_braidsearch = importlib.import_module('peyl.braidsearch')
                    peyl_permutations = importlib.import_module('peyl.permutations')
                    peyl_polymat = importlib.import_module('peyl.polymat')
                    
                    braid_lib = {
                        'GNF': peyl_braid.GNF,
                        'DGNF': peyl_braid.DGNF,
                        'BraidGroup': peyl_braid.BraidGroup,
                        'JonesSummand': getattr(peyl_braidsearch, 'JonesSummand', None),
                        'JonesCellRep': rep_class_module.JonesCellRep,
                        'SymmetricGroup': peyl_permutations.SymmetricGroup,
                        'polymat': peyl_polymat,
                        'available': True
                    }
                    # Cache it
                    global _braid_lib
                    _braid_lib = braid_lib
                except Exception as e:
                    # Last resort: try standard imports
                    try:
                        from peyl.braid import GNF, DGNF, BraidGroup
                        from peyl.braidsearch import JonesSummand
                        from peyl.permutations import SymmetricGroup
                        from peyl import polymat
                        braid_lib = {
                            'GNF': GNF,
                            'DGNF': DGNF,
                            'BraidGroup': BraidGroup,
                            'JonesSummand': JonesSummand,
                            'JonesCellRep': type(rep),  # Use the rep's class
                            'SymmetricGroup': SymmetricGroup,
                            'polymat': polymat,
                            'available': True
                        }
                        global _braid_lib
                        _braid_lib = braid_lib
                    except Exception as e2:
                        raise RuntimeError(
                            f"Failed to import braid library. "
                            f"Rep is from {rep_module}, but imports failed: {e}, {e2}"
                        )
        
        if not braid_lib['available']:
            raise RuntimeError(
                f"Braid library (peyl) not available. "
                f"Rep object type: {type(rep)}, module: {getattr(rep.__class__, '__module__', 'unknown')}."
            )
        
        SymmetricGroup = braid_lib['SymmetricGroup']
        print("  Building symmetric table...", end=" ")
        self._sym_table = symmetric_table_gpu(rep, use_gpu=self.use_gpu)
        print(f"done ({len(self._sym_table)} permutations)")
        
        # Precompute w0 for efficiency
        self._w0 = SymmetricGroup(rep.n).longest_element()
        self._eye = rep.id()
        if self.use_gpu and gpm.GPU_AVAILABLE:
            self._eye_gpu = gpm.to_gpu(self._eye)
            self._anti_gpu = gpm.to_gpu(self._sym_table[self._w0])
        
    def add_braids_images(
        self,
        braids: List["GNF"],
        images: np.ndarray
    ) -> int:
        """Add braids and their images to appropriate buckets."""
        if len(braids) == 0:
            return 0
        
        # Keep images on GPU for computation if possible
        if self.use_gpu and gpm.GPU_AVAILABLE:
            if not gpm.is_gpu_array(images):
                images_gpu = gpm.to_gpu(images)
            else:
                images_gpu = images
            # Compute projlens on GPU (much faster)
            projlens = gpm.to_cpu(gpm.projlen(images_gpu))
            # Convert to CPU only once at the end
            images_cpu = gpm.to_cpu(images_gpu)
        else:
            images_cpu = gpm.to_cpu(images) if gpm.is_gpu_array(images) else images
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
            
        # Process in batches - use larger batches for GPU efficiency
        # For GPU, larger batches are better; for CPU, smaller is fine
        batch_size = 10000 if self.use_gpu else 5000
        
        for batch_start in range(0, len(all_pairs), batch_size):
            batch = all_pairs[batch_start:batch_start + batch_size]
            
            # Pack base images - keep on GPU if possible
            base_images_list = [images[idx] for idx, _, _ in batch]
            left = gpm.pack(base_images_list)
            
            if self.use_gpu and gpm.GPU_AVAILABLE:
                left = gpm.to_gpu(left)
                
            # Evaluate suffixes at each length
            for k in range(1, suffix_length + 1):
                suffix_braids = [suffix.substring(0, k) for _, _, suffix in batch]
                
                # Use GPU-accelerated evaluation - keep on GPU if we're using GPU
                suffix_images = evaluate_braids_of_same_length_gpu(
                    self.rep, suffix_braids, self._sym_table, 
                    use_gpu=self.use_gpu, return_gpu=(self.use_gpu and gpm.GPU_AVAILABLE)
                )
                
                # Ensure both are on GPU for multiplication
                if self.use_gpu and gpm.GPU_AVAILABLE:
                    if not gpm.is_gpu_array(left):
                        left = gpm.to_gpu(left)
                    if not gpm.is_gpu_array(suffix_images):
                        suffix_images = gpm.to_gpu(suffix_images)
                    
                # Batch multiply (both should already be on GPU if use_gpu is True)
                products = gpm.mul(left, suffix_images, p=self.p)
                products = gpm.projectivise(products)
                
                # Add results - this will handle CPU conversion internally
                new_braids = [braid * suffix_k for _, braid, suffix_k in zip(batch, batch, suffix_braids)]
                self.add_braids_images(new_braids, products)
                
    def bootstrap_exhaustive(self, upto_length: int):
        """Bootstrap by exhaustively enumerating all braids up to given length."""
        braid_lib = _get_braid_lib()
        if not braid_lib['available']:
            raise RuntimeError("Braid library not available")
        
        BraidGroup = braid_lib['BraidGroup']
        B = BraidGroup(self.n)
        
        # Add identity
        eye = self.rep.id()
        identity = B.identity()
        self.add_braids_images([identity], eye[None, ...])
        
        for length in range(1, upto_length + 1):
            print(f"Bootstrapping length {length}...", end=" ")
            count = 0
            
            # Enumerate in batches - use larger batches for GPU
            # Import batched function with fallback
            try:
                from peyl.braidsearch import batched
            except ImportError:
                # Fallback implementation
                def batched(iterable, chunk_length):
                    chunk = []
                    for x in iterable:
                        chunk.append(x)
                        if len(chunk) == chunk_length:
                            yield chunk
                            chunk = []
                    if chunk:
                        yield chunk
            
            batch_size = 5000 if self.use_gpu else 1000
            for batch in batched(B.all_of_garside_length(length), batch_size):
                # Use GPU-accelerated evaluation
                images = evaluate_braids_of_same_length_gpu(
                    self.rep, batch, self._sym_table, use_gpu=self.use_gpu
                )
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
