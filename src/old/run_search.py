#!/usr/bin/env python3
"""
GPU-accelerated search for Burau kernel elements.

This is the main entry point for running the search algorithm described in
"4-strand Burau is unfaithful modulo 5" by Gibson, Williamson, and Yacobi.

Usage:
    # Basic search for n=4, p=5
    python run_search.py --p 5
    
    # Search with different parameters
    python run_search.py --p 7 --bucket-size 1000 --max-length 100
    
    # Run on specific GPU
    CUDA_VISIBLE_DEVICES=0 python run_search.py --p 5
    
    # Multiple parallel runs (recommended)
    for i in {1..8}; do
        python run_search.py --p 5 --seed $i --output results_$i.json &
    done
"""

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated Burau kernel element search"
    )
    
    # Representation parameters
    parser.add_argument("--n", type=int, default=4,
                       help="Number of braid strands (default: 4)")
    parser.add_argument("--r", type=int, default=1,
                       help="Representation parameter (default: 1 for Burau)")
    parser.add_argument("--p", type=int, default=5,
                       help="Prime modulus (default: 5, use 0 for exact)")
    
    # Search parameters
    parser.add_argument("--bucket-size", type=int, default=500,
                       help="Max elements per bucket (default: 500)")
    parser.add_argument("--bootstrap-length", type=int, default=8,
                       help="Exhaustively enumerate up to this length (default: 8)")
    parser.add_argument("--max-length", type=int, default=80,
                       help="Maximum Garside length to search (default: 80)")
    parser.add_argument("--suffix-length", type=int, default=1,
                       help="Garside suffix length per step (default: 1)")
    
    # Runtime options
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="results.json",
                       help="Output file path (default: results.json)")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU acceleration")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    return parser.parse_args()


def print_gpu_info():
    """Print GPU information."""
    try:
        import cupy as cp
        print(f"CuPy version: {cp.__version__}")
        
        n_devices = cp.cuda.runtime.getDeviceCount()
        print(f"GPU devices: {n_devices}")
        
        for i in range(n_devices):
            props = cp.cuda.runtime.getDeviceProperties(i)
            mem_free, mem_total = cp.cuda.runtime.memGetInfo()
            print(f"  [{i}] {props['name'].decode()}: "
                  f"{mem_free/1e9:.1f}/{mem_total/1e9:.1f} GB free")
                  
        return True
    except ImportError:
        print("CuPy not available - running on CPU only")
        return False
    except Exception as e:
        print(f"GPU error: {e}")
        return False


def main():
    args = parse_args()
    
    print("="*60)
    print("GPU-Accelerated Burau Kernel Element Search")
    print("="*60)
    print()
    
    # Print configuration
    print("Configuration:")
    print(f"  Braid group: B_{args.n}")
    print(f"  Representation: ({args.n - args.r}, {args.r})")
    print(f"  Prime modulus: {'exact' if args.p == 0 else f'F_{args.p}'}")
    print(f"  Bucket size: {args.bucket_size}")
    print(f"  Max length: {args.max_length}")
    print(f"  Random seed: {args.seed}")
    print()
    
    # Check GPU
    gpu_available = not args.no_gpu and print_gpu_info()
    print()
    
    # Import braid library
    try:
        # Try relative import first (when running as part of package)
        from .jonesrep import JonesCellRep
        from .gpu_tracker import GPUTracker
    except ImportError:
        try:
            # Try adding parent to path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from jonesrep import JonesCellRep
            from gpu_tracker import GPUTracker
        except ImportError:
            # Final fallback - assume files are in same directory
            from jonesrep import JonesCellRep
            from gpu_tracker import GPUTracker
    
    # Create representation
    rep = JonesCellRep(n=args.n, r=args.r, p=args.p)
    print(f"Representation dimension: {rep.dimension()}")
    
    # Create tracker
    tracker = GPUTracker(
        rep=rep,
        bucket_size=args.bucket_size,
        seed=args.seed,
        use_gpu=gpu_available,
    )
    
    # Run search
    print("\nStarting search...")
    start_time = time.time()
    
    try:
        stats = tracker.run_search(
            bootstrap_length=args.bootstrap_length,
            max_length=args.max_length,
            suffix_length=args.suffix_length,
            verbose=not args.quiet,
        )
        
        total_time = time.time() - start_time
        
        # Print results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Braids stored: {stats.total_braids_stored}")
        print(f"Braids seen: {stats.total_braids_seen}")
        print(f"Active buckets: {stats.n_active_buckets}")
        
        if stats.kernel_candidates:
            print(f"\n🎉 KERNEL ELEMENTS FOUND: {len(stats.kernel_candidates)}")
            for i, (length, pl, braid_str) in enumerate(stats.kernel_candidates):
                print(f"\n  [{i+1}] Length {length}, projlen {pl}")
                # Print truncated braid if too long
                if len(braid_str) > 100:
                    print(f"      {braid_str[:100]}...")
                else:
                    print(f"      {braid_str}")
        else:
            print("\nNo kernel elements found in this run.")
            print("Minimum projlen by length (last 10):")
            sorted_lengths = sorted(stats.min_projlen_at_length.keys())[-10:]
            for length in sorted_lengths:
                print(f"  Length {length}: {stats.min_projlen_at_length[length]}")
        
        # Save results
        results = {
            'config': {
                'n': args.n,
                'r': args.r,
                'p': args.p,
                'bucket_size': args.bucket_size,
                'max_length': args.max_length,
                'seed': args.seed,
            },
            'stats': {
                'total_time_seconds': total_time,
                'braids_stored': stats.total_braids_stored,
                'braids_seen': stats.total_braids_seen,
                'active_buckets': stats.n_active_buckets,
                'current_length': stats.current_length,
            },
            'min_projlen_by_length': {str(k): v for k, v in stats.min_projlen_at_length.items()},
            'kernel_candidates': [
                {'length': l, 'projlen': p, 'braid': b}
                for l, p, b in stats.kernel_candidates
            ],
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
        
    except KeyboardInterrupt:
        print("\n\nSearch interrupted by user.")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
