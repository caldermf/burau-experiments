"""
Aggregate results from multiple parallel Burau kernel searches.

Usage:
    python aggregate_results.py --input-dir /path/to/results --output combined.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict
import glob


def load_results(filepath: str) -> Dict:
    """Load results from a JSON file."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return {}


def aggregate_results(input_dir: str) -> Dict:
    """
    Aggregate results from all search files in a directory.
    
    Looks for files matching pattern: search_*.json
    """
    result_files = glob.glob(os.path.join(input_dir, "search_*.json"))
    result_files += glob.glob(os.path.join(input_dir, "results_*.json"))
    
    if not result_files:
        print(f"No result files found in {input_dir}")
        return {}
    
    print(f"Found {len(result_files)} result files")
    
    all_results = [load_results(f) for f in result_files]
    all_results = [r for r in all_results if r]  # Filter empty
    
    if not all_results:
        return {}
    
    # Aggregate kernel candidates
    all_kernel_candidates = []
    for r in all_results:
        candidates = r.get('kernel_candidates', [])
        all_kernel_candidates.extend(candidates)
    
    # Sort by Garside length
    all_kernel_candidates.sort(key=lambda x: x.get('length', float('inf')))
    
    # Deduplicate by braid string
    seen_braids = set()
    unique_candidates = []
    for c in all_kernel_candidates:
        braid_str = c.get('braid', '')
        if braid_str not in seen_braids:
            seen_braids.add(braid_str)
            unique_candidates.append(c)
    
    # Compute aggregate statistics
    total_time = sum(r.get('total_time', 0) for r in all_results)
    max_length = max(r.get('max_length_searched', 0) for r in all_results)
    n_searches = len(all_results)
    
    return {
        'n_searches': n_searches,
        'total_compute_time_seconds': total_time,
        'max_length_searched': max_length,
        'n_kernel_candidates': len(unique_candidates),
        'kernel_candidates': unique_candidates,
        'source_files': result_files
    }


def print_summary(results: Dict):
    """Print a human-readable summary."""
    print("\n" + "="*60)
    print("SEARCH RESULTS SUMMARY")
    print("="*60)
    
    print(f"Number of searches: {results.get('n_searches', 0)}")
    print(f"Total compute time: {results.get('total_compute_time_seconds', 0)/3600:.2f} hours")
    print(f"Max Garside length searched: {results.get('max_length_searched', 0)}")
    print(f"Kernel candidates found: {results.get('n_kernel_candidates', 0)}")
    
    candidates = results.get('kernel_candidates', [])
    if candidates:
        print("\nKernel Candidates:")
        print("-"*40)
        for i, c in enumerate(candidates[:10]):  # Show first 10
            print(f"  {i+1}. Length {c.get('length')}, projlen {c.get('projlen')}")
            braid = c.get('braid', '')
            if len(braid) > 60:
                braid = braid[:60] + "..."
            print(f"      {braid}")
        if len(candidates) > 10:
            print(f"  ... and {len(candidates) - 10} more")
    else:
        print("\nNo kernel candidates found.")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate parallel search results")
    parser.add_argument("--input-dir", required=True, help="Directory containing result files")
    parser.add_argument("--output", default="combined_results.json", help="Output file path")
    args = parser.parse_args()
    
    results = aggregate_results(args.input_dir)
    
    if results:
        # Save combined results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Combined results saved to {args.output}")
        
        # Print summary
        print_summary(results)
    else:
        print("No results to aggregate")


if __name__ == "__main__":
    main()
