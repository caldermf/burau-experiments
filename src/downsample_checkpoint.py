#!/usr/bin/env python3
"""
Downsample a checkpoint to reduce memory usage.

This reduces bucket_size so the checkpoint can be loaded on smaller GPUs.
"""

import torch
import argparse
from pathlib import Path


def downsample_checkpoint(input_path: str, output_path: str, new_bucket_size: int):
    """
    Load a checkpoint and reduce the number of braids per bucket.
    
    Args:
        input_path: Path to original checkpoint
        output_path: Path to save downsampled checkpoint
        new_bucket_size: Max braids to keep per projlen bucket
    """
    print(f"Loading checkpoint from {input_path}...")
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    
    level = checkpoint['level']
    config = checkpoint.get('config', {})
    old_bucket_size = config.get('bucket_size', 'unknown')
    
    print(f"  Level: {level}")
    print(f"  Original bucket_size: {old_bucket_size}")
    print(f"  New bucket_size: {new_bucket_size}")
    
    # Count original braids
    old_total = 0
    for pl, bucket_data in checkpoint['buckets'].items():
        if isinstance(bucket_data, tuple):
            mat, words, lengths = bucket_data
        else:
            mat = bucket_data['matrices']
        old_total += len(mat)
    
    print(f"  Original total braids: {old_total}")
    
    # Estimate memory
    sample_bucket = list(checkpoint['buckets'].values())[0]
    if isinstance(sample_bucket, tuple):
        sample_mat = sample_bucket[0]
    else:
        sample_mat = sample_bucket['matrices']
    
    D = sample_mat.shape[-1]
    bytes_per_matrix = sample_mat.numel() // len(sample_mat) * sample_mat.element_size()
    
    print(f"  D (degree window): {D}")
    print(f"  Bytes per matrix: {bytes_per_matrix:,}")
    
    # Downsample each bucket
    new_buckets = {}
    new_total = 0
    
    for pl, bucket_data in checkpoint['buckets'].items():
        if isinstance(bucket_data, tuple):
            mat, words, lengths = bucket_data
        else:
            mat = bucket_data['matrices']
            words = bucket_data['words']
            lengths = bucket_data['lengths']
        
        n = len(mat)
        
        if n <= new_bucket_size:
            # Keep all
            new_buckets[pl] = (mat, words, lengths)
            new_total += n
        else:
            # Random sample
            idx = torch.randperm(n)[:new_bucket_size]
            new_buckets[pl] = (mat[idx], words[idx], lengths[idx])
            new_total += new_bucket_size
    
    print(f"  New total braids: {new_total}")
    print(f"  Reduction: {old_total} -> {new_total} ({100*new_total/old_total:.1f}%)")
    
    # Estimate new memory usage
    new_memory_gb = new_total * bytes_per_matrix / 1e9
    print(f"  Estimated GPU memory for matrices: {new_memory_gb:.1f} GB")
    
    # Update checkpoint
    checkpoint['buckets'] = new_buckets
    checkpoint['config']['bucket_size'] = new_bucket_size
    
    # Save
    print(f"\nSaving to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    
    # Report file sizes
    import os
    old_size = os.path.getsize(input_path) / 1e9
    new_size = os.path.getsize(output_path) / 1e9
    print(f"  File size: {old_size:.2f} GB -> {new_size:.2f} GB")
    
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Downsample a checkpoint to fit on smaller GPUs"
    )
    parser.add_argument("input", type=str, help="Path to input checkpoint")
    parser.add_argument("output", type=str, help="Path to output checkpoint")
    parser.add_argument("--bucket-size", "-b", type=int, default=10000,
                        help="New max braids per bucket (default: 10000)")
    
    args = parser.parse_args()
    
    downsample_checkpoint(args.input, args.output, args.bucket_size)


if __name__ == "__main__":
    main()
