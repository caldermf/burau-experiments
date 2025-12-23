"""
GPU-accelerated Burau kernel element search.

This package provides CUDA-accelerated implementations of the reservoir
sampling algorithm for finding kernel elements in the Burau representation.

Usage:
    from gpu_burau import GPUTracker
    from gpu_burau import gpu_polymat as gpm
    
    # Or run from command line:
    # python -m gpu_burau.run_search --p 5
"""

from . import gpu_polymat
from .gpu_tracker import GPUTracker, SearchStats

__version__ = "0.1.0"
__all__ = [
    "gpu_polymat",
    "GPUTracker",
    "SearchStats",
]
