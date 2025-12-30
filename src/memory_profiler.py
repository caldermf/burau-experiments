"""
Memory profiling for GPU braid search experiments.

Generates:
1. PyTorch memory snapshots (.pickle) for memviz visualization
2. CSV logs for quick plotting
3. Summary statistics

Usage:
    from memory_profiler import MemoryProfiler
    
    profiler = MemoryProfiler(
        output_dir="mem_profiles",
        params={"bucket_size": 100000, "p": 5, ...},
        record_snapshots=True  # Full memviz snapshots
    )
    
    with profiler:
        # ... run your search ...
        profiler.mark("bootstrap_complete")
        # ... more work ...
    
    # Generates: mem_profiles/profile_<timestamp>_<gpu>_b100000_p5.{pickle,csv,txt}
"""

import torch
import time
import csv
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime
import subprocess


@dataclass
class MemoryStats:
    """Memory statistics at a point in time."""
    timestamp: float
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    event: str = ""
    level: int = 0


class MemoryProfiler:
    """
    GPU memory profiler for braid search experiments.
    
    Features:
    - PyTorch memory snapshots for detailed memviz analysis
    - Periodic CSV logging for quick plotting
    - Event markers (level transitions, sampling, etc.)
    - Automatic GPU detection and labeling
    """
    
    def __init__(
        self,
        output_dir: str = "mem_profiles",
        params: Optional[dict] = None,
        record_snapshots: bool = True,
        log_interval_seconds: float = 0.5,
        enabled: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.params = params or {}
        self.record_snapshots = record_snapshots
        self.log_interval = log_interval_seconds
        self.enabled = enabled and torch.cuda.is_available()
        
        self.stats: list[MemoryStats] = []
        self.events: list[tuple[float, str]] = []
        self.start_time: float = 0
        self.current_level: int = 0
        
        # GPU info
        self.gpu_name = "cpu"
        self.gpu_memory_gb = 0
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Build filename suffix from params
        self.file_suffix = self._build_suffix()
        
    def _build_suffix(self) -> str:
        """Build descriptive filename suffix from params."""
        parts = []
        
        # Sanitize GPU name for filename
        gpu_short = self.gpu_name.replace(" ", "_").replace("-", "_")
        gpu_short = "".join(c for c in gpu_short if c.isalnum() or c == "_")[:20]
        parts.append(gpu_short)
        
        # Key params
        if "bucket_size" in self.params:
            parts.append(f"b{self.params['bucket_size']}")
        if "prime" in self.params or "p" in self.params:
            p = self.params.get("prime") or self.params.get("p")
            parts.append(f"p{p}")
        if "use_best" in self.params and self.params["use_best"] > 0:
            parts.append(f"ub{self.params['use_best']}")
        if "max_length" in self.params:
            parts.append(f"ml{self.params['max_length']}")
            
        return "_".join(parts)
    
    def _get_output_prefix(self) -> str:
        """Get output file prefix with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"profile_{timestamp}_{self.file_suffix}"
    
    def _record_stats(self, event: str = ""):
        """Record current memory statistics."""
        if not self.enabled:
            return
            
        stats = MemoryStats(
            timestamp=time.time() - self.start_time,
            allocated_mb=torch.cuda.memory_allocated() / 1e6,
            reserved_mb=torch.cuda.memory_reserved() / 1e6,
            max_allocated_mb=torch.cuda.max_memory_allocated() / 1e6,
            event=event,
            level=self.current_level
        )
        self.stats.append(stats)
    
    def __enter__(self):
        """Start profiling."""
        if not self.enabled:
            return self
            
        self.start_time = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Reset peak stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Start recording memory history for snapshots
        if self.record_snapshots:
            try:
                torch.cuda.memory._record_memory_history(
                    max_entries=1000000,
                    context="all"
                )
            except Exception as e:
                print(f"Warning: Could not start memory recording: {e}")
                self.record_snapshots = False
        
        self._record_stats("start")
        print(f"\n📊 Memory profiling started")
        print(f"   GPU: {self.gpu_name} ({self.gpu_memory_gb:.1f} GB)")
        print(f"   Output: {self.output_dir}/")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling and save outputs."""
        if not self.enabled:
            return
            
        self._record_stats("end")
        
        prefix = self._get_output_prefix()
        
        # Save memory snapshot for memviz
        if self.record_snapshots:
            try:
                snapshot_path = self.output_dir / f"{prefix}.pickle"
                torch.cuda.memory._dump_snapshot(str(snapshot_path))
                print(f"   Snapshot: {snapshot_path}")
            except Exception as e:
                print(f"   Warning: Could not save snapshot: {e}")
            finally:
                torch.cuda.memory._record_memory_history(enabled=None)
        
        # Save CSV log
        csv_path = self.output_dir / f"{prefix}.csv"
        self._save_csv(csv_path)
        print(f"   CSV log: {csv_path}")
        
        # Save summary
        summary_path = self.output_dir / f"{prefix}_summary.txt"
        self._save_summary(summary_path)
        print(f"   Summary: {summary_path}")
        
        print(f"📊 Memory profiling complete\n")
    
    def mark(self, event: str):
        """Mark an event (shows up in timeline)."""
        if not self.enabled:
            return
        self._record_stats(event)
        self.events.append((time.time() - self.start_time, event))
    
    def set_level(self, level: int):
        """Update current level and record stats."""
        self.current_level = level
        self.mark(f"level_{level}")
    
    def log_periodic(self):
        """Call this periodically in your main loop to log stats."""
        if not self.enabled:
            return
        if not self.stats or (time.time() - self.start_time - self.stats[-1].timestamp) >= self.log_interval:
            self._record_stats()
    
    def _save_csv(self, path: Path):
        """Save stats to CSV."""
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_s", "allocated_mb", "reserved_mb", 
                "max_allocated_mb", "level", "event"
            ])
            for s in self.stats:
                writer.writerow([
                    f"{s.timestamp:.3f}",
                    f"{s.allocated_mb:.1f}",
                    f"{s.reserved_mb:.1f}",
                    f"{s.max_allocated_mb:.1f}",
                    s.level,
                    s.event
                ])
    
    def _save_summary(self, path: Path):
        """Save human-readable summary."""
        with open(path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MEMORY PROFILE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("GPU INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Device: {self.gpu_name}\n")
            f.write(f"Total Memory: {self.gpu_memory_gb:.2f} GB\n\n")
            
            f.write("PARAMETERS\n")
            f.write("-" * 40 + "\n")
            for k, v in sorted(self.params.items()):
                f.write(f"{k}: {v}\n")
            f.write("\n")
            
            if self.stats:
                f.write("MEMORY USAGE\n")
                f.write("-" * 40 + "\n")
                peak_allocated = max(s.allocated_mb for s in self.stats)
                peak_reserved = max(s.reserved_mb for s in self.stats)
                final_allocated = self.stats[-1].allocated_mb
                
                f.write(f"Peak Allocated: {peak_allocated:.1f} MB ({peak_allocated/1000:.2f} GB)\n")
                f.write(f"Peak Reserved:  {peak_reserved:.1f} MB ({peak_reserved/1000:.2f} GB)\n")
                f.write(f"Final Allocated: {final_allocated:.1f} MB\n")
                f.write(f"Utilization: {peak_allocated / (self.gpu_memory_gb * 1000) * 100:.1f}%\n\n")
                
                f.write("EVENTS\n")
                f.write("-" * 40 + "\n")
                for t, event in self.events[:50]:  # First 50 events
                    f.write(f"  {t:8.2f}s  {event}\n")
                if len(self.events) > 50:
                    f.write(f"  ... and {len(self.events) - 50} more events\n")


def get_memory_stats_dict() -> dict:
    """Get current memory stats as a dict (useful for logging)."""
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1e6,
        "reserved_mb": torch.cuda.memory_reserved() / 1e6,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
        "device": torch.cuda.get_device_name(0)
    }


def print_memory_status(prefix: str = ""):
    """Print current GPU memory status."""
    if not torch.cuda.is_available():
        print(f"{prefix}Device: CPU")
        return
        
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"{prefix}GPU Memory: {allocated:.2f} GB allocated, "
          f"{reserved:.2f} GB reserved, {total:.2f} GB total "
          f"({allocated/total*100:.1f}% used)")
