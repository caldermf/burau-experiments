from __future__ import annotations

import argparse
import contextlib
import io
import os
import socket
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch

from a3_gpu_search import SearchConfig, run_search
from plot_search_log import parse_log, render_plot


# Edit these defaults only if you want a different standard manual run.
MAX_G_LENGTH = 100
FIRST_STEPS = 12
BASE_POINT = 1
SEED = 0
DEVICE = "auto"
PRINT_WITNESS_LIMIT = 20
MANUAL_RUNS_DIR = Path(__file__).resolve().parent / "manual runs"


class FilteredTee(io.TextIOBase):
    def __init__(self, console: io.TextIOBase, logfile: io.TextIOBase) -> None:
        self.console = console
        self.logfile = logfile
        self._buffer = ""

    def write(self, text: str) -> int:
        self.logfile.write(text)
        self.logfile.flush()
        self._buffer += text

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit_console_line(line + "\n")
        return len(text)

    def flush(self) -> None:
        self.logfile.flush()
        self.console.flush()

    def finalize(self) -> None:
        if self._buffer:
            self._emit_console_line(self._buffer)
            self._buffer = ""
        self.flush()

    def _emit_console_line(self, line: str) -> None:
        if line.startswith("Finished step "):
            self.console.write(line)
            self.console.flush()


class LiveWitnessWriter:
    def __init__(self, path: Path, config: SearchConfig) -> None:
        self.path = path
        self.handle = path.open("w", encoding="ascii")
        self.handle.write("A3 manual witness log\n")
        self.handle.write("status = running\n")
        self.handle.write(f"started_at = {datetime.now().astimezone().isoformat()}\n")
        self.handle.write(f"host = {socket.gethostname()}\n")
        self.handle.write(f"modulus = {config.modulus}\n")
        self.handle.write(f"base_point = {config.base_point}\n")
        self.handle.write(f"cap = {config.cap_1}\n")
        self.handle.write(f"total_cap = {config.total_cap_1}\n")
        self.handle.write(f"first_steps = {config.first_steps}\n")
        self.handle.write(f"max_g_length = {config.max_g_length}\n")
        self.handle.write(f"seed = {config.seed}\n")
        self.handle.write("\n")
        self.handle.flush()

    def record(self, depth: int, witness: list[list[int]], index: int) -> None:
        flat = [letter for block in witness for letter in block]
        self.handle.write(f"Witness #{index}\n")
        self.handle.write(f"depth = {depth}\n")
        self.handle.write(f"blocks = {witness}\n")
        self.handle.write(f"flat = {flat}\n")
        self.handle.write("\n")
        self.handle.flush()

    def finalize(self, result) -> None:
        depth_hist = dict(sorted(Counter(result.witness_depths).items()))
        self.handle.write("Final summary\n")
        self.handle.write("status = completed\n")
        self.handle.write(f"found = {result.found}\n")
        self.handle.write(f"first_depth = {result.depth}\n")
        self.handle.write(f"unique_witnesses = {len(result.witnesses)}\n")
        self.handle.write(f"total_hits = {result.total_hits}\n")
        self.handle.write(f"witness_depth_histogram = {depth_hist}\n")
        self.handle.flush()

    def finalize_failure(self) -> None:
        self.handle.write("Final summary\n")
        self.handle.write("status = failed\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple manual GPU runner. Usage: python curvemod_run.py P CAP TOTAL_CAP"
    )
    parser.add_argument("p", type=int, help="Modulus.")
    parser.add_argument("cap", type=int, help="Per-spread bucket cap.")
    parser.add_argument("total_cap", type=int, help="Total previous-layer expansion cap.")
    return parser.parse_args()


def build_paths(modulus: int, cap: int, total_cap: int) -> tuple[Path, Path, Path]:
    MANUAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"a3_manual_p{modulus}_b{cap}_u{total_cap}_len{MAX_G_LENGTH}_bp{BASE_POINT}_{timestamp}"
    return (
        MANUAL_RUNS_DIR / f"{stem}.log",
        MANUAL_RUNS_DIR / f"{stem}_witnesses.txt",
        MANUAL_RUNS_DIR / f"{stem}.png",
    )


def write_run_header(handle: io.TextIOBase, modulus: int, cap: int, total_cap: int) -> None:
    handle.write(f"Start: {datetime.now().astimezone().isoformat()}\n")
    handle.write(f"Host: {socket.gethostname()}\n")
    handle.write(f"CUDA visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}\n")
    handle.write(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(unset)')}\n")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total_mib = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        handle.write(f"{name}, {total_mib} MiB\n")
    else:
        handle.write("CUDA unavailable\n")
    handle.write(
        f"Manual config: p={modulus} base_point={BASE_POINT} cap={cap} total_cap={total_cap} "
        f"first_steps={FIRST_STEPS} max_g_length={MAX_G_LENGTH} seed={SEED} device={DEVICE}\n"
    )
    handle.flush()


def main() -> None:
    args = parse_args()
    log_path, witness_path, png_path = build_paths(args.p, args.cap, args.total_cap)

    config = SearchConfig(
        cap_1=args.cap,
        cap_2=args.cap,
        total_cap_1=args.total_cap,
        total_cap_2=args.total_cap,
        first_steps=FIRST_STEPS,
        modulus=args.p,
        max_g_length=MAX_G_LENGTH,
        base_point=BASE_POINT,
        device=DEVICE,
        seed=SEED,
        max_witnesses=None,
        print_witness_limit=PRINT_WITNESS_LIMIT,
    )

    witness_writer = LiveWitnessWriter(witness_path, config)
    config.witness_callback = witness_writer.record

    result = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with log_path.open("w", encoding="ascii") as logfile:
        write_run_header(logfile, args.p, args.cap, args.total_cap)
        tee = FilteredTee(sys.stdout, logfile)

        try:
            with contextlib.redirect_stdout(tee):
                result = run_search(config)
                print("SUMMARY log_path=", log_path)
                print("SUMMARY witness_output=", witness_path)
                print("SUMMARY found=", result.found)
                print("SUMMARY first_depth=", result.depth)
                print("SUMMARY unique_witnesses=", len(result.witnesses))
                print("SUMMARY total_hits=", result.total_hits)
                print("SUMMARY witness_depth_histogram=", dict(sorted(Counter(result.witness_depths).items())))
                if torch.cuda.is_available():
                    print("CUDA max_memory_allocated_bytes:", torch.cuda.max_memory_allocated())
                    print("CUDA max_memory_reserved_bytes:", torch.cuda.max_memory_reserved())
        except Exception:
            traceback.print_exc(file=logfile)
            witness_writer.finalize_failure()
            tee.finalize()
            witness_writer.close()
            raise
        finally:
            tee.finalize()

    witness_writer.finalize(result)
    witness_writer.close()

    parsed = parse_log(log_path)
    render_plot(parsed, png_path, log_path)

    print(f"Done. Log: {log_path}")
    print(f"Witnesses: {witness_path}")
    print(f"PNG: {png_path}")
    print(
        f"Summary: found={result.found} first_depth={result.depth} "
        f"unique_witnesses={len(result.witnesses)} total_hits={result.total_hits}"
    )


if __name__ == "__main__":
    main()
