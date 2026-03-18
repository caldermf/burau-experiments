from __future__ import annotations

import argparse
import ast
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


FINISHED_RE = re.compile(
    r"^Finished step (?P<depth>\d+), minimal spread (?P<min>\d+), max spread (?P<max>\d+), got (?P<drops>\d+) drops$"
)
TOTAL_RE = re.compile(r"^Total number of curves (?P<total>\d+)$")
WITNESS_RE = re.compile(r"^Found witness #\d+ for p=(?P<p>\d+) at depth (?P<depth>\d+):")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a PNG summary from a saved A3 search log.")
    parser.add_argument("log_path", help="Path to the saved log file.")
    parser.add_argument("--output", help="Optional output PNG path. Defaults to log path with .png suffix.")
    return parser.parse_args()


def parse_log(path: Path) -> dict:
    depths: list[int] = []
    min_spreads: list[int] = []
    max_spreads: list[int] = []
    totals: list[int] = []
    drops: list[int] = []
    witness_depths: list[int] = []
    witness_hist: dict[int, int] | None = None
    modulus: int | None = None
    header: str | None = None

    pending_total_depth: int | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("Starting A3 torch search with p="):
            header = line
            parts = line.split()
            try:
                modulus = int(parts[6])
            except (IndexError, ValueError):
                modulus = None

        finished_match = FINISHED_RE.match(line)
        if finished_match:
            depth = int(finished_match.group("depth"))
            depths.append(depth)
            min_spreads.append(int(finished_match.group("min")))
            max_spreads.append(int(finished_match.group("max")))
            drops.append(int(finished_match.group("drops")))
            pending_total_depth = depth
            continue

        total_match = TOTAL_RE.match(line)
        if total_match and pending_total_depth is not None:
            totals.append(int(total_match.group("total")))
            pending_total_depth = None
            continue

        witness_match = WITNESS_RE.match(line)
        if witness_match:
            witness_depths.append(int(witness_match.group("depth")))
            if modulus is None:
                modulus = int(witness_match.group("p"))
            continue

        if line.startswith("SUMMARY witness_depth_histogram="):
            _, payload = line.split("=", 1)
            parsed = ast.literal_eval(payload.strip())
            witness_hist = {int(k): int(v) for k, v in parsed.items()}

    if witness_hist is None:
        witness_hist = dict(sorted(Counter(witness_depths).items()))

    if len(depths) != len(totals):
        raise ValueError(f"Mismatch between finished-step count ({len(depths)}) and total-count lines ({len(totals)}).")

    return {
        "depths": depths,
        "min_spreads": min_spreads,
        "max_spreads": max_spreads,
        "totals": totals,
        "drops": drops,
        "witness_hist": witness_hist,
        "modulus": modulus,
        "header": header,
    }


def render_plot(parsed: dict, output_path: Path, log_path: Path) -> None:
    depths = parsed["depths"]
    min_spreads = parsed["min_spreads"]
    max_spreads = parsed["max_spreads"]
    totals = parsed["totals"]
    witness_hist = parsed["witness_hist"]
    modulus = parsed["modulus"]

    fig, axes = plt.subplots(3, 1, figsize=(13.75, 12.5), dpi=144, sharex=True)
    fig.patch.set_facecolor("#f6f2ea")

    for ax in axes:
        ax.set_facecolor("#fffaf2")
        ax.grid(True, alpha=0.22, linewidth=0.8)

    axes[0].plot(depths, totals, color="#005f73", linewidth=2.2)
    axes[0].fill_between(depths, totals, color="#0a9396", alpha=0.18)
    axes[0].set_ylabel("Live curves")
    axes[0].set_title(f"A3 search summary mod {modulus}" if modulus is not None else "A3 search summary")

    axes[1].plot(depths, min_spreads, color="#bb3e03", linewidth=2.0, label="min spread")
    axes[1].plot(depths, max_spreads, color="#ca6702", linewidth=2.0, label="max spread")
    axes[1].fill_between(depths, min_spreads, max_spreads, color="#ee9b00", alpha=0.18)
    axes[1].set_ylabel("Spread")
    axes[1].legend(loc="upper left")

    if witness_hist:
        w_depths = sorted(witness_hist)
        w_counts = [witness_hist[d] for d in w_depths]
        axes[2].bar(w_depths, w_counts, width=0.9, color="#9b2226")
        axes[2].set_ylabel("Witnesses")
    else:
        axes[2].text(
            0.5,
            0.5,
            "No witnesses found",
            ha="center",
            va="center",
            fontsize=18,
            color="#9b2226",
            transform=axes[2].transAxes,
        )
        axes[2].set_yticks([])
    axes[2].set_xlabel("Depth")

    summary_lines = [
        f"log: {log_path.name}",
        f"peak frontier: {max(totals):,}" if totals else "peak frontier: n/a",
        f"first witness depth: {min(witness_hist) if witness_hist else 'none'}",
        f"total witnesses: {sum(witness_hist.values()):,}",
    ]
    fig.text(0.02, 0.012, " | ".join(summary_lines), fontsize=9, color="#444444")

    plt.tight_layout(rect=(0, 0.03, 1, 0.985))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_path)
    output_path = Path(args.output) if args.output else log_path.with_suffix(".png")
    parsed = parse_log(log_path)
    render_plot(parsed, output_path, log_path)
    print(output_path)


if __name__ == "__main__":
    main()
