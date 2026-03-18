#!/usr/bin/env python3

import argparse
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from tests.run_exact_modp_validation import pairing_poly, single_whisker


ROOT = Path(__file__).resolve().parent
CHECKER = ROOT / ".exact_modp_check"


@dataclass(frozen=True)
class HitInfo:
    level: int
    tuple5: tuple[int, int, int, int, int]


KNOWN_HITS = [
    HitInfo(543, (376, 22, 144, 143, 400)),
    HitInfo(622, (195, 43, 383, 413, 209)),
    HitInfo(640, (79, 241, 319, 253, 387)),
    HitInfo(696, (257, 90, 348, 560, 136)),
    HitInfo(816, (709, 2, 104, 196, 620)),
    HitInfo(818, (136, 7, 674, 497, 321)),
    HitInfo(933, (395, 165, 372, 333, 600)),
    HitInfo(948, (692, 38, 217, 495, 453)),
    HitInfo(959, (540, 204, 214, 423, 536)),
]


def ensure_checker() -> None:
    if CHECKER.exists():
        return
    subprocess.run(
        ["cc", "-O3", "burau_exact_modp_check.c", "-o", str(CHECKER)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def checker_batch(rows: list[tuple[int, int, int, int, int]]) -> list[tuple[int, int, int]]:
    ensure_checker()
    input_text = "".join(f"3 {a} {b} {c} {d} {e}\n" for a, b, c, d, e in rows)
    proc = subprocess.run(
        [str(CHECKER)],
        cwd=ROOT,
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
    )
    out = []
    for line in proc.stdout.strip().splitlines():
        out.append(tuple(int(x) for x in line.split()))
    return out


def root_multiplicity_at_one(coeffs: list[int]) -> int:
    def eval_at_one(poly: list[int]) -> int:
        return sum(poly)

    def deriv(poly: list[int]) -> list[int]:
        return [(i + 1) * poly[i + 1] for i in range(len(poly) - 1)]

    mult = 0
    cur = coeffs[:]
    while cur and eval_at_one(cur) == 0:
        mult += 1
        cur = deriv(cur)
    return mult


def reciprocity_type(coeffs: list[int]) -> str:
    if coeffs == coeffs[::-1]:
        return "pal"
    if coeffs == [-x for x in coeffs[::-1]]:
        return "anti"
    return "none"


def packet(tuple5: tuple[int, int, int, int, int]) -> tuple[int, list[int]]:
    poly = pairing_poly(*tuple5)
    shift = min(poly)
    coeffs = [poly.get(shift + i, 0) // 3 for i in range(max(poly) - shift + 1)]
    return shift, coeffs


def raw_packet(tuple5: tuple[int, int, int, int, int]) -> tuple[int, list[int]]:
    poly = pairing_poly(*tuple5)
    shift = min(poly)
    coeffs = [poly.get(shift + i, 0) for i in range(max(poly) - shift + 1)]
    return shift, coeffs


def _step_type_data(symbol: str) -> tuple[int, dict[int, int]]:
    down = symbol[:2]
    over = symbol[2:]
    delta = 0
    coeffs: dict[int, int] = defaultdict(int)
    if down == "dL":
        delta += 1
        coeffs[delta] += 1
    elif down == "dR":
        coeffs[delta] -= 1
        delta -= 1
    elif down == "eL":
        coeffs[delta] -= 1
        delta += 1
    elif down == "eR":
        delta -= 1
        coeffs[delta] += 1
    else:
        raise ValueError(symbol)

    if over == "aL":
        delta -= 1
    elif over == "aR":
        delta += 1
    elif over == "bL":
        delta += 4
    elif over == "bR":
        delta -= 4
    elif over == "cL":
        delta -= 1
    elif over == "cR":
        delta += 1
    elif over == "E":
        pass
    else:
        raise ValueError(symbol)
    return delta, dict(coeffs)


def symbolic_walk(tuple5: tuple[int, int, int, int, int]) -> list[tuple[str, int]]:
    a, b, c, d, e = tuple5
    bl = 2 * a
    start = bl + b
    cl = start + b + 1
    end = cl + c
    el = 2 * d
    er = el + e
    suma = bl - 1
    sumb = 2 * start
    sumc = 2 * end
    sumd = el - 1
    sume = er + er - 1
    x = start
    h = 0
    out = []

    while True:
        h_before = h
        if x < el:
            down = "dL" if x < d else "dR"
            x = sumd - x
        else:
            down = "eL" if x < er else "eR"
            x = sume - x

        if x < cl:
            if x < bl:
                over = "aL" if x < a else "aR"
                x = suma - x
            else:
                over = "bL" if x < start else "bR"
                x = sumb - x
        else:
            if x < end:
                over = "cL"
                x = sumc - x
            elif x > end:
                over = "cR"
                x = sumc - x
            else:
                over = "E"
                delta, _ = _step_type_data(down + over)
                out.append((down + over, h_before))
                h += delta
                return out

        delta, _ = _step_type_data(down + over)
        out.append((down + over, h_before))
        h += delta


def relative_histogram(tuple5: tuple[int, int, int, int, int]) -> tuple[int, list[int]]:
    coeffs: dict[int, int] = defaultdict(int)
    for symbol, h0 in symbolic_walk(tuple5):
        _, step_coeffs = _step_type_data(symbol)
        for level, value in step_coeffs.items():
            coeffs[h0 + level] += value
    lo = min(coeffs)
    hi = max(coeffs)
    return lo, [coeffs.get(i, 0) for i in range(lo, hi + 1)]


def trimmed_relative_histogram(tuple5: tuple[int, int, int, int, int]) -> tuple[int, list[int]]:
    lo, coeffs = relative_histogram(tuple5)
    start = 0
    while start < len(coeffs) and coeffs[start] == 0:
        start += 1
    end = len(coeffs) - 1
    while end >= 0 and coeffs[end] == 0:
        end -= 1
    if start > end:
        return lo, []
    return lo + start, coeffs[start : end + 1]


def relative_histogram_edges(tuple5: tuple[int, int, int, int, int]) -> tuple[int, int, int, int]:
    coeffs: dict[int, int] = defaultdict(int)
    for symbol, h0 in symbolic_walk(tuple5):
        _, step_coeffs = _step_type_data(symbol)
        for level, value in step_coeffs.items():
            coeffs[h0 + level] += value
    lo = min(coeffs)
    hi = max(coeffs)
    return lo, coeffs[lo], hi, coeffs[hi]


def trimmed_relative_histogram_edges(tuple5: tuple[int, int, int, int, int]) -> tuple[int, int, int, int]:
    lo, coeffs = trimmed_relative_histogram(tuple5)
    if not coeffs:
        return lo, 0, lo, 0
    hi = lo + len(coeffs) - 1
    return lo, coeffs[0], hi, coeffs[-1]


def half_balance(tuple5: tuple[int, int, int, int, int]) -> tuple[dict[str, int], dict[str, int]]:
    down_counts = Counter()
    over_counts = Counter()
    for symbol, _ in symbolic_walk(tuple5):
        down = symbol[:2]
        over = symbol[2:]
        down_counts[down] += 1
        over_counts[over] += 1
    return dict(down_counts), dict(over_counts)


def block_decomposition(
    tuple5: tuple[int, int, int, int, int]
) -> list[tuple[int, tuple[str, ...], tuple[int, list[int]]]]:
    walk = symbolic_walk(tuple5)
    blocks: list[tuple[int, tuple[str, ...], tuple[int, list[int]]]] = []
    start = 0
    while start < len(walk):
        block_start_h = walk[start][1]
        end = start
        while True:
            symbol, _ = walk[end]
            if symbol[2:] != "aL" and symbol[2:] != "aR":
                block_symbols = tuple(step for step, _ in walk[start : end + 1])
                coeffs: dict[int, int] = defaultdict(int)
                h = 0
                for step_symbol in block_symbols:
                    delta, step_coeffs = _step_type_data(step_symbol)
                    for level, value in step_coeffs.items():
                        coeffs[h + level] += value
                    h += delta
                lo = min(coeffs)
                hi = max(coeffs)
                blocks.append(
                    (block_start_h, block_symbols, (lo, [coeffs.get(i, 0) for i in range(lo, hi + 1)]))
                )
                start = end + 1
                break
            end += 1
    return blocks


def trimmed_relative_edge_sources(
    tuple5: tuple[int, int, int, int, int]
) -> tuple[
    tuple[int, int],
    list[tuple[int, int, int, tuple[int, ...], int]],
    tuple[int, int],
    list[tuple[int, int, int, tuple[int, ...], int]],
]:
    lo, coeffs = trimmed_relative_histogram(tuple5)
    if not coeffs:
        return (lo, 0), [], (lo, 0), []
    hi = lo + len(coeffs) - 1
    left_sources: list[tuple[int, int, int, tuple[int, ...], int]] = []
    right_sources: list[tuple[int, int, int, tuple[int, ...], int]] = []
    for start, block_symbols, (packet_lo, packet_coeffs) in block_decomposition(tuple5):
        for offset, value in enumerate(packet_coeffs):
            if value == 0:
                continue
            level = start + packet_lo + offset
            item = (len(block_symbols), start, packet_lo, tuple(packet_coeffs), value)
            if level == lo:
                left_sources.append(item)
            if level == hi:
                right_sources.append(item)
    return (lo, coeffs[0]), left_sources, (hi, coeffs[-1]), right_sources


def exact_mod3_cost(tuple5: tuple[int, int, int, int, int]) -> tuple[int, int]:
    poly = pairing_poly(*tuple5)
    shift = min(poly)
    coeffs = [poly.get(shift + i, 0) for i in range(max(poly) - shift + 1)]
    bad = sum(c % 3 != 0 for c in coeffs)
    residue_l1 = sum(0 if c % 3 == 0 else 1 for c in coeffs)
    return bad, residue_l1


def reduced_regime(tuple5: tuple[int, int, int, int, int]) -> tuple[str, str, tuple[int, int, int, int]]:
    a, b, c, d, e = tuple5
    if d < a:
        return "a-split", "eE", (d, a - d, b, c)
    if d < a + b:
        return "b-split", "eE", (a, d - a, a + b - d, c)

    # Determine which side the terminal branch uses in the c-split regime.
    bl = 2 * a
    start = bl + b
    cl = start + b + 1
    end = cl + c
    el = 2 * d
    er = el + e
    suma = bl - 1
    sumb = 2 * start
    sumc = 2 * end
    sumd = el - 1
    sume = er + er - 1
    x = start
    endpoint = "eE"
    while True:
        if x < el:
            down = "d"
            x = sumd - x
        else:
            down = "e"
            x = sume - x
        if x < cl:
            x = (suma - x) if x < bl else (sumb - x)
            continue
        if x == end:
            endpoint = f"{down}E"
            break
        x = sumc - x

    if endpoint == "dE":
        return "c-split", endpoint, (a, b, d - a - b - 1, e)
    return "c-split", endpoint, (a, b, d - a - b, e - 1)


def tuple_from_split(kind: str, split: tuple[int, int, int, int]) -> tuple[int, int, int, int, int]:
    if kind == "a":
        x, y, z, w = split
        level = x + y + z + w + 1
        return (x + y, z, w, x, level - x)
    if kind == "b":
        a, u, v, c = split
        return (a, u + v, c, a + u, v + c + 1)
    if kind == "ce":
        a, b, u, v = split
        return (a, b, u + v, a + b + u, v + 1)
    if kind == "cd":
        a, b, u, e = split
        return (a, b, u + e, a + b + u + 1, e)
    raise ValueError(kind)


def classify_search_kind(hit: HitInfo) -> tuple[str, tuple[int, int, int, int]]:
    regime, endpoint, split = reduced_regime(hit.tuple5)
    if regime == "a-split":
        return "a", split
    if regime == "b-split":
        return "b", split
    if endpoint == "eE":
        return "ce", split
    return "cd", split


def analyze_hit(hit: HitInfo) -> str:
    shift, coeffs = packet(hit.tuple5)
    regime, endpoint, split = reduced_regime(hit.tuple5)
    return (
        f"level={hit.level} tuple={hit.tuple5} regime={regime} endpoint={endpoint} "
        f"split={split} shift={shift} span={len(coeffs) - 1} reciprocity={reciprocity_type(coeffs)} "
        f"mult1={root_multiplicity_at_one(coeffs)} coeffs={coeffs}"
    )


def analyze_reciprocity(tuple5: tuple[int, int, int, int, int]) -> str:
    level = sum(tuple5[:3]) + 1
    shift, coeffs = raw_packet(tuple5)
    rel_shift, rel_coeffs = relative_histogram(tuple5)
    trim_rel_shift, trim_rel_coeffs = trimmed_relative_histogram(tuple5)
    left_edge, left_sources, right_edge, right_sources = trimmed_relative_edge_sources(tuple5)
    regime, endpoint, split = reduced_regime(tuple5)
    down_counts, over_counts = half_balance(tuple5)
    blocks = block_decomposition(tuple5)
    block_lengths = Counter(len(block_symbols) for _, block_symbols, _ in blocks)
    block_words = Counter(block_symbols for _, block_symbols, _ in blocks)
    block_packets = Counter((len(block_symbols), data[0], tuple(data[1])) for _, block_symbols, data in blocks)
    lines = [
        f"level={level} tuple={tuple5}",
        f"regime={regime} endpoint={endpoint} split={split}",
        f"packet_shift={shift} packet={coeffs}",
        f"relative_shift={rel_shift} relative_hist={rel_coeffs}",
        f"trimmed_relative_shift={trim_rel_shift} trimmed_relative_hist={trim_rel_coeffs}",
        f"trimmed_edges=left{left_edge} right{right_edge}",
        "left_edge_sources=" + repr(left_sources),
        "right_edge_sources=" + repr(right_sources),
        f"reciprocity={reciprocity_type(coeffs)} mult1={root_multiplicity_at_one(coeffs)}",
        f"down_half_counts={dict(sorted(down_counts.items()))}",
        f"over_half_counts={dict(sorted(over_counts.items()))}",
        f"block_lengths={dict(sorted(block_lengths.items()))}",
        "common_blocks=" + repr(block_words.most_common(8)),
        "common_block_packets=" + repr(block_packets.most_common(8)),
    ]
    return "\n".join(lines)


def search_scaled_family(
    base: HitInfo,
    *,
    scale: int,
    radius: int,
    keep: int,
) -> list[tuple[tuple[int, int, int, int, int], tuple[int, int, int, int], tuple[int, int], str, int]]:
    kind, split0 = classify_search_kind(base)
    target = [scale * x for x in split0]
    total = sum(target)
    rows = []
    splits = []
    for x0 in range(max(0, target[0] - radius), target[0] + radius + 1):
        for x1 in range(max(0, target[1] - radius), target[1] + radius + 1):
            for x2 in range(max(0, target[2] - radius), target[2] + radius + 1):
                x3 = total - x0 - x1 - x2
                if x3 < 0 or abs(x3 - target[3]) > radius:
                    continue
                split = (x0, x1, x2, x3)
                tup = tuple_from_split(kind, split)
                rows.append(tup)
                splits.append(split)

    triples = checker_batch(rows)
    hits = []
    near = []
    for tup, split, triple in zip(rows, splits, triples):
        single, _, exact = triple
        if not single:
            continue
        if exact:
            shift, coeffs = packet(tup)
            hits.append((tup, split, (0, 0), reciprocity_type(coeffs), root_multiplicity_at_one(coeffs)))
            continue
        cost = exact_mod3_cost(tup)
        near.append((cost, tup, split))

    near.sort(key=lambda item: (item[0], item[1]))
    out = hits[:]
    for cost, tup, split in near[:keep]:
        shift, coeffs = packet(tup)
        out.append((tup, split, cost, reciprocity_type(coeffs), root_multiplicity_at_one(coeffs)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--analyze-tuple", type=str, default=None)
    parser.add_argument("--search-base-level", type=int, default=None)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--radius", type=int, default=12)
    parser.add_argument("--keep", type=int, default=12)
    args = parser.parse_args()

    if args.analyze:
        for hit in KNOWN_HITS:
            print(analyze_hit(hit))
        return

    if args.analyze_tuple is not None:
        tup = tuple(int(x.strip()) for x in args.analyze_tuple.split(","))
        if len(tup) != 5:
            raise SystemExit("--analyze-tuple expects five comma-separated integers")
        print(analyze_reciprocity(tup))
        return

    if args.search_base_level is None:
        raise SystemExit("--search-base-level is required unless --analyze is set")

    base = next((hit for hit in KNOWN_HITS if hit.level == args.search_base_level), None)
    if base is None:
        raise SystemExit(f"unknown base level: {args.search_base_level}")

    print(f"searching around base level {base.level}, scale={args.scale}, radius={args.radius}")
    rows = search_scaled_family(base, scale=args.scale, radius=args.radius, keep=args.keep)
    for tup, split, cost, rec, mult1 in rows:
        level = sum(tup[:3]) + 1
        tag = "HIT" if cost == (0, 0) else "near"
        print(
            f"{tag} level={level} tuple={tup} split={split} cost={cost} reciprocity={rec} mult1={mult1}"
        )


if __name__ == "__main__":
    main()
