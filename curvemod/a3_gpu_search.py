from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from a3_gpu_burau import (
    CompiledOperator,
    apply_operator,
    compile_simple_operators,
    make_initial_state,
    normalize_states,
)
from a3_gpu_tables import A3TableData, build_a3_table_data


@dataclass
class SearchConfig:
    cap_1: int = 500
    cap_2: int = 500
    total_cap_1: int = 50000
    total_cap_2: int = 50000
    first_steps: int = 12
    modulus: int = 7
    max_g_length: int = 1000
    device: str = "auto"
    seed: int = 0


@dataclass
class Bucket:
    states: torch.Tensor
    last_simple_ids: torch.Tensor
    parent_indices: torch.Tensor
    record_indices: Optional[torch.Tensor] = None

    @property
    def size(self) -> int:
        return int(self.last_simple_ids.numel())


@dataclass
class SearchResult:
    witness: Optional[List[List[int]]]
    depth: Optional[int]
    runtime_seconds: float
    device: str
    found: bool
    spread_counts_by_depth: Dict[int, Dict[int, int]]


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _priority_cap(bucket: Bucket, cap: int, generator: torch.Generator) -> Bucket:
    if bucket.size <= cap:
        return bucket

    priorities = torch.rand(bucket.size, generator=generator, device=bucket.states.device)
    keep = torch.topk(priorities, k=cap, largest=False).indices
    return Bucket(
        states=bucket.states.index_select(0, keep),
        last_simple_ids=bucket.last_simple_ids.index_select(0, keep),
        parent_indices=bucket.parent_indices.index_select(0, keep),
    )


def _concat_bucket_parts(parts: list[Bucket]) -> Bucket:
    if len(parts) == 1:
        return parts[0]
    return Bucket(
        states=torch.cat([part.states for part in parts], dim=0),
        last_simple_ids=torch.cat([part.last_simple_ids for part in parts], dim=0),
        parent_indices=torch.cat([part.parent_indices for part in parts], dim=0),
    )


def _register_depth(
    depth: int,
    buckets: Dict[int, Bucket],
    history_last_simple_ids: List[Optional[torch.Tensor]],
    history_parent_indices: List[Optional[torch.Tensor]],
) -> None:
    spreads = sorted(buckets.keys())
    total = sum(bucket.size for bucket in buckets.values())
    last_simple_ids = torch.empty(total, dtype=torch.long)
    parent_indices = torch.empty(total, dtype=torch.long)

    offset = 0
    for spread in spreads:
        bucket = buckets[spread]
        count = bucket.size
        record_indices = torch.arange(offset, offset + count, dtype=torch.long, device=bucket.states.device)
        bucket.record_indices = record_indices
        last_simple_ids[offset : offset + count] = bucket.last_simple_ids.detach().cpu()
        parent_indices[offset : offset + count] = bucket.parent_indices.detach().cpu()
        offset += count

    while len(history_last_simple_ids) <= depth:
        history_last_simple_ids.append(None)
        history_parent_indices.append(None)

    history_last_simple_ids[depth] = last_simple_ids
    history_parent_indices[depth] = parent_indices


def _reconstruct_path(
    simple_words: List[List[int]],
    history_last_simple_ids: List[Optional[torch.Tensor]],
    history_parent_indices: List[Optional[torch.Tensor]],
    depth: int,
    record_index: int,
) -> List[List[int]]:
    out: List[List[int]] = []
    current_depth = depth
    current_index = record_index
    while current_depth >= 1 and current_index >= 0:
        last_simple = int(history_last_simple_ids[current_depth][current_index].item())
        out.append(simple_words[last_simple])
        current_index = int(history_parent_indices[current_depth][current_index].item())
        current_depth -= 1
    return out


def _build_initial_buckets(
    config: SearchConfig,
    tables: A3TableData,
    operators: List[CompiledOperator],
    width: int,
    device: torch.device,
    generator: torch.Generator,
) -> Dict[int, Bucket]:
    e1 = make_initial_state(width=width, device=device)
    buckets: dict[int, list[Bucket]] = {}
    for simple_id in tables.start_simple_ids:
        state = apply_operator(e1, operators[simple_id], config.modulus)
        state, spread = normalize_states(state, config.modulus)
        spread_value = int(spread[0].item())
        if spread_value != 1:
            continue
        entry = Bucket(
            states=state,
            last_simple_ids=torch.tensor([simple_id], dtype=torch.long, device=device),
            parent_indices=torch.tensor([-1], dtype=torch.long, device=device),
        )
        buckets.setdefault(spread_value, []).append(entry)

    capped: Dict[int, Bucket] = {}
    for spread, parts in buckets.items():
        bucket = _concat_bucket_parts(parts)
        capped[spread] = _priority_cap(bucket, config.cap_1, generator)
    return capped


def _selected_prev_spreads(buckets: Dict[int, Bucket], total_cap: int) -> List[int]:
    if not buckets:
        return []
    spreads = sorted(buckets.keys())
    selected = []
    total = 0
    for spread in spreads:
        selected.append(spread)
        total += buckets[spread].size
        if total >= total_cap:
            break
    return selected


def run_search(config: SearchConfig) -> SearchResult:
    if config.modulus <= 0:
        raise ValueError("The GPU rewrite currently supports only the Fp case, so modulus must be positive.")

    device = _resolve_device(config.device)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(config.seed)

    tables = build_a3_table_data(modulus=config.modulus, device=device)
    operators = compile_simple_operators(tables.simple_words, device=device)

    global_min_shift = min(operator.min_shift for operator in operators)
    global_max_shift = max(operator.max_shift for operator in operators)
    operator_span = global_max_shift - global_min_shift
    width = config.max_g_length + operator_span + 1

    history_last_simple_ids: List[Optional[torch.Tensor]] = [None]
    history_parent_indices: List[Optional[torch.Tensor]] = [None]

    start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if start_event is not None:
        start_event.record()
    else:
        import time

        wall_start = time.time()

    print(
        "Starting A3 torch search with p=",
        config.modulus,
        "caps",
        config.cap_1,
        config.cap_2,
        "totalcaps",
        config.total_cap_1,
        config.total_cap_2,
        "FIRST_STEPS",
        config.first_steps,
        "device",
        device,
    )
    print(f"We have {len(tables.dual_words)} atoms and {len(tables.simple_words)} dual simples")

    current_buckets = _build_initial_buckets(config, tables, operators, width, device, generator)
    _register_depth(1, current_buckets, history_last_simple_ids, history_parent_indices)
    spread_counts_by_depth: Dict[int, Dict[int, int]] = {
        1: {spread: bucket.size for spread, bucket in sorted(current_buckets.items())}
    }
    print("Initialized step 1 with spreads:", spread_counts_by_depth[1])

    witness: Optional[List[List[int]]] = None
    witness_depth: Optional[int] = None

    with torch.no_grad():
        for cur in range(2, config.max_g_length + 1):
            if not current_buckets:
                break

            if cur < config.first_steps:
                cap = config.cap_1
                total_cap = config.total_cap_1
            else:
                cap = config.cap_2
                total_cap = config.total_cap_2

            selected_spreads = _selected_prev_spreads(current_buckets, total_cap)
            if not selected_spreads:
                break

            print(f"Starting step {cur}")
            next_candidates: dict[int, list[Bucket]] = {}
            total_drops = 0

            for prev_spread in selected_spreads:
                bucket = current_buckets[prev_spread]
                print(f"Analyzing {bucket.size} states in spread {prev_spread}")
                unique_ids = torch.unique(bucket.last_simple_ids)
                for simple_id_tensor in unique_ids:
                    simple_id = int(simple_id_tensor.item())
                    mask = bucket.last_simple_ids == simple_id
                    group_states = bucket.states[mask]
                    group_parents = bucket.record_indices[mask]
                    successors = tables.allowed_successors[simple_id]
                    if not successors:
                        continue

                    for next_simple_id in successors:
                        candidate_states = apply_operator(group_states, operators[next_simple_id], config.modulus)
                        candidate_states, spreads = normalize_states(candidate_states, config.modulus)

                        valid = spreads >= 0
                        if cur == 2:
                            valid &= spreads != 0

                        if not valid.any():
                            continue

                        drops_mask = valid & (spreads < prev_spread)
                        total_drops += int(drops_mask.sum().item())

                        hit_mask = valid & (spreads == 0)
                        if hit_mask.any():
                            hit_index = int(torch.nonzero(hit_mask, as_tuple=False)[0, 0].item())
                            parent_record = int(group_parents[hit_index].item())
                            witness = [tables.simple_words[next_simple_id]] + _reconstruct_path(
                                tables.simple_words,
                                history_last_simple_ids,
                                history_parent_indices,
                                cur - 1,
                                parent_record,
                            )
                            witness_depth = cur
                            print("Found one for p=", config.modulus, witness)
                            break

                        valid &= spreads <= (config.max_g_length - cur + 1)
                        if not valid.any():
                            continue

                        kept_states = candidate_states[valid]
                        kept_spreads = spreads[valid]
                        kept_parents = group_parents[valid]
                        kept_simple_ids = torch.full(
                            (kept_states.shape[0],),
                            next_simple_id,
                            dtype=torch.long,
                            device=device,
                        )

                        for spread_value in torch.unique(kept_spreads):
                            spread_int = int(spread_value.item())
                            spread_mask = kept_spreads == spread_value
                            part = Bucket(
                                states=kept_states[spread_mask],
                                last_simple_ids=kept_simple_ids[spread_mask],
                                parent_indices=kept_parents[spread_mask],
                            )
                            next_candidates.setdefault(spread_int, []).append(part)

                    if witness is not None:
                        break
                if witness is not None:
                    break

            if witness is not None:
                break

            next_buckets: Dict[int, Bucket] = {}
            for spread, parts in next_candidates.items():
                bucket = _concat_bucket_parts(parts)
                next_buckets[spread] = _priority_cap(bucket, cap, generator)

            current_buckets = next_buckets
            if not current_buckets:
                spread_counts_by_depth[cur] = {}
                for later_depth in range(cur + 1, config.max_g_length + 1):
                    spread_counts_by_depth[later_depth] = {}
                break

            _register_depth(cur, current_buckets, history_last_simple_ids, history_parent_indices)
            counts = {spread: bucket.size for spread, bucket in sorted(current_buckets.items())}
            spread_counts_by_depth[cur] = counts
            print(
                f"Finished step {cur}, minimal spread {min(counts)}, max spread {max(counts)}, got {total_drops} drops"
            )
            print(f"Total number of curves {sum(counts.values())}")

    if end_event is not None:
        end_event.record()
        torch.cuda.synchronize(device)
        runtime_seconds = start_event.elapsed_time(end_event) / 1000.0
    else:
        import time

        runtime_seconds = time.time() - wall_start

    print("Altogether, runtime=", runtime_seconds)

    return SearchResult(
        witness=witness,
        depth=witness_depth,
        runtime_seconds=runtime_seconds,
        device=str(device),
        found=witness is not None,
        spread_counts_by_depth=spread_counts_by_depth,
    )
