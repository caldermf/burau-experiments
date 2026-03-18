from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from a3_exact_check import commutator_is_identity
from a3_gpu_burau import (
    CompiledOperatorTable,
    apply_operator_ids,
    compile_simple_operators,
    make_initial_state,
    normalize_states,
    stack_compiled_operators,
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
    base_point: int = 1
    device: str = "auto"
    seed: int = 0
    max_witnesses: Optional[int] = 1
    print_witness_limit: int = 20
    witness_callback: Optional[Callable[[int, List[List[int]], int], None]] = None
    exact_commutator_generator: Optional[int] = None


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
    witnesses: List[List[List[int]]]
    witness_depths: List[int]
    total_hits: int
    accepted_hits: int
    rejected_hits: int


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
    out.reverse()
    return out


def _build_initial_buckets(
    config: SearchConfig,
    tables: A3TableData,
    operator_table: CompiledOperatorTable,
    width: int,
    device: torch.device,
    generator: torch.Generator,
) -> Dict[int, Bucket]:
    if not tables.start_simple_ids:
        return {}

    e1 = make_initial_state(width=width, device=device, base_point=config.base_point)
    start_simple_ids = torch.tensor(tables.start_simple_ids, dtype=torch.long, device=device)
    start_states = apply_operator_ids(e1, operator_table, start_simple_ids, config.modulus).squeeze(1)
    start_states, start_spreads = normalize_states(start_states, config.modulus)

    buckets: dict[int, list[Bucket]] = {}
    for spread_value in torch.unique(start_spreads):
        spread_int = int(spread_value.item())
        if spread_int != 1:
            continue
        mask = start_spreads == spread_value
        entry = Bucket(
            states=start_states[mask],
            last_simple_ids=start_simple_ids[mask],
            parent_indices=torch.full((int(mask.sum().item()),), -1, dtype=torch.long, device=device),
        )
        buckets.setdefault(spread_int, []).append(entry)

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


def _witness_key(witness: List[List[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(block) for block in witness)


def run_search(config: SearchConfig) -> SearchResult:
    if config.modulus <= 0:
        raise ValueError("The GPU rewrite currently supports only the Fp case, so modulus must be positive.")
    if config.base_point < 1 or config.base_point > 3:
        raise ValueError("base_point must be one of 1, 2, 3.")
    if config.max_witnesses is not None and config.max_witnesses < 0:
        raise ValueError("max_witnesses must be non-negative or None.")
    if config.print_witness_limit < 0:
        raise ValueError("print_witness_limit must be non-negative.")
    if config.exact_commutator_generator is not None and config.exact_commutator_generator not in (1, 2, 3):
        raise ValueError("exact_commutator_generator must be one of 1, 2, 3, or None.")

    device = _resolve_device(config.device)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(config.seed)

    tables = build_a3_table_data(modulus=config.modulus, device=device, base_point=config.base_point)
    operators = compile_simple_operators(tables.simple_words, device=device)
    operator_table = stack_compiled_operators(operators, device=device)

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
        "base_point",
        config.base_point,
        "device",
        device,
    )
    print(f"We have {len(tables.dual_words)} atoms and {len(tables.simple_words)} dual simples")

    current_buckets = _build_initial_buckets(config, tables, operator_table, width, device, generator)
    _register_depth(1, current_buckets, history_last_simple_ids, history_parent_indices)
    spread_counts_by_depth: Dict[int, Dict[int, int]] = {
        1: {spread: bucket.size for spread, bucket in sorted(current_buckets.items())}
    }
    print("Initialized step 1 with spreads:", spread_counts_by_depth[1])

    witness: Optional[List[List[int]]] = None
    witness_depth: Optional[int] = None
    witnesses: List[List[List[int]]] = []
    witness_depths: List[int] = []
    seen_witnesses: set[tuple[tuple[int, ...], ...]] = set()
    total_hits = 0
    accepted_hits = 0
    rejected_hits = 0
    stop_after_hit = False

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
                    successor_count = int(tables.allowed_count[simple_id].item())
                    if successor_count == 0:
                        continue
                    successor_ids = tables.allowed_suffix_padded[simple_id, :successor_count]
                    candidate_states = apply_operator_ids(group_states, operator_table, successor_ids, config.modulus)
                    num_successors, group_size = candidate_states.shape[:2]
                    candidate_states = candidate_states.view(num_successors * group_size, *candidate_states.shape[2:])
                    candidate_states, spreads = normalize_states(candidate_states, config.modulus)
                    candidate_states = candidate_states.view(num_successors, group_size, *candidate_states.shape[1:])
                    spreads = spreads.view(num_successors, group_size)

                    valid = spreads >= 0
                    if cur == 2:
                        valid &= spreads != 0

                    if not valid.any():
                        continue

                    drops_mask = valid & (spreads < prev_spread)
                    total_drops += int(drops_mask.sum().item())

                    hit_mask = valid & (spreads == 0)
                    if hit_mask.any():
                        hit_positions = torch.nonzero(hit_mask, as_tuple=False)
                        total_hits += int(hit_positions.shape[0])
                        for hit_successor_index, hit_state_index in hit_positions.tolist():
                            next_simple_id = int(successor_ids[hit_successor_index].item())
                            parent_record = int(group_parents[hit_state_index].item())
                            candidate_witness = _reconstruct_path(
                                tables.simple_words,
                                history_last_simple_ids,
                                history_parent_indices,
                                cur - 1,
                                parent_record,
                            ) + [tables.simple_words[next_simple_id]]
                            candidate_key = _witness_key(candidate_witness)
                            if candidate_key in seen_witnesses:
                                continue
                            seen_witnesses.add(candidate_key)
                            if config.exact_commutator_generator is not None:
                                if not commutator_is_identity(
                                    candidate_witness,
                                    modulus=config.modulus,
                                    generator=config.exact_commutator_generator,
                                ):
                                    rejected_hits += 1
                                    continue
                            accepted_hits += 1
                            witnesses.append(candidate_witness)
                            witness_depths.append(cur)
                            if config.witness_callback is not None:
                                config.witness_callback(cur, candidate_witness, len(witnesses))
                            if witness is None:
                                witness = candidate_witness
                                witness_depth = cur
                            if len(witnesses) <= config.print_witness_limit:
                                print(
                                    f"Found witness #{len(witnesses)} for p={config.modulus} at depth {cur}:",
                                    candidate_witness,
                                )
                            elif len(witnesses) == config.print_witness_limit + 1:
                                print(
                                    f"Reached witness print limit ({config.print_witness_limit}); suppressing further witness dumps."
                                )
                            if config.max_witnesses is not None and len(witnesses) >= config.max_witnesses:
                                stop_after_hit = True
                                break
                        if stop_after_hit:
                            break

                    valid &= spreads <= (config.max_g_length - cur + 1)
                    valid &= spreads != 0
                    if not valid.any():
                        continue

                    flat_valid = valid.reshape(-1)
                    flat_states = candidate_states.reshape(num_successors * group_size, *candidate_states.shape[2:])
                    flat_spreads = spreads.reshape(-1)
                    flat_parents = (
                        group_parents.unsqueeze(0).expand(num_successors, -1).reshape(-1)
                    )
                    flat_simple_ids = successor_ids.unsqueeze(1).expand(-1, group_size).reshape(-1)

                    kept_states = flat_states[flat_valid]
                    kept_spreads = flat_spreads[flat_valid]
                    kept_parents = flat_parents[flat_valid]
                    kept_simple_ids = flat_simple_ids[flat_valid]

                    for spread_value in torch.unique(kept_spreads):
                        spread_int = int(spread_value.item())
                        spread_mask = kept_spreads == spread_value
                        part = Bucket(
                            states=kept_states[spread_mask],
                            last_simple_ids=kept_simple_ids[spread_mask],
                            parent_indices=kept_parents[spread_mask],
                        )
                        next_candidates.setdefault(spread_int, []).append(part)

                    if stop_after_hit:
                        break
                if stop_after_hit:
                    break

            if stop_after_hit:
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
        witnesses=witnesses,
        witness_depths=witness_depths,
        total_hits=total_hits,
        accepted_hits=accepted_hits,
        rejected_hits=rejected_hits,
    )
