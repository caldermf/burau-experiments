from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from d4_gpu_common import (
    GAMMA_WORD,
    RANK,
    build_automaton,
    build_automaton_indices,
    build_descents,
    build_garside_gens,
    build_simple_action_tensors,
    compute_oburau_vector,
    ct,
    dense_from_dim_vec,
    find_representative,
    flatten_factors,
    make_fp_vec,
)
from tensor_backend import get_backend


def storage_dtype_for(p: int, requested: str):
    if requested == "int16":
        if p == 0 or p >= 32768:
            raise ValueError("int16 storage requires 0 < p < 32768")
        return np.int16
    if requested == "int32":
        return np.int32
    if p != 0 and p < 32768:
        return np.int16
    return np.int32


def torch_dtype_for(backend, dtype):
    return backend._torch_dtype(dtype)


@dataclass
class TableBundle:
    simple_words: list[list[int]]
    gamma_idx: int
    valid_suffixes: np.ndarray
    num_valid_suffixes: np.ndarray
    init_states: np.ndarray
    init_words: np.ndarray
    init_lengths: np.ndarray
    init_last: np.ndarray


class GPUSpreadBuckets:
    def __init__(self, bucket_size: int, backend):
        self.bucket_size = bucket_size
        self.backend = backend
        self.data: dict[int, tuple[object, object, object, object, object]] = {}

    def add_chunk(self, states, words, lengths, last_simple, spreads):
        if states.shape[0] == 0:
            return

        lib = self.backend.lib
        if self.backend.is_torch:
            priorities = lib.rand(states.shape[0], device=self.backend.device)
            unique_spreads = self.backend.to_numpy(lib.unique(spreads)).astype(np.int32)
        else:
            priorities = np.random.rand(states.shape[0])
            unique_spreads = np.unique(spreads).astype(np.int32)

        for spread in unique_spreads.tolist():
            mask = spreads == spread
            new_states = states[mask]
            new_words = words[mask]
            new_lengths = lengths[mask]
            new_last = last_simple[mask]
            new_priorities = priorities[mask]

            if spread not in self.data:
                if new_states.shape[0] <= self.bucket_size:
                    self.data[spread] = (new_states, new_words, new_lengths, new_last, new_priorities)
                else:
                    keep = self._topk_lowest(new_priorities, self.bucket_size)
                    self.data[spread] = (
                        new_states[keep],
                        new_words[keep],
                        new_lengths[keep],
                        new_last[keep],
                        new_priorities[keep],
                    )
                continue

            old_states, old_words, old_lengths, old_last, old_priorities = self.data[spread]
            merged_states = self.backend.concat([old_states, new_states], axis=0)
            merged_words = self.backend.concat([old_words, new_words], axis=0)
            merged_lengths = self.backend.concat([old_lengths, new_lengths], axis=0)
            merged_last = self.backend.concat([old_last, new_last], axis=0)
            merged_priorities = self.backend.concat([old_priorities, new_priorities], axis=0)

            if merged_states.shape[0] <= self.bucket_size:
                self.data[spread] = (merged_states, merged_words, merged_lengths, merged_last, merged_priorities)
            else:
                keep = self._topk_lowest(merged_priorities, self.bucket_size)
                self.data[spread] = (
                    merged_states[keep],
                    merged_words[keep],
                    merged_lengths[keep],
                    merged_last[keep],
                    merged_priorities[keep],
                )

    def _topk_lowest(self, priorities, k: int):
        if self.backend.is_torch:
            _, keep = self.backend.lib.topk(priorities, k, largest=False)
            return keep
        return np.argpartition(priorities, k - 1)[:k]

    def get_bucket_items(self):
        return {spread: values[:4] for spread, values in self.data.items()}

    def total_count(self) -> int:
        return sum(states.shape[0] for states, *_ in self.data.values())


def parse_args():
    parser = argparse.ArgumentParser(description="GPU-native D4 mod-p Burau same-curve bucket search.")
    parser.add_argument("--p", type=int, default=7, help="Work over Z/pZ.")
    parser.add_argument("--max-g-length", type=int, default=1000, help="Maximum dual-Garside length to explore.")
    parser.add_argument("--backend", choices=["auto", "numpy", "torch"], default="torch")
    parser.add_argument("--device", default="cuda", help="Backend device, e.g. cuda or cpu.")
    parser.add_argument("--storage-dtype", choices=["auto", "int16", "int32"], default="auto")
    parser.add_argument("--cap-1", type=int, default=4000, help="Per-spread bucket cap during early steps.")
    parser.add_argument("--cap-2", type=int, default=4000, help="Per-spread bucket cap after early steps.")
    parser.add_argument("--total-cap-1", type=int, default=50000, help="Total low-spread curves selected early.")
    parser.add_argument("--total-cap-2", type=int, default=50000, help="Total low-spread curves selected later.")
    parser.add_argument("--first-steps", type=int, default=12, help="Number of early broad-exploration steps.")
    parser.add_argument("--base-vertex", type=int, default=1, choices=ct.positive_letters)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expansion-chunk-size", type=int, default=65536)
    parser.add_argument("--use-random-subset", type=int, default=0, help="If > 0, randomly limit selected states per level.")
    parser.add_argument("--hit-condition", choices=["spread-zero", "same-curve"], default="spread-zero")
    parser.add_argument("--max-hits", type=int, default=10000, help="Maximum hits to serialize; 0 means unlimited.")
    parser.add_argument("--stop-at-first", action="store_true")
    parser.add_argument("--verbose-tables", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "d4_mod_p_gpu_native_hits.json",
    )
    args = parser.parse_args()
    if args.p < 2:
        raise ValueError("--p must be at least 2 for the mod-p GPU search")
    return args


def build_tables(max_g_length: int, base_vertex: int, p: int, storage_dtype, verbose: bool) -> TableBundle:
    simple_words = build_garside_gens(verbose=verbose)
    simple_to_index = {tuple(word): idx for idx, word in enumerate(simple_words)}
    gamma_rep = find_representative(GAMMA_WORD, simple_words)
    gamma_idx = simple_to_index[tuple(gamma_rep)]
    _, right_descents = build_descents(simple_words)
    automaton = build_automaton(simple_words, right_descents, gamma_rep)
    automaton_idx = build_automaton_indices(simple_words, automaton)

    max_valid = max((len(v) for v in automaton_idx.values()), default=0)
    valid_suffixes = np.full((len(simple_words), max_valid), -1, dtype=np.int16)
    num_valid_suffixes = np.zeros((len(simple_words),), dtype=np.int16)
    for source_idx, targets in automaton_idx.items():
        num_valid_suffixes[source_idx] = len(targets)
        if targets:
            valid_suffixes[source_idx, : len(targets)] = np.array(targets, dtype=np.int16)

    state_width = max_g_length + 3
    init_states = []
    init_words = []
    init_lengths = []
    init_last = []
    base_target = ct.dim_vectors[base_vertex]

    for simple_idx, simple_word in enumerate(simple_words):
        if simple_word == [] or simple_idx == gamma_idx:
            continue

        admissible = True
        for descent in right_descents[str(simple_word)]:
            descent_vec = compute_oburau_vector(descent, base_vertex)
            if ct.poly_normalize_vector(descent_vec) == base_target:
                admissible = False
                break
            if ct.topdeg_vector(descent_vec) - ct.botdeg_vector(descent_vec) != 1:
                admissible = False
                break

        if not admissible:
            continue

        vec = make_fp_vec(compute_oburau_vector(simple_word, base_vertex), p)
        spread = ct.topdeg_vector(vec) - ct.botdeg_vector(vec)
        if spread != 1:
            continue

        word_row = np.full((max_g_length,), -1, dtype=np.int16)
        word_row[0] = simple_idx
        init_states.append(dense_from_dim_vec(vec, state_width, p, storage_dtype))
        init_words.append(word_row)
        init_lengths.append(1)
        init_last.append(simple_idx)

    if init_states:
        init_states_np = np.stack(init_states, axis=0)
        init_words_np = np.stack(init_words, axis=0)
        init_lengths_np = np.array(init_lengths, dtype=np.int16)
        init_last_np = np.array(init_last, dtype=np.int16)
    else:
        init_states_np = np.zeros((0, RANK, state_width), dtype=storage_dtype)
        init_words_np = np.zeros((0, max_g_length), dtype=np.int16)
        init_lengths_np = np.zeros((0,), dtype=np.int16)
        init_last_np = np.zeros((0,), dtype=np.int16)

    return TableBundle(
        simple_words=simple_words,
        gamma_idx=gamma_idx,
        valid_suffixes=valid_suffixes,
        num_valid_suffixes=num_valid_suffixes,
        init_states=init_states_np,
        init_words=init_words_np,
        init_lengths=init_lengths_np,
        init_last=init_last_np,
    )


def build_expansion_indices(last_simple, num_valid_suffixes, valid_suffixes, backend):
    lib = backend.lib
    if last_simple.shape[0] == 0:
        if backend.is_torch:
            empty = lib.empty(0, dtype=lib.int64, device=backend.device)
            return empty, empty
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    if backend.is_torch:
        counts = num_valid_suffixes[last_simple.long()].to(lib.int64)
        total = int(backend.to_numpy(counts.sum()))
    else:
        counts = num_valid_suffixes[last_simple.astype(np.int64)].astype(np.int64, copy=False)
        total = int(counts.sum())

    if total == 0:
        if backend.is_torch:
            empty = lib.empty(0, dtype=lib.int64, device=backend.device)
            return empty, empty
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    if backend.is_torch:
        braid_indices = lib.repeat_interleave(
            lib.arange(last_simple.shape[0], device=backend.device, dtype=lib.int64), counts
        )
        cumsum = lib.cumsum(counts, dim=0)
        starts = cumsum - counts
        global_positions = lib.arange(total, device=backend.device, dtype=lib.int64)
        local_suffix_indices = global_positions - starts[braid_indices]
        expanded_last = last_simple[braid_indices]
        suffix_indices = valid_suffixes[expanded_last.long(), local_suffix_indices.long()]
        return braid_indices.long(), suffix_indices.long()

    braid_indices = np.repeat(np.arange(last_simple.shape[0], dtype=np.int64), counts)
    cumsum = np.cumsum(counts)
    starts = cumsum - counts
    global_positions = np.arange(total, dtype=np.int64)
    local_suffix_indices = global_positions - starts[braid_indices]
    expanded_last = last_simple[braid_indices]
    suffix_indices = valid_suffixes[expanded_last.astype(np.int64), local_suffix_indices.astype(np.int64)]
    return braid_indices, suffix_indices.astype(np.int64)


def apply_simple_actions(states, suffix_indices, simple_actions, p: int, backend):
    lib = backend.lib
    parent = states.to(lib.int32) if backend.is_torch else states.astype(np.int32, copy=False)
    actions = simple_actions[suffix_indices]
    actions = actions.to(lib.int32) if backend.is_torch else actions.astype(np.int32, copy=False)
    batch, rank, width = parent.shape
    out = backend.zeros((batch, rank, width + 1), dtype=np.int32)

    for source_coord in range(rank):
        source = parent[:, source_coord, :]
        coeff0 = actions[:, :, source_coord, 0]
        coeff1 = actions[:, :, source_coord, 1]
        out[:, :, :width] = out[:, :, :width] + coeff0[:, :, None] * source[:, None, :]
        out[:, :, 1:] = out[:, :, 1:] + coeff1[:, :, None] * source[:, None, :]

    if p:
        out %= p
    return out


def normalize_and_score(states, state_width: int, base_vertex: int, backend):
    lib = backend.lib
    alive_mask = lib.any(states != 0, dim=1) if backend.is_torch else np.any(states != 0, axis=1)
    alive_rows = lib.any(alive_mask, dim=1) if backend.is_torch else np.any(alive_mask, axis=1)
    if int(backend.to_numpy(alive_rows.sum())) == 0:
        empty_bool = np.empty(0, dtype=bool)
        empty_i32 = np.empty(0, dtype=np.int32)
        return states[:0, :, :state_width], empty_i32, backend.to_numpy(alive_rows).astype(bool), empty_bool, empty_bool

    filtered = states[alive_rows]
    filtered_mask = alive_mask[alive_rows]
    if backend.is_torch:
        min_idx = lib.argmax(filtered_mask.to(lib.int64), dim=1)
        reversed_mask = lib.flip(filtered_mask, dims=(1,))
        max_idx = filtered.shape[2] - 1 - lib.argmax(reversed_mask.to(lib.int64), dim=1)
    else:
        min_idx = np.argmax(filtered_mask.astype(np.int64), axis=1)
        reversed_mask = np.flip(filtered_mask, axis=1)
        max_idx = filtered.shape[2] - 1 - np.argmax(reversed_mask.astype(np.int64), axis=1)
    spreads = max_idx - min_idx

    if backend.is_torch:
        padded = lib.cat([filtered, backend.zeros((filtered.shape[0], filtered.shape[1], state_width), dtype=np.int32)], dim=2)
        gather_idx = lib.arange(state_width, device=backend.device, dtype=lib.int64)[None, :] + min_idx[:, None].to(lib.int64)
    else:
        padded = np.concatenate(
            [filtered, np.zeros((filtered.shape[0], filtered.shape[1], state_width), dtype=np.int32)],
            axis=2,
        )
        gather_idx = np.arange(state_width, dtype=np.int64)[None, :] + np.asarray(min_idx, dtype=np.int64)[:, None]

    normalized = backend.gather_last_axis(padded, gather_idx)
    degree0 = normalized[:, :, 0]
    target = np.zeros((RANK,), dtype=np.int32)
    target[base_vertex - 1] = 1

    if backend.is_torch:
        target_t = lib.tensor(target, dtype=lib.int32, device=backend.device)
        same_curve = (spreads == 0) & lib.all(degree0 == target_t[None, :], dim=1)
        spread_zero = spreads == 0
        return (
            normalized,
            backend.to_numpy(spreads).astype(np.int32),
            backend.to_numpy(alive_rows).astype(bool),
            backend.to_numpy(spread_zero).astype(bool),
            backend.to_numpy(same_curve).astype(bool),
        )

    same_curve = (spreads == 0) & np.all(degree0 == target[None, :], axis=1)
    spread_zero = spreads == 0
    return normalized, np.asarray(spreads, dtype=np.int32), np.asarray(alive_rows, dtype=bool), spread_zero.astype(bool), same_curve.astype(bool)


def tensor_from_numpy(lib, backend, arr, dtype):
    if backend.is_torch:
        return lib.tensor(arr, dtype=torch_dtype_for(backend, dtype), device=backend.device)
    return arr.astype(dtype, copy=False)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    storage_dtype = storage_dtype_for(args.p, args.storage_dtype)
    backend = get_backend(args.backend, args.device)
    lib = backend.lib
    if backend.is_torch:
        lib.manual_seed(args.seed)
        if backend.device == "cuda":
            lib.cuda.manual_seed_all(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    tables = build_tables(args.max_g_length, args.base_vertex, args.p, storage_dtype, args.verbose_tables)
    state_width = args.max_g_length + 3
    simple_actions_np = build_simple_action_tensors(tables.simple_words, args.p, storage_dtype)

    simple_actions = tensor_from_numpy(lib, backend, simple_actions_np, storage_dtype)
    valid_suffixes = tensor_from_numpy(lib, backend, tables.valid_suffixes, np.int16)
    num_valid_suffixes = tensor_from_numpy(lib, backend, tables.num_valid_suffixes, np.int16)
    init_states = tensor_from_numpy(lib, backend, tables.init_states, storage_dtype)
    init_words = tensor_from_numpy(lib, backend, tables.init_words, np.int16)
    init_lengths = tensor_from_numpy(lib, backend, tables.init_lengths, np.int16)
    init_last = tensor_from_numpy(lib, backend, tables.init_last, np.int16)

    current_buckets = GPUSpreadBuckets(args.cap_1, backend)
    init_spreads_np = np.full((init_states.shape[0],), 1, dtype=np.int32)
    init_spreads = tensor_from_numpy(lib, backend, init_spreads_np, np.int32)
    current_buckets.add_chunk(init_states, init_words, init_lengths, init_last, init_spreads)

    hits = []
    hit_limit_reached = False

    print(f"Backend: {backend.kind} on {backend.device or 'cpu'}")
    print(f"Storage dtype: {np.dtype(storage_dtype).name}")
    print(f"Built {len(tables.simple_words)} D4 Garside simples.")
    print(f"Starting D4 GPU-native search over Z/{args.p}Z with base vertex {args.base_vertex}.")

    for current_length in range(2, args.max_g_length + 1):
        previous = current_buckets.get_bucket_items()
        if not previous:
            break

        step_start = time.time()
        total_cap = args.total_cap_1 if current_length < args.first_steps else args.total_cap_2
        cap = args.cap_1 if current_length < args.first_steps else args.cap_2
        current_buckets = GPUSpreadBuckets(cap, backend)

        selected_spreads = []
        selected_count = 0
        for spread in sorted(previous.keys()):
            count = previous[spread][0].shape[0]
            if selected_count == 0 or selected_count + count <= total_cap:
                selected_spreads.append(spread)
                selected_count += count
            else:
                break

        if not selected_spreads:
            break

        flat_states = backend.concat([previous[spread][0] for spread in selected_spreads], axis=0)
        flat_words = backend.concat([previous[spread][1] for spread in selected_spreads], axis=0)
        flat_lengths = backend.concat([previous[spread][2] for spread in selected_spreads], axis=0)
        flat_last = backend.concat([previous[spread][3] for spread in selected_spreads], axis=0)
        flat_prev_spreads = np.concatenate(
            [np.full(previous[spread][0].shape[0], spread, dtype=np.int32) for spread in selected_spreads]
        )

        if args.use_random_subset > 0 and flat_states.shape[0] > args.use_random_subset:
            if backend.is_torch:
                perm = lib.randperm(flat_states.shape[0], device=backend.device)[: args.use_random_subset]
                flat_states = flat_states[perm]
                flat_words = flat_words[perm]
                flat_lengths = flat_lengths[perm]
                flat_last = flat_last[perm]
                flat_prev_spreads = flat_prev_spreads[backend.to_numpy(perm)]
            else:
                perm = np.random.permutation(flat_states.shape[0])[: args.use_random_subset]
                flat_states = flat_states[perm]
                flat_words = flat_words[perm]
                flat_lengths = flat_lengths[perm]
                flat_last = flat_last[perm]
                flat_prev_spreads = flat_prev_spreads[perm]

        print(
            f"Step {current_length}: analyzing {flat_states.shape[0]} braids "
            f"across spreads {selected_spreads[0]}..{selected_spreads[-1]}"
        )

        braid_indices, suffix_indices = build_expansion_indices(flat_last, num_valid_suffixes, valid_suffixes, backend)
        if braid_indices.shape[0] == 0:
            print(f"Step {current_length}: no admissible expansions")
            break

        step_hits = 0
        spread_drops = 0
        admissible_spread = args.max_g_length - current_length + 1
        stop_after_step = False

        for start_idx in range(0, braid_indices.shape[0], args.expansion_chunk_size):
            end_idx = min(start_idx + args.expansion_chunk_size, braid_indices.shape[0])
            chunk_braid_idx = braid_indices[start_idx:end_idx]
            chunk_suffix_idx = suffix_indices[start_idx:end_idx]

            parent_states = flat_states[chunk_braid_idx]
            parent_words = flat_words[chunk_braid_idx]
            parent_lengths = flat_lengths[chunk_braid_idx]
            parent_prev_spreads = flat_prev_spreads[backend.to_numpy(chunk_braid_idx)]

            expanded_states = apply_simple_actions(parent_states, chunk_suffix_idx, simple_actions, args.p, backend)
            normalized, spread_np, alive_np, spread_zero_np, same_curve_np = normalize_and_score(
                expanded_states, state_width, args.base_vertex, backend
            )
            if normalized.shape[0] == 0:
                continue

            alive_indices = np.flatnonzero(alive_np)
            alive_idx_tensor = tensor_from_numpy(lib, backend, alive_indices.astype(np.int64), np.int64)
            parent_words_alive = parent_words[alive_idx_tensor]
            parent_lengths_alive = parent_lengths[alive_idx_tensor]
            suffix_alive = chunk_suffix_idx[alive_idx_tensor]
            prev_spreads_alive = parent_prev_spreads[alive_indices]

            if backend.is_torch:
                new_words = parent_words_alive.clone()
                row_idx = lib.arange(parent_lengths_alive.shape[0], device=backend.device)
                new_words[row_idx, parent_lengths_alive.long()] = suffix_alive.to(lib.int16)
                new_lengths = parent_lengths_alive + 1
                new_last = suffix_alive.to(lib.int16)
            else:
                new_words = parent_words_alive.copy()
                row_idx = np.arange(parent_lengths_alive.shape[0])
                new_words[row_idx, parent_lengths_alive.astype(np.int64)] = suffix_alive.astype(np.int16)
                new_lengths = parent_lengths_alive + 1
                new_last = suffix_alive.astype(np.int16)

            nonstupid_mask = ~((current_length == 2) & spread_zero_np)
            if args.hit_condition == "same-curve":
                hit_mask = same_curve_np & nonstupid_mask
            else:
                hit_mask = spread_zero_np & nonstupid_mask

            step_hits += int(hit_mask.sum())
            hit_local = np.flatnonzero(hit_mask)
            if hit_local.size:
                hit_idx = tensor_from_numpy(lib, backend, hit_local.astype(np.int64), np.int64)
                hit_words = backend.to_numpy(new_words[hit_idx]).astype(np.int16)
                hit_lengths = backend.to_numpy(new_lengths[hit_idx]).astype(np.int16)
                hit_states = backend.to_numpy(normalized[hit_idx]).astype(np.int32)
                for local_pos, (word_row, length, state) in enumerate(zip(hit_words, hit_lengths, hit_states)):
                    if args.max_hits > 0 and len(hits) >= args.max_hits:
                        hit_limit_reached = True
                        stop_after_step = True
                        break
                    factor_indices = [int(x) for x in word_row[:length] if int(x) >= 0]
                    factor_indices = list(reversed(factor_indices))
                    degree0 = state[:, 0].astype(int).tolist()
                    hits.append(
                        {
                            "simple_indices": factor_indices,
                            "normal_form_factors": [tables.simple_words[idx] for idx in factor_indices],
                            "artin_word": flatten_factors(factor_indices, tables.simple_words),
                            "garside_length": int(length),
                            "spread": 0,
                            "same_base_vector": bool(same_curve_np[hit_local[local_pos]]),
                            "normalized_degree0": degree0,
                        }
                    )
                if args.stop_at_first or stop_after_step:
                    stop_after_step = True

            spread_drops += int((spread_np < prev_spreads_alive).sum())
            keep_mask = (spread_np <= admissible_spread) & nonstupid_mask
            if not keep_mask.any() or stop_after_step:
                if stop_after_step:
                    break
                continue

            keep_local = np.flatnonzero(keep_mask)
            keep_idx = tensor_from_numpy(lib, backend, keep_local.astype(np.int64), np.int64)
            current_buckets.add_chunk(
                normalized[keep_idx].to(torch_dtype_for(backend, storage_dtype)) if backend.is_torch else normalized[keep_idx].astype(storage_dtype, copy=False),
                new_words[keep_idx].to(lib.int16) if backend.is_torch else new_words[keep_idx].astype(np.int16, copy=False),
                new_lengths[keep_idx].to(lib.int16) if backend.is_torch else new_lengths[keep_idx].astype(np.int16, copy=False),
                new_last[keep_idx].to(lib.int16) if backend.is_torch else new_last[keep_idx].astype(np.int16, copy=False),
                tensor_from_numpy(lib, backend, spread_np[keep_local].astype(np.int32), np.int32),
            )

        elapsed = time.time() - step_start
        next_items = current_buckets.get_bucket_items()
        if next_items:
            print(
                f"Finished step {current_length}: min spread {min(next_items.keys())}, "
                f"max spread {max(next_items.keys())}, drops {spread_drops}, hits this step {step_hits}, "
                f"survivors {current_buckets.total_count()}, runtime {elapsed:.2f}s"
            )
        else:
            print(
                f"Finished step {current_length}: no surviving buckets, drops {spread_drops}, "
                f"hits this step {step_hits}, runtime {elapsed:.2f}s"
            )

        if args.stop_at_first and step_hits > 0:
            break
        if hit_limit_reached:
            print(f"Stopping because --max-hits={args.max_hits} was reached.")
            break
        if not next_items:
            break

    payload = {
        "parameters": {
            "p": args.p,
            "max_g_length": args.max_g_length,
            "backend": backend.kind,
            "device": backend.device,
            "storage_dtype": np.dtype(storage_dtype).name,
            "cap_1": args.cap_1,
            "cap_2": args.cap_2,
            "total_cap_1": args.total_cap_1,
            "total_cap_2": args.total_cap_2,
            "first_steps": args.first_steps,
            "base_vertex": args.base_vertex,
            "seed": args.seed,
            "expansion_chunk_size": args.expansion_chunk_size,
            "use_random_subset": args.use_random_subset,
            "hit_condition": args.hit_condition,
            "max_hits": args.max_hits,
        },
        "num_garside_simples": len(tables.simple_words),
        "num_hits": len(hits),
        "hit_limit_reached": hit_limit_reached,
        "runtime_seconds": time.time() - start,
        "hits": hits,
    }

    with args.output.open("w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote {len(hits)} hits to {args.output}")


if __name__ == "__main__":
    main()
