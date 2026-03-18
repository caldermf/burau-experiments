from __future__ import annotations

import argparse
import ast
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import setup_a3 as ct
from tensor_backend import get_backend
from a3_mod_p_bucket_search import (
    build_automaton,
    build_descents,
    build_garside_gens,
    compute_oburau_deg,
    compute_oburau_vector,
    find_representative,
)


def build_simple_action_tensors(simple_words, p: int):
    degree_cap = 1
    num_simples = len(simple_words)
    actions = np.zeros((num_simples, 3, 3, degree_cap + 1), dtype=np.int16)

    for simple_idx, simple_word in enumerate(simple_words):
        for basis_vertex in ct.positive_letters:
            image = compute_oburau_vector(simple_word, basis_vertex)
            for out_coord, poly in enumerate(image):
                for degree, coeff in poly.items():
                    if degree < 0 or degree > degree_cap:
                        raise ValueError(
                            f"Simple {simple_word} has unsupported degree {degree}; expected only 0/1 in dual form"
                        )
                    value = coeff % p if p else coeff
                    actions[simple_idx, out_coord, basis_vertex - 1, degree] = value

    return actions


def dense_from_dim_vec(vec, state_width, p):
    dense = np.zeros((3, state_width), dtype=np.int16)
    for coord, poly in enumerate(vec):
        for degree, coeff in poly.items():
            if degree < 0 or degree >= state_width:
                raise ValueError(f"Degree {degree} exceeds configured width {state_width}")
            dense[coord, degree] = coeff % p if p else coeff
    return dense


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


def build_tables(max_g_length: int, base_vertex: int, p: int) -> TableBundle:
    simple_words = build_garside_gens()
    simple_to_index = {tuple(word): idx for idx, word in enumerate(simple_words)}
    gamma_rep = find_representative(ct.GAMMA_WORD, simple_words)
    gamma_idx = simple_to_index[tuple(gamma_rep)]
    _, right_descents = build_descents(simple_words)
    automaton = build_automaton(simple_words, right_descents, gamma_rep)

    automaton_idx = {
        simple_to_index[tuple(source)]: [simple_to_index[tuple(target)] for target in targets]
        for source_key, targets in automaton.items()
        for source in [ast.literal_eval(source_key)]
        if source != []
    }

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

        vec = compute_oburau_vector(simple_word, base_vertex)
        normalized = ct.poly_normalize_vector(vec)
        if normalized == base_target:
            continue

        spread = ct.topdeg_vector(vec) - ct.botdeg_vector(vec)
        if spread != 1:
            continue

        word_row = np.full((max_g_length,), -1, dtype=np.int16)
        word_row[0] = simple_idx
        init_states.append(dense_from_dim_vec(vec, state_width, p))
        init_words.append(word_row)
        init_lengths.append(1)
        init_last.append(simple_idx)

    if init_states:
        init_states_np = np.stack(init_states, axis=0)
        init_words_np = np.stack(init_words, axis=0)
        init_lengths_np = np.array(init_lengths, dtype=np.int16)
        init_last_np = np.array(init_last, dtype=np.int16)
    else:
        init_states_np = np.zeros((0, 3, state_width), dtype=np.int16)
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


class GPUSpreadBuckets:
    def __init__(self, bucket_size: int, backend):
        self.bucket_size = bucket_size
        self.backend = backend
        self.data: dict[int, tuple[object, object, object, object, object]] = {}

    def add_chunk(self, states, words, lengths, last_simple, spreads):
        if states.shape[0] == 0:
            return

        lib = self.backend.lib
        priorities = lib.rand(states.shape[0], device=self.backend.device) if self.backend.is_torch else np.random.rand(states.shape[0])
        unique_spreads = self.backend.to_numpy(lib.unique(spreads) if self.backend.is_torch else np.unique(spreads)).astype(np.int32)

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
        return np.argpartition(priorities, k)[:k]

    def get_bucket_items(self):
        return {spread: values[:4] for spread, values in self.data.items()}

    def total_count(self) -> int:
        return sum(states.shape[0] for states, *_ in self.data.values())


def parse_args():
    parser = argparse.ArgumentParser(description="GPU-native A3 mod p witness search using burau-faithfulness conventions.")
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--max-g-length", type=int, default=60)
    parser.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cap-1", type=int, default=500)
    parser.add_argument("--cap-2", type=int, default=500)
    parser.add_argument("--total-cap-1", type=int, default=50000)
    parser.add_argument("--total-cap-2", type=int, default=50000)
    parser.add_argument("--first-steps", type=int, default=12)
    parser.add_argument("--base-vertex", type=int, default=1, choices=ct.positive_letters)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expansion-chunk-size", type=int, default=131072)
    parser.add_argument("--use-best", type=int, default=0, help="If > 0, limit gathered states per level to this many lowest-spread states.")
    parser.add_argument("--stop-at-first", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data" / "a3_gpu_native_hits.json")
    return parser.parse_args()


def build_expansion_indices(last_simple, num_valid_suffixes, valid_suffixes, backend):
    lib = backend.lib
    if last_simple.shape[0] == 0:
        if backend.is_torch:
            empty = lib.empty(0, dtype=lib.int64, device=backend.device)
            return empty, empty
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    counts = num_valid_suffixes[last_simple.long()].to(lib.int64) if backend.is_torch else num_valid_suffixes[last_simple.astype(np.int64)].astype(np.int64, copy=False)
    total = int(backend.to_numpy(counts.sum()))
    if total == 0:
        if backend.is_torch:
            empty = lib.empty(0, dtype=lib.int64, device=backend.device)
            return empty, empty
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    braid_indices = lib.repeat_interleave(lib.arange(last_simple.shape[0], device=backend.device, dtype=lib.int64), counts) if backend.is_torch else np.repeat(np.arange(last_simple.shape[0], dtype=np.int64), counts)
    cumsum = lib.cumsum(counts, dim=0) if backend.is_torch else np.cumsum(counts)
    starts = cumsum - counts
    global_positions = lib.arange(total, device=backend.device) if backend.is_torch else np.arange(total, dtype=np.int64)
    local_suffix_indices = global_positions - starts[braid_indices]
    expanded_last = last_simple[braid_indices]
    suffix_indices = valid_suffixes[expanded_last.long(), local_suffix_indices.long()] if backend.is_torch else valid_suffixes[expanded_last.astype(np.int64), local_suffix_indices.astype(np.int64)]
    return braid_indices.long() if backend.is_torch else braid_indices, suffix_indices.long() if backend.is_torch else suffix_indices.astype(np.int64)


def apply_simple_actions(states, suffix_indices, simple_actions, p: int, backend):
    lib = backend.lib
    parent = states.to(lib.int32) if backend.is_torch else states.astype(np.int32, copy=False)
    actions = simple_actions[suffix_indices]
    batch, _, width = parent.shape
    out = backend.zeros((batch, 3, width + 1), dtype=np.int32)

    coeff0 = actions[..., 0].to(lib.int32) if backend.is_torch else actions[..., 0].astype(np.int32, copy=False)
    coeff1 = actions[..., 1].to(lib.int32) if backend.is_torch else actions[..., 1].astype(np.int32, copy=False)

    if backend.is_torch and backend.device == "cuda":
        # CUDA einsum/bmm kernels do not support int32 here on torch 2.5,
        # so do the tiny 3x3 contractions in float32 and cast back exactly.
        parent_f32 = parent.to(lib.float32)
        coeff0_f32 = coeff0.to(lib.float32)
        coeff1_f32 = coeff1.to(lib.float32)
        out[:, :, :width] += lib.einsum("boi,biw->bow", coeff0_f32, parent_f32).round().to(lib.int32)
        out[:, :, 1:] += lib.einsum("boi,biw->bow", coeff1_f32, parent_f32).round().to(lib.int32)
    else:
        out[:, :, :width] += lib.einsum("boi,biw->bow", coeff0, parent)
        out[:, :, 1:] += lib.einsum("boi,biw->bow", coeff1, parent)
    if p:
        out %= p
    return out


def normalize_and_score(states, state_width: int, base_vertex: int, backend):
    lib = backend.lib
    alive_mask = lib.any(states != 0, dim=1) if backend.is_torch else np.any(states != 0, axis=1)
    alive_rows = lib.any(alive_mask, dim=1) if backend.is_torch else np.any(alive_mask, axis=1)
    if int(backend.to_numpy(alive_rows.sum())) == 0:
        if backend.is_torch:
            empty_states = states[:0, :, :state_width]
            empty_i32 = np.empty(0, dtype=np.int32)
            empty_bool = np.empty(0, dtype=bool)
            return empty_states, empty_i32, empty_bool, empty_bool
        return states[:0, :, :state_width], np.empty(0, dtype=np.int32), np.empty(0, dtype=bool), np.empty(0, dtype=bool)

    filtered = states[alive_rows]
    filtered_mask = alive_mask[alive_rows]
    min_idx = lib.argmax(filtered_mask.to(lib.int64) if backend.is_torch else filtered_mask.astype(np.int64), dim=1 if backend.is_torch else 1)
    reversed_mask = lib.flip(filtered_mask, dims=(1,)) if backend.is_torch else np.flip(filtered_mask, axis=1)
    max_idx = filtered.shape[2] - 1 - lib.argmax(reversed_mask.to(lib.int64) if backend.is_torch else reversed_mask.astype(np.int64), dim=1 if backend.is_torch else 1)
    spreads = max_idx - min_idx

    padded = lib.cat([filtered, backend.zeros((filtered.shape[0], filtered.shape[1], state_width), dtype=np.int32)], dim=2) if backend.is_torch else np.concatenate([filtered, np.zeros((filtered.shape[0], filtered.shape[1], state_width), dtype=np.int32)], axis=2)
    gather_idx = (lib.arange(state_width, device=backend.device, dtype=lib.int64)[None, :] + min_idx[:, None].to(lib.int64)) if backend.is_torch else (np.arange(state_width, dtype=np.int64)[None, :] + np.asarray(min_idx, dtype=np.int64)[:, None])
    normalized = backend.gather_last_axis(padded, gather_idx)

    degree0 = normalized[:, :, 0]
    target = np.zeros((3,), dtype=np.int32)
    target[base_vertex - 1] = 1
    if backend.is_torch:
        target_t = lib.tensor(target, dtype=lib.int32, device=backend.device)
        same_curve = (spreads == 0) & lib.all(degree0 == target_t[None, :], dim=1)
        return normalized, backend.to_numpy(spreads).astype(np.int32), backend.to_numpy(alive_rows).astype(bool), backend.to_numpy(same_curve).astype(bool)
    same_curve = (spreads == 0) & np.all(degree0 == target[None, :], axis=1)
    return normalized, np.asarray(spreads, dtype=np.int32), np.asarray(alive_rows, dtype=bool), same_curve.astype(bool)


def gather_level_braids(buckets, total_cap: int, use_best: int):
    if not buckets:
        return None
    sorted_spreads = sorted(buckets.keys())
    chosen_spreads = []
    total = 0
    for spread in sorted_spreads:
        count = buckets[spread][0].shape[0]
        if total == 0 or total + count <= total_cap:
            chosen_spreads.append(spread)
            total += count
        else:
            break

    states = []
    words = []
    lengths = []
    last_simple = []
    for spread in chosen_spreads:
        st, wd, ln, ls = buckets[spread]
        states.append(st)
        words.append(wd)
        lengths.append(ln)
        last_simple.append(ls)

    states_all = states[0] if len(states) == 1 else states[0].__class__.cat(states, dim=0) if hasattr(states[0].__class__, "cat") else None
    return states, words, lengths, last_simple


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = get_backend(args.backend, args.device)
    if backend.kind != "torch":
        raise RuntimeError("The GPU-native search is intended for torch backend; use --backend torch.")

    lib = backend.lib
    if backend.device == "cuda":
        lib.manual_seed(args.seed)
        lib.cuda.manual_seed_all(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    tables = build_tables(args.max_g_length, args.base_vertex, args.p)
    state_width = args.max_g_length + 3
    simple_actions_np = build_simple_action_tensors(tables.simple_words, args.p)
    simple_actions = lib.tensor(simple_actions_np, dtype=lib.int16, device=backend.device)
    valid_suffixes = lib.tensor(tables.valid_suffixes, dtype=lib.int16, device=backend.device)
    num_valid_suffixes = lib.tensor(tables.num_valid_suffixes, dtype=lib.int16, device=backend.device)

    current_buckets = GPUSpreadBuckets(args.cap_1, backend)
    init_states = lib.tensor(tables.init_states, dtype=lib.int16, device=backend.device)
    init_words = lib.tensor(tables.init_words, dtype=lib.int16, device=backend.device)
    init_lengths = lib.tensor(tables.init_lengths, dtype=lib.int16, device=backend.device)
    init_last = lib.tensor(tables.init_last, dtype=lib.int16, device=backend.device)

    init_spreads = np.array([1] * init_states.shape[0], dtype=np.int32)
    current_buckets.add_chunk(init_states, init_words, init_lengths, init_last, lib.tensor(init_spreads, dtype=lib.int32, device=backend.device))
    hits = []

    print(f"Backend: {backend.kind} on {backend.device}")
    print(f"Built {len(tables.simple_words)} Garside simples for A3.")
    print(f"Starting GPU-native search over Z/{args.p}Z with base vertex {args.base_vertex}.")

    for current_length in range(2, args.max_g_length + 1):
        previous = current_buckets.get_bucket_items()
        if not previous:
            break

        step_start = time.time()
        total_cap = args.total_cap_1 if current_length < args.first_steps else args.total_cap_2
        cap = args.cap_1 if current_length < args.first_steps else args.cap_2
        current_buckets = GPUSpreadBuckets(cap, backend)

        sorted_spreads = sorted(previous.keys())
        selected_spreads = []
        selected_count = 0
        for spread in sorted_spreads:
            count = previous[spread][0].shape[0]
            if selected_count == 0 or selected_count + count <= total_cap:
                selected_spreads.append(spread)
                selected_count += count
            else:
                break

        state_parts = []
        word_parts = []
        length_parts = []
        last_parts = []
        spread_parts = []
        for spread in selected_spreads:
            states, words, lengths, last_simple = previous[spread]
            state_parts.append(states)
            word_parts.append(words)
            length_parts.append(lengths)
            last_parts.append(last_simple)
            spread_parts.append(spread)

        flat_states = lib.cat(state_parts, dim=0)
        flat_words = lib.cat(word_parts, dim=0)
        flat_lengths = lib.cat(length_parts, dim=0)
        flat_last = lib.cat(last_parts, dim=0)
        flat_prev_spreads = np.concatenate([np.full(previous[spread][0].shape[0], spread, dtype=np.int32) for spread in selected_spreads])

        if args.use_best > 0 and flat_states.shape[0] > args.use_best:
            perm = lib.randperm(flat_states.shape[0], device=backend.device)[: args.use_best]
            flat_states = flat_states[perm]
            flat_words = flat_words[perm]
            flat_lengths = flat_lengths[perm]
            flat_last = flat_last[perm]
            flat_prev_spreads = flat_prev_spreads[backend.to_numpy(perm)]

        print(f"Step {current_length}: analyzing {flat_states.shape[0]} braids across spreads {selected_spreads[0]}..{selected_spreads[-1]}")

        braid_indices, suffix_indices = build_expansion_indices(flat_last, num_valid_suffixes, valid_suffixes, backend)
        if braid_indices.shape[0] == 0:
            print(f"Step {current_length}: no admissible expansions")
            break

        step_hits = 0
        spread_drops = 0
        admissible_spread = args.max_g_length - current_length + 1

        for start_idx in range(0, braid_indices.shape[0], args.expansion_chunk_size):
            end_idx = min(start_idx + args.expansion_chunk_size, braid_indices.shape[0])
            chunk_braid_idx = braid_indices[start_idx:end_idx]
            chunk_suffix_idx = suffix_indices[start_idx:end_idx]

            parent_states = flat_states[chunk_braid_idx]
            parent_words = flat_words[chunk_braid_idx]
            parent_lengths = flat_lengths[chunk_braid_idx]
            parent_prev_spreads = flat_prev_spreads[backend.to_numpy(chunk_braid_idx)]

            expanded_states = apply_simple_actions(parent_states, chunk_suffix_idx, simple_actions, args.p, backend)
            normalized, spread_np, alive_np, same_curve_np = normalize_and_score(expanded_states, state_width, args.base_vertex, backend)
            if normalized.shape[0] == 0:
                continue

            alive_indices = np.flatnonzero(alive_np)
            parent_words_alive = parent_words[lib.tensor(alive_indices, dtype=lib.int64, device=backend.device)]
            parent_lengths_alive = parent_lengths[lib.tensor(alive_indices, dtype=lib.int64, device=backend.device)]
            suffix_alive = chunk_suffix_idx[lib.tensor(alive_indices, dtype=lib.int64, device=backend.device)]
            prev_spreads_alive = parent_prev_spreads[alive_indices]

            new_words = parent_words_alive.clone()
            row_idx = lib.arange(parent_lengths_alive.shape[0], device=backend.device)
            new_words[row_idx, parent_lengths_alive.long()] = suffix_alive.to(lib.int16)
            new_lengths = parent_lengths_alive + 1
            new_last = suffix_alive.to(lib.int16)

            step_hits += int(same_curve_np.sum())
            hit_local = np.flatnonzero(same_curve_np)
            if hit_local.size:
                hit_idx = lib.tensor(hit_local, dtype=lib.int64, device=backend.device)
                hit_words = backend.to_numpy(new_words[hit_idx]).astype(np.int16)
                hit_lengths = backend.to_numpy(new_lengths[hit_idx]).astype(np.int16)
                for word_row, length in zip(hit_words, hit_lengths):
                    simple_indices = [int(x) for x in word_row[:length] if int(x) >= 0]
                    simple_indices = list(reversed(simple_indices))
                    hits.append(
                        {
                            "simple_indices": simple_indices,
                            "normal_form_factors": [tables.simple_words[idx] for idx in simple_indices],
                            "artin_word": [letter for idx in simple_indices for letter in tables.simple_words[idx]],
                        }
                    )

            spread_drops += int((spread_np < prev_spreads_alive).sum())
            keep_mask = spread_np <= admissible_spread
            if not keep_mask.any():
                continue

            keep_local = np.flatnonzero(keep_mask)
            keep_idx = lib.tensor(keep_local, dtype=lib.int64, device=backend.device)
            current_buckets.add_chunk(
                normalized[keep_idx].to(lib.int16),
                new_words[keep_idx].to(lib.int16),
                new_lengths[keep_idx].to(lib.int16),
                new_last[keep_idx].to(lib.int16),
                lib.tensor(spread_np[keep_local], dtype=lib.int32, device=backend.device),
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
            break

        if args.stop_at_first and step_hits > 0:
            break

    payload = {
        "parameters": {
            "p": args.p,
            "max_g_length": args.max_g_length,
            "backend": backend.kind,
            "device": backend.device,
            "cap_1": args.cap_1,
            "cap_2": args.cap_2,
            "total_cap_1": args.total_cap_1,
            "total_cap_2": args.total_cap_2,
            "first_steps": args.first_steps,
            "base_vertex": args.base_vertex,
            "seed": args.seed,
            "expansion_chunk_size": args.expansion_chunk_size,
            "use_best": args.use_best,
        },
        "num_garside_simples": len(tables.simple_words),
        "num_hits": len(hits),
        "runtime_seconds": time.time() - start,
        "hits": hits,
    }

    with args.output.open("w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote {len(hits)} hits to {args.output}")


if __name__ == "__main__":
    main()
