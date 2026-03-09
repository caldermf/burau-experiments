from __future__ import annotations

import argparse
import ast
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import setup_a3 as ct
from tensor_backend import get_backend


def compute_oburau_vector(braid_word, base_vertex):
    vector = ct.dim_vectors[base_vertex]
    for letter in reversed(braid_word):
        vector = ct.oburau_fns[letter](vector)
    return vector


def compute_oburau_deg(braid_word):
    return max(ct.topdeg_vector(compute_oburau_vector(braid_word, base_vertex)) for base_vertex in ct.positive_letters)


def equal_braids(braid1, braid2):
    return all(
        compute_oburau_vector(braid1, base_vertex) == compute_oburau_vector(braid2, base_vertex)
        for base_vertex in ct.positive_letters
    )


def build_garside_gens():
    garside_gens = [[]] + ct.DUAL_ATOMS
    additions = 1
    while additions != 0:
        to_add = []
        for left in garside_gens:
            for right in garside_gens:
                candidate = left + right
                if compute_oburau_deg(candidate) > 1:
                    continue

                is_new = True
                for existing in to_add:
                    if equal_braids(existing, candidate):
                        if len(candidate) < len(existing):
                            to_add.remove(existing)
                            to_add.append(candidate)
                        is_new = False
                        break

                if not is_new:
                    continue

                for existing in garside_gens:
                    if equal_braids(existing, candidate):
                        if len(candidate) < len(existing):
                            garside_gens.remove(existing)
                            garside_gens.append(candidate)
                        is_new = False
                        break

                if is_new:
                    to_add.append(candidate)

        garside_gens += to_add
        additions = len(to_add)

    return garside_gens


def find_representative(braid_word, garside_gens):
    for candidate in garside_gens:
        if equal_braids(candidate, braid_word):
            return candidate
    raise ValueError(f"No Garside representative found for {braid_word}")


def build_descents(garside_gens):
    left_descents = {}
    right_descents = {}

    for descent in ct.DUAL_ATOMS:
        for gen in garside_gens:
            if compute_oburau_deg(gen + descent) <= 1:
                rep = find_representative(gen + descent, garside_gens)
                right_descents.setdefault(str(rep), []).append(descent)

            if compute_oburau_deg(descent + gen) <= 1:
                rep = find_representative(descent + gen, garside_gens)
                left_descents.setdefault(str(rep), []).append(descent)

    return left_descents, right_descents


def build_automaton(garside_gens, right_descents, gamma_rep):
    automaton = {}
    for target in garside_gens:
        automaton[str(target)] = []
        if target == [] or target == gamma_rep:
            continue

        for source in garside_gens:
            if source == [] or source == gamma_rep:
                continue

            admissible = True
            for descent in right_descents[str(source)]:
                if descent != [] and descent != gamma_rep and compute_oburau_deg(descent + target) <= 1:
                    admissible = False
                    break

            if admissible:
                automaton[str(target)].append(source)

    return automaton


def clone_tensor(x, backend):
    return x.clone() if backend.is_torch else x.copy()


def select_rows(x, indices, backend):
    if isinstance(indices, slice):
        return x[indices]
    if len(indices) == 0:
        return x[:0]
    if len(indices) > 1:
        start = indices[0]
        stop = indices[-1] + 1
        if stop - start == len(indices) and np.array_equal(indices, np.arange(start, stop)):
            return x[start:stop]
    if backend.is_torch:
        idx = backend.lib.tensor(indices, dtype=backend.lib.int64, device=backend.device)
        return x[idx]
    return x[indices]


def pad_shift_component(current, coord, shift, backend):
    batch_size, _, width = current.shape
    out = backend.zeros((batch_size, width), dtype=np.int32)
    if shift == -1:
        out[:, :-1] = current[:, coord, 1:]
    elif shift == 0:
        out[:, :] = current[:, coord, :]
    elif shift == 1:
        out[:, 1:] = current[:, coord, :-1]
    else:
        raise ValueError(f"Unsupported shift {shift}")
    return out


def normalize_states(states, state_width, backend):
    degree_mask = backend.any(states != 0, axis=1)
    alive_np = backend.to_numpy(backend.any(degree_mask, axis=1)).astype(bool)
    if not alive_np.any():
        return states[:0], np.empty(0, dtype=np.int32), alive_np

    alive_indices = np.flatnonzero(alive_np)
    alive_states = select_rows(states, alive_indices, backend)
    alive_mask = backend.any(alive_states != 0, axis=1)
    min_idx = backend.argmax(alive_mask, axis=1)
    flipped = backend.flip(alive_mask, axis=1)
    max_idx = alive_states.shape[2] - 1 - backend.argmax(flipped, axis=1)

    min_idx_np = backend.to_numpy(min_idx).astype(np.int32)
    spread_np = backend.to_numpy(max_idx - min_idx).astype(np.int32)

    padded = backend.concat(
        [alive_states, backend.zeros((alive_states.shape[0], alive_states.shape[1], alive_states.shape[2]), dtype=np.int32)],
        axis=2,
    )
    gather_idx = backend.arange(state_width, dtype=np.int32)[None, :] + backend.array(min_idx_np[:, None], dtype=np.int32)
    normalized = backend.gather_last_axis(padded, gather_idx)
    return normalized, spread_np, alive_np


def same_curve_mask(states, spread_np, base_vertex, backend):
    if len(spread_np) == 0:
        return np.empty(0, dtype=bool)
    degree0 = backend.to_numpy(states[:, :, 0]).astype(np.int32)
    target = np.zeros((degree0.shape[1],), dtype=np.int32)
    target[base_vertex - 1] = 1
    return (spread_np == 0) & np.all(degree0 == target[None, :], axis=1)


def make_fp_tensor(vec, p, backend):
    dense = np.zeros((len(vec), 1), dtype=np.int32)
    for i, poly in enumerate(vec):
        if poly:
            top_deg = max(poly)
            dense = np.pad(dense, ((0, 0), (0, max(0, top_deg + 1 - dense.shape[1]))))
            for degree, coeff in poly.items():
                dense[i, degree] = coeff % p if p else coeff
    return backend.array(dense[:, :, None].transpose(0, 2, 1), dtype=np.int32)


def dense_from_dim_vec(vec, state_width, p, backend):
    dense = np.zeros((3, state_width), dtype=np.int32)
    for i, poly in enumerate(vec):
        for degree, coeff in poly.items():
            if degree >= state_width:
                raise ValueError(f"Degree {degree} exceeds configured state width {state_width}")
            dense[i, degree] = coeff % p if p else coeff
    return backend.array(dense[None, :, :], dtype=np.int32)


def build_letter_rules():
    rules = {}
    for letter in ct.all_letters:
        index = abs(letter) - 1
        terms = []
        if letter > 0:
            terms.append((index, -1, -1))
        else:
            terms.append((index, 1, -1))

        for source_coord in range(3):
            if source_coord == index:
                continue
            if not ct.exists_dynkin_graph_edge(index + 1, source_coord + 1):
                continue
            if ct.exists_dynkin_ograph_edge(index + 1, source_coord + 1):
                shift = 0 if letter > 0 else 1
            else:
                shift = -1 if letter > 0 else 0
            terms.append((source_coord, shift, -1))
        rules[letter] = {"index": index, "terms": terms}
    return rules


def apply_simple_batch(states, simple_word, letter_rules, p, state_width, backend):
    if states.shape[0] == 0:
        return states, np.empty(0, dtype=np.int32), np.empty(0, dtype=bool)

    pad = len(simple_word)
    current_width = state_width + 2 * pad
    current = backend.zeros((states.shape[0], states.shape[1], current_width), dtype=np.int32)
    current[:, :, pad : pad + state_width] = states

    for letter in reversed(simple_word):
        rule = letter_rules[letter]
        updated = backend.zeros((current.shape[0], current_width), dtype=np.int32)
        for source_coord, shift, coeff in rule["terms"]:
            contribution = pad_shift_component(current, source_coord, shift, backend)
            updated = updated + coeff * contribution

        current[:, rule["index"], :] = updated
        current = backend.mod(current, p)

    normalized, spread_np, alive_np = normalize_states(current, state_width, backend)
    return normalized, spread_np, alive_np


@dataclass
class NodeStore:
    parents: list[int]
    simple_indices: list[int]

    def add(self, parent_id: int, simple_index: int) -> int:
        self.parents.append(parent_id)
        self.simple_indices.append(simple_index)
        return len(self.parents) - 1


@dataclass
class StateGroup:
    node_ids: np.ndarray
    states: any


def flatten_word(simple_indices, simple_words):
    word = []
    for idx in simple_indices:
        word.extend(simple_words[idx])
    return word


def reconstruct_factor_indices(parent_id, simple_index, nodes: NodeStore):
    factors = [simple_index]
    current = parent_id
    while current != -1:
        factors.append(nodes.simple_indices[current])
        current = nodes.parents[current]
    return factors


def initialize_groups(simple_words, right_descents_idx, gamma_idx, base_vertex, p, state_width, backend, nodes):
    buckets = {1: defaultdict(list)}
    for simple_idx, simple_word in enumerate(simple_words):
        if simple_word == [] or simple_idx == gamma_idx:
            continue

        admissible = True
        for descent_idx in right_descents_idx[simple_idx]:
            descent_vec = compute_oburau_vector(simple_words[descent_idx], base_vertex)
            normalized = ct.poly_normalize_vector(descent_vec)
            if normalized == ct.dim_vectors[base_vertex]:
                admissible = False
                break
            if ct.topdeg_vector(descent_vec) - ct.botdeg_vector(descent_vec) != 1:
                admissible = False
                break

        if not admissible:
            continue

        dense = dense_from_dim_vec(compute_oburau_vector(simple_word, base_vertex), state_width, p, backend)
        spread = int(ct.topdeg_vector(compute_oburau_vector(simple_word, base_vertex)) - ct.botdeg_vector(compute_oburau_vector(simple_word, base_vertex)))
        if spread != 1:
            continue

        node_id = nodes.add(-1, simple_idx)
        buckets[1][spread].append((simple_idx, node_id, dense))

    grouped = {1: {}}
    for spread, items in buckets[1].items():
        grouped[1][spread] = {}
        by_simple = defaultdict(list)
        for simple_idx, node_id, dense in items:
            by_simple[simple_idx].append((node_id, dense))
        for simple_idx, group_items in by_simple.items():
            node_ids = np.array([item[0] for item in group_items], dtype=np.int64)
            states = backend.concat([item[1] for item in group_items], axis=0)
            grouped[1][spread][simple_idx] = StateGroup(node_ids=node_ids, states=states)
    return grouped


def parse_args():
    parser = argparse.ArgumentParser(description="CUDA-friendly bucket search for Burau nonfaithfulness in type A3 modulo p.")
    parser.add_argument("--p", type=int, default=3, help="Work over Z/pZ. Use 0 for Z.")
    parser.add_argument("--max-g-length", type=int, default=100, help="Maximum Garside length to explore.")
    parser.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto", help="Tensor backend.")
    parser.add_argument("--device", default="auto", help="Backend device, e.g. cuda or cpu.")
    parser.add_argument("--cap-1", type=int, default=500, help="Per-spread bucket cap during early steps.")
    parser.add_argument("--cap-2", type=int, default=500, help="Per-spread bucket cap after early steps.")
    parser.add_argument("--total-cap-1", type=int, default=50000, help="Total curves retained during early steps.")
    parser.add_argument("--total-cap-2", type=int, default=50000, help="Total curves retained after early steps.")
    parser.add_argument("--first-steps", type=int, default=12, help="Number of early broad-exploration steps.")
    parser.add_argument("--base-vertex", type=int, default=1, choices=ct.positive_letters, help="Which simple root alpha_i to track.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--transition-batch-size", type=int, default=4096, help="Chunk size for batched simple applications.")
    parser.add_argument("--stop-at-first", action="store_true", help="Stop once a witness is found.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data" / "a3_bucket_hits.json", help="Where to write hits and metadata.")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    backend = get_backend(args.backend, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    simple_words = build_garside_gens()
    simple_to_index = {tuple(word): idx for idx, word in enumerate(simple_words)}
    gamma_rep = find_representative(ct.GAMMA_WORD, simple_words)
    gamma_idx = simple_to_index[tuple(gamma_rep)]
    _, right_descents = build_descents(simple_words)
    automaton = build_automaton(simple_words, right_descents, gamma_rep)

    right_descents_idx = {
        simple_to_index[tuple(source)]: [simple_to_index[tuple(desc)] for desc in descents]
        for source_key, descents in right_descents.items()
        for source in [ast.literal_eval(source_key)]
    }
    automaton_idx = {
        simple_to_index[tuple(source)]: [simple_to_index[tuple(target)] for target in targets]
        for source_key, targets in automaton.items()
        for source in [ast.literal_eval(source_key)]
        if source != []
    }

    max_simple_len = max(len(word) for word in simple_words)
    state_width = args.max_g_length + max_simple_len + 2
    letter_rules = build_letter_rules()

    nodes = NodeStore(parents=[], simple_indices=[])
    buckets = initialize_groups(simple_words, right_descents_idx, gamma_idx, args.base_vertex, args.p, state_width, backend, nodes)
    hits = []

    print(f"Backend: {backend.kind} on {backend.device or 'cpu'}")
    print(f"Built {len(simple_words)} Garside simples for A3.")
    print(f"Starting search over Z/{args.p}Z with base vertex {args.base_vertex}.")

    current_length = 1
    while current_length < args.max_g_length:
        previous = buckets[current_length]
        if not previous:
            break

        current_length += 1
        step_start = time.time()
        next_buckets = defaultdict(lambda: defaultdict(list))

        if current_length < args.first_steps:
            total_cap = args.total_cap_1
            cap = args.cap_1
        else:
            total_cap = args.total_cap_2
            cap = args.cap_2

        stop_key = min(previous.keys())
        counter = sum(group.node_ids.size for group in previous[stop_key].values())
        while counter < total_cap and stop_key + 1 in previous:
            stop_key += 1
            counter += sum(group.node_ids.size for group in previous[stop_key].values())
        keylist = [key for key in sorted(previous.keys()) if key <= stop_key]

        step_hits = 0
        spread_drops = 0
        admissible_spread = args.max_g_length - current_length + 1

        for previous_spread in keylist:
            num_states = sum(group.node_ids.size for group in previous[previous_spread].values())
            print(f"Step {current_length}: analyzing {num_states} braids in spread {previous_spread}")
            for source_idx, group in previous[previous_spread].items():
                for next_idx in automaton_idx.get(source_idx, []):
                    total = group.node_ids.size
                    for start_idx in range(0, total, args.transition_batch_size):
                        end_idx = min(start_idx + args.transition_batch_size, total)
                        chunk_states = select_rows(group.states, slice(start_idx, end_idx), backend)
                        chunk_parent_ids = group.node_ids[start_idx:end_idx]

                        normalized, spread_np, alive_np = apply_simple_batch(
                            chunk_states,
                            simple_words[next_idx],
                            letter_rules,
                            args.p,
                            state_width,
                            backend,
                        )

                        if not alive_np.any():
                            continue

                        alive_indices = np.flatnonzero(alive_np)
                        parent_alive = chunk_parent_ids[alive_indices]
                        same_np = same_curve_mask(normalized, spread_np, args.base_vertex, backend)
                        step_hits += int(same_np.sum())

                        for local_idx in np.flatnonzero(same_np):
                            hits.append((int(parent_alive[local_idx]), next_idx))

                        if previous_spread is not None:
                            spread_drops += int((spread_np < previous_spread).sum())

                        keep_mask = spread_np <= admissible_spread
                        if not keep_mask.any():
                            continue

                        keep_indices = np.flatnonzero(keep_mask)
                        kept_states = select_rows(normalized, keep_indices, backend)
                        kept_parents = parent_alive[keep_indices]
                        kept_spreads = spread_np[keep_indices]

                        for spread in np.unique(kept_spreads):
                            local = np.flatnonzero(kept_spreads == spread)
                            next_buckets[int(spread)][next_idx].append(
                                (
                                    kept_parents[local].copy(),
                                    select_rows(kept_states, local, backend),
                                )
                            )

        buckets[current_length] = {}
        for spread, by_simple in next_buckets.items():
            buckets[current_length][spread] = {}
            simple_arrays = []
            parent_arrays = []
            state_arrays = []
            for simple_idx, chunks in by_simple.items():
                parent_ids = np.concatenate([chunk[0] for chunk in chunks]) if chunks else np.empty(0, dtype=np.int64)
                states = backend.concat([chunk[1] for chunk in chunks], axis=0) if chunks else backend.zeros((0, 3, state_width), dtype=np.int32)
                if len(parent_ids) == 0:
                    continue
                simple_arrays.append(np.full(len(parent_ids), simple_idx, dtype=np.int64))
                parent_arrays.append(parent_ids)
                state_arrays.append(states)

            if not parent_arrays:
                continue

            flat_simple = np.concatenate(simple_arrays)
            flat_parent = np.concatenate(parent_arrays)
            flat_states = backend.concat(state_arrays, axis=0)

            if len(flat_parent) > cap:
                chosen = np.array(random.sample(range(len(flat_parent)), cap), dtype=np.int64)
                flat_simple = flat_simple[chosen]
                flat_parent = flat_parent[chosen]
                flat_states = select_rows(flat_states, chosen, backend)

            for simple_idx in np.unique(flat_simple):
                local = np.flatnonzero(flat_simple == simple_idx)
                node_ids = np.array([nodes.add(int(flat_parent[idx]), int(simple_idx)) for idx in local], dtype=np.int64)
                states = select_rows(flat_states, local, backend)
                buckets[current_length][spread][simple_idx] = StateGroup(node_ids=node_ids, states=states)

        buckets[current_length - 1] = {}
        elapsed = time.time() - step_start
        if buckets[current_length]:
            print(
                f"Finished step {current_length}: min spread {min(buckets[current_length].keys())}, "
                f"max spread {max(buckets[current_length].keys())}, drops {spread_drops}, "
                f"hits this step {step_hits}, runtime {elapsed:.2f}s"
            )
        else:
            print(
                f"Finished step {current_length}: no surviving buckets, drops {spread_drops}, "
                f"hits this step {step_hits}, runtime {elapsed:.2f}s"
            )
            break

        if args.stop_at_first and step_hits > 0:
            break

    serialized_hits = []
    for parent_id, simple_idx in hits:
        factor_indices = reconstruct_factor_indices(parent_id, simple_idx, nodes)
        serialized_hits.append(
            {
                "simple_indices": factor_indices,
                "normal_form_factors": [simple_words[idx] for idx in factor_indices],
                "artin_word": flatten_word(factor_indices, simple_words),
            }
        )

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
            "transition_batch_size": args.transition_batch_size,
        },
        "num_garside_simples": len(simple_words),
        "num_hits": len(serialized_hits),
        "runtime_seconds": time.time() - start,
        "hits": serialized_hits,
    }

    with args.output.open("w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote {len(serialized_hits)} hits to {args.output}")


if __name__ == "__main__":
    main()
