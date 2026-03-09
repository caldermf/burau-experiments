from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tensor_backend import get_backend


def parse_args():
    parser = argparse.ArgumentParser(description="Verify Burau-kernel membership for witness braids.")
    parser.add_argument("--input", type=Path, required=True, help="JSON file produced by a3_mod_p_bucket_search.py")
    parser.add_argument("--output", type=Path, default=None, help="Optional verification JSON output")
    parser.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto", help="Tensor backend")
    parser.add_argument("--device", default="auto", help="Backend device")
    parser.add_argument("--batch-size", type=int, default=256, help="Number of candidates per verification batch")
    return parser.parse_args()


def clone_tensor(x, backend):
    return x.clone() if backend.is_torch else x.copy()


def select_rows(x, indices, backend):
    if len(indices) == 0:
        return x[:0]
    if backend.is_torch:
        idx = backend.lib.tensor(indices, dtype=backend.lib.int64, device=backend.device)
        return x[idx]
    return x[indices]


def pad_shift(arr, shift, backend):
    batch, rows, cols, width = arr.shape
    out = backend.zeros((batch, rows, cols, width), dtype=np.int32)
    if shift == -1:
        out[:, :, :, :-1] = arr[:, :, :, 1:]
    elif shift == 0:
        out[:, :, :, :] = arr
    elif shift == 1:
        out[:, :, :, 1:] = arr[:, :, :, :-1]
    else:
        raise ValueError(f"Unsupported shift {shift}")
    return out


def build_row_rules():
    dynkin_graph = {1: {2}, 2: {1, 3}, 3: {2}}
    dynkin_ograph = {1: {2}, 2: {3}, 3: set()}

    def exists_edge(i, j):
        return j in dynkin_graph.get(i, set())

    def exists_oriented_edge(i, j):
        return j in dynkin_ograph.get(i, set())

    rules = {}
    for letter in [1, 2, 3, -1, -2, -3]:
        index = abs(letter) - 1
        terms = []
        terms.append((index, -1 if letter > 0 else 1, -1))
        for source_row in range(3):
            if source_row == index:
                continue
            if not exists_edge(index + 1, source_row + 1):
                continue
            if exists_oriented_edge(index + 1, source_row + 1):
                shift = 0 if letter > 0 else 1
            else:
                shift = -1 if letter > 0 else 0
            terms.append((source_row, shift, -1))
        rules[letter] = {"row": index, "terms": terms}
    return rules


def apply_letter_batch(mats, letter, row_rules, p, backend):
    rule = row_rules[letter]
    updated = backend.zeros((mats.shape[0], mats.shape[2], mats.shape[3]), dtype=np.int32)
    for source_row, shift, coeff in rule["terms"]:
        shifted = pad_shift(mats[:, source_row : source_row + 1, :, :], shift, backend)[:, 0, :, :]
        updated = updated + coeff * shifted

    new_mats = clone_tensor(mats, backend)
    new_mats[:, rule["row"], :, :] = updated
    if p != 0:
        new_mats = new_mats % p
    return new_mats


def invert_word(word):
    return [-x for x in reversed(word)]


def commutator_word(beta_word, base_vertex):
    return beta_word + [base_vertex] + invert_word(beta_word) + [-base_vertex]


def word_degree_bound(word):
    return len(word) + 4


def identity_matrix_tensor(batch_size, width, offset, backend):
    mats = backend.zeros((batch_size, 3, 3, width), dtype=np.int32)
    for i in range(3):
        mats[:, i, i, offset] = 1
    return mats


def apply_word_batch(words, p, row_rules, backend):
    max_len = max(len(word) for word in words)
    bound = word_degree_bound([None] * max_len)
    width = 2 * bound + 1
    offset = bound
    mats = identity_matrix_tensor(len(words), width, offset, backend)

    for step in range(max_len):
        letters = []
        active_rows = []
        for idx, word in enumerate(words):
            if step < len(word):
                letters.append(word[-1 - step])
                active_rows.append(idx)

        if not active_rows:
            continue

        active_rows = np.array(active_rows, dtype=np.int64)
        batch = select_rows(mats, active_rows, backend)
        for letter in sorted(set(letters)):
            local = np.flatnonzero(np.array(letters) == letter)
            local_rows = active_rows[local]
            local_batch = select_rows(mats, local_rows, backend)
            updated = apply_letter_batch(local_batch, int(letter), row_rules, p, backend)
            if backend.is_torch:
                idx = backend.lib.tensor(local_rows, dtype=backend.lib.int64, device=backend.device)
                mats[idx] = updated
            else:
                mats[local_rows] = updated

    return mats, offset


def is_identity_matrix_batch(mats, offset, backend):
    mats_np = backend.to_numpy(mats).astype(np.int32)
    batch_size = mats_np.shape[0]
    identity = np.zeros_like(mats_np)
    for i in range(3):
        identity[:, i, i, offset] = 1
    return np.all(mats_np == identity, axis=(1, 2, 3))


def verify_hits(data, backend, batch_size):
    p = data["parameters"]["p"]
    base_vertex = data["parameters"]["base_vertex"]
    row_rules = build_row_rules()
    hits = data["hits"]

    results = []
    num_identity = 0
    start = time.time()

    for chunk_start in range(0, len(hits), batch_size):
        chunk = hits[chunk_start : chunk_start + batch_size]
        commutators = [commutator_word(hit["artin_word"], base_vertex) for hit in chunk]
        mats, offset = apply_word_batch(commutators, p, row_rules, backend)
        identity_mask = is_identity_matrix_batch(mats, offset, backend)
        for local_idx, is_identity in enumerate(identity_mask):
            if is_identity:
                num_identity += 1
            results.append(
                {
                    "hit_index": chunk_start + local_idx,
                    "is_identity": bool(is_identity),
                    "beta_word": chunk[local_idx]["artin_word"],
                    "commutator_word": commutators[local_idx],
                }
            )

    return {
        "verified_identity_count": num_identity,
        "total_candidates": len(hits),
        "verification_runtime_seconds": time.time() - start,
        "results": results,
    }


def main():
    args = parse_args()
    backend = get_backend(args.backend, args.device)

    with args.input.open() as handle:
        data = json.load(handle)

    summary = verify_hits(data, backend, args.batch_size)
    payload = {
        "input_file": str(args.input),
        "backend": backend.kind,
        "device": backend.device,
        **summary,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as handle:
            json.dump(payload, handle, indent=2)

    print(f"Verified {payload['verified_identity_count']} / {payload['total_candidates']} candidates as Burau-kernel elements.")
    print(f"Verification runtime: {payload['verification_runtime_seconds']:.6f}s")


if __name__ == "__main__":
    main()
