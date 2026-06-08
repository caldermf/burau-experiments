from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from d4_gpu_common import RANK, build_letter_rules, commutator_word
from tensor_backend import get_backend


def parse_args():
    parser = argparse.ArgumentParser(description="Verify D4 Burau-kernel candidates from d4_mod_p_gpu_native_search.py.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--include-words", action="store_true", help="Include beta and commutator words in verifier output.")
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


def shift_row_polys(rows, shift: int, backend):
    out = backend.zeros(rows.shape, dtype=np.int32)
    if shift == -1:
        out[:, :, :-1] = rows[:, :, 1:]
    elif shift == 0:
        out[:, :, :] = rows
    elif shift == 1:
        out[:, :, 1:] = rows[:, :, :-1]
    else:
        raise ValueError(f"Unsupported shift {shift}")
    return out


def apply_letter_batch(mats, letter: int, row_rules, p: int, backend):
    rule = row_rules[letter]
    updated = backend.zeros((mats.shape[0], RANK, mats.shape[3]), dtype=np.int32)
    for source_row, shift, coeff in rule["terms"]:
        shifted = shift_row_polys(mats[:, source_row, :, :], shift, backend)
        updated = updated + coeff * shifted

    new_mats = clone_tensor(mats, backend)
    new_mats[:, rule["row"], :, :] = updated
    if p != 0:
        new_mats %= p
    return new_mats


def word_degree_bound(words):
    max_len = max((len(word) for word in words), default=0)
    return max_len + 4


def identity_matrix_tensor(batch_size: int, width: int, offset: int, backend):
    mats = backend.zeros((batch_size, RANK, RANK, width), dtype=np.int32)
    for i in range(RANK):
        mats[:, i, i, offset] = 1
    return mats


def apply_word_batch(words, p: int, row_rules, backend):
    bound = word_degree_bound(words)
    width = 2 * bound + 1
    offset = bound
    mats = identity_matrix_tensor(len(words), width, offset, backend)
    max_len = max((len(word) for word in words), default=0)

    for step in range(max_len):
        active_rows = []
        active_letters = []
        for idx, word in enumerate(words):
            if step < len(word):
                active_rows.append(idx)
                active_letters.append(word[-1 - step])

        if not active_rows:
            continue

        active_rows_np = np.array(active_rows, dtype=np.int64)
        active_letters_np = np.array(active_letters, dtype=np.int64)
        for letter in sorted(set(active_letters)):
            local = np.flatnonzero(active_letters_np == letter)
            local_rows = active_rows_np[local]
            local_batch = select_rows(mats, local_rows, backend)
            updated = apply_letter_batch(local_batch, int(letter), row_rules, p, backend)
            if backend.is_torch:
                idx = backend.lib.tensor(local_rows, dtype=backend.lib.int64, device=backend.device)
                mats[idx] = updated
            else:
                mats[local_rows] = updated

    return mats, offset


def is_identity_matrix_batch(mats, offset: int, backend):
    mats_np = backend.to_numpy(mats).astype(np.int32)
    identity = np.zeros_like(mats_np)
    for i in range(RANK):
        identity[:, i, i, offset] = 1
    return np.all(mats_np == identity, axis=(1, 2, 3))


def base_image_info(mats, offset: int, base_vertex: int, backend):
    mats_np = backend.to_numpy(mats).astype(np.int32)
    images = mats_np[:, :, base_vertex - 1, :]
    same_exact = []
    same_up_to_unit = []
    exponents = []
    coeffs = []

    for image in images:
        nz = np.argwhere(image != 0)
        if nz.shape[0] != 1:
            same_exact.append(False)
            same_up_to_unit.append(False)
            exponents.append(None)
            coeffs.append(None)
            continue

        row, degree = (int(nz[0, 0]), int(nz[0, 1]))
        coeff = int(image[row, degree])
        exponents.append(degree - offset)
        coeffs.append(coeff)
        same_exact.append(row == base_vertex - 1 and coeff == 1)
        same_up_to_unit.append(row == base_vertex - 1 and coeff != 0)

    return same_exact, same_up_to_unit, exponents, coeffs


def verify_hits(data, backend, batch_size: int, include_words: bool):
    p = int(data["parameters"]["p"])
    base_vertex = int(data["parameters"]["base_vertex"])
    row_rules = build_letter_rules()
    hits = data.get("hits", [])

    results = []
    identity_count = 0
    same_curve_count = 0
    same_curve_up_to_unit_count = 0
    start = time.time()

    for chunk_start in range(0, len(hits), batch_size):
        chunk = hits[chunk_start : chunk_start + batch_size]
        beta_words = [hit["artin_word"] for hit in chunk]
        beta_mats, beta_offset = apply_word_batch(beta_words, p, row_rules, backend)
        same_exact, same_unit, exponents, coeffs = base_image_info(beta_mats, beta_offset, base_vertex, backend)

        commutators = [commutator_word(word, base_vertex) for word in beta_words]
        commutator_mats, commutator_offset = apply_word_batch(commutators, p, row_rules, backend)
        identity_mask = is_identity_matrix_batch(commutator_mats, commutator_offset, backend)

        for local_idx, is_identity in enumerate(identity_mask):
            if is_identity:
                identity_count += 1
            if same_exact[local_idx]:
                same_curve_count += 1
            if same_unit[local_idx]:
                same_curve_up_to_unit_count += 1

            result = {
                "hit_index": chunk_start + local_idx,
                "commutator_is_identity": bool(is_identity),
                "beta_alpha_base_is_q_power_alpha_base": bool(same_exact[local_idx]),
                "beta_alpha_base_is_unit_q_power_alpha_base": bool(same_unit[local_idx]),
                "base_image_q_exponent": exponents[local_idx],
                "base_image_coefficient": coeffs[local_idx],
            }
            if include_words:
                result["beta_word"] = beta_words[local_idx]
                result["commutator_word"] = commutators[local_idx]
            results.append(result)

    return {
        "verified_commutator_identity_count": identity_count,
        "verified_same_curve_count": same_curve_count,
        "verified_same_curve_up_to_unit_count": same_curve_up_to_unit_count,
        "total_candidates": len(hits),
        "verification_runtime_seconds": time.time() - start,
        "results": results,
    }


def main():
    args = parse_args()
    backend = get_backend(args.backend, args.device)

    with args.input.open() as handle:
        data = json.load(handle)

    summary = verify_hits(data, backend, args.batch_size, args.include_words)
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

    print(
        "Verified "
        f"{payload['verified_commutator_identity_count']} / {payload['total_candidates']} "
        "candidates as Burau-kernel commutators."
    )
    print(
        "Same-curve condition: "
        f"{payload['verified_same_curve_count']} exact, "
        f"{payload['verified_same_curve_up_to_unit_count']} up to unit."
    )
    print(f"Verification runtime: {payload['verification_runtime_seconds']:.6f}s")


if __name__ == "__main__":
    main()
