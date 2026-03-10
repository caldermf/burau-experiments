from __future__ import annotations

import unittest

import torch

import setup_a3 as ct
from a3_gpu_burau import apply_operator, compile_simple_operators, normalize_states
from a3_gpu_search import SearchConfig, run_search
from a3_gpu_tables import build_a3_table_data


def make_fp(pol, modulus: int):
    out = {}
    for degree, coeff in pol.items():
        value = coeff % modulus
        if value != 0:
            out[degree] = value
    return out


def make_fp_vec(vec, modulus: int):
    return [make_fp(pol, modulus) for pol in vec]


def normalize_vec(vec):
    valuation = min((min(pol.keys()) for pol in vec if pol), default=None)
    if valuation is None:
        return vec
    out = []
    for pol in vec:
        out.append({degree - valuation: coeff for degree, coeff in pol.items()})
    return out


def apply_word_cpu(word, vec, modulus: int):
    out = vec
    for letter in reversed(word):
        out = ct.oburau_fns[letter](out)
        out = make_fp_vec(out, modulus)
    return normalize_vec(out)


def spread_vec(vec):
    return ct.topdeg_vector(vec) - ct.botdeg_vector(vec)


def vec_to_tensor(vec, width: int, modulus: int):
    tensor = torch.zeros((1, len(vec), width), dtype=torch.int32)
    for row, pol in enumerate(vec):
        for degree, coeff in pol.items():
            tensor[0, row, degree] = coeff % modulus
    return tensor


def reference_spread_counts(modulus: int, max_depth: int):
    tables = build_a3_table_data(modulus=modulus, device=torch.device("cpu"))
    depth_counts = {}
    buckets = {}

    for simple_id in tables.start_simple_ids:
        vec = apply_word_cpu(tables.simple_words[simple_id], ct.dim_vectors[1], modulus)
        spread = spread_vec(vec)
        if spread == 1:
            buckets.setdefault(spread, []).append((simple_id, vec))

    depth_counts[1] = {spread: len(entries) for spread, entries in sorted(buckets.items())}

    for depth in range(2, max_depth + 1):
        next_buckets = {}
        for prev_spread, entries in buckets.items():
            for last_simple_id, vec in entries:
                for next_simple_id in tables.allowed_successors[last_simple_id]:
                    candidate = apply_word_cpu(tables.simple_words[next_simple_id], vec, modulus)
                    spread = spread_vec(candidate)
                    if depth == 2 and spread == 0:
                        continue
                    if spread <= max_depth - depth + 1:
                        next_buckets.setdefault(spread, []).append((next_simple_id, candidate))
        buckets = next_buckets
        depth_counts[depth] = {spread: len(entries) for spread, entries in sorted(buckets.items())}

    return depth_counts


class A3GpuReferenceTests(unittest.TestCase):
    def test_simple_operators_match_cpu_on_e1(self):
        modulus = 5
        device = torch.device("cpu")
        tables = build_a3_table_data(modulus=modulus, device=device)
        operators = compile_simple_operators(tables.simple_words, device=device)
        width = 16

        e1 = torch.zeros((1, 3, width), dtype=torch.int32)
        e1[0, 0, 0] = 1

        for simple_id, word in enumerate(tables.simple_words):
            gpu_state = apply_operator(e1, operators[simple_id], modulus)
            gpu_state, _ = normalize_states(gpu_state, modulus)

            cpu_vec = apply_word_cpu(word, ct.dim_vectors[1], modulus)
            cpu_state = vec_to_tensor(cpu_vec, width, modulus)

            self.assertTrue(torch.equal(gpu_state.cpu(), cpu_state), msg=f"Mismatch for simple {word}")

    def test_search_counts_match_reference_for_small_depth(self):
        modulus = 5
        max_depth = 4
        reference = reference_spread_counts(modulus=modulus, max_depth=max_depth)

        result = run_search(
            SearchConfig(
                cap_1=10_000,
                cap_2=10_000,
                total_cap_1=10_000,
                total_cap_2=10_000,
                first_steps=10,
                modulus=modulus,
                max_g_length=max_depth,
                device="cpu",
                seed=0,
            )
        )

        self.assertEqual(result.spread_counts_by_depth, reference)
        self.assertFalse(result.found)


if __name__ == "__main__":
    unittest.main()
