from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b5r2_hecke import (
    DIMENSION,
    DELTA,
    artin_generator_matrix,
    commutator_is_identity,
    commutator_matrix,
    e_matrix,
    identity_matrix,
    matrix_projlen,
    multiply_matrices,
    _mul,
    reduce_matrix,
)
from b5r2_dual_tables import _eval_braid, build_b5r2_dual_table_data

try:
    import torch

    from b5r2_gpu_search import (
        SearchConfig,
        apply_operator,
        compile_left_matrix_operator,
        compile_right_matrix_operator,
        make_identity_state,
        pairwise_multiply_states,
        run_search,
    )

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    SearchConfig = None
    apply_operator = None
    compile_left_matrix_operator = None
    compile_right_matrix_operator = None
    make_identity_state = None
    pairwise_multiply_states = None
    run_search = None
    TORCH_AVAILABLE = False


def _flatten_matrix(matrix, width: int, modulus: int) -> torch.Tensor:
    tensor = torch.zeros((1, DIMENSION * DIMENSION, width), dtype=torch.int32)
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            flat = row * DIMENSION + col
            for degree, coeff in matrix[row][col].items():
                tensor[0, flat, degree] = coeff % modulus
    return tensor


def _commutator_word(word, generator: int = 1):
    return [generator] + list(word) + [-generator] + [-letter for letter in reversed(word)]


class B5R2HeckeTests(unittest.TestCase):
    def test_temperley_lieb_relations_hold_on_the_cell_basis(self):
        matrices = [e_matrix(index) for index in (1, 2, 3, 4)]

        def delta_times(matrix):
            out = [[{} for _ in range(DIMENSION)] for _ in range(DIMENSION)]
            for row in range(DIMENSION):
                for col in range(DIMENSION):
                    out[row][col] = _mul(DELTA, matrix[row][col])
            return out

        for matrix in matrices:
            self.assertEqual(multiply_matrices(matrix, matrix), delta_times(matrix))

        for index in range(3):
            left = multiply_matrices(multiply_matrices(matrices[index], matrices[index + 1]), matrices[index])
            self.assertEqual(left, matrices[index])

        for left_index, right_index in ((0, 2), (0, 3), (1, 3)):
            self.assertEqual(
                multiply_matrices(matrices[left_index], matrices[right_index]),
                multiply_matrices(matrices[right_index], matrices[left_index]),
            )

    def test_generators_are_exact_inverses(self):
        identity = identity_matrix()
        for letter in (1, 2, 3, 4):
            forward = artin_generator_matrix(letter)
            backward = artin_generator_matrix(-letter)
            self.assertEqual(multiply_matrices(forward, backward), identity)
            self.assertEqual(multiply_matrices(backward, forward), identity)

    def test_commutator_identity_matches_group_expectation(self):
        self.assertTrue(commutator_is_identity([3], modulus=5, generator=1))
        self.assertFalse(commutator_is_identity([2], modulus=5, generator=1))
        self.assertEqual(matrix_projlen(commutator_matrix([3], modulus=5, generator=1)), 0)

    def test_dual_simple_admissibility_excludes_trivial_braid_commutators(self):
        tables = build_b5r2_dual_table_data(torch.device("cpu"))
        identity = _eval_braid([])

        level_paths = [(simple_id,) for simple_id in tables.start_simple_ids]
        all_paths = list(level_paths)
        for _ in range(2):
            next_level = []
            for path in level_paths:
                right_id = path[0]
                for left_id in tables.allowed_predecessors[right_id]:
                    next_level.append((left_id,) + path)
            all_paths.extend(next_level)
            level_paths = next_level

        for path in all_paths:
            word = []
            for simple_id in path:
                word.extend(tables.simple_words[simple_id])
            self.assertNotEqual(_eval_braid(_commutator_word(word)), identity)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class B5R2HeckeTorchTests(unittest.TestCase):
    def test_fixed_operator_application_matches_exact_matrix(self):
        modulus = 5
        width = 16
        device = torch.device("cpu")
        identity_state = make_identity_state(width=width, device=device)

        sigma = artin_generator_matrix(1)
        sigma_right = compile_right_matrix_operator(sigma)
        sigma_left = compile_left_matrix_operator(sigma)

        right_state = apply_operator(identity_state, sigma_right, modulus).cpu()
        left_state = apply_operator(identity_state, sigma_left, modulus).cpu()
        expected = _flatten_matrix(reduce_matrix(sigma, modulus), width=width, modulus=modulus)

        self.assertTrue(torch.equal(right_state, expected))
        self.assertTrue(torch.equal(left_state, expected))

    def test_pairwise_matrix_product_matches_exact_cpu_product(self):
        modulus = 5
        width = 16
        device = torch.device("cpu")
        sigma1 = _flatten_matrix(reduce_matrix(artin_generator_matrix(1), modulus), width, modulus).to(device)
        sigma3 = _flatten_matrix(reduce_matrix(artin_generator_matrix(3), modulus), width, modulus).to(device)
        product = pairwise_multiply_states(sigma1, sigma3, modulus=modulus, out_width=width).cpu()
        expected = _flatten_matrix(
            reduce_matrix(multiply_matrices(artin_generator_matrix(1), artin_generator_matrix(3)), modulus),
            width,
            modulus,
        )
        self.assertTrue(torch.equal(product, expected))

    def test_small_search_never_accepts_a_trivial_braid_commutator(self):
        result = run_search(
            SearchConfig(
                bucket_cap=128,
                total_cap=128,
                max_depth=3,
                modulus=2,
                device="cpu",
                seed=0,
                max_witnesses=5,
                exact_commutator_generator=1,
                require_sigma2=True,
            )
        )
        identity = _eval_braid([])
        for witness in result.witnesses:
            self.assertNotEqual(_eval_braid(_commutator_word(witness)), identity)
