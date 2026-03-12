import unittest

from burau_modp_search import (
    WeightTuple,
    find_example_53,
    generate_candidates,
    is_right_terminal_state,
    iter_candidates_by_case,
    left_path_index,
    left_thresholds,
    precheck_candidate,
    reduce_poly_mod_p,
    right_path_index,
    right_thresholds,
)


class WeightTupleTests(unittest.TestCase):
    def test_secondary_weights_example_53(self) -> None:
        weights = WeightTuple(w0=22, w1=74, w2=89, w3=21, w14=192)
        self.assertEqual(weights.h, 96)
        self.assertEqual(weights.w8, 0)
        self.assertEqual(weights.w9, 148)
        self.assertEqual(weights.w10, 44)
        self.assertEqual(weights.w11, 164)
        self.assertEqual(weights.w12, 14)
        self.assertEqual(weights.w13, 28)

    def test_expected_intersections_example_53(self) -> None:
        weights = WeightTuple(w0=22, w1=74, w2=89, w3=21, w14=192)
        self.assertEqual(weights.expected_intersections(), 48)
        self.assertTrue(weights.is_admissible_arc_candidate())

    def test_page_13_thresholds_example_53(self) -> None:
        weights = WeightTuple(w0=22, w1=74, w2=89, w3=21, w14=192)
        self.assertEqual(left_thresholds(weights), (0, 74, 148, 170, 192, 118, 118, 44))
        self.assertEqual(right_thresholds(weights), (14, 21, 28, 103, 178))

    def test_initial_left_record_example_53(self) -> None:
        weights = WeightTuple(w0=22, w1=74, w2=89, w3=21, w14=192)
        self.assertEqual(left_path_index(weights, ell=weights.w2), 3)

    def test_right_terminal_detection(self) -> None:
        weights = WeightTuple(w0=22, w1=74, w2=89, w3=21, w14=192)
        self.assertTrue(is_right_terminal_state(weights, ell=21))
        self.assertTrue(is_right_terminal_state(weights, ell=255))
        self.assertFalse(is_right_terminal_state(weights, ell=20))
        self.assertFalse(is_right_terminal_state(weights, ell=178))

    def test_reduce_poly_mod_p(self) -> None:
        poly = {-3: -1, 0: 3, 2: 7}
        self.assertEqual(reduce_poly_mod_p(poly, p=5), {-3: 4, 0: 3, 2: 2})


class CandidateGenerationTests(unittest.TestCase):
    def test_example_53_is_generated(self) -> None:
        weights = find_example_53()
        self.assertIsNotNone(weights)
        self.assertEqual(weights, WeightTuple(w0=22, w1=74, w2=89, w3=21, w14=192))

    def test_p_equals_two_skips_odd_intersections(self) -> None:
        candidates = list(iter_candidates_by_case(p=2, max_intersections=5))
        self.assertTrue(candidates)
        self.assertTrue(all(candidate.intersections % 2 == 0 for candidate in candidates))

    def test_odd_prime_keeps_odd_intersections(self) -> None:
        candidates = list(iter_candidates_by_case(p=3, max_intersections=5))
        odd_levels = {candidate.intersections for candidate in candidates if candidate.intersections % 2 == 1}
        self.assertTrue(odd_levels)

    def test_generate_candidates_dedupes_overlapping_cases(self) -> None:
        all_keys = [weights.key() for weights in generate_candidates(p=3, max_intersections=8)]
        self.assertEqual(len(all_keys), len(set(all_keys)))

    def test_precheck_is_one_sided_safe(self) -> None:
        admissible = WeightTuple(w0=22, w1=74, w2=89, w3=21, w14=192)
        inadmissible = WeightTuple(w0=2, w1=2, w2=1, w3=1, w14=2)
        self.assertTrue(precheck_candidate(admissible, p=3))
        self.assertFalse(precheck_candidate(inadmissible, p=3))

    def test_right_path_index_falls_back_to_six(self) -> None:
        weights = WeightTuple(w0=22, w1=74, w2=89, w3=21, w14=192)
        self.assertEqual(right_path_index(weights, ell=178), 6)


if __name__ == "__main__":
    unittest.main()
