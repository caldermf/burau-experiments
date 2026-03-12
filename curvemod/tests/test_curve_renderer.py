from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from curve_renderer import RenderStyle, curve_endpoints_from_burau, curve_pair_from_word, parse_artin_word, render_curve_pair_with_style


def max_distance_to_segment(polyline: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    norm_sq = float(np.dot(segment, segment))
    if norm_sq == 0.0:
        return float(np.max(np.linalg.norm(polyline - start, axis=1)))
    offsets = polyline - start
    params = np.clip((offsets @ segment) / norm_sq, 0.0, 1.0)
    projections = start + np.outer(params, segment)
    return float(np.max(np.linalg.norm(polyline - projections, axis=1)))


class CurveRendererTests(unittest.TestCase):
    def test_parse_artin_word_accepts_generator_notation(self):
        self.assertEqual(parse_artin_word("s2 s3^-1 s1"), [2, -3, 1])
        self.assertEqual(parse_artin_word("2 -3 1"), [2, -3, 1])

    def test_identity_renders_p1(self):
        pair = curve_pair_from_word([])
        self.assertEqual(pair.moved_endpoint_pair, (1, 2))
        self.assertTrue(np.allclose(pair.base_polyline, pair.moved_polyline))

    def test_sigma1_preserves_p1_as_set(self):
        pair = curve_pair_from_word([1])
        max_deviation = max_distance_to_segment(
            pair.moved_polyline,
            np.array([0.0, 0.0], dtype=float),
            np.array([1.0, 0.0], dtype=float),
        )
        self.assertLess(max_deviation, 1e-8)
        self.assertEqual(pair.moved_endpoint_pair, (1, 2))

    def test_sigma2_moves_p1_to_arc_between_1_and_3(self):
        for word in ([2], [-2]):
            pair = curve_pair_from_word(word)
            self.assertEqual(pair.moved_endpoint_pair, (1, 3))

    def test_inverse_cancellation_returns_to_p1(self):
        pair = curve_pair_from_word([2, -2])
        self.assertEqual(pair.moved_endpoint_pair, (1, 2))
        self.assertLess(np.max(np.linalg.norm(pair.moved_polyline - pair.base_polyline, axis=1)), 1e-8)

    def test_braid_relation_signature_matches(self):
        left = curve_pair_from_word([1, 2, 1])
        right = curve_pair_from_word([2, 1, 2])
        self.assertEqual(left.moved_endpoint_pair, right.moved_endpoint_pair)
        self.assertEqual(left.wall_crossing_signature, right.wall_crossing_signature)

    def test_sigma1_sigma3_commute_on_invariants(self):
        left = curve_pair_from_word([1, 3])
        right = curve_pair_from_word([3, 1])
        self.assertEqual(left.moved_endpoint_pair, right.moved_endpoint_pair)
        self.assertEqual(left.wall_crossing_signature, right.wall_crossing_signature)

    def test_refinement_stability(self):
        coarse = curve_pair_from_word([2, -1, 3], sampling_tolerance=1e-2, max_segment_length=0.15)
        fine = curve_pair_from_word([2, -1, 3], sampling_tolerance=2e-4, max_segment_length=0.03)
        self.assertEqual(coarse.moved_endpoint_pair, fine.moved_endpoint_pair)
        self.assertEqual(coarse.wall_crossing_signature, fine.wall_crossing_signature)

    def test_burau_endpoint_consistency_on_short_words(self):
        words = [
            [],
            [1],
            [-1],
            [2],
            [-2],
            [3],
            [-3],
            [2, -1, 3],
            [1, 2, 1],
            [3, -2, -1],
        ]
        for word in words:
            pair = curve_pair_from_word(word)
            self.assertEqual(pair.moved_endpoint_pair, tuple(sorted(curve_endpoints_from_burau(word))))

    def test_render_saves_svg(self):
        pair = curve_pair_from_word([2, -1, 3])
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "curve.svg"
            figure = render_curve_pair_with_style(pair, RenderStyle(show_base_curve=False), out_path=str(out_path))
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)
            figure.clf()


if __name__ == "__main__":
    unittest.main()
