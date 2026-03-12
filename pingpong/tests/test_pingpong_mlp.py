import unittest

import numpy as np

import pingpong_mlp


class PingPongMLPTests(unittest.TestCase):
    def test_apply_alternating_words_matches_manual_products(self) -> None:
        A, B = pingpong_mlp.build_generators(1.5, dtype_name="float64")
        exponents, A_powers, B_powers = pingpong_mlp.precompute_powers(A, B, power_bound=2)
        exponent_to_index = {int(exp): idx for idx, exp in enumerate(exponents)}

        starting_vector = np.array([1.0, 1.0, 0.0], dtype=np.float64)
        start_generators = np.array([0, 1], dtype=np.int64)
        lengths = np.array([3, 4], dtype=np.int64)
        power_choice_indices = np.zeros((2, 4), dtype=np.int64)

        # Sample 0: A^1 B^-2 A^2 v
        power_choice_indices[0, 0] = exponent_to_index[1]
        power_choice_indices[0, 1] = exponent_to_index[-2]
        power_choice_indices[0, 2] = exponent_to_index[2]

        # Sample 1: B^-1 A^2 B^1 A^-1 v
        power_choice_indices[1, 0] = exponent_to_index[-1]
        power_choice_indices[1, 1] = exponent_to_index[2]
        power_choice_indices[1, 2] = exponent_to_index[1]
        power_choice_indices[1, 3] = exponent_to_index[-1]

        points, labels = pingpong_mlp.apply_alternating_words(
            starting_vector,
            start_generators,
            lengths,
            power_choice_indices,
            A_powers,
            B_powers,
        )

        manual_0 = (
            np.linalg.matrix_power(A, 2)
            @ np.linalg.matrix_power(B, -2)
            @ np.linalg.matrix_power(A, 1)
            @ starting_vector
        )
        manual_1 = (
            np.linalg.matrix_power(A, -1)
            @ np.linalg.matrix_power(B, 1)
            @ np.linalg.matrix_power(A, 2)
            @ np.linalg.matrix_power(B, -1)
            @ starting_vector
        )

        self.assertTrue(np.allclose(points[0], manual_0))
        self.assertTrue(np.allclose(points[1], manual_1))
        self.assertEqual(int(labels[0]), 0)
        self.assertEqual(int(labels[1]), 0)

    def test_generate_dataset_returns_finite_points(self) -> None:
        config = pingpong_mlp.GeneratorConfig(
            v=1.5,
            starting_vector=(1.0, 1.0, 0.0),
            power_bound=2,
            min_length=1,
            max_length=6,
            num_samples=512,
            chunk_size=128,
            seed=7,
        )
        dataset = pingpong_mlp.generate_dataset(config)

        self.assertEqual(dataset["points"].shape, (512, 3))
        self.assertEqual(dataset["labels"].shape, (512,))
        self.assertTrue(np.isfinite(dataset["points"]).all())
        self.assertEqual(set(np.unique(dataset["labels"])), {0, 1})


if __name__ == "__main__":
    unittest.main()
