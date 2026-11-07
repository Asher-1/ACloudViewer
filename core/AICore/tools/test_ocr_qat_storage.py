"""QAT must train the deployed storage operator, not a different int8 scheme."""

import unittest
from types import SimpleNamespace

import gguf
import numpy as np

from ocr_qat_storage import storage_forward
from ocr_qat_graph import head_argmax_counts, teacher_head_feed


class StorageForwardTests(unittest.TestCase):
    def test_q8_matches_independent_gguf_bytes_including_scale_rounding(self):
        rng = np.random.default_rng(71)
        values = rng.normal(size=(9, 64, 1, 1)).astype(np.float32)
        values[0] = 0
        values[1] *= np.float32(1e-6)
        values[2] *= np.float32(37.17)
        expected = gguf.quants.dequantize(
            gguf.quants.quantize(values.reshape(9, 64), gguf.GGMLQuantizationType.Q8_0),
            gguf.GGMLQuantizationType.Q8_0).reshape(values.shape)
        np.testing.assert_array_equal(storage_forward(values, "Q8_0"), expected)

    def test_q8_halfway_rounds_away_from_zero(self):
        values = np.zeros((1, 32), np.float32)
        values[0, :5] = [127, .5, -.5, 1.5, -1.5]
        np.testing.assert_array_equal(storage_forward(values, "Q8_0")[0, :5],
                                      [127, 1, -1, 2, -2])

    def test_f16_matches_serialized_storage_and_f32_does_not_alias_master(self):
        value = np.array([[1.0007, -3.14159, 1e-8]], np.float32)
        np.testing.assert_array_equal(storage_forward(value, "F16"),
                                      value.astype(np.float16).astype(np.float32))
        result = storage_forward(value, "F32")
        np.testing.assert_array_equal(result, value)
        self.assertFalse(np.shares_memory(result, value))

    def test_rejects_invalid_storage_instead_of_training_another_format(self):
        for value, storage in [(np.zeros((1, 31), np.float32), "Q8_0"),
                               (np.zeros(32, np.float32), "Q8_0"),
                               (np.zeros((1, 32), np.float64), "Q8_0"),
                               (np.array([[np.inf]], np.float32), "F16"),
                               (np.array([[70000]], np.float32), "F16"),
                               (np.zeros((1, 32), np.float32), "INT8")]:
            with self.subTest(storage=storage, shape=value.shape):
                with self.assertRaises(ValueError):
                    storage_forward(value, storage)


class LayoutArgmaxAccountingTests(unittest.TestCase):
    def test_counts_class_and_independent_dfl_edge_flips(self):
        reference = [np.zeros((1, 2, 10), np.float32) for _ in range(4)]
        reference += [np.zeros((1, 2, 32), np.float32) for _ in range(4)]
        actual = [value.copy() for value in reference]
        reference[0][0, 0, 2] = 1
        actual[0][0, 0, 3] = 1
        reference[4][0, 1, 3 * 8 + 5] = 1
        actual[4][0, 1, 3 * 8 + 6] = 1

        counts = head_argmax_counts(actual, reference)

        self.assertEqual(counts, dict(class_decisions=8, class_flips=1,
                                      dfl_decisions=32, dfl_flips=1))

    def test_requires_all_finite_heads_with_matching_layout_shapes(self):
        heads = [np.zeros((1, 1, 10), np.float32) for _ in range(4)]
        heads += [np.zeros((1, 1, 32), np.float32) for _ in range(4)]
        with self.assertRaises(ValueError):
            head_argmax_counts(heads[:-1], heads[:-1])
        nonfinite = [value.copy() for value in heads]
        nonfinite[0][0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            head_argmax_counts(nonfinite, heads)


class TeacherHeadFeedTests(unittest.TestCase):
    def setUp(self):
        self.names = [f"head_{i}" for i in range(8)]
        self.truth = [np.zeros((1, 2, 10), np.float32) for _ in range(4)]
        self.truth += [np.zeros((1, 2, 32), np.float32) for _ in range(4)]
        self.targets = {name: SimpleNamespace(name=f"teacher_{i}")
                        for i, name in enumerate(self.names)}

    def test_q8_mse_path_does_not_require_winner_labels(self):
        feed = teacher_head_feed(self.names, self.truth, self.targets, {})

        self.assertEqual(set(feed), {target.name for target in self.targets.values()})
        for target, value in zip(self.targets.values(), self.truth):
            np.testing.assert_array_equal(feed[target.name], value)

    def test_ranking_path_emits_class_and_each_dfl_edge_winner(self):
        winners = {name: SimpleNamespace(name=f"winner_{i}")
                   for i, name in enumerate(self.names)}

        feed = teacher_head_feed(self.names, self.truth, self.targets, winners)

        self.assertEqual(len(feed), 16)
        for index, (name, value) in enumerate(zip(self.names, self.truth)):
            target = winners[name]
            expected = np.argmax(value if value.shape[-1] == 10 else
                                 value.reshape(*value.shape[:-1], 4, 8), axis=-1)
            np.testing.assert_array_equal(feed[target.name], expected)

    def test_rejects_mismatched_target_mapping_with_clear_error(self):
        with self.assertRaisesRegex(ValueError, "target mappings"):
            teacher_head_feed(self.names, self.truth, {}, {})


if __name__ == "__main__":
    unittest.main()
