"""Numeric contracts for conversion-only F16 error compensation."""
import unittest

import numpy as np

from ocr_q8_calibration import activation_f16


class ActivationF16Tests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(31)
        self.weights = rng.normal(size=(8, 32, 1, 1)).astype(np.float32)
        common = rng.normal(size=(1, 512))
        self.inputs = common + rng.normal(scale=.04, size=(32, 512))

    def test_real_f16_reduces_correlated_activation_error(self):
        before = self.weights.copy()
        result = activation_f16(self.weights, self.inputs, .01)
        w = self.weights.reshape(8, 32).astype(np.float64)
        baseline = w.astype(np.float16).astype(np.float64)
        actual = result.reshape(8, 32).astype(np.float64)
        self.assertEqual(result.dtype, np.float16)
        self.assertEqual(result.shape, before.shape)
        np.testing.assert_array_equal(self.weights, before)
        original_error = np.sum(((w - baseline) @ self.inputs) ** 2)
        corrected_error = np.sum(((w - actual) @ self.inputs) ** 2)
        self.assertLess(corrected_error, original_error * .5)
        np.testing.assert_array_equal(result, activation_f16(before, self.inputs, .01))

    def test_exact_weights_remain_exact(self):
        weights = np.full((2, 32), .5, dtype=np.float32)
        np.testing.assert_array_equal(activation_f16(weights, self.inputs, .01), weights)

    def test_invalid_calibration_is_rejected(self):
        for damping in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                activation_f16(self.weights, self.inputs, damping)
        for inputs in (self.inputs[:1], self.inputs[:, :0], self.inputs * np.nan,
                       np.zeros_like(self.inputs)):
            with self.assertRaises(ValueError):
                activation_f16(self.weights, inputs, .01)
        for weights in (np.full((2, 32), 70000), np.zeros((2, 32, 3, 3))):
            with self.assertRaises(ValueError):
                activation_f16(weights, self.inputs, .01)


if __name__ == "__main__":
    unittest.main()
