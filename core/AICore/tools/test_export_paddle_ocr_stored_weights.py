import struct
import unittest

import numpy as np

from export_paddle_ocr_stored_weights import replace_payloads, stored_values


def stream(values):
    result = bytearray()
    for value in values:
        # TensorDesc protobuf: data_type=FP32 (field 1), dimensions (field 2).
        descriptor = bytes([8, 5] + [x for d in value.shape for x in (16, d)])
        result += struct.pack("<IQII", 0, 0, 0, len(descriptor))
        result += descriptor + value.astype("<f4").tobytes()
    return bytes(result)


class StoredWeightsTests(unittest.TestCase):
    def setUp(self):
        self.named = [("a", np.array([.1, .2], np.float32)),
                      ("b", np.arange(6, dtype=np.float32).reshape(2, 3))]
        self.data = stream([v for _, v in self.named])
        self.values = {name: value.astype(np.float16).astype(np.float32)
                       for name, value in self.named}

    def test_preserves_descriptors_and_changes_only_payload(self):
        result = replace_payloads(self.data, self.named, self.values)
        self.assertEqual(result, stream(list(self.values.values())))
        self.assertNotEqual(result, self.data)

    def test_f32_roundtrip_is_byte_identical(self):
        self.assertEqual(replace_payloads(self.data, self.named, dict(self.named)), self.data)

    def test_missing_extra_duplicate_names_and_wrong_order_fail(self):
        for named, values in [(self.named, {"a": self.values["a"]}),
                (self.named, {**self.values, "c": self.values["a"]}),
                ([self.named[0], self.named[0]], self.values),
                (self.named[::-1], self.values)]:
            with self.subTest(named=named), self.assertRaises(ValueError):
                replace_payloads(self.data, named, values)

    def test_shape_dtype_nonfinite_and_truncated_stream_fail(self):
        for value in (np.zeros((2, 1), np.float32), np.zeros(2, np.float16),
                      np.array([np.nan, 0], np.float32)):
            with self.assertRaises(ValueError):
                replace_payloads(self.data, self.named, {**self.values, "a": value})
        for cut in (1, 10, 23):
            with self.assertRaises(ValueError):
                replace_payloads(self.data[:-cut], self.named, self.values)

    def test_lod_is_rejected_instead_of_treating_count_as_byte_length(self):
        bad = bytearray(self.data)
        struct.pack_into("<Q", bad, 4, 1)
        with self.assertRaisesRegex(ValueError, "LoD-free"):
            replace_payloads(bad, self.named, self.values)

    def test_real_gguf_q8_folded_kernel_and_f16_reconstruction(self):
        import tempfile
        from pathlib import Path
        try:
            import gguf
        except ImportError:
            self.skipTest("GGUF conversion package is not installed")

        named = [("kernel", np.linspace(-1, 1, 128, dtype=np.float32).reshape(2, 2, 4, 8)),
                 ("bias", np.array([.12345, -.6789], np.float32))]
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "test.gguf"
            writer = gguf.GGUFWriter(str(model), "ocr")
            quantized = gguf.quants.Q8_0.quantize(named[0][1].reshape(2, 64))
            writer.add_tensor("ocr.kernel", quantized, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor("ocr.bias", named[1][1].astype(np.float16))
            writer.write_header_to_file(); writer.write_kv_data_to_file()
            writer.write_tensors_to_file(); writer.close()
            values, receipt = stored_values(model, named)
            np.testing.assert_array_equal(values["kernel"],
                gguf.quants.dequantize(quantized, gguf.GGMLQuantizationType.Q8_0).reshape(2, 2, 4, 8))
            np.testing.assert_array_equal(values["bias"], named[1][1].astype(np.float16).astype(np.float32))
            self.assertEqual([row['storage'] for row in receipt], ['Q8_0', 'F16'])
            self.assertEqual(len(replace_payloads(stream([v for _, v in named]), named, values)),
                             len(stream([v for _, v in named])))
            with self.assertRaisesRegex(ValueError, "coverage"):
                stored_values(model, named[:1])
            with self.assertRaisesRegex(ValueError, "shape"):
                stored_values(model, [("kernel", named[0][1].reshape(4, 32)), named[1]])


if __name__ == '__main__':
    unittest.main()
