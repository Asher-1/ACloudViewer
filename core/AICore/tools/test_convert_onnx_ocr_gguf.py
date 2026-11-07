#!/usr/bin/env python3
"""Fail-closed conversion tests; real checkpoint accuracy uses manifest probes."""
import subprocess
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from convert_onnx_ocr_gguf import lower, validate_source_identity
from convert_paddle_ocr_gguf import write_gguf, quantized
from onnx_ocr_reference import stored_weight_reference


def model(node, tensors=()):
    graph = helper.make_graph([node], "negative-contract",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
                              initializer=list(tensors))
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class ConverterTests(unittest.TestCase):
    def test_activation_q8_standard_bytes_reduce_correlated_output_error(self):
        import gguf
        from ocr_q8_calibration import activation_q8_0
        rng = np.random.default_rng(7)
        weight = rng.normal(size=(4, 32, 1, 1)).astype(np.float32)
        basis = rng.normal(size=(4, 160))
        activation = rng.normal(size=(32, 4)) @ basis + rng.normal(size=(32, 160)) * .01
        encoded = activation_q8_0(weight, activation, .01)
        restored = gguf.quants.Q8_0.dequantize(encoded).reshape(4, 32)
        ordinary = gguf.quants.Q8_0.dequantize(
            gguf.quants.Q8_0.quantize(weight.reshape(4, 32)))
        expected = weight.reshape(4, 32) @ activation
        self.assertLess(np.linalg.norm(restored @ activation - expected),
                        np.linalg.norm(ordinary @ activation - expected))
        np.testing.assert_array_equal(encoded, activation_q8_0(weight, activation, .01))
        self.assertEqual(encoded.dtype, np.uint8)
        self.assertEqual(encoded.size, weight.size // 32 * 34)
        for bad in [np.zeros((32, 0)), np.full((32, 4), np.nan)]:
            with self.assertRaises(ValueError):
                activation_q8_0(weight, bad, .01)
        for damping in [0, -1, float('nan')]:
            with self.assertRaises(ValueError):
                activation_q8_0(weight, activation, damping)

    def test_activation_calibration_is_sha_bound_before_output_creation(self):
        from ocr_q8_calibration import ALGORITHM, calibrated_overrides
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "word.png").write_bytes(b"wrong calibration input")
            config = {"algorithm": ALGORITHM, "kernels": {"kernel": .01},
                      "images": [{"relative_path": "word.png", "sha256": "0" * 64}]}
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                calibrated_overrides(root / "not-executed.onnx", {}, config, root, "rec")
            config["images"][0]["relative_path"] = "../outside.png"
            with self.assertRaisesRegex(ValueError, "relative path"):
                calibrated_overrides(root / "not-executed.onnx", {}, config, root, "rec")

    def test_calibrated_q8_container_restores_exact_standard_weights(self):
        import gguf
        from ocr_q8_calibration import activation_q8_0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(11)
            weight = rng.normal(size=(3, 32, 1, 1)).astype(np.float32)
            source = root / "source.onnx"
            graph = helper.make_graph([helper.make_node("Conv", ["x", "kernel"], ["y"])], "calibrated",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 32, 8, 8])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])],
                [numpy_helper.from_array(weight, "kernel")])
            onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), source)
            packed = activation_q8_0(weight, rng.normal(size=(32, 100)), .01)
            container = root / "calibrated.gguf"
            write_gguf({"kernel": weight}, {"ocr.source.sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
                       container, "q8_0", [], storage_overrides={"kernel": (packed, gguf.GGMLQuantizationType.Q8_0)})
            restored, storage = stored_weight_reference(source, container)
            self.assertEqual(storage, {"Q8_0": weight.size})
            actual = numpy_helper.to_array(restored.graph.initializer[0])
            np.testing.assert_array_equal(actual, gguf.quants.Q8_0.dequantize(packed).reshape(weight.shape))

    def test_q8_subnormal_zero_scale_has_finite_canonical_storage(self):
        import gguf
        normal = np.linspace(-.7, .7, 32, dtype=np.float32)
        weights = np.stack([np.full(32, 1e-40, np.float32), normal])
        with np.errstate(over="raise", invalid="raise"):
            encoded, kind = quantized("kernel", weights, "q8_0", [])
        restored = gguf.quants.dequantize(encoded, kind)
        self.assertTrue(np.isfinite(restored).all())
        np.testing.assert_array_equal(restored[0], np.zeros(32, np.float32))
        # Ordinary blocks remain byte-for-byte the library's Q8_0 encoding.
        np.testing.assert_array_equal(encoded[1], gguf.quants.Q8_0.quantize(normal[None])[0])

    def test_source_role_and_dictionary_are_bound_to_checkpoint(self):
        assets = [{"sha256": "source", "role": "rec",
                   "dictionary": {"sha256": "v6-dictionary"}}]
        validate_source_identity("source", "rec", "v6-dictionary", assets)
        with self.assertRaisesRegex(ValueError, "dictionary SHA-256 mismatch"):
            validate_source_identity("source", "rec", "v5-dictionary", assets)
        with self.assertRaisesRegex(ValueError, "role mismatch"):
            validate_source_identity("source", "det", None, assets)

    def test_q8_policy_cannot_emit_a_container_without_q8_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.onnx"
            kernel = numpy_helper.from_array(np.ones((3, 32, 1, 1), np.float32), "kernel")
            graph = helper.make_graph([helper.make_node("Conv", ["x", "kernel"], ["y"])], "no-q8",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 32, 8, 8])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])], [kernel])
            onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            policy = root / "policy.json"
            policy.write_text(json.dumps({"schema": 1, "sources": {digest: {
                "q8_0": {"keep_f16": ["kernel"]}}}}))
            completed = subprocess.run([sys.executable, str(Path(__file__).with_name("convert_onnx_ocr_gguf.py")),
                                        str(source), str(root / "output"), "--role", "det",
                                        "--source-sha256", digest, "--quants", "q8_0",
                                        "--precision-policy", str(policy)], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("no Q8_0 weights", completed.stderr)
            self.assertFalse(list(root.glob("*.gguf")))

    def test_precision_policy_rejects_invalid_later_quant_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.onnx"
            weight = numpy_helper.from_array(np.ones((3, 3, 1, 1), np.float32), "kernel")
            onnx.save(model(helper.make_node("Conv", ["x", "kernel"], ["y"]), [weight]), source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            policy = root / "policy.json"
            policy.write_text(json.dumps({"schema": 1, "sources": {digest: {
                "q8_0": {"keep_f16": ["typo"]}}}}))
            completed = subprocess.run([sys.executable, str(Path(__file__).with_name("convert_onnx_ocr_gguf.py")),
                                        str(source), str(root / "output"), "--role", "det",
                                        "--source-sha256", digest, "--quants", "f32,q8_0",
                                        "--precision-policy", str(policy)], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("unknown, non-kernel or conflicting", completed.stderr)
            self.assertFalse(list(root.glob("*.gguf")))

    def test_precision_policy_preserves_measured_kernel_storage(self):
        import gguf
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.onnx"
            kernels = [numpy_helper.from_array(np.linspace(-.7, .7, 96, dtype=np.float32).reshape(3, 32, 1, 1), name)
                       for name in ("first", "second")]
            graph = helper.make_graph([
                helper.make_node("Conv", ["x", "first"], ["a"]),
                helper.make_node("Conv", ["x", "second"], ["b"]),
                helper.make_node("Add", ["a", "b"], ["y"])], "storage",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 32, 8, 8])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])], kernels)
            onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            policy = root / "policy.json"
            policy.write_text(json.dumps({"schema": 1, "sources": {digest: {
                "f16": {"keep_f32": ["first"]}, "q8_0": {"keep_f16": ["first"]}}}}))
            subprocess.run([sys.executable, str(Path(__file__).with_name("convert_onnx_ocr_gguf.py")),
                            str(source), str(root / "output"), "--role", "det",
                            "--source-sha256", digest, "--quants", "f16,q8_0",
                            "--precision-policy", str(policy)], check=True, capture_output=True)
            for kind, expected in (("f16", {"first": "F32", "second": "F16"}),
                                   ("q8_0", {"first": "F16", "second": "Q8_0"})):
                reader = gguf.GGUFReader(root / f"output-{kind}.gguf")
                self.assertEqual({t.name.removeprefix("ocr."): t.tensor_type.name for t in reader.tensors}, expected)

    def test_stored_reference_replaces_folded_alias_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = np.linspace(-.7, .7, 96, dtype=np.float32).reshape(3, 32, 1, 1)
            graph = helper.make_graph([
                helper.make_node("Identity", ["original"], ["kernel"]),
                helper.make_node("Conv", ["x", "kernel"], ["y"])], "alias",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 32, 8, 8])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])],
                [numpy_helper.from_array(weight, "original")])
            source = root / "source.onnx"
            onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), source)
            before = source.read_bytes()
            tensors, _, keep = lower(onnx.load(source))
            for kind in ("f32", "f16", "q8_0"):
                container = root / f"model-{kind}.gguf"
                write_gguf(tensors, {"ocr.source.sha256": hashlib.sha256(before).hexdigest()},
                           container, kind, [], keep_f32=keep)
                reference, storage = stored_weight_reference(source, container)
                onnx.checker.check_model(reference)
                self.assertEqual([node.op_type for node in reference.graph.node], ["Conv"])
                actual = numpy_helper.to_array(reference.graph.initializer[0])
                self.assertEqual(actual.shape, weight.shape)
                self.assertEqual(sum(storage.values()), weight.size)
                self.assertLess(float(np.abs(actual - weight).max()), .004)
                self.assertEqual(source.read_bytes(), before)

    def test_stored_reference_rejects_wrong_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.onnx"
            onnx.save(model(helper.make_node("Relu", ["x"], ["y"])), source)
            container = root / "wrong.gguf"
            write_gguf({}, {"ocr.source.sha256": "0" * 64}, container, "f32", [])
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                stored_weight_reference(source, container)

    def test_unknown_operator_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported ONNX op Sin"):
            lower(model(helper.make_node("Sin", ["x"], ["y"])))

    def test_unknown_attribute_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "attributes"):
            lower(model(helper.make_node("Relu", ["x"], ["y"], unexpected=1)))

    def test_interpolated_resize_is_rejected(self):
        scales = numpy_helper.from_array(np.array([1, 1, 2, 2], np.float32), "scales")
        with self.assertRaisesRegex(ValueError, "Resize semantics"):
            lower(model(helper.make_node("Resize", ["x", "", "scales"], ["y"], mode="linear"), [scales]))

    def test_dynamic_clip_limits_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "constant limits"):
            lower(model(helper.make_node("Clip", ["x", "x", "x"], ["y"])))

    def test_constant_folding_does_not_hide_allowzero(self):
        data = numpy_helper.from_array(np.ones((1, 3, 8, 8), np.float32), "data")
        shape = numpy_helper.from_array(np.array([1, 3, 8, 8], np.int64), "shape")
        with self.assertRaisesRegex(ValueError, "allowzero"):
            lower(model(helper.make_node("Reshape", ["data", "shape"], ["y"], allowzero=1), [data, shape]))

    def test_unaudited_pool_padding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "pooling auto_pad"):
            lower(model(helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[3, 3], auto_pad="SAME_LOWER")))

    def test_half_normalization_stash_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "stash"):
            lower(model(helper.make_node("LayerNormalization", ["x"], ["y"], stash_type=10)))

    def test_wrong_digest_writes_no_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.onnx"
            onnx.save(model(helper.make_node("Relu", ["x"], ["y"])), source)
            completed = subprocess.run([sys.executable, str(Path(__file__).with_name("convert_onnx_ocr_gguf.py")),
                                        str(source), str(root / "output"), "--role", "det",
                                        "--source-sha256", "0" * 64], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("SHA-256 mismatch", completed.stderr)
            self.assertFalse(list(root.glob("*.gguf")))


if __name__ == "__main__":
    unittest.main()
