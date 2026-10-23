"""Reject CPU fallbacks before a CUDA reference can produce acceptance data."""

import hashlib
import importlib.metadata
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


SCRIPTS = Path(__file__).resolve().parent
BENCH = SCRIPTS.parent / "tools/bench_paddle_ocr_graph.py"
LAYOUT = SCRIPTS / "run_paddle_layout_reference.py"


class BeforeInference(Exception):
    pass


class PaddleReferenceDeviceTests(unittest.TestCase):
    def check_entry(self, entry, compiled, enabled):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for name in ("program", "parameters", "input"):
                path = root / name
                path.write_bytes(name.encode())
                files[name] = path
            report = root / "report.json"
            if entry == BENCH:
                arguments = [str(entry), "--device", "cuda", "--warmups", "0",
                             "--runs", "1", "--threads", "1", "--report", str(report)]
                for name, path in files.items():
                    arguments += ["--" + name, str(path), "--" + name + "-sha256",
                                  hashlib.sha256(path.read_bytes()).hexdigest()]
            else:
                post = root / "postprocess.py"
                # The upstream constructor consumes its label file immediately.
                post.write_text("from pathlib import Path\n"
                    "class PicoDetPostProcess:\n"
                    "    def __init__(self, labels, **kwargs):\n"
                    "        Path(labels).read_text()\n"
                    "        Path(labels).unlink()\n")
                arguments = [str(entry), "--device", "gpu", "--model-dir", str(root),
                             "--postprocess-py", str(post), "--output", str(report),
                             str(files["input"])]
            paddle = ModuleType("paddle")
            inference = ModuleType("paddle.inference")
            paddle.is_compiled_with_cuda = Mock(return_value=compiled)
            paddle.inference = inference
            config = Mock()
            config.use_gpu.return_value = enabled
            inference.Config = Mock(return_value=config)
            inference.create_predictor = Mock(side_effect=BeforeInference)
            imported_modules = {"paddle": paddle, "paddle.inference": inference}
            # These guards must fail before any array/image operations; keeping
            # their imports inert also lets runpy exercise repeated cases in
            # one test process without reloading NumPy's compiled extension.
            imported_modules["numpy"] = ModuleType("numpy")
            if entry == LAYOUT:
                imported_modules["cv2"] = ModuleType("cv2")
            with patch.dict(sys.modules, imported_modules), \
                    patch.object(sys, "argv", arguments), \
                    patch.dict(os.environ, {"NVIDIA_TF32_OVERRIDE": "1"}), \
                    patch.object(importlib.metadata, "version", return_value="3.0.0"):
                if not compiled:
                    with self.assertRaisesRegex(RuntimeError, "CPU fallback is forbidden"):
                        runpy.run_path(str(entry), run_name="__main__")
                    inference.Config.assert_not_called()
                elif not enabled:
                    with self.assertRaisesRegex(RuntimeError, "did not enable"):
                        runpy.run_path(str(entry), run_name="__main__")
                else:
                    with self.assertRaises(BeforeInference):
                        runpy.run_path(str(entry), run_name="__main__")
                    inference.create_predictor.assert_called_once_with(config)
                self.assertEqual(os.environ["NVIDIA_TF32_OVERRIDE"], "0")
            if not compiled or not enabled:
                inference.create_predictor.assert_not_called()
            self.assertFalse(report.exists())

    def test_cpu_library_cannot_certify_cuda_even_with_gpu_package_metadata(self):
        for entry in (BENCH, LAYOUT):
            with self.subTest(entry=entry.name):
                self.check_entry(entry, compiled=False, enabled=False)

    def test_disabled_predictor_cannot_certify_cuda(self):
        for entry in (BENCH, LAYOUT):
            with self.subTest(entry=entry.name):
                self.check_entry(entry, compiled=True, enabled=False)

    def test_cuda_predictor_reaches_creation_with_strict_math(self):
        for entry in (BENCH, LAYOUT):
            with self.subTest(entry=entry.name):
                self.check_entry(entry, compiled=True, enabled=True)


if __name__ == "__main__":
    unittest.main()
