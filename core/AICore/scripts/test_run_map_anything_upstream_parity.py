#!/usr/bin/env python3
"""Unit tests for the upstream MapAnything parity row's reference guards."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "run_map_anything_upstream_parity.py"
SPEC = importlib.util.spec_from_file_location("upstream_parity", SCRIPT)
PARITY = importlib.util.module_from_spec(SPEC)
sys.modules["upstream_parity"] = PARITY
SPEC.loader.exec_module(PARITY)


class NewerSourceTest(unittest.TestCase):
    def make_tree(self, root: Path) -> tuple[Path, Path]:
        cpp = root / "cpp_ggml"
        for rel in ("src/vggt_graph.cpp", "include/mapggml/capi.h",
                    "CMakeLists.txt", "third_party/ggml/CMakeLists.txt",
                    "third_party/ggml/src/ggml-cuda/conv2d-transpose.cu",
                    "third_party/ggml/include/ggml.h",
                    "third_party/ggml-patches/0001-k.patch"):
            path = cpp / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rel)
            os.utime(path, (1000, 1000))
        cli = cpp / "build-cuda" / "bin" / "vggt-cli"
        cli.parent.mkdir(parents=True)
        cli.write_text("elf")
        os.utime(cli, (2000, 2000))
        return cpp, cli

    def test_binary_newer_than_every_source_is_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, cli = self.make_tree(Path(tmp))
            self.assertIsNone(PARITY.newer_source(Path(tmp), cli))

    def test_source_edited_after_build_is_stale(self):
        for rel in ("src/vggt_graph.cpp",
                    "third_party/ggml/src/ggml-cuda/conv2d-transpose.cu",
                    "third_party/ggml-patches/0001-k.patch"):
            with self.subTest(rel=rel), tempfile.TemporaryDirectory() as tmp:
                cpp, cli = self.make_tree(Path(tmp))
                os.utime(cpp / rel, (3000, 3000))
                self.assertEqual(PARITY.newer_source(Path(tmp), cli), cpp / rel)

    def test_outputs_outside_the_compiled_sources_do_not_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            cpp, cli = self.make_tree(Path(tmp))
            bench = cpp / "benchmarks" / "RESULTS.md"
            bench.parent.mkdir()
            bench.write_text("later")
            os.utime(bench, (3000, 3000))
            self.assertIsNone(PARITY.newer_source(Path(tmp), cli))


if __name__ == "__main__":
    unittest.main()
