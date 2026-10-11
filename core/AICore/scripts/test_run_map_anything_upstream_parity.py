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


class ReferenceEnvTest(unittest.TestCase):
    def test_every_foreign_ggml_provider_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            build, root = tmp / "snap", tmp / "ref"
            copy = tmp / "aicore_build" / "external" / "lib"
            toolkit = tmp / "cuda" / "lib64"
            own = root / "cpp_ggml" / "build-cuda" / "lib"
            for d in (build / "bin", copy, toolkit, own):
                d.mkdir(parents=True)
            (copy / "libggml-base.so.0").write_text("")
            (own / "libggml-base.so.0").write_text("")
            (toolkit / "libcudart.so.12").write_text("")
            entries = [str(build / "bin"), str(copy), str(toolkit), str(own)]
            saved = os.environ.get("LD_LIBRARY_PATH")
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(entries)
            try:
                env = PARITY.reference_env(build, root)
            finally:
                if saved is None:
                    del os.environ["LD_LIBRARY_PATH"]
                else:
                    os.environ["LD_LIBRARY_PATH"] = saved
            self.assertEqual(env["LD_LIBRARY_PATH"].split(os.pathsep),
                             [str(toolkit), str(own)])


class VendoredTwinTest(unittest.TestCase):
    PATCH = (b"diff --git a/src/x.cu b/src/x.cu\n"
             b"--- a/src/x.cu\n+++ b/src/x.cu\n@@ -1 +1,2 @@\n a\n+b\n"
             b"diff --git a/src/ggml-cuda/mapany/ggml-v0.24.0/k.cuh "
             b"b/src/ggml-cuda/mapany/ggml-v0.24.0/k.cuh\n"
             b"new file mode 100644\n--- /dev/null\n"
             b"+++ b/src/ggml-cuda/mapany/ggml-v0.24.0/k.cuh\n"
             b"@@ -0,0 +1,2 @@\n+#pragma once\n+int k;\n"
             b"\\ No newline at end of file\n")

    def test_only_new_twin_files_are_extracted_byte_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            patch = Path(tmp) / "p.patch"
            patch.write_bytes(self.PATCH)
            self.assertEqual(PARITY.vendored_twin(patch),
                             {"k.cuh": b"#pragma once\nint k;"})

    def test_repository_twin_extracts_the_three_flash_attention_sources(self):
        twin = PARITY.vendored_twin(PARITY.MAPANY_PATCH)
        self.assertEqual(sorted(twin), ["fattn-common.cuh", "fattn-mma-f16.cuh",
                                        "fattn-swizzle.cuh"])
        self.assertTrue(all(data.endswith(b"\n") for data in twin.values()))


if __name__ == "__main__":
    unittest.main()
