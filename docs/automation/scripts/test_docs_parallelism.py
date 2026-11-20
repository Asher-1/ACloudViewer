#!/usr/bin/env python3
"""Verify Sphinx uses the same NPROC budget as the native/CI build."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "make_docs", Path(__file__).resolve().parents[2] / "make_docs.py")
make_docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_docs)


class SphinxParallelismTest(unittest.TestCase):
    def command(self, parallel, limit, cpus=32):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "output/html").mkdir(parents=True)
            (root / "output/html/index.html").write_text("test", encoding="utf-8")
            builder = make_docs.SphinxDocsBuilder(
                str(root), str(root / "output"), False, parallel, strict=True)
            with patch.dict(os.environ, {}, clear=True):
                if limit is not None:
                    os.environ["NPROC"] = limit
                with patch.object(make_docs.multiprocessing, "cpu_count", return_value=cpus), \
                     patch.object(make_docs.shutil, "which", return_value="sphinx-build"), \
                     patch.object(make_docs, "get_git_short_hash", return_value="test"), \
                     patch.object(make_docs.subprocess, "check_call") as call:
                    builder.run()
                    command = call.call_args.args[0]
                    self.assertIn("-W", command)
                    return command[command.index("-j") + 1]

    def test_serial_ignores_parallel_budget(self):
        self.assertEqual(self.command(False, "12"), "1")

    def test_parallel_honors_budget(self):
        self.assertEqual(self.command(True, "12"), "12")

    def test_ci_budget(self):
        self.assertEqual(self.command(True, "4"), "4")

    def test_budget_cannot_exceed_cpu_count(self):
        self.assertEqual(self.command(True, "12", cpus=4), "4")

    def test_missing_budget_retains_default(self):
        self.assertEqual(self.command(True, None), "32")

    def test_nonpositive_budget_uses_one_worker(self):
        self.assertEqual(self.command(True, "0"), "1")

    def test_invalid_budget_fails_instead_of_unbounded_build(self):
        with self.assertRaises(ValueError):
            self.command(True, "invalid")


if __name__ == "__main__":
    unittest.main()
