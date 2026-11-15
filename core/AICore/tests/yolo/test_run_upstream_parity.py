"""Portable matrix orchestration and failure-propagation contracts."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run_upstream_parity as runner
import runtime_environment as runtime


class ParityLauncherTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        models = root / "models"
        models.mkdir()
        for name in ("a.gguf", "b.gguf"):
            (models / name).touch()
        baseline = root / "upstream.jsonl"
        baseline.write_text("{}\n")
        binary = root / "build/bin/Release/test_yolo_capi_performance.exe"
        binary.parent.mkdir(parents=True)
        binary.touch()
        args = runner.parser().parse_args([
            "--build", str(root / "build"), "--models-dir", str(models),
            "--upstream-bench", str(baseline), "--out-dir", str(root / "out"),
            "--devices", "cpu,vulkan", "--no-build"])
        return args, binary

    def test_windows_layout_runs_each_model_and_combines_only_current_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            args, binary = self.fixture(directory)
            args.out_dir.mkdir()
            (args.out_dir / "aicore-all.jsonl").write_text("stale\n")
            processes = []

            def execute(command, **kwargs):
                if "env" in kwargs:
                    processes.append((command, kwargs["env"]))
                    kwargs["stdout"].write(b'{"row":true}\n')
                return SimpleNamespace(returncode=0)

            with mock.patch.dict(os.environ, {"AICORE_BUILD_CONFIG": "Release",
                                              "YOLO_CPU_THREADS": "8", "YOLO_GPU_THREADS": "32"}), \
                    mock.patch.object(runtime, "is_windows", return_value=True), \
                    mock.patch.object(runner, "configured_qt_library_dir", return_value=None), \
                    mock.patch.object(runner, "embedded_qt_library_dir", return_value=None), \
                    mock.patch.object(runner.subprocess, "run", side_effect=execute):
                self.assertEqual(runner.run(args), 0)
            self.assertEqual(len(processes), 4)
            self.assertTrue(all(command == [str(binary)] for command, _ in processes))
            self.assertEqual([env["AICORE_TEST_YOLO_THREADS"] for _, env in processes],
                             ["8", "8", "32", "32"])
            self.assertEqual((args.out_dir / "aicore-all.jsonl").read_bytes(),
                             b'{"row":true}\n' * 4)

    def test_probe_failure_survives_a_successful_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            args, binary = self.fixture(directory)
            with mock.patch.object(runner, "find_build_tool", return_value=binary), \
                    mock.patch.object(runner, "configured_qt_library_dir", return_value=None), \
                    mock.patch.object(runner, "embedded_qt_library_dir", return_value=None), \
                    mock.patch.object(runner.subprocess, "run", side_effect=[
                        SimpleNamespace(returncode=1), *[SimpleNamespace(returncode=0)] * 4]) as run, \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner.run(args), 1)
                self.assertEqual(run.call_count, 5)

    def test_build_failure_stops_before_any_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(directory)
            args.no_build = False
            with mock.patch.object(runner.subprocess, "run", side_effect=
                                   subprocess.CalledProcessError(1, "cmake")) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.run(args)
                self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
