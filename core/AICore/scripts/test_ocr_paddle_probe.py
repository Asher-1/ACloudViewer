#!/usr/bin/env python3
import os
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from run_ocr_paddle_probe import comparison, digest, execute, passes, run, source_environment
from ocr_paddle_metadata import distribution_environment


class PaddleGraphTests(unittest.TestCase):
    def test_vendored_metadata_cannot_hide_the_selected_package_version(self):
        entries = [SimpleNamespace(metadata={"Name": name}, version=version)
                   for name, version in (("typing_extensions", "4.16.0"),
                                         ("typing-extensions", "4.12.2"))]
        with patch("ocr_paddle_metadata.metadata.distributions", return_value=entries), \
                patch("ocr_paddle_metadata.metadata.version", return_value="4.16.0"):
            identity = distribution_environment()
        self.assertEqual(identity["environment"], {"typing-extensions": "4.16.0"})
        self.assertEqual(identity["distribution_candidates"],
                         {"typing-extensions": ["4.12.2", "4.16.0"]})
        with patch("ocr_paddle_metadata.metadata.distributions", return_value=entries), \
                patch("ocr_paddle_metadata.metadata.version", return_value="4.17.0"):
            self.assertNotEqual(distribution_environment(), identity)

    def test_missing_selected_metadata_fails_instead_of_using_a_vendored_fallback(self):
        entry = SimpleNamespace(metadata={"Name": "missing"}, version="1.0")
        with patch("ocr_paddle_metadata.metadata.distributions", return_value=[entry]), \
                patch("ocr_paddle_metadata.metadata.version", side_effect=ValueError("not found")):
            with self.assertRaisesRegex(ValueError, "not found"):
                distribution_environment()

    def test_all_public_graph_quantizations_have_unambiguous_metrics_and_inputs(self):
        scripts = Path(__file__).resolve().parent
        manifest = json.loads((scripts / "validation_manifest.json").read_text())
        rows = [r for r in manifest["scenarios"] if r["id"].startswith("ocr-paddle-graph-")]
        self.assertEqual(len(rows), 9)
        self.assertEqual(len({r["model_glob"] for r in rows}), 9)
        for row in rows:
            args = row["args"]
            role = args[args.index("--source-role") + 1]
            self.assertIn("ocr-paddle-source-" + role, row["input_ids"])
            self.assertIn("ocr-paddle-config-" + role, row["input_ids"])
            self.assertIn("{profile}", args)
            self.assertIn("{reference:ocr_paddle_gpu}", args)
            for check in row["accuracy"]["checks"]:
                self.assertTrue(check["metric"].startswith("json0/"))
            self.assertIn("json0/max_diff", row["accuracy"]["require_metrics"])

    def test_shape_and_nonfinite_values_fail_before_broadcasting(self):
        for other in (np.zeros((2, 2)), np.array([[float("nan"), 1.]]),
                      np.array([[0., float("inf")]])):
            with self.assertRaisesRegex(ValueError, "invalid probability"):
                comparison(np.zeros((1, 2)), other)
        with self.assertRaises(ValueError):
            comparison(np.array([]), np.array([]))

    def test_changed_argmax_fails_even_with_small_probability_error(self):
        metrics = comparison(np.array([[.50001, .49999]]),
                             np.array([[.49999, .50001]]), argmax=True)
        self.assertLess(metrics["max_diff"], .001)
        self.assertFalse(passes(metrics, .001, argmax=True))

    def test_quantized_raw_argmax_is_diagnostic_when_values_are_finite(self):
        metrics = comparison(np.array([[.50001, .49999]]),
                             np.array([[.49999, .50001]]), argmax=True)
        self.assertTrue(passes(metrics, .001, argmax=True, quantized=True))

    def test_mean_and_maximum_are_independent_gates(self):
        metrics = comparison(np.array([[.2, .2]]), np.array([[.1, .1]]))
        self.assertFalse(passes(metrics, .5, .01))
        self.assertFalse(passes(metrics, .01, .5))
        self.assertTrue(passes(metrics, .5, .5))

    def test_source_environment_drops_foreign_python_and_shared_libraries(self):
        with patch.dict(os.environ, {"PYTHONPATH": "/foreign", "LD_PRELOAD": "foreign.so",
                                     "LD_LIBRARY_PATH": "/foreign/lib"}):
            environment = source_environment(Path("/no-such-venv/bin/python"))
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("LD_PRELOAD", environment)
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
        if os.name != "nt":
            self.assertNotIn("/foreign/lib", environment.get("LD_LIBRARY_PATH", ""))

    def test_source_crash_cannot_certify_an_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "source.json"
            report.write_text('{"p50_ms": 1}')
            crashed = subprocess.CompletedProcess(["source"], -6, "", "heap failure")
            with patch("run_ocr_paddle_probe.subprocess.run", return_value=crashed):
                with self.assertRaisesRegex(RuntimeError, "process failed"):
                    execute(["source"], report)
            self.assertIn("heap failure", report.with_suffix(".log").read_text())

    def test_source_does_not_inherit_application_logging_configuration(self):
        parent = {"GLOG_log_dir": "/sandbox/logfiles/", "GLOG_logtostderr": "1",
                  "MLOG_log_dir": "/sandbox/logfiles/", "KEEP_THIS": "value"}
        with patch.dict(os.environ, parent):
            environment = source_environment(Path("/no-such-venv/bin/python"))
            self.assertEqual(os.environ["GLOG_log_dir"], parent["GLOG_log_dir"])
        self.assertFalse(any(k.startswith(("GLOG_", "MLOG_")) for k in environment))
        self.assertEqual(environment["KEEP_THIS"], "value")

    def test_native_accuracy_failure_preserves_outputs_for_diagnosis(self):
        with tempfile.TemporaryDirectory() as directory:
            failure = subprocess.CompletedProcess(["native"], 1, '{"max_diff":.5}', "")
            with patch("run_ocr_paddle_probe.subprocess.run", return_value=failure):
                result, audit = execute(["native"], Path(directory)/"native.json",
                                        require_success=False)
            self.assertEqual(result.returncode, 1)
            self.assertIsNone(audit)

    def test_failed_gpu_audit_cannot_certify_fast_timings(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.CompletedProcess(["native"], 0, '{"p50_ms":1}', "")
            with patch("run_ocr_paddle_probe.run_audited", return_value=(completed, {"passed": False})):
                with self.assertRaisesRegex(RuntimeError, "audit rejected"):
                    execute(["native"], Path(directory)/"native.json", gpu=True, strict=True)

    def test_gpu_release_compares_official_cuda_output_without_running_native_cpu(self):
        scripts = Path(__file__).resolve().parent
        identity = digest(scripts / "ocr_paddle_metadata.py")
        for profile, mismatch in (("release", False), ("release", True),
                                  ("developer", False), ("developer", True)):
            with self.subTest(profile=profile, mismatch=mismatch), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                truth = np.array([[.50001, .49999]], dtype=np.float32)
                np.save(root / "input.npy", np.zeros((1, 3, 48, 320), np.float32))
                np.save(root / "reference.npy", truth)
                for name in ("model", "program", "parameters"):
                    (root / name).write_bytes(name.encode())
                (root / "pins.json").write_text('{}')
                args = SimpleNamespace(input=root / "input.npy", reference=root / "reference.npy",
                    source_pins=root / "pins.json", source_root=root, model=root / "model",
                    source_role="rec_en", role="rec", backend="vulkan", probe=root / "probe",
                    source_python=Path("cpu-source"), source_gpu_python=Path("cuda-source"),
                    report=root / "report.json", warmups=2, runs=10, threads=4,
                    max_diff=.001, mean_diff=.001, source_max_diff=.001,
                    gpu_idle_timeout=10, measurement_profile=profile,
                    input_sha256=digest(root / "input.npy"),
                    reference_sha256=digest(root / "reference.npy"))
                native_backends = []

                def launch(command, target, **kwargs):
                    if command[0] == str(args.probe):
                        native_backends.append(command[4])
                        dump = Path(command[6].removeprefix("dump="))
                        truth.tofile(dump)
                    else:
                        gpu = command[0] == "cuda-source"
                        np.save(target.with_suffix('.npy'),
                                truth[:, ::-1] if gpu and mismatch else truth)
                        target.write_text(json.dumps(dict(distribution_identity_sha256=identity,
                            environment={}, distribution_candidates={}, p50_ms=1, p95_ms=1)))
                    completed = subprocess.CompletedProcess(command, 0,
                        '{"p50_ms":100,"p95_ms":100}' if profile == "developer" else
                        '{"p50_ms":1,"p95_ms":1}', '')
                    audit = dict(passed=True, archive_path="audit.json", sampled_process_peak_mib=20)
                    return completed, audit if kwargs.get('gpu') else None

                files = {key: root / key for key in ("program", "parameters")}
                with patch('run_ocr_paddle_probe.verified_source', return_value=files), \
                        patch('run_ocr_paddle_probe.execute', side_effect=launch):
                    report = run(args)
                self.assertEqual(native_backends, ['vulkan'])
                self.assertEqual(report['exit_code'], int(mismatch))
                self.assertEqual(report['argmax_match'], int(not mismatch))
                self.assertEqual(report['source_gpu_argmax_match'], int(not mismatch))
                self.assertEqual(report['source_performance_qualified'], int(profile == 'release'))
                self.assertEqual(report['source_performance_pass'], int(profile == 'release'))

    def test_gpu_truth_does_not_require_original_or_native_cpu_accuracy(self):
        scripts = Path(__file__).resolve().parent
        identity = digest(scripts / 'ocr_paddle_metadata.py')
        for backend in ('cuda', 'vulkan'):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cpu = np.array([[.50001, .49999]], np.float32)
                cuda = cpu[:, ::-1].copy()
                np.save(root/'input.npy', np.zeros((1, 3, 48, 320), np.float32))
                np.save(root/'reference.npy', cpu)
                for name in ('model', 'program', 'parameters'):
                    (root/name).write_bytes(name.encode())
                (root/'pins.json').write_text('{}')
                args = SimpleNamespace(input=root/'input.npy', reference=root/'reference.npy',
                    source_pins=root/'pins.json', source_root=root, model=root/'model',
                    source_role='rec_en', role='rec', backend=backend, probe=root/'probe',
                    source_python=Path('cpu-source'), source_gpu_python=Path('cuda-source'),
                    report=root/'report.json', warmups=2, runs=10, threads=4,
                    max_diff=.001, mean_diff=.001, source_max_diff=.001,
                    gpu_idle_timeout=10, measurement_profile='release',
                    input_sha256=digest(root/'input.npy'),
                    reference_sha256=digest(root/'reference.npy'))
                called = []

                def launch(command, target, **kwargs):
                    called.append(command[0])
                    if command[0] == str(args.probe):
                        self.assertEqual(command[4], backend)
                        native_reference = np.load(Path(command[1])/'rec_en_reference.npy')
                        self.assertTrue(np.array_equal(native_reference, cuda))
                        cuda.tofile(Path(command[6].removeprefix('dump=')))
                    else:
                        self.assertEqual(command[0], 'cuda-source')
                        np.save(target.with_suffix('.npy'), cuda)
                        target.write_text(json.dumps(dict(distribution_identity_sha256=identity,
                            environment={}, distribution_candidates={}, p50_ms=1, p95_ms=1)))
                    return subprocess.CompletedProcess(command, 0,
                        '{"p50_ms":1,"p95_ms":1}', ''), dict(passed=True,
                            archive_path='audit.json', sampled_process_peak_mib=20)

                with patch('run_ocr_paddle_probe.verified_source', return_value={
                        key:root/key for key in ('program', 'parameters')}), \
                        patch('run_ocr_paddle_probe.execute', side_effect=launch):
                    report = run(args)
                self.assertEqual(called, ['cuda-source', str(args.probe)])
                self.assertEqual(report['exit_code'], 0)
                self.assertEqual(report['argmax_match'], 1)
                self.assertEqual(report['source_gpu_argmax_match'], 1)
                self.assertEqual(report['source_cuda_fixture_comparison']['argmax_match'], 0)


if __name__ == "__main__":
    unittest.main()
