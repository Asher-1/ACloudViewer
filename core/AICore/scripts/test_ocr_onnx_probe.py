#!/usr/bin/env python3
import unittest
import json
import tempfile
import subprocess
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from run_ocr_onnx_probe import (accuracy_crops, accuracy_inputs,
    aggregate_fixtures, detector_input_limit, native_exit_is_diagnostic,
    probability_command, probability_gate, source_performance_pass)


class ProbabilityGateTests(unittest.TestCase):
    def test_quantized_raw_exit_needs_complete_outputs(self):
        self.assertTrue(native_exit_is_diagnostic(1, quantized=True,
                                                  outputs_complete=True))
        self.assertFalse(native_exit_is_diagnostic(1, quantized=True,
                                                   outputs_complete=False))
        self.assertFalse(native_exit_is_diagnostic(1, quantized=False,
                                                   outputs_complete=True))

    def test_release_detector_cannot_use_small_developer_input(self):
        self.assertEqual(detector_input_limit("developer", "det"), 320)
        self.assertEqual(detector_input_limit("release", "det"), 960)
        self.assertEqual(detector_input_limit("release", "det", 960), 960)
        for limit in [320, 640, 1024]:
            with self.assertRaisesRegex(ValueError, "source default limit 960"):
                detector_input_limit("release", "det", limit)

    def test_accuracy_limit_can_be_explicit_and_word_normalization_is_independent(self):
        self.assertEqual(detector_input_limit("developer", "det", 960), 960)
        self.assertEqual(detector_input_limit("release", "rec"), 320)
        self.assertEqual(detector_input_limit("release", "cls"), 320)
        with self.assertRaisesRegex(ValueError, "at least 32"):
            detector_input_limit("developer", "det", 0)

    def test_quantized_source_probability_and_argmax_are_diagnostic(self):
        metrics = dict(max_diff=.02, mean_diff=.001, stored_max_diff=.0001,
                       stored_mean_diff=.00001, argmax_match=.9,
                       stored_argmax_match=.9)
        self.assertTrue(probability_gate(metrics, "rec", quantized=True,
                                         max_diff=.001, stored_max_diff=.001))
        self.assertFalse(probability_gate(metrics, "rec", quantized=False,
                                          max_diff=.001, stored_max_diff=.001))
        metrics["stored_max_diff"] = .002
        self.assertFalse(probability_gate(metrics, "rec", quantized=True,
                                          max_diff=.001, stored_max_diff=.001))

    def fixture(self, **kwargs):
        result = {"role": "rec", "max_diff": .00001, "mean_diff": .000001,
                  "stored_max_diff": .00001, "stored_mean_diff": .000001,
                  "cpu_max_diff": .00001, "cpu_mean_diff": .000001,
                  "argmax_match": 1, "stored_argmax_match": 1, "cpu_argmax_match": 1,
                  "output_sha256": "1234", "p50_ms": 10, "exit_code": 0}
        return dict(result, **kwargs)

    def test_bad_word_cannot_be_hidden_by_good_words_or_median(self):
        rows = [self.fixture() for _ in range(5)]
        rows[2] = self.fixture(max_diff=.1, argmax_match=.9, exit_code=1)
        result = aggregate_fixtures(rows, [0, 0, 1, 0, 0])
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(result["max_diff"], .1)
        self.assertEqual(result["argmax_match"], .9)

    def test_classifier_includes_every_upright_and_rotated_input(self):
        args = SimpleNamespace(role="cls", image=Path("a.png"),
                               extra_image=[Path("b.png")], include_rotated=True)
        self.assertEqual(accuracy_inputs(args), [(Path("a.png"), False),
                         (Path("b.png"), False), (Path("a.png"), True), (Path("b.png"), True)])
        args.role = "rec"
        with self.assertRaisesRegex(ValueError, "classifier"):
            accuracy_inputs(args)

    def test_additional_accuracy_does_not_change_primary_timing(self):
        result = aggregate_fixtures([self.fixture(), self.fixture(p50_ms=500)], [0, 0])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["p50_ms"], 10)
        self.assertEqual(result["accuracy_fixture_count"], 2)
        self.assertEqual(result["accuracy_fixture_requested_count"], 2)
        self.assertEqual(result["fixture_pass_count"], 2)
        self.assertEqual(result["fixture_incomplete_indices"], [])

    def test_preinference_audit_rejection_does_not_count_as_accuracy_coverage(self):
        rejected = {"role": "cls", "exit_code": 1, "gpu_audit_pass": 0,
                    "error": "device was not idle"}
        result = aggregate_fixtures([rejected, self.fixture(role="cls")], [1, 0])
        self.assertEqual(result["accuracy_fixture_requested_count"], 2)
        self.assertEqual(result["accuracy_fixture_count"], 1)
        self.assertEqual(result["fixture_pass_count"], 1)
        self.assertEqual(result["fixture_incomplete_indices"], [0])
        self.assertEqual(result["exit_code"], 1)

    def test_completed_numeric_failure_is_coverage_but_not_a_pass(self):
        result = aggregate_fixtures([self.fixture(argmax_match=.9, exit_code=1)], [1])
        self.assertEqual(result["accuracy_fixture_count"], 1)
        self.assertEqual(result["fixture_pass_count"], 0)
        self.assertEqual(result["fixture_incomplete_indices"], [])
        self.assertEqual(result["exit_code"], 1)

    def test_detector_coverage_does_not_require_classifier_argmax(self):
        fixture = self.fixture(role="det")
        fixture.pop("argmax_match")
        fixture.pop("stored_argmax_match")
        result = aggregate_fixtures([fixture], [0])
        self.assertEqual(result["accuracy_fixture_count"], 1)
        self.assertEqual(result["fixture_pass_count"], 1)
        self.assertEqual(result["exit_code"], 0)

    def test_native_cpu_diagnostics_are_not_required_for_gpu_qualification(self):
        rows = [self.fixture(), self.fixture(cpu_argmax_match=0, cpu_max_diff=1)]
        for row in rows:
            row.pop("cpu_mean_diff")
        self.assertEqual(aggregate_fixtures(rows, [0, 0])["exit_code"], 0)
        for row in rows:
            row.pop("cpu_argmax_match")
            row.pop("cpu_max_diff")
        self.assertEqual(aggregate_fixtures(rows, [0, 0])["exit_code"], 0)

    def test_original_source_and_stored_weight_metrics_remain_required(self):
        for key in ("max_diff", "mean_diff", "argmax_match", "stored_max_diff",
                    "stored_mean_diff", "stored_argmax_match"):
            other = self.fixture()
            other.pop(key)
            self.assertEqual(aggregate_fixtures([self.fixture(), other], [0, 0])["exit_code"], 1)

    def test_page_regression_retains_coordinates_and_does_not_rotate(self):
        args = SimpleNamespace(role="rec", extra_crop=[
            ["page.jpg", "1132", "353", "1195", "353", "1195", "368", "1132", "368"]])
        self.assertEqual(accuracy_crops(args), [(Path("page.jpg"), False,
                         [1132., 353., 1195., 353., 1195., 368., 1132., 368.])])
        args.role = "cls"
        with self.assertRaisesRegex(ValueError, "recognizer"):
            accuracy_crops(args)

    def test_invalid_crop_cannot_be_silently_omitted(self):
        args = SimpleNamespace(role="rec", extra_crop=[["page.jpg", "0"]])
        with self.assertRaisesRegex(ValueError, "eight"):
            accuracy_crops(args)
        args.extra_crop = [["page.jpg", "nan", *["0"] * 7]]
        with self.assertRaisesRegex(ValueError, "finite"):
            accuracy_crops(args)

    def test_failed_fixture_with_missing_metrics_cannot_inherit_primary_pass(self):
        result = aggregate_fixtures([self.fixture(), {"exit_code": 1}], [0, 1])
        self.assertEqual(result["exit_code"], 1)
        self.assertNotIn("max_diff", result)
        self.assertNotIn("output_sha256", result)

    def test_nonfinite_fixture_cannot_disappear_in_maximum(self):
        result = aggregate_fixtures([self.fixture(), self.fixture(max_diff=float("nan"))], [0, 0])
        self.assertEqual(result["exit_code"], 1)
        self.assertNotIn("max_diff", result)

    def test_cpu_comparison_preserves_threads_and_disables_measurement_hold(self):
        args = SimpleNamespace(probe=Path("probe"), model=Path("model.gguf"),
                               role="rec", backend="vulkan", max_diff=.001,
                               warmups=2, runs=10, threads=8,
                               measurement_profile="release")
        native = probability_command(args, Path("fixture"))
        cpu = probability_command(args, Path("fixture"), cpu=True)
        self.assertEqual(native[8:], ["2", "10", "8", "1500"])
        self.assertEqual(cpu[4], "cpu")
        self.assertEqual(cpu[6], "dump=fixture/cpu.bin")
        self.assertEqual(cpu[8:], ["0", "1", "8", "0"])

    def test_source_performance_requires_both_percent_and_absolute_regression(self):
        source = {"p50_ms": 100., "p95_ms": 100.}
        self.assertFalse(source_performance_pass({"p50_ms": 106., "p95_ms": 100.}, source))
        self.assertFalse(source_performance_pass({"p50_ms": 100., "p95_ms": 106.}, source))
        self.assertTrue(source_performance_pass({"p50_ms": 104., "p95_ms": 104.}, source))
        self.assertTrue(source_performance_pass({"p50_ms": 2., "p95_ms": 2.},
                                                {"p50_ms": 1., "p95_ms": 1.}))
        for invalid in (None, 0., float("nan"), float("inf")):
            self.assertFalse(source_performance_pass({"p50_ms": invalid, "p95_ms": 100.}, source))


class CudaTruthFlowTests(unittest.TestCase):
    def test_failed_export_cannot_leave_a_previous_success_report(self):
        from run_ocr_onnx_probe import run_probe
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = root / "report.json"
            report.write_text(json.dumps(ProbabilityGateTests().fixture()))
            args = SimpleNamespace(report=report, role="cls", backend="cpu",
                source=root / "source.onnx", source_sha256="pinned", image=root / "word.png",
                model=root / "model.gguf", max_side=320, threads=4)
            failed = subprocess.CompletedProcess([], 9, "", "export failed")
            with patch("run_ocr_onnx_probe.subprocess.run", return_value=failed):
                self.assertEqual(run_probe(args), 9)
            current = json.loads(report.read_text())
            self.assertEqual(current["exit_code"], 9)
            self.assertEqual(current["error"], "source fixture export failed")
            self.assertNotIn("output_sha256", current)
            self.assertNotIn("argmax_match", current)
            self.assertEqual(aggregate_fixtures([current], [9])["accuracy_fixture_count"], 0)

    def test_gpu_native_is_judged_against_cuda_for_original_and_stored_weights(self):
        import numpy as np
        from run_ocr_onnx_probe import run_probe
        for backend in ("cuda", "vulkan"):
            for matches_cuda in (True, False):
                with self.subTest(backend=backend, matches_cuda=matches_cuda), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    model, python = root / "model.gguf", root / "source-python"
                    model.write_bytes(b"weights")
                    python.touch()
                    cpu = np.array([[[.500001, .499999]]], np.float32)
                    cuda = cpu[..., ::-1].copy()
                    args = SimpleNamespace(report=root / "report.json", role="cls", backend=backend,
                        source=root / "source.onnx", source_sha256="pinned", image=root / "word.png",
                        max_side=320, model=model, threads=4, source_benchmark_python=python,
                        probe=root / "probe", max_diff=.001, stored_max_diff=.001,
                        warmups=0, runs=1, measurement_profile="developer", gpu_idle_timeout=1)
                    export_calls = []
                    def call(command, **kwargs):
                        if command[0] == str(python):
                            export_calls.append(command)
                            self.assertEqual(command[command.index("--provider") + 1], "cuda")
                            self.assertEqual(command[command.index("--weights") + 1], str(model))
                            self.assertIn("env", kwargs)
                            work = Path(command[4])
                            np.save(work / "cls_input.npy", np.ones((1, 3, 48, 192), np.float32))
                            np.save(work / "cls_reference.npy", cuda)
                            np.save(work / "cls_stored_reference.npy", cuda)
                            (work / "reference.json").write_text(json.dumps({
                                "provider": ["CUDAExecutionProvider"],
                                "stored_weights": {"provider": ["CUDAExecutionProvider"]}}))
                            return subprocess.CompletedProcess(command, 0, "", "")
                        self.assertEqual(command[0], str(args.probe))
                        work = Path(command[1])
                        np.testing.assert_array_equal(np.load(work / "cls_reference.npy"), cuda)
                        np.testing.assert_array_equal(np.load(work / "cls_stored_reference.npy"), cuda)
                        (cuda if matches_cuda else cpu).tofile(work / "native.bin")
                        return subprocess.CompletedProcess(command, 0, '{"p50_ms": 1, "p95_ms": 1}', "")
                    def benchmark(command, audit, **kwargs):
                        report = Path(command[command.index("--report") + 1])
                        report.write_text(json.dumps({"p50_ms": 1, "p95_ms": 1}))
                        np.save(report.with_suffix(".npy"), cuda)
                        return subprocess.CompletedProcess(command, 0, "", ""), {
                            "passed": True, "audit_id": "test", "archive_path": "test",
                            "sampled_process_peak_mib": 1}
                    with patch("run_ocr_onnx_probe.subprocess.run", side_effect=call), \
                         patch("run_ocr_onnx_probe.run_audited", side_effect=benchmark):
                        code = run_probe(args)
                    report = json.loads(args.report.read_text())
                    self.assertEqual(len(export_calls), 1)
                    self.assertEqual(code, int(not matches_cuda))
                    self.assertEqual(report["argmax_match"], int(matches_cuda))
                    self.assertEqual(report["stored_argmax_match"], int(matches_cuda))
                    self.assertEqual(report["accuracy_reference"], "original and stored-weight ONNX CUDA")

    def test_extra_accuracy_fixtures_keep_cuda_reference_interpreter(self):
        import sys
        import run_ocr_onnx_probe as probe
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / "report.json"
            calls = []
            def run(args):
                calls.append(args)
                args.report.write_text(json.dumps(ProbabilityGateTests().fixture()))
                return 0
            command = ["probe", "--source", "source", "--source-sha256", "pinned",
                "--image", "a.png", "--extra-image", "b.png", "--model", "model",
                "--probe", "binary", "--backend", "vulkan", "--role", "cls",
                "--source-benchmark-python", "cuda-python", "--report", str(report)]
            with patch.object(sys, "argv", command), patch.object(probe, "run_probe", side_effect=run):
                self.assertEqual(probe.main(), 0)
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(c.source_benchmark_python == Path("cuda-python") for c in calls))
            self.assertEqual((calls[1].warmups, calls[1].runs), (0, 1))


class ProviderContractTests(unittest.TestCase):
    def test_requested_cuda_provider_cannot_silently_fall_back(self):
        import importlib.util
        import sys
        from unittest.mock import Mock
        location = Path(__file__).resolve().parents[1] / "tools/export_onnx_ocr_fixture.py"
        spec = importlib.util.spec_from_file_location("fixture_exporter", location)
        exporter = importlib.util.module_from_spec(spec)
        # Exercise provider selection without requiring optional ORT/OpenCV
        # packages in the fast unit-test environment. Real graphs are separate.
        with patch.dict(sys.modules, {"cv2": Mock(), "onnxruntime": Mock()}):
            spec.loader.exec_module(exporter)
        for model in ("original.onnx", b"stored-weight-original-topology"):
            with self.subTest(model=model):
                wrong = Mock()
                wrong.get_providers.return_value = ["CPUExecutionProvider"]
                with patch.object(exporter.ort, "InferenceSession", return_value=wrong) as create:
                    with self.assertRaisesRegex(RuntimeError, "silently fell back"):
                        exporter.reference_session(model, "cuda", 4)
                    self.assertEqual(create.call_args.kwargs["providers"],
                        [("CUDAExecutionProvider", {"use_tf32": "0"})])
                correct = Mock()
                correct.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                with patch.object(exporter.ort, "InferenceSession", return_value=correct):
                    self.assertIs(exporter.reference_session(model, "cuda", 4), correct)

    def test_orientation_and_ctc_export_use_original_cuda(self):
        import run_ocr_onnx_probe as probe
        for backend in ("cuda", "vulkan"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                python, model = root / "cuda-python", root / "model.gguf"
                python.touch(); model.write_bytes(b"weights")
                args = SimpleNamespace(report=root / "report.json", role="orientation", backend=backend,
                    source=root / "source.onnx", source_sha256="pinned", image=root / "words",
                    max_side=320, model=model, threads=4, source_benchmark_python=python,
                    probe=root / "probe", det=root / "det", cls=root / "cls",
                    rec_source=root / "rec.onnx", rec_source_sha256="rec-pinned",
                    dictionary=root / "dictionary", dictionary_sha256="dictionary-pinned")
                def call(command, **kwargs):
                    if command[0] == str(python):
                        self.assertEqual(command[command.index("--provider") + 1], "cuda")
                        self.assertEqual(command[command.index("--rec-source") + 1], str(args.rec_source))
                        self.assertEqual(command[command.index("--rec-source-sha256") + 1], "rec-pinned")
                        self.assertEqual(command[command.index("--dictionary-sha256") + 1], "dictionary-pinned")
                    else:
                        self.assertEqual(command[0], str(args.probe))
                        Path(command[5]).write_text(json.dumps({"providers": ["CUDAExecutionProvider"]}))
                    return subprocess.CompletedProcess(command, 0, "", "")
                with patch.object(probe.subprocess, "run", side_effect=call):
                    self.assertEqual(probe.run_probe(args), 0)
                self.assertEqual(json.loads(args.report.read_text())["accuracy_reference"],
                    "original ONNX CUDA orientation and CTC")


if __name__ == "__main__":
    unittest.main()
