#!/usr/bin/env python3
"""Contract checks for GPU overlap rejection and baseline library isolation."""
import os
import sys
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import demo_reference_probe
import tagging_reference
import cpu_precision_reference
from upstream_metrics_probe import (active_graphics_processes,
                                    aicore_latency_ms, execute_audited,
                                    gpu_overlap_pids, include_graphics_audit,
                                    metrics_for_backend, probability_alignment,
                                    truth_cache, upstream_backend_env)


class IndependentAccuracyTests(unittest.TestCase):
    def test_cpu_q8_reference_uses_block_scaling_and_half_scale_storage(self):
        import torch
        value = torch.zeros(2, 32)
        value[0, 0] = 2.0
        value[0, 1] = 2.0 / 127.0 * 2.5
        value[0, 2] = -2.0 / 127.0 * 3.5
        restored, = cpu_precision_reference.quantized_input(None, (value,))
        scale = torch.tensor(2.0 / 127.0).half().float()
        self.assertEqual(restored[0, 0], 127 * scale)
        self.assertEqual(restored[0, 1], 2 * scale)
        self.assertEqual(restored[0, 2], -4 * scale)
        self.assertTrue(torch.isfinite(restored).all())
        self.assertTrue(torch.equal(restored[1], torch.zeros(32)))

    def test_reference_device_overrides_upstream_batch_helper_default(self):
        # Upstream's Tag2Text helper chooses CUDA at import time even when the
        # evaluator explicitly requests CPU. Exercise the complete adapter
        # without loading checkpoint weights or leaving generated fixtures.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = root / "cpp_ggml/benchmarks/python_eval.py"
            evaluator.parent.mkdir(parents=True)
            evaluator.write_text(
                "import argparse\n"
                "import batch_inference\n"
                "def parse_args():\n"
                "    parser = argparse.ArgumentParser()\n"
                "    parser.add_argument('--device', required=True)\n"
                "    return parser.parse_args()\n"
                "def main():\n"
                "    assert batch_inference.device == parse_args().device\n"
                "    return 7\n")
            helper = SimpleNamespace(device="cuda")
            with patch.dict(sys.modules, {
                    "batch_inference": helper,
                    "torch": SimpleNamespace(device=lambda value: value)}), \
                    patch.object(sys, "argv", ["tagging_reference.py", "--upstream",
                                               str(root), "--device", "cpu"]), \
                    patch.object(sys, "path", list(sys.path)):
                self.assertEqual(tagging_reference.main(), 7)
                self.assertEqual(helper.device, "cpu")

    def test_gpu_report_keeps_upstream_deltas_but_not_cpu_compatibility_claims(self):
        metrics = {
            "upstream_logit_max_error": 0.05,
            "compatibility_upstream_logit_max_error": 0.05,
            "compatibility_pytorch_cross_f1_drop": 0.0,
            "cross_f1": 0.995,
        }
        gpu = metrics_for_backend(metrics, "cuda")
        self.assertEqual(gpu["upstream_logit_max_error"], 0.05)
        self.assertEqual(gpu["cross_f1"], 0.995)
        self.assertNotIn("compatibility_upstream_logit_max_error", gpu)
        self.assertNotIn("compatibility_pytorch_cross_f1_drop", gpu)
        self.assertEqual(metrics_for_backend(metrics, "cpu"), metrics)

    def test_same_quantized_upstream_error_has_no_integration_excess(self):
        import numpy as np
        truth = np.array([.49, .9, .1], dtype=np.float32)
        quantized = np.array([.51, .89, .11], dtype=np.float32)
        metrics = probability_alignment(quantized, quantized, truth)
        self.assertGreater(metrics["pytorch_probability_mae"], 0)
        self.assertEqual(metrics["pytorch_probability_error_excess"], 0)

    def test_changed_scores_are_detected_even_when_selected_tags_do_not_change(self):
        import numpy as np
        truth = np.array([.2, .9], dtype=np.float32)
        upstream = np.array([.21, .89], dtype=np.float32)
        changed = np.array([.3, .8], dtype=np.float32)
        self.assertEqual(list(upstream > .5), list(changed > .5))
        self.assertGreater(probability_alignment(changed, upstream, truth)[
            "pytorch_probability_error_excess"], .08)

    def test_one_improved_score_cannot_hide_another_worsened_score(self):
        import numpy as np
        truth = np.array([.2, .9], dtype=np.float32)
        upstream = np.array([.3, .8], dtype=np.float32)
        changed = np.array([.2, .79], dtype=np.float32)
        metrics = probability_alignment(changed, upstream, truth)
        self.assertLess(metrics["pytorch_probability_mae"],
                        metrics["upstream_pytorch_probability_mae"])
        self.assertGreater(metrics["pytorch_probability_error_excess"], .009)


class TimingAuditTests(unittest.TestCase):
    def test_aicore_latency_uses_typed_pipeline_timing_not_caller_decode_wall(self):
        self.assertEqual(aicore_latency_ms({"latency_ms": 12.0,
                                            "e2e_ms": 9.5}), 9.5)
        self.assertEqual(aicore_latency_ms({"latency_ms": 12.0}), 12.0)

    def test_demo_wrapper_forwards_probe_inputs_and_threads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "ram_swin_large_14m-q8_0.gguf"
            image = root / "demo1.jpg"
            cache = root / "cache"
            model.write_bytes(b"model")
            image.write_bytes(b"image")
            cache.mkdir()
            (cache / "ram_swin_large_14m-q8_0-demo1-pytorch-gguf-logits.f32").write_bytes(
                b"reference")
            args = type("Args", (), {
                "probe": root / "test_tagging_probe",
                "model": model,
                "image": image,
                "backend": "cuda",
                "reference_cache": cache,
                "threads": 5,
                "cpu_precision": 0,
            })()
            completed = type("Result", (), {"stdout": '{"accuracy_max_diff": 0}\n',
                                             "stderr": "", "returncode": 0})()
            with patch.dict(os.environ, {"AICORE_TEST_RAM_UPSTREAM": str(root)}), \
                    patch.object(demo_reference_probe, "cache_for", return_value=cache), \
                    patch.object(demo_reference_probe.subprocess, "run", return_value=completed) as run:
                self.assertEqual(demo_reference_probe.materialize(args, ["--repeat", "2"]), 0)
            command = run.call_args.args[0]
            self.assertEqual(command[:7], [str(args.probe.resolve()), "--model",
                                            str(model.resolve()), "--backend", "cuda",
                                            "--image", str(image.resolve())])
            self.assertIn(("--threads", "5"), tuple(zip(command, command[1:])))
            self.assertIn(("--repeat", "2"), tuple(zip(command, command[1:])))
            self.assertEqual(command[-2:], ["--reference-dir", str(cache.resolve())])

    def test_cpu_truth_cannot_reuse_a_cuda_reference_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, checkpoint, manifest = (root / name for name in
                                           ("image", "checkpoint", "manifest"))
            image.write_bytes(b"image bytes")
            checkpoint.write_bytes(b"checkpoint bytes")
            manifest.write_text(str(image) + "\n")
            args = (root, "revision", "tag2text", "caption", manifest,
                    [{"image": str(image)}], [], checkpoint)
            default = truth_cache(*args)
            self.assertEqual(default, truth_cache(*args, reference_device="cuda"))
            self.assertNotEqual(default, truth_cache(*args, reference_device="cpu"))

    def test_desktop_graphics_do_not_override_allowed_compute_contexts(self):
        samples = [{"pids": [17, 42, 99], "active_graphics_pids": [42, 99]}]
        self.assertEqual(gpu_overlap_pids(samples, {17, 42}, 99), [])

    def test_external_compute_is_rejected_while_desktop_graphics_are_observational(self):
        samples = [{"pids": [17, 42], "active_graphics_pids": [99]}]
        self.assertEqual(gpu_overlap_pids(samples, {17}, None), [42])

    def test_idle_allowed_context_and_monitored_child_are_accepted(self):
        samples = [{"pids": [17, 42, 99], "active_graphics_pids": [99]}]
        self.assertEqual(gpu_overlap_pids(samples, {17, 42}, 99), [])

    def test_graphics_audit_detects_active_desktop_and_combined_processes(self):
        output = """# gpu pid type sm mem enc dec jpg ofa command
0 3235 G 27 15 - - - - Xorg
0 170927 C+G 73 42 - - - - test_tagging_pr
0 3282618 G 0 0 - - - - chrome
0 1225 G - - - - - - Xorg
0 165593 C 0 0 - - - - python3
"""
        self.assertEqual(active_graphics_processes(output), {3235, 170927})

    def test_graphics_audit_preserves_original_compute_processes(self):
        class Evaluation:
            @staticmethod
            def gpu_process_ids():
                return {17}
        with patch("upstream_metrics_probe.subprocess.run") as run:
            run.return_value.stdout = "0 42 G 10 1 - - - - Xorg\n"
            include_graphics_audit(Evaluation)
            self.assertEqual(Evaluation.gpu_process_ids(), {17})
            run.assert_called_once()

    def test_persistent_graphics_activity_is_observational(self):
        class Evaluation:
            @staticmethod
            def gpu_process_ids():
                return set()
        with patch("upstream_metrics_probe.subprocess.run") as run, \
                patch("upstream_metrics_probe.time.monotonic", side_effect=[0, 61]):
            run.return_value.stdout = "0 42 G 10 1 - - - - Xorg\n"
            include_graphics_audit(Evaluation, idle_timeout=60)
            self.assertEqual(Evaluation.gpu_process_ids(), set())
            self.assertEqual(Evaluation.gpu_process_ids(), set())

    def test_overlap_is_preserved_and_retried(self):
        calls = []

        class Evaluation:
            @staticmethod
            def run_command_monitored(command, log, allowed):
                calls.append((command, allowed, os.environ.get("RAM_TEST_ENV")))
                log.write_text("overlap" if len(calls) == 1 else "clean")
                return {"transient_gpu_pids": [42] if len(calls) == 1 else []}

        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "sample.log"
            with patch.dict(os.environ, {"RAM_TEST_ENV": "candidate"}):
                env = dict(os.environ, RAM_TEST_ENV="baseline")
                execute_audited([Path("upstream")], log, Evaluation, {17}, env)
                self.assertEqual(os.environ["RAM_TEST_ENV"], "candidate")
            self.assertEqual(log.read_text(), "clean")
            self.assertEqual(log.with_name("sample.overlap-1.log").read_text(), "overlap")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], (["upstream"], {17}, "baseline"))

    def test_persistent_overlap_fails_instead_of_using_dirty_timings(self):
        class Evaluation:
            @staticmethod
            def run_command_monitored(command, log, allowed):
                log.write_text("overlap")
                return {"transient_gpu_pids": [42]}

        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "sample.log"
            with self.assertRaisesRegex(RuntimeError, "overlap persisted") as failure:
                execute_audited(["upstream"], log, Evaluation, set(), retries=1)
            self.assertFalse(log.exists())
            self.assertEqual(len(list(Path(directory).glob("*.overlap-*.log"))), 2)
            retained = log.with_name("sample.overlap-2.log")
            self.assertTrue(str(failure.exception).endswith(str(retained)))
            self.assertTrue(retained.is_file())

    def test_failed_command_restores_candidate_environment(self):
        class Evaluation:
            @staticmethod
            def run_command_monitored(command, log, allowed):
                raise RuntimeError("failed command")

        with patch.dict(os.environ, {"RAM_TEST_ENV": "candidate"}):
            with self.assertRaisesRegex(RuntimeError, "failed command"):
                execute_audited(["upstream"], Path("unused"), Evaluation, set(),
                                dict(os.environ, RAM_TEST_ENV="baseline"))
            self.assertEqual(os.environ["RAM_TEST_ENV"], "candidate")

    def test_baseline_uses_its_own_ggml_libraries(self):
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": "/candidate/bin",
                                     "GGML_BACKEND_PATH": "/candidate/backends"}):
            env = upstream_backend_env(Path("/reference"), "cuda")
        self.assertNotIn("GGML_BACKEND_PATH", env)
        if os.name != "nt":
            self.assertTrue(env.get("LD_LIBRARY_PATH", env.get("DYLD_LIBRARY_PATH", "")).startswith(
                "/reference/cpp_ggml/build-cuda/third_party/ggml/src"))

    def test_alternate_baseline_build_does_not_load_default_build_libraries(self):
        env = upstream_backend_env(Path("/reference"), "cpu", Path("/matched-flags"))
        variable = "PATH" if os.name == "nt" else "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        self.assertTrue(env[variable].startswith("/matched-flags/third_party/ggml/src"))


if __name__ == "__main__":
    unittest.main()
