#!/usr/bin/env python3
import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import gpu_audit


class AuditTests(unittest.TestCase):
    def sample(self, pid=123, kind="C", memory=42, utilization=0):
        return {"devices": [{"uuid": "gpu", "utilization_pct": utilization,
                             "used_mib": 1024, "processes": [
                                 {"pid": pid, "type": kind, "used_mib": memory}]}],
                "activity": [{"pid": pid, "type": kind, "sm_pct": 0}]}

    def test_graphics_memory_belongs_to_vulkan_owner(self):
        summary = gpu_audit.summarize_samples([self.sample(kind="G")], 123)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["sampled_process_peak_mib"], 42)

    def test_idle_external_compute_context_invalidates_measurement(self):
        sample = self.sample()
        sample["devices"][0]["processes"].append(
            {"pid": 99, "type": "C", "used_mib": 1})
        self.assertFalse(gpu_audit.summarize_samples([sample], 123)["passed"])

    def test_graphics_only_activity_does_not_invalidate_measurement(self):
        sample = self.sample()
        sample["devices"][0]["utilization_pct"] = 82
        sample["activity"].append({"pid": 99, "type": "G", "sm_pct": 96})
        self.assertEqual(gpu_audit.external_activity(sample, 123), [])
        summary = gpu_audit.summarize_samples([sample], 123)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["normal_graphics_pids"], [99])

    def test_external_compute_and_mixed_compute_graphics_contexts_block(self):
        for kind in ("C", "C+G"):
            sample = self.sample()
            sample["devices"][0]["processes"].append(
                {"pid": 99, "type": kind, "used_mib": 1})
            sample["activity"].append(
                {"pid": 99, "type": kind, "sm_pct": 0})
            self.assertEqual(gpu_audit.external_activity(sample, 123), [99])
            self.assertFalse(gpu_audit.summarize_samples([sample], 123)["passed"])

    def test_missing_process_memory_cannot_pass(self):
        for samples in ([], [self.sample(memory=None)], [self.sample(pid=99, kind="G")]):
            self.assertFalse(gpu_audit.summarize_samples(samples, 123)["passed"])
        self.assertFalse(gpu_audit.summarize_samples([self.sample()], 123, ["error"])["passed"])

    def test_parse_process_activity_including_unknown_utilization(self):
        result = gpu_audit.parse_activity("# header\n0 123 G - - test\n0 234 C 12 0 test\n")
        self.assertIsNone(result[0]["sm_pct"])
        self.assertEqual(result[1]["sm_pct"], 12)
        with self.assertRaises(ValueError):
            gpu_audit.parse_activity("bad telemetry")
        with self.assertRaises(ValueError):
            gpu_audit.parse_activity("")
        with self.assertRaises(ValueError):
            gpu_audit.parse_devices("<nvidia_smi_log><gpu/></nvidia_smi_log>")

    def test_missing_telemetry_stops_strict_probe_and_keeps_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / "audit.json"
            with patch.object(gpu_audit, "sample_driver", side_effect=OSError("missing")), \
                 patch.object(subprocess, "Popen") as process:
                with self.assertRaisesRegex(RuntimeError, "telemetry unavailable"):
                    gpu_audit.run_audited(["probe"], report, strict=True)
            process.assert_not_called()
            self.assertIn('"passed": false', report.read_text())

    def test_strict_preflight_allows_desktop_graphics_and_high_utilization(self):
        desktop = self.sample(pid=99, kind="G", utilization=82)
        desktop["activity"][0]["sm_pct"] = 96
        measured = self.sample(utilization=82)
        measured["devices"][0]["processes"].append(
            {"pid": 99, "type": "G", "used_mib": 100})
        measured["activity"].append({"pid": 99, "type": "G", "sm_pct": 96})
        sampled = 0
        monitor_sampled = threading.Event()

        def driver_sample():
            nonlocal sampled
            sampled += 1
            if sampled <= 3:
                return desktop
            monitor_sampled.set()
            return measured

        with tempfile.TemporaryDirectory() as temp:
            child = unittest.mock.Mock(pid=123, returncode=0)
            def communicate():
                if not monitor_sampled.wait(3):
                    child.returncode = 1
                    return "", "monitor timeout"
                return "ok", ""
            child.communicate.side_effect = communicate
            with patch.object(gpu_audit, "sample_driver", side_effect=driver_sample), \
                 patch.object(subprocess, "Popen", return_value=child):
                result, audit = gpu_audit.run_audited(
                    ["probe"], Path(temp) / "audit.json", strict=True)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(audit["preflight_passed"])
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["normal_graphics_pids"], [99])
        self.assertEqual(audit["waiting_samples"][-1]["devices"][0]["utilization_pct"], 82)

    def test_observational_run_never_becomes_controlled_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            child = unittest.mock.Mock(pid=123, returncode=0)
            child.communicate.return_value = ("ok", "")
            with patch.object(gpu_audit, "sample_driver", return_value=self.sample()), \
                 patch.object(subprocess, "Popen", return_value=child):
                result, audit = gpu_audit.run_audited(["probe"], Path(temp) / "audit.json")
            self.assertEqual(result.returncode, 0)
            self.assertFalse(audit["passed"])

    def test_repeated_failures_keep_distinct_audits(self):
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / "audit.json"
            attempts = []
            with patch.object(gpu_audit, "sample_driver", side_effect=OSError("missing")), \
                 patch.object(subprocess, "Popen") as process:
                for _ in range(2):
                    with self.assertRaisesRegex(RuntimeError, "telemetry unavailable"):
                        gpu_audit.run_audited(["probe"], report, strict=True)
                    attempts.append(json.loads(report.read_text()))
            process.assert_not_called()
            self.assertNotEqual(attempts[0]["audit_id"], attempts[1]["audit_id"])
            archives = sorted(report.with_suffix(".d").glob("*.json"))
            self.assertEqual(len(archives), 2)
            for attempt in attempts:
                self.assertFalse(attempt["passed"])
                self.assertEqual(json.loads(Path(attempt["archive_path"]).read_text()),
                                 attempt)
            self.assertEqual(json.loads(report.read_text()), attempts[1])


if __name__ == "__main__":
    unittest.main()
