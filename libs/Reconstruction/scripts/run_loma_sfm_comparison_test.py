#!/usr/bin/env python3
"""Tests for the L3 upstream binary provenance gate."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_loma_sfm_comparison as comparison


class RevisionMatchTest(unittest.TestCase):

    def test_accepts_full_and_abbreviated_revisions(self) -> None:
        revision = "3e5c915ca426e9661041d0696aff088dfc49fd90"
        self.assertTrue(comparison.revisions_match(revision, revision))
        self.assertTrue(comparison.revisions_match(revision, "3e5c915c"))
        self.assertTrue(comparison.revisions_match("3e5c915c", revision))

    def test_rejects_invalid_or_different_revisions(self) -> None:
        self.assertFalse(comparison.revisions_match("3e5c915c", "a2bd15c2"))
        self.assertFalse(comparison.revisions_match("main", "3e5c915c"))
        self.assertFalse(comparison.revisions_match("3e5c91", "3e5c915c"))


class BinaryRevisionTest(unittest.TestCase):

    @mock.patch.object(comparison.subprocess, "run")
    def test_parses_embedded_commit(self, run: mock.Mock) -> None:
        run.return_value = subprocess.CompletedProcess(
            ["colmap", "version"], 0,
            "COLMAP 4.3.0.dev0 (Commit 3e5c915c on 2026-10-12)\n")
        revision = comparison.colmap_binary_revision(Path("colmap"), [])
        self.assertEqual(revision, "3e5c915c")
        self.assertEqual(run.call_args.args[0], ["colmap", "version"])

    @mock.patch.object(comparison.subprocess, "run")
    def test_rejects_failed_version_command(self, run: mock.Mock) -> None:
        run.return_value = subprocess.CompletedProcess(
            ["colmap", "version"], 2, "unknown command\n")
        with self.assertRaisesRegex(RuntimeError, "exit 2"):
            comparison.colmap_binary_revision(Path("colmap"), [])

    @mock.patch.object(comparison.subprocess, "run")
    def test_rejects_unversioned_binary(self, run: mock.Mock) -> None:
        run.return_value = subprocess.CompletedProcess(
            ["colmap", "version"], 0, "COLMAP 4.3.0.dev0\n")
        with self.assertRaisesRegex(RuntimeError, "cannot parse"):
            comparison.colmap_binary_revision(Path("colmap"), [])


class ThresholdPolicyTest(unittest.TestCase):
    """The committed thresholds must separate the two measured populations."""

    THRESHOLDS = Path(__file__).resolve().parent / "l3_mini6_thresholds.json"
    COUNT_KEYS = ("registered_images_relative_difference",
                  "sparse_points_relative_difference",
                  "track_observations_relative_difference",
                  "mean_track_length_relative_difference")
    CONTINUOUS_KEYS = ("point_reprojection_rms_px_relative_difference",
                       "camera_center_p95", "camera_rotation_p95_deg",
                       "point_nn_p95")

    @classmethod
    def setUpClass(cls) -> None:
        cls.thresholds = comparison.json.loads(
            cls.THRESHOLDS.read_text(encoding="utf-8"))
        cls.provenance = cls.thresholds["provenance"]

    @staticmethod
    def observation(**values: float) -> tuple[dict, dict, dict]:
        """Build evaluate_l3 inputs whose observed metrics equal `values`."""
        baseline = {"registered_images": 6, "sparse_points": 406,
                    "track_observations": 1237, "mean_track_length": 3.0,
                    "point_reprojection_rms_px": 1.0}
        candidate = dict(baseline)
        track = values.get("track_observations_relative_difference", 0.0)
        candidate["track_observations"] = 1237 * (1.0 + track)
        candidate["mean_track_length"] = 3.0 * (1.0 + track)
        candidate["point_reprojection_rms_px"] = 1.0 + values.get(
            "point_reprojection_rms_px_relative_difference", 0.0)
        geometry = {
            "camera_center_error": {"p95": values.get("camera_center_p95", 0.0)},
            "camera_rotation_error_deg": {
                "p95": values.get("camera_rotation_p95_deg", 0.0)},
            "symmetric_point_nn_distance": {
                "p95": values.get("point_nn_p95", 0.0)},
        }
        return candidate, baseline, geometry

    def test_counts_are_exact(self) -> None:
        for key in self.COUNT_KEYS:
            self.assertEqual(self.thresholds[key], 0.0, key)

    def test_thresholds_follow_the_log_midpoint_rule(self) -> None:
        floor = self.provenance["noise_floor"]
        divergence = self.provenance["smallest_algorithmic_divergence"]
        for key in self.CONTINUOUS_KEYS:
            midpoint = (floor[key] * divergence[key]) ** 0.5
            self.assertLess(floor[key], self.thresholds[key], key)
            self.assertLess(self.thresholds[key], divergence[key], key)
            # One-significant-digit rounding stays within a factor of 2.
            self.assertLess(abs(comparison.math.log10(
                self.thresholds[key] / midpoint)), comparison.math.log10(2.0), key)

    def test_noise_floor_passes(self) -> None:
        floor = self.provenance["noise_floor"]
        values = {key: floor[key] for key in self.CONTINUOUS_KEYS}
        self.assertEqual(comparison.evaluate_l3(
            *self.observation(**values), self.thresholds), [])

    def test_measured_algorithmic_divergence_fails_every_metric(self) -> None:
        divergence = self.provenance["smallest_algorithmic_divergence"]
        values = {key: divergence[key] for key in self.CONTINUOUS_KEYS}
        values["track_observations_relative_difference"] = divergence[
            "track_observations_relative_difference"]
        violations = comparison.evaluate_l3(*self.observation(**values),
                                            self.thresholds)
        failed = {violation.split("=", 1)[0] for violation in violations}
        self.assertEqual(failed, set(self.COUNT_KEYS[2:]) | set(self.CONTINUOUS_KEYS))

    def test_thresholds_bind_their_fixture_configuration(self) -> None:
        expected = self.thresholds["configuration"]
        args = comparison.argparse.Namespace(
            max_features=expected["max_features"],
            camera_model=expected["camera_model"],
            single_camera=expected["single_camera"],
            sift_gpu=expected["sift_gpu"], seed=expected["seed"],
            mapper_option=list(reversed(expected["mapper_options"])))
        manifest = [{"path": "a.jpg", "sha256": "0" * 64}]
        actual = comparison.l3_configuration(args, manifest)
        mismatches = comparison.configuration_mismatches(expected, actual)
        # Only the synthetic image manifest differs; option order is irrelevant.
        self.assertEqual(len(mismatches), 1)
        self.assertTrue(mismatches[0].startswith("input_manifest_sha256"))
        args.seed += 1
        self.assertEqual(len(comparison.configuration_mismatches(
            expected, comparison.l3_configuration(args, manifest))), 2)

    def test_thresholds_without_configuration_fail_closed(self) -> None:
        actual = {"seed": 1, "max_features": 2048}
        self.assertEqual(len(comparison.configuration_mismatches({}, actual)), 2)


class DeterministicArgvTest(unittest.TestCase):
    """A seeded run must pin every run-dependent COLMAP input."""

    def run_variant_commands(self, seed: int | None) -> dict[str, list[str]]:
        args = comparison.argparse.Namespace(
            output=None, harness="harness", images=Path("images"),
            camera_model="SIMPLE_RADIAL", max_features=2048, single_camera=True,
            sift_gpu=False, seed=seed, mapper_option=[], runtime_library_dir=[])
        commands: dict[str, list[str]] = {}

        def record(command, log_path, colmap_binary, runtime_library_dirs):
            commands[log_path.name] = command
            return 0.0

        with tempfile.TemporaryDirectory() as output, \
                mock.patch.object(comparison, "run_stage", side_effect=record), \
                mock.patch.object(comparison, "choose_model", return_value=Path("m")), \
                mock.patch.object(comparison, "parse_text_model", return_value={}), \
                mock.patch.object(comparison, "database_counts", return_value={}):
            args.output = Path(output)
            comparison.run_variant(args, "candidate", "sift", None, Path("colmap"), 6)
        return commands

    def test_seed_pins_extraction_order_and_matching_ransac(self) -> None:
        commands = self.run_variant_commands(20261009)
        extract = commands["extract.log"]
        match = commands["match.log"]
        sparse = commands["mapper.log"]
        self.assertEqual(extract[extract.index("--num-threads") + 1], "1")
        self.assertEqual(match[match.index("--random-seed") + 1], "20261009")
        self.assertEqual(sparse[sparse.index("--random-seed") + 1], "20261009")

    def test_unseeded_run_keeps_colmap_defaults(self) -> None:
        commands = self.run_variant_commands(None)
        self.assertNotIn("--num-threads", commands["extract.log"])
        self.assertNotIn("--random-seed", commands["match.log"])
        self.assertNotIn("--random-seed", commands["mapper.log"])


if __name__ == "__main__":
    unittest.main()
