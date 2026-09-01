#!/usr/bin/env python3

import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).with_name("validate_all.py")
SPEC = importlib.util.spec_from_file_location("aicore_validate_all", MODULE_PATH)
assert SPEC and SPEC.loader
VALIDATE_ALL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATE_ALL
SPEC.loader.exec_module(VALIDATE_ALL)


class ValidateAllTests(unittest.TestCase):
    @staticmethod
    def report(rows):
        return {
            "backend": "cuda", "repeats": 2, "warmup_runs": 2,
            "inference_runs": 10, "threads": 0, "trellis_steps": 12,
            "manifest": {"sha256": "manifest"}, "inputs": {},
            "host": {"machine": "x86_64", "cpu_count": 8,
                     "nvidia": "gpu"},
            "results": rows,
        }

    def test_parse_validation_json_and_rmbg_summary(self):
        output = "\n".join([
            '{"suite":"aicore-validation","task":"depth",'
            '"inference_ms":12.5,"e2e_ms":15.0,'
            '"output_hash":"abc123"}',
            "[rmbg-perf] device=cuda median_ms=561.5 p95_ms=580.4 "
            "output_hash=12345",
        ])
        metrics, fingerprints = VALIDATE_ALL.parse_output(
            output, "rmbg", Path("/does/not/exist"))
        self.assertEqual(metrics["inference_p50_ms"], 561.5)
        self.assertEqual(metrics["inference_p95_ms"], 580.4)
        self.assertEqual(metrics["json0/inference_ms"], 12.5)
        self.assertEqual(fingerprints["output_hash"], "12345")
        self.assertEqual(fingerprints["json0/output_hash"], "abc123")

    def test_fingerprint_stability_requires_every_attempt_to_match(self):
        Attempt = VALIDATE_ALL.Attempt
        stable, failures = VALIDATE_ALL.stable_fingerprints([
            Attempt(0, 1.0, fingerprints={"output": "aa"}),
            Attempt(0, 1.0, fingerprints={"output": "bb"}),
        ])
        self.assertFalse(stable)
        self.assertEqual(len(failures), 1)

    def test_compare_uses_asset_content_not_absolute_path(self):
        asset_a = {"path": "/baseline/model.gguf", "bytes": 10,
                   "sha256": "feed"}
        asset_b = {"path": "/candidate/model.gguf", "bytes": 10,
                   "sha256": "feed"}
        baseline = self.report([{
            "key": "depth/depth/model.gguf",
            "model_assets": [asset_a],
            "metrics": {"inference_ms": 100.0},
            "fingerprints": {"output": "a"},
        }])
        current = self.report([{
            "key": "depth/depth/model.gguf",
            "model_assets": [asset_b],
            "metrics": {"inference_ms": 106.0},
            "fingerprints": {"output": "b"},
            "fingerprint_policy": "stability_only",
        }])
        failures = VALIDATE_ALL.compare_report(current, baseline, 5.0, 3.0)
        self.assertEqual(len(failures), 1)
        self.assertIn("regressed 6.00%", failures[0])

    def test_accuracy_metrics_are_not_treated_as_latency(self):
        row = {
            "key": "sam3/sam3/model.gguf", "model_assets": [],
            "metrics": {"mask_iou": 0.98, "json0/cuda/total/p50_ms": 10.0},
            "fingerprints": {"output": "a"},
            "fingerprint_policy": "stability_only",
        }
        baseline = self.report([row | {
            "metrics": {"mask_iou": 1.0, "json0/cuda/total/p50_ms": 10.0}}])
        current = self.report([row])
        self.assertEqual(VALIDATE_ALL.compare_report(
            current, baseline, 5.0, 3.0), [])

    def test_exact_policy_rejects_changed_output(self):
        base_row = {
            "key": "depth/depth/model.gguf", "model_assets": [],
            "metrics": {"inference_ms": 10.0},
            "fingerprints": {"output": "a"}, "fingerprint_policy": "exact",
        }
        current_row = base_row | {"fingerprints": {"output": "b"}}
        failures = VALIDATE_ALL.compare_report(
            self.report([current_row]), self.report([base_row]), 5.0, 3.0)
        self.assertEqual(failures,
                         ["depth/depth/model.gguf: output fingerprint changed"])

    def test_explicitly_failed_baseline_is_rejected(self):
        row = {
            "key": "depth/depth/model.gguf", "model_assets": [],
            "metrics": {"inference_ms": 10.0},
            "fingerprints": {"output": "a"}, "fingerprint_policy": "exact",
        }
        failures = VALIDATE_ALL.compare_report(
            self.report([row]), self.report([row | {"status": "fail"}]),
            5.0, 3.0)
        self.assertEqual(
            failures, ["depth/depth/model.gguf: baseline status is fail"])

    def test_nested_timing_report_is_flattened(self):
        metrics, _ = VALIDATE_ALL.flatten_json(
            {"graph_ms": {"p50": 12.0, "p90": 14.0}})
        self.assertEqual(metrics,
                         {"graph_ms/p50": 12.0, "graph_ms/p90": 14.0})

    def test_catalog_parser_requires_every_selected_task(self):
        row = {
            "task": "depth", "relative_path": "da3_models/model.gguf",
            "url": "https://example.invalid/model.gguf",
            "sha256": "a" * 64, "size_bytes": 123,
        }
        assets = VALIDATE_ALL.parse_catalog_output(
            json.dumps(row) + "\n", {"depth"})
        self.assertEqual(len(assets), 1)
        self.assertEqual(str(assets[0].relative_path),
                         "da3_models/model.gguf")
        with self.assertRaisesRegex(ValueError, "no assets for tasks: yolo"):
            VALIDATE_ALL.parse_catalog_output(
                json.dumps(row) + "\n", {"depth", "yolo"})

    def test_catalog_parser_rejects_path_traversal(self):
        row = {
            "task": "depth", "relative_path": "../model.gguf",
            "url": "https://example.invalid/model.gguf",
            "sha256": "a" * 64, "size_bytes": 0,
        }
        with self.assertRaisesRegex(ValueError, "unsafe model catalog"):
            VALIDATE_ALL.parse_catalog_output(
                json.dumps(row) + "\n", {"depth"})

    def test_catalog_parser_deduplicates_shared_model_destination(self):
        rows = [
            {
                "task": "rmbg", "relative_path": "rmbg_models/rmbg.gguf",
                "url": "https://example.invalid/rmbg.gguf",
                "sha256": "a" * 64, "size_bytes": 0,
            },
            {
                "task": "rmbg",
                "relative_path": "rmbg_models/rmbg.gguf",
                "url": "https://example.invalid/rmbg.gguf",
                "sha256": "a" * 64, "size_bytes": 0,
            },
        ]
        assets = VALIDATE_ALL.parse_catalog_output(
            "\n".join(json.dumps(row) for row in rows), {"rmbg"})
        self.assertEqual(
            {str(asset.relative_path) for asset in assets},
            {"rmbg_models/rmbg.gguf"})

    def test_missing_model_is_downloaded_to_task_cache_and_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "published.gguf"
            payload = b"official model bytes"
            source.write_bytes(payload)
            asset = VALIDATE_ALL.ModelAsset(
                "depth", VALIDATE_ALL.PurePosixPath("da3_models/model.gguf"),
                source.as_uri(), hashlib.sha256(payload).hexdigest(),
                len(payload))
            summary, failures = VALIDATE_ALL.ensure_model_assets(
                [asset], root / "extract", offline=False, timeout=5,
                retries=1, jobs=1)
            self.assertEqual(failures, [])
            self.assertEqual(summary["downloaded"], 1)
            self.assertEqual(
                (root / "extract/da3_models/model.gguf").read_bytes(), payload)
            self.assertEqual(list((root / "extract/da3_models").glob("*.part")),
                             [])

    def test_corrupt_model_is_replaced_but_offline_mode_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "published.gguf"
            payload = b"verified replacement"
            source.write_bytes(payload)
            destination = root / "extract/yolo_models/model.gguf"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"corrupt")
            asset = VALIDATE_ALL.ModelAsset(
                "yolo", VALIDATE_ALL.PurePosixPath("yolo_models/model.gguf"),
                source.as_uri(), hashlib.sha256(payload).hexdigest(),
                len(payload))
            offline_summary, offline_failures = VALIDATE_ALL.ensure_model_assets(
                [asset], root / "extract", offline=True, timeout=5,
                retries=1, jobs=1)
            self.assertEqual(len(offline_failures), 1)
            self.assertEqual(offline_summary["unavailable"],
                             ["yolo_models/model.gguf"])
            summary, failures = VALIDATE_ALL.ensure_model_assets(
                [asset], root / "extract", offline=False, timeout=5,
                retries=1, jobs=1)
            self.assertEqual(failures, [])
            self.assertEqual(summary["downloaded"], 1)
            self.assertEqual(destination.read_bytes(), payload)

    def test_download_failure_is_returned_as_preflight_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            asset = VALIDATE_ALL.ModelAsset(
                "depth", VALIDATE_ALL.PurePosixPath("da3_models/missing.gguf"),
                (root / "does-not-exist.gguf").as_uri(), "a" * 64, 1)
            summary, failures = VALIDATE_ALL.ensure_model_assets(
                [asset], root / "extract", offline=False, timeout=1,
                retries=1, jobs=1)
            self.assertEqual(summary["downloaded"], 0)
            self.assertEqual(len(failures), 1)
            self.assertIn("download failed", failures[0])
            self.assertEqual(summary["unavailable"],
                             ["da3_models/missing.gguf"])

    def test_allow_incomplete_filters_scenarios_using_unavailable_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "extract"
            available = root / "demo_models/available.gguf"
            unavailable = root / "demo_models/unavailable.gguf"
            available.parent.mkdir(parents=True)
            available.write_bytes(b"available")
            unavailable.write_bytes(b"corrupt")
            common = {
                "task": "demo", "command": ("probe",), "env": {},
                "accuracy_gate": "demo", "metric_parser": "generic",
                "fingerprint_policy": "exact", "require_fingerprint": True,
                "report_path": root / "report.json",
            }
            specs = [
                VALIDATE_ALL.RunSpec(
                    scenario_id="available", model_id="available",
                    model_paths=(available,), **common),
                VALIDATE_ALL.RunSpec(
                    scenario_id="unavailable", model_id="unavailable",
                    model_paths=(unavailable,), **common),
            ]
            runnable, skipped = VALIDATE_ALL.filter_unavailable_specs(
                specs, root, ["demo_models/unavailable.gguf"])
            self.assertEqual([spec.scenario_id for spec in runnable],
                             ["available"])
            self.assertEqual(len(skipped), 1)
            self.assertIn("demo_models/unavailable.gguf", skipped[0])

    def test_allow_incomplete_comparison_ignores_missing_model_rows(self):
        row = {
            "key": "demo/model/present", "model_assets": [],
            "metrics": {"inference_ms": 10.0},
            "fingerprints": {"output": "a"},
            "fingerprint_policy": "exact", "status": "pass",
        }
        missing = row | {"key": "demo/model/missing"}
        current = self.report([row])
        baseline = self.report([row, missing])
        self.assertEqual(VALIDATE_ALL.compare_report(
            current, baseline, 5.0, 3.0, allow_incomplete=True), [])
        self.assertEqual(VALIDATE_ALL.compare_report(
            current, baseline, 5.0, 3.0),
            ["missing candidate row: demo/model/missing"])

    def test_expand_flags_uncovered_model_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = root / "assets"
            models = assets / "demo_models"
            models.mkdir(parents=True)
            (models / "covered.gguf").write_bytes(b"covered")
            (models / "new.gguf").write_bytes(b"new")
            manifest = {
                "schema": 1,
                "tasks": ["demo"],
                "scenarios": [{
                    "id": "demo",
                    "task": "demo",
                    "model_glob": "demo_models/covered.gguf",
                    "covered_globs": ["demo_models/*.gguf"],
                    "for_each_model": True,
                    "executable": "binary:probe",
                    "env": {},
                    "accuracy_gate": "demo",
                }],
            }
            args = SimpleNamespace(
                tasks="", assets=assets, repo=root, build=root / "build",
                backend="cpu", image="image", image2="image2",
                face_image="face", sam_image="sam", yolo_image="yolo",
                threads=1, warmup_runs=1,
                inference_runs=1, trellis_steps=1,
            )
            specs, diagnostics = VALIDATE_ALL.expand_specs(
                manifest, args, root / "reports")
            self.assertEqual(len(specs), 1)
            self.assertEqual(diagnostics["uncovered_assets"],
                             ["demo: demo_models/new.gguf"])

    def test_local_only_scenario_is_optional_when_bundle_is_unpublished(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = root / "assets"
            models = assets / "demo_models"
            models.mkdir(parents=True)
            (models / "published_q8.gguf").write_bytes(b"published")
            manifest = {
                "schema": 1,
                "tasks": ["demo"],
                "scenarios": [
                    {
                        "id": "demo-local-f16", "task": "demo",
                        "local_only": True,
                        "covered_globs": ["demo_models/unpublished_f16.gguf"],
                        "executable": "binary:probe", "env": {},
                        "accuracy_gate": "developer-local bundle",
                    },
                    {
                        "id": "demo-published-q8", "task": "demo",
                        "covered_globs": ["demo_models/published_q8.gguf"],
                        "executable": "binary:probe", "env": {},
                        "accuracy_gate": "published bundle",
                    },
                ],
            }
            args = SimpleNamespace(
                tasks="", assets=assets, repo=root, build=root / "build",
                backend="cpu", image="image", image2="image2",
                face_image="face", sam_image="sam", yolo_image="yolo",
                threads=1, warmup_runs=1,
                inference_runs=1, trellis_steps=1,
            )
            specs, diagnostics = VALIDATE_ALL.expand_specs(
                manifest, args, root / "reports")
            self.assertEqual([spec.scenario_id for spec in specs],
                             ["demo-published-q8"])
            self.assertEqual(diagnostics["missing_assets"], [])
            self.assertEqual(diagnostics["uncovered_assets"], [])
            self.assertEqual(len(diagnostics["local_only_scenarios"]), 1)

    def test_complete_manifest_expands_installed_yolo_text_towers(self):
        assets = Path.home() / "cloudViewer_data" / "extract"
        models = assets / "yolo_models"
        if not models.is_dir():
            self.skipTest("local AICore assets are not installed")
        manifest = VALIDATE_ALL.load_manifest(
            MODULE_PATH.with_name("validation_manifest.json"))
        args = SimpleNamespace(
            tasks="yolo", assets=assets, repo=MODULE_PATH.parents[3],
            build=MODULE_PATH.parents[3] / "build_app", backend="cuda",
            image="image", image2="image2", face_image="face",
            sam_image="sam", yolo_image="yolo", threads=1,
            warmup_runs=1, inference_runs=1, trellis_steps=1,
        )
        specs, diagnostics = VALIDATE_ALL.expand_specs(
            manifest, args, args.build / "Testing")
        expected = {
            path.name for path in models.glob("*.gguf")
            if path.name.startswith(("clip-", "mobileclip", "mclip-"))
        }
        expanded = {
            spec.model_id for spec in specs
            if spec.scenario_id.startswith("yolo-text-")
        }
        self.assertEqual(expanded, expected)
        self.assertEqual(diagnostics["uncovered_assets"], [])


if __name__ == "__main__":
    unittest.main()
