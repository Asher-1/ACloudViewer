#!/usr/bin/env python3

import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
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
            "revision": "", "results": rows,
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

    def test_runtime_library_path_preserves_configured_qt_directory(self):
        env = {"AICORE_QT_LIBRARY_DIR": "/opt/qt/lib"}
        VALIDATE_ALL.add_runtime_library_path(env, Path("/tmp/aicore-build"))
        variable = ("PATH" if VALIDATE_ALL.os.name == "nt" else
                    "DYLD_LIBRARY_PATH" if VALIDATE_ALL.sys.platform == "darwin"
                    else "LD_LIBRARY_PATH")
        self.assertEqual(env[variable].split(VALIDATE_ALL.os.pathsep),
                         ["/tmp/aicore-build/bin", "/opt/qt/lib"])

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

    def test_missing_input_fixture_is_downloaded_and_archive_member_extracted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "fixture.zip"
            payload = b"pinned input image"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("images/input.jpg", payload)
            archive_bytes = source.read_bytes()
            asset = VALIDATE_ALL.InputAsset(
                "fixture", frozenset({"depth"}),
                VALIDATE_ALL.PurePosixPath("fixture.zip"), source.as_uri(),
                hashlib.sha256(archive_bytes).hexdigest(), len(archive_bytes),
                True, ((VALIDATE_ALL.PurePosixPath("images/input.jpg"),
                        hashlib.sha256(payload).hexdigest(), len(payload)),))
            summary, failures = VALIDATE_ALL.ensure_input_assets(
                [asset], root / "extract", offline=False, timeout=5,
                retries=1, jobs=1)
            self.assertEqual(failures, [])
            self.assertEqual(summary["downloaded"], 1)
            self.assertEqual((root / "extract/images/input.jpg").read_bytes(),
                             payload)

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

    # ------------------------------------------------------------------
    # Tiering: default light subset for tasks that declare one, --full
    # restores the complete matrix; preflight and coverage audit shrink
    # with the tier.
    # ------------------------------------------------------------------

    @staticmethod
    def tier_manifest():
        return {
            "schema": 1,
            "tasks": ["demo", "plain"],
            "scenarios": [
                {
                    "id": "demo", "task": "demo",
                    "model_glob": "demo_models/*.gguf",
                    "light_globs": ["demo_models/tiny_q8.gguf"],
                    "for_each_model": True,
                    "executable": "binary:probe", "env": {},
                    "accuracy_gate": "demo",
                },
                {
                    "id": "plain", "task": "plain",
                    "model_glob": "plain_models/ok.gguf",
                    "for_each_model": True,
                    "executable": "binary:probe", "env": {},
                    "accuracy_gate": "plain",
                },
            ],
        }

    @staticmethod
    def tier_args(assets, root, full):
        return SimpleNamespace(
            tasks="", assets=assets, repo=root, build=root / "build",
            backend="cpu", image="image", image2="image2",
            face_image="face", sam_image="sam", yolo_image="yolo",
            threads=1, warmup_runs=1, inference_runs=1, trellis_steps=1,
            full=full)

    def test_light_tier_runs_only_declared_light_subset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = root / "assets"
            (assets / "demo_models").mkdir(parents=True)
            (assets / "plain_models").mkdir(parents=True)
            (assets / "demo_models/tiny_q8.gguf").write_bytes(b"tiny")
            (assets / "demo_models/heavy_f16.gguf").write_bytes(b"heavy")
            (assets / "plain_models/ok.gguf").write_bytes(b"ok")
            manifest = self.tier_manifest()
            specs, diagnostics = VALIDATE_ALL.expand_specs(
                manifest, self.tier_args(assets, root, full=False),
                root / "reports")
            self.assertEqual([spec.key for spec in specs],
                             ["demo/demo/tiny_q8.gguf",
                              "plain/plain/ok.gguf"])
            # The heavy model is deliberately not executed in the light
            # tier, so it must not be reported as an uncovered asset.
            self.assertEqual(diagnostics["uncovered_assets"], [])
            specs, diagnostics = VALIDATE_ALL.expand_specs(
                manifest, self.tier_args(assets, root, full=True),
                root / "reports")
            self.assertEqual([spec.key for spec in specs],
                             ["demo/demo/heavy_f16.gguf",
                              "demo/demo/tiny_q8.gguf",
                              "plain/plain/ok.gguf"])
            self.assertEqual(diagnostics["uncovered_assets"], [])

    def test_light_tier_reports_missing_light_models_like_any_asset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = root / "assets"
            (assets / "demo_models").mkdir(parents=True)
            manifest = self.tier_manifest()
            specs, diagnostics = VALIDATE_ALL.expand_specs(
                manifest, self.tier_args(assets, root, full=False),
                root / "reports")
            self.assertEqual(
                diagnostics["missing_assets"],
                ["demo: demo_models/tiny_q8.gguf",
                 "plain: plain_models/ok.gguf"])
            self.assertEqual(specs, [])

    def test_manifest_model_patterns_shrink_to_light_subset(self):
        manifest = self.tier_manifest()
        tiered = VALIDATE_ALL.tiered_tasks(manifest)
        self.assertEqual(tiered, {"demo"})
        selected = {"demo", "plain"}
        light = VALIDATE_ALL.manifest_model_patterns(
            manifest, selected, full=False, tiered=tiered)
        self.assertEqual(light,
                         {"demo_models/tiny_q8.gguf", "plain_models/ok.gguf"})
        complete = VALIDATE_ALL.manifest_model_patterns(
            manifest, selected, full=True, tiered=tiered)
        self.assertIn("demo_models/*.gguf", complete)

    def test_light_catalog_preflight_keeps_only_light_rows(self):
        def row(task, relative):
            return {
                "task": task, "relative_path": relative,
                "url": f"https://example.invalid/{relative}",
                "sha256": "a" * 64, "size_bytes": 1,
            }

        output = "\n".join(json.dumps(item) for item in (
            row("demo", "demo_models/tiny_q8.gguf"),
            row("demo", "demo_models/heavy_f16.gguf"),
            row("plain", "plain_models/ok.gguf"),
        ))
        selected = {"demo", "plain"}
        patterns = VALIDATE_ALL.manifest_model_patterns(
            self.tier_manifest(), selected, full=False)
        light_rows = VALIDATE_ALL.parse_catalog_output(
            output, selected, patterns,
            task_light_patterns={"demo": {"demo_models/tiny_q8.gguf"}})
        self.assertEqual(
            {str(asset.relative_path) for asset in light_rows},
            {"demo_models/tiny_q8.gguf", "plain_models/ok.gguf"})
        complete_rows = VALIDATE_ALL.parse_catalog_output(
            output, selected,
            VALIDATE_ALL.manifest_model_patterns(
                self.tier_manifest(), selected, full=True))
        self.assertEqual(len(complete_rows), 3)

    # ------------------------------------------------------------------
    # Per-task probe-output cleanup: intermediate files are deleted as
    # soon as a task's rows are summarized.
    # ------------------------------------------------------------------

    def test_sweep_task_probe_outputs_removes_only_that_task(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            probes = root / "probes"
            baseline_probes = root / "baseline-probes"
            probes.mkdir()
            baseline_probes.mkdir()
            (probes / "demo-a.json").write_text("{}")
            (probes / "demo-b.json").write_text("{}")
            (probes / "plain-a.json").write_text("{}")
            (baseline_probes / "demo-a.json").write_text("{}")
            removed = VALIDATE_ALL.sweep_task_probe_outputs("demo", root)
            self.assertEqual(removed, 3)
            self.assertTrue((probes / "plain-a.json").exists())
            self.assertEqual(list(probes.glob("demo-*.json")), [])
            self.assertEqual(list(baseline_probes.glob("demo-*.json")), [])

    def test_sweep_task_probe_outputs_tolerates_missing_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(
                VALIDATE_ALL.sweep_task_probe_outputs("demo", Path(temp)), 0)

    # ------------------------------------------------------------------
    # Per-task cleanup: probe outputs and consumed model-cache files are
    # released only when explicitly enabled and only after the task's last
    # spec; shared models wait for their last referencing task.
    # ------------------------------------------------------------------

    @staticmethod
    def cleanup_args(root, clean_probe_outputs=False, clean_model_cache=False):
        return SimpleNamespace(
            output=SimpleNamespace(parent=root), assets=root / "assets",
            clean_probe_outputs=clean_probe_outputs,
            clean_model_cache=clean_model_cache)

    def test_task_outputs_are_kept_unless_cleanup_is_explicitly_enabled(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            probes = root / "probes"
            probes.mkdir()
            (probes / "demo-a.json").write_text("{}")
            model = root / "assets/demo_models/m.gguf"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"m")
            specs_left = {"demo": 1}
            consumed = {"demo": {model}}
            holders = {model: {"demo"}}
            # Default policy: everything survives the run for inspection.
            self.assertEqual(VALIDATE_ALL.release_finished_task_outputs(
                "demo", specs_left, consumed, holders,
                self.cleanup_args(root)), (0, []))
            self.assertTrue((probes / "demo-a.json").exists())
            self.assertTrue(model.exists())
            # Explicit opt-in (--clean-probe-outputs --clean-model-cache):
            # the task's last summarized spec releases both.
            probe_removed, model_removed = VALIDATE_ALL.release_finished_task_outputs(
                "demo", specs_left, consumed, holders,
                self.cleanup_args(root, clean_probe_outputs=True,
                                  clean_model_cache=True))
            self.assertEqual(probe_removed, 1)
            self.assertEqual(model_removed, ["demo_models/m.gguf"])
            self.assertFalse((probes / "demo-a.json").exists())
            self.assertFalse(model.exists())

    def test_task_output_cleanup_waits_for_the_task_last_spec(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            probes = root / "probes"
            probes.mkdir()
            (probes / "demo-a.json").write_text("{}")
            (probes / "demo-b.json").write_text("{}")
            specs_left = {"demo": 2}
            args = self.cleanup_args(root, clean_probe_outputs=True)
            self.assertEqual(VALIDATE_ALL.release_finished_task_outputs(
                "demo", specs_left, {}, {}, args), (0, []))
            self.assertEqual(len(list(probes.glob("demo-*.json"))), 2)
            self.assertEqual(VALIDATE_ALL.release_finished_task_outputs(
                "demo", specs_left, {}, {}, args), (2, []))
            self.assertEqual(list(probes.glob("demo-*.json")), [])
            # Unknown tasks (e.g. filtered out by --allow-incomplete) are inert.
            self.assertEqual(VALIDATE_ALL.release_finished_task_outputs(
                "ghost", specs_left, {}, {}, args), (0, []))

    def test_model_cache_prune_keeps_shared_models_until_last_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shared = root / "assets/shared_models/dep.gguf"
            shared.parent.mkdir(parents=True)
            shared.write_bytes(b"dep")
            holders = {shared: {"demo", "other"}}
            consumed = {"demo": {shared}, "other": {shared}}
            args = self.cleanup_args(root, clean_model_cache=True)
            self.assertEqual(VALIDATE_ALL.release_finished_task_outputs(
                "demo", {"demo": 1, "other": 1}, consumed, holders,
                args), (0, []))
            self.assertTrue(shared.exists())
            self.assertEqual(VALIDATE_ALL.release_finished_task_outputs(
                "other", {"demo": 1, "other": 1}, consumed, holders,
                args), (0, ["shared_models/dep.gguf"]))
            self.assertFalse(shared.exists())

    def test_model_cache_prune_only_touches_conforming_consumed_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = root / "assets"
            conforming = assets / "demo_models/m.gguf"
            conforming.parent.mkdir(parents=True)
            conforming.write_bytes(b"m")
            decoys = {
                assets / "other_dir/a.gguf": b"x",      # not *_models
                assets / "demo_models/notes.txt": b"x",  # not .gguf
                root / "outside.gguf": b"x",             # outside assets root
            }
            for path, payload in decoys.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            consumed_paths = {conforming, *decoys}
            holders = {path: {"demo"} for path in consumed_paths}
            removed = VALIDATE_ALL.prune_consumed_model_cache(
                "demo", consumed_paths, holders, assets)
            self.assertEqual(removed, ["demo_models/m.gguf"])
            self.assertFalse(conforming.exists())
            for path in decoys:
                self.assertTrue(path.exists())

    # ------------------------------------------------------------------
    # Explicit --models selection: narrows the run to named models (or
    # bundle scenario ids), bypasses the tier, and shrinks the audit and
    # the preflight to the selection.
    # ------------------------------------------------------------------

    def test_model_id_matches_names_stems_and_globs(self):
        matches = VALIDATE_ALL.model_id_matches
        self.assertTrue(matches("sam3-f16.gguf", ["sam3-f16"]))
        self.assertTrue(matches("sam3-f16.gguf", ["sam3-f16.gguf"]))
        self.assertTrue(matches("sam3-f16.gguf", ["sam3*"]))
        self.assertFalse(matches("sam2_hiera_tiny.gguf", ["sam3*"]))
        self.assertTrue(matches("trellis-coarse-q8", ["trellis-coarse-q8"]))
        self.assertTrue(matches("trellis-coarse-q8", ["*-q8"]))

    def test_models_filter_selects_named_models_bypassing_tier(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = root / "assets"
            (assets / "demo_models").mkdir(parents=True)
            (assets / "plain_models").mkdir(parents=True)
            (assets / "demo_models/tiny_q8.gguf").write_bytes(b"t")
            (assets / "demo_models/heavy_f16.gguf").write_bytes(b"h")
            (assets / "plain_models/ok.gguf").write_bytes(b"o")
            manifest = self.tier_manifest()
            args = self.tier_args(assets, root, full=False)
            args.models = "heavy_f16"
            specs, diagnostics = VALIDATE_ALL.expand_specs(
                manifest, args, root / "reports")
            # The explicit selection bypasses the tier (heavy model runs) and
            # the unselected plain task drops out entirely.
            self.assertEqual([spec.key for spec in specs],
                             ["demo/demo/heavy_f16.gguf"])
            self.assertEqual(diagnostics["uncovered_assets"], [])
            self.assertEqual(diagnostics["missing_assets"], [])

    def test_models_filter_rejects_patterns_that_match_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = root / "assets"
            (assets / "demo_models").mkdir(parents=True)
            (assets / "demo_models/tiny_q8.gguf").write_bytes(b"t")
            args = self.tier_args(assets, root, full=False)
            args.models = "tiny_q8,does-not-exist"
            with self.assertRaisesRegex(ValueError, "does-not-exist"):
                VALIDATE_ALL.expand_specs(
                    self.tier_manifest(), args, root / "reports")

    def test_models_filter_runs_matching_bundle_scenario(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = root / "assets"
            (assets / "bundle_models").mkdir(parents=True)
            for name in ("dino_q8.gguf", "dino_f16.gguf"):
                (assets / f"bundle_models/{name}").write_bytes(b"m")
            manifest = {
                "schema": 1,
                "tasks": ["bundle"],
                "scenarios": [
                    {
                        "id": "bundle-coarse-q8", "task": "bundle",
                        "covered_globs": ["bundle_models/dino_q8.gguf"],
                        "executable": "binary:probe", "env": {},
                        "accuracy_gate": "bundle",
                    },
                    {
                        "id": "bundle-coarse-f16", "task": "bundle",
                        "covered_globs": ["bundle_models/dino_f16.gguf"],
                        "executable": "binary:probe", "env": {},
                        "accuracy_gate": "bundle",
                    },
                ],
            }
            args = self.tier_args(assets, root, full=False)
            args.models = "bundle-coarse-q8"
            specs, _ = VALIDATE_ALL.expand_specs(
                manifest, args, root / "reports")
            self.assertEqual([spec.key for spec in specs],
                             ["bundle/bundle-coarse-q8/bundle-coarse-q8"])

    def test_catalog_selection_prefetches_named_models_and_dependencies(self):
        def row(task, relative):
            return {
                "task": task, "relative_path": relative,
                "url": f"https://example.invalid/{relative}",
                "sha256": "a" * 64, "size_bytes": 1,
            }

        output = "\n".join(json.dumps(item) for item in (
            row("demo", "demo_models/tiny_q8.gguf"),
            row("demo", "demo_models/heavy_f16.gguf"),
            row("plain", "plain_models/ok.gguf"),
        ))
        selection = {
            "demo": {"*/heavy_f16", "*/heavy_f16.gguf"},
            # plain has no matching row: its catalog rows drop out entirely
            "plain": {"*/does-not-match"},
        }
        assets = VALIDATE_ALL.parse_catalog_output(
            output, {"demo", "plain"}, task_selection=selection)
        self.assertEqual(
            {str(asset.relative_path) for asset in assets},
            {"demo_models/heavy_f16.gguf"})
        with self.assertRaisesRegex(ValueError, "matched no catalog rows"):
            VALIDATE_ALL.parse_catalog_output(
                output, {"demo", "plain"},
                task_selection={
                    "demo": {"*/nothing"}, "plain": {"*/nothing"}})

    def test_light_yolo_matrix_receives_light_model_globs(self):
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
            warmup_runs=1, inference_runs=1, trellis_steps=1, full=False)
        specs, _ = VALIDATE_ALL.expand_specs(
            manifest, args, args.build / "Testing")
        matrix = [spec for spec in specs if spec.scenario_id == "yolo"]
        self.assertEqual(len(matrix), 1)
        command = " ".join(matrix[0].command)
        self.assertIn("--model-globs", command)
        self.assertIn("yolo26n-q8_0.gguf", command)
        self.assertNotIn("yolo26x", command)  # heavy scales stay out
        # full tier passes the complete covered glob instead
        args.full = True
        specs, _ = VALIDATE_ALL.expand_specs(
            manifest, args, args.build / "Testing")
        matrix = [spec for spec in specs if spec.scenario_id == "yolo"]
        self.assertIn("yolo*.gguf", " ".join(matrix[0].command))

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

    # ------------------------------------------------------------------
    # VRAM gate: estimate, fail-open, skip row synthesis, reporting
    # ------------------------------------------------------------------

    @staticmethod
    def make_spec(tmp, **overrides):
        model = Path(tmp) / "demo_models/model.gguf"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"x" * (16 * 1024 * 1024))  # 16 MiB
        fields = {
            "scenario_id": "demo", "task": "demo", "model_id": "model.gguf",
            "model_paths": (model,), "command": ("probe",), "env": {},
            "accuracy_gate": "demo", "metric_parser": "generic",
            "fingerprint_policy": "exact", "require_fingerprint": True,
            "report_path": Path(tmp) / "probe.json",
        }
        fields.update(overrides)
        return VALIDATE_ALL.RunSpec(**fields)

    @staticmethod
    def gate_args(backend="cuda", overhead=1024.0):
        return SimpleNamespace(backend=backend, vram_overhead_mib=overhead)

    def test_vram_estimate_prefers_declared_value_over_model_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            declared = self.make_spec(temp, vram_estimate_mib=8192.0)
            self.assertEqual(
                VALIDATE_ALL.estimate_spec_vram_mib(declared, 1024.0), 8192.0)
            derived = self.make_spec(
                temp, vram_overhead_mib=512.0)
            self.assertAlmostEqual(
                VALIDATE_ALL.estimate_spec_vram_mib(derived, 1024.0),
                16.0 + 512.0)
            # Scenario without an override falls back to the runner default.
            defaulted = self.make_spec(temp)
            self.assertAlmostEqual(
                VALIDATE_ALL.estimate_spec_vram_mib(defaulted, 2048.0),
                16.0 + 2048.0)

    def test_vram_gate_fails_open_without_nvidia_smi(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = self.make_spec(temp)
            original = VALIDATE_ALL.command_output
            VALIDATE_ALL.command_output = lambda command: ""
            try:
                self.assertIsNone(
                    VALIDATE_ALL.vram_gate_decision(spec, self.gate_args()))
            finally:
                VALIDATE_ALL.command_output = original

    def test_vram_gate_skips_only_when_need_exceeds_free(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = self.make_spec(temp)
            original = VALIDATE_ALL.command_output
            VALIDATE_ALL.command_output = lambda command: "2048\n"  # MiB
            try:
                gate = VALIDATE_ALL.vram_gate_decision(
                    spec, self.gate_args(overhead=256.0))
                self.assertIsNone(gate)  # 16 + 256 fits in 2048
                gate = VALIDATE_ALL.vram_gate_decision(
                    spec, self.gate_args(overhead=512.0))
                self.assertIsNone(gate)  # 16 + 512 still fits
            finally:
                VALIDATE_ALL.command_output = original
            VALIDATE_ALL.command_output = lambda command: "1024\n"
            try:
                gate = VALIDATE_ALL.vram_gate_decision(
                    spec, self.gate_args(overhead=4096.0))
                self.assertIsNotNone(gate)
                self.assertAlmostEqual(gate["needed_mib"], 16.0 + 4096.0)
                self.assertAlmostEqual(gate["free_mib"], 1024.0)
            finally:
                VALIDATE_ALL.command_output = original

    def test_vram_gate_never_runs_on_cpu_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = self.make_spec(temp)
            original = VALIDATE_ALL.command_output
            VALIDATE_ALL.command_output = lambda command: "64\n"
            try:
                self.assertIsNone(VALIDATE_ALL.vram_gate_decision(
                    spec, self.gate_args(backend="cpu")))
            finally:
                VALIDATE_ALL.command_output = original

    def test_vram_skipped_row_reports_need_free_and_keeps_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = self.make_spec(temp)
            row = VALIDATE_ALL.vram_skipped_row(
                spec, {"needed_mib": 9000.0, "free_mib": 4000.0})
            self.assertEqual(row["status"], "vram_skipped")
            self.assertEqual(row["return_codes"], [])
            self.assertEqual(len(row["model_assets"]), 1)
            self.assertAlmostEqual(
                row["metrics"]["vram_needed_mib"], 9000.0)
            self.assertIn("9000", row["error_summary"])
            self.assertIn("4000", row["error_summary"])

    def test_compare_report_ignores_vram_skipped_rows(self):
        row = {
            "key": "demo/demo/model.gguf", "model_assets": [],
            "metrics": {}, "fingerprints": {},
            "fingerprint_policy": "exact", "status": "vram_skipped",
        }
        baseline = self.report([row | {
            "status": "pass", "metrics": {"inference_ms": 10.0}}])
        current = self.report([row])
        self.assertEqual(VALIDATE_ALL.compare_report(
            current, baseline, 5.0, 3.0), [])

    def test_error_summary_extracts_probe_root_cause_lines(self):
        tail = "\n".join([
            "ggml_cuda_init: found 1 CUDA devices",
            "failed to load image: /data/missing.jpg",
            "",
            "FAIL /path/test.cpp:47: depth != nullptr",
            "FAIL /path/test.cpp:48: height > 0 && width > 0",
            '{"suite":"aicore-validation","valid_fields":0}',
        ])
        summary = VALIDATE_ALL.extract_error_summary(tail)
        self.assertIn("failed to load image", summary)
        self.assertIn("FAIL /path/test.cpp:48", summary)
        self.assertNotIn("ggml_cuda_init", summary)
        self.assertNotIn("suite", summary)

    def test_markdown_reports_totals_and_vram_section(self):
        report = self.report([
            {"key": "demo/demo/ok.gguf", "task": "demo",
             "scenario_id": "demo", "model_id": "ok.gguf",
             "model_assets": [], "accuracy_gate": "demo",
             "fingerprint_policy": "exact", "status": "pass",
             "return_codes": [0], "metrics": {"e2e_ms": 1.0},
             "fingerprints": {}, "error_summary": ""},
            {"key": "demo/demo/bad.gguf", "task": "demo",
             "scenario_id": "demo", "model_id": "bad.gguf",
             "model_assets": [], "accuracy_gate": "demo",
             "fingerprint_policy": "exact", "status": "fail",
             "return_codes": [1], "metrics": {}, "fingerprints": {},
             "error_summary": "FAIL test.cpp:9: inference failed"},
            {"key": "demo/demo/huge.gguf", "task": "demo",
             "scenario_id": "demo", "model_id": "huge.gguf",
             "model_assets": [], "accuracy_gate": "demo",
             "fingerprint_policy": "exact", "status": "vram_skipped",
             "return_codes": [],
             "metrics": {"vram_needed_mib": 18432.0,
                         "vram_free_mib": 11000.0},
             "fingerprints": {},
             "error_summary": "skipped before launch: estimated VRAM need "
                              "18432 MiB exceeds 11000 MiB free (OOM guard)"},
        ])
        report["verdict"] = "INCOMPLETE"
        report["failures"] = ["demo/demo/bad.gguf: fail, return codes=[1]"]
        report["incomplete_reasons"] = ["vram insufficient: demo/demo/huge.gguf"]
        report["coverage"] = {"models_by_task": {"demo": 1}}
        text = VALIDATE_ALL.render_markdown(report)
        self.assertIn("- total scenarios: **3**", text)
        self.assertIn("PASS **1**", text)
        self.assertIn("FAIL **1**", text)
        self.assertIn("VRAM-SKIP **1**", text)
        self.assertIn("pass rate: **33.3%**", text)
        self.assertIn("fail rate: **33.3%**", text)
        self.assertIn("skip rate: **33.3%**", text)
        self.assertIn("## Failure details", text)
        self.assertIn("FAIL test.cpp:9: inference failed", text)
        self.assertIn("## VRAM-gated skips", text)
        self.assertIn("| 18432 | 11000 |", text)
        self.assertIn("vram insufficient: demo/demo/huge.gguf", text)
        self.assertIn("█", text)  # distribution bars present


if __name__ == "__main__":
    unittest.main()
