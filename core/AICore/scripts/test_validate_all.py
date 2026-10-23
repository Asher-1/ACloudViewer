#!/usr/bin/env python3

import importlib.util
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
import zipfile
import tarfile
import io
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MODULE_PATH = Path(__file__).resolve().with_name("validate_all.py")
SPEC = importlib.util.spec_from_file_location("aicore_validate_all", MODULE_PATH)
assert SPEC and SPEC.loader
VALIDATE_ALL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATE_ALL
SPEC.loader.exec_module(VALIDATE_ALL)


class ValidateAllTests(unittest.TestCase):
    def test_facedetect_bnkps_default_and_quantized_catalog_rows_have_consumers(self):
        manifest = VALIDATE_ALL.load_manifest(
            MODULE_PATH.with_name("validation_manifest.json"))
        names = ["scrfd_10g_bnkps.gguf", "scrfd_10g_bnkps-f32.gguf",
                 "scrfd_10g_bnkps-q8_0.gguf"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            models = assets / "facedetect_models"
            models.mkdir(parents=True)
            for name in names:
                (models / name).write_bytes(b"fixture")
            # Keep the embedding and landmark consumers present as well: they
            # must not accidentally claim a detection-only default model.
            (models / "buffalo_l.gguf").write_bytes(b"fixture")
            (models / "landmarks-2d106-1k3d68.gguf").write_bytes(b"fixture")
            args = self.relation_args(assets, root, full=True)
            args.tasks = "facedetect"
            specs, diagnostics = VALIDATE_ALL.expand_specs(
                manifest, args, root / "reports")
        self.assertEqual(diagnostics["missing_assets"], [])
        self.assertEqual(diagnostics["uncovered_assets"], [])
        for name in names:
            consumers = [s for s in specs if s.model_id == name]
            self.assertEqual(len(consumers), 1, name)
            self.assertEqual(consumers[0].scenario_id, "facedetect-bnkps")
            self.assertEqual(Path(consumers[0].command[0]).name,
                             "test_facedetect_detect_only")

    def test_execution_artifacts_identify_commands_runtime_and_build_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "build"
            binary = build / "bin"
            binary.mkdir(parents=True)
            expected = [binary / "test_probe", binary / "AICore.dll",
                        binary / "libggml-cuda.so", binary / "libggml.1.dylib",
                        root / "probe.py", build / "CMakeCache.txt"]
            for path in expected:
                path.write_bytes(b"original executable or configuration")
            model = binary / "fixture.gguf"
            model.write_bytes(b"model is recorded separately")
            spec = SimpleNamespace(command=(str(expected[0]), str(expected[4]), str(model)))
            artifacts = VALIDATE_ALL.execution_artifacts([spec], build)
            self.assertTrue({str(p) for p in expected} <= artifacts.keys())
            self.assertNotIn(str(model), artifacts)
            for path in expected:
                self.assertEqual(artifacts[str(path)]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_execution_artifact_changes_fail_even_with_successful_probe_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            binary = build / "bin"
            binary.mkdir()
            library = binary / "libAICore.so"
            library.write_bytes(b"original")
            before = VALIDATE_ALL.execution_artifacts([], build)
            self.assertEqual(VALIDATE_ALL.execution_artifact_failures(before, before, "candidate"), [])
            library.write_bytes(b"rebuilt")
            after = VALIDATE_ALL.execution_artifacts([], build)
            self.assertEqual(len(VALIDATE_ALL.execution_artifact_failures(before, after, "candidate")), 1)
            library.unlink()
            after = VALIDATE_ALL.execution_artifacts([], build)
            self.assertEqual(len(VALIDATE_ALL.execution_artifact_failures(before, after, "baseline")), 1)
            extra = binary / "ggml-vulkan.dll"
            extra.write_bytes(b"new backend")
            after = VALIDATE_ALL.execution_artifacts([], build)
            self.assertEqual(len(VALIDATE_ALL.execution_artifact_failures(before, after, "candidate")), 2)

    def test_controlled_ab_rejects_different_cpu_variant_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            candidate = root / "candidate"
            for build in (baseline, candidate):
                (build / "bin").mkdir(parents=True)
            (baseline / "bin/libggml-cpu.so").touch()
            (baseline / "bin/libggml-cpu-haswell.so").touch()
            (candidate / "bin/libggml-cpu.so").touch()
            (baseline / "CMakeCache.txt").write_text(
                "AICore_CPU_ALL_VARIANTS:BOOL=ON\n"
                "GGML_CPU_ALL_VARIANTS:BOOL=ON\n")
            (candidate / "CMakeCache.txt").write_text(
                "GGML_CPU_ALL_VARIANTS:BOOL=OFF\n")
            baseline_profile = VALIDATE_ALL.cpu_backend_profile(baseline)
            candidate_profile = VALIDATE_ALL.cpu_backend_profile(candidate)
            failures = VALIDATE_ALL.cpu_backend_profile_failures(
                candidate_profile, baseline_profile)
            self.assertIn("CPU backend configuration mismatch", " ".join(failures))
            self.assertIn("candidate missing ['haswell']", " ".join(failures))

    def test_controlled_ab_accepts_matching_cpu_variant_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builds = [root / "baseline", root / "candidate"]
            for build in builds:
                (build / "bin").mkdir(parents=True)
                (build / "bin/libggml-cpu.so").touch()
                (build / "bin/libggml-cpu-haswell.so").touch()
            profiles = [VALIDATE_ALL.cpu_backend_profile(build) for build in builds]
            self.assertEqual(VALIDATE_ALL.cpu_backend_profile_failures(
                profiles[1], profiles[0]), [])

    def test_staged_candidate_reads_cpu_variants_from_llama_externalproject(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "build_app"
            candidate = root / "stride-fix/validation-v5"
            for build in (baseline, candidate):
                (build / "bin").mkdir(parents=True)
                (build / "bin/libggml-cpu.so").touch()
                (build / "bin/libggml-cpu-haswell.so").touch()
            (baseline / "CMakeCache.txt").write_text(
                "AICore_CPU_ALL_VARIANTS:BOOL=ON\n")
            external = root / "llama/src/ext_llama-build"
            external.mkdir(parents=True)
            (external / "CMakeCache.txt").write_text(
                "GGML_CPU_ALL_VARIANTS:BOOL=ON\n")

            baseline_profile = VALIDATE_ALL.cpu_backend_profile(baseline)
            candidate_profile = VALIDATE_ALL.cpu_backend_profile(candidate)
            self.assertEqual(candidate_profile["all_variants_settings"],
                             {"GGML_CPU_ALL_VARIANTS": "ON"})
            self.assertEqual(VALIDATE_ALL.cpu_backend_profile_failures(
                candidate_profile, baseline_profile), [])

    def test_report_comparison_accepts_equivalent_cpu_settings_with_different_keys(self):
        baseline = {"modules": ["generic", "haswell"],
                    "all_variants_settings": {
                        "AICore_CPU_ALL_VARIANTS": "ON",
                        "GGML_CPU_ALL_VARIANTS": "ON"},
                    "all_variants_consistent": True}
        candidate = {"modules": ["generic", "haswell"],
                     "all_variants_settings": {"GGML_CPU_ALL_VARIANTS": "TRUE"},
                     "all_variants_consistent": True}
        self.assertEqual(VALIDATE_ALL.cpu_backend_profile_failures(
            candidate, baseline), [])

    def test_controlled_ab_accepts_renamed_base_module_with_same_dispatch_score(self):
        baseline = {
            "modules": ["generic", "haswell", "x64"],
            "module_scores": {"generic": 64, "haswell": 64, "x64": 1},
            "best_dispatch_score": 64,
            "all_variants_settings": {"GGML_CPU_ALL_VARIANTS": "ON"},
            "all_variants_consistent": True,
        }
        candidate = {
            "modules": ["haswell", "x64"],
            "module_scores": {"haswell": 64, "x64": 1},
            "best_dispatch_score": 64,
            "all_variants_settings": {"GGML_CPU_ALL_VARIANTS": "ON"},
            "all_variants_consistent": True,
        }
        self.assertEqual(VALIDATE_ALL.cpu_backend_profile_failures(
            candidate, baseline), [])

    def test_controlled_ab_rejects_different_cpu_dispatch_scores(self):
        baseline = {
            "modules": ["generic", "haswell"],
            "best_dispatch_score": 64,
            "all_variants_settings": {"GGML_CPU_ALL_VARIANTS": "ON"},
            "all_variants_consistent": True,
        }
        candidate = {
            "modules": ["generic", "x64"],
            "best_dispatch_score": 1,
            "all_variants_settings": {"GGML_CPU_ALL_VARIANTS": "ON"},
            "all_variants_consistent": True,
        }
        failures = VALIDATE_ALL.cpu_backend_profile_failures(
            candidate, baseline)
        self.assertIn("CPU backend dispatch capability mismatch",
                      " ".join(failures))

    def test_manifest_identity_describes_consumed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            original = VALIDATE_ALL.DEFAULT_MANIFEST.read_bytes()
            path.write_bytes(original)
            identity = {}
            parsed = VALIDATE_ALL.load_manifest(path, identity=identity)
            path.write_bytes(original + b"\n")
            self.assertTrue(parsed["scenarios"])
            self.assertEqual(identity["bytes"], len(original))
            self.assertEqual(identity["sha256"], hashlib.sha256(original).hexdigest())
            self.assertNotEqual(identity, VALIDATE_ALL.file_identity(path))

    def test_published_ocr_table_scope_does_not_download_local_onnx_sources(self):
        manifest = VALIDATE_ALL.load_manifest(VALIDATE_ALL.DEFAULT_MANIFEST)
        published_ids = {s["id"] for s in manifest["scenarios"]
                         if s["task"] in {"ocr", "table"} and
                         not s.get("local_only")}
        inputs = VALIDATE_ALL.parse_input_assets(
            manifest, {"ocr", "table"}, published_ids)
        input_ids = {asset.asset_id for asset in inputs}
        self.assertIn("ppocr-testdata", input_ids)
        self.assertIn("table-fixtures-index", input_ids)
        self.assertIn("ocr-rec-reference", input_ids)
        self.assertFalse(any(a.startswith("ocr-onnx-") for a in input_ids))
        local_ids = {s["id"] for s in manifest["scenarios"]
                     if "run_ocr_onnx_probe.py" in str(s.get("args", []))}
        local_inputs = VALIDATE_ALL.parse_input_assets(
            manifest, {"ocr"}, local_ids)
        self.assertEqual(len([a for a in local_inputs
                             if a.asset_id.startswith("ocr-onnx-")]), 12)

    def test_ocr_local_onnx_rows_own_pinned_sources_and_numeric_gates(self):
        manifest = VALIDATE_ALL.load_manifest(VALIDATE_ALL.DEFAULT_MANIFEST)
        rows = [s for s in manifest["scenarios"]
                if "run_ocr_onnx_probe.py" in str(s.get("args", []))]
        self.assertEqual(len(rows), 54)
        sources = {a["id"]: a for a in manifest["input_assets"]}
        self.assertEqual(len([a for a in sources if a.startswith("ocr-onnx-")]), 12)
        for row in rows:
            self.assertTrue(row["local_only"])  # no invented published GGUF URLs
            self.assertEqual(row["executable"], "python")
            self.assertIn("{scenario_report}", row["args"])
            self.assertIn("{warmup_runs}", row["args"])
            source = sources[row["input_ids"][0]]
            self.assertEqual(row["args"][row["args"].index("--source-sha256") + 1],
                             source["sha256"])
            self.assertEqual(source["size_bytes"] > 0, True)
            self.assertTrue(source["url"].endswith(".onnx"))
            gate = VALIDATE_ALL.normalize_accuracy_gate(row)
            orientation = "orientation" in row["id"]
            if not orientation:
                self.assertIn("--measurement-profile", row["args"])
                self.assertEqual(row["args"][row["args"].index("--measurement-profile") + 1],
                                 "{profile}")
                self.assertIn("{gpu_idle_timeout}", row["args"])
                if row["id"].startswith("ppocr-cls-onnx-"):
                    self.assertIn("--include-rotated", row["args"])
                if "-det-onnx-" in row["id"]:
                    self.assertEqual(row["args"][row["args"].index("--max-side") + 1], "960")
            if orientation:
                source_path = row["args"][row["args"].index("--rec-source") + 1]
                rec_source = next(a for a in sources.values()
                                  if source_path.endswith(a["relative_path"]))
                self.assertIn(rec_source["id"], row["input_ids"])
                self.assertEqual(row["args"][row["args"].index("--rec-source-sha256") + 1],
                                 rec_source["sha256"])
                self.assertEqual(len(row["args"][row["args"].index("--dictionary-sha256") + 1]), 64)
            good = ({"orientation_rows": 10, "failures": 0, "thresholds": 2, "repeats": 2} if orientation
                    else {"max_diff": 0, "mean_diff": 0, "argmax_match": 1,
                          "stored_max_diff": 0, "stored_argmax_match": 1,
                          "cpu_max_diff": 0, "cpu_argmax_match": 1,
                          "accuracy_fixture_count": (1 + row["args"].count("--extra-image")) *
                          (2 if "--include-rotated" in row["args"] else 1) +
                          row["args"].count("--extra-crop")})
            if row["id"].startswith("ppocrv5-mobile-rec-onnx-"):
                self.assertIn("ocr-detection-labelled-50", row["input_ids"])
                self.assertEqual(good["accuracy_fixture_count"], 5)
            with tempfile.TemporaryDirectory() as temp:
                report = Path(temp) / "report.json"
                report.write_text(json.dumps(good if orientation else dict(good, fixtures=[good])))
                metrics, _ = VALIDATE_ALL.parse_output(
                    "OCR probe report", "generic", report,
                    VALIDATE_ALL.gate_metric_names(gate))
                self.assertEqual(VALIDATE_ALL.evaluate_accuracy_gate(gate, metrics), [])
            bad = dict(good)
            bad["failures" if orientation else "max_diff"] = 1
            self.assertTrue(VALIDATE_ALL.evaluate_accuracy_gate(gate, bad))
            self.assertTrue(VALIDATE_ALL.evaluate_accuracy_gate(gate, {}))

    def test_ocr_dataset_top_level_metrics_remain_unambiguous_with_nested_metrics(self):
        manifest = VALIDATE_ALL.load_manifest(VALIDATE_ALL.DEFAULT_MANIFEST)
        for row in manifest["scenarios"]:
            if row["task"] != "ocr" or "--ground-truth" not in row.get("args", []):
                continue
            gate = VALIDATE_ALL.normalize_accuracy_gate(row)
            payload = {"cer": .1, "wer": .2, "hmean": .9,
                       "cer_delta": 0, "wer_delta": 0, "hmean_delta": 0,
                       "e2e_cer_delta": 0, "e2e_wer_delta": 0,
                       "passes_source_gt_gate": 1,
                       "aggregates": {"recognition": {"cer": .1, "wer": .2},
                                      "source_recognition": {"cer": .1, "wer": .2},
                                      "detection": {"hmean": .9}}}
            with tempfile.TemporaryDirectory() as temp:
                report = Path(temp) / "report.json"
                report.write_text(json.dumps(payload))
                metrics, _ = VALIDATE_ALL.parse_output("", "generic", report,
                                                     VALIDATE_ALL.gate_metric_names(gate))
            self.assertEqual(VALIDATE_ALL.evaluate_accuracy_gate(gate, metrics), [])
            del metrics["json0/cer"]
            # Source/nested CER must not substitute for a missing native CER.
            self.assertTrue(VALIDATE_ALL.evaluate_accuracy_gate(gate, metrics))

    def test_ocr_source_pipeline_requires_complete_real_page_comparison(self):
        manifest = VALIDATE_ALL.load_manifest(VALIDATE_ALL.DEFAULT_MANIFEST)
        rows = [s for s in manifest["scenarios"]
                if "run_ocr_source_pipeline.py" in str(s.get("args", []))
                and "--ground-truth" not in s["args"]]
        self.assertEqual(len(rows), 12)
        for row in rows:
            self.assertTrue(row["local_only"])
            self.assertIn("{reference:xanylabeling}", row["args"])
            self.assertEqual(len(row["input_ids"]), 4)
            gate = VALIDATE_ALL.normalize_accuracy_gate(row)
            selected = "--selected-regions" in row["args"]
            good = {"json0/source_count": 10 if selected else 73,
                    "json0/missing_or_extra": 0,
                    "json0/text_mismatches": 0, "json0/quad_error": 0,
                    "json0/repeats": 2}
            self.assertEqual(VALIDATE_ALL.evaluate_accuracy_gate(gate, good), [])
            for count in ((9, 11, 73) if selected else (0, 10, 69)):
                bad_count = dict(good, **{"json0/source_count": count})
                self.assertTrue(VALIDATE_ALL.evaluate_accuracy_gate(gate, bad_count))
            for metric in ("missing_or_extra", "text_mismatches", "quad_error"):
                bad = dict(good)
                bad["json0/" + metric] = 3
                self.assertTrue(VALIDATE_ALL.evaluate_accuracy_gate(gate, bad))
            self.assertTrue(VALIDATE_ALL.evaluate_accuracy_gate(gate, {}))

    def test_ocr_labelled_gates_cover_all_presets_and_quants_without_using_source_as_gt(self):
        manifest = VALIDATE_ALL.load_manifest(VALIDATE_ALL.DEFAULT_MANIFEST)
        rows = [s for s in manifest["scenarios"] if "--ground-truth" in s.get("args", [])
                and "run_ocr_source_pipeline.py" in str(s["args"])]
        self.assertEqual(len(rows), 54)
        coverage = {(row["args"][row["args"].index("--stem") + 1],
                     row["args"][row["args"].index("--quant") + 1],
                     "receipts" if "-sroie-receipts-" in row["id"] else
                     "xfund" if "--evaluation" in row["args"] else "scenes") for row in rows}
        self.assertEqual(coverage, {(stem, quant, dataset)
                         for stem in ("ppocrv4-source", "ppocrv3-japan", "ppocrv5-mobile",
                                      "ppocrv6-tiny", "ppocrv6-small", "ppocrv6-medium")
                         for quant in ("f32", "f16", "q8_0")
                         for dataset in ("scenes", "xfund", "receipts")})
        assets = {a["id"]: a for a in manifest["input_assets"]}
        scenes = assets["ocr-detection-labelled-50"]
        self.assertTrue(scenes["archive"])
        self.assertEqual(len(scenes["extracted_files"]), 51)
        for row in rows:
            args = row["args"]
            self.assertTrue(row["local_only"])
            self.assertIn(args[args.index("--quant") + 1], ("f32", "f16", "q8_0"))
            label_path = args[args.index("--ground-truth") + 1]
            digest = args[args.index("--ground-truth-sha256") + 1]
            candidates = [item for asset_id in row["input_ids"]
                          for item in assets[asset_id].get("extracted_files", [assets[asset_id]])]
            if "-sroie-receipts-" in row["id"]:
                labels = sorted((item for item in candidates
                                 if item["relative_path"].endswith(".csv")),
                                key=lambda item: item["relative_path"])
                images = [item for item in candidates if item["relative_path"].endswith(".jpg")]
                self.assertEqual({Path(item["relative_path"]).name for item in labels},
                                 {f"{i:03d}.csv" for i in range(50)})
                self.assertEqual({Path(item["relative_path"]).name for item in images},
                                 {f"{i:03d}.jpg" for i in range(50)})
                combined = hashlib.sha256()
                for item in labels:
                    self.assertTrue(label_path.endswith(str(Path(item["relative_path"]).parent)))
                    combined.update(Path(item["relative_path"]).name.encode() + b"\0")
                    combined.update(bytes.fromhex(item["sha256"]))
                self.assertEqual(digest, combined.hexdigest())
            else:
                pin = next(item for item in candidates if label_path.endswith(item["relative_path"]))
                self.assertEqual(digest, pin["sha256"])
            gate = VALIDATE_ALL.normalize_accuracy_gate(row)
            names = VALIDATE_ALL.gate_metric_names(gate)
            self.assertIn("cer_delta", names)
            self.assertIn("wer_delta", names)
            recognition = "--evaluation" in args
            self.assertEqual("hmean_delta" in names, not recognition)
            good = {name: 0 for name in names}
            good.update({"json0/hmean": 1, "json0/cer": 0, "json0/wer": 0})
            good["passes_source_gt_gate"] = 1
            self.assertEqual(VALIDATE_ALL.evaluate_accuracy_gate(gate, good), [])
            for metric in ("cer_delta", "wer_delta"):
                self.assertTrue(VALIDATE_ALL.evaluate_accuracy_gate(gate, dict(good, **{metric: .006})))
            self.assertTrue(VALIDATE_ALL.evaluate_accuracy_gate(gate, {}))

    def test_relation_summary_reaches_numeric_and_performance_gates(self):
        report={"metrics": {"native_max_diff": 0.0,
                            "performance_ratio": 1.06},
                "fingerprint": "12345678901234567890",
                "rows": [{"native_max_diff": 0.0},
                         {"native_max_diff": 0.0}]}
        parsed, hashes=VALIDATE_ALL.parse_output(
            json.dumps(report), "relation", Path("/absent"))
        self.assertEqual(parsed["native_max_diff"], 0.0)
        self.assertEqual(hashes["relation_output"], report["fingerprint"])
        gate={"require_metrics": ["native_max_diff", "performance_ratio"],
              "checks": [{"metric": "performance_ratio", "op": "<=",
                          "value": 1.05}]}
        failures=VALIDATE_ALL.evaluate_accuracy_gate(gate, parsed)
        self.assertEqual(len(failures), 1)
        self.assertIn("performance_ratio", failures[0])

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
            "[rmbg-perf] device=cuda profile=strict median_ms=561.5 "
            "p95_ms=580.4 output_hash=12345",
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

    def test_ram_dataset_and_caption_metrics_enforce_structured_gates(self):
        report = {
            "cross_f1": 0.995, "upstream_cross_f1": 1.0,
            "latency_over_upstream": 1.06, "mAP": 0.21, "CP": 0.12,
            "CR": 0.15, "caption_exact": 1.0, "caption_word_f1": 1.0,
            "pytorch_caption_exact": 0.98, "pytorch_caption_word_f1": 0.99,
            "datasets": [{"cross_f1": 0.995, "mAP": 0.21}],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "probe.json"
            path.write_text(json.dumps(report))
            metrics, _ = VALIDATE_ALL.parse_output("RAM report ready", "generic", path)
        gate = {"require_metrics": ["json0/mAP", "json0/CP", "json0/CR"],
                "checks": [
                    {"metric": "json0/latency_over_upstream", "op": "<=", "value": 1.05},
                    {"metric": "json0/pytorch_caption_exact", "op": ">=", "value": 0.99},
                ]}
        failures = VALIDATE_ALL.evaluate_accuracy_gate(gate, metrics)
        self.assertEqual(len(failures), 2)
        self.assertIn("latency_over_upstream", failures[0])
        self.assertIn("pytorch_caption_exact", failures[1])

    def test_ram_new_arithmetic_contracts_are_candidate_only_after_normalization(self):
        manifest = json.loads(
            MODULE_PATH.with_name("validation_manifest.json").read_text())
        expected = {
            "ram-upstream-datasets": {
                "json0/arithmetic_probability_max_error",
                "json0/arithmetic_probability_mae",
                "json0/latency_over_upstream",
            },
            "ram-upstream-caption": {"json0/latency_over_upstream"},
        }
        for scenario_id, metrics_to_check in expected.items():
            scenario = next(row for row in manifest["scenarios"]
                            if row["id"] == scenario_id)
            gate = VALIDATE_ALL.normalize_accuracy_gate(scenario)
            checks = {check["metric"]: check for check in gate["checks"]}
            for metric in metrics_to_check:
                check = checks[metric]
                self.assertTrue(check.get("candidate_only"),
                                f"{scenario_id}: {metric}")
                isolated_gate = {"require_metrics": [metric], "checks": [check]}
                expected_value = float(check["value"])
                failing_value = (expected_value + 1.0 if check["op"] in ("<", "<=")
                                 else expected_value - 1.0)
                values = {metric: failing_value}
                self.assertEqual(
                    VALIDATE_ALL.evaluate_accuracy_gate(
                        isolated_gate, values, backend="cuda", baseline=True), [])
                failures = VALIDATE_ALL.evaluate_accuracy_gate(
                    isolated_gate, values, backend="cuda")
                self.assertTrue(any(metric in failure for failure in failures),
                                f"{scenario_id}: {metric}")

    def test_quantization_alignment_metrics_reach_the_accuracy_gate(self):
        report = {
            "pytorch_probability_max_error": 0.06,
            "pytorch_probability_error_excess": 0.0,
            "pytorch_cross_f1_drop": 0.0,
            "pytorch_annotation_error_excess": 0.0,
            "upstream_pytorch_cross_f1": 0.989,
            "upstream_pytorch_probability_mae": 0.0005,
            "upstream_pytorch_probability_max_error": 0.06,
            "upstream_logit_max_error": 0.0,
            "upstream_probability_max_error": 0.0,
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "probe.json"
            path.write_text(json.dumps(report))
            metrics, _ = VALIDATE_ALL.parse_output("probe complete", "generic", path)
        for key, value in report.items():
            self.assertEqual(metrics[f"json0/{key}"], value)
        gate = {"require_metrics": [f"json0/{key}" for key in report],
                "checks": [{"metric": "json0/pytorch_probability_error_excess",
                            "op": "<=", "value": 1e-6}]}
        self.assertEqual(VALIDATE_ALL.evaluate_accuracy_gate(gate, metrics), [])
        metrics["json0/pytorch_probability_error_excess"] = 0.001
        failures = VALIDATE_ALL.evaluate_accuracy_gate(gate, metrics)
        self.assertEqual(len(failures), 1)
        self.assertIn("probability_error_excess", failures[0])

    def test_ram_cpu_q8_gate_keeps_independent_pytorch_probability_checks(self):
        manifest = json.loads(MODULE_PATH.with_name("validation_manifest.json").read_text())
        gate = next(row["accuracy"] for row in manifest["scenarios"]
                    if row["id"] == "ram-upstream-demo1")
        report = {"max_diff": 0.15, "mean_diff": 0.008,
                  "accuracy_max_diff": 0.0, "upstream_cpu_max_diff": 0.0,
                  "compatibility_accuracy_max_diff": 0.0,
                  "cpu_arithmetic_probability_max_diff": 0.0,
                  "cpu_arithmetic_probability_mean_diff": 0.0,
                  "probability_max_diff": 0.03, "probability_mean_diff": 0.001}
        metrics, _ = VALIDATE_ALL.parse_output(json.dumps(report), "generic", Path("/absent"))
        failures = VALIDATE_ALL.evaluate_accuracy_gate(gate, metrics)
        self.assertEqual(len(failures), 1)
        self.assertIn("probability_max_diff", failures[0])
        report["probability_max_diff"] = 0.011
        report["compatibility_accuracy_max_diff"] = 0.03
        metrics, _ = VALIDATE_ALL.parse_output(json.dumps(report), "generic", Path("/absent"))
        failures = VALIDATE_ALL.evaluate_accuracy_gate(gate, metrics)
        self.assertEqual(len(failures), 1)
        self.assertIn("accuracy_max_diff", failures[0])

    def test_ram_dataset_gate_rejects_upstream_matching_f1_below_original_truth_bound(self):
        manifest = json.loads(MODULE_PATH.with_name("validation_manifest.json").read_text())
        gate = next(row["accuracy"] for row in manifest["scenarios"]
                    if row["id"] == "ram-upstream-datasets")
        # Every other bound is met, including exact upstream agreement. That
        # agreement cannot make a failed original-checkpoint F1 pass.
        metrics = {key: 0.0 for key in gate["require_metrics"]}
        metrics.update({"json0/cross_f1": 0.989010989010989,
                        "json0/upstream_cross_f1": 1.0,
                        "json0/compatibility_upstream_cross_f1": 1.0,
                        "json0/latency_over_upstream": 1.0})
        failures = VALIDATE_ALL.evaluate_accuracy_gate(
            gate, metrics, backend="cuda")
        self.assertEqual(len(failures), 1)
        self.assertIn("cross_f1", failures[0])
        metrics["json0/cross_f1"] = 0.99
        self.assertEqual(
            VALIDATE_ALL.evaluate_accuracy_gate(gate, metrics, backend="cuda"), [])

    def test_backend_specific_accuracy_check_only_applies_to_declared_backend(self):
        gate = {
            "require_metrics": [],
            "checks": [{
                "metric": "compatibility_upstream_logit_max_error",
                "op": "<=",
                "value": 1e-4,
                "backends": ["cpu"],
            }],
        }
        metrics = {"compatibility_upstream_logit_max_error": 0.05}
        self.assertEqual(
            VALIDATE_ALL.evaluate_accuracy_gate(gate, metrics, backend="cuda"), [])
        failures = VALIDATE_ALL.evaluate_accuracy_gate(gate, metrics, backend="cpu")
        self.assertEqual(len(failures), 1)
        self.assertIn("compatibility_upstream_logit_max_error", failures[0])

    def test_candidate_only_accuracy_check_is_observational_for_baseline(self):
        gate = {
            "require_metrics": ["new_contract_error"],
            "checks": [{
                "metric": "new_contract_error",
                "op": "<=",
                "value": 0.02,
                "candidate_only": True,
            }],
        }
        metrics = {"new_contract_error": 0.12}
        self.assertEqual(
            VALIDATE_ALL.evaluate_accuracy_gate(gate, metrics, baseline=True), [])
        failures = VALIDATE_ALL.evaluate_accuracy_gate(gate, metrics)
        self.assertEqual(len(failures), 1)
        self.assertIn("new_contract_error", failures[0])

    def test_scenario_selection_narrows_without_mutating_manifest(self):
        manifest = {
            "tasks": {"tagging": {}},
            "scenarios": [
                {"id": "demo", "task": "tagging"},
                {"id": "datasets", "task": "tagging"},
                {"id": "caption", "task": "tagging"},
            ],
        }
        narrowed = VALIDATE_ALL.select_scenarios(
            manifest, "datasets,caption")
        self.assertEqual([row["id"] for row in narrowed["scenarios"]],
                         ["datasets", "caption"])
        self.assertEqual(len(manifest["scenarios"]), 3)
        with self.assertRaisesRegex(ValueError, "unknown scenarios: missing"):
            VALIDATE_ALL.select_scenarios(manifest, "missing")

    def test_cpu_accuracy_improvement_cannot_hide_failed_upstream_compatibility(self):
        manifest = json.loads(MODULE_PATH.with_name("validation_manifest.json").read_text())
        gate = next(row["accuracy"] for row in manifest["scenarios"]
                    if row["id"] == "ram-upstream-datasets")
        metrics = {key: 0.0 for key in gate["require_metrics"]}
        metrics.update({
            "json0/cross_f1": 1.0,
            "json0/compatibility_upstream_cross_f1": 1.0,
            "json0/compatibility_upstream_probability_max_error": 0.0,
            "json0/compatibility_pytorch_probability_error_excess": 0.0,
            "json0/compatibility_pytorch_cross_f1_drop": 0.0,
            "json0/compatibility_pytorch_annotation_error_excess": 0.0,
            "json0/latency_over_upstream": 1.0,
            "json0/compatibility_upstream_logit_max_error": 0.001,
        })
        failures = VALIDATE_ALL.evaluate_accuracy_gate(
            gate, metrics, backend="cpu")
        self.assertEqual(len(failures), 1)
        self.assertIn("compatibility_upstream_logit_max_error", failures[0])
        metrics["json0/compatibility_upstream_logit_max_error"] = 0
        metrics["json0/arithmetic_probability_max_error"] = .03
        failures = VALIDATE_ALL.evaluate_accuracy_gate(
            gate, metrics, backend="cpu")
        self.assertEqual(len(failures), 1)
        self.assertIn("arithmetic_probability_max_error", failures[0])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "probe.json"
            report = {key.removeprefix("json0/"): value for key, value in metrics.items()}
            output.write_text(json.dumps(report))
            parsed, _ = VALIDATE_ALL.parse_output("probe complete", "generic", output)
        self.assertEqual(parsed, metrics)
    def test_runtime_library_path_recovers_qt_from_cmake_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            build = Path(temp) / "build"
            qt_lib = Path(temp) / "qt/lib"
            qt_core_dir = qt_lib / "cmake/Qt5Core"
            qt_core_dir.mkdir(parents=True)
            expected = qt_lib.parent / "bin" if VALIDATE_ALL.os.name == "nt" \
                else qt_lib
            expected.mkdir(parents=True, exist_ok=True)
            build.mkdir()
            (build / "CMakeCache.txt").write_text(
                f"Qt5Core_DIR:PATH={qt_core_dir}\n", encoding="utf-8")
            env = {}
            VALIDATE_ALL.add_runtime_library_path(env, build)
            variable = ("PATH" if VALIDATE_ALL.os.name == "nt" else
                        "DYLD_LIBRARY_PATH"
                        if VALIDATE_ALL.sys.platform == "darwin"
                        else "LD_LIBRARY_PATH")
            self.assertEqual(env[variable].split(VALIDATE_ALL.os.pathsep),
                             [str(build / "bin"), str(expected.resolve())])

    def test_fingerprint_stability_requires_every_attempt_to_match(self):
        Attempt = VALIDATE_ALL.Attempt
        stable, failures = VALIDATE_ALL.stable_fingerprints([
            Attempt(0, 1.0, fingerprints={"output": "aa"}),
            Attempt(0, 1.0, fingerprints={"output": "bb"}),
        ])
        self.assertFalse(stable)
        self.assertEqual(len(failures), 1)

    def test_stability_only_policy_skips_cross_attempt_hash_gate(self):
        # GPU generators with parallel extractors (e.g. trellis mesh hashmap
        # insertion) have no run-stable vertex order: byte hashes must not
        # gate stability, the probe's own finite/degeneracy checks do.
        Attempt = VALIDATE_ALL.Attempt
        spec = VALIDATE_ALL.RunSpec(
            scenario_id="s", task="trellis", model_id="m",
            model_paths=(), command=(), env={},
            accuracy_gate="fixed-seed finite mesh geometry",
            metric_parser="trellis", fingerprint_policy="stability_only",
            require_fingerprint=True, report_path=VALIDATE_ALL.Path("/tmp/r"),
        )
        stable, failures = VALIDATE_ALL.stable_fingerprints([
            Attempt(0, 1.0, fingerprints={"json0/geometry_sha12": "aaa"}),
            Attempt(0, 1.0, fingerprints={"json0/geometry_sha12": "bbb"}),
        ], spec)
        self.assertEqual((stable, failures), ({}, []))

    def test_exact_policy_still_fails_on_changing_fingerprint(self):
        Attempt = VALIDATE_ALL.Attempt
        spec = VALIDATE_ALL.RunSpec(
            scenario_id="s", task="depth", model_id="m",
            model_paths=(), command=(), env={},
            accuracy_gate="finite depth", metric_parser="generic",
            fingerprint_policy="exact", require_fingerprint=True,
            report_path=VALIDATE_ALL.Path("/tmp/r"),
        )
        stable, failures = VALIDATE_ALL.stable_fingerprints([
            Attempt(0, 1.0, fingerprints={"output_hash": "aa"}),
            Attempt(0, 1.0, fingerprints={"output_hash": "bb"}),
        ], spec)
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
        baseline = self.report([{**row, **{
            "metrics": {"mask_iou": 1.0, "json0/cuda/total/p50_ms": 10.0}}}])
        current = self.report([row])
        self.assertEqual(VALIDATE_ALL.compare_report(
            current, baseline, 5.0, 3.0), [])

    def test_exact_policy_rejects_changed_output(self):
        base_row = {
            "key": "depth/depth/model.gguf", "model_assets": [],
            "metrics": {"inference_ms": 10.0},
            "fingerprints": {"output": "a"}, "fingerprint_policy": "exact",
        }
        current_row = {**base_row, **{"fingerprints": {"output": "b"}}}
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
            self.report([row]), self.report([{**row, **{"status": "fail"}}]),
            5.0, 3.0)
        self.assertEqual(
            failures, ["depth/depth/model.gguf: baseline status is fail"])

    def test_nested_timing_report_is_flattened(self):
        metrics, _ = VALIDATE_ALL.flatten_json(
            {"graph_ms": {"p50": 12.0, "p90": 14.0}})
        self.assertEqual(metrics,
                         {"graph_ms/p50": 12.0, "graph_ms/p90": 14.0})

    def test_ab_compares_candidate_timings_and_peak_not_reference_or_pid_samples(self):
        row = {"key": "table/table/model.gguf", "model_assets": [],
               "metrics": {"json0/native/p50_ms": 10.0,
                           "json0/gpu_sampled_process_peak_mib": 100.0,
                           "json0/source_p50_ms": 20.0,
                           "json0/sources/0/metadata/forward_ms/0": 20.0,
                           "json0/sources/0/audit/samples/0/used_mib": 100.0},
               "fingerprints": {}}
        candidate = {**row, "metrics": {
            "json0/native/p50_ms": 14.0,
            "json0/gpu_sampled_process_peak_mib": 120.0,
            "json0/source_p50_ms": 50.0,
            "json0/sources/0/metadata/forward_ms/0": 50.0,
            "json0/sources/0/audit/samples/1/used_mib": 300.0}}
        failures = VALIDATE_ALL.compare_report(
            self.report([candidate]), self.report([row]), 5.0, 3.0)
        self.assertEqual(len(failures), 2)
        self.assertTrue(any("native/p50_ms regressed" in f for f in failures))
        self.assertTrue(any("gpu_sampled_process_peak_mib regressed" in f
                            for f in failures))

    def test_metrics_named_by_structured_gate_are_flattened(self):
        gate = VALIDATE_ALL.normalize_accuracy_gate({
            "id": "depth-backend-parity", "accuracy_gate": "prose",
            "accuracy": {
                "reference": "cpu_reference",
                "reference_id": "probe",
                "require_metrics": ["depth_relative_mae"],
                "checks": [{"metric": "pose_relative_mae", "op": "<=",
                            "value": 0.02}],
            }})
        line = ('{"cpu_output_hash":"ab","depth_relative_mae":0.001,'
                '"pose_relative_mae":0.0001,"faces":6}')
        bare, _ = VALIDATE_ALL.parse_output(line, "generic", Path("/nonexistent"))
        self.assertEqual(bare, {})
        metrics, fingerprints = VALIDATE_ALL.parse_output(
            line, "generic", Path("/nonexistent"),
            VALIDATE_ALL.gate_metric_names(gate))
        self.assertEqual(metrics, {"json0/depth_relative_mae": 0.001,
                                   "json0/pose_relative_mae": 0.0001})
        self.assertEqual(fingerprints, {"json0/cpu_output_hash": "ab"})
        self.assertEqual(VALIDATE_ALL.evaluate_accuracy_gate(gate, metrics), [])

    def test_ocr_original_gpu_metrics_keep_zero_and_unqualified_timing(self):
        metrics, _ = VALIDATE_ALL.parse_output(
            '{"source_gpu_max_diff":0,"source_gpu_mean_diff":0,'
            '"source_gpu_argmax_match":1,"source_performance_qualified":0,'
            '"source_performance_pass":0,"source_gpu_audit_pass":0,'
            '"gpu_sampled_process_peak_mib":120,"frames":2}',
            "generic", Path("/nonexistent"))
        self.assertEqual(metrics, {
            "json0/source_gpu_max_diff": 0.0,
            "json0/source_gpu_mean_diff": 0.0,
            "json0/source_gpu_argmax_match": 1.0,
            "json0/source_performance_qualified": 0.0,
            "json0/source_performance_pass": 0.0,
            "json0/source_gpu_audit_pass": 0.0,
            "json0/gpu_sampled_process_peak_mib": 120.0,
        })

    def test_lingbot_parity_metrics_are_flattened(self):
        metrics, _ = VALIDATE_ALL.flatten_json({
            "pose_t_rmse": 3.3e-05,
            "pose_quat_maxdiff": 1.9e-05,
            "intr_maxdiff": 0.012,
            "depth_pool_rel_rmse": 2.4e-04,
            "conf_mean_maxdiff": 8.1e-04,
            "frames": 12,  # informational leaf: not whitelisted
        })
        self.assertEqual(
            metrics,
            {"pose_t_rmse": 3.3e-05,
             "pose_quat_maxdiff": 1.9e-05,
             "intr_maxdiff": 0.012,
             "depth_pool_rel_rmse": 2.4e-04,
             "conf_mean_maxdiff": 8.1e-04})

    def test_structured_accuracy_gate_resolves_leaf_metric_and_rejects_threshold(self):
        gate = {
            "reference": "cpu_reference",
            "checks": [{"metric": "kpt_median_px", "op": "<=", "value": 0.005}],
            "require_metrics": ["descriptor_cosine_median"],
        }
        self.assertEqual(
            VALIDATE_ALL.evaluate_accuracy_gate(
                gate, {"json0/kpt_median_px": 0.006,
                       "json0/descriptor_cosine_median": 0.9999}),
            ["accuracy gate failed: kpt_median_px 0.006 <= 0.005"])
        self.assertIn(
            "required accuracy metric missing/non-finite: descriptor_cosine_median",
            VALIDATE_ALL.evaluate_accuracy_gate(
                gate, {"json0/kpt_median_px": 0.001}))

    def test_legacy_accuracy_gate_is_explicitly_probe_owned(self):
        scenario = {"id": "demo", "accuracy_gate": "finite output"}
        normalized = VALIDATE_ALL.normalize_accuracy_gate(scenario)
        self.assertFalse(normalized["structured"])
        self.assertEqual(normalized["reference"], "probe_legacy")

    def test_structured_accuracy_requires_metrics_unless_probe_invariant(self):
        invalid = {"id": "demo", "accuracy": {
            "reference": "upstream", "reference_id": "fixture-v1",
            "checks": [], "require_metrics": []}}
        with self.assertRaisesRegex(ValueError, "checks or require_metrics"):
            VALIDATE_ALL.normalize_accuracy_gate(invalid)

        probe_owned = {"id": "demo", "accuracy": {
            "reference": "probe_invariant", "reference_id": "probe-v1",
            "checks": [], "require_metrics": []}}
        normalized = VALIDATE_ALL.normalize_accuracy_gate(probe_owned)
        self.assertTrue(normalized["structured"])
        self.assertEqual(normalized["reference"], "probe_invariant")

    def test_summary_fails_when_structured_accuracy_check_fails(self):
        spec = VALIDATE_ALL.RunSpec(
            scenario_id="s", task="demo", model_id="m", model_paths=(),
            command=(), env={}, accuracy_gate="numeric", metric_parser="generic",
            fingerprint_policy="none", require_fingerprint=False,
            report_path=Path("/tmp/report"), accuracy={
                "reference": "cpu_reference",
                "checks": [{"metric": "mask_iou", "op": ">=", "value": 0.9}],
                "require_metrics": [], "structured": True,
            })
        row, failures = VALIDATE_ALL.summarize_attempts(
            spec, [VALIDATE_ALL.Attempt(
                0, 1.0, metrics={"mask_iou": 0.8})], False)
        self.assertEqual(row["status"], "fail")
        self.assertEqual(row["accuracy_failures"],
                         ["accuracy gate failed: mask_iou 0.8 >= 0.9"])
        self.assertIn("accuracy gate failed", row["error_summary"])
        self.assertTrue(any("accuracy gate failed" in item for item in failures))

    def test_summary_checks_every_attempt_before_taking_medians(self):
        spec = VALIDATE_ALL.RunSpec(
            scenario_id="s", task="demo", model_id="m", model_paths=(),
            command=(), env={}, accuracy_gate="numeric", metric_parser="generic",
            fingerprint_policy="none", require_fingerprint=False,
            report_path=Path("/tmp/report"), accuracy={
                "checks": [{"metric": "mask_iou", "op": ">=", "value": 0.9}],
                "require_metrics": ["mask_iou"], "structured": True,
            })
        for bad_metrics in ({"mask_iou": 0.8}, {}):
            with self.subTest(bad_metrics=bad_metrics):
                attempts = [VALIDATE_ALL.Attempt(0, 1.0, metrics=value)
                            for value in ({"mask_iou": 1.0}, bad_metrics,
                                          {"mask_iou": 1.0})]
                row, failures = VALIDATE_ALL.summarize_attempts(
                    spec, attempts, False)
                self.assertEqual(row["metrics"]["mask_iou"], 1.0)
                self.assertEqual(row["status"], "fail")
                self.assertIn("attempt 2:", row["error_summary"])
                self.assertTrue(row["attempts"][1]["accuracy_failures"])
                self.assertTrue(any("attempt 2:" in item for item in failures))

    def test_summary_preserves_error_from_an_earlier_failed_attempt(self):
        spec = VALIDATE_ALL.RunSpec(
            scenario_id="s", task="demo", model_id="m", model_paths=(),
            command=(), env={}, accuracy_gate="probe", metric_parser="generic",
            fingerprint_policy="none", require_fingerprint=False,
            report_path=Path("/tmp/report"))
        row, failures = VALIDATE_ALL.summarize_attempts(spec, [
            VALIDATE_ALL.Attempt(1, 1.0, output_tail="FAIL native latency ratio"),
            VALIDATE_ALL.Attempt(0, 1.0, output_tail="PASS"),
        ], False)
        self.assertEqual(row["status"], "fail")
        self.assertEqual(row["error_summary"],
                         "attempt 1: FAIL native latency ratio")
        self.assertTrue(failures)

    def test_profile_defaults_and_release_baseline_requirement(self):
        with mock.patch.object(sys, "argv", ["validate_all.py"]):
            args = VALIDATE_ALL.parse_args()
        self.assertEqual((args.profile, args.repeats, args.warmup_runs,
                          args.inference_runs), ("developer", 2, 1, 2))
        with mock.patch.object(sys, "argv", ["validate_all.py", "--profile",
                                               "release"]):
            with self.assertRaises(SystemExit) as error:
                VALIDATE_ALL.parse_args()
        self.assertEqual(error.exception.code, 2)

    def test_gpu_idle_timeout_defaults_and_positive_validation(self):
        with mock.patch.object(sys, "argv", ["validate_all.py"]):
            args = VALIDATE_ALL.parse_args()
        self.assertEqual(args.gpu_idle_timeout, 60.0)
        with mock.patch.object(sys, "argv", ["validate_all.py",
                                             "--gpu-idle-timeout", "900"]):
            args = VALIDATE_ALL.parse_args()
        self.assertEqual(args.gpu_idle_timeout, 900.0)
        with mock.patch.object(sys, "argv", ["validate_all.py",
                                             "--gpu-idle-timeout", "0"]):
            with self.assertRaises(SystemExit) as error:
                VALIDATE_ALL.parse_args()
        self.assertEqual(error.exception.code, 2)

    def test_gpu_idle_timeout_reaches_dataset_and_caption_probe_args(self):
        """The shared-host tolerance reaches only the probes that accept it:
        the two upstream dataset/caption scenarios declare the placeholder,
        while the demo1 wrapper forwards unknown arguments it does not
        understand and therefore must not declare it."""
        manifest = VALIDATE_ALL.load_manifest(VALIDATE_ALL.DEFAULT_MANIFEST)
        declared = {scenario["id"]: scenario["args"]
                    for scenario in manifest["scenarios"]
                    if scenario["id"].startswith("ram-upstream-")}
        self.assertIn("--gpu-idle-timeout", declared["ram-upstream-datasets"])
        self.assertIn("--gpu-idle-timeout", declared["ram-upstream-caption"])
        self.assertNotIn("--gpu-idle-timeout", declared["ram-upstream-demo1"])
        self.assertEqual(
            VALIDATE_ALL.resolve_token("{gpu_idle_timeout}",
                                       {"gpu_idle_timeout": "600"},
                                       Path("/build")),
            "600")

    def test_release_profile_rejects_allow_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.json"
            baseline.write_text("{}", encoding="utf-8")
            with mock.patch.object(
                    sys, "argv", ["validate_all.py", "--profile", "release",
                                  "--baseline", str(baseline),
                                  "--allow-incomplete"]):
                with self.assertRaises(SystemExit) as error:
                    VALIDATE_ALL.parse_args()
        self.assertEqual(error.exception.code, 2)

    def test_release_accuracy_contract_rejects_legacy_prose(self):
        common = dict(
            scenario_id="s", task="demo", model_id="m", model_paths=(),
            command=(), env={}, accuracy_gate="finite output",
            metric_parser="generic", fingerprint_policy="none",
            require_fingerprint=False, report_path=Path("/tmp/report"))
        legacy = VALIDATE_ALL.RunSpec(**common, accuracy={
            "reference": "probe_legacy", "structured": False})
        structured = VALIDATE_ALL.RunSpec(**common, accuracy={
            "reference": "pytorch", "reference_id": "commit:fixture",
            "checks": [{"metric": "mae", "op": "<=", "value": 0.01}],
            "require_metrics": [], "structured": True})
        failures = VALIDATE_ALL.release_accuracy_contract_failures(
            [legacy, structured])
        self.assertEqual(len(failures), 1)
        self.assertIn(legacy.key, failures[0])

    def test_release_manifest_contract_fails_before_download_and_honors_model_selection(self):
        manifest = {
            "tasks": {"demo": {}},
            "scenarios": [
                {"id": "old", "task": "demo", "for_each_model": True,
                 "model_glob": "demo_models/old*.gguf",
                 "accuracy_gate": "legacy invariant"},
                {"id": "new", "task": "demo", "for_each_model": True,
                 "model_glob": "demo_models/new*.gguf",
                 "accuracy_gate": "numeric invariant", "accuracy": {
                     "reference": "golden", "reference_id": "fixture-v1",
                     "checks": [{"metric": "mae", "op": "<=",
                                 "value": 0.01}],
                     "require_metrics": ["mae"]}},
            ],
        }
        assets = [VALIDATE_ALL.ModelAsset(
            "demo", VALIDATE_ALL.PurePosixPath("demo_models/new-f16.gguf"),
            "https://example.invalid/new-f16.gguf", "a" * 64)]
        selected = {"demo"}
        full_args = SimpleNamespace(models="", full=True)
        failures = VALIDATE_ALL.release_manifest_accuracy_contract_failures(
            manifest, full_args, selected, assets)
        self.assertEqual(len(failures), 1)
        self.assertIn("demo/old", failures[0])

        selected_args = SimpleNamespace(models="new-f16", full=False)
        self.assertEqual(
            VALIDATE_ALL.release_manifest_accuracy_contract_failures(
                manifest, selected_args, selected, assets), [])

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

    def test_manifest_pattern_naming_unpublished_asset_is_drift(self):
        catalog = "\n".join(json.dumps({
            "task": "yolo", "relative_path": f"yolo_models/{name}",
            "url": "https://example.invalid/m.gguf",
            "sha256": "a" * 64, "size_bytes": 0,
        }) for name in ("yolov8s-world-q8_0.gguf", "clip-f16.gguf"))
        manifest = {"scenarios": [
            {"id": "yolo", "task": "yolo",
             "owned_globs": ["yolo_models/yolo*.gguf"],
             "light_globs": ["yolo_models/yolov8s-worldv2-q8_0.gguf"],
             "required_globs": ["yolo_models/clip-f16.gguf"]},
            {"id": "user", "task": "yolo", "local_only": True,
             "model_glob": "yolo_models/private.gguf"},
            {"id": "other", "task": "depth",
             "model_glob": "da3_models/absent.gguf"},
        ]}
        self.assertEqual(
            VALIDATE_ALL.manifest_catalog_drift(manifest, catalog, {"yolo"}),
            ["yolo.light_globs: yolo_models/yolov8s-worldv2-q8_0.gguf"])
        manifest["scenarios"][0]["light_globs"] = [
            "yolo_models/yolov8s-world-q8_0.gguf"]
        self.assertEqual(
            VALIDATE_ALL.manifest_catalog_drift(manifest, catalog, {"yolo"}),
            [])

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

    def test_tar_fixture_extracts_only_pinned_regular_members(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "fixture.tar"
            payload = b"pinned labels"
            member = VALIDATE_ALL.PurePosixPath("dataset/val.txt")
            asset = VALIDATE_ALL.InputAsset(
                "tar-fixture", frozenset({"ocr"}), VALIDATE_ALL.PurePosixPath("fixture.tar"),
                source.as_uri(), "0" * 64, 0, True,
                ((member, hashlib.sha256(payload).hexdigest(), len(payload)),))
            with tarfile.open(source, "w") as archive:
                entry = tarfile.TarInfo(str(member))
                entry.size = len(payload)
                archive.addfile(entry, io.BytesIO(payload))
                # Unpinned entries must never be extracted.
                link = tarfile.TarInfo("../outside")
                link.type, link.linkname = tarfile.SYMTYPE, "/outside"
                archive.addfile(link)
            VALIDATE_ALL.extract_input_archive(asset, source, root / "extract")
            self.assertEqual((root / "extract" / str(member)).read_bytes(), payload)
            self.assertFalse((root / "outside").exists())
            with tarfile.open(source, "w") as archive:
                link = tarfile.TarInfo(str(member))
                link.type, link.linkname = tarfile.SYMTYPE, "/outside"
                archive.addfile(link)
            with self.assertRaisesRegex(RuntimeError, "invalid archive member"):
                VALIDATE_ALL.extract_input_archive(asset, source, root / "extract")
            self.assertEqual((root / "extract" / str(member)).read_bytes(), payload)

    def test_archive_dot_prefix_normalization_and_duplicate_rejection(self):
        for kind in ("tar", "zip"):
            for names, error in ((["./dataset/input"], None),
                                 (["dataset/input", "./dataset/input"], "ambiguous"),
                                 (["dataset/input", "dataset/input"], "ambiguous"),
                                 (["../dataset/input", "/dataset/input"], "missing")):
                with self.subTest(kind=kind, names=names), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    source = root / ("fixture." + kind)
                    payload = b"original source tensor"
                    asset = VALIDATE_ALL.InputAsset(
                        "fixture", frozenset({"table"}),
                        VALIDATE_ALL.PurePosixPath(source.name), source.as_uri(),
                        "0" * 64, 0, True,
                        ((VALIDATE_ALL.PurePosixPath("dataset/input"),
                          hashlib.sha256(payload).hexdigest(), len(payload)),))
                    if kind == "tar":
                        with tarfile.open(source, "w") as archive:
                            for name in names:
                                entry = tarfile.TarInfo(name)
                                entry.size = len(payload)
                                archive.addfile(entry, io.BytesIO(payload))
                    else:
                        with zipfile.ZipFile(source, "w") as archive:
                            for name in names:
                                archive.writestr(name, payload)
                    if error:
                        with self.assertRaisesRegex(RuntimeError, error):
                            VALIDATE_ALL.extract_input_archive(asset, source, root / "extract")
                        self.assertFalse((root / "extract/dataset/input").exists())
                    else:
                        VALIDATE_ALL.extract_input_archive(asset, source, root / "extract")
                        self.assertEqual((root / "extract/dataset/input").read_bytes(), payload)

    def test_dot_prefixed_archive_links_are_rejected_even_with_matching_payload(self):
        for kind in ("tar", "zip"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / ("fixture." + kind)
                payload = b"link-target"
                asset = VALIDATE_ALL.InputAsset(
                    "fixture", frozenset({"table"}),
                    VALIDATE_ALL.PurePosixPath(source.name), source.as_uri(),
                    "0" * 64, 0, True,
                    ((VALIDATE_ALL.PurePosixPath("dataset/input"),
                      hashlib.sha256(payload).hexdigest(), len(payload)),))
                if kind == "tar":
                    with tarfile.open(source, "w") as archive:
                        entry = tarfile.TarInfo("./dataset/input")
                        entry.type, entry.linkname = tarfile.SYMTYPE, payload.decode()
                        archive.addfile(entry)
                else:
                    with zipfile.ZipFile(source, "w") as archive:
                        entry = zipfile.ZipInfo("./dataset/input")
                        entry.create_system = 3
                        entry.external_attr = (0o120777 << 16)
                        archive.writestr(entry, payload)
                with self.assertRaisesRegex(RuntimeError, "invalid archive member"):
                    VALIDATE_ALL.extract_input_archive(asset, source, root / "extract")
                self.assertFalse((root / "extract/dataset/input").exists())

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
        missing = {**row, **{"key": "demo/model/missing"}}
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
                gpu_idle_timeout=60.0,
            )
            specs, diagnostics = VALIDATE_ALL.expand_specs(
                manifest, args, root / "reports")
            self.assertEqual(len(specs), 1)
            self.assertEqual(diagnostics["uncovered_assets"],
                             ["demo: demo_models/new.gguf"])

    def test_same_model_in_multiple_scenarios_keeps_distinct_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = root / "assets"
            (assets / "demo_models").mkdir(parents=True)
            (assets / "demo_models/model.gguf").write_bytes(b"model")
            manifest = {"tasks": ["demo"], "scenarios": [
                {"id": scenario, "task": "demo", "model_glob": "demo_models/*.gguf",
                 "for_each_model": True, "executable": "python",
                 "args": ["--output", "{scenario_report}"], "accuracy_gate": "demo"}
                for scenario in ("tags", "caption")]}
            args = SimpleNamespace(
                tasks="", assets=assets, repo=root, build=root / "build",
                backend="cpu", image="image", image2="image2", face_image="face",
                sam_image="sam", yolo_image="yolo", threads=1, warmup_runs=1,
                inference_runs=1, trellis_steps=1, gpu_idle_timeout=60.0)
            specs, _ = VALIDATE_ALL.expand_specs(manifest, args, root / "reports")
            self.assertEqual(len(specs), 2)
            self.assertNotEqual(specs[0].report_path, specs[1].report_path)
            self.assertEqual(specs[0].report_path.parent.name, "tags")
            self.assertEqual(specs[1].report_path.parent.name, "caption")

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
            gpu_idle_timeout=60.0, full=full)

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

    def test_sweep_task_owned_scenarios_removes_raw_sidecars(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scenario = root / "probes" / "demo" / "caption"
            scenario.mkdir(parents=True)
            (scenario / "demo-model.json").write_text("{}")
            (scenario / "logits.f32").write_bytes(b"raw")
            work = scenario / "raw-run"
            work.mkdir()
            (work / "timing.log").write_text("timing")
            unrelated = root / "probes" / "other"
            unrelated.mkdir()
            (unrelated / "other-model.json").write_text("{}")
            self.assertEqual(VALIDATE_ALL.sweep_task_probe_outputs("demo", root), 1)
            self.assertFalse((root / "probes" / "demo").exists())
            self.assertTrue((unrelated / "other-model.json").exists())
    def test_sweep_task_probe_outputs_removes_probe_scratch_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scratch = root / "probes" / "demo-a.json.upstream.d"
            (scratch / "ref").mkdir(parents=True)
            (scratch / "ref" / "up.depth.bin").write_bytes(b"\0" * 4)
            other = root / "probes" / "plain-a.json.upstream.d"
            other.mkdir()
            self.assertEqual(
                VALIDATE_ALL.sweep_task_probe_outputs("demo", root), 1)
            self.assertFalse(scratch.exists())
            self.assertTrue(other.exists())

    # ------------------------------------------------------------------
    # External truth references: {reference:NAME} tokens resolve from
    # --reference NAME=PATH; an unconfigured reference expands empty so the
    # consuming probe exits 77 and the row fails unless --allow-incomplete.
    # ------------------------------------------------------------------

    def test_reference_token_resolves_configured_and_empties_missing(self):
        build = Path("/nonexistent/build")
        references = {"upstream": "/opt/upstream"}
        self.assertEqual(VALIDATE_ALL.resolve_token(
            "--reference={reference:upstream} --b {backend}",
            {"backend": "cuda"}, build, references),
            "--reference=/opt/upstream --b cuda")
        self.assertEqual(VALIDATE_ALL.resolve_token(
            "{reference:absent}", {}, build, references), "")
        self.assertEqual(VALIDATE_ALL.resolve_token(
            "{reference:upstream}", {}, build), "")

    def test_parse_references_rejects_malformed_items(self):
        parsed = VALIDATE_ALL.parse_references(["up=/tmp/x"])
        self.assertEqual(parsed, {"up": str(Path("/tmp/x").resolve())})
        for bad in ("up", "=/tmp/x", "up="):
            with self.assertRaisesRegex(ValueError, "NAME=PATH"):
                VALIDATE_ALL.parse_references([bad])

    def test_reference_interpreter_preserves_virtual_environment(self):
        import venv
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "reference-env"
            venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(root)
            python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            references = VALIDATE_ALL.parse_references([f"python={python}"])
            actual = subprocess.check_output(
                [references["python"], "-c", "import sys; print(sys.prefix)"],
                text=True).strip()
            self.assertEqual(Path(actual), root)

    def test_reference_row_without_reference_is_a_gate_failure(self):
        spec = VALIDATE_ALL.RunSpec(
            scenario_id="upstream", task="demo", model_id="m",
            model_paths=(), command=(), env={}, accuracy_gate="bit-exact",
            metric_parser="generic", fingerprint_policy="exact",
            require_fingerprint=True, report_path=Path("/tmp/report"),
            accuracy={"reference": "upstream_cli", "reference_id": "rev",
                      "checks": [{"metric": "max_diff", "op": "<=",
                                  "value": 0.0}],
                      "require_metrics": ["raw_files_compared"],
                      "structured": True})
        skipped = [VALIDATE_ALL.Attempt(77, 1.0)]
        row, failures = VALIDATE_ALL.summarize_attempts(spec, skipped, False)
        self.assertEqual(row["status"], "skipped")
        self.assertEqual(len(failures), 1)
        _, failures = VALIDATE_ALL.summarize_attempts(spec, skipped, True)
        self.assertEqual(failures, [])
        # A passing probe that compared no files is not evidence.
        empty = [VALIDATE_ALL.Attempt(0, 1.0, metrics={"max_diff": 0.0},
                                      fingerprints={"output_hash": "1"})]
        row, failures = VALIDATE_ALL.summarize_attempts(spec, empty, False)
        self.assertEqual(row["status"], "fail")
        self.assertTrue(any("raw_files_compared" in item for item in failures))
        metrics, _ = VALIDATE_ALL.flatten_json(
            {"upstream_parity": {"max_diff": 0.0, "raw_files_compared": 3,
                                 "views": 3}})
        self.assertEqual(metrics, {"upstream_parity/max_diff": 0.0,
                                   "upstream_parity/raw_files_compared": 3.0})

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

    def test_baseline_model_identity_is_summarized_before_cache_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "assets/demo_models/m.gguf"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"verified model")

            def make_spec():
                return VALIDATE_ALL.RunSpec(
                    scenario_id="s", task="demo", model_id="m",
                    model_paths=(model,), command=(), env={},
                    accuracy_gate="probe", metric_parser="generic",
                    fingerprint_policy="none", require_fingerprint=False,
                    report_path=root / "report.json")

            attempt = [VALIDATE_ALL.Attempt(0, 1.0)]
            row, failures, baseline_row, baseline_failures = (
                VALIDATE_ALL.summarize_attempt_pair(
                    make_spec(), attempt, make_spec(), attempt, False))
            self.assertEqual((row["status"], baseline_row["status"]),
                             ("pass", "pass"))
            self.assertEqual(failures, [])
            self.assertEqual(baseline_failures, [])
            self.assertEqual(row["model_assets"], baseline_row["model_assets"])

            VALIDATE_ALL.release_finished_task_outputs(
                "demo", {"demo": 1}, {"demo": {model}}, {model: {"demo"}},
                self.cleanup_args(root, clean_model_cache=True))
            self.assertFalse(model.exists())
            self.assertEqual(row["model_assets"][0]["sha256"],
                             baseline_row["model_assets"][0]["sha256"])

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
            row("yolo", "yolo_models/yolo26n-obb-q8_0.gguf"),
            row("yolo", "yolo_models/yolo26l-cls-f32.gguf"),
            # A shared catalog owner may describe the selected physical file.
            row("reid", "yolo_models/yolo26n-obb-q8_0.gguf"),
            row("reid", "yolo_models/yolo26l-cls-f32.gguf"),
        ))
        selection = {
            "*/yolo26n-obb-q8_0.gguf",
        }
        assets = VALIDATE_ALL.parse_catalog_output(
            output, {"yolo"}, dependency_patterns=selection,
            task_selection={"yolo": selection})
        self.assertEqual(
            {str(asset.relative_path) for asset in assets},
            {"yolo_models/yolo26n-obb-q8_0.gguf"})
        with self.assertRaisesRegex(ValueError, "matched no catalog rows"):
            VALIDATE_ALL.parse_catalog_output(
                output, {"yolo"}, task_selection={"yolo": {"*/nothing"}})

    def test_load_catalog_explicit_model_excludes_unrelated_task_models(self):
        def row(task, relative):
            return {
                "task": task, "relative_path": relative,
                "url": f"https://example.invalid/{relative}",
                "sha256": "a" * 64, "size_bytes": 1,
            }

        output = "\n".join(json.dumps(item) for item in (
            row("demo", "demo_models/selected.gguf"),
            row("demo", "demo_models/sibling.gguf"),
            row("other", "shared_models/common.gguf"),
            row("other", "other_models/unrelated.gguf"),
        ))
        manifest = {
            "tasks": ["demo", "other"],
            "scenarios": [
                {"id": "demo-selected", "task": "demo",
                 "owned_globs": ["demo_models/selected.gguf"],
                 "required_globs": ["shared_models/common.gguf"],
                 "for_each_model": True},
                {"id": "demo-sibling", "task": "demo",
                 "owned_globs": ["demo_models/sibling.gguf"],
                 "for_each_model": True},
                {"id": "other", "task": "other",
                 "owned_globs": ["other_models/unrelated.gguf"],
                 "for_each_model": True},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "test_catalog_dump_urls"
            binary.touch()
            args = SimpleNamespace(
                catalog_binary=binary, build=root, models="selected",
                full=False)
            completed = SimpleNamespace(
                returncode=0, stdout=output, stderr="")
            with mock.patch.object(VALIDATE_ALL.subprocess, "run",
                                   return_value=completed):
                assets = VALIDATE_ALL.load_model_catalog(
                    args, manifest, {"demo"})
        self.assertEqual(
            {str(asset.relative_path) for asset in assets},
            {"demo_models/selected.gguf", "shared_models/common.gguf"})

    def test_bundle_partial_selection_prefetches_fixed_dependencies(self):
        manifest = {
            "tasks": ["ocr"],
            "scenarios": [{
                "id": "ocr-layout-regions-f32", "task": "ocr",
                "covered_globs": ["ocr_models/pplcnet-layout-f32.gguf"],
                "required_globs": [
                    "ocr_models/ppocr-det-ch-f32.gguf",
                    "ocr_models/ppocr-rec-ch-f32.gguf",
                ],
            }],
        }
        selection = VALIDATE_ALL.model_selection_globs(
            manifest, {"ocr"}, ["pplcnet-layout-f32"])
        self.assertEqual(selection, {
            "ocr_models/ppocr-det-ch-f32.gguf",
            "ocr_models/ppocr-rec-ch-f32.gguf",
        })

    def test_input_assets_shrink_to_declared_selected_scenarios(self):
        def input_row(asset_id):
            return {
                "id": asset_id, "tasks": ["ocr"],
                "relative_path": f"fixtures/{asset_id}.npy",
                "url": f"https://example.invalid/{asset_id}.npy",
                "sha256": "a" * 64, "size_bytes": 1,
            }

        manifest = {
            "tasks": ["ocr"],
            "scenarios": [
                {"id": "layout", "task": "ocr", "input_ids": ["layout"]},
                {"id": "regions", "task": "ocr", "input_ids": ["regions"]},
            ],
            "input_assets": [input_row("layout"), input_row("regions")],
        }
        assets = VALIDATE_ALL.parse_input_assets(
            manifest, {"ocr"}, {"layout"})
        self.assertEqual([asset.asset_id for asset in assets], ["layout"])

    def test_input_assets_keep_task_fallback_for_legacy_scenarios(self):
        manifest = {
            "tasks": ["ocr"],
            "scenarios": [{"id": "legacy", "task": "ocr"}],
            "input_assets": [{
                "id": "fixture", "tasks": ["ocr"],
                "relative_path": "fixtures/input.npy",
                "url": "https://example.invalid/input.npy",
                "sha256": "a" * 64, "size_bytes": 1,
            }],
        }
        assets = VALIDATE_ALL.parse_input_assets(
            manifest, {"ocr"}, {"legacy"})
        self.assertEqual([asset.asset_id for asset in assets], ["fixture"])

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
            warmup_runs=1, inference_runs=1, trellis_steps=1,
            gpu_idle_timeout=60.0, full=False)
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
                gpu_idle_timeout=60.0,
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
            gpu_idle_timeout=60.0,
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
        baseline = self.report([{**row, **{
            "status": "pass", "metrics": {"inference_ms": 10.0}}}])
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
            '"official_max_error": 0.0,',
            '"official_error_excess": 0.0,',
        ])
        summary = VALIDATE_ALL.extract_error_summary(tail)
        self.assertIn("failed to load image", summary)
        self.assertIn("FAIL /path/test.cpp:48", summary)
        self.assertNotIn("ggml_cuda_init", summary)
        self.assertNotIn("suite", summary)
        self.assertNotIn("official_", summary)

    def test_gpu_audit_failure_summary_distinguishes_waiting_from_failure(self):
        tail = "\n".join([
            "waiting for transient GPU activity: [42]",
            "Discarding GPU-overlapped sample [42]; audit: /tmp/sample.overlap-1.log",
            "GPU overlap persisted after 4 attempts: /tmp/sample.log",
        ])
        self.assertEqual(VALIDATE_ALL.extract_error_summary(tail),
                         "GPU overlap persisted after 4 attempts: /tmp/sample.log")
        self.assertIn("/tmp/idle.log", VALIDATE_ALL.extract_error_summary(
            "GPU preflight timed out with foreign compute activity; audit: /tmp/idle.log"))

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


    # ------------------------------------------------------------------
    # relation: manifest consumers for every catalog model
    # ------------------------------------------------------------------

    REPO = MODULE_PATH.parents[3]

    @classmethod
    def relation_catalog_files(cls):
        source = (cls.REPO / "core/AICore/src/tasks/relation/"
                  "model_catalog.cpp").read_text(encoding="utf-8")
        names = re.findall(r'^\s*\{"([^"]+\.(?:gguf|json))",\s*$', source,
                           re.MULTILINE)
        return names

    @staticmethod
    def relation_args(assets, root, full):
        return SimpleNamespace(
            tasks="relation", assets=assets, repo=root,
            build=root / "build", backend="cpu", image="image",
            image2="image2", face_image="face", sam_image="sam",
            yolo_image="yolo", threads=1, warmup_runs=1, inference_runs=1,
            trellis_steps=1, gpu_idle_timeout=60.0, full=full)

    def expand_relation(self, full):
        names = self.relation_catalog_files()
        manifest = VALIDATE_ALL.load_manifest(
            MODULE_PATH.with_name("validation_manifest.json"))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = root / "assets"
            (assets / "relation_models").mkdir(parents=True)
            for name in names:
                (assets / "relation_models" / name).write_bytes(b"x")
            specs, diagnostics = VALIDATE_ALL.expand_specs(
                manifest, self.relation_args(assets, root, full),
                root / "reports")
        return names, specs, diagnostics

    def test_relation_catalog_has_twenty_five_published_entries(self):
        names = self.relation_catalog_files()
        self.assertEqual(len(names), 25)
        self.assertEqual(
            len([n for n in names if n.startswith("relsgg-convnext-")]), 12)

    def test_relation_manifest_consumes_every_catalog_model_in_full_matrix(self):
        names, specs, diagnostics = self.expand_relation(full=True)
        self.assertEqual(diagnostics["missing_assets"], [])
        self.assertEqual(diagnostics["uncovered_assets"], [])
        for name in names:
            if name.endswith("-text-student-f32.gguf") or \
                    name.endswith("tokenizer.json"):
                continue
            plain = {s.scenario_id for s in specs if s.model_id == name}
            self.assertIn(
                "relation-" + name[len("relsgg-"):-len(".gguf")], plain,
                name)
            self.assertIn(
                "relation-" + name[len("relsgg-"):-len(".gguf")] + "-default",
                plain, name)

    def test_relation_convnext_rows_mirror_vit_storage_gates_and_tiering(self):
        manifest = VALIDATE_ALL.load_manifest(
            MODULE_PATH.with_name("validation_manifest.json"))
        rows = {s["id"]: s for s in manifest["scenarios"]}
        for size in ("tiny", "small", "base", "large"):
            for quant in ("f16", "f32", "q8_0"):
                for suffix in ("", "-default"):
                    row = rows[f"relation-convnext-{size}-{quant}{suffix}"]
                    ref = rows[f"relation-vitb16-{quant}{suffix}"]
                    name = f"relation_models/relsgg-convnext-{size}-{quant}.gguf"
                    self.assertEqual(row["owned_globs"], [name])
                    self.assertEqual(row["model_glob"], name)
                    self.assertTrue(row["for_each_model"])
                    self.assertEqual(row["light"], ref["light"])
                    self.assertEqual(row["args"], ref["args"])
                    self.assertEqual(row["accuracy"], ref["accuracy"])

    def test_relation_light_tier_runs_default_rows_for_all_twelve_convnext(self):
        names, specs, diagnostics = self.expand_relation(full=False)
        self.assertEqual(diagnostics["missing_assets"], [])
        self.assertEqual(diagnostics["uncovered_assets"], [])
        convnext = {n for n in names if n.startswith("relsgg-convnext-")}
        light = {s.model_id for s in specs
                 if s.scenario_id.startswith("relation-convnext-")}
        self.assertEqual(light, convnext)
        self.assertTrue(all(
            s.scenario_id.endswith("-default") for s in specs
            if s.scenario_id.startswith("relation-convnext-")))
        # Plugin-level temporal rows are full-matrix only.
        self.assertFalse([s for s in specs
                          if s.scenario_id.startswith("relation-temporal")])

    def test_relation_temporal_rows_are_model_free_full_matrix_probes(self):
        _, specs, _ = self.expand_relation(full=True)
        by_id = {s.scenario_id: s for s in specs}
        tracker = by_id["relation-temporal-tracker"]
        self.assertEqual(Path(tracker.command[0]).name,
                         "test_relate_temporal")
        self.assertEqual(tracker.model_paths, ())
        self.assertEqual(tracker.accuracy["reference"], "probe_invariant")
        self.assertFalse(tracker.require_fingerprint)
        parity = by_id["relation-temporal-upstream-parity"]
        self.assertIn("compare_temporal_upstream.py", parity.command[1])
        self.assertEqual(Path(parity.command[-1]).name,
                         "test_relate_temporal_replay")

    def test_find_binary_searches_plugin_test_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            build = Path(temp)
            folder = build / "bin" / "plugin_test"
            folder.mkdir(parents=True)
            (folder / "test_relate_temporal").write_bytes(b"")
            self.assertEqual(
                VALIDATE_ALL.find_binary(build, "test_relate_temporal"),
                (folder / "test_relate_temporal").resolve())

    def test_missing_probe_executable_is_a_row_failure_not_a_crash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = self.make_spec(
                temp, command=(str(root / "absent_probe"),),
                report_path=root / "probe.json")
            args = SimpleNamespace(repo=root, build=root, timeout=5)
            attempt = VALIDATE_ALL.run_attempt(spec, args)
            self.assertEqual(attempt.returncode, 127)
            self.assertIn("cannot execute probe", attempt.output_tail)


if __name__ == "__main__":
    unittest.main()
