#!/usr/bin/env python3
"""Fail-closed source identity and GT-region ordering for the Paddle probe."""

import hashlib
import copy
import ctypes as C
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from run_ocr_source_pipeline import aicore_library_path
import run_ocr_paddle_pipeline as probe


class PaddleReferenceTest(unittest.TestCase):
    def test_aicore_path_keeps_backend_search_directory_when_library_is_symlinked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "new-runtime"
            build_bin = root / "candidate" / "bin"
            runtime.mkdir()
            build_bin.mkdir(parents=True)
            target = runtime / "libAICore.so"
            target.write_bytes(b"runtime")
            link = build_bin / "libAICore.so"
            try:
                link.symlink_to(target)
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")

            resolved = aicore_library_path(build_bin)
            self.assertEqual(resolved, link.absolute())
            self.assertEqual(resolved.parent, build_bin.absolute())
            self.assertEqual(resolved.resolve(), target.resolve())

    def test_recognition_sort_keeps_original_gt_indices(self):
        crops = [np.zeros((10, 40, 3)), np.zeros((10, 10, 3)), np.zeros((10, 20, 3))]
        recognize = Mock(return_value=[{"rec_text": text, "rec_score": .9}
                                       for text in ("second", "third", "first")])
        ocr = SimpleNamespace(paddlex_pipeline=SimpleNamespace(
            _crop_by_polys=Mock(return_value=crops), text_rec_model=recognize))
        reference = probe.PaddleReference(ocr)
        boxes = [[[i, 0], [i + 1, 0], [i + 1, 1], [i, 1]] for i in range(3)]
        quads, texts, _, indices = reference(np.zeros((10, 10, 3)), boxes)
        self.assertEqual(indices, [1, 2, 0])
        self.assertEqual(quads, [boxes[i] for i in indices])
        restored = [None] * 3
        for (text, _), i in zip(texts, indices):
            restored[i] = text
        self.assertEqual(restored, ["first", "second", "third"])
        ordered = recognize.call_args.args[0]
        self.assertEqual([crop.shape[1] for crop in ordered], [10, 20, 40])

    def test_missing_region_fails_instead_of_silent_zip_truncation(self):
        ocr = SimpleNamespace(paddlex_pipeline=SimpleNamespace(
            _crop_by_polys=lambda *_: [np.zeros((10, 20, 3))], text_rec_model=lambda _: []))
        with self.assertRaisesRegex(ValueError, "lost fixed GT"):
            probe.PaddleReference(ocr)(None, [[[0, 0], [1, 0], [1, 1], [0, 1]]])

    def test_detector_retains_unfiltered_official_boxes(self):
        result = {"dt_polys": ["low-score", "kept"], "rec_polys": ["kept"],
                  "rec_texts": ["word"], "rec_scores": [.8]}
        ocr = SimpleNamespace(paddlex_pipeline=None, predict=Mock(return_value=[result]))
        reference = probe.PaddleReference(ocr)
        image = np.zeros((10, 10, 3))
        self.assertEqual(reference(image)[0], ["kept"])
        self.assertEqual(reference.text_detector(image), ["low-score", "kept"])
        self.assertEqual(ocr.predict.call_count, 1)


class SourceIdentityTest(unittest.TestCase):
    def test_source_page_threads_are_positive_and_runner_owned(self):
        import run_ocr_source_pipeline as source
        scripts = Path(__file__).resolve().parent
        manifest = json.loads((scripts / "validation_manifest.json").read_text())
        rows = [s for s in manifest["scenarios"] if any(
            arg.endswith("/run_ocr_source_pipeline.py")
            for arg in s.get("args", []))]
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(scenario=row["id"]):
                self.assertEqual(row["args"].count("--threads"), 1)
                index = row["args"].index("--threads")
                self.assertEqual(row["args"][index + 1], "{threads}")
        argv = ["probe", "--upstream", "unused", "--assets", "unused",
                "--library", "unused", "--backend", "cpu", "--stem",
                "ppocrv4-source", "--report", "unused", "--threads", "-1"]
        with patch.object(sys, "argv", argv), \
                patch.object(sys, "stderr", io.StringIO()) as error, \
                self.assertRaises(SystemExit) as failure:
            source.main()
        self.assertEqual(failure.exception.code, 2)
        self.assertIn("--threads must be nonnegative", error.getvalue())

    def test_holdout_pins_are_disjoint_and_cover_both_languages_and_quants(self):
        scripts = Path(__file__).resolve().parent
        manifest = json.loads((scripts / "validation_manifest.json").read_text())
        assets = {a["id"]: a for a in manifest["input_assets"]}
        rows = [s for s in manifest["scenarios"] if s["id"].startswith("ppocr-heldout-")]
        self.assertEqual({(r["args"][r["args"].index("--language") + 1],
                          r["args"][r["args"].index("--quant") + 1]) for r in rows},
                         {(lang, quant) for lang in ("ch", "en")
                          for quant in ("f32", "f16", "q8_0")})
        for row in rows:
            inputs = [assets[name] for name in row["input_ids"]
                      if name.startswith("ocr-sroie-holdout-")]
            self.assertEqual(len(inputs), 100)
            self.assertFalse(any(name.startswith("ocr-sroie-receipt-") for name in row["input_ids"]))
            labels = sorted((a for a in inputs if a["relative_path"].endswith(".csv")),
                            key=lambda a: a["relative_path"])
            self.assertEqual({Path(a["relative_path"]).stem for a in labels},
                             {f"{i:03d}" for i in range(50, 100)})
            self.assertEqual({Path(a["relative_path"]).stem for a in inputs
                              if a["relative_path"].endswith(".jpg")},
                             {f"{i:03d}" for i in range(50, 100)})
            digest = hashlib.sha256()
            for asset in labels:
                digest.update(Path(asset["relative_path"]).name.encode() + b"\0")
                digest.update(bytes.fromhex(asset["sha256"]))
            args = row["args"]
            self.assertEqual(args[args.index("--ground-truth-sha256") + 1], digest.hexdigest())
            metrics = {c["metric"] for c in row["accuracy"]["checks"]}
            self.assertTrue({"hmean_delta", "cer_delta", "wer_delta", "e2e_cer_delta",
                             "e2e_wer_delta", "passes_source_gt_gate"} <= metrics)

    def test_page_identity_retains_selected_and_vendored_versions(self):
        import ocr_paddle_metadata
        entries = [SimpleNamespace(metadata={"Name": "typing_extensions"}, version=version)
                   for version in ("4.16.0", "4.12.2")]
        paddle = SimpleNamespace(base=SimpleNamespace(libpaddle=SimpleNamespace(__file__="paddle.so")))
        with patch.dict(sys.modules, {"paddle": paddle}), \
                patch.object(ocr_paddle_metadata.metadata, "distributions", return_value=entries), \
                patch.object(ocr_paddle_metadata.metadata, "version", return_value="4.16.0"), \
                patch.object(probe, "sha256", return_value="digest"):
            identity = probe.reference_identity()
        self.assertEqual(identity["source_environment"], {"typing-extensions": "4.16.0"})
        self.assertEqual(identity["source_distribution_candidates"],
                         {"typing-extensions": ["4.12.2", "4.16.0"]})
        self.assertEqual(identity["source_distribution_identity_sha256"], "digest")

    def test_every_source_and_dictionary_config_is_a_pinned_runner_input(self):
        scripts = Path(__file__).resolve().parent
        pins = json.loads((scripts.parent / "tools/ocr_paddle_sources.json").read_text())
        manifest = json.loads((scripts / "validation_manifest.json").read_text())
        assets = {a["id"]: a for a in manifest["input_assets"]}
        for role, source in pins.items():
            members = {m["relative_path"]: m["sha256"]
                       for m in assets[f"ocr-paddle-source-{role}"]["extracted_files"]}
            for key in ("program", "parameters"):
                entry = source[key]
                self.assertEqual(members[entry["relative_path"]], entry["sha256"])
            config = source["config"]
            asset = assets[f"ocr-paddle-config-{role}"]
            self.assertEqual(asset["relative_path"], config["archive_relative_path"])
            self.assertEqual(asset["sha256"], config["archive_sha256"])
        rows = [s for s in manifest["scenarios"] if s["id"].startswith("ppocr-public-")]
        self.assertEqual(len(rows), 15)
        for row in rows:
            lang = row["args"][row["args"].index("--language") + 1]
            for role in ("det_ch", "rec_" + lang):
                for kind in ("source", "config"):
                    self.assertIn(f"ocr-paddle-{kind}-{role}", row["input_ids"])

    def test_source_config_and_gguf_metadata_are_independently_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = {"name": "original"}
            for key in ("program", "parameters", "config"):
                path = root / key
                path.write_bytes(key.encode())
                row[key] = {"relative_path": key, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            metadata = {"ocr.source.name": "original", "ocr.source.sha256": row["parameters"]["sha256"]}
            with patch.object(probe, "Gguf", return_value=SimpleNamespace(kv=metadata)):
                self.assertEqual(len(probe.verified_source(root, {"det": row}, "det", root / "test.gguf")), 3)
                metadata["ocr.source.name"] = "same-family-other-checkpoint"
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    probe.verified_source(root, {"det": row}, "det", root / "test.gguf")
                metadata["ocr.source.name"] = "original"
                (root / "config").write_text("different dictionary")
                with self.assertRaisesRegex(ValueError, "digest mismatch: det/config"):
                    probe.verified_source(root, {"det": row}, "det", root / "test.gguf")

    def test_archive_config_is_read_only_and_checks_member_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            program, parameters = root / "program", root / "parameters"
            program.write_bytes(b"program")
            parameters.write_bytes(b"parameters")
            archive, payload = root / "config.tar", b"independent dictionary"
            with tarfile.open(archive, "w") as tar:
                info = tarfile.TarInfo("./official/inference.yml")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            row = {"name": "original", **{key: {"relative_path": key,
                   "sha256": hashlib.sha256((root / key).read_bytes()).hexdigest()}
                   for key in ("program", "parameters")}, "config": {
                       "archive_relative_path": archive.name,
                       "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                       "member": "./official/inference.yml", "size_bytes": len(payload),
                       "sha256": hashlib.sha256(payload).hexdigest()}}
            metadata = {"ocr.source.name": "original", "ocr.source.sha256": row["parameters"]["sha256"]}
            with patch.object(probe, "Gguf", return_value=SimpleNamespace(kv=metadata)):
                files = probe.verified_source(root, {"det": row}, "det", root / "test.gguf")
                self.assertEqual(files["config"], payload)
                self.assertFalse((root / "official").exists())
                row["config"]["sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "digest mismatch: det/config"):
                    probe.verified_source(root, {"det": row}, "det", root / "test.gguf")


class ReplayTest(unittest.TestCase):
    def reference_case(self):
        args = SimpleNamespace(ground_truth_sha256="gt", ground_truth=Path("gt.json"),
            source_pins=Path("pins.json"), image_root=Path("images"), evaluation="pipeline",
            language="ch", threads=4)
        result = dict(source_process_returncode=0, ground_truth_sha256="gt",
            source_pins_sha256="pins", probe_sha256="probe", source_files={"det": "weights"},
            source_distribution_identity_sha256="metadata",
            source_distribution_candidates={"selected": ["1"]},
            source_evaluation="pipeline", source_language="ch", source_threads=4,
            pages=[dict(image="a.png", image_sha256="image")])
        return args, result

    def validate(self, args, result):
        def digest(path):
            return {"pins.json": "pins", "run_ocr_paddle_pipeline.py": "probe",
                    "ocr_paddle_metadata.py": "metadata",
                    "a.png": "image"}[Path(path).name]
        with patch.object(probe, "sha256", side_effect=digest), \
                patch.object(probe, "source_digests", return_value={"det": "weights"}), \
                patch("ocr_gt_eval.ground_truth_digest", return_value="gt"), \
                patch("ocr_gt_eval.load_ground_truth", return_value={"a.png": []}):
            probe.validate_reference(args, result, {})

    def test_replay_requires_normal_teardown_receipt_and_exact_source_contract(self):
        args, result = self.reference_case()
        self.validate(args, result)
        for key, wrong in (("source_process_returncode", None), ("source_process_returncode", -6),
                ("source_process_returncode", False), ("source_files", {"det": "other"}),
                ("source_evaluation", "recognition"), ("source_language", "en"),
                ("source_threads", 8), ("probe_sha256", "old")):
            invalid = copy.deepcopy(result)
            invalid[key] = wrong
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                self.validate(args, invalid)

    def test_replay_rejects_legacy_or_changed_distribution_identity(self):
        args, result = self.reference_case()
        for key, wrong in (("source_distribution_identity_sha256", None),
                ("source_distribution_identity_sha256", "old-helper"),
                ("source_distribution_candidates", {})):
            invalid = copy.deepcopy(result)
            invalid[key] = wrong
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                self.validate(args, invalid)

    def test_replay_rejects_missing_duplicate_or_changed_pages(self):
        args, result = self.reference_case()
        for pages in ([], result["pages"] * 2,
                      [dict(image="a.png", image_sha256="changed")]):
            invalid = copy.deepcopy(result)
            invalid["pages"] = pages
            with self.assertRaises(ValueError):
                self.validate(args, invalid)

    def test_duplicate_images_keep_distinct_gt_and_tuple_points(self):
        image = np.zeros((10, 10, 3))
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        first = [[0, 0], [1, 0], [1, 1], [0, 1]]
        second = [[2, 2], [3, 2], [3, 3], [2, 3]]
        pages = [{"decoded_sha256": digest, "regions": [quad],
                  "texts": [(text, .9)], "indices": [0], "e2e": ([], []), "detection": []}
                 for quad, text in [(first, "first"), (second, "second")]]
        reference = probe.ReplayedReference({"pages": pages})
        self.assertEqual(reference(image, [[tuple(p) for p in second]])[1][0][0], "second")
        with self.assertRaisesRegex(ValueError, "GT-region mismatch"):
            reference(image, [[[9, 9], [10, 9], [10, 10], [9, 10]]])


class PaddleRegionAPITest(unittest.TestCase):
    def test_uses_paddle_quad_api_without_selected_polygon_normalization(self):
        from run_ocr_source_pipeline import TextRegion
        native = Mock(return_value=123)
        library = SimpleNamespace(aicore_ocr_recognize=native)
        adapter = probe.PaddleRegionAPI(library)
        points = (C.c_float * 8)(1, 8, 8, 1, 10, 3, 3, 10)
        regions = (TextRegion * 1)(TextRegion(points, 4))
        view = probe.View(0, 20, 20, 60, 4)
        result = adapter.aicore_ocr_recognize_regions(9, C.byref(view), regions, 1)
        self.assertEqual(result, 123)
        self.assertEqual(list(native.call_args.args[2]), list(points))
        self.assertEqual(native.call_args.args[3], 1)

    def test_out_of_image_gt_fails_instead_of_silent_crop_contract_change(self):
        from run_ocr_source_pipeline import TextRegion
        library = SimpleNamespace(aicore_ocr_recognize=Mock())
        adapter = probe.PaddleRegionAPI(library)
        points = (C.c_float * 8)(-1, 0, 8, 0, 8, 4, -1, 4)
        regions = (TextRegion * 1)(TextRegion(points, 4))
        with self.assertRaisesRegex(ValueError, "API bounds"):
            adapter.aicore_ocr_recognize_regions(9, C.byref(probe.View(0, 20, 20, 60, 4)), regions, 1)
        library.aicore_ocr_recognize.assert_not_called()


class GpuPageReferenceTest(unittest.TestCase):
    def test_cpu_reference_cannot_qualify_gpu_page_comparison(self):
        test = ReplayTest()
        args, result = test.reference_case()
        args.source_device = "cuda"
        for value in (None, "cpu"):
            with self.subTest(value=value):
                if value is None: result.pop("source_device", None)
                else: result["source_device"] = value
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    test.validate(args, result)
        result["source_device"] = "cuda"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            test.validate(args, result)
        result["source_tf32"] = False
        test.validate(args, result)

    def test_onnx_page_provider_must_be_cuda_with_tf32_disabled(self):
        import run_ocr_source_pipeline as source
        session = Mock()
        for actual in (["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]):
            session.get_providers.return_value = actual
            with patch.object(source.ort, "InferenceSession", return_value=session) as create:
                if actual[0] == "CPUExecutionProvider":
                    with self.assertRaisesRegex(RuntimeError, "silently fell back"):
                        source.source_session(Path("source.onnx"), 4, True)
                else:
                    self.assertIs(source.source_session(Path("source.onnx"), 4, True), session)
                self.assertEqual(create.call_args.kwargs["providers"],
                    [("CUDAExecutionProvider", {"use_tf32": "0"})])

    def test_original_cuda_page_child_requires_successful_normal_exit(self):
        import run_ocr_source_pipeline as source
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            python = root / "source-python"; python.touch()
            report = root / "report.json"
            args = SimpleNamespace(source_python=python, report=report)
            for code in (0, -6):
                report.write_text(json.dumps({"source_provider": "CUDAExecutionProvider"}))
                def execute(*arguments, **kwargs):
                    self.assertFalse(report.exists(), "stale report must be removed before child")
                    report.write_text(json.dumps({"source_provider": "CUDAExecutionProvider"}))
                    return SimpleNamespace(returncode=code)
                with patch.object(source.subprocess, "run", side_effect=execute) as run:
                    self.assertEqual(source.run_source_child(args, ["--backend", "vulkan"]), code)
                    self.assertEqual(run.call_args.args[0][0], str(python))
                    self.assertIn("--source-cuda-child", run.call_args.args[0])
                actual = json.loads(report.read_text())
                self.assertEqual("source_process_returncode" in actual, code == 0)
            report.write_text(json.dumps({"source_provider": "CPUExecutionProvider"}))
            def cpu_child(*arguments, **kwargs):
                report.write_text(json.dumps({"source_provider": "CPUExecutionProvider"}))
                return SimpleNamespace(returncode=0)
            with patch.object(source.subprocess, "run", side_effect=cpu_child), \
                    self.assertRaisesRegex(ValueError, "did not report CUDA"):
                source.run_source_child(args, [])

    def test_all_page_and_orientation_manifest_consumers_have_cuda_source(self):
        manifest = json.loads(Path(__file__).with_name("validation_manifest.json").read_text())
        counts = {"onnx": 0, "paddle": 0, "orientation": 0}
        for row in manifest["scenarios"]:
            args = row.get("args", [])
            if any(a.endswith("/run_ocr_source_pipeline.py") for a in args):
                self.assertIn("--source-root", args, row["id"]);counts["onnx"] += 1
            if any(a.endswith("/run_ocr_paddle_pipeline.py") for a in args):
                self.assertIn("--source-gpu-python", args, row["id"]);counts["paddle"] += 1
            if any(a.endswith("/run_ocr_onnx_probe.py") for a in args) and "orientation" in args:
                self.assertIn("--source-benchmark-root", args, row["id"]);counts["orientation"] += 1
        self.assertEqual(counts, {"onnx": 66, "paddle": 21, "orientation": 18})


if __name__ == "__main__":
    unittest.main()
