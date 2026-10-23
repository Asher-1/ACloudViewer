#!/usr/bin/env python3
"""Single real-page source-vs-C-ABI diagnostic, scheduled by validate_all.

Uses the pinned source TextSystem and parse_args without importing the GUI.
ONNX Runtime and source dependencies are validation-only. No downloads or
model catalog ownership; all assets are supplied by the common runner.
"""
import argparse
import ast
import ctypes as C
import hashlib
import importlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import onnxruntime as ort
import yaml

class Options(C.Structure):
    _fields_ = [("struct_size", C.c_size_t), ("abi_version", C.c_uint32),
                ("det", C.c_char_p), ("rec", C.c_char_p), ("layout", C.c_char_p),
                ("kie", C.c_char_p), ("maxside", C.c_int32), ("thresh", C.c_float),
                ("unclip", C.c_float), ("batch", C.c_int32), ("device", C.c_char_p),
                ("threads", C.c_int32), ("boxthresh", C.c_float), ("limit", C.c_int32),
                ("cls", C.c_char_p), ("clsthresh", C.c_float), ("clsbatch", C.c_int32),
                ("candidates", C.c_int32)]


class View(C.Structure):
    _fields_ = [("data", C.c_void_p), ("width", C.c_int32), ("height", C.c_int32),
                ("stride", C.c_size_t), ("format", C.c_int)]


class Box(C.Structure):
    _fields_ = [("quad", C.c_float * 8), ("text", C.c_char_p),
                ("score", C.c_float), ("language", C.c_char * 8)]


class TextRegion(C.Structure):
    _fields_ = [("points_xy", C.POINTER(C.c_float)), ("point_count", C.c_int32)]


class Preset(C.Structure):
    _fields_ = [(name, C.c_char_p) for name in
                ("id", "display_name", "det_stem", "rec_stem", "cls_stem")] + [
                ("det_thresh", C.c_float), ("det_box_thresh", C.c_float),
                ("det_unclip_ratio", C.c_float), ("det_max_candidates", C.c_int32),
                ("det_max_side", C.c_int32), ("rec_batch", C.c_int32),
                ("cls_batch", C.c_int32), ("cls_thresh", C.c_float),
                ("drop_score", C.c_float)]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aicore_library_path(path):
    """Return an absolute lexical path, preserving symlink-based build layouts.

    AICore discovers dynamically loaded ggml backends beside the path used to
    load libAICore. Resolving a build/bin symlink can redirect that search to a
    staging directory that contains the core library but not its backends.
    """
    library = Path(os.path.abspath(os.fspath(path)))
    if not library.is_dir():
        return library
    name = ("AICore.dll" if os.name == "nt" else
            "libAICore.dylib" if sys.platform == "darwin" else
            "libAICore.so")
    for candidate in (library / name, library / "Release" / name):
        if candidate.is_file():
            return candidate
    raise ValueError(f"built AICore library unavailable under {library}")


def evaluate_labeled_pages(args, system, library, ctx, preset, identity):
    """Compare detector, fixed-GT-region recognition and complete pipelines.

    GT transcription drives CER/WER. Source predictions are evaluated against
    those same labels, never substituted for them as generated ground truth.
    """
    import ocr_gt_eval as gt

    if not args.image_root or not args.ground_truth_sha256:
        raise ValueError("labelled evaluation requires --image-root and --ground-truth-sha256")
    if gt.ground_truth_digest(args.ground_truth) != args.ground_truth_sha256:
        raise ValueError("ground-truth SHA-256 mismatch")
    truth = gt.load_ground_truth(args.ground_truth)
    recognition_only = args.evaluation == "recognition"
    if args.ground_truth.suffix.lower() == ".json":
        if "documents" in json.loads(args.ground_truth.read_text()) and not recognition_only:
            raise ValueError("XFUND field annotations require --evaluation recognition; not text-line detection GT")
    if not truth or not any(not a.ignore for rows in truth.values() for a in rows):
        raise ValueError("dataset has no evaluable text")
    library.aicore_ocr_recognize_regions.argtypes = [C.c_void_p, C.POINTER(View), C.POINTER(TextRegion), C.c_int32]
    library.aicore_ocr_recognize_regions.restype = C.c_void_p

    def native(image, annotations=None):
        padded = np.zeros((image.shape[0], image.shape[1] + 7, 3), np.uint8)
        padded[:, :image.shape[1]] = image
        view = View(padded.ctypes.data, image.shape[1], image.shape[0], padded.strides[0], 4)
        if annotations is None:
            result = library.aicore_ocr_run(ctx, C.byref(view))
        else:
            buffers = [np.asarray(a.points, np.float32) for a in annotations]
            regions = (TextRegion * len(buffers))(*[TextRegion(
                p.ctypes.data_as(C.POINTER(C.c_float)), len(p)) for p in buffers])
            result = library.aicore_ocr_recognize_regions(ctx, C.byref(view), regions, len(buffers))
        if not result:
            raise RuntimeError(library.aicore_ocr_last_error().decode())
        try:
            rows = []
            for i in range(library.aicore_ocr_box_count(result)):
                box = library.aicore_ocr_box_at(result, i).contents
                points = tuple(zip(box.quad[::2], box.quad[1::2]))
                rows.append(gt.Prediction(points, box.text.decode(), box.score))
            return rows
        finally:
            library.aicore_ocr_result_release(result)

    pages = []
    for name, annotations in truth.items():
        image_path = args.image_root / name
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"dataset image decode failed: {image_path}")
        # Restore the configured drop score: source custom-region calls set it
        # to zero, and otherwise leak that setting into the following page.
        source_e2e, source_detection, actual = [], [], []
        if not recognition_only:
            system.drop_score = preset.drop_score
            source_result = system(image)
            quads, texts = source_result[:2]
            source_e2e = [gt.Prediction(tuple(map(tuple, q)), text, float(score))
                          for q, (text, score) in zip([] if quads is None else quads,
                                                     [] if texts is None else texts)]
            detected = system.text_detector(image)
            source_detection = [gt.Prediction(tuple(map(tuple, q)), "") for q in
                                ([] if detected is None else detected)]
            actual = native(image)
        valid = [a for a in annotations if not a.ignore]
        source_recognition, native_recognition = [], []
        if valid:
            _, texts, _, indices = system(image, dt_boxes=[list(a.points) for a in valid])
            if len(texts) != len(valid) or sorted(indices) != list(range(len(valid))):
                raise ValueError("source lost fixed-GT recognition regions")
            source_recognition = [gt.Prediction(valid[i].points, text, float(score))
                                  for (text, score), i in zip(texts, indices)]
            fixed = native(image, valid)
            if len(fixed) != len(valid):
                raise ValueError("native lost fixed-GT recognition regions")
            native_recognition = [gt.Prediction(a.points, p.text, p.score) for a, p in zip(valid, fixed)]
        page = {"image": name, "image_sha256": sha256(image_path)}
        groups = [
                ("source_recognition", source_recognition, valid),
                ("recognition", native_recognition, valid)]
        if not recognition_only:
            groups += [("source_detection", source_detection, annotations),
                ("detection", actual, annotations),
                ("source_e2e", source_e2e, annotations),
                ("e2e", [p for p in actual if p.score >= preset.drop_score], annotations)]
        for key, predictions, labels in groups:
            page[key] = gt.evaluate(predictions, labels, .5, True)
        page["native_predictions"] = [{"points": p.points, "text": p.text, "score": p.score}
                                      for p in (native_recognition if recognition_only else actual)]
        page["source_predictions"] = [{"points": [list(map(float, point)) for point in p.points],
                                        "text": p.text, "score": p.score}
                                      for p in (source_recognition if recognition_only else source_e2e)]
        pages.append(page)
        # Keep long full-page runs observable without exposing a partial report
        # under the final filename consumed by validate_all.
        progress = args.report.with_suffix(".progress")
        progress.parent.mkdir(parents=True, exist_ok=True)
        temporary = progress.with_name(progress.name + ".tmp")
        temporary.write_text(json.dumps({
            "complete": False, "completed_pages": len(pages),
            "total_pages": len(truth), "last_image": name,
            "backend": args.backend, "quant": args.quant,
            "ground_truth_sha256": args.ground_truth_sha256,
        }) + "\n")
        temporary.replace(progress)
        print(f"OCR labelled pages: {len(pages)}/{len(truth)} {name}", flush=True)
    keys = ["recognition", "source_recognition"]
    if not recognition_only:
        keys += ["detection", "source_detection", "e2e", "source_e2e"]
    aggregates = {key: gt.aggregate(p[key] for p in pages) for key in keys}
    metrics = {"cer": aggregates["recognition"]["cer"], "wer": aggregates["recognition"]["wer"],
               "cer_delta": aggregates["recognition"]["cer"] - aggregates["source_recognition"]["cer"],
               "wer_delta": aggregates["recognition"]["wer"] - aggregates["source_recognition"]["wer"]}
    checks = ["cer_delta", "wer_delta"]
    if not recognition_only:
        metrics.update({"hmean": aggregates["detection"]["hmean"],
               "hmean_delta": aggregates["source_detection"]["hmean"] - aggregates["detection"]["hmean"],
               "e2e_cer_delta": aggregates["e2e"]["cer"] - aggregates["source_e2e"]["cer"],
               "e2e_wer_delta": aggregates["e2e"]["wer"] - aggregates["source_e2e"]["wer"]})
        checks += ["hmean_delta", "e2e_cer_delta", "e2e_wer_delta"]
    passed = all(metrics[key] <= .005 for key in checks)
    report = {**identity, **metrics, "passes_source_gt_gate": int(passed),
              "ground_truth_sha256": args.ground_truth_sha256,
              "ground_truth": args.ground_truth.name, "pages": pages, "aggregates": aggregates,
              "ground_truth_format": "sroie-receipt-csv" if args.ground_truth.is_dir() else "label-file",
              "evaluation": args.evaluation,
              "output_sha256": hashlib.sha256(json.dumps(
                  [p["native_predictions"] for p in pages], sort_keys=True).encode()).hexdigest(),
              "scope": "labelled dataset accuracy; no throughput or VRAM claim"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.report)
    progress = args.report.with_suffix(".progress")
    temporary = progress.with_name(progress.name + ".tmp")
    temporary.write_text(json.dumps({
        "complete": True, "completed_pages": len(pages), "total_pages": len(truth),
        "backend": args.backend, "quant": args.quant,
        "passes_source_gt_gate": int(passed),
        "ground_truth_sha256": args.ground_truth_sha256,
    }) + "\n")
    temporary.replace(progress)
    return int(not passed)


def source_session(path, threads, gpu):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    expected = "CUDAExecutionProvider" if gpu else "CPUExecutionProvider"
    providers = [(expected, {"use_tf32": "0"})] if gpu else [expected]
    session = ort.InferenceSession(str(path), options, providers=providers)
    if session.get_providers()[0] != expected:
        raise RuntimeError("source page provider silently fell back")
    return session


def run_source_child(args, arguments):
    """Normal subprocess exit is required; no original CPU numeric fallback."""
    from run_ocr_onnx_probe import source_environment
    if not args.source_python or not args.source_python.is_file():
        raise ValueError("GPU pages require an original-source CUDA interpreter")
    args.report.unlink(missing_ok=True)
    completed = subprocess.run([str(args.source_python), str(Path(__file__).resolve()),
        *arguments, "--source-cuda-child"], env=source_environment(args.source_python))
    if completed.returncode == 0:
        result = json.loads(args.report.read_text())
        if result.get("source_provider") != "CUDAExecutionProvider":
            raise ValueError("source page child did not report CUDA truth")
        result["source_process_returncode"] = 0
        args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stem", required=True, choices=("ppocrv4-source", "ppocrv3-japan", "ppocrv5-mobile", "ppocrv6-tiny", "ppocrv6-small", "ppocrv6-medium"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--quant", choices=("f32", "f16", "q8_0"), default="f32")
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--ground-truth-sha256")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--evaluation", choices=("pipeline", "recognition"), default="pipeline",
                        help="XFUND semantic-field labels support recognition only")
    parser.add_argument("--selected-regions", action="store_true",
                        help="Compare source custom polygons, including skew, fractional and border-crossing regions")
    parser.add_argument("--source-python", type=Path,
                        help="isolated original ONNX CUDA interpreter for GPU comparisons")
    parser.add_argument("--source-root", type=Path,
                        help="original ONNX CUDA environment directory")
    parser.add_argument("--source-cuda-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.source_root:
        args.source_python = args.source_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    # The common runner uses zero for the probe default. Resolve it once
    # before configuring either the original graph or the native context.
    if args.threads == 0:
        args.threads = 4
    if args.threads < 1:
        parser.error("--threads must be nonnegative")
    if args.ground_truth and args.selected_regions:
        parser.error("labelled evaluation and custom-region diagnostic are separate scenarios")
    if args.backend in ("cuda", "vulkan") and not args.source_cuda_child:
        return run_source_child(args, sys.argv[1:])
    sources = json.loads((Path(__file__).parents[1] / "tools/ocr_onnx_sources.json").read_text())
    upstream = Path(args.upstream).resolve()
    if not args.upstream or not (upstream / ".git").exists():
        print("pinned X-AnyLabeling checkout unavailable", file=sys.stderr)
        return 77
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    if revision != sources["upstream_commit"]:
        raise ValueError("source checkout revision mismatch")
    scoped = "anylabeling/services/auto_labeling"
    if subprocess.check_output(["git", "diff", "HEAD", "--", scoped, "anylabeling/configs/auto_labeling"], cwd=upstream):
        raise ValueError("source OCR implementation/config has local changes")
    modelstem = args.stem.replace("-", "_")
    names = [f"{modelstem}_det_infer.onnx", f"{modelstem}_rec_infer.onnx", "ch_ppocr_mobile_v2.0_cls_infer.onnx"]
    exact_v4 = args.stem in ("ppocrv4-source", "ppocrv3-japan")
    if exact_v4:
        names[:2] = ["ch_PP-OCRv4_det_infer.onnx",
                     "japan_PP-OCRv3_rec_infer.onnx" if "japan" in args.stem
                     else "ch_PP-OCRv4_rec_infer.onnx"]
    source_paths = [args.assets / "ocr_onnx" / name for name in names]
    pins = {row["name"]: row["sha256"] for row in sources["assets"]}
    for path in source_paths:
        if sha256(path) != pins[path.name]:
            raise ValueError(f"source model digest mismatch: {path.name}")
    package = types.ModuleType("validation_source_ppocr")
    package.__path__ = [str(upstream / scoped / "utils/ppocr_utils")]
    sys.modules[package.__name__] = package
    text_system = importlib.import_module(package.__name__ + ".text_system")
    source = ast.parse((upstream / scoped / "ppocr_v4.py").read_text())
    klass = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "PPOCRv4")
    method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
    scope = {"Args": SimpleNamespace, "os": os}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "source_parse_args", "exec"), scope)
    sessions = []
    for path in source_paths:
        sessions.append(source_session(path, args.threads, args.backend in ("cuda", "vulkan")))
    suffix = "v5" if "v5" in args.stem else "v6_" + args.stem.rsplit("-", 1)[1]
    config_name = ("japan_ppocr" if "japan" in args.stem else "ch_ppocr_v4") if exact_v4 else "ch_chinese_cht_en_japan_ppocr_" + suffix
    config = yaml.safe_load((upstream / f"anylabeling/configs/auto_labeling/{config_name}.yaml").read_text())
    dictionary = ("japan_dict.txt" if "japan" in args.stem else
                  "ppocr_keys_v1.txt" if exact_v4 else
                  "ppocrv5_dict.txt" if "v5" in args.stem else
                  "ppocrv6_tiny_dict.txt" if "tiny" in args.stem else "ppocrv6_dict.txt")
    fake = SimpleNamespace(det_net=sessions[0], rec_net=sessions[1], cls_net=sessions[2],
                           drop_score=config.get("drop_score", .5), use_angle_cls=config["use_angle_cls"],
                           rec_char_dict_path=str(upstream / scoped / "configs/ppocr" / dictionary), current_dir=".")
    source_args = scope["parse_args"](fake)
    for key in ("det_db_thresh", "det_db_box_thresh", "det_db_unclip_ratio", "det_db_max_candidates"):
        if key in config: setattr(source_args, key, config[key])
    system = text_system.TextSystem(source_args)
    if not args.ground_truth:
        image_path = args.assets / "ppocr_testdata/tables/ppstructure_table.jpg"
        image = cv2.imread(str(image_path))
        if image is None: raise ValueError("input decode failed")
        quads, texts, _, _ = system(image)
        expected = [{"quad": quad.tolist(), "text": text, "score": float(score)}
                    for quad, (text, score) in zip(quads, texts)]
        if not expected: raise ValueError("source produced no text")
    custom, normalized = None, None
    if args.selected_regions:
        custom = [np.asarray(quad, np.float32).copy() for quad in quads[:8]]
        custom[0][0] += [3, 2]  # skewed four-corner annotation
        custom[1] = np.insert(custom[1], 1, (custom[1][0] + custom[1][1]) / 2 + [0, 2], axis=0)
        custom[2] = custom[2][[0, 1, 3]]  # triangle
        custom += [np.array([[-8.75, -4.25], [175.75, -2.25], [180.5, 65.75], [-5.5, 62.25]], np.float32),
                   np.array([[image.shape[1] - 90, image.shape[0] - 65], [image.shape[1] + 9, image.shape[0] - 60],
                             [image.shape[1] + 7, image.shape[0] + 8], [image.shape[1] - 85, image.shape[0] + 11]], np.float32)]
        custom = custom[::-1]  # input ownership differs from reading order
        integer_custom = [np.trunc(points).astype(np.int32).tolist() for points in custom]
        normalized = system.normalize_custom_detection_boxes(integer_custom)
        _, selected_texts, _, indices = system(image, dt_boxes=integer_custom)
        expected = [None] * len(custom)
        for (text, score), index in zip(selected_texts, indices):
            expected[index] = {"quad": normalized[index].tolist(), "text": text, "score": float(score)}
        if any(row is None for row in expected):
            raise ValueError("source custom workflow did not retain all regions")
    library_path = aicore_library_path(args.library)
    library = C.CDLL(str(library_path))
    library.aicore_ocr_context_create.argtypes = [C.POINTER(Options)]
    library.aicore_ocr_context_create.restype = C.c_void_p
    library.aicore_ocr_context_destroy.argtypes = [C.c_void_p]
    library.aicore_ocr_run.argtypes = [C.c_void_p, C.POINTER(View)]
    library.aicore_ocr_run.restype = C.c_void_p
    if args.selected_regions:
        library.aicore_ocr_recognize_regions.argtypes = [C.c_void_p, C.POINTER(View), C.POINTER(TextRegion), C.c_int32]
        library.aicore_ocr_recognize_regions.restype = C.c_void_p
    library.aicore_ocr_box_count.argtypes = [C.c_void_p]
    library.aicore_ocr_box_at.argtypes = [C.c_void_p, C.c_int]
    library.aicore_ocr_box_at.restype = C.POINTER(Box)
    library.aicore_ocr_result_release.argtypes = [C.c_void_p]
    library.aicore_ocr_last_error.restype = C.c_char_p
    models_dir = args.models_dir or args.assets / "ocr_models"
    models = [models_dir / f"{args.stem}-{role}-{args.quant}.gguf" for role in ("det", "rec")]
    if args.stem == "ppocrv3-japan":
        models[0] = models_dir / f"ppocrv4-source-det-{args.quant}.gguf"
    models.append(models_dir / f"ppocr-cls-{args.quant}.gguf")
    library.aicore_ocr_preset_by_rec_filename.argtypes = [C.c_char_p]
    library.aicore_ocr_preset_by_rec_filename.restype = C.POINTER(Preset)
    preset = library.aicore_ocr_preset_by_rec_filename(models[1].name.encode())
    if not preset:
        raise ValueError("source recognizer has no public task recipe")
    preset = preset.contents
    source_defaults = {
        "det_thresh": source_args.det_db_thresh,
        "det_box_thresh": source_args.det_db_box_thresh,
        "det_unclip_ratio": source_args.det_db_unclip_ratio,
        "det_max_candidates": getattr(source_args, "det_db_max_candidates", 1000),
        "det_max_side": source_args.det_limit_side_len,
        "rec_batch": source_args.rec_batch_num,
        "cls_batch": source_args.cls_batch_num,
        "cls_thresh": source_args.cls_thresh,
        "drop_score": fake.drop_score,
    }
    for key, value in source_defaults.items():
        if abs(getattr(preset, key) - value) > 1e-6:
            raise ValueError(f"public recipe/source default mismatch: {key}")
    for key, path in zip(("det_stem", "rec_stem", "cls_stem"), models):
        if getattr(preset, key).decode() + f"-{args.quant}.gguf" != path.name:
            raise ValueError(f"public recipe/source model pairing mismatch: {key}")
    options = Options()
    options.struct_size, options.abi_version = C.sizeof(options), 1
    options.det, options.rec, options.cls = [str(path).encode() for path in models]
    options.device, options.batch, options.clsbatch = args.backend.encode(), preset.rec_batch, preset.cls_batch
    options.threads = args.threads
    options.maxside, options.limit = preset.det_max_side, 1
    options.thresh, options.boxthresh = preset.det_thresh, preset.det_box_thresh
    options.unclip, options.candidates = preset.det_unclip_ratio, preset.det_max_candidates
    options.clsthresh = preset.cls_thresh
    ctx = library.aicore_ocr_context_create(C.byref(options))
    if not ctx: raise RuntimeError(library.aicore_ocr_last_error().decode())
    if args.ground_truth:
        try:
            return evaluate_labeled_pages(args, system, library, ctx, preset, {
                "backend": args.backend, "quant": args.quant, "source_revision": revision,
                "threads": args.threads,
                "preset_id": preset.id.decode(), "preset_defaults_match": True,
                "source_defaults": source_defaults,
                "source_models": {p.name: sha256(p) for p in source_paths},
                "models": {p.name: sha256(p) for p in models},
                "library_sha256": sha256(library_path),
                "probe_sha256": sha256(Path(__file__)),
                "onnxruntime": ort.__version__,
                "source_provider": sessions[0].get_providers()[0],
                "source_provider_options": [v.get_provider_options() for v in sessions]})
        finally:
            library.aicore_ocr_context_destroy(ctx)
    # Padded BGR rows through the public borrowed-image ABI.
    padded = np.zeros((image.shape[0], image.shape[1] + 7, 3), np.uint8)
    padded[:, :image.shape[1]] = image
    view = View(padded.ctypes.data, image.shape[1], image.shape[0], padded.strides[0], 4)
    attempts = []
    buffers = [np.ascontiguousarray(points, np.float32) for points in custom] if custom is not None else []
    regions = (TextRegion * len(buffers))(*[TextRegion(points.ctypes.data_as(C.POINTER(C.c_float)), len(points)) for points in buffers])
    try:
        for _ in range(2):
            result = (library.aicore_ocr_recognize_regions(ctx, C.byref(view), regions, len(buffers))
                      if args.selected_regions else library.aicore_ocr_run(ctx, C.byref(view)))
            if not result: raise RuntimeError(library.aicore_ocr_last_error().decode())
            try:
                actual = []
                for i in range(library.aicore_ocr_box_count(result)):
                    box = library.aicore_ocr_box_at(result, i).contents
                    if args.selected_regions or box.score >= fake.drop_score:
                        actual.append({"quad": np.array(box.quad).reshape(4, 2).tolist(),
                                       "text": box.text.decode(), "score": float(box.score)})
            finally:
                library.aicore_ocr_result_release(result)
            rows, used = [], set()
            # O(n^2) nearest unused center: independent of source/native box
            # enumeration and sufficient for the bounded 73/81-box fixture.
            for source in expected:
                center = np.array(source["quad"]).mean(0)
                pairs = [(float(np.linalg.norm(np.array(row["quad"]).mean(0) - center)), i, row)
                         for i, row in enumerate(actual) if i not in used and
                         (not args.selected_regions or i == len(rows))]
                if not pairs: break
                distance, i, row = min(pairs)
                used.add(i)
                rows.append({"center_distance": distance, "quad_error": float(np.max(np.abs(np.array(source["quad"]) - row["quad"]))),
                             "source": source["text"], "native": row["text"], "same_text": source["text"] == row["text"]})
            attempts.append({"source_count": len(expected), "native_count": len(actual), "rows": rows})
        if args.selected_regions:
            # Real loaded context: malformed polygons must fail before any
            # float-to-int cast; degenerate valid polygons preserve ownership.
            for points, count in ((np.array([[0, 0], [1, 1]], np.float32), 2),
                                  (np.array([[0, 0], [1, 1], [np.nan, 2]], np.float32), 3),
                                  (np.array([[0, 0], [1, 1], [2**31, 2]], np.float32), 3)):
                invalid = TextRegion(points.ctypes.data_as(C.POINTER(C.c_float)), count)
                if library.aicore_ocr_recognize_regions(ctx, C.byref(view), C.byref(invalid), 1):
                    raise ValueError("malformed selected region accepted")
            points = np.zeros((3, 2), np.float32)
            degenerate = TextRegion(points.ctypes.data_as(C.POINTER(C.c_float)), 3)
            result = library.aicore_ocr_recognize_regions(ctx, C.byref(view), C.byref(degenerate), 1)
            if not result: raise ValueError("degenerate region lost its result slot")
            try:
                box = library.aicore_ocr_box_at(result, 0).contents
                if library.aicore_ocr_box_count(result) != 1 or box.text or box.score != 0:
                    raise ValueError("degenerate selected region contract changed")
            finally:
                library.aicore_ocr_result_release(result)
    finally:
        library.aicore_ocr_context_destroy(ctx)
    metrics = {"missing_or_extra": max(abs(a["source_count"] - a["native_count"]) for a in attempts),
               "source_count": len(expected), "repeats": len(attempts),
               "text_mismatches": max(sum(not row["same_text"] for row in a["rows"]) for a in attempts),
               "quad_error": max(row["quad_error"] for a in attempts for row in a["rows"])}
    report = {**metrics, "attempts": attempts, "backend": args.backend, "source_revision": revision,
              "threads": args.threads,
              "source_provider": sessions[0].get_providers()[0],
              "source_provider_options": [v.get_provider_options() for v in sessions],
              "preset_id": preset.id.decode(), "preset_defaults_match": True,
              "source_defaults": source_defaults,
              "input_sha256": sha256(image_path), "source_models": {p.name: sha256(p) for p in source_paths},
              "models": {p.name: sha256(p) for p in models}, "onnxruntime": ort.__version__,
              "scope": "selected polygons from one real page vs pinned source" if args.selected_regions else "one real page vs pinned source; not dataset accuracy or release performance"}
    if args.selected_regions:
        report["custom_regions"] = [points.tolist() for points in buffers]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"OCR source pipeline report: {args.report}")
    return int(metrics["missing_or_extra"] != 0 or metrics["text_mismatches"] != 0 or
               metrics["quad_error"] > (0.001 if args.selected_regions else 2))


if __name__ == "__main__":
    raise SystemExit(main())
