#!/usr/bin/env python3
"""Labelled-page checks for published OCR weights against PaddleOCR 3.0.1.

The common validation runner provisions every source/config/GT file. This
probe never downloads models and never substitutes a same-family checkpoint.
PaddleOCR owns detection, crops, batching, decoding and the complete pipeline.
It is an accuracy probe; it does not measure throughput or memory.
"""

import argparse
import ctypes as C
import gc
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace

from gen_ocr_fixtures import Gguf
from ocr_paddle_environment import source_environment
from run_ocr_source_pipeline import (Box, Options, View, aicore_library_path,
                                     evaluate_labeled_pages, sha256)


class PaddleReference:
    """Adapt the pinned official pipeline to the shared GT metric consumer."""

    def __init__(self, ocr):
        self.ocr = ocr
        self.pipeline = ocr.paddlex_pipeline
        self.drop_score = 0.0
        self._last_image = None
        self._last_result = None

    def __call__(self, image, dt_boxes=None):
        if dt_boxes is None:
            result = self.ocr.predict(image, text_rec_score_thresh=self.drop_score)[0]
            self._last_image, self._last_result = image, result
            return result["rec_polys"], list(zip(result["rec_texts"], result["rec_scores"])), None, None
        # Use the exact official crop and recognizer objects already owned by
        # this pipeline, including its dictionary and six-crop padding groups.
        crops = list(self.pipeline._crop_by_polys(image, dt_boxes))
        indices = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / crops[i].shape[0])
        results = list(self.pipeline.text_rec_model([crops[i] for i in indices]))
        if len(results) != len(indices):
            raise ValueError("official Paddle recognizer lost fixed GT regions")
        texts = [(r["rec_text"], float(r["rec_score"])) for r in results]
        return [dt_boxes[i] for i in indices], texts, None, indices

    def text_detector(self, image):
        if image is not self._last_image:
            self(image)
        return self._last_result["dt_polys"]


def verified_source(root, pins, role, model):
    """Verify independent source files and the GGUF conversion identity."""
    row = pins[role]
    files = {}
    for key in ("program", "parameters", "config"):
        entry = row[key]
        if key == "config" and "archive_relative_path" in entry:
            archive = root / entry["archive_relative_path"]
            if sha256(archive) != entry["archive_sha256"]:
                raise ValueError(f"Paddle config archive digest mismatch: {role}")
            # The YAML is metadata inside a pinned official archive. Read the
            # exact regular member into memory; never extract untrusted paths.
            with tarfile.open(archive) as stream:
                member = stream.getmember(entry["member"])
                if not member.isfile() or member.size != entry["size_bytes"]:
                    raise ValueError(f"invalid Paddle config archive member: {role}")
                payload = stream.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError(f"Paddle source digest mismatch: {role}/{key}")
            files[key] = payload
            continue
        path = root / entry["relative_path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Paddle source digest mismatch: {role}/{key}")
        files[key] = path
    source = Gguf(str(model)).kv
    if (source.get("ocr.source.name") != row["name"] or
            source.get("ocr.source.sha256") != row["parameters"]["sha256"]):
        raise ValueError(f"GGUF/Paddle source identity mismatch: {model.name}")
    return files


def source_digests(sources):
    return {role: {key: hashlib.sha256(value).hexdigest() if isinstance(value, bytes)
                  else sha256(value) for key, value in files.items()}
            for role, files in sources.items()}


def reference_identity():
    import paddle
    import platform
    from ocr_paddle_metadata import distribution_environment
    environment = distribution_environment()
    return dict(source_environment=environment["environment"],
        source_distribution_candidates=environment["distribution_candidates"],
        source_distribution_identity_sha256=sha256(
            Path(distribution_environment.__code__.co_filename)),
        source_python_version=platform.python_version(),
        source_library_sha256=sha256(Path(paddle.base.libpaddle.__file__)),
        source_isolation_sha256=sha256(Path(source_environment.__code__.co_filename)),
        logging_environment_keys=sorted(k for k in os.environ
                                        if k.startswith(("GLOG_", "MLOG_"))))


def validate_reference(args, result, sources):
    import ocr_gt_eval as gt
    from ocr_paddle_metadata import distribution_environment
    code = result.get("source_process_returncode")
    if (type(code) is not int or code != 0 or
            result["ground_truth_sha256"] != args.ground_truth_sha256 or
            result["source_pins_sha256"] != sha256(args.source_pins) or
            result["probe_sha256"] != sha256(Path(__file__)) or
            result["source_files"] != source_digests(sources) or
            result.get("source_distribution_identity_sha256") != sha256(
                Path(distribution_environment.__code__.co_filename)) or
            not result.get("source_distribution_candidates") or
            result.get("source_evaluation") != args.evaluation or
            result.get("source_language") != args.language or
            result.get("source_threads") != args.threads or
            result.get("source_device", "cpu") != getattr(args, "source_device", "cpu") or
            (getattr(args, "source_device", "cpu") == "cuda" and
             result.get("source_tf32") is not False)):
        raise ValueError("independent Paddle reference identity mismatch")
    if gt.ground_truth_digest(args.ground_truth) != args.ground_truth_sha256:
        raise ValueError("ground-truth SHA-256 mismatch")
    names = [page["image"] for page in result["pages"]]
    if len(set(names)) != len(names) or set(names) != set(gt.load_ground_truth(args.ground_truth)):
        raise ValueError("independent Paddle reference page coverage mismatch")
    for page in result["pages"]:
        if sha256(args.image_root / page["image"]) != page["image_sha256"]:
            raise ValueError("source image changed before native evaluation")


def export_reference(args, pins, sources):
    """Run Paddle alone: its native libraries never share an AICore process."""
    import cv2
    import ocr_gt_eval as gt
    from ocr_paddle_metadata import distribution_environment
    code_paths = (Path(__file__), Path(source_environment.__code__.co_filename),
                  Path(distribution_environment.__code__.co_filename))
    code_before = {str(path): sha256(path) for path in code_paths}
    # Paddle 3.0 has no allow_tf32 flag; match the independent graph probe.
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    import paddle
    versions = {name: importlib.metadata.version(name) for name in ("paddleocr", "paddlex")}
    versions["paddlepaddle"] = paddle.__version__
    if args.source_device == "cuda":
        if not paddle.is_compiled_with_cuda():
            raise ValueError("official Paddle page interpreter has no CUDA support")
        paddle.set_device("gpu:0")
    if any(versions[name] != version for name, version in
           (("paddleocr", "3.0.1"), ("paddlex", "3.0.1"), ("paddlepaddle", "3.0.0"))):
        raise ValueError(f"official Paddle reference version mismatch: {versions}")
    if gt.ground_truth_digest(args.ground_truth) != args.ground_truth_sha256:
        raise ValueError("ground-truth SHA-256 mismatch")
    if args.source_reference:
        if sha256(args.source_reference) != args.source_reference_sha256:
            raise ValueError("replayed source reference SHA-256 mismatch")
        result = json.loads(args.source_reference.read_text())
        validate_reference(args, result, sources)
        current = reference_identity()
        if any(result.get(key) != value for key, value in current.items()):
            raise ValueError("replayed source environment changed")
        if result["source_versions"] != versions:
            raise ValueError("replayed source framework version changed")
        result["replayed_source_sha256"] = args.source_reference_sha256
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return 0
    with tempfile.TemporaryDirectory(prefix="ocr-paddle-source-") as temporary:
        dirs = {}
        for role, files in sources.items():
            target = Path(temporary) / role
            target.mkdir()
            for key, name in (("program", "inference.pdmodel"), ("parameters", "inference.pdiparams")):
                (target / name).symlink_to(files[key].resolve())
            config = files["config"]
            (target / "inference.yml").write_bytes(config if isinstance(config, bytes) else config.read_bytes())
            dirs[role] = str(target)
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(text_detection_model_name="PP-OCRv4_mobile_det",
            text_detection_model_dir=dirs["det_ch"],
            text_recognition_model_name=pins["rec_" + args.language]["official_model"],
            text_recognition_model_dir=dirs["rec_" + args.language],
            use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_textline_orientation=False, enable_mkldnn=False,
            cpu_threads=args.threads, device="gpu:0" if args.source_device == "cuda" else "cpu")
        if args.source_device == "cuda" and not paddle.get_device().startswith("gpu:"):
            raise ValueError("official Paddle page pipeline silently fell back to CPU")
        pipeline = ocr.paddlex_pipeline
        defaults = {name: getattr(pipeline, "text_det_" + name) for name in
                    ("limit_side_len", "limit_type", "max_side_limit", "thresh", "box_thresh", "unclip_ratio")}
        expected = dict(limit_side_len=64, limit_type="min", max_side_limit=4000,
                        thresh=.3, box_thresh=.6, unclip_ratio=1.5)
        if defaults != expected or pipeline.text_rec_score_thresh != 0:
            raise ValueError(f"official/default C ABI contract mismatch: {defaults}")
        reference = PaddleReference(ocr)
        identity = reference_identity()
        pages = []
        for name, labels in gt.load_ground_truth(args.ground_truth).items():
            path = args.image_root / name
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"image decode failed: {path}")
            e2e, detection = ([], []), []
            if args.evaluation == "pipeline":
                quads, texts, _, _ = reference(image)
                e2e = ([q.tolist() for q in quads], [(t, float(s)) for t, s in texts])
                detection = [q.tolist() for q in reference.text_detector(image)]
            boxes = [[list(p) for p in label.points] for label in labels if not label.ignore]
            texts, indices = [], []
            if boxes:
                _, texts, _, indices = reference(image, boxes)
            pages.append(dict(image=name, image_sha256=sha256(path),
                decoded_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                detection=detection, e2e=e2e, regions=boxes, texts=texts, indices=indices))
            print(f"Paddle labelled pages: {len(pages)} {name}", flush=True)
        # Explicit cleanup is checked by the child's successful exit. A crash
        # after writing predictions remains a failed reference execution.
        del reference, pipeline, ocr
        gc.collect()
    if (identity != reference_identity() or
            code_before != {str(path): sha256(path) for path in code_paths}):
        raise ValueError("Paddle page reference environment/code changed during export")
    result = dict(pages=pages, source_versions=versions, source_defaults=defaults,
        source_evaluation=args.evaluation, source_language=args.language,
        source_threads=args.threads, source_device=args.source_device, source_tf32=False,
        **identity,
        source_files=source_digests(sources),
        ground_truth_sha256=args.ground_truth_sha256,
        source_pins_sha256=sha256(args.source_pins), probe_sha256=sha256(Path(__file__)))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return 0


class ReplayedReference:
    """Consume source predictions only after the independent process exits 0."""

    def __init__(self, result):
        self.pages = {}
        for page in result["pages"]:
            self.pages.setdefault(page["decoded_sha256"], []).append(page)
        self.drop_score = 0.0

    def page(self, image):
        return self.pages[hashlib.sha256(image.tobytes()).hexdigest()][0]

    def __call__(self, image, dt_boxes=None):
        page = self.page(image)
        if dt_boxes is None:
            quads, texts = page["e2e"]
            return quads, texts, None, None
        boxes = [[list(point) for point in quad] for quad in dt_boxes]
        matches = [p for p in self.pages[hashlib.sha256(image.tobytes()).hexdigest()]
                   if p["regions"] == boxes]
        if not matches:
            raise ValueError("replayed source GT-region mismatch")
        page = matches[0]
        return [dt_boxes[i] for i in page["indices"]], page["texts"], None, page["indices"]

    def text_detector(self, image):
        return self.page(image)["detection"]


class PaddleRegionAPI:
    """Use Paddle's existing quad API in the shared GT metric consumer.

    recognize_regions explicitly implements X-AnyLabeling's selected-polygon
    convention. PaddleX recognition instead fits and orders each quad like
    its detector path, which is the public recognize() contract.
    """

    def __init__(self, library):
        self.library = library
        native = library.aicore_ocr_recognize
        native.argtypes = [C.c_void_p, C.POINTER(View), C.POINTER(C.c_float), C.c_int32]
        native.restype = C.c_void_p

        def recognize(ctx, view, regions, count):
            values = []
            image = C.cast(view, C.POINTER(View)).contents
            for i in range(count):
                if regions[i].point_count != 4:
                    raise ValueError("Paddle fixed-region gate requires quad GT")
                quad = [regions[i].points_xy[j] for j in range(8)]
                # recognize() clips to the image; never silently compare it
                # with Paddle's unbounded crops on an out-of-image label.
                if any(not 0 <= v <= (image.width if j % 2 == 0 else image.height)
                       for j, v in enumerate(quad)):
                    raise ValueError("Paddle fixed-region GT exceeds the public quad API bounds")
                values.extend(quad)
            quads = (C.c_float * len(values))(*values)
            return native(ctx, view, quads, count)

        self.aicore_ocr_recognize_regions = recognize

    def __getattr__(self, name):
        return getattr(self.library, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--source-pins", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--language", choices=("ch", "en"), required=True)
    parser.add_argument("--quant", choices=("f32", "f16", "q8_0"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--ground-truth-sha256", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--evaluation", choices=("pipeline", "recognition"), default="pipeline")
    parser.add_argument("--source-python", type=Path,
                        help="isolated Paddle 3.0.0/PaddleOCR 3.0.1 interpreter")
    parser.add_argument("--source-gpu-python", type=Path,
                        help="isolated original Paddle CUDA interpreter for GPU page comparisons")
    parser.add_argument("--source-reference", type=Path,
                        help="reuse a completed, SHA-pinned source page export")
    parser.add_argument("--source-reference-sha256")
    parser.add_argument("--export-reference", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.threads == 0:
        args.threads = 4
    if args.threads < 1:
        parser.error("--threads must be nonnegative")
    if bool(args.source_reference) != bool(args.source_reference_sha256):
        parser.error("source reference and SHA-256 must be supplied together")
    args.source_device = "cuda" if args.backend in ("cuda", "vulkan") else "cpu"
    if args.source_device == "cuda":
        if not args.source_gpu_python or not args.source_gpu_python.is_file():
            parser.error("GPU pages require --source-gpu-python with original Paddle CUDA")
        args.source_python = args.source_gpu_python
    pins = json.loads(args.source_pins.read_text())
    det = args.assets / "ocr_models" / f"ppocr-det-ch-{args.quant}.gguf"
    rec = args.assets / "ocr_models" / f"ppocr-rec-{args.language}-{args.quant}.gguf"
    sources = {role: verified_source(args.source_root, pins, role, model)
               for role, model in (("det_ch", det), ("rec_" + args.language, rec))}
    if args.export_reference:
        return export_reference(args, pins, sources)
    if not args.source_python or not args.source_python.is_file():
        parser.error("--source-python must name the isolated official Paddle interpreter")
    source_path = args.report.with_suffix(".paddle.json")
    source_path.parent.mkdir(parents=True, exist_ok=True)
    if args.source_reference and args.source_reference.resolve() == source_path.resolve():
        parser.error("source reference must be distinct from the current output")
    source_path.unlink(missing_ok=True)
    environment = source_environment(args.source_python)
    command = [str(args.source_python), str(Path(__file__).resolve()), *sys.argv[1:],
               "--export-reference", "--report", str(source_path)]
    with source_path.with_suffix(".log").open("w") as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    source_result = json.loads(source_path.read_text())
    # Only the parent can stamp successful teardown. An export written by a
    # child that subsequently aborts is never eligible for reference replay.
    source_result["source_process_returncode"] = 0
    source_path.write_text(json.dumps(source_result, ensure_ascii=False, indent=2) + "\n")
    validate_reference(args, source_result, sources)
    actual_sources = source_digests(sources)
    if source_result["source_files"] != actual_sources:
        raise ValueError("Paddle source changed before native evaluation")
    args.library = aicore_library_path(args.library)
    library = C.CDLL(str(args.library))
    library.aicore_ocr_context_create.argtypes = [C.POINTER(Options)]
    library.aicore_ocr_context_create.restype = C.c_void_p
    library.aicore_ocr_context_destroy.argtypes = [C.c_void_p]
    library.aicore_ocr_run.argtypes = [C.c_void_p, C.POINTER(View)]
    library.aicore_ocr_run.restype = C.c_void_p
    library.aicore_ocr_box_count.argtypes = [C.c_void_p]
    library.aicore_ocr_box_at.argtypes = [C.c_void_p, C.c_int]
    library.aicore_ocr_box_at.restype = C.POINTER(Box)
    library.aicore_ocr_result_release.argtypes = [C.c_void_p]
    library.aicore_ocr_last_error.restype = C.c_char_p
    options = Options()
    options.struct_size, options.abi_version = C.sizeof(options), 1
    options.det, options.rec = str(det).encode(), str(rec).encode()
    options.device, options.threads = args.backend.encode(), args.threads
    ctx = library.aicore_ocr_context_create(C.byref(options))
    if not ctx:
        raise RuntimeError(library.aicore_ocr_last_error().decode())
    try:
        return evaluate_labeled_pages(args, ReplayedReference(source_result), PaddleRegionAPI(library), ctx,
                SimpleNamespace(drop_score=0.0), {
                    "backend": args.backend, "quant": args.quant, "language": args.language,
                    "models": {p.name: sha256(p) for p in (det, rec)},
                    "source_files": actual_sources,
                    "source_pins_sha256": sha256(args.source_pins),
                    "source_versions": source_result["source_versions"],
                    "source_environment": source_result["source_environment"],
                    "source_distribution_candidates": source_result["source_distribution_candidates"],
                    "source_distribution_identity_sha256": source_result["source_distribution_identity_sha256"],
                    "source_python_version": source_result["source_python_version"],
                    "source_library_sha256": source_result["source_library_sha256"],
                    "replayed_source_sha256": source_result.get("replayed_source_sha256"),
                    "source_defaults": source_result["source_defaults"],
                    "fixed_region_native_api": "aicore_ocr_recognize (Paddle quad contract; in-image GT)",
                    "independent_source_report_sha256": sha256(source_path),
                    "library_sha256": sha256(args.library),
                    "probe_sha256": sha256(Path(__file__)),
                    "metric_consumer_sha256": sha256(Path(__file__).with_name("run_ocr_source_pipeline.py")),
                    "scope": "official same-weight PaddleOCR labelled accuracy; no performance claim"})
    finally:
        library.aicore_ocr_context_destroy(ctx)


if __name__ == "__main__":
    raise SystemExit(main())
