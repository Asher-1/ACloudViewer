#!/usr/bin/env python3
"""Export layout regions through the public borrowed-image OCR C ABI.

The existing API requires det/rec models. For layout accuracy isolation only,
DB threshold=1 suppresses text regions; layout still consumes the original
full-resolution image with its unchanged score=.5/NMS=.5 policy. Assert zero
text boxes rather than silently measuring a different workflow. No timing or
OCR pipeline acceptance is inferred from this accuracy-only configuration.
"""

import argparse
import ctypes as C
import json
from pathlib import Path

import cv2
import numpy as np

from layout_gt_eval import CDLA_LABELS, digest, verified_path
from run_ocr_source_pipeline import Options, View, aicore_library_path


class Region(C.Structure):
    _fields_ = [("class_id", C.c_int32), ("reading_order", C.c_int32),
                ("quad", C.c_float * 8), ("score", C.c_float)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("library", "det", "rec", "layout", "image-root", "pins", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    pins = json.loads(args.pins.read_text())
    if pins["labels"] != list(CDLA_LABELS) or not pins["pages"]:
        raise ValueError("invalid CDLA selection/label mapping")
    names = [page["image"]["path"] for page in pins["pages"]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate CDLA page")
    dependencies = {str(path): digest(path) for path in (Path(__file__), args.library,
        args.det, args.rec, args.layout, args.pins, Path(digest.__code__.co_filename),
        Path(__file__).with_name("run_ocr_source_pipeline.py"))}
    args.library = aicore_library_path(args.library)
    library = C.CDLL(str(args.library))
    for name, inputs, output in (
        ("context_create", [C.POINTER(Options)], C.c_void_p),
        ("context_destroy", [C.c_void_p], None),
        ("context_device", [C.c_void_p], C.c_char_p),
        ("context_ready", [C.c_void_p], C.c_int),
        ("label_count", [C.c_void_p], C.c_int),
        ("label_at", [C.c_void_p, C.c_int], C.c_char_p),
        ("run", [C.c_void_p, C.POINTER(View)], C.c_void_p),
        ("box_count", [C.c_void_p], C.c_int),
        ("region_count", [C.c_void_p], C.c_int),
        ("region_at", [C.c_void_p, C.c_int], C.POINTER(Region)),
        ("result_release", [C.c_void_p], None),
        ("last_error", [], C.c_char_p)):
        function = getattr(library, "aicore_ocr_" + name)
        function.argtypes, function.restype = inputs, output
    options = Options()
    options.struct_size, options.abi_version = C.sizeof(options), 1
    options.det, options.rec, options.layout = [str(path.resolve()).encode()
                                                for path in (args.det, args.rec, args.layout)]
    options.device, options.threads = args.backend.encode(), args.threads
    options.maxside, options.limit, options.thresh, options.boxthresh = 32, 1, 1, 1
    ctx = library.aicore_ocr_context_create(C.byref(options))
    if not ctx:
        raise RuntimeError(library.aicore_ocr_last_error().decode())
    try:
        device = library.aicore_ocr_context_device(ctx).decode()
        if not library.aicore_ocr_context_ready(ctx) or not device.lower().startswith(args.backend):
            raise ValueError("layout context not ready or requested backend was not selected")
        labels = [library.aicore_ocr_label_at(ctx, i).decode()
                  for i in range(library.aicore_ocr_label_count(ctx))]
        if labels != list(CDLA_LABELS):
            raise ValueError("GGUF/native layout class mapping differs from CDLA")
        report = dict(backend=args.backend, device=device, dependencies=dependencies,
            pins_sha256=digest(args.pins), score_threshold=.5, nms_threshold=.5,
            images=[], scope="Layout page accuracy isolation only; no OCR, timing or release claim",
            detector=dict(max_side=32, limit_type="max", binarization_threshold=1,
                          box_score_threshold=1, required_text_boxes=0))
        for index, page in enumerate(pins["pages"]):
            path = verified_path(args.image_root, page["image"])
            verified_path(args.image_root, page["annotation"])
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"cannot decode {path.name}")
            height, width = image.shape[:2]
            padded = np.zeros((height, width + 7, 3), dtype=np.uint8)
            padded[:, :width] = image
            view = View(padded.ctypes.data, width, height, padded.strides[0], 4)
            result = library.aicore_ocr_run(ctx, C.byref(view))
            if not result:
                raise RuntimeError(library.aicore_ocr_last_error().decode())
            try:
                if library.aicore_ocr_box_count(result) != 0:
                    raise ValueError("text detector was not suppressed for layout isolation")
                error = library.aicore_ocr_last_error()
                if error:
                    raise RuntimeError(error.decode())
                regions = []
                for i in range(library.aicore_ocr_region_count(result)):
                    row = library.aicore_ocr_region_at(result, i).contents
                    regions.append(dict(class_id=row.class_id, score=float(row.score),
                        bbox=[min(row.quad[::2]), min(row.quad[1::2]),
                              max(row.quad[::2]), max(row.quad[1::2])]))
                report["images"].append(dict(image=path.name, sha256=page["image"]["sha256"],
                    width=width, height=height, regions=regions))
            finally:
                library.aicore_ocr_result_release(result)
            if (index + 1) % 50 == 0:
                print(json.dumps(dict(pages=index + 1, planned=len(pins["pages"]))), flush=True)
    finally:
        library.aicore_ocr_context_destroy(ctx)
    if any(digest(Path(path)) != sha for path, sha in dependencies.items()):
        raise ValueError("layout consumer execution artifact changed during export")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
