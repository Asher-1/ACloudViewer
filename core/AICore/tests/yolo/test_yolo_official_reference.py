#!/usr/bin/env python3
"""Official PyTorch checkpoint truth with identical pixels/canvas, not a CPU-backend substitute."""
from __future__ import annotations

import argparse
import ctypes as c
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


class ImageView(c.Structure):
    _fields_ = [("data", c.POINTER(c.c_uint8)), ("width", c.c_int32),
                ("height", c.c_int32), ("stride", c.c_size_t), ("format", c.c_int)]


class Detection(c.Structure):
    _fields_ = [("x1", c.c_float), ("y1", c.c_float), ("x2", c.c_float),
                ("y2", c.c_float), ("score", c.c_float), ("class_id", c.c_int32)]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def runtime_library(build):
    """Select runtime libraries only from the specified build, with no host-install fallback."""
    for directory in (build / "bin", build / "bin/Release"):
        for name in ("libAICore.so", "libAICore.dylib", "AICore.dll"):
            path = directory / name
            if path.is_file():
                return path
    raise FileNotFoundError(f"AICore runtime unavailable in {build}")


def source_identity(source, expected_revision):
    """Audit consumed Python sources; record ggml diffs not consumed by PyTorch separately."""
    git = ["git", "-C", str(source)]
    revision = subprocess.check_output(git + ["rev-parse", "HEAD"]).decode().strip()
    if revision != expected_revision:
        raise RuntimeError("official source revision mismatch")
    consumed = subprocess.check_output(
        git + ["status", "--porcelain", "--untracked-files=all", "--", "ultralytics"])
    # Python caches are not sources; added .py/.yaml files can change loading behavior and must also be rejected.
    changes = [line for line in consumed.decode().splitlines()
               if "__pycache__/" not in line and not line.endswith((".pyc", ".pyo"))]
    if changes:
        raise RuntimeError("modified official Python reference: " + "; ".join(changes))
    status = subprocess.check_output(git + ["status", "--porcelain", "--untracked-files=no"])
    submodules = subprocess.check_output(git + ["submodule", "status", "--recursive"])
    delta = subprocess.check_output(git + ["diff", "HEAD", "--binary", "--submodule=diff"])
    return {"source_revision": revision, "consumed_python_clean": True,
            "working_tree_status": status.decode(), "submodule_status": submodules.decode(),
            "working_tree_diff_sha256": hashlib.sha256(delta).hexdigest()}


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * \
                   max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + \
            max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def compare(reference, candidate):
    """O(R*C): match small detection sets by same-class IoU, preserving per-box errors that averages could hide."""
    used = set()
    rows = []
    for index, expected in enumerate(reference):
        choices = [(overlap(expected, actual), j) for j, actual in enumerate(candidate)
                   if j not in used and int(actual[5]) == int(expected[5])]
        iou, match = max(choices, default=(0.0, -1))
        if match < 0 or iou < 0.9:
            rows.append({"reference_index": index, "matched": False})
            continue
        used.add(match)
        actual = candidate[match]
        center = max(abs((expected[0] + expected[2] - actual[0] - actual[2]) / 2),
                     abs((expected[1] + expected[3] - actual[1] - actual[3]) / 2))
        rows.append({"reference_index": index, "candidate_index": match,
                     "matched": True, "iou": iou,
                     "score_error": abs(expected[4] - actual[4]),
                     "center_error_px": center})
    matches = [row for row in rows if row["matched"]]
    return {"reference_count": len(reference), "candidate_count": len(candidate),
            "unmatched_reference": len(reference) - len(matches),
            "unmatched_candidate": len(candidate) - len(used),
            "minimum_iou": min((row["iou"] for row in matches), default=0.0),
            "maximum_score_error": max((row["score_error"] for row in matches), default=0.0),
            "maximum_center_error_px": max((row["center_error_px"] for row in matches), default=0.0),
            "per_detection": rows}


def bind(library):
    signatures = {
        "options_new": (c.c_void_p, []),
        "options_free": (None, [c.c_void_p]),
        "options_set_device": (None, [c.c_void_p, c.c_char_p]),
        "options_set_threads": (None, [c.c_void_p, c.c_int]),
        "options_set_conf_thres": (None, [c.c_void_p, c.c_float]),
        "options_set_iou_thres": (None, [c.c_void_p, c.c_float]),
        "options_set_input_size": (None, [c.c_void_p, c.c_int, c.c_int]),
        "load_opts": (c.c_void_p, [c.c_char_p, c.c_void_p]),
        "free": (None, [c.c_void_p]),
        "last_error": (c.c_char_p, [c.c_void_p]),
        "context_device": (c.c_char_p, [c.c_void_p]),
        "detect_image": (c.c_int, [c.c_void_p, c.POINTER(ImageView)]),
        "detection_count": (c.c_int, [c.c_void_p]),
        "detection_at": (Detection, [c.c_void_p, c.c_int]),
    }
    for name, (result, arguments) in signatures.items():
        symbol = getattr(library, "aicore_yolo_" + name)
        symbol.restype, symbol.argtypes = result, arguments
    # Frozen old libraries validate the original mode only; a missing API does not mean mode 1 took effect.
    for name, arguments in (
            ('options_set_attention_accumulation', [c.c_void_p, c.c_int]),
            ('context_attention_accumulation', [c.c_void_p])):
        symbol = getattr(library, 'aicore_yolo_' + name, None)
        if symbol is not None:
            symbol.restype, symbol.argtypes = c.c_int, arguments
    return library


def infer(lib, args, rgb):
    view = ImageView(rgb.ctypes.data_as(c.POINTER(c.c_uint8)), rgb.shape[1],
                     rgb.shape[0], rgb.strides[0], 1)
    options = lib.aicore_yolo_options_new()
    if not options:
        raise RuntimeError("cannot allocate YOLO options")
    context = None
    try:
        lib.aicore_yolo_options_set_device(options, args.backend.encode())
        lib.aicore_yolo_options_set_threads(options, args.threads)
        lib.aicore_yolo_options_set_conf_thres(options, args.confidence)
        lib.aicore_yolo_options_set_iou_thres(options, 0.7)
        lib.aicore_yolo_options_set_input_size(options, args.size, args.size)
        requested_mode = getattr(args, 'attention_accumulation', 0)
        setter = getattr(lib, 'aicore_yolo_options_set_attention_accumulation', None)
        getter = getattr(lib, 'aicore_yolo_context_attention_accumulation', None)
        if setter is None or getter is None:
            if requested_mode != 0:
                raise RuntimeError('runtime does not support explicit attention accumulation')
        elif setter(options, requested_mode) != 0:
            raise RuntimeError('runtime rejected attention accumulation mode')
        context = lib.aicore_yolo_load_opts(str(args.model).encode(), options)
        if not context:
            raise RuntimeError(lib.aicore_yolo_last_error(None).decode())
        configured_mode = getter(context) if getter is not None else 0
        if configured_mode != requested_mode:
            raise RuntimeError('loaded context attention accumulation mismatch')
        device = lib.aicore_yolo_context_device(context).decode()
        if args.backend.lower() not in device.lower():
            raise RuntimeError(f"requested {args.backend}, resolved {device}")
        hashes, durations, first = [], [], None
        for iteration in range(args.warmups + args.repeats):
            start = time.perf_counter()
            if lib.aicore_yolo_detect_image(context, c.byref(view)) < 0:
                raise RuntimeError(lib.aicore_yolo_last_error(context).decode())
            detections = []
            for index in range(lib.aicore_yolo_detection_count(context)):
                row = lib.aicore_yolo_detection_at(context, index)
                detections.append([row.x1, row.y1, row.x2, row.y2, row.score, row.class_id])
            if iteration >= args.warmups:
                durations.append((time.perf_counter() - start) * 1000)
                hashes.append(hashlib.sha256(json.dumps(detections).encode()).hexdigest())
                if first is None:
                    first = detections
        return first, device, hashes, durations, configured_mode
    finally:
        lib.aicore_yolo_free(context)
        lib.aicore_yolo_options_free(options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    runtime = parser.add_mutually_exclusive_group(required=True)
    runtime.add_argument("--library", type=Path)
    runtime.add_argument("--build", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--attention-accumulation", type=int, choices=(0, 1), default=0,
                        help="PSA accumulation request: 0 retains upstream, 1 requests F32; storage types are unchanged")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--canonical-canvas", action="store_true",
                        help="Diagnostic: both sides consume the same official LetterBox uint8 canvas, isolating resize differences")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    if not args.source:
        print("official source unavailable: configure --reference ultralytics=PATH", file=sys.stderr)
        return 77
    args.source = Path(args.source)
    if args.build:
        args.library = runtime_library(args.build)
    for path in (args.library, args.model, args.image, args.checkpoint, args.source):
        if not path.exists():
            print(f"reference asset unavailable: {path}", file=sys.stderr)
            return 77
    if args.repeats < 1 or args.warmups < 0 or args.size <= 0:
        parser.error("invalid repeat/warmup/canvas size")
    identity = source_identity(args.source, args.source_revision)
    actual_revision = identity["source_revision"]
    if digest(args.checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("official checkpoint pin mismatch")
    sys.path.insert(0, str(args.source.resolve()))
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO
    import ultralytics
    if not Path(ultralytics.__file__).resolve().is_relative_to(args.source.resolve()):
        raise RuntimeError("wrong ultralytics module loaded")
    rgb = np.asarray(Image.open(args.image).convert("RGB")).copy()
    decoded_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
    if args.canonical_canvas:
        from ultralytics.data.augment import LetterBox
        rgb = LetterBox(args.size, auto=True, stride=32)(image=rgb)
    # Identical decoded pixels; AICore LetterBox uses auto=True (stride-reduced
    # padding), so input_size does not force non-square images square. Official inference also requires rect=True.
    model = YOLO(str(args.checkpoint))
    official = model.predict(
        source=rgb[:, :, ::-1].copy(), imgsz=args.size, rect=True,
        conf=args.confidence, iou=0.7, max_det=300, device="cpu", verbose=False)[0]
    reference = official.boxes.data.cpu().numpy().astype(float).tolist()
    if not reference:
        raise RuntimeError("empty official detection reference")
    lib = bind(c.CDLL(str(args.library.resolve())))
    candidate, device, hashes, times, configured_mode = infer(lib, args, rgb)
    quant = next((q for q in ("f32", "f16", "q8_0")
                  if f"-{q}." in args.model.name.lower()), None)
    if quant is None:
        raise RuntimeError("model quantization must be explicit: f32/f16/q8_0")
    report = compare(reference, candidate)
    report.update({"reference_id": f"ultralytics@{actual_revision}:{args.checkpoint_sha256}",
                   "source_provenance": identity,
                   "runtime_sha256": digest(args.library),
                   "model_sha256": digest(args.model), "image_sha256": digest(args.image),
                   "decoded_sha256": decoded_hash,
                   "pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                   "input_mode": "canonical_canvas" if args.canonical_canvas else "decoded_image",
                   "device": device, "quantization": quant,
                   "attention_accumulation_requested": args.attention_accumulation,
                   "attention_accumulation_configured": configured_mode,
                   "attention_accumulation": "f32" if configured_mode == 1 else "upstream",
                   "attention_options_api": hasattr(lib, 'aicore_yolo_options_set_attention_accumulation'),
                   "requested_size": args.size, "letterbox_auto": True,
                   "canvas": list(model.predictor.pre_transform([rgb[:, :, ::-1]])[0].shape[:2]),
                   "confidence": args.confidence, "warmups": args.warmups,
                   "repeats": args.repeats, "token_hash": hashes[0],
                   "stable": len(set(hashes)) == 1,
                   "e2e_p50_ms": float(np.percentile(times, 50)),
                   "e2e_p95_ms": float(np.percentile(times, 95)),
                   "official_detections": reference, "candidate_detections": candidate})
    # Retain existing detection-error bounds per box; averages must not cancel individual box errors.
    report["pass"] = (report["stable"] and report["unmatched_reference"] == 0 and
                      report["unmatched_candidate"] == 0 and
                      report["maximum_score_error"] <= (0.005 if quant == "f32" else 0.02) and
                      report["maximum_center_error_px"] <= (1.0 if quant == "f32" else 2.5))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
