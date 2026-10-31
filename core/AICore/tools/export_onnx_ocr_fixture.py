#!/usr/bin/env python3
"""Export independent ONNX OCR probabilities for the native OCR probe.

Uses the source PP-OCR normalization and a real image. ONNX Runtime/OpenCV
are test/conversion dependencies only. A sidecar records input/model hashes
and the resolved engine; the model's exact source digest is mandatory.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_crop(image, quad):
    """Match the source get_rotate_crop_image before recognizer normalization."""
    points = np.asarray(quad, np.float32).reshape(4, 2)
    if not np.isfinite(points).all():
        raise ValueError("crop quadrilateral must be finite")
    width = int(max(np.linalg.norm(points[0] - points[1]),
                    np.linalg.norm(points[2] - points[3])))
    height = int(max(np.linalg.norm(points[0] - points[3]),
                     np.linalg.norm(points[1] - points[2])))
    if width <= 0 or height <= 0:
        raise ValueError("crop quadrilateral has an empty edge")
    target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    transform = cv2.getPerspectiveTransform(points, target)
    crop = cv2.warpPerspective(image, transform, (width, height),
                               borderMode=cv2.BORDER_REPLICATE,
                               flags=cv2.INTER_CUBIC)
    return np.rot90(crop) if height / width >= 1.5 else crop


def preprocess(image, role, max_side=960):
    h, w = image.shape[:2]
    if role == "det":
        ratio = min(1.0, max_side / max(h, w))
        rh, rw = max(32, round(int(h * ratio) / 32) * 32), max(32, round(int(w * ratio) / 32) * 32)
        # Source TextDetector applies these constants to cv2's BGR planes.
        x = cv2.resize(image, (rw, rh)).astype(np.float32) / 255
        x = (x - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        return x.transpose(2, 0, 1)[None].copy()
    width = 192 if role == "cls" else max(320, int(48 * w / h))
    rw = min(width, math.ceil(48 * w / h))
    resized = cv2.resize(image, (rw, 48)).astype(np.float32).transpose(2, 0, 1)
    x = np.zeros((1, 3, 48, width), np.float32)
    x[0, :, :, :rw] = (resized / 255 - .5) / .5
    return x


def reference_session(model, provider, threads):
    """Require the requested independent provider, including stored-weight graphs."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    expected = "CUDAExecutionProvider" if provider == "cuda" else "CPUExecutionProvider"
    providers = [(expected, {"use_tf32": "0"})] if provider == "cuda" else [expected]
    session = ort.InferenceSession(model, options, providers=providers)
    if session.get_providers()[0] != expected:
        raise RuntimeError("source accuracy provider silently fell back")
    return session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--role", choices=("det", "rec", "cls"), required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--max-side", type=int, default=960)
    parser.add_argument("--rotate180", action="store_true",
                        help="include the classifier's exact upside-down input transform")
    parser.add_argument("--quad", type=float, nargs=8,
                        help="source page quadrilateral for a recognizer regression input")
    parser.add_argument("--weights", type=Path,
                        help="also export an independent reference with this GGUF's stored weights")
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1: parser.error("thread count must be positive")
    if digest(args.model) != args.source_sha256: raise SystemExit("model digest mismatch")
    image = cv2.imread(str(args.image))
    if image is None: raise SystemExit("image decode failed")
    if args.quad is not None:
        if args.role != "rec": parser.error("quadrilateral fixture is recognizer-only")
        image = source_crop(image, args.quad)
    if args.rotate180:
        if args.role != "cls": parser.error("rotation fixture is classifier-only")
        image = image[::-1, ::-1].copy()
    x = preprocess(image, args.role, args.max_side)
    session = reference_session(str(args.model), args.provider, args.threads)
    y = session.run(None, {session.get_inputs()[0].name: x})[0]
    if not np.isfinite(y).all(): raise SystemExit("source produced non-finite output")
    args.output.mkdir(parents=True, exist_ok=True)
    stem = {"det": "det_ch", "rec": "rec_ch", "cls": "cls"}[args.role]
    np.save(args.output / f"{stem}_input.npy", x)
    np.save(args.output / f"{stem}_reference.npy", y.astype(np.float32))
    report = {"model_sha256": digest(args.model), "image_sha256": digest(args.image),
              "image": args.image.name, "role": args.role,
              "input_shape": list(x.shape), "output_shape": list(y.shape),
              "onnxruntime_version": ort.__version__, "provider": session.get_providers(),
              "provider_options": session.get_provider_options(), "threads": args.threads,
              "preprocess": "X-AnyLabeling PP-OCR", "max_side": args.max_side,
              "rotation_degrees": 180 if args.rotate180 else 0}
    if args.quad is not None:
        report["source_quad"] = args.quad
        report["crop_shape"] = list(image.shape)
        report["crop_pixels_sha256"] = hashlib.sha256(image.tobytes()).hexdigest()
    if args.weights:
        from onnx_ocr_reference import stored_weight_reference
        reference, storage = stored_weight_reference(args.model, args.weights)
        rounded = reference_session(reference.SerializeToString(), args.provider, args.threads)
        z = rounded.run(None, {rounded.get_inputs()[0].name: x})[0]
        if z.shape != y.shape or not np.isfinite(z).all():
            raise SystemExit("stored-weight reference has invalid output")
        np.save(args.output / f"{stem}_stored_reference.npy", z.astype(np.float32))
        delta = np.abs(z.astype(np.float64) - y)
        report["stored_weights"] = {
            "container_sha256": digest(args.weights), "storage_elements": storage,
            "provider": rounded.get_providers(), "provider_options": rounded.get_provider_options(),
            "source_max_diff": float(delta.max()), "source_mean_diff": float(delta.mean()),
            "source_argmax_match": float(np.mean(z.argmax(-1) == y.argmax(-1)))
                if args.role != "det" else None}
    (args.output / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
