#!/usr/bin/env python3
"""Measure one-kernel-at-a-time OCR storage sensitivity in source ONNX.

This conversion diagnostic does not certify a model or own a validation
matrix. Every resulting mixed-precision container still needs independent
source, labelled-page, backend and performance gates.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import gguf
import numpy as np
import onnx
import onnxruntime as ort

from convert_onnx_ocr_gguf import lower
from convert_paddle_ocr_gguf import quantized
from export_onnx_ocr_fixture import preprocess
from onnx_ocr_reference import replace_constants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--role", choices=("det", "rec", "cls"), required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--include-rotated", action="store_true")
    parser.add_argument("--quant", choices=("f16", "q8_0"), required=True)
    parser.add_argument("--max-side", type=int, default=320)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.include_rotated and args.role != "cls":
        parser.error("rotated sensitivity inputs require the classifier role")
    data = args.model.read_bytes()
    if hashlib.sha256(data).hexdigest() != args.source_sha256:
        parser.error("source SHA-256 mismatch")
    tensors, _, keep_f32 = lower(onnx.load_model_from_string(data))
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    options.log_severity_level = 3

    def session(model):
        return ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])

    inputs, input_identity = [], []
    for path in args.images:
        image = cv2.imread(str(path))
        if image is None:
            parser.error(f"image decode failed: {path}")
        inputs.append(preprocess(image, args.role, args.max_side))
        input_identity.append({"name": path.name,
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "rotation_degrees": 0})
        if args.include_rotated:
            inputs.append(preprocess(image[::-1, ::-1].copy(), args.role, args.max_side))
            input_identity.append(dict(input_identity[-1], rotation_degrees=180))
    original = session(data)
    expected = [original.run(None, {original.get_inputs()[0].name: x})[0] for x in inputs]
    del original
    rows = []
    # O(k * graph_cost), one live ORT session at a time. Evaluating a single
    # changed tensor attributes rounding error without k^2 pairwise models.
    for name, tensor in tensors.items():
        if name in keep_f32:
            continue
        rounded, kind = quantized(name, tensor, args.quant, [])
        restored = gguf.quants.dequantize(rounded, kind).reshape(tensor.shape)
        model = replace_constants(onnx.load_model_from_string(data), {name: restored})
        candidate = session(model.SerializeToString())
        metrics = []
        for x, y in zip(inputs, expected):
            z = candidate.run(None, {candidate.get_inputs()[0].name: x})[0]
            delta = np.abs(z.astype(np.float64) - y)
            metrics.append({"max_diff": float(delta.max()), "mean_diff": float(delta.mean()),
                            "argmax_match": float(np.mean(z.argmax(-1) == y.argmax(-1)))
                            if args.role != "det" else None})
        del candidate
        row = {"tensor": name, "shape": list(tensor.shape), "storage": kind.name,
               "max_diff": max(m["max_diff"] for m in metrics), "per_image": metrics}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "per_image"}), flush=True)
    report = {"source_sha256": args.source_sha256, "quant": args.quant,
              "onnxruntime_version": ort.__version__, "provider": "CPUExecutionProvider",
              "images": input_identity,
              "rows": sorted(rows, key=lambda row: row["max_diff"], reverse=True),
              "scope": "single-kernel source sensitivity; not release acceptance"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
