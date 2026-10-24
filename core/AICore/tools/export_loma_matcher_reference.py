#!/usr/bin/env python3
"""Create an ONNX-reference fixture for the ggml LoMa matcher P/R gate.

This is conversion-host tooling only.  It requires numpy and onnxruntime; the
fixture is a small binary input/output record consumed by a C++ test that links
only libAICore and ggml.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import onnxruntime as ort


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def expected_sha256(variant: str) -> str:
    source_path = Path(__file__).with_name("loma_sources.json")
    with source_path.open(encoding="utf-8") as source:
        models = json.load(source)["models"]
    if variant not in models:
        raise SystemExit(f"unknown pinned LoMa matcher variant: {variant}")
    return models[variant]["sha256"]


def normalized_pixels(rng: np.random.Generator, count: int, width: int,
                      height: int) -> np.ndarray:
    pixels = rng.uniform([4.0, 4.0], [width - 4.0, height - 4.0],
                         size=(count, 2)).astype(np.float32)
    return pixels


def descriptors(rng: np.random.Generator, count: int, dimension: int) -> np.ndarray:
    # LoMa consumes DeDoDe's native descriptor scale. Do not L2-normalize as
    # LightGlue does: that changes the assignment confidence and invalidates
    # COLMAP's 0.1 filtering threshold.
    return rng.standard_normal((count, dimension), dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", default="matcher_B",
                        help="pinned loma_sources.json matcher key")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--n0", type=int, default=64)
    parser.add_argument("--n1", type=int, default=80)
    args = parser.parse_args()
    if args.n1 < args.n0 or args.n0 <= 0:
        raise SystemExit("require 0 < n0 <= n1")
    actual_digest = digest(args.model)
    if actual_digest != expected_sha256(args.variant):
        raise SystemExit(f"reference model digest does not match pinned {args.variant}")

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    descriptor_input = next((item for item in session.get_inputs()
                             if item.name == "desc0"), None)
    if descriptor_input is None or len(descriptor_input.shape) != 3 or \
            not isinstance(descriptor_input.shape[2], int) or \
            descriptor_input.shape[2] <= 0:
        raise SystemExit("LoMa matcher desc0 input does not expose a descriptor dimension")
    descriptor_dimension = descriptor_input.shape[2]

    width, height = 640, 480
    rng = np.random.default_rng(args.seed)
    pixels0 = normalized_pixels(rng, args.n0, width, height)
    pixels1 = np.empty((args.n1, 2), dtype=np.float32)
    pixels1[:args.n0] = np.clip(
            pixels0 + rng.normal(0.0, 1.25, pixels0.shape).astype(np.float32),
            [0.0, 0.0], [width - 1.0, height - 1.0])
    pixels1[args.n0:] = normalized_pixels(rng, args.n1 - args.n0, width, height)
    desc0 = descriptors(rng, args.n0, descriptor_dimension)
    desc1 = np.empty((args.n1, descriptor_dimension), dtype=np.float32)
    desc1[:args.n0] = desc0
    desc1[args.n0:] = descriptors(rng, args.n1 - args.n0, descriptor_dimension)
    kpts0 = 2.0 * pixels0 / np.array([width, height], dtype=np.float32) - 1.0
    kpts1 = 2.0 * pixels1 / np.array([width, height], dtype=np.float32) - 1.0

    m0, _m1, scores0, _scores1 = session.run(
            None, {"kpts0": kpts0[None], "kpts1": kpts1[None],
                   "desc0": desc0[None], "desc1": desc1[None]})
    accepted = [(index, int(m0[0, index]), float(scores0[0, index]))
                for index in range(args.n0)
                if m0[0, index] >= 0 and scores0[0, index] >= args.threshold]
    if not accepted:
        raise SystemExit("reference fixture contains no accepted matches")

    with args.output.open("wb") as target:
        target.write(struct.pack("<8s7I", b"LOMRF01\0", 1, args.n0, args.n1,
                                 descriptor_dimension, width, height, len(accepted)))
        for array in (pixels0, pixels1, desc0, desc1):
            target.write(np.ascontiguousarray(array, dtype="<f4").tobytes())
        for index0, index1, score in accepted:
            target.write(struct.pack("<iif", index0, index1, score))
    print(f"wrote {args.output}: matches={len(accepted)} sha256={actual_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
