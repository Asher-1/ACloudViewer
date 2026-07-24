#!/usr/bin/env python3
"""DEV/CI ONLY — generate LGINP01 fixtures for AICore contract tests.

Not used by the qLightGlue GUI plugin (which uses OpenCV SIFT + GGML, or will
use ONNX for ALIKED following the COLMAP pattern).
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    from lightglue import ALIKED
    from lightglue.utils import load_image
except ImportError as exc:
    print(f"Missing dependency: {exc}", file=sys.stderr)
    print("Install: pip install torch lightglue", file=sys.stderr)
    raise SystemExit(2)


def to_numpy(value: torch.Tensor) -> np.ndarray:
    return value[0].detach().cpu().numpy().astype(np.float32)


def write_fixture(path: Path, records: list[dict]) -> None:
    m = records[0]["kpts"].shape[0]
    n = records[1]["kpts"].shape[0]
    dim = records[0]["desc"].shape[1]
    with path.open("wb") as stream:
        stream.write(b"LGINP01\0")
        stream.write(struct.pack("<I", 1))  # version
        stream.write(struct.pack("<I", m))
        stream.write(struct.pack("<I", n))
        stream.write(struct.pack("<I", dim))
        stream.write(struct.pack("<I", records[0]["width"]))
        stream.write(struct.pack("<I", records[0]["height"]))
        stream.write(struct.pack("<I", records[1]["width"]))
        stream.write(struct.pack("<I", records[1]["height"]))
        stream.write(struct.pack("<I", 0))  # flags
        stream.write(records[0]["kpts"].astype(np.float32).tobytes())
        stream.write(records[1]["kpts"].astype(np.float32).tobytes())
        stream.write(records[0]["desc"].astype(np.float32).tobytes())
        stream.write(records[1]["desc"].astype(np.float32).tobytes())
        stream.write(records[0]["scales"].astype(np.float32).tobytes())
        stream.write(records[1]["scales"].astype(np.float32).tobytes())
        stream.write(records[0]["oris"].astype(np.float32).tobytes())
        stream.write(records[1]["oris"].astype(np.float32).tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image0", required=True, type=Path)
    parser.add_argument("--image1", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--resize", type=int, default=1024)
    args = parser.parse_args()

    extractor = ALIKED(
        max_num_keypoints=args.max_keypoints, detection_threshold=-1
    ).eval()

    records = []
    for path in (args.image0, args.image1):
        image = load_image(path)
        with torch.inference_mode():
            features = extractor.extract(image, resize=args.resize)
        count = features["keypoints"].shape[1]
        records.append(
            {
                "kpts": to_numpy(features["keypoints"]),
                "desc": to_numpy(features["descriptors"]),
                "scales": to_numpy(features["scales"])
                if "scales" in features
                else np.ones(count, dtype=np.float32),
                "oris": to_numpy(features["oris"])
                if "oris" in features
                else np.zeros(count, dtype=np.float32),
                "width": int(features["image_size"][0, 0]),
                "height": int(features["image_size"][0, 1]),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_fixture(args.output, records)
    print(
        f"aliked: {records[0]['kpts'].shape[0]} x "
        f"{records[1]['kpts'].shape[0]} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
