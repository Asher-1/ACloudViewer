#!/usr/bin/env python3
"""Export the final DINOv2 token trace used by the DeDoDe-G ggml gate."""

import argparse
import struct
from pathlib import Path


def load_descriptor_fixture(path: Path):
    with path.open("rb") as source:
        if source.read(8) != b"LOMDDB1\0":
            raise SystemExit("expected a DeDoDe descriptor reference fixture")
        version, width, height, count, dimension, rgb_bytes = struct.unpack(
            "<6I", source.read(24)
        )
        if version != 1 or width != 784 or height != 784 or rgb_bytes != width * height * 3:
            raise SystemExit("invalid DeDoDe descriptor reference fixture")
        rgb = source.read(rgb_bytes)
        if len(rgb) != rgb_bytes:
            raise SystemExit("truncated DeDoDe descriptor reference fixture")
    return width, height, rgb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-model", required=True, type=Path)
    parser.add_argument("--descriptor-fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import numpy as np
    import onnxruntime as ort

    width, height, rgb = load_descriptor_fixture(args.descriptor_fixture)
    image = np.frombuffer(rgb, dtype=np.uint8).reshape(height, width, 3)
    nchw = image.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    keypoints = np.zeros((1, 1, 2), dtype=np.float32)
    session = ort.InferenceSession(str(args.trace_model), providers=["CPUExecutionProvider"])
    descriptions, tokens = session.run(None, {"image": nchw, "keypoints": keypoints})
    del descriptions
    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.shape != (1, 3137, 1024):
        raise SystemExit(f"unexpected DINOv2 trace shape: {tokens.shape}")
    with args.output.open("wb") as target:
        target.write(b"LOMDGT1\0")
        target.write(struct.pack("<6I", 1, width, height, 3137, 1024, len(rgb)))
        target.write(rgb)
        target.write(tokens.tobytes())
    print(f"wrote {args.output}: 3137 x 1024 DINOv2 tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
