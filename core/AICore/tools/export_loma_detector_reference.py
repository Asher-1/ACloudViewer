#!/usr/bin/env python3
"""Write a real-image DaD ONNX reference fixture for the ggml CTest.

This development-only script deliberately imports ONNX Runtime and Pillow. The
fixture contains only RGB bytes and source outputs; the product test links
libAICore and GGUF exclusively.
"""

import argparse
import struct
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()
    if not 1 <= args.max_keypoints <= 2048:
        raise SystemExit("--max-keypoints must be in [1, 2048]")

    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    image = Image.open(args.image).convert("RGB")
    if args.width is not None or args.height is not None:
        if args.width is None or args.height is None:
            raise SystemExit("--width and --height must be supplied together")
        image = image.resize((args.width, args.height))
    width, height = image.size
    rgb = image.tobytes()
    nchw = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    keypoints, scores = session.run(None, {
        "image": nchw,
        "num_keypoints": np.asarray([args.max_keypoints], dtype=np.int64),
    })
    keypoints = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    # COLMAP's LoMa bridge returns source pixel coordinates, not [-1, 1].
    keypoints[:, 0] = 0.5 * (keypoints[:, 0] + 1.0) * width
    keypoints[:, 1] = 0.5 * (keypoints[:, 1] + 1.0) * height
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if keypoints.shape[0] != args.max_keypoints or scores.shape[0] != args.max_keypoints:
        raise SystemExit("DaD ONNX output did not honor requested keypoint count")
    with args.output.open("wb") as output:
        output.write(b"LOMDAD1\0")
        output.write(struct.pack("<IIIII", 1, width, height,
                                 args.max_keypoints, len(rgb)))
        output.write(rgb)
        output.write(keypoints.tobytes())
        output.write(scores.tobytes())
    print(f"wrote {args.output}: {width}x{height}, {args.max_keypoints} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
