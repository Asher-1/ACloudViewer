#!/usr/bin/env python3
"""Write an ONNX reference fixture for DaD-selected DeDoDe-B descriptors.

This is a development utility. Production code and the CTest executable use
only the RGB bytes, GGUF weights, and libAICore's ggml graph.
"""

import argparse
import struct
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-model", required=True, type=Path)
    parser.add_argument("--descriptor-model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-keypoints", type=int, default=64)
    parser.add_argument("--width", type=int, default=784)
    parser.add_argument("--height", type=int, default=784)
    args = parser.parse_args()
    if not 1 <= args.max_keypoints <= 2048:
        raise SystemExit("--max-keypoints must be in [1, 2048]")

    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    image = Image.open(args.image).convert("RGB").resize((args.width, args.height))
    rgb = image.tobytes()
    nchw = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    detector = ort.InferenceSession(
        str(args.detector_model), providers=["CPUExecutionProvider"]
    )
    normalized_keypoints, _ = detector.run(
        None,
        {
            "image": nchw,
            "num_keypoints": np.asarray([args.max_keypoints], dtype=np.int64),
        },
    )
    normalized_keypoints = np.asarray(normalized_keypoints, dtype=np.float32)
    descriptor = ort.InferenceSession(
        str(args.descriptor_model), providers=["CPUExecutionProvider"]
    )
    descriptions = descriptor.run(
        None, {"image": nchw, "keypoints": normalized_keypoints}
    )[0]
    descriptions = np.asarray(descriptions, dtype=np.float32).reshape(args.max_keypoints, -1)
    keypoints = normalized_keypoints.reshape(args.max_keypoints, 2).copy()
    keypoints[:, 0] = 0.5 * (keypoints[:, 0] + 1.0) * args.width
    keypoints[:, 1] = 0.5 * (keypoints[:, 1] + 1.0) * args.height
    with args.output.open("wb") as output:
        output.write(b"LOMDDB1\0")
        output.write(
            struct.pack(
                "<IIIIII",
                1,
                args.width,
                args.height,
                args.max_keypoints,
                descriptions.shape[1],
                len(rgb),
            )
        )
        output.write(rgb)
        output.write(keypoints.tobytes())
        output.write(descriptions.tobytes())
    print(
        f"wrote {args.output}: {args.width}x{args.height}, "
        f"{args.max_keypoints} x {descriptions.shape[1]} descriptors"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
