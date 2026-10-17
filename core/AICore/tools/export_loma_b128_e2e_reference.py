#!/usr/bin/env python3
"""Create a real-image DaD -> DeDoDe-B -> LoMa-B128 reference fixture.

The ONNX models are used only to create a development fixture. The C++ gate
uses the production GGUF graphs and public AICore ABI exclusively.
"""

import argparse
import struct
from pathlib import Path


def load_rgb(path: Path, width: int, height: int):
    from PIL import Image
    import numpy as np

    image = Image.open(path).convert("RGB").resize((width, height))
    rgb = image.tobytes()
    nchw = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return rgb, nchw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-model", required=True, type=Path)
    parser.add_argument("--descriptor-model", required=True, type=Path)
    parser.add_argument("--matcher-model", required=True, type=Path)
    parser.add_argument("--image0", required=True, type=Path)
    parser.add_argument("--image1", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-keypoints", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=784)
    parser.add_argument("--height", type=int, default=784)
    args = parser.parse_args()
    if not 1 <= args.max_keypoints <= 2048:
        raise SystemExit("--max-keypoints must be in [1, 2048]")

    import numpy as np
    import onnxruntime as ort

    detector = ort.InferenceSession(
        str(args.detector_model), providers=["CPUExecutionProvider"]
    )
    descriptor = ort.InferenceSession(
        str(args.descriptor_model), providers=["CPUExecutionProvider"]
    )
    matcher = ort.InferenceSession(
        str(args.matcher_model), providers=["CPUExecutionProvider"]
    )
    rgb0, input0 = load_rgb(args.image0, args.width, args.height)
    rgb1, input1 = load_rgb(args.image1, args.width, args.height)

    def extract(nchw):
        keypoints, scores = detector.run(
            None,
            {"image": nchw,
             "num_keypoints": np.asarray([args.max_keypoints], dtype=np.int64)},
        )
        descriptions = descriptor.run(
            None, {"image": nchw, "keypoints": keypoints}
        )[0]
        return (np.asarray(keypoints, dtype=np.float32).reshape(args.max_keypoints, 2),
                np.asarray(scores, dtype=np.float32).reshape(args.max_keypoints),
                np.asarray(descriptions, dtype=np.float32).reshape(args.max_keypoints, -1))

    points0, _scores0, desc0 = extract(input0)
    points1, _scores1, desc1 = extract(input1)
    m0, _m1, match_scores, _match_scores1 = matcher.run(
        None,
        {"kpts0": points0[None], "kpts1": points1[None],
         "desc0": desc0[None], "desc1": desc1[None]},
    )
    accepted = [(index, int(m0[0, index]), float(match_scores[0, index]))
                for index in range(args.max_keypoints)
                if m0[0, index] >= 0 and match_scores[0, index] >= args.threshold]
    if not accepted:
        raise SystemExit("real-image LoMa-B128 reference contains no accepted matches")
    if desc0.shape[1] != 128:
        raise SystemExit("LoMa-B128 reference did not produce 128-dimensional descriptors")
    with args.output.open("wb") as output:
        output.write(b"LOMB128\0")
        output.write(struct.pack("<IIIIIII", 1, args.width, args.height,
                                 args.max_keypoints, desc0.shape[1], len(rgb0),
                                 len(accepted)))
        output.write(rgb0)
        output.write(rgb1)
        for index0, index1, score in accepted:
            output.write(struct.pack("<iif", index0, index1, score))
    print(f"wrote {args.output}: {args.max_keypoints} features, "
          f"{len(accepted)} matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
