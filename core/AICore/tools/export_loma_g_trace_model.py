#!/usr/bin/env python3
"""Append DINOv2's final token tensor to a pinned DeDoDe-G ONNX graph.

This is a development-only reference fixture utility. The runtime consumes
only GGUF through ggml; ONNX is never linked or loaded by ACloudViewer.
"""

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import onnx

    model = onnx.load(str(args.input), load_external_data=False)
    token_name = "layer_norm_48"
    if not any(token_name in node.output for node in model.graph.node):
        raise SystemExit(f"could not find final DINOv2 token tensor: {token_name}")
    if any(value.name == token_name for value in model.graph.output):
        raise SystemExit("DINOv2 token tensor is already an output")
    model.graph.output.append(
        onnx.helper.make_tensor_value_info(
            token_name, onnx.TensorProto.FLOAT, [1, 3137, 1024]))
    onnx.checker.check_model(model)
    onnx.save(model, str(args.output))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
