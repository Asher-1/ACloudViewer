#!/usr/bin/env python3
"""Validate a COLMAP LoMa ONNX asset before its offline GGUF conversion.

This is a development-only tool. It is intentionally not imported by CMake or
the AICore runtime: the shipped application remains free of ONNX libraries.
"""

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path


# These are the operators exercised by the first audited upstream matcher
# (loma_matcher_B.onnx). A converter must explicitly add support when another
# upstream model expands this set; accepting unknown operators would create a
# GGUF that appears valid but cannot reproduce the source graph.
MATCHER_B_OPS = frozenset(
    {
        "Add", "And", "ArgMax", "Concat", "Cos", "Div", "Einsum",
        "Equal", "Erf", "Expand", "Gather", "GatherElements", "Greater",
        "LayerNormalization", "MatMul", "Mul", "Neg", "Range", "ReduceMax",
        "Reshape", "Shape", "Sin", "Slice", "Softmax", "Squeeze",
        "Transpose", "Unsqueeze", "Where",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--sha256", required=True,
                        help="Pinned SHA-256 from COLMAP resources.h")
    parser.add_argument("--allow-unknown-ops", action="store_true")
    args = parser.parse_args()

    actual_digest = sha256(args.model)
    if actual_digest.lower() != args.sha256.lower():
        print(f"digest mismatch: expected {args.sha256}, got {actual_digest}",
              file=sys.stderr)
        return 2

    try:
        import onnx  # Development-only conversion dependency.
    except ImportError:
        print("install the Python 'onnx' package only in the converter environment",
              file=sys.stderr)
        return 3

    model = onnx.load(str(args.model), load_external_data=False)
    operators = Counter(node.op_type for node in model.graph.node)
    unknown = sorted(set(operators) - MATCHER_B_OPS)
    print(f"sha256={actual_digest}")
    print(f"initializers={len(model.graph.initializer)} nodes={len(model.graph.node)}")
    for name, count in sorted(operators.items()):
        print(f"{name}={count}")
    if unknown and not args.allow_unknown_ops:
        print("unimplemented operators: " + ", ".join(unknown), file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
