#!/usr/bin/env python3
"""Run the pinned upstream evaluator on its explicitly requested device.

Upstream Tag2Text's batch helper reads a module-level device selected at
import time. Bind that helper to the evaluator's --device option before
delegating; weights, transforms, forward operations and scoring stay upstream.
"""

import argparse
import importlib.util
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--upstream", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    upstream = args.upstream.resolve()
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "cpp_ggml/scripts"))
    spec = importlib.util.spec_from_file_location(
        "ram_pytorch_eval", upstream / "cpp_ggml/benchmarks/python_eval.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    sys.argv = [str(spec.origin), *remaining]
    options = evaluation.parse_args()
    import batch_inference
    import torch
    batch_inference.device = torch.device(options.device)
    return evaluation.main()


if __name__ == "__main__":
    raise SystemExit(main())
