#!/usr/bin/env python3
"""Produce F32 / F16 / Q8_0 GGUF variants for vendored CNN models.

Usage (after building AICore with quantize CLI or from Python export):

  # 1) Export F32 from upstream PyTorch (sibling repos):
  #    DeepLSD: scripts/convert_deeplsd_to_gguf.py
  #
  # 2) Quantize with AICore tool:
  #    aicore_gguf_quantize  in.gguf out-f16.gguf f16
  #    aicore_gguf_quantize  in.gguf out-q8_0.gguf q8_0
  #
  Release naming (cloudViewer_downloads):
    DeepLSD/deeplsd_wireframe-f32.gguf
    DeepLSD/deeplsd_wireframe-f16.gguf
    DeepLSD/deeplsd_wireframe-q8_0.gguf
    DeepLSD/deeplsd_md-f32.gguf
    DeepLSD/deeplsd_md-f16.gguf
    DeepLSD/deeplsd_md-q8_0.gguf

Parity: run backend verify in sibling cpp trees (CPU/CUDA/Vulkan).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quantize_bin", help="path to aicore_gguf_quantize binary")
    parser.add_argument("input_f32", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--stem", required=True, help="e.g. deeplsd_wireframe")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    f32 = args.output_dir / f"{args.stem}-f32.gguf"
    f16 = args.output_dir / f"{args.stem}-f16.gguf"
    q8 = args.output_dir / f"{args.stem}-q8_0.gguf"

    if not f32.exists():
        f32.write_bytes(args.input_f32.read_bytes())

    for out, typ in ((f16, "f16"), (q8, "q8_0")):
        cmd = [args.quantize_bin, str(f32), str(out), typ]
        print("+", " ".join(cmd))
        subprocess.check_call(cmd)
    print("Wrote:", f32, f16, q8)
    return 0


if __name__ == "__main__":
    sys.exit(main())
