#!/usr/bin/env python3
"""Preprocess scene images into the engine's raw-float input.

Mirrors FreeSplatter's scene path (runner.run_views_to_scene): per image,
center-crop to a square, resize to 512x512 (bicubic, antialias), scale to [0,1],
lay out as CHW; stack views. Writes a flat float32 file (view-major NCHW) that
free_splatter-cli consumes directly.

  python scripts/preprocess_scene.py out.f32 view1.jpg view2.jpg [...]
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image


def load(path: str, size: int = 512) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    img = img.resize((size, size), Image.BICUBIC)
    a = np.asarray(img, dtype=np.float32) / 255.0   # HWC
    return np.transpose(a, (2, 0, 1))               # CHW


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: preprocess_scene.py out.f32 img1 img2 [...]", file=sys.stderr)
        return 2
    out, imgs = sys.argv[1], sys.argv[2:]
    stack = np.stack([load(p) for p in imgs], axis=0)   # (N,3,512,512)
    stack.astype("<f4").tofile(out)
    print(f"wrote {out}: {stack.shape[0]} views, {stack.nbytes} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
