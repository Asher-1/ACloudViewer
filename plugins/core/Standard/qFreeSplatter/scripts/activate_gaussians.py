#!/usr/bin/env python3
"""Derive the activated (render-ready) gaussians reference from gaussians_raw.

forward_gaussians returns RAW logits; the renderer (GaussianModel.get_*) applies
the activations when it builds splats. This reproduces those activations for the
scene config (configs/freesplatter-scene.yaml) so the engine's emitted, activated
output can be parity-checked. Channel layout [xyz3, sh12, opacity1, scale3, rot4]:

  xyz (0:3)      identity (rescale = 1)
  sh  (3:15)     identity (sh_degree>0 -> raw SH)
  opacity (15)   sigmoid
  scale (16:19)  scale_min + (scale_max - scale_min) * sigmoid     [0.0001, 0.02]
  rot (19:23)    L2-normalize (quaternion, order w,x,y,z)

  python scripts/activate_gaussians.py FIXTURE_DIR [--scale-min 1e-4 --scale-max 0.02]

Reads <dir>/gaussians_raw.f32 (+ meta.json), writes <dir>/gaussians.f32 and adds
it to meta.json. Idempotent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def activate(raw: np.ndarray, smin: float, smax: float) -> np.ndarray:
    g = raw.astype(np.float64).copy()
    g[:, 15] = sigmoid(g[:, 15])
    g[:, 16:19] = smin + (smax - smin) * sigmoid(g[:, 16:19])
    rot = g[:, 19:23]
    g[:, 19:23] = rot / (np.linalg.norm(rot, axis=1, keepdims=True) + 1e-12)
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--scale-min", type=float, default=1e-4)
    ap.add_argument("--scale-max", type=float, default=0.02)
    args = ap.parse_args()

    meta = json.loads((args.dir / "meta.json").read_text())
    shape = meta["taps"]["gaussians_raw"]["shape"]
    raw = np.fromfile(args.dir / "gaussians_raw.f32", dtype="<f4").reshape(shape)
    g = activate(raw, args.scale_min, args.scale_max).astype("<f4")
    g.tofile(args.dir / "gaussians.f32")
    meta["taps"]["gaussians"] = {"shape": list(g.shape), "dtype": "f32"}
    (args.dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.dir}/gaussians.f32 {list(g.shape)}")
    return 0


if __name__ == "__main__":
    main()
