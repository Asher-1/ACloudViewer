#!/usr/bin/env python3
"""Convert the engine's activated gaussians (free_splatter-cli --out) to a
.splat file (antimatter15 layout) for the WebGL viewer.

The engine output is already activated (opacity sigmoid'd, scale mapped, quat
normalized), channels [xyz(3), SH(12), opacity(1), scale(3), rot(4)] = 23. This:
  - prunes splats with opacity <= --opacity-threshold (default 5e-3, matching
    FreeSplatter's save_gaussian),
  - colour from the SH DC term: rgb = clip(0.5 + C0 * f_dc, 0, 1),
  - packs 32 bytes/splat: pos(3 f32), scale(3 f32), rgba(4 u8), rot(4 u8: w,x,y,z),
  - sorts by importance (opacity x volume) descending, so a progressive loader
    reveals the dominant structure first (the "scene builds up" effect).

  python scripts/to_splat.py gaussians.f32 out.splat [--opacity-threshold 5e-3]
"""
from __future__ import annotations

import argparse

import numpy as np

C0 = 0.28209479177387814
GC = 23


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gaussians")
    ap.add_argument("out")
    ap.add_argument("--opacity-threshold", type=float, default=5e-3)
    args = ap.parse_args()

    g = np.fromfile(args.gaussians, dtype="<f4").reshape(-1, GC)
    xyz   = g[:, 0:3]
    dc    = g[:, 3:6]        # SH DC (first coeff per channel)
    opac  = g[:, 15]         # already sigmoid'd
    scale = g[:, 16:19]      # already mapped to physical scale
    rot   = g[:, 19:23]      # already-normalized quaternion (w,x,y,z)

    keep = opac > args.opacity_threshold
    xyz, dc, opac, scale, rot = xyz[keep], dc[keep], opac[keep], scale[keep], rot[keep]

    # importance = opacity * volume; dominant splats first (progressive reveal)
    importance = opac * np.prod(np.maximum(scale, 1e-9), axis=1)
    order = np.argsort(-importance)
    xyz, dc, opac, scale, rot = xyz[order], dc[order], opac[order], scale[order], rot[order]

    rgb = np.clip(0.5 + C0 * dc, 0.0, 1.0)
    rgba = np.concatenate([rgb, opac[:, None]], axis=1)
    # OpenCV (y down, z forward) -> OpenGL (y up): 180deg about X = diag(1,-1,-1).
    xyz = xyz * np.array([1.0, -1.0, -1.0], dtype=np.float32)
    w, x, y, z = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]   # (w,x,y,z)
    rot = np.stack([-x, w, -z, y], axis=1)                    # (-x, w, -z, y)
    rot = rot / (np.linalg.norm(rot, axis=1, keepdims=True) + 1e-12)

    dt = np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3),
                   ("rgba", "u1", 4), ("rot", "u1", 4)])
    arr = np.zeros(len(xyz), dt)
    arr["pos"]   = xyz
    arr["scale"] = scale
    arr["rgba"]  = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)
    arr["rot"]   = np.clip(rot * 128.0 + 128.0, 0, 255).astype(np.uint8)
    arr.tofile(args.out)
    print(f"wrote {args.out}: {len(xyz)} splats ({arr.nbytes} bytes), "
          f"pruned {(~keep).sum()} of {len(keep)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
