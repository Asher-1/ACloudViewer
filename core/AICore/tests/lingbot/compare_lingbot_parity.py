#!/usr/bin/env python3
"""Element-wise A/B between the upstream lingbot-map-cli and the AICore
LingBot-Map integration.

Both sides dump per-frame LBF3 binaries (magic 0x4C424633 + idx32 + h + w +
pose_enc[9] + depth + depth_conf + c2w[16] + intrinsics[4], little-endian
float32) — the upstream CLI via --stream-dir, the AICore side via
tests/lingbot/dump_lingbot_frames. This script reads both directories and
reports the max absolute difference of depth, c2w and intrinsics per frame,
plus the throughput numbers if the RESULT lines are passed.

Usage:
  compare_lingbot_parity.py <upstream_dir> <aicore_dir> [--tol 1e-6]
                             [--upstream-result LINE] [--aicore-result LINE]

Exit code 0 = within tolerance (or identical), 1 = out of tolerance.
"""
from __future__ import annotations

import argparse
import glob
import os
import struct
import sys

LBF3_MAGIC = 0x4C424633


def read_lbf3(path: str):
    with open(path, "rb") as f:
        data = f.read()
    magic, idx, h, w = struct.unpack_from("<IIII", data, 0)
    assert magic == LBF3_MAGIC, f"bad magic in {path}"
    off = 16
    floats = struct.unpack_from(f"<{9}f", data, off)
    pose_enc = list(floats)
    off += 9 * 4
    plane = h * w
    depth = struct.unpack_from(f"<{plane}f", data, off)
    off += plane * 4
    depth_conf = struct.unpack_from(f"<{plane}f", data, off)
    off += plane * 4
    c2w = struct.unpack_from("<16f", data, off)
    off += 16 * 4
    intrinsics = struct.unpack_from("<4f", data, off)
    return idx, h, w, pose_enc, depth, depth_conf, c2w, intrinsics


def max_abs_diff(a, b) -> float:
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def mean_abs_diff(a, b) -> float:
    if not a:
        return 0.0
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def rmse(a, b) -> float:
    if not a:
        return 0.0
    return (sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)) ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("upstream_dir")
    ap.add_argument("aicore_dir")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--upstream-result", default="")
    ap.add_argument("--aicore-result", default="")
    args = ap.parse_args()

    upstream = sorted(
        glob.glob(os.path.join(args.upstream_dir, "frame_*.bin")))
    aicore = sorted(glob.glob(os.path.join(args.aicore_dir, "frame_*.bin")))
    if not upstream or not aicore:
        print("FAIL: missing frames "
              f"(upstream={len(upstream)} aicore={len(aicore)})")
        return 1
    if len(upstream) != len(aicore):
        print(f"FAIL: frame count mismatch {len(upstream)} vs {len(aicore)}")
        return 1

    overall = {"depth": 0.0, "depth_conf": 0.0, "c2w": 0.0, "intrinsics": 0.0}
    for up_path, ai_path in zip(upstream, aicore):
        u = read_lbf3(up_path)
        a = read_lbf3(ai_path)
        assert u[0] == a[0] and u[1] == a[1] and u[2] == a[2], (
            f"frame header mismatch: upstream={u[:3]} aicore={a[:3]}")
        diffs = {
            "depth": max_abs_diff(u[4], a[4]),
            "depth_conf": max_abs_diff(u[5], a[5]),
            "c2w": max_abs_diff(u[6], a[6]),
            "intrinsics": max_abs_diff(u[7], a[7]),
        }
        means = {
            "depth": mean_abs_diff(u[4], a[4]),
            "c2w": mean_abs_diff(u[6], a[6]),
        }
        rmses = {
            "depth": rmse(u[4], a[4]),
        }
        for k, v in diffs.items():
            overall[k] = max(overall[k], v)
        print(
            f"frame {u[0]:04d} ({u[1]}x{u[2]}): "
            f"depth_maxdiff={diffs['depth']:.3e} "
            f"depth_mean={means['depth']:.3e} depth_rmse={rmses['depth']:.3e} "
            f"conf_maxdiff={diffs['depth_conf']:.3e} "
            f"c2w_maxdiff={diffs['c2w']:.3e} c2w_mean={means['c2w']:.3e} "
            f"intr_maxdiff={diffs['intrinsics']:.3e}")

    print("overall max diffs:", {k: f"{v:.3e}" for k, v in overall.items()})
    if args.upstream_result:
        print("upstream:", args.upstream_result)
    if args.aicore_result:
        print("aicore:  ", args.aicore_result)

    worst = max(overall.values())
    if worst <= args.tol:
        print(f"PASS: max diff {worst:.3e} <= tol {args.tol:.1e}")
        return 0
    print(f"FAIL: max diff {worst:.3e} > tol {args.tol:.1e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
