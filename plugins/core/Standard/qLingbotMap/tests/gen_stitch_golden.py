#!/usr/bin/env python3
"""Generate the golden header for test_lingbot_window_stitch.

Runs the official (verified) numpy port in the upstream lingbot-map-ggml
repo (ggml_demo.py _split_windows/_pairwise_alignment/_warp_window/
_stitch_windows/_quat_to_mat/_mat_to_quat) on deterministic synthetic data
and emits a C++ header with inputs + expected outputs.

Usage:
  python3 gen_stitch_golden.py /path/to/lingbot-map-ggml > out.h
"""
import sys

import numpy as np

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
import ggml_demo  # noqa: E402

OUT_W, OUT_H = 8, 8
PLANE = OUT_W * OUT_H
SCALE_FRAMES = 8


def make_pose_enc(rng, frames):
    pe = rng.uniform(-1.0, 1.0, size=(frames, 9)).astype(np.float32)
    # Valid random rotations: random quaternion, normalized, w>=0.
    q = rng.normal(size=(frames, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 3] < 0] *= -1.0
    pe[:, 3:7] = q
    pe[:, 7] = np.float32(1.2)
    pe[:, 8] = np.float32(1.4)
    return pe


def pose_enc_to_c2w(pe):
    """Engine decode: c2w rot = R^T, c2w t = -R^T t (w2c pose_enc)."""
    frames = pe.shape[0]
    c2w = np.zeros((frames, 4, 4), dtype=np.float32)
    for i in range(frames):
        R = ggml_demo._quat_to_mat(pe[i, 3:7].astype(np.float32))
        t = pe[i, :3].astype(np.float32)
        RT = R.T
        c2w[i, 0, 0:3] = RT[0]
        c2w[i, 1, 0:3] = RT[1]
        c2w[i, 2, 0:3] = RT[2]
        c2w[i, 0, 3] = -RT[0] @ t
        c2w[i, 1, 3] = -RT[1] @ t
        c2w[i, 2, 3] = -RT[2] @ t
        c2w[i, 3, 3] = 1.0
    return c2w


def make_window(rng, start, frames, with_degenerate=False):
    pe = make_pose_enc(rng, frames)
    depth = rng.uniform(0.5, 5.0, size=(frames, OUT_H, OUT_W)).astype(np.float32)
    if with_degenerate:
        depth[1, 2, 3] = np.float32("nan")
        depth[2, 4, 5] = 0.0  # |db| <= eps -> masked out of the ratio scale
    conf = rng.uniform(0.0, 3.0, size=(frames, PLANE)).astype(np.float32)
    intr = np.tile(
        np.array([10.0, 11.0, 4.0, 3.5], dtype=np.float32), (frames, 1)
    )
    return {
        "pose_enc": pe,
        "depth": depth.reshape(frames, PLANE),
        "depth_conf": conf,
        "c2w": pose_enc_to_c2w(pe),
        "intrinsics": intr,
    }


def hdr_array(name, arr):
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    lines = []
    lines.append(f"static const float {name}[] = {{")
    vals = [
        "std::numeric_limits<float>::quiet_NaN()" if np.isnan(v) else f"{v:.9e}f"
        for v in flat
    ]
    for i in range(0, len(vals), 4):
        lines.append("    " + ", ".join(vals[i : i + 4]) + ",")
    lines.append("};")
    return "\n".join(lines)


def emit_chain(f, rng, total, window_size, overlap, tag):
    plan = ggml_demo._split_windows(total, window_size, overlap)
    windows = [make_window(rng, s, e - s, with_degenerate=(i == 0))
               for i, (s, e) in enumerate(plan)]
    warped = []
    aligns = []
    for wi, win in enumerate(windows):
        if wi > 0:
            s_ab, R_ab, t_ab = ggml_demo._pairwise_alignment(
                warped[-1], win, overlap)
            aligns.append((float(s_ab), R_ab, t_ab))
            win = ggml_demo._warp_window(win, s_ab, R_ab, t_ab)
        warped.append(win)
    merged = ggml_demo._stitch_windows(warped, overlap)

    f.write(f"// ---- chain {tag}: total={total} window={window_size} "
            f"overlap={overlap} ----\n")
    f.write(f"constexpr int k{tag}Total = {total};\n")
    f.write(f"constexpr int k{tag}WindowSize = {window_size};\n")
    f.write(f"constexpr int k{tag}Overlap = {overlap};\n")
    f.write(f"constexpr int k{tag}NumWindows = {len(plan)};\n")
    for wi, (s, e) in enumerate(plan):
        f.write(f"constexpr int k{tag}Win{wi}Start = {s};\n")
        f.write(f"constexpr int k{tag}Win{wi}End = {e};\n")
    f.write(hdr_array(f"k{tag}Win0PoseEnc", windows[0]["pose_enc"]))
    f.write("\n")
    f.write(hdr_array(f"k{tag}Win0Depth", windows[0]["depth"]))
    f.write("\n")
    f.write(hdr_array(f"k{tag}Win0Conf", windows[0]["depth_conf"]))
    f.write("\n")
    f.write(hdr_array(f"k{tag}Win0C2w", windows[0]["c2w"]))
    f.write("\n")
    f.write(hdr_array(f"k{tag}Win0Intr", windows[0]["intrinsics"]))
    f.write("\n")
    f.write(hdr_array(f"k{tag}Win1PoseEnc", windows[1]["pose_enc"]))
    f.write("\n")
    f.write(hdr_array(f"k{tag}Win1Depth", windows[1]["depth"]))
    f.write("\n")
    f.write(hdr_array(f"k{tag}Win1Conf", windows[1]["depth_conf"]))
    f.write("\n")
    f.write(hdr_array(f"k{tag}Win1C2w", windows[1]["c2w"]))
    f.write("\n")
    f.write(hdr_array(f"k{tag}Win1Intr", windows[1]["intrinsics"]))
    f.write("\n")
    if len(windows) > 2:
        f.write(hdr_array(f"k{tag}Win2PoseEnc", windows[2]["pose_enc"]))
        f.write("\n")
        f.write(hdr_array(f"k{tag}Win2Depth", windows[2]["depth"]))
        f.write("\n")
        f.write(hdr_array(f"k{tag}Win2Conf", windows[2]["depth_conf"]))
        f.write("\n")
        f.write(hdr_array(f"k{tag}Win2C2w", windows[2]["c2w"]))
        f.write("\n")
        f.write(hdr_array(f"k{tag}Win2Intr", windows[2]["intrinsics"]))
        f.write("\n")
    for ai, (s_ab, R_ab, t_ab) in enumerate(aligns):
        f.write(f"constexpr float k{tag}Align{ai}S = {s_ab:.9e}f;\n")
        f.write(hdr_array(f"k{tag}Align{ai}R", R_ab))
        f.write("\n")
        f.write(hdr_array(f"k{tag}Align{ai}T", t_ab))
        f.write("\n")
    for wi, win in enumerate(warped):
        if wi == 0:
            continue  # warp is identity for the first window
        f.write(hdr_array(f"k{tag}Warped{wi}PoseEnc", win["pose_enc"]))
        f.write("\n")
        f.write(hdr_array(f"k{tag}Warped{wi}Depth", win["depth"]))
        f.write("\n")
        f.write(hdr_array(f"k{tag}Warped{wi}C2w", win["c2w"]))
        f.write("\n")
    f.write(f"constexpr int k{tag}MergedFrames = {merged['pose_enc'].shape[0]};\n")
    f.write(hdr_array(f"k{tag}MergedPoseEnc", merged["pose_enc"]))
    f.write("\n")
    f.write(hdr_array(f"k{tag}MergedDepth", merged["depth"]))
    f.write("\n")
    f.write(hdr_array(f"k{tag}MergedC2w", merged["c2w"]))
    f.write("\n")
    f.write(hdr_array(f"k{tag}MergedIntr", merged["intrinsics"]))
    f.write("\n\n")


def main():
    f = sys.stdout
    f.write("// GENERATED by gen_stitch_golden.py — do not edit.\n")
    f.write("// Golden values of the official windowed pipeline (ggml_demo.py\n")
    f.write("// numpy port, upstream-verified) on deterministic synthetic data.\n")
    f.write("#pragma once\n\n#include <limits>\n\n")
    f.write(f"constexpr int kGoldenW = {OUT_W};\n")
    f.write(f"constexpr int kGoldenH = {OUT_H};\n\n")

    # quat<->mat round-trip cases.
    qrng = np.random.default_rng(7)
    q = qrng.normal(size=(8, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 3] < 0] *= -1.0
    f.write("constexpr int kQuatCases = 8;\n")
    f.write(hdr_array("kQuatIn", q))
    f.write("\n")
    mats = np.stack([ggml_demo._quat_to_mat(qi) for qi in q])
    f.write(hdr_array("kQuatMat", mats))
    f.write("\n")
    back = np.stack([ggml_demo._mat_to_quat(m) for m in mats])
    f.write(hdr_array("kQuatBack", back))
    f.write("\n\n")

    rng = np.random.default_rng(42)
    # Two-window chain (also covers the degenerate NaN/eps depth mask).
    emit_chain(f, rng, 20, 16, 8, "TwoWin")
    # Three-window chain exercising warped[-1] chained alignment.
    emit_chain(f, rng, 30, 16, 8, "ThreeWin")


if __name__ == "__main__":
    main()
