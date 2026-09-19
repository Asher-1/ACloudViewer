// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Windowed long-sequence orchestration math for LingBot-Map reconstruction.
//
// Pure float math port of the official Python windowed pipeline at the
// windowed defaults (keyframe_interval=1, every frame a keyframe):
// gct_stream_window.py's `_pairwise_alignment` / `_warp_predictions` /
// `_stitch_windows` as verified by the upstream ggml_demo.py numpy port
// (scripts/verify_windowed.py: per-window raw pose 7.9e-05 / depth 5.3e-04
// against the official PyTorch `inference_windowed`).
//
// Deliberately Qt-free and dependency-free so it can be unit tested without
// a GUI or the AICore runtime. The engine (AICore) stays untouched: each
// window is the validated streaming primitive run over a fresh KV cache
// (aicore_lingbot_stream_reset + aicore_lingbot_infer_stream).

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace lingbot_stitch {

//! pose_enc layout: [t(0:3), quat_xyzw scalar-last(3:7), fov_y(7), fov_x(8)],
//! w2c semantics (the decoder image of [quat_mat | pe[:3]]).
constexpr int kPoseEncSize = 9;
//! Row-major 4x4 camera-to-world sidecar emitted by the engine.
constexpr int kC2WSize = 16;
//! [fx, fy, cx, cy] in pixels of the processed frame.
constexpr int kIntrinsicsSize = 4;

//! One window's predictions, frame-major. `start` is the global frame index
//! of local frame 0 in the source sequence.
struct WindowData {
    int start = 0;
    int frames = 0;
    int width = 0;
    int height = 0;
    std::vector<float> poseEnc;    /**< [frames][kPoseEncSize] */
    std::vector<float> depth;      /**< [frames][width * height] */
    std::vector<float> depthConf;  /**< [frames][width * height] */
    std::vector<float> c2w;        /**< [frames][kC2WSize] */
    std::vector<float> intrinsics; /**< [frames][kIntrinsicsSize] */
};

//! Window list plan following the official fixed-interval rules.
struct WindowPlan {
    std::vector<std::pair<int, int>> windows; /**< [start, end) ranges */
    int effWindow = 0;                        /**< actual frames per window */
    int effOverlap = 0;                       /**< overlap in actual frames */
};

//! Resolve the windowing parameters exactly like run_ggml_windowed /
//! inference_windowed's fixed-interval branch at keyframe_interval=1:
//! eff_window = min(scale + max(window_size - scale, 0), total),
//! eff_overlap = min(overlap, total - 1) for total > 1 (0 otherwise).
WindowPlan planWindows(int total,
                       int windowSize,
                       int overlapSize,
                       int scaleFrames);

//! Window list per inference_windowed's fixed-interval branch (public for
//! testing; planWindows is the entry point used by the worker).
std::vector<std::pair<int, int>> splitWindows(int total,
                                              int effWindow,
                                              int effOverlap);

//! Similarity (s, R, t) mapping `curr` into `prev`'s coordinate frame,
//! estimated on the overlap. The anchor is the last overlap frame and the
//! depth-ratio scale is the median over every finite overlap pixel ratio
//! (all-true paired-keyframe mask at keyframe_interval=1).
struct Similarity {
    float s = 1.0f;
    float R[9] = {1.f, 0.f, 0.f, 0.f, 1.f,
                  0.f, 0.f, 0.f, 1.f}; /**< row-major 3x3 */
    float t[3] = {0.f, 0.f, 0.f};
};

Similarity pairwiseAlignment(const WindowData& prev,
                             const WindowData& curr,
                             int overlap);

//! Apply the official similarity warp in place: pose_enc center/quat,
//! depth *= s, and the c2w sidecar as R'_c2w = R_c2w @ R^T,
//! t'_c2w = s * t_c2w - R_c2w @ (R^T t) so merged c2w stays the exact
//! decoder image of merged pose_enc.
void warpWindow(WindowData& win, const Similarity& sim);

//! Concatenate per-window predictions de-duplicating the overlaps: every
//! non-final window contributes [0, frames - overlap). The merged window's
//! `start` is the first contributing window's start.
WindowData stitchWindows(const std::vector<WindowData>& windows, int overlap);

//! Per-merged-frame source table [(windowIndex, localFrameIndex)] — the
//! exact slice table stitchWindows consumes, so metadata (source file, sky
//! mask state) can be resolved for each merged frame by the caller.
std::vector<std::pair<int, int>> contributionTable(
        const std::vector<WindowData>& windows, int overlap);

//! Scalar-last (x, y, z, w) quaternion -> row-major 3x3 rotation matrix
//! (port of lingbot_map.utils.rotation.quat_to_mat / PyTorch3D).
void quatToMat(const float q[4], float outR[9]);

//! Row-major 3x3 rotation matrix -> scalar-last quaternion with w >= 0
//! (port of lingbot_map.utils.rotation.mat_to_quat).
void matToQuat(const float m[9], float outQ[4]);

}  // namespace lingbot_stitch
