// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Pure geometry helpers that turn the typed MapAnything inference outputs
// into a world-frame colored point cloud. Qt-free and AICore-free so the
// math is unit-testable without a loaded model.
//
// Camera conventions (ported from the official Python side, see the
// upstream vggt_omega/utils/pose_enc.py + rotation.py shipped in
// map-anything-ggml):
//   - pose_enc = [T(3), quat(4) scalar-LAST xyzw, fov_h(rad), fov_w(rad)];
//     the extrinsic [R|T] is camera-from-world, so world = R^T * (cam - T).
//   - pi3-family pose = camera-TO-world 4x4 row-major.
//   - intrinsics from FoV: fx = (W/2)/tan(fov_w/2), fy = (H/2)/tan(fov_h/2),
//     cx = W/2, cy = H/2 over the DENSE output grid.

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace MapAnythingGeometry {

/** Rotation matrix (row-major 3x3) from a scalar-last (xyzw) quaternion —
 *  the official vggt_omega/utils/rotation.py quat_to_mat. */
void quat_xyzw_to_mat3(const float* q, float* R);

/** Camera-from-world rotation/translation from a VGGT-family pose_enc row
 *  (9 floats): world = R^T * (cam - T). */
void pose_enc_to_w2c(const float* pose9, float* R, float* t);

/** fx, fy, cx, cy from the VGGT-family FoV pair over a W x H grid. */
void fov_to_intrinsics(
        float fov_h, float fov_w, int width, int height, float* out_k);

/** Unproject one depth pixel through K to the camera frame, then into the
 *  world via the inverse of a camera-from-world [R|t]. */
void unproject_depth_pixel(const float* R,
                           const float* t,
                           const float* k,
                           int u,
                           int v,
                           float depth,
                           float* out_world);

/** Sigmoid (pi3-family raw-logit confidence normalization). */
float sigmoid(float x);

/** Transform a camera-frame point by a camera-TO-world 4x4 row-major pose
 *  (pi3 family): world = M[:3,:3] * cam + M[:3,3]. */
void transform_by_c2w(const float* pose16, const float* cam, float* out_world);

// ---- official end-to-end filter stack (upstream e2e_full_scene.py) -------

/** Official visual_util.depth_edge: per-pixel 1 when the 3x3 depth window
 *  jumps more than \p rtol (relative to |depth|) - depth discontinuities
 *  (flying pixels at object boundaries) must not enter the cloud.
 *  \p out_mask has h*w entries (1 = edge, drop it). */
void depth_edge_mask(
        const float* depth, int h, int w, float rtol, uint8_t* out_mask);

/** Official percentile confidence floor over the SCENE-WIDE raw-confidence
 *  population: max(2.0, percentile(conf, p)). \p p in [0,100]. */
float official_conf_floor(const float* conf, size_t n, float percentile);

/** Confidence-priority voxel merge (official e2e pipeline): per voxel keep
 *  the HIGHEST-conf observation - sharpens surfaces while staying
 *  deterministic. \p pts/\p cols/\p confs are n-sized; survivors are
 *  compacted in-place into the same buffers and the new count returned. */
size_t conf_priority_voxel(float* pts,
                           uint8_t* cols,
                           float* confs,
                           size_t n,
                           float voxel_size,
                           std::vector<size_t>* survivor_indices = nullptr);

/** Closed-form Horn Sim3: dst ~= s * R * src + t. Returns s; R (3x3
 *  row-major) and t (3) are written. Returns 0 on success, -1 on degenerate
 *  input (all points identical / non-finite). */
int horn_sim3(const float* src,
              const float* dst,
              size_t n,
              float* out_s,
              float* out_R,
              float* out_t);

/** Robust Horn Sim3: 2 rejection rounds dropping residuals beyond
 *  median + 3*1.4826*MAD. Same contract as horn_sim3. */
int robust_sim3(const float* src,
                const float* dst,
                size_t n,
                float* out_s,
                float* out_R,
                float* out_t);

/** Apply a Sim3 to n points: out = s * R * p + t (row-major R, in-place
 *  safe when out == pts). */
void apply_sim3(const float* R,
                const float* t,
                float s,
                const float* pts,
                size_t n,
                float* out);

}  // namespace MapAnythingGeometry
