// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// splat.h — the single, shared encoder for an antimatter15 .splat record.
//
// There are two producers of .splat output: the single-run path (write_splat,
// straight from the engine's gaussians) and the accumulated-cloud path
// (write_cloud_splat, from pose::AccumPoint). They MUST encode a gaussian
// identically — same OpenCV->OpenGL convention, same quaternion remap, same
// byte packing, same opacity->alpha. They used to be two copies and drifted:
// the cloud path silently dropped first the rotation/scale, then the opacity
// (rendering a fully-opaque, swirling, blurry soup instead of a proper
// alpha-blended surface).
//
// This is the ONE definition both call, so they cannot diverge again. It is
// pinned byte-for-byte by tests/test_pose.cpp::test_splat_record.
#ifndef FREE_SPLATTER_SPLAT_H
#define FREE_SPLATTER_SPLAT_H

#include <cmath>
#include <cstring>

namespace aicore {
namespace gaussian {

// One 32-byte .splat record: pos[3]f32, scale[3]f32, rgba[4]u8, rot[4]u8
// (w,x,y,z). Inputs are in the engine's OpenCV frame (y down, z forward); `rgb`
// and `opacity` in [0,1]; `quat_wxyz` as (w,x,y,z). Writes exactly 32 bytes to
// `out`.
inline void encode_splat_record(unsigned char out[32],
                                const float pos[3],
                                const float scale[3],
                                const float quat_wxyz[4],
                                const float rgb[3],
                                float opacity) {
    auto u8 = [](float v) -> unsigned char {
        float t = v < 0 ? 0 : (v > 255 ? 255 : v);
        return (unsigned char)t;
    };
    auto c01 = [](float v) -> float {
        return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    };
    // .splat / antimatter15 web viewer only: OpenCV (y down, z forward) ->
    // viewer OpenGL (y up) via 180deg about X = diag(1,-1,-1). Do NOT use this
    // flip for CloudViewer DB, VTK, or COLMAP world coordinates.
    const float p[3] = {pos[0], -pos[1], -pos[2]};
    float q[4] = {-quat_wxyz[1], quat_wxyz[0], -quat_wxyz[3], quat_wxyz[2]};
    const float nrm =
            std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) +
            1e-12f;
    std::memcpy(out + 0, p, 12);
    std::memcpy(out + 12, scale, 12);
    out[24] = u8(c01(rgb[0]) * 255.0f);
    out[25] = u8(c01(rgb[1]) * 255.0f);
    out[26] = u8(c01(rgb[2]) * 255.0f);
    out[27] =
            u8(c01(opacity) * 255.0f);  // opacity -> alpha (NOT forced opaque)
    for (int c = 0; c < 4; c++) out[28 + c] = u8(q[c] / nrm * 128.0f + 128.0f);
}

}  // namespace gaussian
}  // namespace aicore

#endif  // FREE_SPLATTER_SPLAT_H
