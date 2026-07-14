#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace da {

// World points produced by back-projecting per-frame depth into a shared world
// frame, mirroring the reference `_depths_to_world_points_with_colors`.
// All parallel arrays are indexed by point (0..num_points-1) in the order the
// points were appended: frames outer, then row-major pixel order (v outer,
// u inner) within each frame, keeping only valid pixels.
struct WorldPoints {
    std::vector<float> xyz;   // 3*num_points (x,y,z per point)
    std::vector<uint8_t> rgb; // 3*num_points (r,g,b per point)
    std::vector<int> frame;   // num_points: source frame index
    std::vector<int> u, v;    // num_points: source pixel column/row
};

// Back-project depth maps into world space with per-pixel colors.
//   depth, conf : N*H*W row-major (frame, row, col).
//   K           : per-frame 3x3 intrinsics, row-major.
//   ext_w2c     : per-frame 4x4 world-to-camera extrinsics, row-major.
//   images_u8   : per-frame pointer to H*W*3 RGB uint8 image.
// A pixel is valid when isfinite(d) && d>0 && conf>=conf_thr. For each valid
// pixel: ray = inv(K) @ [u,v,1]; Xc = ray*d; Xw = (inv(ext) @ [Xc;1])[:3].
WorldPoints back_project(const std::vector<float>& depth,
                         const std::vector<float>& conf,
                         const std::vector<std::array<float, 9>>& K,
                         const std::vector<std::array<float, 16>>& ext_w2c,
                         const std::vector<const uint8_t*>& images_u8,
                         int H, int W, int N, float conf_thr);

// numpy.percentile with linear interpolation: sort a copy, take index
// q_percent/100*(n-1), interpolate between floor/ceil. Does not mutate input.
double percentile_linear(const std::vector<float>& v, double q_percent);

// 3x3 inverse (row-major). Returns false if singular (out left unchanged).
bool inv3(const std::array<float, 9>& m, std::array<float, 9>& out);

// 4x4 inverse (row-major). Returns false if singular (out left unchanged).
bool inv4(const std::array<float, 16>& m, std::array<float, 16>& out);

// Rotation matrix (row-major 3x3) -> COLMAP-order quaternion (qw,qx,qy,qz)
// with qw>=0, ported from read_write_model.py's rotmat2qvec.
std::array<float, 4> rotmat2qvec(const std::array<float, 9>& R);

} // namespace da
