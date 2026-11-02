// PLY export for the predicted gaussians — byte-compatible with
// scripts/rendering/gs.py GaussianRenderer.save_ply (binary little endian,
// float32 properties, standard 3DGS field set).
//
// Field transforms (must match gs.py exactly):
//   filter:  opacity >= threshold (before any transform)
//   f_dc:    sh degree 0: (color - 0.5) * (1 / 0.28209479177387814f) (CUDA
//            scalar division = multiply by the f32 reciprocal);
//            degree > 0: color[.., 0:3] as-is (already dc/0.28209...)
//   f_rest:  degree > 0 only: sh[:, 1:].transpose(1, 2) -> channel-major
//            f_rest[c*(K-1) + k-1] = color[3 + (k-1)*3 + c]
//   header:  plyfile layout, no comment lines
//   opacity: logit(clamp(o, 1e-8, 1-1e-8))
//   scale:   log(scale + 1e-8)
//   rot:     as-is (w-last XYZW, matching the gaussian buffer)
#pragma once

#include "querysplat_activations.hpp"

#include <string>
#include <vector>

namespace qs {

// Writes the opacity-filtered gaussian PLY. Returns false on IO failure.
bool save_gaussian_ply(const GaussianOutput& g, const std::string& path,
                       float opacity_threshold);

// Raw gaussian dump for parity gates: N * (11 + color_dim) f32.
bool save_gaussians_bin(const GaussianOutput& g, const std::string& path);

// Official save_colored_pointcloud parity: vertex-only colored PLY (x/y/z
// f4 + red/green/blue u1) of the gaussian centers with opacity >= threshold.
// RGB = color_to_rgb: sh0 -> color[0:3]; sh>0 -> color[0:3]*C0 + 0.5 clamped.
bool save_colored_pointcloud_ply(const GaussianOutput& g,
                                 const std::string& path, float threshold);

}  // namespace qs
