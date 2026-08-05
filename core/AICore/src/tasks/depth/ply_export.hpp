#pragma once
#include "gs_adapter.hpp"
#include <string>

namespace aicore {
namespace depth {

// Write world-space 3D Gaussians as a standard 3DGS (INRIA gaussian-splatting)
// binary-little-endian .ply. Vertex properties (in order):
//   x y z              float   gaussian means (world space)
//   f_dc_0 f_dc_1 f_dc_2 float SH DC term (harmonics[:, :, 0])
//   opacity            float   stored as logit (inverse-sigmoid); viewers sigmoid it
//   scale_0..2         float   stored as log(scale); viewers exp() them
//   rot_0..3           float   quaternion wxyz (viewers normalize)
// Returns false on I/O error.
bool write_gaussian_ply(const std::string& path, const Gaussians& g);

} // namespace depth
} // namespace aicore
