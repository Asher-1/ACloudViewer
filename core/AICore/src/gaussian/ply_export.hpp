// PLY export for SIBR Gaussian viewer compatibility.
// Converts free-splatter's activated Gaussian parameters to the PLY format
// expected by SIBR's GaussianView (3DGS rasterizer).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace aicore {
namespace gaussian {

// Export gaussians as a SIBR-compatible PLY file.
// gaussians: n_views * height * width * gaussian_channels float32 (activated).
//   Channel layout (scene, 23ch): xyz[0:3] SH[3:15] opacity[15] scale[16:19] rot[19:23].
// sh_degree: SH degree of the model (1 for free-splatter scene/object).
// opacity_threshold: prune gaussians with opacity <= threshold.
// Returns true on success, false on error (sets err).
bool export_ply_sibr(const float* gaussians, int32_t n_views,
                     int32_t height, int32_t width,
                     int32_t gaussian_channels, int32_t sh_degree,
                     float opacity_threshold,
                     const std::string& out_path, std::string& err);

// Same as export_ply_sibr but returns binary PLY bytes (for in-memory SIBR loading).
bool export_ply_sibr_to_buffer(const float* gaussians, int32_t n_views,
                               int32_t height, int32_t width,
                               int32_t gaussian_channels, int32_t sh_degree,
                               float opacity_threshold,
                               std::vector<uint8_t>& out, std::string& err);

} // namespace gaussian
} // namespace aicore
