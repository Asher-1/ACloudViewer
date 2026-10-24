// Native CUDA Gaussian renderer matching SAM 3D Objects' Inria render path.
#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "asset_io.hpp"

namespace sam3d {

// The values consumed by CudaRasterizer::Rasterizer::forward. They are already
// activated exactly as Gaussian.get_xyz/get_features/get_opacity/get_scaling/
// get_rotation expose them to the official renderer.
struct GaussianSplatSet {
    std::vector<float> positions;   // xyz, P * 3
    std::vector<float> sh0;         // RGB SH coefficient 0, P * 3
    std::vector<float> opacities;   // sigmoid(opacity), P
    std::vector<float> scales;      // exp(log_scale), P * 3
    std::vector<float> rotations;   // normalized wxyz quaternion, P * 4

    size_t size() const { return opacities.size(); }
    bool valid() const;
};

// Row-major OpenCV extrinsics/intrinsics, matching the tensors returned by
// render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics.
struct GaussianCamera {
    std::array<float, 16> extrinsics{};
    std::array<float, 9> intrinsics{};
};

struct GaussianRenderConfig {
    int width = 1024;
    int height = 1024;
    float near_plane = 0.8f;
    float far_plane = 1.6f;
    float radius = 2.0f;
    float fov_degrees = 40.0f;
    // gsplat parity: the official render paths all go through the gsplat
    // backend, where GaussianRenderer.pipe.kernel_size is never consumed and
    // the rasterizer applies its default eps2d = 0.3 screen-space low-pass.
    // The vendored fork reads this value directly (see the fork's
    // UPSTREAM.md "Local deltas"), so 0.3 reproduces the official dilation.
    float kernel_size = 0.3f;
};

// Read the binary-little-endian PLY emitted by Gaussian.save_ply or the native
// session writer. The PLY stores inverse activations; this loader applies the
// same sigmoid/exp/quaternion normalization used before rasterization.
bool load_gaussian_splat_ply(const std::string& path, GaussianSplatSet& splats,
                             std::string& error);

// Exact sphere_hammersley_sequence + yaw_pitch_r_fov camera construction used
// by render_multiview. The default call produces the official 100 cameras.
std::vector<GaussianCamera> make_gaussian_hammersley_cameras(
    int view_count, const GaussianRenderConfig& config, std::string& error);

// Render every supplied camera with the unmodified Mip-Splatting Inria CUDA
// rasterizer. Outputs are top-left-origin RGBA8 images, matching the NumPy
// image orientation returned by the official Python render_multiview routine.
bool render_gaussian_views_cuda(const GaussianSplatSet& splats,
                                const std::vector<GaussianCamera>& cameras,
                                const GaussianRenderConfig& config,
                                std::vector<RgbaImage>& images,
                                std::string& error);

}  // namespace sam3d
