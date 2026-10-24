// Native CUDA implementation of SAM 3D Objects' optimized PBR texture bake.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "asset_io.hpp"
#include "gaussian_renderer.hpp"

namespace sam3d {

struct TextureBakeConfig {
    int texture_size = 1024;
    int steps = 2500;
    float learning_rate = 0.01f;
    float eta_min = 1.0e-5f;
    float tv_weight = 0.01f;
    uint32_t random_seed = 0;
    float near_plane = 0.1f;
    float far_plane = 10.0f;
    // The production contract applies OpenCV Telea after optimization. Tests
    // may disable it to compare the optimizer result before hole repair.
    bool apply_telea = true;
};

// Return the learning rate used by a given Adam update.  The official Python
// loop performs update zero at the configured initial rate, then assigns the
// cosine value for completed update N as the rate for update N + 1.
float texture_bake_learning_rate_for_update(const TextureBakeConfig& config,
                                            int completed_updates);

// Frozen per-view nvdiffrast contract used to diagnose the native bake before
// its Adam/TV loop. Values retain nvdiffrast's bottom-left tensor orientation.
struct TextureBakeRaster {
    int width = 0;
    int height = 0;
    std::vector<float> uv;
    std::vector<float> uv_derivatives;
    std::vector<uint8_t> coverage;
};

bool rasterize_texture_bake_view_official_cuda(const NativeMesh& mesh,
                                               const GaussianCamera& camera,
                                               const TextureBakeConfig& config,
                                               int width, int height,
                                               TextureBakeRaster& result,
                                               std::string& error);

// Matches postprocessing_utils.bake_texture(mode="opt"): source observations
// are transformed to the nvdiffrast bottom-left convention, sampled with
// linear-mipmap-linear filtering, optimized for 2500 Adam/TV steps, flipped
// back, and finally repaired with OpenCV Telea. `hole_mask` is optional and
// receives the final UV-atlas holes in the same orientation used by official
// cv2.inpaint (non-zero means a missing texel).
bool bake_texture_official_cuda(const NativeMesh& mesh,
                                const std::vector<RgbaImage>& observations,
                                const std::vector<GaussianCamera>& cameras,
                                const TextureBakeConfig& config,
                                RgbaImage& texture,
                                std::vector<uint8_t>* hole_mask,
                                std::vector<float>* optimized_texture,
                                std::string& error);

}  // namespace sam3d
