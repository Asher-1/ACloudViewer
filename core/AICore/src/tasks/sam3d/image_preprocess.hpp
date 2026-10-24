// Native implementation of the image, mask, and point-map preprocessing
// contract in checkpoints/hf/pipeline.yaml.  This layer deliberately does
// not depend on Python, Torch, or a ggml backend.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "asset_io.hpp"

namespace sam3d {

struct NativePreprocessConfig {
    int target_size = 518;
    float crop_box_size_factor = 1.2f;
};

// Tensor storage is contiguous NCHW with N=1.  The SAMT writer records the
// corresponding ggml dimensions [W, H, C, 1].
struct NativeConditionInputs {
    int width = 0;
    int height = 0;
    std::vector<float> image;               // [3, H, W], object crop
    std::vector<float> mask;                // [1, H, W], object crop
    std::vector<float> pointmap;            // [3, H, W], object crop
    std::vector<float> rgb_image;           // [3, H, W], full image
    std::vector<float> rgb_image_mask;      // [1, H, W], full image
    std::vector<float> rgb_pointmap;        // [3, H, W], full image
    std::array<float, 3> pointmap_scale{};
    std::array<float, 3> pointmap_shift{};
};

// Match InferencePipelinePointMap.preprocess_image with the configured
// ObjectCentricSSI and the ss_preprocessor transform sequence. `pointmap`
// must be camera-transformed, CHW storage, and share the input image size.
bool preprocess_ss_conditions(const RgbaImage& rgba, const std::vector<float>& pointmap,
                              int pointmap_width, int pointmap_height,
                              const NativePreprocessConfig& config,
                              NativeConditionInputs& output, std::string& error);

// Write only the runtime condition tensors. Reference-only metadata and
// Python-generated artifacts are intentionally excluded from this directory.
bool write_ss_conditions(const std::string& directory, const NativeConditionInputs& input,
                         std::string& error);

}  // namespace sam3d
