// Native raw-image MoGe inference matching MoGeModel.infer().
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "asset_io.hpp"

namespace sam3d {

class Backend;
class GGUFModel;

struct MogeInferenceOptions {
    int num_tokens = 2500;
    bool apply_mask = true;
    bool force_projection = false;
    // Regression-only boundary captures. Disabled for ordinary inference so
    // full-resolution runs do not retain another point-map-sized buffer.
    bool capture_intermediates = false;
    // Regression-only DINO block captures at the internal token resolution.
    // These are intentionally opt-in because all 24 MLP intermediates are
    // substantial even when the final point map is small.
    bool capture_block_outputs = false;
};

struct MogeInferenceResult {
    int width = 0;
    int height = 0;
    // HWC, after the official MoGe shift/mask postprocess.
    std::vector<float> points_moge;
    // CHW, transformed from R3 into the convention consumed by the SAM 3D
    // point-map preprocessor. Invalid values remain IEEE infinity.
    std::vector<float> pointmap_pytorch3d;
    // HWC scalar tensors at the original image resolution.
    std::vector<float> mask_logits;
    std::vector<float> mask_probability;
    std::vector<float> depth;
    std::vector<uint8_t> mask;
    std::array<float, 9> intrinsics{};
    float focal = 0.0f;
    float shift = 0.0f;
    // HWC captures from MoGeModel.forward(), populated only when requested.
    int resized_width = 0;
    int resized_height = 0;
    std::vector<float> resized_rgb;
    std::vector<float> forward_points;
    int backbone_tokens = 0;
    int backbone_hidden = 0;
    int backbone_mlp_hidden = 0;
    int backbone_image_width = 0;
    int backbone_image_height = 0;
    std::vector<float> backbone_image;
    std::vector<float> backbone_patch_tokens;
    std::vector<float> backbone_position_tokens;
    std::vector<float> backbone_input;
    std::vector<float> backbone_attention_context;
    std::vector<float> backbone_q;
    std::vector<float> backbone_k;
    std::vector<float> backbone_v;
    std::vector<std::vector<float>> backbone_attention_projection;
    std::vector<std::vector<float>> backbone_attention;
    std::vector<std::vector<float>> backbone_mlp_fc1;
    std::vector<std::vector<float>> backbone_mlp_gelu;
    std::vector<std::vector<float>> backbone_mlp;
    std::vector<std::vector<float>> backbone_blocks;
};

// Exact integer target dimensions used by MoGeModel.forward().
bool moge_resized_dimensions(int original_width, int original_height, int num_tokens,
                             int& resized_width, int& resized_height, std::string& error);

// Execute the official MoGe raw-image inference contract without Python or
// Torch. The supplied backend owns all ggml allocations and can be CPU, CUDA,
// or Vulkan.
bool run_moge_inference(const RgbaImage& image, const GGUFModel& model, Backend& backend,
                        const MogeInferenceOptions& options, MogeInferenceResult& output,
                        std::string& error);

}  // namespace sam3d
