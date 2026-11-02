// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Shared types for the VLX-Seek vision sub-chain.
//
// Ported from the upstream Asher-1/VLX-Seek-GGML cpp_ggml/src/vlx_types.hpp.
// Every numeric dimension is read from GGUF metadata, never hard-coded: the
// upstream converter writes the values it extracted from the checkpoint, so a
// future 0.6B/3B/Qwen3.8 variant only changes metadata.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace vlx {

// One candidate region in original-image pixel coordinates (x1, y1, x2, y2).
struct Box {
    float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
};

// One spatial feature level in NCHW float layout, as consumed by HFRE.
struct FeatureLevel {
    int channels = 0;
    int height = 0;
    int width = 0;
    std::vector<float> data;  // NCHW, length = channels*height*width
};

// C-RADIOv4-SO400M runtime config (mirrors RADIOConfig / VLX aux config).
struct RadioConfig {
    int patch_size = 16;
    int embed_dim = 1152;      // SO400M width
    int depth = 27;            // blocks
    int num_heads = 16;
    int mlp_hidden = 4304;     // 4304/1152 ratio
    int num_cls_tokens = 3;    // 10B config: 3 cls + 7 registers (skip=10)
    int num_registers = 7;
    int pos_rows = 128;        // CPE max 2048 / patch16 -> 128x128 grid
    int pos_cols = 128;
    int out_indices[4] = {6, 13, 20, 26};  // forward_intermediates indices

    // Input conditioner fallback for GGUFs without the checkpoint buffers:
    // the 10B checkpoint's bf16 norm_mean / norm_std (OpenAI CLIP rounded to
    // bf16). The GGUF tensors radio.input_conditioner.* take precedence.
    float norm_mean[3] = {0.482421875f, 0.45703125f, 0.408203125f};
    float norm_std[3] = {0.26953125f, 0.26171875f, 0.275390625f};

    int num_skip() const { return num_cls_tokens + num_registers; }
};

// HFRE runtime config (mirrors HybridFineGrainedRegionEncoder init args that
// the 10B checkpoint actually sets). The mean-fusion path with
// use_vision_tower_object_feature=true is the configuration the released
// 10B checkpoint uses; other branches exist in Python but are not converted
// (YAGNI until a checkpoint ships them).
struct HfreConfig {
    int roi_output_size = 14;         // 10B: mm_roi_output_size=14
    float aux_spatial_scale = 1.f / 16.f;   // 1 / aux patch_size
    float vt_spatial_scale = 1.f / 32.f;    // 1 / (patch16 * merge2)
    bool add_pos_embedding = true;
    int pos_embedding_dim = 6656;     // mm_object_hidden_size = 4608 aux + 2048 vt
    int vt_object_feature_dim = 4096; // main-tower out_hidden_size (SimpleFPN input)
    bool use_vt_object_feature = true;
    bool use_simple_fpn_vt = true;    // 10B: mm_use_simpleFPN_for_vt=True
    int simple_fpn_out_channels = 512; // 4 levels x 512 = 2048 vt concat
    bool roi_aligned = false;         // torchvision roi_align aligned flag
    std::string combination = "concat"; // mm_object_feature_combination
    int final_dim = 6656;             // concat(4608 aux, 2048 vt)
    int n_threads = 1;                // host RoIAlign threads (split over boxes)
};

// Result of the vision sub-chain for one image.
struct VisionOutput {
    RadioConfig radio;
    std::vector<FeatureLevel> aux_levels;  // 4 NCHW levels (RADIO out_indices)
    FeatureLevel vt_level;                 // main-tower merged feature (NCHW)
};

}  // namespace vlx
