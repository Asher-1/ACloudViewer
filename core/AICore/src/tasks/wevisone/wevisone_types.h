// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace wwv {

// Released decoder architecture
// (aicore_wevisone_vl.mask_decoder.WeVisOneMaskDecoder).
struct DecoderConfig {
    // Architecture version; the graph builder refuses anything else.
    std::string architecture_version = "wevisone_vl_v1";
    // Pre-merger ViT layers tapped for the mask decoder.
    std::vector<int32_t> vision_feature_layers = {5, 11, 17, 23};

    int32_t hidden_dim = 192;
    int32_t mask_feature_dim = 64;
    int32_t num_fourier_freqs = 32;

    int32_t msda_encoder_layers = 3;
    int32_t msda_num_levels = 4;
    int32_t msda_num_points = 4;

    int32_t transformer_layers = 6;
    int32_t transformer_heads = 8;
    int32_t transformer_ffn_dim = 768;

    int32_t group_norm_groups = 32;
    int32_t stem_mid_dim = 128;
    int32_t alpha_cascade_dim = 48;
    int32_t alpha_cascade_blocks = 2;
    int32_t cnn_image_size = 1024;
};

// Everything the runtime needs to locate and validate the GGUF model set.
// Populated explicitly from aicore_wevisone_options — no environment access.
struct ModelPaths {
    std::string vlm_gguf;      // merged mmproj + LLM (llama.cpp qwen3vl)
    std::string decoder_gguf;  // mask decoder sidecar (wevisone_mask arch)
};

// A single parsed result box in original-image pixels.
struct ResultBox {
    float x1 = 0.f;
    float y1 = 0.f;
    float x2 = 0.f;
    float y2 = 0.f;
};

}  // namespace wwv
