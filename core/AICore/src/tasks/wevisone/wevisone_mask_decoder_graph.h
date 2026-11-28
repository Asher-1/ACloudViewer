// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ggml.h"
#include "tasks/wevisone/wevisone_types.h"

namespace wwv {

// Explicit libtorch path for the bit-exact-with-torch float log (vmsLn).
// Set from the task options BEFORE the first decoder run; empty keeps the
// standard pip-torch layout auto-discovery. Replaces the upstream
// WWV_MKL_LOG_LIB environment read (options-only contract).
void set_mkl_log_lib_path(const std::string& path);

// Inputs handed to the graph as pre-filled leaves.
struct MaskDecoderInputs {
    ggml_tensor* cnn_image = nullptr;  // (S, S, 3, 1) f32 in [0, 1]
    ggml_tensor* levels[4] = {nullptr, nullptr, nullptr, nullptr};
    // (Wg, Hg, C, 1) f32 row-major maps
    ggml_tensor* t_seg = nullptr;    // (llm_dim, Q) f32 LM hidden states
    ggml_tensor* boxes01 = nullptr;  // (4, Q) f32 normalized xyxy
    // host copy of the (4, Q) box values: lets the builder precompute the
    // BoxFourierEncoder log terms in double precision instead of the
    // in-graph ggml_log (whose 1-2 ulp divergence from torch's Sleef u10
    // is amplified by the Fourier phase into the query position band).
    const float* boxes01_host = nullptr;
    // stages-in isolation switch: precomputed query position (d, Q) that
    // replaces the in-graph box-Fourier branch when non-null.
    ggml_tensor* query_position = nullptr;
};

struct MaskDecoderOutputs {
    ggml_tensor* mask_logits_s1 = nullptr;  // (S, S, 1, Q)
    ggml_tensor* mask_logits_s2 = nullptr;  // (S/2, S/2, 1, Q)
    ggml_tensor* mask_logits_coarse = nullptr;
};

// Weight set resolved from the decoder GGUF by the runtime (all leaves of
// the weight ggml_context; missing optional weights may be nullptr only
// where the Python module has none).
struct MaskDecoderWeights {
    // SameScaleMSDAEncoder
    ggml_tensor* input_proj_w[4] = {nullptr, nullptr, nullptr, nullptr};
    ggml_tensor* input_proj_gn_w[4] = {nullptr, nullptr, nullptr, nullptr};
    ggml_tensor* input_proj_gn_b[4] = {nullptr, nullptr, nullptr, nullptr};
    ggml_tensor* level_embed = nullptr;  // (4, 192)
    // Host-built constant: sine 2D position embedding + level_embed, laid
    // out (d, levels*hw) in the same token order as the level concat.
    ggml_tensor* encoder_pos_leaf = nullptr;
    // encoder layers (3 x): {depthwise (192,1,3,3), pointwise (192,192),
    //  norm1 w, norm1 b, linear1 w, linear1 b, linear2 w, linear2 b,
    //  norm2 w, norm2 b}
    std::vector<std::array<ggml_tensor*, 10>> encoder_layers;
    // MaskQueryBuilder
    ggml_tensor* content_proj_w = nullptr;
    ggml_tensor* content_proj_b = nullptr;
    ggml_tensor* content_norm_w = nullptr;
    ggml_tensor* content_norm_b = nullptr;
    ggml_tensor* box_mlp_w1 = nullptr;  // (192, 256)
    ggml_tensor* box_mlp_b1 = nullptr;
    ggml_tensor* box_mlp_w2 = nullptr;  // (192, 192)
    ggml_tensor* box_mlp_b2 = nullptr;
    ggml_tensor* box_freqs = nullptr;  // (32, 1) host constant 2^arange
    ggml_tensor* position_norm_w = nullptr;
    ggml_tensor* position_norm_b = nullptr;
    // transformer layers (6 x): {self in_proj w, b, self out w, b,
    //  norm1 w, b, msda val w, b, msda off w, b, msda att w, b,
    //  msda out w, b, norm2 w, b, ffn1 w, b, ffn2 w, b, norm3 w, b} = 20
    std::vector<std::array<ggml_tensor*, 20>> transformer_layers;
    // norm3 (post-FFN) per layer (w, b).
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> transformer_norms;
    // pixel fusion
    ggml_tensor* fpn_lateral_w[4] = {nullptr, nullptr, nullptr, nullptr};
    std::vector<std::array<ggml_tensor*, 6>> fpn_output;  // conv w, gn w/b
    ggml_tensor* up_conv0_w = nullptr;  // PixelShuffleStage 0 (768, 192, 3, 3)
    ggml_tensor* up_conv0_b = nullptr;
    ggml_tensor* up_conv1_w = nullptr;  // PixelShuffleStage 1 (256, 192, 3, 3)
    ggml_tensor* up_conv1_b = nullptr;
    ggml_tensor* s2_up_conv_w =
            nullptr;  // ResidualPixelShuffle (256, 64, 3, 3)
    ggml_tensor* s2_up_conv_b = nullptr;
    ggml_tensor* stem_conv0_w = nullptr;  // (128, 3, 3, 3)
    ggml_tensor* stem_gn0_w = nullptr;
    ggml_tensor* stem_gn0_b = nullptr;
    ggml_tensor* stem_conv1_w = nullptr;  // (64, 128, 3, 3)
    ggml_tensor* stem_gn1_w = nullptr;
    ggml_tensor* stem_gn1_b = nullptr;
    ggml_tensor* fpn_proj_w = nullptr;  // (64, 64, 1, 1)
    ggml_tensor* cnn_proj_w = nullptr;  // (64, 64, 1, 1)
    ggml_tensor* smooth_gn_w = nullptr;
    ggml_tensor* smooth_gn_b = nullptr;
    ggml_tensor* smooth_conv_w = nullptr;  // (64, 64, 3, 3)
    // mask predictor
    ggml_tensor* kernel_w1 = nullptr;  // (192, 192)
    ggml_tensor* kernel_b1 = nullptr;
    ggml_tensor* kernel_w2 = nullptr;  // (65, 192)
    ggml_tensor* kernel_b2 = nullptr;
    // alpha cascade head
    ggml_tensor* cascade_semantic_w = nullptr;  // (48, 64, 3, 3)
    ggml_tensor* cascade_semantic_gn_w = nullptr;
    ggml_tensor* cascade_semantic_gn_b = nullptr;
    ggml_tensor* rgb_s4_w = nullptr;
    ggml_tensor* rgb_s4_gn_w = nullptr;
    ggml_tensor* rgb_s4_gn_b = nullptr;
    ggml_tensor* rgb_s2_w = nullptr;
    ggml_tensor* rgb_s2_gn_w = nullptr;
    ggml_tensor* rgb_s2_gn_b = nullptr;
    ggml_tensor* rgb_s1_w = nullptr;
    ggml_tensor* rgb_s1_gn_w = nullptr;
    ggml_tensor* rgb_s1_gn_b = nullptr;
    ggml_tensor* coarse_s4_w = nullptr;
    ggml_tensor* coarse_s4_gn_w = nullptr;
    ggml_tensor* coarse_s4_gn_b = nullptr;
    ggml_tensor* coarse_s2_w = nullptr;
    ggml_tensor* coarse_s2_gn_w = nullptr;
    ggml_tensor* coarse_s2_gn_b = nullptr;
    ggml_tensor* previous_s2_w = nullptr;
    ggml_tensor* previous_s2_gn_w = nullptr;
    ggml_tensor* previous_s2_gn_b = nullptr;
    ggml_tensor* previous_s1_w = nullptr;
    ggml_tensor* previous_s1_gn_w = nullptr;
    ggml_tensor* previous_s1_gn_b = nullptr;
    ggml_tensor* fuse_s4_w = nullptr;   // (48, 144, 1, 1)
    ggml_tensor* fuse_s2_w = nullptr;   // (48, 240, 1, 1)
    ggml_tensor* fuse_s1_w = nullptr;   // (48, 192, 1, 1)
    ggml_tensor* up_s4_s2_w = nullptr;  // ConvT (48, 48, 4, 4)
    ggml_tensor* up_s2_s1_w = nullptr;  // ConvT (48, 48, 4, 4)
    ggml_tensor* query_s4_w = nullptr;  // (96, 192)
    ggml_tensor* query_s4_b = nullptr;
    ggml_tensor* query_s2_w = nullptr;
    ggml_tensor* query_s2_b = nullptr;
    ggml_tensor* query_s1_w = nullptr;
    ggml_tensor* query_s1_b = nullptr;
    ggml_tensor* output_s4_w = nullptr;  // (1, 48, 1, 1)
    ggml_tensor* output_s4_b = nullptr;
    ggml_tensor* output_s2_w = nullptr;
    ggml_tensor* output_s2_b = nullptr;
    ggml_tensor* output_s1_w = nullptr;
    ggml_tensor* output_s1_b = nullptr;
    // residual stacks: per stage, blocks * (gn_w, gn_b, conv_w) x 2
    std::vector<std::array<ggml_tensor*, 12>>
            cascade_blocks;  // 3 stages x 2 blocks
};

// Builds the decoder graph. Q (query count) and the level/grid shapes are
// fixed at build time (per-run graph). The returned graph must be executed
// with a backend that can see every input leaf (all in `ctx0`).
// Host-staged f32 constants created by the builder (uploaded by the caller
// after ggml_gallocr_alloc_graph; cleared on every builder call).
void wwv_clear_f32_leaves();
const std::vector<std::pair<ggml_tensor*, float>>& wwv_f32_leaves();

// PixelShuffle perm-table staging (uploaded with the f32 leaves).
void wwv_clear_perm_staged();
const std::vector<std::pair<ggml_tensor*, std::vector<int32_t>>>&
wwv_perm_staged();

// (1, N) f32 vector leaf staging (uploaded with the f32 leaves).
void wwv_clear_f32vec_staged();
const std::vector<std::pair<ggml_tensor*, std::vector<float>>>&
wwv_f32vec_staged();

ggml_cgraph* build_mask_decoder_graph(
        ggml_context* ctx0,
        const DecoderConfig& cfg,
        int32_t q,
        int32_t grid_w,
        int32_t grid_h,
        int32_t vit_channels,
        int32_t llm_dim,
        const MaskDecoderWeights& w,
        const MaskDecoderInputs& in,
        MaskDecoderOutputs* out,
        std::string* err,
        std::vector<std::pair<std::string, ggml_tensor*>>* debug_taps,
        const char* taps_filter,
        bool taps_direct = false);

}  // namespace wwv
