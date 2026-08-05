// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Weights, graph build and forward pass for the "free-splatter" architecture
// (pieces 1-3: patch tokenizer -> self-attention transformer -> Gaussian head).
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "backend.hpp"
#include "gguf_loader.h"

namespace aicore {
namespace gaussian {

// One pre-norm self-attention transformer block. All Linears are bias-free; the
// only norm parameter is the LayerNorm gamma (no beta).
struct layer_weights {
    ggml_tensor* attn_norm = nullptr;  // norm1.weight
    ggml_tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    ggml_tensor* ffn_norm = nullptr;  // norm2.weight
    ggml_tensor *ffn_up = nullptr, *ffn_down = nullptr;
};

// Sink for a named intermediate, given as row-major [n_rows, n_cols] f32.
// Called once per tap during forward and freed immediately, so a full-depth
// tapped run never holds all activations (~14 GB) in host memory at once.
using tap_sink = std::function<void(const std::string& name,
                                    const float* data,
                                    int64_t n_rows,
                                    int64_t n_cols)>;

struct model {
    model_file file;
    engine_backend be;

    // Piece 1 (tokenizer)
    ggml_tensor* patch_embed =
            nullptr;  // conv kernel, ne=[KW,KH,IC,OC]=[8,8,3,1024]
    ggml_tensor* pos_embed = nullptr;  // [n_embd, tokens_per_view]
    ggml_tensor* view_ref = nullptr;   // [n_embd] reference-view embedding
    ggml_tensor* view_src = nullptr;   // [n_embd] source-view embedding
    // Piece 2 weights
    std::vector<layer_weights> layers;
    // Piece 3 (head)
    ggml_tensor* output_norm = nullptr;  // final norm.weight
    ggml_tensor* unpatch_w = nullptr;    // [n_embd, ph*pw*gaussian_channels]
    ggml_tensor* unpatch_b = nullptr;    // [ph*pw*gaussian_channels]

    ggml_backend_buffer_t weights_buf =
            nullptr;  // CPU zero-copy wrapper or device buffer
    ggml_context* device_ctx = nullptr;  // device tensor mirror (non-CPU)

    std::string error;

    bool load(const std::string& gguf_path,
              const std::string& device,
              int n_threads);
    void release();
    ~model() { release(); }

    const hparams& hp() const { return file.hp; }

    // Forward over `images` (validated, view-major NCHW f32, n_views*C*H*W).
    // `out` is filled with the activated, render-ready Gaussian params,
    // row-major [n_views*H*W, gaussian_channels]. If `sink` is set, every named
    // tap is streamed to it (slower: marks them as graph outputs).
    bool forward(const float* images,
                 int32_t n_views,
                 std::vector<float>& out,
                 const tap_sink& sink = {});

private:
    bool map_tensors();
    bool realize_weights();
};

}  // namespace gaussian
}  // namespace aicore
