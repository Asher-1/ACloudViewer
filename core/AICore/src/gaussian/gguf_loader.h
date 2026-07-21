// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// GGUF model file for the "free-splatter" architecture: metadata,
// hyperparameters, tensor access. Written by scripts/convert.py.
#pragma once

#include <ggml.h>
#include <gguf.h>

#include <string>

namespace aicore {
namespace gaussian {

// Hyperparameters of pieces 1-3 (patch tokenizer -> transformer -> Gaussian
// head).
struct hparams {
    int32_t n_layer = 0;      // free-splatter.block_count            (24)
    int32_t n_embd = 0;       // free-splatter.embedding_length       (1024)
    int32_t n_ff = 0;         // free-splatter.feed_forward_length    (4096)
    int32_t n_head = 0;       // free-splatter.attention.head_count   (16)
    int32_t head_dim = 0;     // free-splatter.attention.key_length   (64)
    int32_t patch_size = 0;   // free-splatter.vision.patch_size      (8)
    int32_t image_size = 0;   // free-splatter.vision.image_size      (512)
    int32_t in_channels = 0;  // free-splatter.vision.in_channels       (3)
    int32_t gaussian_channels =
            0;              // free-splatter.vision.gaussian_channels (23)
    int32_t sh_degree = 0;  // free-splatter.sh_degree              (1)

    bool sh_residual = false;  // free-splatter.sh_residual  (scene: true)
    bool use_2dgs = false;     // free-splatter.use_2dgs     (false)

    float ln_eps = 1e-5f;         // free-splatter.attention.layer_norm_epsilon
    float scale_min_act = 1e-4f;  // free-splatter.scale_min_act
    float scale_max_act = 0.02f;  // free-splatter.scale_max_act

    // Derived helpers.
    int32_t tokens_per_view() const {
        const int32_t g = image_size / patch_size;
        return g * g;
    }
};

struct model_file {
    gguf_context *guf = nullptr;
    ggml_context *ctx =
            nullptr;  // tensor metadata; tensor data iff opened with_data
    hparams hp{};
    std::string error;

    model_file() = default;
    model_file(const model_file &) = delete;
    model_file &operator=(const model_file &) = delete;
    ~model_file() { close(); }

    // with_data=false reads metadata only (cheap); true maps tensor data too.
    bool open(const std::string &path, bool with_data);
    void close();

    ggml_tensor *tensor(const char *name) const;  // nullptr if absent
    ggml_tensor *require(const char *name);       // sets error if absent
};

}  // namespace gaussian
}  // namespace aicore
