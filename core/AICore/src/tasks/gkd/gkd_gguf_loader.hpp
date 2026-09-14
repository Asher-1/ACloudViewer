// GKDT ggml runtime - GGUF model loader.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gkd {

// Architecture hyper-parameters, read from the `gkd.*` KV vocabulary written by
// scripts/convert_gkd_to_gguf.py.
struct ModelParams {
    // visual encoder (DINOv3 ViT)
    int D = 1024;             // embed dim
    int blocks = 24;          // transformer depth
    int heads = 16;           // attention heads
    int patch = 16;           // patch size
    int n_storage = 4;        // storage (register) tokens
    float norm_eps = 1e-5f;
    bool has_layerscale = true;
    bool mask_k_bias = true;
    int img_size = 384;
    int feat_w = 24;          // img_size / patch
    // text encoder (dinotxt)
    int T_D = 1280;
    int T_layers = 24;
    int T_heads = 20;
    int T_ctx = 77;
    int T_vocab = 49408;
    int T_proj_dim = 2048;
    float T_norm_eps = 1e-5f;
    // text adaptation net
    int A_in = 2048;
    int A_model = 1280;
    int A_blocks = 1;
    int A_heads = 20;
    int A_out = 2048;
    int text_half = 1024;     // take the lower half of text features
    // kernel generation transformer
    int K_D = 1024;
    int K_blocks = 2;
    int K_heads = 16;
    int K_ff = 1024;
    bool K_use_mask_token = true;
    float K_norm_eps = 1e-5f;
    // detection head
    int up_scale = 4;
    bool kernel_norm = true;
    // prompting / preprocessing
    float sigma = 14.0f;
    int pad_kps = 80;
    float img_mean[3] = {0.485f, 0.456f, 0.406f};
    float img_std[3] = {0.229f, 0.224f, 0.225f};

    int n_tokens_per_im() const { return 1 + n_storage + feat_w * feat_w; }
    int n_patches() const { return feat_w * feat_w; }
    int heat_w() const { return feat_w * up_scale; }
};

// Loaded model: metadata + weights resident on the active backend.
struct GkdModel {
    ModelParams P;
    gguf_context* gguf = nullptr;
    ggml_context* meta_ctx = nullptr;      // tensor metadata (no data)
    ggml_backend_buffer_t weight_buf = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // tokenizer arrays (owned copies, extracted from GGUF)
    std::vector<std::string> bpe_vocab;
    std::vector<std::string> bpe_merges;
};

// Load a GGUF model onto the backend described by buft. Returns nullptr on
// failure (reason is logged).
std::unique_ptr<GkdModel> load_gkd_model(const std::string& path, ggml_backend_buffer_type_t buft);

// KV helpers (throw / abort via log on missing keys of the wrong kind).
namespace kv {
    int32_t      i32(const gguf_context* g, const char* key, int32_t def);
    float        f32(const gguf_context* g, const char* key, float def);
    bool         boolean(const gguf_context* g, const char* key, bool def);
    std::string  str(const gguf_context* g, const char* key, const std::string& def);
    std::vector<std::string> arr_str(const gguf_context* g, const char* key);
    std::vector<float>       arr_f32(const gguf_context* g, const char* key);
}

}  // namespace gkd
