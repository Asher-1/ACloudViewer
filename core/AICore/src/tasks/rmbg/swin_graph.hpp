#pragma once
#include "ggml-backend.h"
#include "tasks/rmbg/swin_backbone.hpp"

#include <string>
#include <vector>

namespace rmbg {

/// GGML compute graph for Swin patch_embed (conv 4x4 s4 + layer norm).
/// Phase-2 graph path: validate against SwinBackboneForward::forward_patch_embed.
class SwinPatchEmbedGraph {
public:
    bool init(ggml_backend_t backend, const WeightMap & w, std::string & err);
    void free();

    /// Input NCHW [3,H,W]. Writes NCHW patch map and BLC tokens (post norm).
    bool forward(const std::vector<float> & nchw, int H, int W,
                 std::vector<float> & patch_nchw,
                 std::vector<float> & tokens_blc,
                 std::string & err);

    ggml_backend_t backend() const { return backend_; }

private:
    ggml_backend_t backend_ = nullptr;
    ggml_backend_buffer_t buf_weights_ = nullptr;
    ggml_context * ctx_w_ = nullptr;
    ggml_context * ctx_g_ = nullptr;
    ggml_cgraph  * graph_ = nullptr;

    ggml_tensor * w_conv_ = nullptr;
    ggml_tensor * b_conv_ = nullptr;
    ggml_tensor * w_ln_  = nullptr;
    ggml_tensor * b_ln_  = nullptr;
    ggml_tensor * inp_   = nullptr;
    ggml_tensor * out_conv_ = nullptr;
    ggml_tensor * out_nchw_ = nullptr;

    int embed_dim_ = 0;
};

} // namespace rmbg
