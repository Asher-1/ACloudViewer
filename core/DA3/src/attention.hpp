#pragma once
#include "ggml.h"
#include "model_loader.hpp"
#include "rope2d.hpp"
namespace da {
struct AttnWeights {
    ggml_tensor *qkv_w=nullptr,*qkv_b=nullptr,*proj_w=nullptr,*proj_b=nullptr;
    ggml_tensor *qn_w=nullptr,*qn_b=nullptr,*kn_w=nullptr,*kn_b=nullptr; // null if no qk_norm
};
AttnWeights load_attn(const ModelLoader& ml, int i);
// x: [embed, tokens]. qkv -> optional qk_norm -> optional rope (if cosb!=null) -> sdpa -> proj.
ggml_tensor* attention(ggml_context* ctx, ggml_tensor* x, const AttnWeights& w,
                       int num_heads, int head_dim, float ln_eps,
                       ggml_tensor* cosb, ggml_tensor* sinb);
}
