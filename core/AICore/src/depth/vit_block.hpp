#pragma once
#include "ggml.h"
#include "model_loader.hpp"
#include "attention.hpp"
namespace aicore {
namespace depth {
struct BlockWeights {
    ggml_tensor *n1_w,*n1_b,*n2_w,*n2_b,*ls1,*ls2,*fc1_w,*fc1_b,*fc2_w,*fc2_b;
    // SwiGLU FFN (ffn_type=="swiglu"): w12 = Linear(dim->2*hidden), w3 = Linear(hidden->dim).
    ggml_tensor *w12_w=nullptr,*w12_b=nullptr,*w3_w=nullptr,*w3_b=nullptr;
    bool swiglu=false;
    AttnWeights attn;
};
BlockWeights load_block(const ModelLoader& ml, int i);
// x: [embed, tokens] -> [embed, tokens]. cosb/sinb = rope tables for this layer (or null).
ggml_tensor* vit_block(ggml_context* ctx, ggml_tensor* x, const BlockWeights& w,
                       int num_heads, int head_dim, float ln_eps,
                       ggml_tensor* cosb, ggml_tensor* sinb);
} // namespace depth
} // namespace aicore
