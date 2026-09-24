// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/depth/attention.hpp"

#include <cmath>
#include <string>

#include "common/ggml_extend.hpp"

namespace aicore {
namespace depth {
// The QKᵀ/softmax/×V core is always the fused flash attention
// (ggml_flash_attn_ext, F32 k/v, F32 softmax precision): ~24% faster backbone
// than the old materialized-scores path with bit-tight parity on CPU, and
// ~20% faster on CUDA (GB10: forward 59 -> 47 ms, corr=0.99999 vs manual).
// The historical manual/skip A/B variants and the F16 k/v experiment
// (DA_ATTN / DA_ATTN_F16) were development-time scaffolding and are removed;
// F32 k/v keeps bit-tight feature parity over the 12 layers.
AttnWeights load_attn(const ModelLoader& ml, int i) {
    auto t = [&](const char* s) {
        return ml.tensor("vit.blk." + std::to_string(i) + "." + s);
    };
    AttnWeights w;
    w.qkv_w = t("attn_qkv.weight");
    w.qkv_b = t("attn_qkv.bias");
    w.proj_w = t("attn_proj.weight");
    w.proj_b = t("attn_proj.bias");
    w.qn_w = t("attn_qnorm.weight");
    w.qn_b = t("attn_qnorm.bias");
    w.kn_w = t("attn_knorm.weight");
    w.kn_b = t("attn_knorm.bias");
    return w;
}
ggml_tensor* attention(ggml_context* ctx,
                       ggml_tensor* x,
                       const AttnWeights& w,
                       int H,
                       int D,
                       float eps,
                       ggml_tensor* cosb,
                       ggml_tensor* sinb) {
    const int tok = (int)x->ne[1], embed = H * D;
    ggml_tensor* qkv = linear(ctx, w.qkv_w, x, w.qkv_b);  // [3*embed, tok]
    ggml_tensor* qkv4 = ggml_reshape_4d(ctx, qkv, D, H, 3, tok);  // [D,H,3,tok]
    auto take = [&](int idx) {
        return ggml_cont(ctx, ggml_view_4d(ctx, qkv4, D, H, 1, tok, qkv4->nb[1],
                                           qkv4->nb[2], qkv4->nb[3],
                                           (size_t)idx * qkv4->nb[2]));
    };
    ggml_tensor* q = ggml_reshape_3d(ctx, take(0), D, H, tok);
    ggml_tensor* k = ggml_reshape_3d(ctx, take(1), D, H, tok);
    ggml_tensor* v = ggml_reshape_3d(ctx, take(2), D, H, tok);
    if (w.qn_w) {
        q = layernorm(ctx, q, w.qn_w, w.qn_b, eps);
        k = layernorm(ctx, k, w.kn_w, w.kn_b, eps);
    }
    if (cosb) {
        q = apply_rope(ctx, q, cosb, sinb, D);
        k = apply_rope(ctx, k, cosb, sinb, D);
    }
    ggml_tensor* qp =
            ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [D,tok,H]
    ggml_tensor* kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor* vp = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    const float scale = 1.0f / std::sqrt((float)D);
    // Fused scaled QKᵀ + softmax + ×V. q/k/v already in flash layout
    // [D, tok, H, 1] (ne0=D, ne1=tokens, ne2=heads, ne3=batch). No causal mask
    // (full bidirectional attention). F32 k/v keeps bit-tight parity with the
    // historical manual path.
    ggml_tensor* o =
            ggml_flash_attn_ext(ctx, qp, kp, vp, nullptr, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);  // match manual F32 softmax
    o = ggml_reshape_2d(ctx, o, embed,
                        tok);  // out is [D,H,tok,1] -> [embed,tok]
    return linear(ctx, w.proj_w, o, w.proj_b);
}
}  // namespace depth
}  // namespace aicore
