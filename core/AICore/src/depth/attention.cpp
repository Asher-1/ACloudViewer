// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "attention.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "compute_mode.hpp"
#include "ggml_extend.hpp"
namespace aicore {
namespace depth {
// A/B toggle for the QKᵀ/softmax/×V core. Default is "flash"
// (ggml_flash_attn_ext fusion with F32 k/v — ~24% faster backbone, bit-tight
// parity with the manual path). "manual" forces the old materialized-scores
// path; "skip" is a no-op (wrong output) used only to bound the attention-core
// cost in profiling.
//
// The explicit DA_ATTN env override takes precedence (read once). Otherwise the
// default is Flash on BOTH CPU and GPU: the CUDA ggml_flash_attn_ext (F32 prec)
// is ~20% faster than the manual path on the GB10 (forward 59 -> 47 ms, tying
// PyTorch/cuDNN) at corr=0.99999 vs manual (the GPU path is not bit-exact vs
// CPU anyway). DA_ATTN=manual restores the old materialized-scores path.
enum class AttnMode { Manual, Flash, Skip };
static AttnMode attn_mode() {
    static const int env = [] {
        const char* e = std::getenv("DA_ATTN");
        if (e && std::strcmp(e, "manual") == 0) return 1;  // forced manual
        if (e && std::strcmp(e, "skip") == 0) return 2;    // forced skip
        if (e && std::strcmp(e, "flash") == 0) return 3;   // forced flash
        return 0;                                          // unset -> auto
    }();
    switch (env) {
        case 1:
            return AttnMode::Manual;
        case 2:
            return AttnMode::Skip;
        case 3:
            return AttnMode::Flash;
        default:
            return AttnMode::Flash;  // Flash on both CPU and GPU (CUDA
                                     // flash_attn_ext, F32 prec)
    }
}
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
    const AttnMode mode = attn_mode();
    if (mode == AttnMode::Flash) {
        // Fused scaled QKᵀ + softmax + ×V. q/k/v already in flash layout
        // [D, tok, H, 1] (ne0=D, ne1=tokens, ne2=heads, ne3=batch). No causal
        // mask (full bidirectional attention). CPU fattn accepts F32 k/v, which
        // both keeps bit-tight parity with the manual path AND avoids the cast
        // nodes — DA_ATTN_F16 forces the F16 k/v variant for A/B (slightly
        // faster-to-cast but loses feature parity over 12 layers).
        const bool f16kv = std::getenv("DA_ATTN_F16") != nullptr;
        ggml_tensor* kf = f16kv ? ggml_cast(ctx, kp, GGML_TYPE_F16) : kp;
        ggml_tensor* vf = f16kv ? ggml_cast(ctx, vp, GGML_TYPE_F16) : vp;
        ggml_tensor* o = ggml_flash_attn_ext(ctx, qp, kf, vf, nullptr, scale,
                                             0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(
                o, GGML_PREC_F32);  // match manual F32 softmax
        o = ggml_reshape_2d(ctx, o, embed,
                            tok);  // out is [D,H,tok,1] -> [embed,tok]
        return linear(ctx, w.proj_w, o, w.proj_b);
    }
    if (mode == AttnMode::Skip) {
        // No-op core (wrong output, valid shape): bounds the QKᵀ/softmax/×V
        // cost. Keep qp/kp reachable (×0) so the rope cos/sin host inputs
        // aren't pruned out of the graph (else their buffers go unallocated).
        // These add/scale ops are trivial vs the QKᵀ/×V matmuls we are bounding
        // out.
        ggml_tensor* o = ggml_reshape_2d(ctx, vp, embed, tok);
        o = ggml_add(
                ctx, o,
                ggml_scale(ctx, ggml_reshape_2d(ctx, qp, embed, tok), 0.0f));
        o = ggml_add(
                ctx, o,
                ggml_scale(ctx, ggml_reshape_2d(ctx, kp, embed, tok), 0.0f));
        return linear(ctx, w.proj_w, o, w.proj_b);
    }
    ggml_tensor* sc = ggml_mul_mat(ctx, kp, qp);  // [tok_k,tok_q,H]
    ggml_mul_mat_set_prec(sc, GGML_PREC_F32);
    sc = ggml_soft_max_ext(ctx, sc, nullptr, scale, 0.0f);
    ggml_tensor* vt =
            ggml_cont(ctx, ggml_permute(ctx, vp, 1, 0, 2, 3));  // [tok,D,H]
    ggml_tensor* o = ggml_mul_mat(ctx, vt, sc);                 // [D,tok_q,H]
    o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));       // [D,H,tok]
    o = ggml_reshape_2d(ctx, o, embed, tok);
    return linear(ctx, w.proj_w, o, w.proj_b);
}
}  // namespace depth
}  // namespace aicore
