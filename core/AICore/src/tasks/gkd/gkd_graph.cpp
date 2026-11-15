// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// GKDT ggml runtime - session and graph builders (implementation).
//
// Three ggml graphs mirror the official PyTorch forward 1:1:
//   vision_graph : images {384,384,3,B}   -> DINOv3 patch tokens {D, 576, B}
//   text_graph   : token ids {77, T}      -> adapted text features {2048, 77,
//   T} detect_graph : qf {fw,fw,D,1} + prompts {D,N} + mask {1,N}
//                -> fused heatmaps {N, heat_w^2}   (per query image)
//
// Host side: BPE tokenize, EOT argmax-pool + lower-half slice, soft-fiber
// Gaussian visual prompt pooling, official PAD_KPS padding, heatmap decode.
#include "tasks/gkd/gkd_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "tasks/gkd/gkd_common.hpp"

namespace gkd {

// ---------------------------------------------------------------------------
// graph helpers
// ---------------------------------------------------------------------------
static std::string vname(int i, const char* suffix) {
    return "vis.b" + std::to_string(i) + "." + suffix;
}
static std::string tname(int i, const char* suffix) {
    return "txt.b" + std::to_string(i) + "." + suffix;
}
static std::string aname(int i, const char* suffix) {
    return "anet.b" + std::to_string(i) + "." + suffix;
}
static std::string kname(int i, const char* suffix) {
    return "kg.b" + std::to_string(i) + "." + suffix;
}

// Return the raw weight tensor (resident in the backend weight buffer).
// NOTE: wrap it in ggml_view_tensor only if you need a reshaped alias —
// view+reshape chains of backend-buffer tensors must go through ggml_cont
// (or be avoided) because mul_mat on such reshaped views silently
// mis-addresses memory in some builds.
static ggml_tensor* weight(GkdModel& m,
                           ggml_context* build,
                           const std::string& name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) {
        GKD_LOG_ERROR("missing tensor '%s' in GGUF", name.c_str());
        return nullptr;
    }
    return it->second;
}

// y = LayerNorm(x, eps) * w + b   (torch nn.LayerNorm over ne[0])
static ggml_tensor* ln(ggml_context* b,
                       ggml_tensor* x,
                       ggml_tensor* w,
                       ggml_tensor* bias,
                       float eps) {
    ggml_tensor* n = ggml_norm(b, x, eps);
    ggml_tensor* y = ggml_mul(b, n, w);
    if (bias) y = ggml_add(b, y, bias);
    return y;
}

// x / max(||x||, eps) along ne[0]   (torch F.normalize(p=2, dim=channel))
static ggml_tensor* l2_normalize_rows(ggml_context* b,
                                      ggml_tensor* x,
                                      float eps) {
    ggml_tensor* nrm = ggml_sqrt(b, ggml_sum_rows(b, ggml_sqr(b, x)));
    return ggml_div(b, x, ggml_clamp(b, nrm, eps, 3.4e38f));
}

// Fused packed QKV: in {D, T(, B)} (+bias) -> per-head q/k/v {dh, T, heads(,
// B)} realized as strided views (no data movement); heads live in ne2 per the
// ggml flash_attn_ext convention.
struct QkvHeads {
    ggml_tensor *q, *k, *v;
};
static QkvHeads qkv_heads(ggml_context* b,
                          ggml_tensor* qkv,
                          ggml_tensor* bias,
                          int D,
                          int heads) {
    const int dh = D / heads;
    const size_t es = ggml_element_size(qkv);
    const int64_t T = qkv->ne[1];
    const int64_t Bn = qkv->ne[2];
    const size_t row = qkv->nb[1];  // token stride
    const size_t bat = qkv->nb[2];  // batch stride
    auto mk = [&](int part) -> ggml_tensor* {
        // ggml flash convention: {dh, seq, heads, batch}; src {D*3, seq,
        // batch}: token stride nb1 = row (D*3*es), head stride nb2 = dh*es,
        // batch nb3 = bat
        ggml_tensor* t = ggml_view_4d(b, qkv, dh, T, heads, Bn, row, dh * es,
                                      bat, (size_t)part * D * es);
        if (bias) {
            ggml_tensor* bp = ggml_reshape_4d(
                    b, ggml_view_1d(b, bias, D, (size_t)part * D * es), dh, 1,
                    heads, 1);
            t = ggml_add(b, t, bp);
        }
        return t;
    };
    return {mk(0), mk(1), mk(2)};
}

// dinov3 RoPE: out = q*cos + rot_half(q)*sin with rot_half = cat(-x2, x1),
// expanded per half to avoid UNARY(-) on strided views (unsupported on CUDA):
//   out1 = x1*cos1 - x2*sin1 ; out2 = x2*cos2 + x1*sin2
// x: {dh, T, heads, B}; cos/sin: {dh, T, 1, 1} broadcast over heads and batch.
static ggml_tensor* rope_apply(ggml_context* b,
                               ggml_tensor* x,
                               ggml_tensor* cos_t,
                               ggml_tensor* sin_t) {
    const int64_t h = x->ne[0] / 2;
    const size_t es = ggml_element_size(x);
    ggml_tensor* x1 = ggml_view_4d(b, x, h, x->ne[1], x->ne[2], x->ne[3],
                                   x->nb[1], x->nb[2], x->nb[3], 0);
    ggml_tensor* x2 = ggml_view_4d(b, x, h, x->ne[1], x->ne[2], x->ne[3],
                                   x->nb[1], x->nb[2], x->nb[3], h * es);
    ggml_tensor* c1 = ggml_view_4d(b, cos_t, h, cos_t->ne[1], 1, 1,
                                   cos_t->nb[1], cos_t->nb[2], cos_t->nb[3], 0);
    ggml_tensor* c2 =
            ggml_view_4d(b, cos_t, h, cos_t->ne[1], 1, 1, cos_t->nb[1],
                         cos_t->nb[2], cos_t->nb[3], h * es);
    ggml_tensor* s1 = ggml_view_4d(b, sin_t, h, sin_t->ne[1], 1, 1,
                                   sin_t->nb[1], sin_t->nb[2], sin_t->nb[3], 0);
    ggml_tensor* s2 =
            ggml_view_4d(b, sin_t, h, sin_t->ne[1], 1, 1, sin_t->nb[1],
                         sin_t->nb[2], sin_t->nb[3], h * es);
    ggml_tensor* out1 = ggml_sub(b, ggml_mul(b, x1, c1), ggml_mul(b, x2, s1));
    ggml_tensor* out2 = ggml_add(b, ggml_mul(b, x2, c2), ggml_mul(b, x1, s2));
    return ggml_concat(b, out1, out2, 0);
}

// Patch embedding: the host prepares the im2col matrix {IC*KH*KW, L, B}
// (k = kw + kh*KW + ic*KH*KW, token order w + h*W), so the "conv" with
// kernel == stride reduces to one mul_mat plus bias. Equivalent to torch
// conv2d in fp32 (ggml_conv_2d would round the im2col through F16).
static ggml_tensor* patch_embed_f32(ggml_context* b,
                                    ggml_tensor* w /*{KW,KH,IC,OC}*/,
                                    ggml_tensor* bias,
                                    ggml_tensor* im2col /*{K,L,B}*/) {
    const int64_t KW = w->ne[0], KH = w->ne[1], IC = w->ne[2], OC = w->ne[3];
    ggml_tensor* w2 = ggml_reshape_2d(b, w, KW * KH * IC, OC);
    ggml_tensor* r = ggml_mul_mat(b, w2, im2col);  // {OC, L, B}
    r = ggml_add(b, r, ggml_reshape_2d(b, bias, OC, 1));
    return r;
}

// ---------------------------------------------------------------------------
// session lifecycle
// ---------------------------------------------------------------------------
GkdSession::~GkdSession() {
    if (static_buf_) ggml_backend_buffer_free(static_buf_);
    if (static_ctx_) ggml_free(static_ctx_);
    free_backend_ctx(bctx_);
}

GkdSession::VisionCache::~VisionCache() {
    if (galloc) ggml_gallocr_free(galloc);
    if (ctx) ggml_free(ctx);
}
GkdSession::TextCache::~TextCache() {
    if (galloc) ggml_gallocr_free(galloc);
    if (ctx) ggml_free(ctx);
}
GkdSession::DetectCache::~DetectCache() {
    if (galloc) ggml_gallocr_free(galloc);
    if (ctx) ggml_free(ctx);
}

std::unique_ptr<GkdSession> GkdSession::create(const std::string& gguf_path,
                                               int n_threads,
                                               const std::string& device) {
    auto s = std::unique_ptr<GkdSession>(new GkdSession());
    s->bctx_ = init_backend_ctx(n_threads, device);
    if (!s->bctx_.cpu) return nullptr;

    s->model_ = load_gkd_model(gguf_path, backend_weight_buft(s->bctx_));
    if (!s->model_) return nullptr;

    s->tokenizer_ok_ =
            s->tokenizer_.init(s->model_->bpe_vocab, s->model_->bpe_merges);
    if (!s->tokenizer_ok_) {
        GKD_LOG_WARN("tokenizer init failed; text prompts unavailable");
    }

    s->compute_rope_tables();
    GKD_LOG_DEBUG("session ready (backend: %s, threads: %d)", s->backend(),
                  s->bctx_.n_threads);
    return s;
}

void GkdSession::compute_rope_tables() {
    const ModelParams& P = model_->P;
    const int n_patches = P.n_patches();
    const int dh = P.D / P.heads;
    const int D4 = dh / 4;

    ggml_tensor* periods_t = model_->tensors.count("vis.rope_periods")
                                     ? model_->tensors["vis.rope_periods"]
                                     : nullptr;
    std::vector<float> periods(D4);
    if (periods_t) {
        ggml_backend_tensor_get(periods_t, periods.data(), 0,
                                sizeof(float) * D4);
    } else {
        for (int i = 0; i < D4; i++) {
            periods[i] = std::pow(100.0f, 2.0f * i / (dh / 2));
        }
    }

    const float FW = (float)P.feat_w;
    std::vector<float> ch(P.feat_w), cw(P.feat_w);
    for (int i = 0; i < P.feat_w; i++) {
        ch[i] = (0.5f + i) / FW;  // normalize_coords = "separate"
        cw[i] = (0.5f + i) / FW;
    }
    const float two_pi = 6.2831853f;  // float(2 * math.pi)
    // full tables over ALL tokens: rows [0, T-NP) are the cls/storage prefix
    // (identity RoPE: cos=1, sin=0), rows [T-NP, T) are the patch rows.
    const int T = P.n_tokens_per_im();
    std::vector<float> sin_tab((size_t)T * dh, 0.0f),
            cos_tab((size_t)T * dh, 1.0f);
    int64_t t = T - n_patches;
    for (int hh = 0; hh < P.feat_w; hh++) {
        for (int ww = 0; ww < P.feat_w; ww++, t++) {
            float c[2] = {2.0f * ch[hh] - 1.0f, 2.0f * cw[ww] - 1.0f};
            float ang[640];
            for (int axis = 0; axis < 2; axis++) {
                for (int j = 0; j < D4; j++) {
                    ang[axis * D4 + j] = two_pi * c[axis] / periods[j];
                }
            }
            for (int d = 0; d < dh; d++) {
                float a = ang[d % (dh / 2)];  // angles.tile(2)
                cos_tab[(size_t)t * dh + d] = std::cos(a);
                sin_tab[(size_t)t * dh + d] = std::sin(a);
            }
        }
    }

    // Allocate the constant tables in a backend buffer (weight buft) so the
    // backend scheduler can assign devices to their graph views.
    struct ggml_init_params ip {};
    ip.mem_size = ggml_tensor_overhead() * 8;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context* sctx = ggml_init(ip);
    // created directly as 4D {dh, T, 1, 1} so graph views need no reshaping
    rope_sin_ = ggml_new_tensor_4d(sctx, GGML_TYPE_F32, dh, T, 1, 1);
    rope_cos_ = ggml_new_tensor_4d(sctx, GGML_TYPE_F32, dh, T, 1, 1);
    const int C = P.T_ctx;
    // flash_attn_ext requires F16 masks on some backends (e.g. CPU)
    txt_causal_mask_ = ggml_new_tensor_2d(sctx, GGML_TYPE_F16, C, C);
    static_ctx_ = sctx;
    static_buf_ = ggml_backend_alloc_ctx_tensors_from_buft(
            sctx, backend_weight_buft(bctx_));
    if (!static_buf_) {
        GKD_LOG_ERROR("failed to allocate the static constant buffer");
        return;
    }

    std::vector<float> msk32((size_t)T * dh);
    for (size_t i = 0; i < msk32.size(); i++) msk32[i] = sin_tab[i];
    ggml_backend_tensor_set(rope_sin_, msk32.data(), 0,
                            msk32.size() * sizeof(float));
    for (size_t i = 0; i < msk32.size(); i++) msk32[i] = cos_tab[i];
    ggml_backend_tensor_set(rope_cos_, msk32.data(), 0,
                            msk32.size() * sizeof(float));

    std::vector<ggml_fp16_t> msk((size_t)C * C);
    ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < C; q++) {
        for (int k = 0; k < C; k++) {
            msk[(size_t)q * C + k] = (k > q) ? neg_inf : (ggml_fp16_t)0;
        }
    }
    ggml_backend_tensor_set(txt_causal_mask_, msk.data(), 0,
                            msk.size() * sizeof(ggml_fp16_t));
}

// ---------------------------------------------------------------------------
// vision graph: DINOv3 ViT -> patch tokens {D, 576, B}
// ---------------------------------------------------------------------------
// NOTE graphs are rebuilt on every call: reusing a cached ggml graph across
// backend_sched allocations produced stale-pointer failures on CUDA. The
// rebuild costs are negligible next to the GPU compute.
bool GkdSession::build_vision_graph(int n_images) {
    const ModelParams& P = model_->P;
    const int D = P.D, T = P.n_tokens_per_im(), NP = P.n_patches();
    const int FW = P.img_size, heads = P.heads, dh = D / heads;

    if (vcache_.ctx) {
        ggml_gallocr_free(vcache_.galloc);
        ggml_free(vcache_.ctx);
        vcache_ = VisionCache{};
    }
    struct ggml_init_params ip {};
    ip.mem_size = ggml_tensor_overhead() * (size_t)(P.blocks * 64 + 256) +
                  ggml_graph_overhead_custom(2048, false);
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    vcache_.ctx = ggml_init(ip);
    ggml_context* b = vcache_.ctx;
    GkdModel& m = *model_;

    // input = per-image im2col matrix {IC*KH*KW, NP, B} (host-prepared)
    vcache_.input = ggml_new_tensor_3d(b, GGML_TYPE_F32, 3 * P.patch * P.patch,
                                       NP, n_images);
    ggml_set_name(vcache_.input, "vision_input");
    ggml_set_input(vcache_.input);

    ggml_tensor* x = patch_embed_f32(b, weight(m, b, "vis.patch_w"),
                                     weight(m, b, "vis.patch_b"),
                                     vcache_.input);  // {D, NP, B} tokens

    // tokens = [cls | storage | patches]
    ggml_tensor* cls =
            ggml_repeat(b, weight(m, b, "vis.cls_token"),
                        ggml_new_tensor_3d(b, GGML_TYPE_F32, D, 1, n_images));
    ggml_tensor* st = ggml_repeat(
            b, weight(m, b, "vis.storage_tokens"),
            ggml_new_tensor_3d(b, GGML_TYPE_F32, D, P.n_storage, n_images));
    x = ggml_concat(b, ggml_concat(b, cls, st, 1), x, 1);  // {D, T, B}

    // rope tables {dh, 1, T, 1}: prefix rows are cos=1/sin=0 (identity),
    // precomputed on the host in compute_rope_tables().
    ggml_tensor* sin_t = ggml_view_tensor(b, rope_sin_);
    ggml_tensor* cos_t = ggml_view_tensor(b, rope_cos_);

    const float scale = 1.0f / std::sqrt((float)dh);
    for (int i = 0; i < P.blocks; i++) {
        ggml_tensor* t = ln(b, x, weight(m, b, vname(i, "norm1_w")),
                            weight(m, b, vname(i, "norm1_b")), P.norm_eps);
        ggml_tensor* qkv =
                ggml_add(b, ggml_mul_mat(b, weight(m, b, vname(i, "qkv_w")), t),
                         weight(m, b, vname(i, "qkv_b")));
        QkvHeads h = qkv_heads(b, qkv, nullptr, D,
                               heads);  // {dh, heads, T, B} strided views
        h.q = rope_apply(b, h.q, cos_t, sin_t);
        h.k = rope_apply(b, h.k, cos_t, sin_t);
        ggml_tensor* attn = ggml_flash_attn_ext(b, h.q, h.k, h.v, nullptr,
                                                scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        attn = ggml_reshape_3d(b, attn, D, T, n_images);
        ggml_tensor* proj = ggml_add(
                b, ggml_mul_mat(b, weight(m, b, vname(i, "proj_w")), attn),
                weight(m, b, vname(i, "proj_b")));
        x = ggml_add(b, x, ggml_mul(b, proj, weight(m, b, vname(i, "ls1"))));

        t = ln(b, x, weight(m, b, vname(i, "norm2_w")),
               weight(m, b, vname(i, "norm2_b")), P.norm_eps);
        ggml_tensor* f1 =
                ggml_add(b, ggml_mul_mat(b, weight(m, b, vname(i, "fc1_w")), t),
                         weight(m, b, vname(i, "fc1_b")));
        ggml_tensor* f2 =
                ggml_add(b,
                         ggml_mul_mat(b, weight(m, b, vname(i, "fc2_w")),
                                      ggml_gelu(b, f1)),
                         weight(m, b, vname(i, "fc2_b")));
        x = ggml_add(b, x, ggml_mul(b, f2, weight(m, b, vname(i, "ls2"))));
    }

    x = ln(b, x, weight(m, b, "vis.norm_w"), weight(m, b, "vis.norm_b"),
           P.norm_eps);

    // output: patch tokens only, contiguous {D, NP, B}
    const size_t es = ggml_element_size(x);
    vcache_.output =
            ggml_cont(b, ggml_view_3d(b, x, D, NP, n_images, x->nb[1], x->nb[2],
                                      (size_t)(1 + P.n_storage) * D * es));
    ggml_set_name(vcache_.output, "vision_output");

    vcache_.graph = ggml_new_graph_custom(b, 2048, false);
    ggml_build_forward_expand(vcache_.graph, vcache_.output);
    vcache_.n_images = n_images;
    vcache_.out_tokens = NP;
    return true;
}

// ---------------------------------------------------------------------------
// text graph: dinotxt + anet -> {A_out, 77, T}
// ---------------------------------------------------------------------------
bool GkdSession::build_text_graph(int n_texts) {
    const ModelParams& P = model_->P;
    const int TD = P.T_D, C = P.T_ctx, TH = P.T_heads, TDH = TD / TH;
    const int ffn_h = (int)model_->tensors[tname(0, "fc1_w")]->ne[1];

    if (tcache_.ctx) {
        ggml_gallocr_free(tcache_.galloc);
        ggml_free(tcache_.ctx);
        tcache_ = TextCache{};
    }
    struct ggml_init_params ip {};
    ip.mem_size = ggml_tensor_overhead() *
                          (size_t)(P.T_layers * 40 + P.A_blocks * 40 + 128) +
                  ggml_graph_overhead_custom(2048, false);
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    tcache_.ctx = ggml_init(ip);
    ggml_context* b = tcache_.ctx;
    GkdModel& m = *model_;

    tcache_.input = ggml_new_tensor_2d(b, GGML_TYPE_I32, C, n_texts);
    ggml_set_name(tcache_.input, "text_input");
    ggml_set_input(tcache_.input);

    // get_rows takes 1D ids: flatten (t, c) then restore {TD, C, T}
    ggml_tensor* ids_flat = ggml_reshape_1d(b, tcache_.input, C * n_texts);
    ggml_tensor* x = ggml_get_rows(b, weight(m, b, "txt.tok_emb"),
                                   ids_flat);  // {TD, C*T}
    x = ggml_reshape_3d(b, x, TD, C, n_texts);
    x = ggml_add(b, x, weight(m, b, "txt.pos_emb"));  // broadcast over T

    const float scale = 1.0f / std::sqrt((float)TDH);
    for (int i = 0; i < P.T_layers; i++) {
        ggml_tensor* t =
                ln(b, x, weight(m, b, tname(i, "attn_norm_w")),
                   weight(m, b, tname(i, "attn_norm_b")), P.T_norm_eps);
        ggml_tensor* qkv =
                ggml_mul_mat(b, weight(m, b, tname(i, "qkv_w")), t);  // no bias
        QkvHeads h = qkv_heads(b, qkv, nullptr, TD, TH);
        ggml_tensor* attn = ggml_flash_attn_ext(
                b, h.q, h.k, h.v, ggml_view_tensor(b, txt_causal_mask_), scale,
                0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        attn = ggml_reshape_3d(b, attn, TD, C, n_texts);
        ggml_tensor* proj = ggml_add(
                b, ggml_mul_mat(b, weight(m, b, tname(i, "proj_w")), attn),
                weight(m, b, tname(i, "proj_b")));
        x = ggml_add(b, x, proj);

        t = ln(b, x, weight(m, b, tname(i, "ffn_norm_w")),
               weight(m, b, tname(i, "ffn_norm_b")), P.T_norm_eps);
        ggml_tensor* f1 =
                ggml_add(b, ggml_mul_mat(b, weight(m, b, tname(i, "fc1_w")), t),
                         weight(m, b, tname(i, "fc1_b")));
        ggml_tensor* f2 =
                ggml_add(b,
                         ggml_mul_mat(b, weight(m, b, tname(i, "fc2_w")),
                                      ggml_gelu(b, f1)),
                         weight(m, b, tname(i, "fc2_b")));
        x = ggml_add(b, x, f2);
    }

    x = ln(b, x, weight(m, b, "txt.ln_final_w"), weight(m, b, "txt.ln_final_b"),
           P.T_norm_eps);
    x = ggml_mul_mat(b, weight(m, b, "txt.head_proj_w"),
                     x);  // {A_in=2048, C, T}

    // ---- adaptation net (CLIP-style transformer blocks, batch-first) ----
    // proj_in: LN(2048) + Linear(2048 -> 1280, no bias)
    x = ln(b, x, weight(m, b, "anet.norm_in_w"), weight(m, b, "anet.norm_in_b"),
           1e-5f);
    x = ggml_mul_mat(b, weight(m, b, "anet.proj_in_w"), x);  // {A_model, C, T}
    for (int i = 0; i < P.A_blocks; i++) {
        ggml_tensor* t = ln(b, x, weight(m, b, aname(i, "ln1_w")),
                            weight(m, b, aname(i, "ln1_b")), 1e-5f);
        ggml_tensor* qkv =
                ggml_add(b, ggml_mul_mat(b, weight(m, b, aname(i, "in_w")), t),
                         weight(m, b, aname(i, "in_b")));
        QkvHeads h = qkv_heads(b, qkv, nullptr, P.A_model, P.A_heads);
        ggml_tensor* attn = ggml_flash_attn_ext(
                b, h.q, h.k, h.v, nullptr,
                1.0f / std::sqrt((float)(P.A_model / P.A_heads)), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        attn = ggml_reshape_3d(b, attn, P.A_model, C, n_texts);
        ggml_tensor* proj = ggml_add(
                b, ggml_mul_mat(b, weight(m, b, aname(i, "out_w")), attn),
                weight(m, b, aname(i, "out_b")));
        x = ggml_add(b, x, proj);

        t = ln(b, x, weight(m, b, aname(i, "ln2_w")),
               weight(m, b, aname(i, "ln2_b")), 1e-5f);
        ggml_tensor* f1 =
                ggml_add(b, ggml_mul_mat(b, weight(m, b, aname(i, "fc_w")), t),
                         weight(m, b, aname(i, "fc_b")));
        // CLIP-style QuickGELU: x * sigmoid(1.702 * x)
        ggml_tensor* act =
                ggml_mul(b, f1, ggml_sigmoid(b, ggml_scale(b, f1, 1.702f)));
        ggml_tensor* f2 = ggml_add(
                b, ggml_mul_mat(b, weight(m, b, aname(i, "proj_w")), act),
                weight(m, b, aname(i, "proj_b")));
        x = ggml_add(b, x, f2);
    }
    x = ln(b, x, weight(m, b, "anet.ln_w"), weight(m, b, "anet.ln_b"), 1e-5f);
    x = ggml_mul_mat(b, weight(m, b, "anet.proj"),
                     x);  // x @ proj, {A_out, C, T}
    x = ln(b, x, weight(m, b, "anet.last_norm_w"),
           weight(m, b, "anet.last_norm_b"), 1e-5f);

    tcache_.output = x;
    ggml_set_name(tcache_.output, "text_output");
    (void)ffn_h;

    tcache_.graph = ggml_new_graph_custom(b, 2048, false);
    ggml_build_forward_expand(tcache_.graph, tcache_.output);
    tcache_.n_texts = n_texts;
    return true;
}

// ---------------------------------------------------------------------------
// detect graph: KGTransformer + DetectionHead + fusion -> {N, heat_w^2}
// ---------------------------------------------------------------------------
bool GkdSession::build_detect_graph(int n_t_pad, int n_v_pad) {
    const ModelParams& P = model_->P;
    const int D = P.D, heads = P.K_heads, dh = D / heads;
    const int fw = P.feat_w, ups = P.up_scale;
    const int N = n_t_pad + n_v_pad;
    if (N <= 0) {
        GKD_LOG_ERROR("detect graph requires at least one prompt");
        return false;
    }

    if (dcache_.ctx) {
        ggml_gallocr_free(dcache_.galloc);
        ggml_free(dcache_.ctx);
        dcache_ = DetectCache{};
    }
    struct ggml_init_params ip {};
    ip.mem_size = ggml_tensor_overhead() * (size_t)(P.K_blocks * 80 + 256) +
                  ggml_graph_overhead_custom(1024, false);
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    dcache_.ctx = ggml_init(ip);
    ggml_context* b = dcache_.ctx;
    GkdModel& m = *model_;

    dcache_.qf = ggml_new_tensor_4d(b, GGML_TYPE_F32, fw, fw, D, 1);
    ggml_set_name(dcache_.qf, "query_features");
    ggml_set_input(dcache_.qf);
    dcache_.prompts = ggml_new_tensor_2d(b, GGML_TYPE_F32, D, N);
    ggml_set_name(dcache_.prompts, "prompts");
    ggml_set_input(dcache_.prompts);
    dcache_.mask = ggml_new_tensor_2d(b, GGML_TYPE_F32, 1, N);
    ggml_set_name(dcache_.mask, "prompt_mask");
    ggml_set_input(dcache_.mask);

    // cross-attention context: {D, fw*fw} (token order w + h*fw, as torch)
    // ggml_permute places old axis i at position axis_i: (1,2,0,3) maps
    // (x, y, c) -> (c, x, y) so ne becomes {D, fw, fw} with c-major strides.
    ggml_tensor* context =
            ggml_cont(b, ggml_permute(b, dcache_.qf, 1, 2, 0, 3));
    context = ggml_reshape_2d(b, context, D, fw * fw);
    // bilinear x4 upsample (align_corners=False semantics, matches torch)
    ggml_tensor* up = ggml_interpolate(b, dcache_.qf, fw * ups, fw * ups, D, 1,
                                       GGML_SCALE_MODE_BILINEAR);
    ggml_tensor* upc =
            ggml_reshape_2d(b, ggml_cont(b, ggml_permute(b, up, 1, 2, 0, 3)), D,
                            fw * fw * ups * ups);

    // mask-token replacement: x = m*prompt + (1-m)*mask_token
    ggml_tensor* m01 = dcache_.mask;
    ggml_tensor* one =
            ggml_fill(b, ggml_new_tensor_2d(b, GGML_TYPE_F32, 1, N), 1.0f);
    ggml_tensor* mtok = ggml_repeat(b, weight(m, b, "kg.mask_token"),
                                    ggml_new_tensor_2d(b, GGML_TYPE_F32, D, N));
    ggml_tensor* x = ggml_add(b, ggml_mul(b, dcache_.prompts, m01),
                              ggml_mul(b, mtok, ggml_sub(b, one, m01)));

    const float scale = 1.0f / std::sqrt((float)dh);
    for (int i = 0; i < P.K_blocks; i++) {
        // self attention between prompts
        ggml_tensor* t = ln(b, x, weight(m, b, kname(i, "norm1_w")),
                            weight(m, b, kname(i, "norm1_b")), P.K_norm_eps);
        ggml_tensor* qkv = ggml_add(
                b, ggml_mul_mat(b, weight(m, b, kname(i, "sa_in_w")), t),
                weight(m, b, kname(i, "sa_in_b")));
        QkvHeads h = qkv_heads(b, qkv, nullptr, D, heads);
        ggml_tensor* attn = ggml_flash_attn_ext(b, h.q, h.k, h.v, nullptr,
                                                scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        attn = ggml_reshape_2d(b, attn, D, N);
        ggml_tensor* proj = ggml_add(
                b, ggml_mul_mat(b, weight(m, b, kname(i, "sa_out_w")), attn),
                weight(m, b, kname(i, "sa_out_b")));
        // NOTE official KGTransformer reassigns x = norm1(x) before the
        // residual:
        //   x = norm1(x); x = x + attn(x)   (residual base is the NORMED x)
        x = ggml_add(b, t, proj);

        // cross attention against the query features
        t = ln(b, x, weight(m, b, kname(i, "nca1_w")),
               weight(m, b, kname(i, "nca1_b")), P.K_norm_eps);
        ggml_tensor* ctx2 = ln(b, context, weight(m, b, kname(i, "nca2_w")),
                               weight(m, b, kname(i, "nca2_b")), P.K_norm_eps);
        // cross-attn in_proj is stored as three separate q/k/v weights
        ggml_tensor* q = ggml_add(
                b, ggml_mul_mat(b, weight(m, b, kname(i, "ca_q_w")), t),
                weight(m, b, kname(i, "ca_q_b")));
        ggml_tensor* k = ggml_add(
                b, ggml_mul_mat(b, weight(m, b, kname(i, "ca_k_w")), ctx2),
                weight(m, b, kname(i, "ca_k_b")));
        ggml_tensor* v = ggml_add(
                b, ggml_mul_mat(b, weight(m, b, kname(i, "ca_v_w")), ctx2),
                weight(m, b, kname(i, "ca_v_b")));
        {
            const size_t es = ggml_element_size(q);
            // head views of {D, seq} tensors: nb1 = token stride (D*es), nb2 =
            // head stride (dh*es)
            ggml_tensor* qv = ggml_view_4d(b, q, dh, N, heads, 1, q->nb[1],
                                           dh * es, dh * es * heads, 0);
            ggml_tensor* kv =
                    ggml_view_4d(b, k, dh, fw * fw, heads, 1, k->nb[1], dh * es,
                                 dh * es * heads, 0);
            ggml_tensor* vv =
                    ggml_view_4d(b, v, dh, fw * fw, heads, 1, v->nb[1], dh * es,
                                 dh * es * heads, 0);
            attn = ggml_flash_attn_ext(b, qv, kv, vv, nullptr, scale, 0.0f,
                                       0.0f);
        }
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        attn = ggml_reshape_2d(b, attn, D, N);
        proj = ggml_add(
                b, ggml_mul_mat(b, weight(m, b, kname(i, "ca_out_w")), attn),
                weight(m, b, kname(i, "ca_out_b")));
        // same non-standard residual: base is the norm_ca1 output (t)
        x = ggml_add(b, t, proj);

        // feed forward
        t = ln(b, x, weight(m, b, kname(i, "norm3_w")),
               weight(m, b, kname(i, "norm3_b")), P.K_norm_eps);
        ggml_tensor* f1 =
                ggml_add(b, ggml_mul_mat(b, weight(m, b, kname(i, "ff1_w")), t),
                         weight(m, b, kname(i, "ff1_b")));
        ggml_tensor* f2 =
                ggml_add(b,
                         ggml_mul_mat(b, weight(m, b, kname(i, "ff2_w")),
                                      ggml_gelu(b, f1)),
                         weight(m, b, kname(i, "ff2_b")));
        x = ggml_add(b, x, f2);
    }

    x = ln(b, x, weight(m, b, "kg.final_norm_w"),
           weight(m, b, "kg.final_norm_b"), P.K_norm_eps);

    // detection head (parameter-free for the released configs)
    ggml_tensor* kn = P.kernel_norm ? l2_normalize_rows(b, x, 1e-12f) : x;
    ggml_tensor* xu = l2_normalize_rows(b, upc, 1e-12f);
    ggml_tensor* heat = ggml_mul_mat(b, kn, xu);  // {N, heat_w^2}

    // openkd_heatmap_fuse: heat rows are [text prompts (n_t_pad) | visual
    // prompts (n_v_pad)]. Text-only / visual-only: per-row mask, zero rows stay
    // zero. Multimodal (official semantics require n_t_raw == n_v_raw): text
    // row i and visual row i fuse element-wise: (hm_t*mt + hm_v*mv) / (mt + mv
    // + 1e-12), leaving max(n_t_pad, n_v_pad) output rows.
    ggml_tensor* msk_c =
            ggml_reshape_2d(b, dcache_.mask, N, 1);  // {N, 1} row broadcast
    const size_t hes = ggml_element_size(heat);
    ggml_tensor* fused;
    if (n_t_pad > 0 && n_v_pad > 0) {
        GGML_ASSERT(n_t_pad == n_v_pad);
        ggml_tensor* hm_t =
                ggml_view_2d(b, heat, n_t_pad, heat->ne[1], heat->nb[1], 0);
        ggml_tensor* hm_v =
                ggml_view_2d(b, heat, n_v_pad, heat->ne[1], heat->nb[1],
                             (size_t)n_t_pad * heat->nb[0]);
        ggml_tensor* mt = ggml_view_2d(b, msk_c, n_t_pad, 1, msk_c->nb[1], 0);
        ggml_tensor* mv = ggml_view_2d(b, msk_c, n_v_pad, 1, msk_c->nb[1],
                                       (size_t)n_t_pad * msk_c->nb[0]);
        ggml_tensor* num =
                ggml_add(b, ggml_mul(b, hm_t, mt), ggml_mul(b, hm_v, mv));
        ggml_tensor* den = ggml_add(
                b, ggml_add(b, mt, mv),
                ggml_fill(b, ggml_new_tensor_1d(b, GGML_TYPE_F32, 1), 1e-12f));
        fused = ggml_div(b, num, den);
    } else {
        fused = ggml_div(
                b, ggml_mul(b, heat, msk_c),
                ggml_add(b, msk_c,
                         ggml_fill(b, ggml_new_tensor_1d(b, GGML_TYPE_F32, 1),
                                   1e-12f)));
    }

    dcache_.output = fused;
    ggml_set_name(dcache_.output, "heatmaps_fused");

    dcache_.graph = ggml_new_graph_custom(b, 1024, false);
    ggml_build_forward_expand(dcache_.graph, dcache_.output);
    dcache_.n_t_pad = n_t_pad;
    dcache_.n_v_pad = n_v_pad;
    return true;
}

// ---------------------------------------------------------------------------
// host-side helpers
// ---------------------------------------------------------------------------
void GkdSession::gaussian_pooling(const float* feats /*D x 576*/,
                                  int n_kps,
                                  const float* kps /*n x 2*/,
                                  const uint8_t* vis,
                                  float* out /*n x D*/) const {
    const ModelParams& P = model_->P;
    const int D = P.D, fw = P.feat_w;
    const int img = P.img_size;
    const float stride = (float)(img / fw);
    const float start = stride / 2.0f - 0.5f;
    const float sigma = P.sigma;
    const float inv2s2 = 1.0f / (2.0f * sigma * sigma);

    std::vector<float> hm((size_t)fw * fw);
    for (int j = 0; j < n_kps; j++) {
        int cx, cy;
        if (vis && !vis[j]) {
            cx = cy = (int)((0.0f / 2 + 0.5f) *
                            img);  // 192: python's padded center
        } else {
            float lx = (kps[j * 2 + 0] / 2.0f + 0.5f) * img;
            float ly = (kps[j * 2 + 1] / 2.0f + 0.5f) * img;
            cx = std::min(std::max((int)lx, 0), img - 1);
            cy = std::min(std::max((int)ly, 0), img - 1);
        }
        float sum = 0.0f;
        for (int y = 0; y < fw; y++) {
            float dy = (y * stride + start) - (float)cy;
            for (int x = 0; x < fw; x++) {
                float dx = (x * stride + start) - (float)cx;
                float e = (dx * dx + dy * dy) * inv2s2;
                float v = (e <= 4.6052f) ? std::exp(-e) : 0.0f;
                hm[(size_t)y * fw + x] = v;
                sum += v;
            }
        }
        if (sum > 0.0f) {
            for (auto& v : hm) v /= sum;
        }
        // fiber[c] = sum_{h,w} feats[c, h*fw + w] * hm[h, w]
        float* o = out + (size_t)j * D;
        std::memset(o, 0, sizeof(float) * D);
        for (int y = 0; y < fw; y++) {
            for (int x = 0; x < fw; x++) {
                float w = hm[(size_t)y * fw + x];
                if (w == 0.0f) continue;
                const float* f = feats + (size_t)(y * fw + x) * D;
                for (int c = 0; c < D; c++) o[c] += w * f[c];
            }
        }
    }
}

void GkdSession::decode_heatmaps(const float* fused,
                                 int n_pad,
                                 int n_valid,
                                 int heat_w,
                                 const ScaleTrans& trans,
                                 DetectOutput& out) const {
    const int HW = heat_w * heat_w;
    out.n_prompts = n_valid;
    out.kps_norm.assign((size_t)n_valid * 2, 0.0f);
    out.scores.assign(n_valid, 0.0f);
    out.heatmaps.assign((size_t)n_valid * HW, 0.0f);
    // fused is the ggml {N_pad, HW} mul_mat output: (j, pix) at j + pix*N_pad.
    // Re-lay it to the torch (N, HW) row-major while decoding.
    for (int j = 0; j < n_valid; j++) {
        int best = 0;
        float best_v = fused[j];
        for (int i = 1; i < HW; i++) {
            float v = fused[(size_t)i * n_pad + j];
            out.heatmaps[(size_t)j * HW + i] = v;
            if (v > best_v) {
                best_v = v;
                best = i;
            }
        }
        out.heatmaps[(size_t)j * HW + 0] = fused[j];
        int gx = best % heat_w;
        int gy = best / heat_w;
        out.kps_norm[(size_t)j * 2 + 0] = ((gx + 0.5f) / heat_w - 0.5f) * 2.0f;
        out.kps_norm[(size_t)j * 2 + 1] = ((gy + 0.5f) / heat_w - 0.5f) * 2.0f;
        out.scores[j] = best_v;
    }
    (void)trans;
}

void GkdSession::dump_tap(const char* name,
                          const std::vector<int64_t>& shape,
                          const float* data) const {
    if (dump_dir_.empty()) return;
    dump_f32(dump_dir_ + "/cpp_" + name + ".bin", shape, data);
}

// ---------------------------------------------------------------------------
// detect orchestration
// ---------------------------------------------------------------------------
bool GkdSession::detect(const RgbImage& image,
                        const float* bboxes,
                        int n_bbox,
                        const DetectInput& extra,
                        std::vector<DetectOutput>& out,
                        StageTiming* timing) {
    const ModelParams& P = model_->P;
    StageTiming st{};
    const double t0 = now_ms();
    out.clear();

    float full_bbox[4] = {0, 0, (float)(image.w - 1), (float)(image.h - 1)};
    if (n_bbox <= 0) {
        bboxes = full_bbox;
        n_bbox = 1;
    }

    // ---------------- 1) preprocessing ----------------
    std::vector<RgbImage> in_ims;
    std::vector<ScaleTrans> q_trans(n_bbox);
    std::vector<std::vector<float>> s_kps_norm;
    std::vector<std::vector<uint8_t>> s_kps_valid;

    if (extra.has_support) {
        float sb[4] = {0, 0, (float)(extra.support_image.w - 1),
                       (float)(extra.support_image.h - 1)};
        int n_kps = (int)extra.support_kps_xy.size() / 2;
        PreprocessResult sp = preprocess_roi(
                extra.support_image, sb, P.img_size,
                extra.support_kps_xy.data(), n_kps,
                extra.support_kps_vis.empty() ? nullptr
                                              : extra.support_kps_vis.data());
        in_ims.push_back(std::move(sp.img));
        s_kps_norm.push_back(std::move(sp.kps_norm));
        s_kps_valid.push_back(std::move(sp.kps_valid));
    }
    for (int i = 0; i < n_bbox; i++) {
        float bb[4] = {bboxes[i * 4 + 0], bboxes[i * 4 + 1], bboxes[i * 4 + 2],
                       bboxes[i * 4 + 3]};
        PreprocessResult qp =
                preprocess_roi(image, bb, P.img_size, nullptr, 0, nullptr);
        q_trans[i] = qp.trans;
        in_ims.push_back(std::move(qp.img));
    }
    st.preprocess = now_ms() - t0;
    const double t1 = now_ms();

    // ---------------- 2) vision graph ----------------
    const int B = (int)in_ims.size();
    if (!build_vision_graph(B)) return false;
    if (!backend_graph_alloc(bctx_, vcache_.graph, vcache_.input,
                             vcache_.output))
        return false;
    {
        std::vector<float> blob((size_t)B * 3 * P.patch * P.patch *
                                P.n_patches());
        for (int i = 0; i < B; i++) {
            image_to_im2col_normalized(
                    in_ims[i], P.img_mean, P.img_std, P.patch,
                    blob.data() +
                            (size_t)i * 3 * P.patch * P.patch * P.n_patches());
        }
        ggml_backend_tensor_set(vcache_.input, blob.data(), 0,
                                blob.size() * sizeof(float));
    }
    if (backend_graph_compute(bctx_, vcache_.graph) !=
        (int)GGML_STATUS_SUCCESS) {
        GKD_LOG_ERROR("vision graph compute failed");
        return false;
    }
    const int n_out_tokens = vcache_.out_tokens;
    const size_t out_floats = (size_t)ggml_nelements(vcache_.output);
    std::vector<float> vis_feats(out_floats);
    ggml_backend_tensor_get(vcache_.output, vis_feats.data(), 0,
                            out_floats * sizeof(float));
    if (bctx_.sched) ggml_backend_sched_reset(bctx_.sched);
    st.vision = now_ms() - t1;

    // parity taps (official layouts)
    if (!dump_dir_.empty()) {
        std::vector<float> in_chw((size_t)B * 3 * P.img_size * P.img_size);
        std::vector<int64_t> in_shape = {(int64_t)B, 3, P.img_size, P.img_size};
        for (int i = 0; i < B; i++) {
            image_to_chw_normalized(
                    in_ims[i], P.img_mean, P.img_std,
                    in_chw.data() + (size_t)i * 3 * P.img_size * P.img_size);
        }
        dump_tap("in_ims", in_shape, in_chw.data());
        std::vector<float> vt((size_t)B * n_out_tokens * P.D);
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < n_out_tokens; t++) {
                std::memcpy(
                        vt.data() + ((size_t)b * n_out_tokens + t) * P.D,
                        vis_feats.data() + ((size_t)b * n_out_tokens + t) * P.D,
                        sizeof(float) * P.D);
            }
        }
        dump_tap("vis_tokens", {(int64_t)B, n_out_tokens, P.D}, vt.data());
    }
    const double t2 = now_ms();

    // ---------------- 3) text graph ----------------
    const int n_t = (int)extra.kps_texts.size();
    std::vector<float> txt_proto;  // n_t x text_half
    if (n_t > 0) {
        if (!tokenizer_ok_) {
            GKD_LOG_ERROR("text prompts require a working tokenizer");
            return false;
        }
        std::vector<int32_t> ids((size_t)n_t * P.T_ctx);
        std::vector<int32_t> eot_pos(n_t);
        for (int i = 0; i < n_t; i++) {
            std::vector<int32_t> tok =
                    tokenizer_.tokenize(extra.kps_texts[i], P.T_ctx);
            std::memcpy(ids.data() + (size_t)i * P.T_ctx, tok.data(),
                        tok.size() * sizeof(int32_t));
            eot_pos[i] = (int)(std::max_element(tok.begin(), tok.end()) -
                               tok.begin());
        }
        if (!build_text_graph(n_t)) return false;
        if (!backend_graph_alloc(bctx_, tcache_.graph, tcache_.input,
                                 tcache_.output))
            return false;
        ggml_backend_tensor_set(tcache_.input, ids.data(), 0,
                                ids.size() * sizeof(int32_t));
        if (backend_graph_compute(bctx_, tcache_.graph) !=
            (int)GGML_STATUS_SUCCESS) {
            GKD_LOG_ERROR("text graph compute failed");
            return false;
        }
        std::vector<float> anet((size_t)n_t * P.T_ctx * P.A_out);
        ggml_backend_tensor_get(tcache_.output, anet.data(), 0,
                                anet.size() * sizeof(float));
        if (bctx_.sched) ggml_backend_sched_reset(bctx_.sched);

        {
            std::vector<float> ids_f(ids.size());
            for (size_t k = 0; k < ids.size(); k++) ids_f[k] = (float)ids[k];
            dump_tap("tok_ids", {n_t, P.T_ctx}, ids_f.data());
        }
        dump_tap("anet_out", {n_t, P.T_ctx, P.A_out}, anet.data());

        // EOT argmax pooling + lower-half slice (official: get_cls_tokens then
        // [:, text_half:])
        txt_proto.resize((size_t)n_t * P.text_half);
        for (int i = 0; i < n_t; i++) {
            const float* row =
                    anet.data() + ((size_t)i * P.T_ctx + eot_pos[i]) * P.A_out;
            std::memcpy(txt_proto.data() + (size_t)i * P.text_half,
                        row + P.A_out - P.text_half,
                        sizeof(float) * P.text_half);
        }
    }
    st.text = now_ms() - t2;
    const double t3 = now_ms();

    // ---------------- 4) prompts (padding per the official PAD_KPS) --------
    const int n_v = extra.has_support ? (int)s_kps_norm[0].size() / 2 : 0;
    // The official model refuses these two cases (openkd_heatmap_fuse cannot
    // broadcast N_t != N_v heatmaps and forward() asserts on empty prompts):
    if (n_t == 0 && n_v == 0) {
        GKD_LOG_ERROR(
                "no prompts given: need --kps-texts and/or --support-image + "
                "--support-kps");
        return false;
    }
    if (n_t > 0 && n_v > 0 && n_t != n_v) {
        GKD_LOG_ERROR(
                "multimodal prompts require matching counts: %d texts vs %d "
                "support keypoints (the official fuse pairs row i with row i)",
                n_t, n_v);
        return false;
    }
    std::vector<float> vis_proto;
    if (n_v > 0) {
        vis_proto.resize((size_t)n_v * P.D);
        gaussian_pooling(vis_feats.data() /* support is image 0 */, n_v,
                         s_kps_norm[0].data(), s_kps_valid[0].data(),
                         vis_proto.data());
    }
    const int n_t_pad = n_t > 0 ? std::max(n_t, P.pad_kps) : 0;
    const int n_v_pad = n_v > 0 ? std::max(n_v, P.pad_kps) : 0;
    const int N = n_t_pad + n_v_pad;

    std::vector<float> prompts((size_t)N * P.D,
                               0.0f);  // {D, N}: column = prompt
    std::vector<float> mask((size_t)N, 0.0f);
    for (int i = 0; i < n_t; i++) {
        for (int c = 0; c < P.text_half; c++) {
            prompts[(size_t)i * P.D + c] =
                    txt_proto[(size_t)i * P.text_half + c];
        }
        mask[i] = 1.0f;
    }
    for (int j = 0; j < n_v; j++) {
        int col = n_t_pad + j;
        std::memcpy(prompts.data() + (size_t)col * P.D,
                    vis_proto.data() + (size_t)j * P.D, sizeof(float) * P.D);
        mask[col] = 1.0f;
    }
    st.prompt_prep = now_ms() - t3;

    dump_tap("prompts", {N, P.D}, prompts.data());
    dump_tap("prompt_mask", {N}, mask.data());
    if (!vis_proto.empty()) dump_tap("vis_proto", {n_v, P.D}, vis_proto.data());

    // ---------------- 5) detect graph per query image ----------------
    std::vector<float> heat_q((size_t)N * P.heat_w() * P.heat_w());
    std::vector<DetectOutput> results(n_bbox);
    double detect_ms = 0, decode_ms = 0;
    for (int i = 0; i < n_bbox; i++) {
        // NOTE the graph is REBUILT for every ROI: re-running alloc+compute on
        // one cached graph produced garbage on CUDA for the second ROI (same
        // family as the cross-call graph-cache issue; rebuild cost is
        // negligible next to the GPU compute).
        const double tb0 = now_ms();
        if (!build_detect_graph(n_t_pad, n_v_pad)) return false;
        // query features for ROI i: vision batch index = (has_support ? 1 : 0)
        // + i
        const int bidx = (extra.has_support ? 1 : 0) + i;
        std::vector<float> qf_in((size_t)P.feat_w * P.feat_w * P.D);
        // graph input ne={fw, fw, D} == channel-major (CHW) memory: index
        // c*(fw*fw) + token. Vision output is {D, 576, B}: index token*D + c.
        for (int tk = 0; tk < P.n_patches(); tk++) {
            for (int c = 0; c < P.D; c++) {
                qf_in[(size_t)c * P.n_patches() + tk] =
                        vis_feats[(((size_t)bidx * P.n_patches()) + tk) * P.D +
                                  c];
            }
        }
        dump_tap("context", {P.n_patches(), P.D},
                 [&] {
                     // torch layout (token, channel) row-major for the parity
                     // dump
                     std::vector<float> tc((size_t)P.n_patches() * P.D);
                     for (int tk = 0; tk < P.n_patches(); tk++)
                         for (int c = 0; c < P.D; c++)
                             tc[(size_t)tk * P.D + c] =
                                     qf_in[(size_t)c * P.n_patches() + tk];
                     return tc;
                 }()
                         .data());
        const double td0 = now_ms();
        if (!backend_graph_alloc(bctx_, dcache_.graph, dcache_.qf,
                                 dcache_.output))
            return false;
        ggml_backend_tensor_set(dcache_.qf, qf_in.data(), 0,
                                qf_in.size() * sizeof(float));
        ggml_backend_tensor_set(dcache_.prompts, prompts.data(), 0,
                                prompts.size() * sizeof(float));
        ggml_backend_tensor_set(dcache_.mask, mask.data(), 0,
                                mask.size() * sizeof(float));
        if (backend_graph_compute(bctx_, dcache_.graph) !=
            (int)GGML_STATUS_SUCCESS) {
            GKD_LOG_ERROR("detect graph compute failed");
            return false;
        }
        const size_t out_elems = (size_t)ggml_nelements(dcache_.output);
        ggml_backend_tensor_get(dcache_.output, heat_q.data(), 0,
                                out_elems * sizeof(float));
        if (bctx_.sched) ggml_backend_sched_reset(bctx_.sched);
        const double compute_ms = now_ms() - td0;
        detect_ms += compute_ms;

        const double tc0 = now_ms();
        // official: N_origin = N_v when visual prompts exist, else N_t
        // (multimodal fuses text row i with visual row i)
        const int n_valid = (n_v > 0) ? n_v : n_t;
        const int n_rows = (int)(out_elems / P.heat_w() / P.heat_w());
        decode_heatmaps(heat_q.data(), n_rows, n_valid, P.heat_w(), q_trans[i],
                        results[i]);
        // keep the ROI transform so the C API can restore source-pixel coords
        results[i].trans = q_trans[i];
        const double decode_ms_roi = now_ms() - tc0;
        decode_ms += decode_ms_roi;
        // Per-ROI breakdown: the rebuild-per-ROI design makes graph prep
        // the first suspect when a multi-object run slows down. Kept at
        // Debug level so a 10-ROI run does not spam the console; raise the
        // log level via aicore_gkd_options_set_log_level() when diagnosing
        // (2026-09-16: a 6-ROI run sat >150 s with no per-stage visibility).
        GKD_LOG_DEBUG(
                "roi %d/%d: graph prep %.1f ms, compute %.1f ms, heatmap "
                "decode %.1f ms",
                i + 1, n_bbox, td0 - tb0, compute_ms, decode_ms_roi);
    }
    st.detect = detect_ms;
    st.decode = decode_ms;
    st.total = now_ms() - t0;

    dump_tap("heatmaps_fused",
             {results.empty() ? 0 : results[0].n_prompts, P.heat_w(),
              P.heat_w()},
             results.empty() ? nullptr : results[0].heatmaps.data());

    out = std::move(results);
    if (timing) *timing = st;
    return true;
}

}  // namespace gkd
