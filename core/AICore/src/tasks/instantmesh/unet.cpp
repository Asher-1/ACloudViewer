// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// UNet2DConditionModel + RefOnly attention. See unet.hpp for the structure
// and layout contract (same as vae.cpp: one contiguous [W,H,C,B] world).
//
// Numerics mirror diffusers 0.39 UNet2DConditionModel with
//   attention_head_dim=[5,10,20,20] (= heads per level; head_dim 64),
//   use_linear_projection, norm_eps 1e-5, GEGLU ff, downsample_padding=1
//   (SYMMETRIC — the VAE's asymmetric downsample is a VAE-only quirk).
// RefOnly: w-pass stores every attn1's post-norm hidden states; r-pass runs
// attn1 as cross-attention with K/V source cat(h_r, h_w). Both passes live
// in one graph so the w intermediates feed the r K/V directly.
#include "unet.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "numerics.hpp"

namespace instantmesh {

namespace {

ggml_tensor *get_t(const UnetModel &m, const std::string &name) {
    auto it = m.gguf.tensors.find(name);
    if (it == m.gguf.tensors.end()) {
        std::fprintf(stderr, "unet: missing tensor '%s'\n", name.c_str());
        std::abort();
    }
    return it->second;
}
bool has_t(const UnetModel &m, const std::string &name) {
    return m.gguf.tensors.count(name) != 0;
}

// torch GroupNorm(G, eps) + affine on [W,H,C,B] contiguous, built from
// explicit reductions. (The library ggml_group_norm computes this exact math
// correctly in isolation but misbehaves inside the large UNet graph for
// reasons that resisted five dump rounds — the primitive composition here is
// verifiable end-to-end, so it replaces the op throughout this model.)
//   sums = ones[WH]·[WH, C*B]  → per-channel spatial sums
//   group sums via reshape [cpg, G, B] and ones[cpg]·  → mean/var per (g,b)
//   expand per-group scalars back per-channel → normalize + affine.
static std::vector<ggml_tensor *> g_u_fill_ones_wh;  // host-filled with 1.0f
static std::vector<ggml_tensor *> g_u_fill_ones_cpg;
static std::vector<ggml_tensor *> g_u_fill_eps;
static std::vector<ggml_tensor *> g_u_fill_ones_gb;
ggml_tensor *gn32(ggml_context *ctx,
                  const UnetModel &m,
                  ggml_tensor *x,
                  const std::string &affine_prefix) {
    const int64_t Wd = x->ne[0], Hd = x->ne[1], C = x->ne[2], B = x->ne[3];
    const int64_t WH = Wd * Hd;
    const int64_t G = m.hp.norm_num_groups;
    const int64_t cpg = C / G;
    ggml_tensor *ones_wh = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, WH, 1);
    ggml_tensor *ones_cpg = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cpg, 1);
    ggml_set_input(ones_wh);
    ggml_set_input(ones_cpg);
    g_u_fill_ones_wh.push_back(ones_wh);
    g_u_fill_ones_cpg.push_back(ones_cpg);

    ggml_tensor *t = ggml_reshape_2d(ctx, x, WH, C * B);  // view
    ggml_tensor *sums = mul_mat_exact(ctx, ones_wh, t);   // [1, C*B]
    ggml_tensor *sq = ggml_mul(ctx, t, t);
    ggml_tensor *sums2 = mul_mat_exact(ctx, ones_wh, sq);  // [1, C*B]
    ggml_tensor *s3 = ggml_reshape_3d(ctx, sums, cpg, G, B);
    ggml_tensor *s23 = ggml_reshape_3d(ctx, sums2, cpg, G, B);
    ggml_tensor *gs = mul_mat_exact(ctx, ones_cpg, s3);    // [1, G, B]
    ggml_tensor *gs2 = mul_mat_exact(ctx, ones_cpg, s23);  // [1, G, B]
    const float inv_n = 1.0f / (float)(cpg * WH);
    ggml_tensor *mean_g = ggml_scale(ctx, gs, inv_n);
    ggml_tensor *ex_g = ggml_scale(ctx, gs2, inv_n);  // E[x^2]
    ggml_tensor *var_g = ggml_sub(ctx, ex_g, ggml_mul(ctx, mean_g, mean_g));
    ggml_tensor *eps_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, G, B);
    ggml_set_input(eps_t);
    g_u_fill_eps.push_back(eps_t);
    ggml_tensor *ones_gb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, G, B);
    ggml_set_input(ones_gb);
    g_u_fill_ones_gb.push_back(ones_gb);
    ggml_tensor *scale_g =
            ggml_div(ctx, ones_gb, ggml_sqrt(ctx, ggml_add(ctx, var_g, eps_t)));
    // expand per-group [1,G,B] scalars to per-channel [C,B]
    ggml_tensor *mean_c = ggml_repeat(
            ctx, mean_g, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cpg, G, B));
    ggml_tensor *scale_c = ggml_repeat(
            ctx, scale_g, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cpg, G, B));
    mean_c = ggml_reshape_2d(ctx, mean_c, C, B);
    scale_c = ggml_reshape_2d(ctx, scale_c, C, B);
    // normalize in [C, WH, B]
    ggml_tensor *th = ggml_cont(
            ctx,
            ggml_permute(ctx, ggml_reshape_3d(ctx, x, WH, C, B), 1, 0, 2, 3));
    ggml_tensor *y = ggml_sub(ctx, th, ggml_reshape_3d(ctx, mean_c, C, 1, B));
    y = ggml_mul(ctx, y, ggml_reshape_3d(ctx, scale_c, C, 1, B));
    if (!affine_prefix.empty()) {
        ggml_tensor *gw = get_t(m, affine_prefix + ".weight");
        ggml_tensor *gb = get_t(m, affine_prefix + ".bias");
        y = ggml_add(ctx, ggml_mul(ctx, y, ggml_reshape_3d(ctx, gw, C, 1, 1)),
                     ggml_reshape_3d(ctx, gb, C, 1, 1));
    }
    y = ggml_reshape_4d(ctx, y, C, Hd, Wd, B);
    return ggml_cont(ctx, ggml_permute(ctx, y, 2, 1, 0, 3));
}

// ── convs (same contract as vae.cpp) ─────────────────────────────────────
ggml_tensor *conv(ggml_context *ctx,
                  ggml_tensor *x,
                  ggml_tensor *w,
                  ggml_tensor *b,
                  int stride,
                  int pad) {
    // As in VAE, retain F16 im2col/layout and explicitly request official F32
    // accumulation to preserve small terms.
    ggml_tensor *columns = ggml_im2col(
            ctx, w, x, stride, stride, pad, pad, 1, 1, true,
            w->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16);
    ggml_tensor *product = ggml_mul_mat(
            ctx,
            ggml_reshape_2d(ctx, columns, columns->ne[0],
                            columns->ne[1] * columns->ne[2] * columns->ne[3]),
            ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1] * w->ne[2], w->ne[3]));
    GGML_ASSERT(ggml_prec_set_acc(product, GGML_PREC_F32));
    ggml_tensor *y = ggml_reshape_4d(ctx, product, columns->ne[1],
                                     columns->ne[2], columns->ne[3], w->ne[3]);
    y = ggml_cont(ctx, ggml_permute(ctx, y, 0, 1, 3, 2));
    if (b) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        y = ggml_add(ctx, y, b);
    }
    return y;
}
// F32-im2col conv (weights may still be f16 — mul_mat upcasts; the point is
// the f32 window vectors). Same output layout as conv().
ggml_tensor *conv_f32(ggml_context *ctx,
                      ggml_tensor *x,
                      ggml_tensor *w,
                      ggml_tensor *b,
                      int stride,
                      int pad) {
    ggml_tensor *im2col = ggml_im2col(ctx, w, x, stride, stride, pad, pad, 1, 1,
                                      true, GGML_TYPE_F32);
    ggml_tensor *wk = ggml_cast(ctx, w, GGML_TYPE_F32);
    ggml_tensor *result = mul_mat_exact(
            ctx,
            ggml_reshape_2d(ctx, im2col, im2col->ne[0],
                            im2col->ne[3] * im2col->ne[2] * im2col->ne[1]),
            ggml_reshape_2d(ctx, wk, wk->ne[0] * wk->ne[1] * wk->ne[2],
                            wk->ne[3]));
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2],
                             im2col->ne[3], w->ne[3]);
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2));
    if (b) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        result = ggml_add(ctx, result, b);
    }
    return result;
}

// torch NCHW bytes → [W,H,C,B] conv input (identity flat map, any B).
ggml_tensor *to_conv(ggml_context *ctx, ggml_tensor *x) {
    return x;  // ne=[W,H,C,B] labeling only; bytes are already NCHW
}
// [W,H,C,B] → torch NCHW bytes (identity flat map).
ggml_tensor *from_conv(ggml_context *ctx, ggml_tensor *x) {
    return ggml_cont(ctx, x);
}
// [W,H,C,B] → torch-order [C,H,W,B] gather (attention flatten input).
ggml_tensor *gather_chw(ggml_context *ctx, ggml_tensor *x) {
    return ggml_cont(ctx, ggml_permute(ctx, x, 2, 1, 0, 3));
}
// [C,H,W,B] → [W,H,C,B] scatter back.
ggml_tensor *scatter_whc(ggml_context *ctx, ggml_tensor *x) {
    return ggml_cont(ctx, ggml_permute(ctx, x, 2, 1, 0, 3));
}

// LayerNorm over ne0 (= channel dim of [C,HW,B]) + affine (1-D weights).
ggml_tensor *ln(ggml_context *ctx,
                ggml_tensor *x,
                ggml_tensor *w,
                ggml_tensor *b,
                float eps) {
    ggml_tensor *n = ggml_norm(ctx, x, eps);
    return ggml_add(ctx, ggml_mul(ctx, n, w), b);
}

// Multi-head attention over [C,HW,B] queries.
//   kv: [C_kv, L_kv, B] torch-ordered source (itself for self-attn).
//   per-head views + softmax; o written back via head-segments cat.
ggml_tensor *attn_qkv(ggml_context *ctx,
                      const UnetModel &m,
                      ggml_tensor *h,
                      ggml_tensor *kv,
                      const std::string &p,
                      int heads) {
    const int64_t C = h->ne[0];  // query channels
    const int64_t HW = h->ne[1];
    const int64_t B = h->ne[2];
    const int64_t L = kv->ne[1];  // key/value sequence length
    const int64_t Ci = get_t(m, p + "to_q.weight")->ne[1];  // inner = heads*hd
    const int64_t hd = Ci / heads;
    const float scale = 1.0f / std::sqrt((float)hd);

    ggml_tensor *q = mul_mat_exact(ctx, get_t(m, p + "to_q.weight"), h);
    ggml_tensor *k = mul_mat_exact(ctx, get_t(m, p + "to_k.weight"), kv);
    ggml_tensor *v = mul_mat_exact(ctx, get_t(m, p + "to_v.weight"), kv);
    // biases (q/k/v and out all have bias in SD weights)
    if (has_t(m, p + "to_q.bias")) {
        q = ggml_add(ctx, q, get_t(m, p + "to_q.bias"));
        k = ggml_add(ctx, k, get_t(m, p + "to_k.bias"));
        v = ggml_add(ctx, v, get_t(m, p + "to_v.bias"));
    }
    // head views: [Ci, N, B] → per-head [hd, N, B]
    // att per head then cat along ne0 to rebuild [Ci, HW, B].
    std::vector<ggml_tensor *> o_heads(heads);
    for (int hi = 0; hi < heads; ++hi) {
        // cont() matters: strided views as matmul operands make the Vulkan
        // backend copy them to F16 (x/y_non_contig path), silently rounding.
        ggml_tensor *qh = ggml_cont(
                ctx, ggml_view_3d(ctx, q, hd, HW, B, q->nb[1], q->nb[2],
                                  (size_t)hi * hd * sizeof(float)));
        ggml_tensor *kh = ggml_cont(
                ctx, ggml_view_3d(ctx, k, hd, L, B, k->nb[1], k->nb[2],
                                  (size_t)hi * hd * sizeof(float)));
        ggml_tensor *vh = ggml_view_3d(ctx, v, hd, L, B, v->nb[1], v->nb[2],
                                       (size_t)hi * hd * sizeof(float));
        // att[i,j] = softmax_i(q_j · k_i * scale)
        ggml_tensor *att = ggml_scale(ctx, mul_mat_exact(ctx, kh, qh), scale);
        att = ggml_soft_max_ext(ctx, att, nullptr, 1.0f, 0.0f);  // [L, HW, B]
        // o_h[j,c] = Σ_i att(i,j)·v(i,c): transpose v head to [L, hd, B]
        ggml_tensor *vt = ggml_cont(
                ctx, ggml_permute(ctx, vh, 1, 0, 2, 3));  // [L, hd, B]
        ggml_tensor *oh = mul_mat_exact(ctx, att, vt);    // [HW, hd, B]
        // back to [hd, HW, B] segment (c fastest within the head block)
        o_heads[hi] = ggml_cont(ctx, ggml_permute(ctx, oh, 1, 0, 2, 3));
    }
    ggml_tensor *o = o_heads[0];
    for (int hi = 1; hi < heads; ++hi) o = ggml_concat(ctx, o, o_heads[hi], 0);
    o = ggml_cont(ctx, o);  // cat inputs were views/conts — normalize
    ggml_tensor *out = mul_mat_exact(ctx, get_t(m, p + "to_out.0.weight"), o);
    if (has_t(m, p + "to_out.0.bias"))
        out = ggml_add(ctx, out, get_t(m, p + "to_out.0.bias"));
    return out;  // [C, HW, B]
}

// BasicTransformerBlock on [C,HW,B]. ref: when non-null, attn1 uses
// cat(h, ref) as its K/V source (RefOnly r-mode); when the string is "w",
// the caller records h via the returned hook input (handled by caller).
ggml_tensor *transformer_block(ggml_context *ctx,
                               const UnetModel &m,
                               ggml_tensor *h,
                               ggml_tensor *context,
                               const std::string &p,
                               int heads,
                               ggml_tensor *ref_in,
                               ggml_tensor **ref_out) {
    // 1) self-attention (or RefOnly r cross-with-self). RefOnly 'w' stores
    // exactly this post-LN input (ReferenceOnlyAttnProc wraps attn1).
    ggml_tensor *h0 = h;  // residual target is the PRE-norm hidden state
    h = ln(ctx, h, get_t(m, p + "norm1.weight"), get_t(m, p + "norm1.bias"),
           m.hp.norm_eps);
    if (ref_out) *ref_out = h;
    ggml_tensor *kv = ref_in ? ggml_concat(ctx, h, ref_in, 1) : h;
    ggml_tensor *a = attn_qkv(ctx, m, h, kv, p + "attn1.", heads);
    h = ggml_add(ctx, h0, a);
    // 2) cross-attention on context
    h0 = h;
    h = ln(ctx, h, get_t(m, p + "norm2.weight"), get_t(m, p + "norm2.bias"),
           m.hp.norm_eps);
    a = attn_qkv(ctx, m, h, context, p + "attn2.", heads);
    h = ggml_add(ctx, h0, a);
    // 3) GEGLU feed-forward
    h0 = h;
    h = ln(ctx, h, get_t(m, p + "norm3.weight"), get_t(m, p + "norm3.bias"),
           m.hp.norm_eps);
    {
        ggml_tensor *g = mul_mat_exact(
                ctx, get_t(m, p + "ff.net.0.proj.weight"), h);  // [2I, HW, B]
        if (has_t(m, p + "ff.net.0.proj.bias"))
            g = ggml_add(ctx, g, get_t(m, p + "ff.net.0.proj.bias"));
        const int64_t I = g->ne[0] / 2;
        ggml_tensor *u = ggml_view_3d(ctx, g, I, g->ne[1], g->ne[2], g->nb[1],
                                      g->nb[2], 0);
        ggml_tensor *gate = ggml_view_3d(ctx, g, I, g->ne[1], g->ne[2],
                                         g->nb[1], g->nb[2], I * sizeof(float));
        // CUDA unary ops require a contiguous src0; the half-split view is
        // strided (nb1 skips the other half), so materialize it first.
        gate = ggml_gelu(ctx, ggml_cont(ctx, gate));
        ggml_tensor *ff = ggml_mul(ctx, u, gate);
        ff = mul_mat_exact(ctx, get_t(m, p + "ff.net.2.weight"), ff);
        if (has_t(m, p + "ff.net.2.bias"))
            ff = ggml_add(ctx, ff, get_t(m, p + "ff.net.2.bias"));
        h = ggml_add(ctx, h0, ff);
    }
    return h;
}

// Transformer2DModel: GN → proj_in → block(s) → proj_out, residual outside.
ggml_tensor *transformer(ggml_context *ctx,
                         const UnetModel &m,
                         ggml_tensor *x,
                         ggml_tensor *context,
                         const std::string &p,
                         int heads,
                         ggml_tensor *ref_in,
                         ggml_tensor **ref_out) {
    const int64_t C = x->ne[2];
    const int64_t HW = x->ne[0] * x->ne[1];
    const int64_t B = x->ne[3];
    ggml_tensor *g =
            ggml_group_norm(ctx, x, m.hp.norm_num_groups, m.hp.norm_eps);
    {
        ggml_tensor *gw = get_t(m, p + "norm.weight");
        ggml_tensor *gb = get_t(m, p + "norm.bias");
        gw = ggml_reshape_4d(ctx, gw, 1, 1, gw->ne[0], 1);
        gb = ggml_reshape_4d(ctx, gb, 1, 1, gb->ne[0], 1);
        g = ggml_add(ctx, ggml_mul(ctx, g, gw), gb);
    }
    ggml_tensor *h = gather_chw(ctx, g);  // [C, HW, B] torch
    h = ggml_reshape_3d(ctx, h, C, HW, B);
    h = mul_mat_exact(ctx, get_t(m, p + "proj_in.weight"),
                      h);  // linear proj (with bias)
    if (has_t(m, p + "proj_in.bias"))
        h = ggml_add(ctx, h, get_t(m, p + "proj_in.bias"));
    h = transformer_block(ctx, m, h, context, p + "transformer_blocks.0.",
                          heads, ref_in, ref_out);
    h = mul_mat_exact(ctx, get_t(m, p + "proj_out.weight"), h);
    if (has_t(m, p + "proj_out.bias"))
        h = ggml_add(ctx, h, get_t(m, p + "proj_out.bias"));
    h = ggml_reshape_4d(ctx, h, C, x->ne[1], x->ne[0], B);  // [C,H,W,B]
    h = scatter_whc(ctx, h);                                // [W,H,C,B]
    return ggml_add(ctx, x, h);
}

// ResnetBlock2D with time-emb. ch_out read from conv1 weights.
ggml_tensor *resnet(ggml_context *ctx,
                    const UnetModel &m,
                    ggml_tensor *x,
                    ggml_tensor *temb_col,
                    const std::string &p) {
    ggml_tensor *h =
            ggml_group_norm(ctx, x, m.hp.norm_num_groups, m.hp.norm_eps);
    {
        ggml_tensor *gw = get_t(m, p + "norm1.weight");
        ggml_tensor *gb = get_t(m, p + "norm1.bias");
        gw = ggml_reshape_4d(ctx, gw, 1, 1, gw->ne[0], 1);
        gb = ggml_reshape_4d(ctx, gb, 1, 1, gb->ne[0], 1);
        h = ggml_add(ctx, ggml_mul(ctx, h, gw), gb);
    }
    h = ggml_silu(ctx, h);
    h = conv(ctx, h, get_t(m, p + "conv1.weight"), get_t(m, p + "conv1.bias"),
             1, 1);
    // temb: silu → linear → [:,:,None,None]; temb_col is [1280, B]
    ggml_tensor *te = ggml_silu(ctx, temb_col);
    te = mul_mat_exact(ctx, get_t(m, p + "time_emb_proj.weight"),
                       te);  // [C, B]
    const int64_t C = te->ne[0], B = te->ne[1];
    te = ggml_reshape_4d(ctx, te, 1, 1, C, B);  // broadcast view
    if (has_t(m, p + "time_emb_proj.bias")) {
        ggml_tensor *tb = ggml_reshape_4d(
                ctx, get_t(m, p + "time_emb_proj.bias"), 1, 1, C, 1);
        te = ggml_add(ctx, te, tb);  // ne3 broadcasts
    }
    h = ggml_add(ctx, h, te);
    ggml_tensor *hn =
            ggml_group_norm(ctx, h, m.hp.norm_num_groups, m.hp.norm_eps);
    {
        ggml_tensor *gw = get_t(m, p + "norm2.weight");
        ggml_tensor *gb = get_t(m, p + "norm2.bias");
        gw = ggml_reshape_4d(ctx, gw, 1, 1, gw->ne[0], 1);
        gb = ggml_reshape_4d(ctx, gb, 1, 1, gb->ne[0], 1);
        h = ggml_add(ctx, ggml_mul(ctx, hn, gw), gb);
    }
    h = ggml_silu(ctx, h);
    h = conv(ctx, h, get_t(m, p + "conv2.weight"), get_t(m, p + "conv2.bias"),
             1, 1);
    const std::string sc = p + "conv_shortcut.weight";
    if (has_t(m, sc)) {
        x = conv(ctx, x, get_t(m, sc), get_t(m, p + "conv_shortcut.bias"), 1,
                 0);
    }
    return ggml_add(ctx, h, x);
}

// sinusoidal timestep embedding → [320, B] column-shared
// sinusoidal timestep embedding → [320, B] column-shared. The [320] vector
// (cos|sin — flip_sin_to_cos) is computed on the HOST and returned as an
// input tensor for the caller to fill after gallocr allocation.
ggml_tensor *time_embedding(ggml_context *ctx,
                            const UnetModel &m,
                            float t,
                            int B,
                            ggml_tensor **emb_in) {
    std::vector<float> e(320);  // flip_sin_to_cos: [cos, sin]
    {
        const int half = 320 / 2;
        std::vector<float> f(half);
        for (int i = 0; i < half; ++i)
            f[i] = std::exp(-(float)std::log(10000.0) * (float)i /
                            (float)(half - m.hp.freq_shift));
        for (int i = 0; i < half; ++i) {
            e[i] = std::cos(t * f[i]);
            e[half + i] = std::sin(t * f[i]);
        }
    }
    ggml_tensor *emb0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 320);
    ggml_set_input(emb0);
    *emb_in = emb0;
    ggml_tensor *emb = ggml_reshape_2d(ctx, emb0, 320, 1);
    emb = ggml_repeat(ctx, emb, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 320, B));
    ggml_tensor *te =
            mul_mat_exact(ctx, get_t(m, "time_embedding.linear_1.weight"), emb);
    te = ggml_add(ctx, te, get_t(m, "time_embedding.linear_1.bias"));
    te = ggml_silu(ctx, te);
    te = mul_mat_exact(ctx, get_t(m, "time_embedding.linear_2.weight"), te);
    te = ggml_add(ctx, te, get_t(m, "time_embedding.linear_2.bias"));
    return te;  // [1280, B]
}

}  // namespace

bool unet_load(const std::string &path,
               ggml_backend_t backend,
               UnetModel &out,
               std::string *error) {
    if (!load_gguf(path, backend, out.gguf, error)) return false;
    out.backend = backend;
    const gguf_context *g = out.gguf.gguf;
    out.hp.in_channels = kv_i32(g, "unet.in_channels", 4);
    out.hp.out_channels = kv_i32(g, "unet.out_channels", 4);
    out.hp.layers_per_block = kv_i32(g, "unet.layers_per_block", 2);
    out.hp.cross_attention_dim = kv_i32(g, "unet.cross_attention_dim", 1024);
    out.hp.norm_num_groups = kv_i32(g, "unet.norm_num_groups", 32);
    out.hp.norm_eps = kv_f32(g, "unet.norm_eps", 1e-5f);
    out.hp.use_linear_projection =
            kv_bool(g, "unet.use_linear_projection", true);
    out.hp.flip_sin_to_cos = kv_bool(g, "unet.flip_sin_to_cos", true);
    out.hp.freq_shift = kv_i32(g, "unet.freq_shift", 0);
    const char *boc = kv_str(g, "unet.block_out_channels", "320,640,1280,1280");
    std::sscanf(boc, "%d,%d,%d,%d", &out.hp.block_out_channels[0],
                &out.hp.block_out_channels[1], &out.hp.block_out_channels[2],
                &out.hp.block_out_channels[3]);
    const char *ahd = kv_str(g, "unet.attention_head_dim", "5,10,20,20");
    std::sscanf(ahd, "%d,%d,%d,%d", &out.hp.num_heads[0], &out.hp.num_heads[1],
                &out.hp.num_heads[2], &out.hp.num_heads[3]);
    return true;
}

float *unet_forward_refonly(const UnetModel &m,
                            const float *sample,
                            const float *ref_sample,
                            const float *context,
                            int context_len,
                            int B,
                            int H,
                            int W,
                            int ref_h,
                            int ref_w,
                            float timestep,
                            int *out_h,
                            int *out_w) {
    ggml_init_params ip = {(size_t)8 << 30, nullptr, true};
    ggml_context *ctx = ggml_init(ip);
    const auto &hp = m.hp;
    const int L = context_len;

    // inputs (torch NCHW bytes / [B,L,1024]); the w-pass condition latent
    // may differ in resolution from the r-pass sample (zero123pp denoises at
    // 120x80 while cond is 64x64), so ref_t carries its own ref_h/ref_w.
    ggml_tensor *sample_t =
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, hp.in_channels, B);
    ggml_tensor *ref_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ref_w, ref_h,
                                            hp.in_channels, B);
    ggml_tensor *context_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                hp.cross_attention_dim, L, B);
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, 65536, false);
    ggml_tensor *emb_in = nullptr;
    ggml_tensor *temb =
            time_embedding(ctx, m, timestep, B, &emb_in);  // [1280, B]

    // ── shared pass builder. `ref` maps each attn1 name to its w-pass input
    // (null during the w-pass itself).
    ggml_tensor *ref_store[16] = {};  // down 2x3 + mid 1 + up 3x3
    auto build_pass = [&](ggml_tensor *x_in, bool is_w) {
        std::vector<ggml_tensor *> skips;
        ggml_tensor *cur =
                conv_f32(ctx, to_conv(ctx, x_in), get_t(m, "conv_in.weight"),
                         get_t(m, "conv_in.bias"), 1, 1);
        skips.push_back(cur);
        // ── down ──
        for (int lvl = 0; lvl < 4; ++lvl) {
            const std::string bp = "down_blocks." + std::to_string(lvl) + ".";
            for (int j = 0; j < hp.layers_per_block; ++j) {
                // CrossAttnDownBlock2D: resnet then transformer per layer. The
                // skip connection must carry the POST-attention state (torch's
                // CrossAttnDownBlock2D appends output_states after attn);
                // DownBlock2D (lvl 3) has no attention.
                cur = resnet(ctx, m, cur, temb,
                             bp + "resnets." + std::to_string(j) + ".");
                if (lvl < 3) {
                    const int ridx = lvl * hp.layers_per_block + j;
                    cur = transformer(
                            ctx, m, cur, context_t,
                            bp + "attentions." + std::to_string(j) + ".",
                            hp.num_heads[lvl], is_w ? nullptr : ref_store[ridx],
                            is_w ? &ref_store[ridx] : nullptr);
                }
                skips.push_back(cur);
            }
            if (lvl < 3) {
                // symmetric stride-2 pad-1 downsample (downsample_padding=1)
                cur = conv_f32(ctx, cur,
                               get_t(m, bp + "downsamplers.0.conv.weight"),
                               get_t(m, bp + "downsamplers.0.conv.bias"), 2, 1);
                skips.push_back(cur);
            }
        }
        // ── mid ──
        cur = resnet(ctx, m, cur, temb, "mid_block.resnets.0.");
        cur = transformer(ctx, m, cur, context_t, "mid_block.attentions.0.",
                          hp.num_heads[3], is_w ? nullptr : ref_store[6],
                          is_w ? &ref_store[6] : nullptr);
        cur = resnet(ctx, m, cur, temb, "mid_block.resnets.1.");
        // ── up ──
        for (int lvl = 0; lvl < 4; ++lvl) {
            const std::string bp = "up_blocks." + std::to_string(lvl) + ".";
            const bool has_attn = lvl > 0;  // UpBlock2D, CrossAttnUpBlock2D×3
            for (int j = 0; j < hp.layers_per_block + 1; ++j) {
                const std::string rp =
                        bp + "resnets." + std::to_string(j) + ".";
                ggml_tensor *skip = skips.back();
                skips.pop_back();
                cur = ggml_concat(ctx, cur, skip, 2);  // channel cat on ne2
                cur = resnet(ctx, m, cur, temb, rp);
                if (has_attn) {
                    const int heads = hp.num_heads[3 - lvl];
                    const int ridx = 7 + (lvl - 1) * 3 + j;  // up1..3, j0..2
                    cur = transformer(
                            ctx, m, cur, context_t,
                            bp + "attentions." + std::to_string(j) + ".", heads,
                            is_w ? nullptr : ref_store[ridx],
                            is_w ? &ref_store[ridx] : nullptr);
                }
            }
            if (lvl < 3) {
                cur = ggml_upscale(ctx, cur, 2, GGML_SCALE_MODE_NEAREST);
                cur = conv_f32(ctx, cur,
                               get_t(m, bp + "upsamplers.0.conv.weight"),
                               get_t(m, bp + "upsamplers.0.conv.bias"), 1, 1);
            }
        }
        cur = ggml_group_norm(ctx, cur, hp.norm_num_groups, hp.norm_eps);
        {
            ggml_tensor *gw = get_t(m, "conv_norm_out.weight");
            ggml_tensor *gb = get_t(m, "conv_norm_out.bias");
            gw = ggml_reshape_4d(ctx, gw, 1, 1, gw->ne[0], 1);
            gb = ggml_reshape_4d(ctx, gb, 1, 1, gb->ne[0], 1);
            cur = ggml_add(ctx, ggml_mul(ctx, cur, gw), gb);
        }
        cur = ggml_silu(ctx, cur);
        cur = conv_f32(ctx, cur, get_t(m, "conv_out.weight"),
                       get_t(m, "conv_out.bias"), 1, 1);
        return cur;
    };

    ggml_tensor *w_out = build_pass(ref_t, /*is_w=*/true);  // ref_vec filled
    ggml_tensor *r_out = build_pass(sample_t, /*is_w=*/false);

    ggml_tensor *result_t = ggml_cont(ctx, r_out);
    ggml_set_output(result_t);

    ggml_gallocr_t alloc =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    // per-head attention explodes the node count — size generously
    ggml_build_forward_expand(gf, w_out);     // tail of the w chain
    ggml_build_forward_expand(gf, result_t);  // tail of the r chain
    ggml_gallocr_alloc_graph(alloc, gf);
    {
        std::vector<float> ev(320);
        const int half = 160;
        for (int i = 0; i < half; ++i)
            ev[i] = std::exp(-(float)std::log(10000.0) * (float)i /
                             (float)(half - m.hp.freq_shift));
        std::vector<float> e(320);
        for (int i = 0; i < half; ++i) {
            e[i] = std::cos(timestep * ev[i]);
            e[half + i] = std::sin(timestep * ev[i]);
        }
        ggml_backend_tensor_set(emb_in, e.data(), 0, e.size() * sizeof(float));
    }
    {
        std::vector<float> one(65536, 1.0f);  // max WH; slice per tensor
        for (auto *t : g_u_fill_ones_wh) {
            ggml_backend_tensor_set(t, one.data(), 0, t->ne[0] * sizeof(float));
        }
        for (auto *t : g_u_fill_ones_cpg) {
            ggml_backend_tensor_set(t, one.data(), 0, t->ne[0] * sizeof(float));
        }
        std::vector<float> epsv((size_t)m.hp.norm_num_groups * B,
                                m.hp.norm_eps);
        for (auto *t : g_u_fill_eps) {
            ggml_backend_tensor_set(t, epsv.data(), 0,
                                    epsv.size() * sizeof(float));
        }
        for (auto *t : g_u_fill_ones_gb) {
            ggml_backend_tensor_set(t, one.data(), 0, t->ne[0] * sizeof(float));
        }
    }
    ggml_backend_tensor_set(sample_t, sample, 0,
                            (size_t)B * hp.in_channels * H * W * sizeof(float));
    ggml_backend_tensor_set(
            ref_t, ref_sample, 0,
            (size_t)B * hp.in_channels * ref_h * ref_w * sizeof(float));
    ggml_backend_tensor_set(
            context_t, context, 0,
            (size_t)B * hp.cross_attention_dim * L * sizeof(float));
    ggml_backend_graph_compute(m.backend, gf);
    float *result = (float *)malloc((size_t)B * hp.out_channels * H * W *
                                    sizeof(float));
    ggml_backend_tensor_get(
            result_t, result, 0,
            (size_t)B * hp.out_channels * H * W * sizeof(float));
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    if (out_h) *out_h = H;
    if (out_w) *out_w = W;
    return result;
}

}  // namespace instantmesh
