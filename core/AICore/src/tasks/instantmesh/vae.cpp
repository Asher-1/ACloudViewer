// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// AutoencoderKL implementation. See vae.hpp for conventions.
//
// LAYOUT CONTRACT (ggml v0.21, pinned by tests/test_conv_layout.cpp against
// in-process naive references — the earlier "ne mislabeled" theory in
// docs/ALIGNMENT.md was a misdiagnosis; a contiguous ggml tensor cannot
// disagree with its own labels):
//   * ggml_conv_2d consumes a [W,H,IC,N]-shaped CONTIGUOUS input (im2col
//     steps channels via nb12, reads each channel plane as row-major [H,W])
//     and returns a plain contiguous [OW,OH,OC,N] tensor (memory == torch
//     NCHW). Its numerics floor is the internal F16 im2col window
//     quantization (~f16 eps, relative).
//   * ggml_group_norm normalizes over ne0*ne1 with groups over ne2 — on a
//     contiguous [W,H,C,N] tensor that is exactly torch GroupNorm (groups =
//     consecutive channels = consecutive H*W memory blocks).
//   * Therefore the whole model runs in ONE layout: ne=[W,H,C,B] contiguous,
//     whose byte order IS torch NCHW [B,C,H,W] (w fastest, c second-slowest,
//     b slowest — identical flat index for any B). Host image/latent buffers
//     in torch NCHW byte order map onto ne=[W,H,C,B] tensors with a plain
//     byte copy: NO conversion at either boundary, NO conv output reorder.
//     The only genuine gathers are inside attention ([C,HW] flatten for
//     mul_mat and back).
#include "vae.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "numerics.hpp"

namespace instantmesh {

namespace {

ggml_tensor *get_t(const VaeModel &m, const char *name) {
    auto it = m.gguf.tensors.find(name);
    if (it == m.gguf.tensors.end()) {
        std::fprintf(stderr, "vae: missing tensor '%s'\n", name);
        std::abort();
    }
    return it->second;
}

ggml_tensor *conv(ggml_context *ctx,
                  ggml_tensor *x,
                  ggml_tensor *w,
                  ggml_tensor *b,
                  int stride,
                  int pad) {
    // Retain official conv_2d F16 im2col/layout; explicitly request F32 GEMM
    // accumulation. Default Vulkan F16 accumulation loses small terms; F32
    // output alone does not prove accumulation precision.
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
        // per-channel bias on [W,H,OC,B] memory: a [1,1,OC,1] view broadcasts.
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        y = ggml_add(ctx, y, b);
    }
    return y;
}

// Same layout contract as conv(), but keeps the im2col windows in F32:
// ggml_conv_2d hardcodes a F16 im2col for non-bf16 kernels (a ~f16-eps
// relative error per conv that accumulates over the deep stack), which is
// unacceptable for the encoder whose cond_lat conditions 75 denoising
// steps. Identical output layout; costs 2x im2col memory + f32 gemm time.
ggml_tensor *conv_f32(ggml_context *ctx,
                      ggml_tensor *x,
                      ggml_tensor *w,
                      ggml_tensor *b,
                      int stride,
                      int pad) {
    ggml_tensor *im2col =
            ggml_im2col(ctx, w, x, stride, stride, pad, pad, 1, 1, true,
                        GGML_TYPE_F32);                  // [ICKHKW, OW, OH, N]
    ggml_tensor *wk = ggml_cast(ctx, w, GGML_TYPE_F32);  // keep src1 F32
    ggml_tensor *result = mul_mat_exact(
            ctx,
            ggml_reshape_2d(ctx, im2col, im2col->ne[0],
                            im2col->ne[3] * im2col->ne[2] * im2col->ne[1]),
            ggml_reshape_2d(ctx, wk, wk->ne[0] * wk->ne[1] * wk->ne[2],
                            wk->ne[3]));
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2],
                             im2col->ne[3], w->ne[3]);
    result = ggml_cont(ctx,
                       ggml_permute(ctx, result, 0, 1, 3, 2));  // [W,H,OC,N]
    if (b) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        result = ggml_add(ctx, result, b);
    }
    return result;
}

// GroupNorm(32) + affine + SiLU on torch-order [C,H,W,B]. ggml_group_norm
// needs the spatial plane on ne0*ne1, so permute in and out.
ggml_tensor *gn_silu(ggml_context *ctx,
                     ggml_tensor *x,
                     const VaeModel &m,
                     const char *wname,
                     const char *bname) {
    // x is already [W,H,C,B] memory — exactly what ggml_group_norm consumes
    // (spatial plane on ne0*ne1, groups over ne2).
    ggml_tensor *p =
            ggml_group_norm(ctx, x, m.hp.norm_num_groups, m.hp.norm_eps);
    ggml_tensor *w = get_t(m, wname);
    ggml_tensor *b = get_t(m, bname);
    w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], 1);
    b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
    p = ggml_silu(ctx, ggml_add(ctx, ggml_mul(ctx, p, w), b));
    return p;  // stays [W,H,C,B] — conv_2d's input shape
}

// ResnetBlock2D with temb=None. f32conv: build convs with F32 im2col
// (encoder path; see conv_f32).
ggml_tensor *resnet(ggml_context *ctx,
                    const VaeModel &m,
                    ggml_tensor *x,
                    const std::string &prefix,
                    bool f32conv) {
    auto conv_fn = f32conv ? conv_f32 : conv;
    ggml_tensor *p =
            ggml_group_norm(ctx, x, m.hp.norm_num_groups, m.hp.norm_eps);
    ggml_tensor *h;
    {
        ggml_tensor *w = get_t(m, (prefix + "norm1.weight").c_str());
        ggml_tensor *b = get_t(m, (prefix + "norm1.bias").c_str());
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], 1);
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        h = ggml_silu(ctx, ggml_add(ctx, ggml_mul(ctx, p, w), b));
    }
    h = conv_fn(ctx, h, get_t(m, (prefix + "conv1.weight").c_str()),
                get_t(m, (prefix + "conv1.bias").c_str()), 1, 1);
    h = gn_silu(ctx, h, m, (prefix + "norm2.weight").c_str(),
                (prefix + "norm2.bias").c_str());
    h = conv_fn(ctx, h, get_t(m, (prefix + "conv2.weight").c_str()),
                get_t(m, (prefix + "conv2.bias").c_str()), 1, 1);
    char nm[256];
    std::snprintf(nm, sizeof(nm), "%sconv_shortcut.weight", prefix.c_str());
    if (m.gguf.tensors.count(nm)) {
        std::snprintf(nm, sizeof(nm), "%sconv_shortcut.bias", prefix.c_str());
        x = conv_fn(ctx, x, get_t(m, (prefix + "conv_shortcut.weight").c_str()),
                    get_t(m, nm), 1, 0);
    }
    return ggml_add(ctx, h, x);
}

// Vanilla spatial attention, heads=1 (GroupNorm → to_q/k/v → softmax →
// to_out, residual). torch flatten order is a plain reshape here.
ggml_tensor *attention(ggml_context *ctx,
                       const VaeModel &m,
                       ggml_tensor *x,
                       const std::string &prefix) {
    const int C = x->ne[2];  // channels live on ne2 in [W,H,C,B]
    const int HW = x->ne[0] * x->ne[1];
    const int B = x->ne[3];
    const int W_dim = x->ne[0], H_dim = x->ne[1];
    ggml_tensor *residual = x;
    // x is [W,H,C,B] old-layout memory: GN + affine in place, then reorder
    // into torch [C,HW] (gather permute + cont) so mul_mat's channel dim
    // lines up. NOTE: the AttentionBlock group_norm HAS affine params.
    ggml_tensor *h =
            ggml_group_norm(ctx, x, m.hp.norm_num_groups, m.hp.norm_eps);
    {
        ggml_tensor *gw = get_t(m, (prefix + "group_norm.weight").c_str());
        ggml_tensor *gb = get_t(m, (prefix + "group_norm.bias").c_str());
        gw = ggml_reshape_4d(ctx, gw, 1, 1, gw->ne[0], 1);
        gb = ggml_reshape_4d(ctx, gb, 1, 1, gb->ne[0], 1);
        h = ggml_add(ctx, ggml_mul(ctx, h, gw), gb);
    }
    h = ggml_cont(ctx, ggml_permute(ctx, h, 2, 1, 0, 3));  // [C,H,W,B]
    h = ggml_reshape_3d(ctx, h, C, HW, B);                 // [C,HW,B]

    ggml_tensor *q =
            mul_mat_exact(ctx, get_t(m, (prefix + "to_q.weight").c_str()), h);
    ggml_tensor *k =
            mul_mat_exact(ctx, get_t(m, (prefix + "to_k.weight").c_str()), h);
    ggml_tensor *v =
            mul_mat_exact(ctx, get_t(m, (prefix + "to_v.weight").c_str()), h);
    if (m.gguf.tensors.count(prefix + "to_q.bias")) {
        q = ggml_add(ctx, q, get_t(m, (prefix + "to_q.bias").c_str()));
        k = ggml_add(ctx, k, get_t(m, (prefix + "to_k.bias").c_str()));
        v = ggml_add(ctx, v, get_t(m, (prefix + "to_v.bias").c_str()));
    }
    const float scale = 1.0f / std::sqrt((float)C);
    // att[i,j] = softmax_k(q_j . k_i * scale): mul_mat(k,q) → [HW_k, HW_q, B].
    ggml_tensor *att = ggml_scale(ctx, mul_mat_exact(ctx, k, q), scale);
    att = ggml_soft_max_ext(ctx, att, nullptr, 1.0f, 0.0f);
    ggml_tensor *vt =
            ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));  // [HW, C, B]
    ggml_tensor *o = mul_mat_exact(ctx, att, vt);              // [HW_q, C, B]
    ggml_tensor *ot =
            ggml_cont(ctx, ggml_permute(ctx, o, 1, 0, 2, 3));  // [C, HW_q, B]
    ggml_tensor *out = mul_mat_exact(
            ctx, get_t(m, (prefix + "to_out.0.weight").c_str()), ot);
    if (m.gguf.tensors.count(prefix + "to_out.0.bias")) {
        out = ggml_add(ctx, out, get_t(m, (prefix + "to_out.0.bias").c_str()));
    }
    out = ggml_reshape_4d(ctx, out, C, H_dim, W_dim,
                          B);  // torch-order [C,H,W,B]
    out = ggml_cont(ctx,
                    ggml_permute(ctx, out, 2, 1, 0, 3));  // back to [W,H,C,B]
    return ggml_add(ctx, residual, out);
}

}  // namespace

bool vae_load(const std::string &path,
              ggml_backend_t backend,
              VaeModel &out,
              std::string *error) {
    if (!load_gguf(path, backend, out.gguf, error)) return false;
    out.backend = backend;
    const gguf_context *g = out.gguf.gguf;
    const char *chs = kv_str(g, "vae.block_out_channels", "128,256,512,512");
    if (std::sscanf(chs, "%d,%d,%d,%d", &out.hp.block_out_channels[0],
                    &out.hp.block_out_channels[1],
                    &out.hp.block_out_channels[2],
                    &out.hp.block_out_channels[3]) != 4) {
        out.hp.block_out_channels[0] = 128;
        out.hp.block_out_channels[1] = 256;
        out.hp.block_out_channels[2] = 512;
        out.hp.block_out_channels[3] = 512;
    }
    out.hp.latent_channels = kv_i32(g, "vae.latent_channels", 4);
    out.hp.norm_num_groups = kv_i32(g, "vae.norm_num_groups", 32);
    out.hp.norm_eps = kv_f32(g, "vae.norm_eps", 1e-6f);
    out.hp.scaling_factor = kv_f32(g, "vae.scaling_factor", 0.18215f);
    return true;
}

float *vae_encode_mode(const VaeModel &m,
                       const float *image,
                       int B,
                       int H,
                       int W,
                       int *out_h,
                       int *out_w) {
    ggml_init_params ip = {(size_t)1 << 30, nullptr, true};
    ggml_context *ctx = ggml_init(ip);

    // Host image arrives as torch NCHW [B,3,H,W] bytes. ne=[W,H,3,B] has the
    // identical flat index (w fastest, c 2nd-slowest, b slowest) for any B —
    // a plain byte copy maps it onto conv_2d's native input layout.
    ggml_tensor *image_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, 3, B);
    ggml_tensor *k_in = get_t(m, "encoder.conv_in.weight");
    ggml_tensor *y = conv_f32(ctx, image_t, k_in, nullptr, 1, 1);
    // conv_2d output is already [W,H,OC,B] (LAYOUT CONTRACT).
    ggml_tensor *cur = y;
    cur = ggml_add(ctx, cur,
                   ggml_reshape_4d(ctx, get_t(m, "encoder.conv_in.bias"), 1, 1,
                                   get_t(m, "encoder.conv_in.bias")->ne[0], 1));
    for (int i = 0; i < 4; ++i) {
        const std::string p = "encoder.down_blocks." + std::to_string(i) + ".";
        cur = resnet(ctx, m, cur, p + "resnets.0.", /*f32conv=*/true);
        cur = resnet(ctx, m, cur, p + "resnets.1.", /*f32conv=*/true);
        if (i < 3) {
            // diffusers Downsample2D (padding=0): asymmetric zero-pad of the
            // bottom/right edge (F.pad(x,(0,1,0,1))) THEN a stride-2 padding-0
            // conv. ggml_pad p0/p1/p2/p3 pad AFTER ne0/ne1/ne2/ne3 resp.
            cur = conv_f32(ctx, ggml_pad(ctx, cur, 1, 1, 0, 0),
                           get_t(m, (p + "downsamplers.0.conv.weight").c_str()),
                           get_t(m, (p + "downsamplers.0.conv.bias").c_str()),
                           2, 0);
        }
    }
    cur = resnet(ctx, m, cur, "encoder.mid_block.resnets.0.", /*f32conv=*/true);
    cur = attention(ctx, m, cur, "encoder.mid_block.attentions.0.");
    cur = resnet(ctx, m, cur, "encoder.mid_block.resnets.1.", /*f32conv=*/true);
    cur = gn_silu(ctx, cur, m, "encoder.conv_norm_out.weight",
                  "encoder.conv_norm_out.bias");
    cur = conv_f32(ctx, cur, get_t(m, "encoder.conv_out.weight"),
                   get_t(m, "encoder.conv_out.bias"), 1, 1);
    cur = conv_f32(ctx, cur, get_t(m, "quant_conv.weight"),
                   get_t(m, "quant_conv.bias"), 1, 0);
    // cur is [W',H',2*latent,B] whose memory is torch [B,2C,H',W']; the
    // posterior mode = the first `latent_channels` channels — a plain ne2
    // slice.
    ggml_tensor *mode =
            ggml_view_4d(ctx, cur, cur->ne[0], cur->ne[1], m.hp.latent_channels,
                         B, cur->nb[1], cur->nb[2], cur->nb[3], 0);
    ggml_tensor *result_t =
            ggml_cont(ctx, mode);  // [W',H',C,B] == torch [B,C,H',W'] bytes

    ggml_gallocr_t alloc =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result_t);
    // Mark everything read post-compute as a graph output: gallocr never
    // recycles output-flagged buffers (ggml-alloc.c "graph outputs are never
    // freed"), while ordinary dead tensors are reused by later nodes right
    // after their direct consumers run — which previously corrupted every
    // staged dump (the historical "galloc reuse interference").
    ggml_set_output(result_t);
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return nullptr;
    }
    ggml_backend_tensor_set(image_t, image, 0,
                            (size_t)B * 3 * H * W * sizeof(float));
    ggml_backend_graph_compute(m.backend, gf);

    const int lh = H / 8, lw = W / 8;
    float *result = (float *)malloc((size_t)B * m.hp.latent_channels * lh * lw *
                                    sizeof(float));
    ggml_backend_tensor_get(
            result_t, result, 0,
            (size_t)B * m.hp.latent_channels * lh * lw * sizeof(float));
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    if (out_h) *out_h = lh;
    if (out_w) *out_w = lw;
    return result;
}

float *vae_decode(const VaeModel &m,
                  const float *latents,
                  int B,
                  int h,
                  int w,
                  int *out_h,
                  int *out_w) {
    const int H = h * 8, W = w * 8;
    ggml_init_params ip = {(size_t)4 << 30, nullptr, true};
    ggml_context *ctx = ggml_init(ip);

    // latents arrive as torch [B,C,h,w] bytes → ne=[w,h,C,B] (same flat
    // index for any B); the scale keeps that layout, which is conv_2d's
    // native input shape — feed it directly.
    ggml_tensor *lat_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w, h,
                                            m.hp.latent_channels, B);
    ggml_tensor *scaled = ggml_scale(
            ctx, lat_t, 1.0f / m.hp.scaling_factor);  // decode(latents/scaling)
    ggml_tensor *cur = scaled;
    cur = conv(ctx, cur, get_t(m, "post_quant_conv.weight"),
               get_t(m, "post_quant_conv.bias"), 1, 0);
    cur = conv(ctx, cur, get_t(m, "decoder.conv_in.weight"),
               get_t(m, "decoder.conv_in.bias"), 1, 1);
    cur = resnet(ctx, m, cur, "decoder.mid_block.resnets.0.",
                 /*f32conv=*/false);
    cur = attention(ctx, m, cur, "decoder.mid_block.attentions.0.");
    cur = resnet(ctx, m, cur, "decoder.mid_block.resnets.1.",
                 /*f32conv=*/false);
    for (int i = 0; i < 4; ++i) {
        const std::string p = "decoder.up_blocks." + std::to_string(i) + ".";
        for (int j = 0; j < 3; ++j) {
            cur = resnet(ctx, m, cur, p + "resnets." + std::to_string(j) + ".",
                         /*f32conv=*/false);
        }
        if (i < 3) {
            // nearest-neighbor ×2 on the spatial plane (ne0,ne1 of [W,H,C,B])
            cur = ggml_upscale(ctx, cur, 2, GGML_SCALE_MODE_NEAREST);
            cur = conv(ctx, cur,
                       get_t(m, (p + "upsamplers.0.conv.weight").c_str()),
                       get_t(m, (p + "upsamplers.0.conv.bias").c_str()), 1, 1);
        }
    }
    cur = gn_silu(ctx, cur, m, "decoder.conv_norm_out.weight",
                  "decoder.conv_norm_out.bias");
    cur = conv(ctx, cur, get_t(m, "decoder.conv_out.weight"),
               get_t(m, "decoder.conv_out.bias"), 1, 1);
    // cur is [W,H,3,B] whose memory is already torch [B,3,H,W] — read as-is.
    ggml_tensor *result_t = cur;
    ggml_set_output(
            result_t);  // read post-compute; keep gallocr from recycling it

    ggml_gallocr_t alloc =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result_t);
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return nullptr;
    }
    ggml_backend_tensor_set(
            lat_t, latents, 0,
            (size_t)B * m.hp.latent_channels * h * w * sizeof(float));
    ggml_backend_graph_compute(m.backend, gf);

    float *result = (float *)malloc((size_t)B * 3 * H * W * sizeof(float));
    ggml_backend_tensor_get(result_t, result, 0,
                            (size_t)B * 3 * H * W * sizeof(float));
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    if (out_h) *out_h = H;
    if (out_w) *out_w = W;
    return result;
}

}  // namespace instantmesh
