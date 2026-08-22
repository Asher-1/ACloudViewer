// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/rmbg/swin_graph.hpp"

#include <cmath>
#include <cstring>

namespace rmbg {

static ggml_tensor* make_tensor_1d(ggml_context* ctx, const char* name, int n) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_name(t, name);
    return t;
}

static ggml_tensor* make_tensor_4d(ggml_context* ctx,
                                   const char* name,
                                   int64_t d0,
                                   int64_t d1,
                                   int64_t d2,
                                   int64_t d3) {
    ggml_tensor* t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d0, d1, d2, d3);
    ggml_set_name(t, name);
    return t;
}

static ggml_tensor* conv_2d_f32(ggml_context* ctx,
                                ggml_tensor* weight,
                                ggml_tensor* input,
                                int stride,
                                int pad) {
    ggml_tensor* col = ggml_im2col(ctx, weight, input, stride, stride, pad, pad,
                                   1, 1, true, GGML_TYPE_F32);
    ggml_tensor* out = ggml_mul_mat(
            ctx,
            ggml_reshape_2d(ctx, col, col->ne[0],
                            col->ne[1] * col->ne[2] * col->ne[3]),
            ggml_reshape_2d(ctx, weight,
                            weight->ne[0] * weight->ne[1] * weight->ne[2],
                            weight->ne[3]));
    out = ggml_reshape_4d(ctx, out, col->ne[1], col->ne[2], col->ne[3],
                          weight->ne[3]);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 1, 3, 2));
}

static void layer_norm_blc(std::vector<float>& x,
                           int rows,
                           int cols,
                           const std::vector<float>& gamma,
                           const std::vector<float>& beta,
                           float eps) {
    for (int r = 0; r < rows; ++r) {
        float* row = x.data() + (size_t)r * cols;
        double mean = 0, var = 0;
        for (int c = 0; c < cols; ++c) mean += row[c];
        mean /= cols;
        for (int c = 0; c < cols; ++c) {
            double d = row[c] - mean;
            var += d * d;
        }
        var /= cols;
        double inv = 1.0 / std::sqrt(var + eps);
        for (int c = 0; c < cols; ++c)
            row[c] = (float)((row[c] - mean) * inv * gamma[c] + beta[c]);
    }
}

bool SwinPatchEmbedGraph::init(ggml_backend_t backend,
                               const WeightMap& w,
                               std::string& err) {
    free();
    backend_ = backend;
    const std::vector<float>* pw = w.get_f32("bb_patch_embed_proj_weight");
    const std::vector<float>* pb = w.get_f32("bb_patch_embed_proj_bias");
    const std::vector<float>* nw = w.get_f32("bb_patch_embed_norm_weight");
    const std::vector<float>* nb = w.get_f32("bb_patch_embed_norm_bias");
    if (!pw || !pb || !nw || !nb) {
        err = "patch_embed weights missing";
        return false;
    }
    embed_dim_ = (int)nw->size();
    (void)nb;

    ggml_init_params p_w{ggml_tensor_overhead() * 8, nullptr, true};
    ctx_w_ = ggml_init(p_w);
    w_conv_ = make_tensor_4d(ctx_w_, "pe_conv_w", 4, 4, 3, embed_dim_);
    b_conv_ = make_tensor_1d(ctx_w_, "pe_conv_b", embed_dim_);
    w_ln_ = make_tensor_1d(ctx_w_, "pe_ln_w", embed_dim_);
    b_ln_ = make_tensor_1d(ctx_w_, "pe_ln_b", embed_dim_);

    buf_weights_ = ggml_backend_alloc_ctx_tensors(ctx_w_, backend_);
    if (!buf_weights_) {
        err = "weight buffer alloc failed";
        return false;
    }

    std::vector<float> gw((size_t)embed_dim_ * 3 * 4 * 4);
    for (int oc = 0; oc < embed_dim_; ++oc)
        for (int ic = 0; ic < 3; ++ic)
            for (int kh = 0; kh < 4; ++kh)
                for (int kw = 0; kw < 4; ++kw)
                    gw[(size_t)kw +
                       4 * ((size_t)kh + 4 * ((size_t)ic + 3 * oc))] =
                            (*pw)[(((size_t)oc * 3 + ic) * 4 + kh) * 4 + kw];

    ggml_backend_tensor_set(w_conv_, gw.data(), 0, gw.size() * sizeof(float));
    ggml_backend_tensor_set(b_conv_, pb->data(), 0, pb->size() * sizeof(float));
    ggml_backend_tensor_set(w_ln_, nw->data(), 0, nw->size() * sizeof(float));
    ggml_backend_tensor_set(b_ln_, nb->data(), 0, nb->size() * sizeof(float));
    return true;
}

void SwinPatchEmbedGraph::free() {
    if (buf_weights_) {
        ggml_backend_buffer_free(buf_weights_);
        buf_weights_ = nullptr;
    }
    if (ctx_w_) {
        ggml_free(ctx_w_);
        ctx_w_ = nullptr;
    }
    if (ctx_g_) {
        ggml_free(ctx_g_);
        ctx_g_ = nullptr;
    }
    graph_ = nullptr;
    backend_ = nullptr;
}

bool SwinPatchEmbedGraph::forward(const std::vector<float>& nchw,
                                  int H,
                                  int W,
                                  std::vector<float>& patch_nchw,
                                  std::vector<float>& tokens_blc,
                                  std::string& err) {
    if (!backend_ || !ctx_w_) {
        err = "graph not initialized";
        return false;
    }
    if ((int)nchw.size() != 3 * H * W) {
        err = "input size mismatch";
        return false;
    }

    const int Wh = H / 4, Ww = W / 4;
    if (ctx_g_) {
        ggml_free(ctx_g_);
        ctx_g_ = nullptr;
    }

    ggml_init_params p_g{ggml_tensor_overhead() * 12 + ggml_graph_overhead(),
                         nullptr, true};
    ctx_g_ = ggml_init(p_g);

    inp_ = make_tensor_4d(ctx_g_, "inp", W, H, 3, 1);
    ggml_set_input(inp_);
    out_conv_ = conv_2d_f32(ctx_g_, w_conv_, inp_, 4, 0);
    ggml_set_output(out_conv_);

    graph_ = ggml_new_graph(ctx_g_);
    ggml_build_forward_expand(graph_, out_conv_);

    ggml_backend_buffer_t buf_compute =
            ggml_backend_alloc_ctx_tensors(ctx_g_, backend_);
    if (!buf_compute) {
        err = "compute buffer alloc failed";
        return false;
    }

    // ggml [W,H,C,N] is contiguous in W first, exactly matching NCHW storage.
    ggml_backend_tensor_set(inp_, nchw.data(), 0, nchw.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend_, graph_) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf_compute);
        err = "graph compute failed";
        return false;
    }

    std::vector<float> raw((size_t)Ww * Wh * embed_dim_);
    ggml_backend_tensor_get(out_conv_, raw.data(), 0,
                            raw.size() * sizeof(float));

    std::vector<float> bias(embed_dim_);
    ggml_backend_tensor_get(b_conv_, bias.data(), 0,
                            bias.size() * sizeof(float));

    tokens_blc.assign((size_t)Wh * Ww * embed_dim_, 0.f);
    patch_nchw.assign((size_t)embed_dim_ * Wh * Ww, 0.f);
    for (int h = 0; h < Wh; ++h)
        for (int w = 0; w < Ww; ++w)
            for (int c = 0; c < embed_dim_; ++c) {
                const float v =
                        raw[w + (size_t)h * Ww + (size_t)c * Ww * Wh] + bias[c];
                tokens_blc[((size_t)h * Ww + w) * embed_dim_ + c] = v;
                patch_nchw[((size_t)c * Wh + h) * Ww + w] = v;
            }

    std::vector<float> lnw(embed_dim_), lnb(embed_dim_);
    ggml_backend_tensor_get(w_ln_, lnw.data(), 0, lnw.size() * sizeof(float));
    ggml_backend_tensor_get(b_ln_, lnb.data(), 0, lnb.size() * sizeof(float));
    layer_norm_blc(tokens_blc, Wh * Ww, embed_dim_, lnw, lnb, 1e-5f);

    for (int c = 0; c < embed_dim_; ++c)
        for (int h = 0; h < Wh; ++h)
            for (int w = 0; w < Ww; ++w)
                patch_nchw[((size_t)c * Wh + h) * Ww + w] =
                        tokens_blc[((size_t)h * Ww + w) * embed_dim_ + c];

    ggml_backend_buffer_free(buf_compute);
    return true;
}

}  // namespace rmbg
