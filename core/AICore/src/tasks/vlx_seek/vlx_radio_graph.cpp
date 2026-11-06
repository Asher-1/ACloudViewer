// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#include "tasks/vlx_seek/vlx_radio_graph.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "tasks/vlx_seek/vlx_ggml_util.h"
#include "tasks/vlx_seek/vlx_gguf_loader.h"
#include "tasks/vlx_seek/vlx_image_prep.h"

namespace vlx {

namespace {

constexpr float kLayerNormEps = 1e-5f;  // torch nn.LayerNorm default

// Bilinear resize, align_corners=False (half-pixel centers): matches
// torch F.interpolate(mode='bilinear', align_corners=False) used by the
// eval CPE path (pos grid -> (max,max), then top-left crop to (gh,gw)).
void bilinear_resize_nchw(
        const float* src, int c, int sh, int sw, float* dst, int dh, int dw) {
    if (sh == dh && sw == dw) {
        std::memcpy(dst, src, sizeof(float) * size_t(c) * sh * sw);
        return;
    }
    const float ry = float(sh) / float(dh);
    const float rx = float(sw) / float(dw);
    auto clamp_v = [](float v, int len) {
        v = v < 0.f ? 0.f : v;
        return v > float(len - 1) ? float(len - 1) : v;
    };
    for (int y = 0; y < dh; ++y) {
        const float fy = clamp_v((y + 0.5f) * ry - 0.5f, sh);
        const int y0 = int(std::floor(fy));
        const int y1 = std::min(y0 + 1, sh - 1);
        const float wy = fy - float(y0);
        for (int x = 0; x < dw; ++x) {
            const float fx = clamp_v((x + 0.5f) * rx - 0.5f, sw);
            const int x0 = int(std::floor(fx));
            const int x1 = std::min(x0 + 1, sw - 1);
            const float wx = fx - float(x0);
            const float w00 = (1 - wy) * (1 - wx), w01 = (1 - wy) * wx;
            const float w10 = wy * (1 - wx), w11 = wy * wx;
            for (int ch = 0; ch < c; ++ch) {
                const float* p = src + size_t(ch) * sh * sw;
                dst[(size_t(ch) * dh + y) * dw + x] =
                        w00 * p[size_t(y0) * sw + x0] +
                        w01 * p[size_t(y0) * sw + x1] +
                        w10 * p[size_t(y1) * sw + x0] +
                        w11 * p[size_t(y1) * sw + x1];
            }
        }
    }
}

}  // namespace

struct RadioGraph::Impl {
    ggml_backend* backend = nullptr;
    const GgufFile* gguf = nullptr;
    RadioConfig cfg;

    // ONE graph for the whole tower (conditioning + 27 blocks). The former
    // per-block segment graphs died of the cross-graph expand trap (see the
    // mmproj post-mortem): segment inputs carried src pointers into the
    // producer's context, every gallocr re-pointed the shared tensors' data,
    // and later allocators won — garbage inputs, O(depth^2) work, and the
    // real-weight L deltas (0.103/0.138/0.191/0.809) that were wrongly
    // attributed to flash_attn_ext.

    // weights (built once per resolution change; referenced by all segments)
    ggml_tensor *w_patch = nullptr, *w_cls = nullptr;
    ggml_tensor *w_ln1[32] = {}, *b_ln1[32] = {};
    ggml_tensor *w_ln2[32] = {}, *b_ln2[32] = {};
    ggml_tensor *w_qkv[32] = {}, *b_qkv[32] = {};
    ggml_tensor *w_proj[32] = {}, *b_proj[32] = {};
    ggml_tensor *g_ls1[32] = {}, *g_ls2[32] = {};
    ggml_tensor *w_fc1[32] = {}, *b_fc1[32] = {};
    ggml_tensor *w_fc2[32] = {}, *b_fc2[32] = {};

    // Weights: one ggml_context allocated in a backend buffer, live tensors
    // referenced directly by the graph (quantized weights stay quantized and
    // are consumed by MUL_MAT without dequant copies, per the sibling repos).
    ggml_context* wctx = nullptr;
    ggml_backend_buffer* wbuf = nullptr;
    std::vector<std::pair<ggml_tensor*, const TensorInfo*>> bind;
    // Per-resolution compute state.
    ggml_context* gctx = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_gallocr* allocr = nullptr;
    // Inputs live in their own context + buffer, allocated BEFORE the graph
    // pool (map-anything pattern) so gallocr skips them and their addresses
    // survive across runs.
    ggml_context* ictx = nullptr;
    ggml_backend_buffer* ibuf = nullptr;
    ggml_tensor *in_img = nullptr, *in_pids = nullptr, *in_pos = nullptr;
    std::vector<float> host_img;  // conditioned input (host scratch)
    // Level taps: ggml_cpy targets living in ictx with NCHW shapes (gw, gh, C)
    // — c slowest, matching FeatureLevel and the torch dumps byte for byte.
    ggml_tensor* out_levels[4] = {};
    int cur_h = 0, cur_w = 0, n_patch = 0;
    bool graph_ready = false;

    ~Impl() {
        if (allocr) ggml_gallocr_free(allocr);
        if (gctx) ggml_free(gctx);
        if (ibuf) ggml_backend_buffer_free(ibuf);
        if (ictx) ggml_free(ictx);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
    }
};

RadioGraph::RadioGraph(ggml_backend* backend,
                       const RadioConfig& cfg,
                       const GgufFile* gguf) {
    impl_ = new Impl();
    impl_->backend = backend;
    impl_->gguf = gguf;
    impl_->cfg = cfg;
}

RadioGraph::~RadioGraph() { delete impl_; }

void RadioGraph::prepare_host_tables(int H, int W) {
    const RadioConfig& cfg = impl_->cfg;
    const int P = cfg.patch_size;
    const int gh = H / P, gw = W / P;
    const int C = cfg.embed_dim;
    impl_->cur_h = H;
    impl_->cur_w = W;
    impl_->n_patch = gh * gw;

    // Patch gather table, p-major; per-patch kernel order k = c*(P*P)+ky*P+kx.
    // GGML (3, W, H) stores ne0=3 fastest, i.e. flat = (y*W+x)*3 + c — the
    // same HWC interleaved layout as the host dump (verified by parity).
    const int kdim = 3 * P * P;
    host_pids_.assign(size_t(kdim) * impl_->n_patch, 0);
    for (int pi = 0; pi < gh; ++pi)
        for (int pj = 0; pj < gw; ++pj) {
            const int p = pi * gw + pj;
            for (int c = 0; c < 3; ++c)
                for (int ky = 0; ky < P; ++ky)
                    for (int kx = 0; kx < P; ++kx) {
                        host_pids_[size_t(p) * kdim +
                                   size_t(c * P * P + ky * P + kx)] =
                                ((pi * P + ky) * W + (pj * P + kx)) * 3 + c;
                    }
        }

    // CPE positional embedding (eval path):
    //   pos_embed (1, rows*cols, C) -> CHW (C, rows, cols)
    //   -> bilinear to (max_dim, max_dim) -> crop top-left (gh, gw)
    //   -> flatten to (C, gh*gw) token-major for the graph add.
    const int rows = cfg.pos_rows, cols = cfg.pos_cols;
    const float* pe = impl_->gguf->tensor_f32("radio.pos_embed");
    std::vector<float> pe_chw(size_t(C) * rows * cols);
    for (int h = 0; h < rows; ++h)
        for (int w = 0; w < cols; ++w) {
            const float* src = pe + size_t(h * cols + w) * C;
            for (int c = 0; c < C; ++c)
                pe_chw[(size_t(c) * rows + h) * cols + w] = src[c];
        }
    const int max_dim = std::max(gh, gw);
    std::vector<float> resized(size_t(C) * max_dim * max_dim);
    bilinear_resize_nchw(pe_chw.data(), C, rows, cols, resized.data(), max_dim,
                         max_dim);
    // Graph consumes pos as (token, dim) rows (like the torch (1, N, C) add):
    // token p = y*gw + x is the fastest index, channel the slow one.
    host_pos_.assign(size_t(C) * impl_->n_patch, 0.f);
    for (int y = 0; y < gh; ++y)
        for (int x = 0; x < gw; ++x)
            for (int c = 0; c < C; ++c)
                host_pos_[(size_t(y) * gw + x) * C + c] =
                        resized[(size_t(c) * max_dim + y) * max_dim + x];
}

bool RadioGraph::ensure_graph(int H, int W, std::string* err) {
    const RadioConfig& cfg = impl_->cfg;
    if (H % cfg.patch_size || W % cfg.patch_size) {
        if (err) *err = "aux input must be a multiple of patch_size";
        return false;
    }
    if (impl_->graph_ready && impl_->cur_h == H && impl_->cur_w == W)
        return true;

    static const char* kRequired[] = {"radio.patch_embed.weight",
                                      "radio.pos_embed", "radio.cls_token"};
    for (const char* n : kRequired) {
        if (!impl_->gguf->tensor(n)) {
            if (err) *err = std::string("missing tensor: ") + n;
            return false;
        }
    }
    prepare_host_tables(H, W);
    build_weights();
    build_graphs(H, W);
    return true;
}

bool RadioGraph::ensure_weights(std::string* err) {
    static const char* kRequired[] = {"radio.patch_embed.weight",
                                      "radio.pos_embed", "radio.cls_token"};
    for (const char* n : kRequired) {
        if (!impl_->gguf->tensor(n)) {
            if (err) *err = std::string("missing tensor: ") + n;
            return false;
        }
    }
    if (!impl_->wbuf) build_weights();
    return true;
}

// One block-0 component per single-output graph, fed by the verified postcls
// token stream (T rows of C, token-major). The requested value is the ONLY
// terminal node — immune to gallocr reuse ambiguity.
bool RadioGraph::run_diag(const float* postcls_tokens,
                          const char* point,
                          int H,
                          int W,
                          std::vector<float>* out,
                          std::string* err) {
    if (!ensure_weights(err)) return false;
    const RadioConfig& cfg = impl_->cfg;
    const int64_t C = cfg.embed_dim;
    const int64_t n_patch_diag =
            int64_t(H / cfg.patch_size) * int64_t(W / cfg.patch_size);
    const int64_t T = int64_t(cfg.num_skip()) + n_patch_diag;
    const int64_t hd = C / cfg.num_heads;
    const float attn_scale = 1.f / std::sqrt(float(hd));
    if (!impl_->wbuf) {
        if (err) *err = "weights not built";
        return false;
    }

    struct ggml_init_params ip = {
            ggml_tensor_overhead() * 256 +
                    ggml_graph_overhead_custom(512, false),
            nullptr, true};
    ggml_context* ctx = ggml_init(ip);
    struct Cleanup {
        ggml_context* c;
        ~Cleanup() { ggml_free(c); }
    } cleanup{ctx};

    ggml_tensor* X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, T);
    ggml_set_input(X);

    auto lnorm = [&](ggml_tensor* in, int which) {
        return ggml_add(
                ctx,
                ggml_mul(ctx, ggml_norm(ctx, in, kLayerNormEps),
                         which == 1 ? impl_->w_ln1[0] : impl_->w_ln2[0]),
                which == 1 ? impl_->b_ln1[0] : impl_->b_ln2[0]);
    };
    auto lin = [&](ggml_tensor* in, ggml_tensor* w, ggml_tensor* b) {
        ggml_tensor* o = mul_mat_w(ctx, w, in);
        return b ? ggml_add(ctx, o, b) : o;
    };

    const std::string pt = point;
    ggml_tensor* out_t = nullptr;
    ggml_tensor* n1 = lnorm(X, 1);
    if (pt == "n1") {
        out_t = n1;
    } else {
        ggml_tensor* qkv = lin(n1, impl_->w_qkv[0], impl_->b_qkv[0]);
        if (pt == "qkv") {
            out_t = qkv;
        } else {
            ggml_tensor* q =
                    ggml_view_3d(ctx, qkv, hd, cfg.num_heads, qkv->ne[1],
                                 hd * sizeof(float), 3 * C * sizeof(float), 0);
            ggml_tensor* k = ggml_view_3d(
                    ctx, qkv, hd, cfg.num_heads, qkv->ne[1], hd * sizeof(float),
                    3 * C * sizeof(float), C * sizeof(float));
            ggml_tensor* v = ggml_view_3d(
                    ctx, qkv, hd, cfg.num_heads, qkv->ne[1], hd * sizeof(float),
                    3 * C * sizeof(float), 2 * C * sizeof(float));
            q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
            k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
            v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
            ggml_tensor* o = ggml_flash_attn_ext(ctx, q, k, v, nullptr,
                                                 attn_scale, 0.f, 0.f);
            o = ggml_reshape_2d(ctx, o, C, qkv->ne[1]);
            o = lin(o, impl_->w_proj[0], impl_->b_proj[0]);
            o = impl_->g_ls1[0] ? ggml_mul(ctx, o, impl_->g_ls1[0]) : o;
            if (pt == "attnbr") {
                out_t = o;
            } else {
                ggml_tensor* x = ggml_add(ctx, X, o);
                ggml_tensor* n2 = lnorm(x, 2);
                ggml_tensor* h1 = ggml_gelu_erf(
                        ctx, lin(n2, impl_->w_fc1[0], impl_->b_fc1[0]));
                ggml_tensor* h2 = lin(h1, impl_->w_fc2[0], impl_->b_fc2[0]);
                h2 = impl_->g_ls2[0] ? ggml_mul(ctx, h2, impl_->g_ls2[0]) : h2;
                if (pt == "mlpbr")
                    out_t = h2;
                else if (pt == "x")
                    out_t = ggml_add(ctx, x, h2);
                else {
                    if (err) *err = "unknown diag point";
                    return false;
                }
            }
        }
    }
    ggml_set_output(out_t);
    ggml_cgraph* g = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(g, out_t);
    ggml_gallocr* al = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(impl_->backend));
    if (!ggml_gallocr_alloc_graph(al, g)) {
        if (err) *err = "diag alloc failed";
        ggml_gallocr_free(al);
        return false;
    }
    ggml_backend_tensor_set(X, postcls_tokens, 0, ggml_nbytes(X));
    ggml_backend_graph_compute(impl_->backend, g);
    out->resize(ggml_nbytes(out_t) / sizeof(float));
    ggml_backend_tensor_get(out_t, out->data(), 0, ggml_nbytes(out_t));
    ggml_gallocr_free(al);
    return true;
}

void RadioGraph::build_weights() {
    if (impl_->wbuf) {
        ggml_backend_buffer_free(impl_->wbuf);
        impl_->wbuf = nullptr;
    }
    if (impl_->wctx) ggml_free(impl_->wctx);
    struct ggml_init_params wip = {};
    wip.mem_size = ggml_tensor_overhead() * 1024;
    wip.mem_buffer = nullptr;
    wip.no_alloc = true;
    impl_->wctx = ggml_init(wip);
    impl_->bind.clear();
    auto load_w = [&](const std::string& n) -> ggml_tensor* {
        const TensorInfo* ti = impl_->gguf->tensor(n);
        if (!ti) return nullptr;
        const int64_t ne_i64[4] = {int64_t(ti->ne[0]), int64_t(ti->ne[1]),
                                   int64_t(ti->ne[2]), int64_t(ti->ne[3])};
        ggml_tensor* t = ggml_new_tensor(impl_->wctx, to_ggml_type(ti->dtype),
                                         int(ti->n_dims), ne_i64);
        impl_->bind.emplace_back(t, ti);
        return t;
    };
    auto opt = [&](const std::string& n) {
        return impl_->gguf->tensor(n) ? load_w(n) : nullptr;
    };
    const RadioConfig& cfg = impl_->cfg;
    impl_->w_patch = load_w("radio.patch_embed.weight");
    impl_->w_cls = load_w("radio.cls_token");
    for (int i = 0; i < cfg.depth; ++i) {
        const std::string p = "radio.blocks." + std::to_string(i);
        impl_->w_ln1[i] = load_w(p + ".norm1.weight");
        impl_->b_ln1[i] = load_w(p + ".norm1.bias");
        impl_->w_ln2[i] = load_w(p + ".norm2.weight");
        impl_->b_ln2[i] = load_w(p + ".norm2.bias");
        impl_->w_qkv[i] = load_w(p + ".attn.qkv.weight");
        impl_->b_qkv[i] = opt(p + ".attn.qkv.bias");
        impl_->w_proj[i] = load_w(p + ".attn.proj.weight");
        impl_->b_proj[i] = opt(p + ".attn.proj.bias");
        impl_->g_ls1[i] = opt(p + ".ls1.gamma");
        impl_->g_ls2[i] = opt(p + ".ls2.gamma");
        impl_->w_fc1[i] = load_w(p + ".mlp.fc1.weight");
        impl_->b_fc1[i] = opt(p + ".mlp.fc1.bias");
        impl_->w_fc2[i] = load_w(p + ".mlp.fc2.weight");
        impl_->b_fc2[i] = opt(p + ".mlp.fc2.bias");
    }
    impl_->wbuf = ggml_backend_alloc_ctx_tensors(impl_->wctx, impl_->backend);
    for (auto& [t, ti] : impl_->bind)
        ggml_backend_tensor_set(t, ti->bytes.data(), 0, ggml_nbytes(t));
}

void RadioGraph::build_graphs(int H, int W) {
    const RadioConfig& cfg = impl_->cfg;
    const int P = cfg.patch_size;
    const int64_t C = cfg.embed_dim;
    const int64_t hd = C / cfg.num_heads;
    const float attn_scale = 1.f / std::sqrt(float(hd));
    const int gh = H / P, gw = W / P;

    if (impl_->allocr) {
        ggml_gallocr_free(impl_->allocr);
        impl_->allocr = nullptr;
    }
    if (impl_->gctx) ggml_free(impl_->gctx);
    if (impl_->ibuf) {
        ggml_backend_buffer_free(impl_->ibuf);
        impl_->ibuf = nullptr;
    }
    if (impl_->ictx) ggml_free(impl_->ictx);

    // host inputs + level taps live in their own ctx + buffer (gallocr
    // skips everything that already owns a buffer).
    struct ggml_init_params iip = {ggml_tensor_overhead() * 64, nullptr, true};
    impl_->ictx = ggml_init(iip);
    ggml_context* ictx = impl_->ictx;
    impl_->in_img = ggml_new_tensor_3d(ictx, GGML_TYPE_F32, 3, W, H);
    impl_->in_pids = ggml_new_tensor_1d(ictx, GGML_TYPE_I32,
                                        int64_t(3 * P * P) * impl_->n_patch);
    impl_->in_pos = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, C, impl_->n_patch);
    for (int li = 0; li < 4; ++li)
        impl_->out_levels[li] =
                ggml_new_tensor_3d(ictx, GGML_TYPE_F32, gw, gh, C);  // NCHW
    for (auto* t : {impl_->in_img, impl_->in_pids, impl_->in_pos})
        ggml_set_input(t);
    impl_->ibuf = ggml_backend_alloc_ctx_tensors(ictx, impl_->backend);

    struct ggml_init_params ip = {
            ggml_tensor_overhead() * 4096 +
                    ggml_graph_overhead_custom(4096, false),
            nullptr, true};
    impl_->gctx = ggml_init(ip);
    ggml_context* ctx = impl_->gctx;

    // ---- patch gather (input already conditioned on host), pos, cls ----
    ggml_tensor* gathered = ggml_get_rows(
            ctx, ggml_reshape_2d(ctx, impl_->in_img, 1, 3 * W * H),
            impl_->in_pids);
    ggml_tensor* patches =
            ggml_reshape_2d(ctx, gathered, 3 * P * P, impl_->n_patch);
    ggml_tensor* x = mul_mat_w(ctx, impl_->w_patch, patches);  // (C, n_patch)
    x = ggml_add(ctx, x, impl_->in_pos);
    x = ggml_concat(ctx, impl_->w_cls, x, 1);  // (C, skip+n_patch)

    // ---- all blocks, one chain, no graph boundaries ----
    const size_t skip = size_t(cfg.num_skip());
    ggml_tensor* taps[4];
    for (int bi = 0; bi < cfg.depth; ++bi) {
        x = this->build_block(ctx, bi, x, attn_scale);
        const int li_at = [](int bi, const RadioConfig& c) {
            for (int li = 0; li < 4; ++li)
                if (c.out_indices[li] == bi) return li;
            return -1;
        }(bi, cfg);
        if (li_at < 0) continue;
        // Tap: strip the cls prefix (view over the spatial tokens), split the
        // flat token axis into (w, h), permute token-major -> NCHW, cpy into
        // the ictx leaf (external buffer: survives the whole-graph compute).
        ggml_tensor* spatial =
                ggml_view_2d(ctx, x, C, impl_->n_patch, C * sizeof(float),
                             skip * C * sizeof(float));
        ggml_tensor* chw =
                ggml_reshape_3d(ctx, spatial, C, gw, gh);  // (C, W, H)
        ggml_tensor* nchw =
                ggml_permute(ctx, chw, 2, 0, 1, 3);  // C->2,W->0,H->1
        taps[li_at] = ggml_cpy(ctx, nchw, impl_->out_levels[li_at]);
    }

    impl_->graph = ggml_new_graph_custom(ctx, 4096, false);
    for (int li = 0; li < 4; ++li)
        ggml_build_forward_expand(impl_->graph, taps[li]);
    impl_->allocr = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(impl_->backend));
    ggml_gallocr_alloc_graph(impl_->allocr, impl_->graph);
    impl_->graph_ready = true;
}

ggml_tensor* RadioGraph::build_block(ggml_context* ctx,
                                     int i,
                                     ggml_tensor* x,
                                     float attn_scale) {
    const RadioConfig& cfg = impl_->cfg;
    const int64_t C = cfg.embed_dim;
    const int64_t hd = C / cfg.num_heads;
    const std::string p = "radio.blocks." + std::to_string(i);

    auto lnorm = [&](ggml_tensor* in, ggml_tensor* w, ggml_tensor* b) {
        ggml_tensor* nn = ggml_norm(ctx, in, kLayerNormEps);
        return ggml_add(ctx, ggml_mul(ctx, nn, w), b);
    };
    auto lin = [&](ggml_tensor* in, ggml_tensor* w, ggml_tensor* b) {
        ggml_tensor* o = mul_mat_w(ctx, w, in);
        if (b) o = ggml_add(ctx, o, b);
        return o;
    };
    auto ls = [&](ggml_tensor* in, ggml_tensor* g) {
        return g ? ggml_mul(ctx, in, g) : in;
    };

    ggml_tensor* n1 = lnorm(x, impl_->w_ln1[i], impl_->b_ln1[i]);
    ggml_tensor* qkv = lin(n1, impl_->w_qkv[i], impl_->b_qkv[i]);

    ggml_tensor* q = ggml_view_3d(ctx, qkv, hd, cfg.num_heads, qkv->ne[1],
                                  hd * sizeof(float), 3 * C * sizeof(float), 0);
    ggml_tensor* k = ggml_view_3d(ctx, qkv, hd, cfg.num_heads, qkv->ne[1],
                                  hd * sizeof(float), 3 * C * sizeof(float),
                                  C * sizeof(float));
    ggml_tensor* v = ggml_view_3d(ctx, qkv, hd, cfg.num_heads, qkv->ne[1],
                                  hd * sizeof(float), 3 * C * sizeof(float),
                                  2 * C * sizeof(float));
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    ggml_tensor* o =
            ggml_flash_attn_ext(ctx, q, k, v, nullptr, attn_scale, 0.f, 0.f);
    o = ggml_reshape_2d(ctx, o, C, qkv->ne[1]);  // head-major, c = h*hd+d
    o = lin(o, impl_->w_proj[i], impl_->b_proj[i]);
    o = ls(o, impl_->g_ls1[i]);
    x = ggml_add(ctx, x, o);

    ggml_tensor* n2 = lnorm(x, impl_->w_ln2[i], impl_->b_ln2[i]);
    ggml_tensor* h1 =
            ggml_gelu_erf(ctx, lin(n2, impl_->w_fc1[i], impl_->b_fc1[i]));
    ggml_tensor* h2 = lin(h1, impl_->w_fc2[i], impl_->b_fc2[i]);
    h2 = ls(h2, impl_->g_ls2[i]);
    return ggml_add(ctx, x, h2);
}

bool RadioGraph::run(const float* img_hwc,
                     int H,
                     int W,
                     std::vector<FeatureLevel>* out_levels,
                     std::string* err,
                     bool keep_on_device) {
    if (!ensure_graph(H, W, err)) return false;

    // InputConditioner as the official model actually runs it: load_model()
    // keeps norm_mean/norm_std in fp32, but builder.py then calls
    // vision_tower_aux.to(dtype=bf16), which casts the buffers too. x is bf16,
    // so y = (x - mean) / std is two bf16 ops, each rounded (torch CPU bf16
    // binary ops compute in f32 and round the result). Done on the host: the
    // bf16 rounding has no ggml op. Buffers come from the checkpoint (bf16
    // values) when the GGUF carries them.
    const float* mean = impl_->gguf->tensor("radio.input_conditioner.norm_mean")
                                ? impl_->gguf->tensor_f32(
                                          "radio.input_conditioner.norm_mean")
                                : impl_->cfg.norm_mean;
    const float* stdv = impl_->gguf->tensor("radio.input_conditioner.norm_std")
                                ? impl_->gguf->tensor_f32(
                                          "radio.input_conditioner.norm_std")
                                : impl_->cfg.norm_std;
    const size_t npx = size_t(H) * W;
    impl_->host_img.resize(npx * 3);
    for (size_t i = 0; i < npx; ++i)
        for (int c = 0; c < 3; ++c)
            impl_->host_img[i * 3 + c] = bf16_round(
                    bf16_round(img_hwc[i * 3 + c] - mean[c]) / stdv[c]);
    ggml_backend_tensor_set(impl_->in_img, impl_->host_img.data(), 0,
                            ggml_nbytes(impl_->in_img));
    ggml_backend_tensor_set(impl_->in_pids, host_pids_.data(), 0,
                            ggml_nbytes(impl_->in_pids));
    ggml_backend_tensor_set(impl_->in_pos, host_pos_.data(), 0,
                            ggml_nbytes(impl_->in_pos));

    ggml_backend_graph_compute(impl_->backend, impl_->graph);

    out_levels->clear();
    for (int li = 0; li < 4; ++li) {
        ggml_tensor* t = impl_->out_levels[li];  // NCHW leaf, (gw, gh, C)
        FeatureLevel lev;
        lev.channels = int(t->ne[2]);
        lev.height = int(t->ne[1]);
        lev.width = int(t->ne[0]);
        if (!keep_on_device) {
            lev.data.resize(ggml_nbytes(t) / sizeof(float));
            ggml_backend_tensor_get(t, lev.data.data(), 0, ggml_nbytes(t));
        }
        out_levels->push_back(std::move(lev));
    }
    return true;
}

ggml_tensor* RadioGraph::out_level_gpu(int li) const {
    if (li < 0 || li >= 4) return nullptr;
    return impl_->out_levels[li];
}

}  // namespace vlx
