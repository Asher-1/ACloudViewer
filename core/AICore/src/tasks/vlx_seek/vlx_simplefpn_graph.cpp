// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#include "vlx_simplefpn_graph.h"

#include <cstdio>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "vlx_ggml_util.h"
#include "vlx_gguf_loader.h"

namespace vlx {

namespace {

// Per-level structure read straight off the 10B checkpoint keys:
//   simfp_2 (stride 4) : ConvT x3 + 1x1(512)   + 3x3
//   simfp_3 (stride 8) : ConvT x2 + 1x1(1024)  + 3x3
//   simfp_4 (stride 16): ConvT x1 + 1x1(2048)  + 3x3
//   simfp_5 (stride 32):           + 1x1(4096) + 3x3
// Sequential indices: ConvT at 3i, LN(3i+1)+GELU(3i+2) only BETWEEN ConvTs,
// trailing 1x1/3x3 pair at (n_convt ? 3*n-2 : 0) and +1.
struct LevelSpec {
    int stage;
    int n_convt;
    int in_dim;
};
constexpr LevelSpec kSpecs[4] = {
        {2, 3, 4096}, {3, 2, 4096}, {4, 1, 4096}, {5, 0, 4096}};

}  // namespace

struct SimpleFpnGraph::Impl {
    ggml_backend* backend = nullptr;
    const GgufFile* gguf = nullptr;
    SimpleFpnConfig cfg;

    // weights (radio/mmproj pattern: create-only ctx, one buffer, then upload)
    ggml_context* wctx = nullptr;
    ggml_backend_buffer* wbuf = nullptr;
    bool weights_ready = false;

    // per-resolution graph state (ONE graph — no cross-graph references)
    ggml_context* gctx = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_gallocr* allocr = nullptr;

    ggml_context* ictx = nullptr;
    ggml_backend_buffer* ibuf = nullptr;
    ggml_tensor* in_pooler = nullptr;  // (vt_dim, W2, H2) token-major
    ggml_tensor* out_levels[4] = {};   // (512, W_s, H_s) ictx leaves
    int cur_h2 = 0, cur_w2 = 0;
    bool graph_ready = false;

    // weight tensors
    ggml_tensor* w_convt[4][3] = {};  // merged (4*out, in) GEMM form
    ggml_tensor* b_convt[4][3] = {};
    ggml_tensor* w_ln1[4][3] = {};  // standalone LN after ConvT i<n-1
    ggml_tensor* b_ln1[4][3] = {};
    ggml_tensor* w_1x1[4] = {};
    ggml_tensor* w_3x3[4] = {};
    ggml_tensor* w_lnc[4] = {};  // LN after 1x1
    ggml_tensor* b_lnc[4] = {};
    ggml_tensor* w_lnf[4] = {};  // LN after 3x3
    ggml_tensor* b_lnf[4] = {};

    ~Impl() {
        if (allocr) ggml_gallocr_free(allocr);
        if (gctx) ggml_free(gctx);
        if (ibuf) ggml_backend_buffer_free(ibuf);
        if (ictx) ggml_free(ictx);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
    }
};

SimpleFpnGraph::SimpleFpnGraph(ggml_backend* backend,
                               const SimpleFpnConfig& cfg,
                               const GgufFile* gguf)
    : impl_(new Impl) {
    impl_->backend = backend;
    impl_->gguf = gguf;
    impl_->cfg = cfg;
}

SimpleFpnGraph::~SimpleFpnGraph() { delete impl_; }

void SimpleFpnGraph::build_weights() {
    Impl* impl = impl_;
    struct ggml_init_params ip = {size_t(256) << 20, nullptr, true};
    impl->wctx = ggml_init(ip);

    std::vector<std::pair<ggml_tensor*, const TensorInfo*>> bind;
    auto load_w = [&](const char* n) -> ggml_tensor* {
        const TensorInfo* ti = impl->gguf->tensor(n);
        // load path above enumerated the GGUF; a miss is an invariant breach.
        if (!ti) GGML_ABORT("fpn: missing weight %s", n);
        const int64_t ne[4] = {int64_t(ti->ne[0]), int64_t(ti->ne[1]),
                               int64_t(ti->ne[2]), int64_t(ti->ne[3])};
        ggml_tensor* t = ggml_new_tensor(impl->wctx, to_ggml_type(ti->dtype),
                                         ti->n_dims, ne);
        bind.emplace_back(t, ti);
        return t;
    };
    char p[128];
    for (int li = 0; li < 4; ++li) {
        const LevelSpec& sp = kSpecs[li];
        const int n = sp.n_convt;
        for (int i = 0; i < n; ++i) {
            std::snprintf(p, sizeof(p), "fpn.simfp_%d.%d.weight_t", sp.stage,
                          3 * i);
            impl->w_convt[li][i] = load_w(p);
            std::snprintf(p, sizeof(p), "fpn.simfp_%d.%d.bias", sp.stage,
                          3 * i);
            impl->b_convt[li][i] = load_w(p);
            if (i < n - 1) {
                std::snprintf(p, sizeof(p), "fpn.simfp_%d.%d.weight", sp.stage,
                              3 * i + 1);
                impl->w_ln1[li][i] = load_w(p);
                std::snprintf(p, sizeof(p), "fpn.simfp_%d.%d.bias", sp.stage,
                              3 * i + 1);
                impl->b_ln1[li][i] = load_w(p);
            }
        }
        const int j = n ? 3 * n - 2 : 0;
        std::snprintf(p, sizeof(p), "fpn.simfp_%d.%d.weight", sp.stage, j);
        impl->w_1x1[li] = load_w(p);
        std::snprintf(p, sizeof(p), "fpn.simfp_%d.%d.norm.weight", sp.stage, j);
        impl->w_lnc[li] = load_w(p);
        std::snprintf(p, sizeof(p), "fpn.simfp_%d.%d.norm.bias", sp.stage, j);
        impl->b_lnc[li] = load_w(p);
        std::snprintf(p, sizeof(p), "fpn.simfp_%d.%d.weight", sp.stage, j + 1);
        impl->w_3x3[li] = load_w(p);
        std::snprintf(p, sizeof(p), "fpn.simfp_%d.%d.norm.weight", sp.stage,
                      j + 1);
        impl->w_lnf[li] = load_w(p);
        std::snprintf(p, sizeof(p), "fpn.simfp_%d.%d.norm.bias", sp.stage,
                      j + 1);
        impl->b_lnf[li] = load_w(p);
    }
    impl->wbuf = ggml_backend_alloc_ctx_tensors(impl->wctx, impl->backend);
    for (auto& [t, ti] : bind)
        ggml_backend_tensor_set(t, ti->bytes.data(), 0, ti->bytes.size());
    impl->weights_ready = true;
}

void SimpleFpnGraph::build_graph(int grid_h, int grid_w) {
    Impl* impl = impl_;
    const int C = impl->cfg.out_channels;
    const int H2 = grid_h / 2, W2 = grid_w / 2;

    struct ggml_init_params ip = {size_t(256) << 20, nullptr, true};
    impl->gctx = ggml_init(ip);
    ggml_context* ctx = impl->gctx;

    // inputs + level taps live in ictx (dedicated buffer; gallocr never
    // re-points external-buffer tensors — the cross-graph expand lesson).
    struct ggml_init_params iip = {size_t(64) << 20, nullptr, true};
    impl->ictx = ggml_init(iip);
    impl->in_pooler = ggml_new_tensor_3d(impl->ictx, GGML_TYPE_F32,
                                         impl->cfg.vt_dim, W2, H2);
    for (int li = 0; li < 4; ++li) {
        const int s = impl->cfg.strides[3] /
                      impl->cfg.strides[li];  // upsample factor
        // NCHW leaf: ne=(W, H, C) — c SLOWEST, matching FeatureLevel's
        // data[(ch*H + y)*W + x] contract and torch's NCHW dump bytes.
        impl->out_levels[li] = ggml_new_tensor_3d(impl->ictx, GGML_TYPE_F32,
                                                  W2 * s, H2 * s, C);
    }
    impl->ibuf = ggml_backend_alloc_ctx_tensors(impl->ictx, impl->backend);

    auto lnorm = [&](ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
        // token-major (C, W, H): reshape (C, W*H) normalizes over C exactly;
        // return reshaped back to 3D so every graph tensor keeps the (C,W,H)
        // contract.
        ggml_tensor* r = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2]);
        ggml_tensor* y =
                ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, r, 1e-6f), w), b);
        return ggml_reshape_3d(ctx, y, x->ne[0], x->ne[1], x->ne[2]);
    };

    // ConvT k2s2 as merged 4-slice GEMM + pixel shuffle. Input/output are
    // token-major (C, W, H); the mul_mat runs on the flattened (C, H*W).
    auto convt = [&](ggml_tensor* x, ggml_tensor* w_all, ggml_tensor* bias) {
        const int64_t C_in = x->ne[0];
        const int64_t W = x->ne[1], H = x->ne[2];
        const int64_t C_out = w_all->ne[1] / 4;  // w_all ne=(C_in, 4*C_out)
        ggml_tensor* r =
                mul_mat_w(ctx, w_all, ggml_reshape_2d(ctx, x, C_in, W * H));
        // slice k holds rows [k*C_out, (k+1)*C_out) of r, k = 2a+b. The view
        // MUST keep the graph-wide token-major axis order (C, W, H) — i1 = w,
        // i2 = h — otherwise the downstream reshape re-interprets the cat0
        // rows with h/w swapped (silently equivalent to the k=2b+a mapping).
        auto slice = [&](int k) {
            return ggml_view_3d(ctx, r, C_out, W, H,
                                /*nb1(w)=*/4 * C_out * sizeof(float),
                                /*nb2(h)=*/4 * C_out * W * sizeof(float),
                                /*offset=*/k * C_out * sizeof(float));
        };
        // b into W: cat0 (k=2a, k=2a+1) -> (2*C_out, W, H) contiguous, whose
        // linear map already equals the (C_out, 2W, H) token-major interleave.
        ggml_tensor* g0 = ggml_reshape_3d(
                ctx, ggml_concat(ctx, slice(0), slice(1), 0), C_out, 2 * W, H);
        ggml_tensor* g1 = ggml_reshape_3d(
                ctx, ggml_concat(ctx, slice(2), slice(3), 0), C_out, 2 * W, H);
        // a into H: cat0 -> (2*C_out, 2W, H) whose rows are (a, c) pairs;
        // reinterpret as (c, x, a, h) with the a-axis stride = C_out (NOT the
        // cat row stride), cont, reshape (C, 2W, 2H) = y=2h+a interleave.
        ggml_tensor* cat = ggml_concat(ctx, g0, g1, 0);
        ggml_tensor* v4 = ggml_view_4d(ctx, cat, C_out, 2 * W, 2, H, cat->nb[1],
                                       C_out * sizeof(float), cat->nb[2], 0);
        ggml_tensor* pv = ggml_cont(ctx, v4);
        ggml_tensor* sh = ggml_reshape_3d(ctx, pv, C_out, 2 * W, 2 * H);
        return ggml_add(ctx, sh, bias);
    };

    auto conv1x1 = [&](ggml_tensor* x, ggml_tensor* w) {
        const int64_t C_in = x->ne[0];
        ggml_tensor* r = mul_mat_w(
                ctx, w, ggml_reshape_2d(ctx, x, C_in, x->ne[1] * x->ne[2]));
        return ggml_reshape_3d(ctx, r, w->ne[1], x->ne[1], x->ne[2]);
    };

    auto conv3x3 = [&](ggml_tensor* x, ggml_tensor* w2d) {
        // token-major (C, W, H) -> NCHW ne=(W, H, C) for im2col. ggml_permute
        // maps OLD axis i to NEW axis p_i: C->2, W->0, H->1 => (2, 0, 1).
        ggml_tensor* xn = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));
        const int64_t IC = x->ne[0];
        // im2col only reads the kernel operand's shape (KW, KH, IC); a tiny F32
        // shape leaf stands in so w2d may be f16/q8_0 (q8_0 rows of IC*9 cannot
        // be reshaped to KW=3).
        GGML_ASSERT(w2d->ne[0] == IC * 9);
        ggml_tensor* w4 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, IC, 1);
        ggml_tensor* patches =
                ggml_im2col(ctx, w4, xn, 1, 1, 1, 1, 1, 1, true, GGML_TYPE_F32);
        // (OC, OW, OH) — token-major over the output grid, same layout.
        return mul_mat_w(ctx, w2d, patches);
    };

    ggml_tensor* taps[4];
    for (int li = 0; li < 4; ++li) {
        const LevelSpec& sp = kSpecs[li];
        const int n = sp.n_convt;
        // Each stage is an INDEPENDENT branch over the pooled feature map
        // (SimpleFP.forward runs every stage on the same input) — NOT a chain.
        ggml_tensor* x = impl->in_pooler;
        for (int i = 0; i < n; ++i) {
            x = convt(x, impl->w_convt[li][i], impl->b_convt[li][i]);
            if (i < n - 1) {
                x = lnorm(x, impl->w_ln1[li][i], impl->b_ln1[li][i]);
                x = ggml_gelu_erf(ctx, x);  // nn.GELU() = exact erf
            }
        }
        const int j = n ? 3 * n - 2 : 0;
        x = conv1x1(x, impl->w_1x1[li]);
        x = lnorm(x, impl->w_lnc[li], impl->b_lnc[li]);
        x = conv3x3(x, impl->w_3x3[li]);
        x = lnorm(x, impl->w_lnf[li], impl->b_lnf[li]);
        // Level tap: the graph chain is token-major (C, W, H), c FASTEST;
        // FeatureLevel/torch want NCHW (c slowest). Transpose in-graph
        // (old C->new 2, W->0, H->1) and cpy into the NCHW-shaped leaf.
        ggml_tensor* nchw = ggml_permute(ctx, x, 2, 0, 1, 3);
        x = ggml_cpy(ctx, nchw, impl->out_levels[li]);
        taps[li] = x;
    }

    impl->graph = ggml_new_graph_custom(ctx, 2048, false);
    for (int li = 0; li < 4; ++li)
        ggml_build_forward_expand(impl->graph, taps[li]);
    impl->allocr = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(impl->backend));
    ggml_gallocr_alloc_graph(impl->allocr, impl->graph);
    impl->graph_ready = true;
}

bool SimpleFpnGraph::run(const float* pooler,
                         int grid_h,
                         int grid_w,
                         std::vector<FeatureLevel>* out_levels,
                         std::string* err,
                         bool keep_on_device) {
    Impl* impl = impl_;
    if (grid_h % 2 || grid_w % 2) {
        if (err) *err = "grid must be even (merge_size 2)";
        return false;
    }
    if (!impl->weights_ready) build_weights();
    const int H2 = grid_h / 2, W2 = grid_w / 2;
    if (!impl->graph_ready || H2 != impl->cur_h2 || W2 != impl->cur_w2) {
        build_graph(grid_h, grid_w);
        impl->cur_h2 = H2;
        impl->cur_w2 = W2;
    }

    ggml_backend_tensor_set(impl->in_pooler, pooler, 0,
                            ggml_nbytes(impl->in_pooler));
    ggml_backend_graph_compute(impl->backend, impl->graph);

    out_levels->clear();
    for (int li = 0; li < 4; ++li) {
        ggml_tensor* t = impl->out_levels[li];
        FeatureLevel lv;
        lv.width = int(t->ne[0]);
        lv.height = int(t->ne[1]);
        lv.channels = int(t->ne[2]);  // NCHW leaf: ne=(W, H, C)
        if (!keep_on_device) {        // host path: read back; GPU-resident
            lv.data.resize(ggml_nbytes(t) / sizeof(float));  // path fills the
            ggml_backend_tensor_get(t, lv.data.data(), 0,
                                    ggml_nbytes(t));  // data lazily
        }  // via out_level_gpu() + tensor_get
        out_levels->push_back(std::move(lv));
    }
    return true;
}

ggml_tensor* SimpleFpnGraph::out_level_gpu(int li) const {
    if (li < 0 || li >= 4) return nullptr;
    return impl_->out_levels[li];
}

}  // namespace vlx
