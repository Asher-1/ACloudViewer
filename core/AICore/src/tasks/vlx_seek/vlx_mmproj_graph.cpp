// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#include "vlx_mmproj_graph.h"
#include "vlx_gguf_loader.h"
#include "vlx_ggml_util.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace vlx {

namespace {

// Decode a flat patch index in spatial-merge-block order into (row, col) grid
// coordinates — the inverse of the block-major token layout used by the
// Qwen2VL-family processor (block_row, block_col, in_row, in_col).
void decode_block_major(int n, int grid_h, int grid_w, int merge, int* row, int* col) {
    const int blocks_w = grid_w / merge;
    const int within = n % (grid_h * grid_w);
    const int in_col = within % merge;
    const int in_row = (within / merge) % merge;
    const int block_col = (within / (merge * merge)) % blocks_w;
    const int block_row = within / (merge * merge * blocks_w);
    *row = block_row * merge + in_row;
    *col = block_col * merge + in_col;
}

// Bilinear taps with align_corners=True on one axis (endpoints map to 0 and
// side-1 exactly): src = idx*(side-1)/(size-1), border-clamped 2 taps.
void axis_taps(int idx, int size, int side, int* t0, int* t1, float* w1) {
    const float src = float(idx) * float(side - 1) / float(size - 1);
    int f = int(std::floor(src));
    f = f < 0 ? 0 : (f > side - 2 ? side - 2 : f);
    *t0 = f;
    *t1 = f + 1;
    *w1 = src - float(f);  // weight of the second tap; first gets 1-w
}

}  // namespace

struct MmprojGraph::Impl {
    ggml_backend* backend = nullptr;
    const GgufFile* gguf = nullptr;
    MmprojConfig cfg;

    // ONE graph for the whole tower (seg0 + blocks + merger). The former
    // per-block segment graphs were killed by the cross-graph expand trap:
    // a segment input (prev segment's out_x) carries src pointers into the
    // previous graph's context, so build_forward_expand pulls the ENTIRE
    // preceding chain into every segment, and each segment's gallocr then
    // re-points the SHARED tensor objects' data into its own pool — later
    // allocators win, so every segment computed against an unwritten pool
    // address (garbage input), with O(depth^2) work as a side effect.
    // The diag pattern proved the single-graph chain bit-exact; one graph
    // collapses all cross-boundary assumptions to zero.
    ggml_context* gctx = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_gallocr* allocr = nullptr;

    // weights (built once; referenced by all segments)
    ggml_context* wctx = nullptr;
    ggml_backend_buffer* wbuf = nullptr;
    ggml_tensor* w_patch = nullptr;
    ggml_tensor* b_patch = nullptr;
    ggml_tensor* w_pos = nullptr;      // (C, pos_side^2) raw table
    ggml_tensor* in_pos = nullptr;     // (C, N) interpolated, per resolution
    ggml_tensor* in_cossin = nullptr;  // (144, N): [cos72 | sin72] token-major
    ggml_tensor* rope_ones = nullptr;  // per-block cossin materialization input
    static constexpr int kMaxBlocks = 27;
    ggml_tensor *w_ln1[kMaxBlocks] = {}, *b_ln1[kMaxBlocks] = {};
    ggml_tensor *w_ln2[kMaxBlocks] = {}, *b_ln2[kMaxBlocks] = {};
    ggml_tensor *w_qkv[kMaxBlocks] = {}, *b_qkv[kMaxBlocks] = {};
    ggml_tensor *w_proj[kMaxBlocks] = {}, *b_proj[kMaxBlocks] = {};
    ggml_tensor *w_fc1[kMaxBlocks] = {}, *b_fc1[kMaxBlocks] = {};
    ggml_tensor *w_fc2[kMaxBlocks] = {}, *b_fc2[kMaxBlocks] = {};
    ggml_tensor *w_mnorm = nullptr, *b_mnorm = nullptr;
    ggml_tensor *w_mfc1 = nullptr, *b_mfc1 = nullptr;
    ggml_tensor *w_mfc2 = nullptr, *b_mfc2 = nullptr;

    // per-resolution compute state
    ggml_context* ictx = nullptr;
    ggml_backend_buffer* ibuf = nullptr;
    ggml_tensor* in_patches = nullptr;  // (K, N) input, ictx leaf
    ggml_tensor* out_final = nullptr;   // (C, N) block26 tap, ictx leaf + buffer
                                        // (read back through ggml_cpy — the
                                        // address must survive gallocr reuse)
    ggml_tensor* out_pooler = nullptr;  // (hidden_out, N/4) graph terminal
    int cur_n = 0, cur_gh = 0, cur_gw = 0;
    bool weights_ready = false, graphs_ready = false;

    ~Impl() {
        if (allocr) ggml_gallocr_free(allocr);
        if (gctx) ggml_free(gctx);
        if (ibuf) ggml_backend_buffer_free(ibuf);
        if (ictx) ggml_free(ictx);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
    }
};

MmprojGraph::MmprojGraph(ggml_backend* backend, const MmprojConfig& cfg, const GgufFile* gguf)
    : impl_(new Impl) {
    impl_->backend = backend;
    impl_->gguf = gguf;
    impl_->cfg = cfg;
}

MmprojGraph::~MmprojGraph() { delete impl_; }

// ---- weights ----

void MmprojGraph::build_weights() {
    const MmprojConfig& cfg = impl_->cfg;
    struct ggml_init_params ip = {size_t(64) << 20, nullptr, true};
    impl_->wctx = ggml_init(ip);

    // Create-only here: the backend buffer is allocated once after ALL weight
    // tensors exist (no_alloc ctx), then every tensor is uploaded (radio
    // pattern — an early tensor_set hits "buffer not set").
    std::vector<std::pair<ggml_tensor*, const TensorInfo*>> bind;
    auto load_w = [&](const char* n) -> ggml_tensor* {
        const TensorInfo* ti = impl_->gguf->tensor(n);
        if (!ti) return nullptr;
        const int64_t ne_i64[4] = {int64_t(ti->ne[0]), int64_t(ti->ne[1]),
                                   int64_t(ti->ne[2]), int64_t(ti->ne[3])};
        ggml_tensor* t = ggml_new_tensor(impl_->wctx, to_ggml_type(ti->dtype),
                                         int(ti->n_dims), ne_i64);
        bind.emplace_back(t, ti);
        return t;
    };
    const int D = cfg.depth;
    impl_->w_patch = load_w("mmproj.patch_embed.weight");
    impl_->b_patch = load_w("mmproj.patch_embed.bias");
    impl_->w_pos = load_w("mmproj.pos_embed");
    for (int i = 0; i < D; ++i) {
        char p[128];
        auto nm = [&](const char* suffix) {
            std::snprintf(p, sizeof(p), "mmproj.blocks.%d.%s", i, suffix);
            return p;
        };
        impl_->w_ln1[i] = load_w(nm("norm1.weight"));
        impl_->b_ln1[i] = load_w(nm("norm1.bias"));
        impl_->w_ln2[i] = load_w(nm("norm2.weight"));
        impl_->b_ln2[i] = load_w(nm("norm2.bias"));
        impl_->w_qkv[i] = load_w(nm("attn.qkv.weight"));
        impl_->b_qkv[i] = load_w(nm("attn.qkv.bias"));
        impl_->w_proj[i] = load_w(nm("attn.proj.weight"));
        impl_->b_proj[i] = load_w(nm("attn.proj.bias"));
        impl_->w_fc1[i] = load_w(nm("mlp.fc1.weight"));
        impl_->b_fc1[i] = load_w(nm("mlp.fc1.bias"));
        impl_->w_fc2[i] = load_w(nm("mlp.fc2.weight"));
        impl_->b_fc2[i] = load_w(nm("mlp.fc2.bias"));
    }
    impl_->w_mnorm = load_w("mmproj.merger.norm.weight");
    impl_->b_mnorm = load_w("mmproj.merger.norm.bias");
    impl_->w_mfc1 = load_w("mmproj.merger.fc1.weight");
    impl_->b_mfc1 = load_w("mmproj.merger.fc1.bias");
    impl_->w_mfc2 = load_w("mmproj.merger.fc2.weight");
    impl_->b_mfc2 = load_w("mmproj.merger.fc2.bias");
    impl_->wbuf = ggml_backend_alloc_ctx_tensors(impl_->wctx, impl_->backend);
    for (auto& [t, ti] : bind)
        ggml_backend_tensor_set(t, ti->bytes.data(), 0, ti->bytes.size());
    impl_->weights_ready = true;
}

// ---- segment graph construction ----
// Axial-2D rope as an explicit complex rotation (avoids neg/concat on
// strided views): with c[d] = cos[d] = cos[d+36], s[d] = sin[d] = sin[d+36],
//   out[d]    = q[d]*c[d] - q[d+36]*s[d]      (d < half)
//   out[d+half] = q[d+half]*c[d] + q[d]*s[d]
// All inputs are strided-view safe (ggml mul handles strided srcs); the
// caller passes MATERIALIZED (contiguous) cos/sin tensors of (hd, 1, N).
static ggml_tensor* rope_axial(ggml_context* ctx, ggml_tensor* x,
                               ggml_tensor* cos72, ggml_tensor* sin72) {
    const int64_t hd = x->ne[0], nh = x->ne[1], N = x->ne[2];
    const int64_t half = hd / 2;
    // cos72/sin72 already tile per axis ([ch,cw,ch,cw]): broadcast the FULL
    // (hd,1,N) rows to (hd,nh,N) — repeat requires equal-or-1 dims, so a
    // half-sized source cannot be broadcast directly.
    ggml_tensor* cb = ggml_repeat(ctx, cos72, x);   // (hd, nh, N)
    ggml_tensor* sb = ggml_repeat(ctx, sin72, x);
    // first-half views of the broadcast tensors (strided views are fine for mul)
    ggml_tensor* c1 = ggml_view_3d(ctx, cb, half, nh, N,
                                   hd * sizeof(float), hd * nh * sizeof(float), 0);
    ggml_tensor* s1 = ggml_view_3d(ctx, sb, half, nh, N,
                                   hd * sizeof(float), hd * nh * sizeof(float), 0);
    ggml_tensor* q1 = ggml_view_3d(ctx, x, half, nh, N,
                                   hd * sizeof(float), hd * nh * sizeof(float), 0);
    ggml_tensor* q2 = ggml_view_3d(ctx, x, half, nh, N,
                                   hd * sizeof(float), hd * nh * sizeof(float),
                                   half * sizeof(float));
    ggml_tensor* real = ggml_sub(ctx, ggml_mul(ctx, q1, c1), ggml_mul(ctx, q2, s1));
    ggml_tensor* imag = ggml_add(ctx, ggml_mul(ctx, q2, c1), ggml_mul(ctx, q1, s1));
    return ggml_concat(ctx, real, imag, 0);
}

void MmprojGraph::build_graphs(int gh, int gw) {
    Impl* impl = impl_;
    const MmprojConfig& cfg = impl->cfg;
    struct ggml_init_params ip = {size_t(256) << 20, nullptr, true};
    impl->gctx = ggml_init(ip);
    ggml_context* ctx = impl->gctx;
    const int C = cfg.embed_dim, hd = cfg.head_dim(), nh = cfg.num_heads;
    const int64_t N = int64_t(gh) * gw;

    // ---- seg0: patch GEMM + bias + interpolated pos embed ----
    ggml_tensor* x;
    {
        ggml_tensor* xp = mul_mat_w(ctx, impl->w_patch, impl->in_patches);
        x = ggml_add(ctx, ggml_add(ctx, xp, impl->b_patch), impl->in_pos);
    }

    // ---- blocks 1..D (one chain, no graph boundaries) ----
    for (int i = 0; i < cfg.depth; ++i) {
        auto lnorm = [&](ggml_tensor* in, ggml_tensor* w, ggml_tensor* b) {
            return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, in, cfg.ln_eps), w), b);
        };
        auto lin = [&](ggml_tensor* in, ggml_tensor* w, ggml_tensor* b) {
            ggml_tensor* o = mul_mat_w(ctx, w, in);
            return b ? ggml_add(ctx, o, b) : o;
        };

        // ---- attention branch ----
        ggml_tensor* n1 = lnorm(x, impl->w_ln1[i], impl->b_ln1[i]);
        ggml_tensor* qkv = lin(n1, impl->w_qkv[i], impl->b_qkv[i]);  // (3C, N)
        // q/k/v as (hd, nh, N) views over the per-token 3C block (radio
        // pattern, parity-verified): token n's q/k/v start at n*3C + {0,C,2C}.
        ggml_tensor* q = ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, nh, N,
                                      hd * sizeof(float), 3 * C * sizeof(float), 0));
        ggml_tensor* k = ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, nh, N,
                                      hd * sizeof(float), 3 * C * sizeof(float),
                                      C * sizeof(float)));
        ggml_tensor* v = ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, nh, N,
                                      hd * sizeof(float), 3 * C * sizeof(float),
                                      2 * C * sizeof(float)));

        // cos/sin views of in_cossin (144, N): per-token row = [cos72 | sin72].
        // The strided views are materialized per block via an all-ones mul
        // (reshape_3d demands contiguity); rope_ones is a constant ictx leaf
        // uploaded once in prepare_host_tables — its address is outside every
        // gallocr pool, so all blocks share it safely.
        ggml_tensor* cosv = ggml_view_2d(ctx, impl->in_cossin, hd, N,
                                         2 * hd * sizeof(float), 0);
        ggml_tensor* sinv = ggml_view_2d(ctx, impl->in_cossin, hd, N,
                                         2 * hd * sizeof(float), hd * sizeof(float));
        ggml_tensor* cos_m = ggml_mul(ctx, cosv, impl->rope_ones);
        ggml_tensor* sin_m = ggml_mul(ctx, sinv, impl->rope_ones);
        ggml_tensor* cosr = ggml_reshape_3d(ctx, cos_m, hd, 1, N);
        ggml_tensor* sinr = ggml_reshape_3d(ctx, sin_m, hd, 1, N);

        ggml_tensor* q_rot = rope_axial(ctx, q, cosr, sinr);
        ggml_tensor* k_rot = rope_axial(ctx, k, cosr, sinr);

        // FA layout (hd, N, nh): cont after permute (radio final form keeps
        // F32 k/v — the F16 cast measurably diverges vs eager f32 attention).
        ggml_tensor* qf = ggml_cont(ctx, ggml_permute(ctx, q_rot, 0, 2, 1, 3));
        ggml_tensor* kf = ggml_cont(ctx, ggml_permute(ctx, k_rot, 0, 2, 1, 3));
        ggml_tensor* vf = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        const float scale = 1.f / std::sqrt(float(hd));
        ggml_tensor* o = ggml_flash_attn_ext(ctx, qf, kf, vf, nullptr, scale, 0.f, 0.f);
        // FA result ne = (v->ne[0], T, nh) with an internal permute: its memory
        // is already token-major (T, nh*hd), c = h*hd + d — a plain reshape.
        o = ggml_reshape_2d(ctx, o, C, N);
        ggml_tensor* attnbr = lin(o, impl->w_proj[i], impl->b_proj[i]);
        x = ggml_add(ctx, x, attnbr);

        // ---- MLP branch (hidden_act = gelu_pytorch_tanh -> ggml_gelu) ----
        ggml_tensor* n2 = lnorm(x, impl->w_ln2[i], impl->b_ln2[i]);
        ggml_tensor* h = ggml_gelu(ctx, lin(n2, impl->w_fc1[i], impl->b_fc1[i]));
        ggml_tensor* mlpbr = lin(h, impl->w_fc2[i], impl->b_fc2[i]);
        x = ggml_add(ctx, x, mlpbr);
    }

    // ---- final tap + merger, same graph ----
    // The HFRE tap must be read back AFTER the whole graph runs, so its
    // address cannot live in the gallocr pool: ggml_cpy into the ictx leaf
    // (llama.cpp dump-tensor pattern; the copy node's dst has an external
    // buffer, gallocr leaves it alone).
    ggml_tensor* tap = ggml_cpy(ctx, x, impl->out_final);
    {
        ggml_tensor* n = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, cfg.ln_eps),
                                                impl->w_mnorm), impl->b_mnorm);
        // Block-major token order makes the 2x2 spatial merge a plain reshape:
        // merged token m = tokens 4m..4m+3 channels concatenated.
        ggml_tensor* m = ggml_reshape_2d(ctx, n, C * cfg.merge * cfg.merge,
                                         N / cfg.merge / cfg.merge);
        ggml_tensor* h = ggml_add(ctx, mul_mat_w(ctx, impl->w_mfc1, m), impl->b_mfc1);
        h = ggml_gelu_erf(ctx, h);   // merger uses exact-erf nn.GELU
        x = ggml_add(ctx, mul_mat_w(ctx, impl->w_mfc2, h), impl->b_mfc2);
    }
    ggml_set_output(x);
    impl->out_pooler = x;
    impl->graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(impl->graph, tap);   // off-chain copy into the leaf
    ggml_build_forward_expand(impl->graph, x);     // terminal: pooler output
    impl->allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl->backend));
    ggml_gallocr_alloc_graph(impl->allocr, impl->graph);
    impl->graphs_ready = true;
}

// Host precompute: interpolated pos embeds + axial cos/sin, per resolution.
void MmprojGraph::prepare_host_tables(int gh, int gw) {
    Impl* impl = impl_;
    const MmprojConfig& cfg = impl->cfg;
    const int C = cfg.embed_dim, side = cfg.pos_side, merge = cfg.merge;
    const int N = gh * gw;

    // ---- pos embeds: bilinear align_corners=True from the (side, side) table,
    // emitted in block-major patch order (matches torch's
    // (pos_embed(idx) * w).sum(1) over the same index stream). ----
    const TensorInfo* ti = impl->gguf->tensor("mmproj.pos_embed");
    const float* table = impl->gguf->tensor_f32("mmproj.pos_embed");
    std::vector<float> pos(size_t(C) * N);
    for (int n = 0; n < N; ++n) {
        int row, col;
        decode_block_major(n, gh, gw, merge, &row, &col);
        int r0, r1, c0, c1;
        float wr1, wc1;
        axis_taps(row, gh, side, &r0, &r1, &wr1);
        axis_taps(col, gw, side, &c0, &c1, &wc1);
        const float w00 = (1.f - wr1) * (1.f - wc1);
        const float w01 = (1.f - wr1) * wc1;
        const float w10 = wr1 * (1.f - wc1);
        const float w11 = wr1 * wc1;
        const float* p00 = table + (size_t(r0) * side + c0) * C;
        const float* p01 = table + (size_t(r0) * side + c1) * C;
        const float* p10 = table + (size_t(r1) * side + c0) * C;
        const float* p11 = table + (size_t(r1) * side + c1) * C;
        for (int c = 0; c < C; ++c)
            pos[size_t(n) * C + c] =
                w00 * p00[c] + w01 * p01[c] + w10 * p10[c] + w11 * p11[c];
    }

    // ---- axial cos/sin: inv_freq (hd/4 per axis), position ids block-major.
    // Per-token row (2*hd floats): [cos_h(18) cos_w(18)]x2 | [sin_h sin_w]x2. ----
    const int hd = cfg.head_dim();
    const int half = hd / 2;   // 36 rotation dims per axis pair
    const int nf = half / 2;   // 18 freqs per axis
    std::vector<float> inv(nf);
    for (int i = 0; i < nf; ++i)
        inv[i] = std::pow(cfg.rope_theta, -2.f * i / float(half));

    std::vector<float> cossin(size_t(2 * hd) * N);
    for (int n = 0; n < N; ++n) {
        int row, col;
        decode_block_major(n, gh, gw, merge, &row, &col);
        float* rp = &cossin[size_t(n) * 2 * hd];
        // Row layout = [cos72 | sin72] (row length 2*hd): cos occupies
        // segments 0..3 (k*nf), sin starts at hd (= 4*nf), NOT 2*hd —
        // 2*hd is past the end of the row.
        for (int i = 0; i < nf; ++i) {
            const float ch = std::cos(row * inv[i]);
            const float cw = std::cos(col * inv[i]);
            const float sh = std::sin(row * inv[i]);
            const float sw = std::sin(col * inv[i]);
            rp[0 * nf + i] = ch;             rp[1 * nf + i] = cw;
            rp[2 * nf + i] = ch;             rp[3 * nf + i] = cw;
            rp[hd + 0 * nf + i] = sh;        rp[hd + 1 * nf + i] = sw;
            rp[hd + 2 * nf + i] = sh;        rp[hd + 3 * nf + i] = sw;
        }
    }

    // Inputs + read-back tap live in their own context + buffer so the
    // gallocr pool skips them entirely (stable addresses, radio pattern).
    struct ggml_init_params ip = {size_t(64) << 20, nullptr, true};
    impl->ictx = ggml_init(ip);
    impl->in_patches = ggml_new_tensor_2d(impl->ictx, GGML_TYPE_F32, cfg.kdim(), N);
    impl->in_pos = ggml_new_tensor_2d(impl->ictx, GGML_TYPE_F32, C, N);
    impl->in_cossin = ggml_new_tensor_2d(impl->ictx, GGML_TYPE_F32, 2 * hd, N);
    impl->rope_ones = ggml_new_tensor_2d(impl->ictx, GGML_TYPE_F32, hd, N);
    impl->out_final = ggml_new_tensor_2d(impl->ictx, GGML_TYPE_F32, C, N);
    impl->ibuf = ggml_backend_alloc_ctx_tensors(impl->ictx, impl->backend);
    ggml_backend_tensor_set(impl->in_pos, pos.data(), 0, ggml_nbytes(impl->in_pos));
    ggml_backend_tensor_set(impl->in_cossin, cossin.data(), 0, ggml_nbytes(impl->in_cossin));
    {   // rope ones are constant for the buffer's lifetime (ictx is never
        // written by graph compute), so one upload here covers all runs.
        std::vector<float> ones(size_t(hd) * N, 1.f);
        ggml_backend_tensor_set(impl->rope_ones, ones.data(), 0,
                                ggml_nbytes(impl->rope_ones));
    }
}

// ---- public API ----

bool MmprojGraph::run(const float* patches, int grid_h, int grid_w,
                      std::vector<float>* out_ll_tokens,
                      std::vector<float>* out_final, std::string* err) {
    const MmprojConfig& cfg = impl_->cfg;
    const int64_t N = int64_t(grid_h) * grid_w;
    if (!impl_->weights_ready) build_weights();
    if (N != impl_->cur_n) {
        prepare_host_tables(grid_h, grid_w);
        impl_->cur_n = int(N);
        impl_->cur_gh = grid_h;
        impl_->cur_gw = grid_w;
    }
    if (!impl_->graphs_ready) build_graphs(grid_h, grid_w);

    ggml_backend_tensor_set(impl_->in_patches, patches, 0,
                            ggml_nbytes(impl_->in_patches));
    ggml_backend_graph_compute(impl_->backend, impl_->graph);

    if (out_final) {
        ggml_tensor* t = impl_->out_final;   // ictx leaf, filled by the tap cpy
        out_final->resize(ggml_nbytes(t) / sizeof(float));
        ggml_backend_tensor_get(t, out_final->data(), 0, ggml_nbytes(t));
    }
    if (out_ll_tokens) {
        ggml_tensor* t = impl_->out_pooler;   // graph terminal (4096, N/4)
        out_ll_tokens->resize(ggml_nbytes(t) / sizeof(float));
        ggml_backend_tensor_get(t, out_ll_tokens->data(), 0, ggml_nbytes(t));
    }
    return true;
}

// Dedicated single-output diagnostic graph: seg0 chain + block-0 up to the
// requested component (radio diag pattern — immune to gallocr reuse ambiguity).
// Weights / pos / cossin are shared with the main graphs; the graph itself is
// built, computed and read back entirely within this call.
bool MmprojGraph::run_diag(const float* patches, int grid_h, int grid_w, const char* point,
                           std::vector<float>* out, std::string* err) {
    (void)err;
    const MmprojConfig& cfg = impl_->cfg;
    const int64_t N = int64_t(grid_h) * grid_w;
    if (!impl_->weights_ready) build_weights();
    if (N != impl_->cur_n) {
        prepare_host_tables(grid_h, grid_w);
        impl_->cur_n = int(N);
        impl_->cur_gh = grid_h;
        impl_->cur_gw = grid_w;
        impl_->graphs_ready = false;   // main graphs (if built) are stale-sized
    }

    const int C = cfg.embed_dim, hd = cfg.head_dim(), nh = cfg.num_heads;
    const int K = cfg.kdim();
    const std::string pt(point);

    struct ggml_init_params ip = {size_t(512) << 20, nullptr, true};
    ggml_context* ctx = ggml_init(ip);

    ggml_tensor* inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_set_input(inp);
    ggml_tensor* x = mul_mat_w(ctx, impl_->w_patch, inp);
    x = ggml_add(ctx, x, impl_->b_patch);
    x = ggml_add(ctx, x, impl_->in_pos);

    auto lnorm = [&](ggml_tensor* in, ggml_tensor* w, ggml_tensor* b) {
        return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, in, cfg.ln_eps), w), b);
    };
    auto lin = [&](ggml_tensor* in, ggml_tensor* w, ggml_tensor* b) {
        ggml_tensor* o = mul_mat_w(ctx, w, in);
        return b ? ggml_add(ctx, o, b) : o;
    };

    ggml_tensor* out_t = nullptr;
    ggml_tensor* ones_t = nullptr;    // cos/sin tap materialization input
    ggml_tensor* rope_ones_t = nullptr;  // q_rot-path cossin materialization
    ggml_tensor* n1 = lnorm(x, impl_->w_ln1[0], impl_->b_ln1[0]);
    if (pt == "n1") {
        out_t = n1;
    } else {
        ggml_tensor* qkv = lin(n1, impl_->w_qkv[0], impl_->b_qkv[0]);
        if (pt == "qkv") {
            out_t = qkv;
        } else {
            ggml_tensor* q = ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, nh, N,
                                          hd * sizeof(float), 3 * C * sizeof(float), 0));
            ggml_tensor* k = ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, nh, N,
                                          hd * sizeof(float), 3 * C * sizeof(float),
                                          C * sizeof(float)));
            ggml_tensor* v = ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, nh, N,
                                          hd * sizeof(float), 3 * C * sizeof(float),
                                          2 * C * sizeof(float)));
            ggml_tensor* cosv = ggml_view_2d(ctx, impl_->in_cossin, hd, N,
                                             2 * hd * sizeof(float), 0);
            ggml_tensor* sinv = ggml_view_2d(ctx, impl_->in_cossin, hd, N,
                                             2 * hd * sizeof(float), hd * sizeof(float));
            if (pt == "cos" || pt == "sin") {
                // Materialize the strided view via an elementwise mul against
                // an all-ones input (reshape_3d/cont reject non-contiguous
                // sources — and the view IS the thing we want to see).
                ggml_tensor* ones = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, N);
                ggml_set_input(ones);
                ggml_set_output(ones);
                ones_t = ones;
                out_t = ggml_mul(ctx, pt == "cos" ? cosv : sinv, ones);
            } else {
            // Materialize the strided cossin views first: reshape_3d demands
            // contiguity and the views (nb1 = parent row stride) are not
            // contiguous — a mul-ones copy is the safe materialization.
            ggml_tensor* ones2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, N);
            ggml_set_input(ones2);
            ggml_tensor* cos_m = ggml_mul(ctx, cosv, ones2);
            ggml_tensor* sin_m = ggml_mul(ctx, sinv, ones2);
            ggml_tensor* cosr = ggml_reshape_3d(ctx, cos_m, hd, 1, N);
            ggml_tensor* sinr = ggml_reshape_3d(ctx, sin_m, hd, 1, N);
            rope_ones_t = ones2;
            ggml_tensor* q_rot = rope_axial(ctx, q, cosr, sinr);
            if (pt == "real") {
                // concat's first half (the real part of the rotation) — isolates
                // repeat/mul semantics from concat's ne0 assembly.
                out_t = ggml_view_3d(ctx, q_rot, q_rot->ne[0] / 2, q_rot->ne[1],
                                     q_rot->ne[2], q_rot->nb[1], q_rot->nb[2], 0);
                ggml_set_output(out_t);
                // view of a graph tensor: materialize via mul-ones for a safe read
                ggml_tensor* ones3 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                        q_rot->ne[0] / 2, q_rot->ne[1],
                                                        q_rot->ne[2]);
                ggml_set_input(ones3);
                rope_ones_t = ones3;   // uploaded on the on-chain path
                out_t = ggml_mul(ctx, out_t, ones3);
            } else if (pt == "q_rot") {
                out_t = q_rot;
            } else {
                ggml_tensor* k_rot = rope_axial(ctx, k, cosr, sinr);
                ggml_tensor* qf = ggml_cont(ctx, ggml_permute(ctx, q_rot, 0, 2, 1, 3));
                ggml_tensor* kf = ggml_cont(ctx, ggml_permute(ctx, k_rot, 0, 2, 1, 3));
                ggml_tensor* vf = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
                const float scale = 1.f / std::sqrt(float(hd));
                ggml_tensor* o = ggml_flash_attn_ext(ctx, qf, kf, vf, nullptr, scale, 0.f, 0.f);
                o = ggml_reshape_2d(ctx, o, C, N);
                ggml_tensor* attnbr = lin(o, impl_->w_proj[0], impl_->b_proj[0]);
                if (pt == "attnbr") {
                    out_t = attnbr;
                } else {
                    x = ggml_add(ctx, x, attnbr);
                    ggml_tensor* n2 = lnorm(x, impl_->w_ln2[0], impl_->b_ln2[0]);
                    ggml_tensor* h = ggml_gelu(ctx, lin(n2, impl_->w_fc1[0], impl_->b_fc1[0]));
                    ggml_tensor* mlpbr = lin(h, impl_->w_fc2[0], impl_->b_fc2[0]);
                    if (pt == "mlpbr") {
                        out_t = mlpbr;
                    } else {   // "x"
                        out_t = ggml_add(ctx, x, mlpbr);
                    }
                }
            }
            }
        }
    }
    if (!out_t) { ggml_free(ctx); return false; }
    ggml_set_output(out_t);
    ggml_cgraph* g = ggml_new_graph_custom(ctx, 4096, false);
    // cos/sin taps are off the x chain: keeping x in the graph would make this
    // a TWO-output graph, and gallocr reuse corrupts the extra output (the
    // radio multi-output lesson). A single-output graph must stay single.
    const bool off_chain = (pt == "cos" || pt == "sin");
    if (!off_chain) ggml_build_forward_expand(g, x);
    ggml_build_forward_expand(g, out_t);
    ggml_gallocr* al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->backend));
    ggml_gallocr_alloc_graph(al, g);
    if (!off_chain) {
        ggml_backend_tensor_set(inp, patches, 0, ggml_nbytes(inp));
        if (rope_ones_t) {
            std::vector<float> ones(ggml_nbytes(rope_ones_t) / sizeof(float), 1.f);
            ggml_backend_tensor_set(rope_ones_t, ones.data(), 0, ggml_nbytes(rope_ones_t));
        }
    } else {
        if (ones_t) {
            std::vector<float> ones(ggml_nbytes(ones_t) / sizeof(float), 1.f);
            ggml_backend_tensor_set(ones_t, ones.data(), 0, ggml_nbytes(ones_t));
        }
        if (rope_ones_t) {
            std::vector<float> ones(ggml_nbytes(rope_ones_t) / sizeof(float), 1.f);
            ggml_backend_tensor_set(rope_ones_t, ones.data(), 0, ggml_nbytes(rope_ones_t));
        }
    }
    ggml_backend_graph_compute(impl_->backend, g);
    out->resize(ggml_nbytes(out_t) / sizeof(float));
    ggml_backend_tensor_get(out_t, out->data(), 0, ggml_nbytes(out_t));
    ggml_gallocr_free(al);
    ggml_free(ctx);
    return true;
}

}  // namespace vlx
