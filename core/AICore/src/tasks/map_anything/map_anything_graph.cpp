// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// VGGT-Omega compute graph for ggml (map_anything). See vggt_graph.hpp for the
// layout contract. Mirrors facebookresearch/vggt-omega (see
// third_party/vggt-omega-src): patch 16, DINOv3-style backbone with axial
// RoPE and 4 storage tokens, frame + inter-frame (global|register) blocks
// with QK-norm and LinearKMaskedBias (masks folded into the biases by the
// converter), single-pass camera head, and a pixel-shuffle dense head.
//
// Dimension conventions:
//   - GGUF tensors store ggml ne[] = reverse(torch shape) (done by gguf-py).
//   - Conv2d kernels ne {KW, KH, IC, OC}; ConvTranspose2d {KW, KH, OC, IC}.
//   - Conv2d output {OW, OH, OC, N}.
//   - Tokens {C, T, B}; attention {dh, nh, T, B} -> FA {dh, T, nh, B}.
//   - Rope rows: per-token cos/sin vectors of length dh (axial layout
//     [Ah|Aw|Ah|Aw], rotate pairs (i, i+dh/2)), identity rows for the
//     camera/register/cls/storage prefix, fed as {dh, 1, T, 1} inputs.

#include "map_anything_graph.hpp"

#include <string.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "map_anything_graph_builder.hpp"

namespace map_anything {

// Parity stage-dump hooks are compiled only when requested at configure
// time (cmake -DMAP_ANYTHING_ENABLE_DUMP=ON); production builds drop them to
// save binary size and compile time. At runtime they are driven by
// RuntimeOptions::dump_dir (explicitly passed through the VGGTRuntime
// constructor — no environment variables).
#ifdef MAP_ANYTHING_ENABLE_DUMP
#define MAP_DUMP_ACTIVE() (!opts.dump_dir.empty())
#define MAP_DUMP_DIR() (opts.dump_dir.c_str())
#else
#define MAP_DUMP_ACTIVE() false
#define MAP_DUMP_DIR() ""
#endif

namespace {

constexpr float kEpsLN = 1e-5f;   // torch nn.LayerNorm default (all LNs here)
constexpr float kFovMin = 0.01f;  // camera activation fov = relu(x) + 0.01

// RopePositionEmbedding (base=100): periods_j = base^(2j/(D/2)), j < D/4.
// Row layout for one head: angles = [Ah(D/4) | Aw(D/4)] tiled twice, so the
// rotate pairs (i, i+D/2) reuse the same angle entries.
void build_rope_rows(int Hp,
                     int Wp,
                     int D_head,
                     int prefix_rows,
                     bool sin_table,
                     const float* periods_ckpt,
                     std::vector<float>& out) {  // out: (T, dh) rows
    // periods come from the checkpoint buffer (f16-rounded values!) —
    // recomputing from the formula shifts every angle and breaks parity
    const int grid = Hp * Wp;
    const int n_tok = prefix_rows + grid;
    const int D4 = D_head / 4;
    std::vector<double> periods(D4);
    for (int j = 0; j < D4; j++) periods[j] = periods_ckpt[j];
    const double max_hw = std::max(Hp, Wp);
    out.assign((size_t)n_tok * D_head, sin_table ? 0.0f : 1.0f);
    for (int h = 0; h < Hp; h++) {
        for (int w = 0; w < Wp; w++) {
            const double ch = 2.0 * ((h + 0.5) / max_hw) - 1.0;
            const double cw = 2.0 * ((w + 0.5) / max_hw) - 1.0;
            float* row = &out[(size_t)(prefix_rows + h * Wp + w) * D_head];
            for (int j = 0; j < D4; j++) {
                const double ah = 2.0 * M_PI * ch / periods[j];
                const double aw = 2.0 * M_PI * cw / periods[j];
                // axial layout [Ah | Aw | Ah | Aw]
                const double vals[4] = {ah, aw, ah, aw};
                for (int k = 0; k < 4; k++) {
                    row[j + k * D4] = sin_table ? (float)std::sin(vals[k])
                                                : (float)std::cos(vals[k]);
                }
            }
        }
    }
}

// make_sincos_pos_embed (omega_0=100, torch computes omega in float64):
// row = [sin(pos*omega)_0..D/2-1 | cos(pos*omega)_0..D/2-1].
void make_sincos(int D,
                 const std::vector<double>& pos,
                 std::vector<float>& out) {
    const int half = D / 2;
    out.assign(pos.size() * (size_t)D, 0.0f);
    for (size_t m = 0; m < pos.size(); m++) {
        for (int j = 0; j < half; j++) {
            const double omega =
                    1.0 / std::pow(100.0, (double)j / (double)half);
            const double ang = pos[m] * omega;
            out[m * (size_t)D + (size_t)j] = (float)std::sin(ang);
            out[m * (size_t)D + (size_t)half + (size_t)j] =
                    (float)std::cos(ang);
        }
    }
}

// DenseHead _apply_pos_embed source values (pre ratio=0.1) on an (Hp, Wp)
// grid with oc channels: per pixel [sincos_x(oc) | sincos_y(oc)].
// create_uv_grid spans [-span*(N-1)/N, +span*(N-1)/N] (torch utils.py), NOT
// the full [-span, +span]; with omega_0=100 even a 3% span error shifts the
// j=0 phase by ~2.6 rad and corrupts the whole depth field.
void build_dpt_uv_embed(int Hp, int Wp, int oc, std::vector<float>& out) {
    const double ar = (double)Wp / (double)Hp;
    const double diag = std::sqrt(ar * ar + 1.0);
    const double span_x = ar / diag, span_y = 1.0 / diag;
    const double ex_x = span_x * (double)(Wp - 1) / (double)Wp;
    const double ey_y = span_y * (double)(Hp - 1) / (double)Hp;
    std::vector<double> xs(Wp), ys(Hp);
    for (int i = 0; i < Wp; i++) {
        xs[i] = -ex_x + 2.0 * ex_x * ((double)i / (double)(Wp - 1));
    }
    for (int i = 0; i < Hp; i++) {
        ys[i] = -ey_y + 2.0 * ey_y * ((double)i / (double)(Hp - 1));
    }
    std::vector<double> px((size_t)Hp * Wp), py((size_t)Hp * Wp);
    for (int h = 0; h < Hp; h++) {
        for (int w = 0; w < Wp; w++) {
            px[(size_t)h * Wp + w] = xs[w];
            py[(size_t)h * Wp + w] = ys[h];
        }
    }
    std::vector<float> ex, ey;
    // torch position_grid_to_embed: each axis gets embed_dim/2 channels from
    // its own make_sincos (=[sin(oc/4) | cos(oc/4)]), then concat ->
    // [sin_x | cos_x | sin_y | cos_y]. Passing the full oc here drops all
    // cos channels and doubles the per-axis frequency span.
    make_sincos(oc / 2, px, ex);
    make_sincos(oc / 2, py, ey);
    out.resize((size_t)Hp * Wp * oc);
    // torch concatenates [sincos_x | sincos_y] on the channel dim
    // (position_grid_to_embed), it does NOT add them; adding sin_x+sin_y
    // keeps the magnitude but biases every pixel systematically.
    // ggml tensor ne = {Wp, Hp, oc, 1}: linear idx = c*Wp*Hp + h*Wp + w.
    for (int h = 0; h < Hp; h++) {
        for (int w = 0; w < Wp; w++) {
            const size_t m = (size_t)h * Wp + w;
            const int half = oc / 2;
            for (int c = 0; c < oc; c++) {
                const float v = (c < half) ? ex[m * half + c]
                                           : ey[m * half + (c - half)];
                out[(size_t)c * Wp * Hp + (size_t)h * Wp + w] = 0.1f * v;
            }
        }
    }
}

// Upstream-parity node naming. The map-anything-ggml reference runs on
// pristine ggml + its own k==s CUDA kernel (part A); our chain carries
// overrides (RMBG F32-direct F16 GEMM output, the SAM3 2x2/s2 conv_transpose
// cuBLAS fast path, the cuBLAS route for small-N F32 GEMMs, the sam3d
// precise-math SiLU) that perturb every stage from the patch embed onward. The
// CUDA backend keeps the pristine kernel selection for nodes whose name starts
// with "mapany_" (ggml_cuda_mapany_pristine_node, map_anything_merged/0001),
// so every mul_mat / conv_transpose / SiLU output built here must carry that
// prefix. The Vulkan backend does the same for its flash-attention and
// conv_transpose overrides (RMBG F32-K/V scalar FA, SAM3 2x2/s2 fast path,
// lingbot F32 conv_transpose scalar routing; ggml_vk_mapany_pristine_node,
// same patch part D). The prefix is inert on CPU/Metal.
constexpr const char* kMatmulName = "mapany_mm";
constexpr const char* kConvTransposeName = "mapany_convT";
constexpr const char* kSiluName = "mapany_silu";
constexpr const char* kFlashAttnName = "mapany_fa";

ggml_tensor* silu(ggml_context* ctx, ggml_tensor* x) {
    ggml_tensor* r = ggml_silu(ctx, x);
    ggml_set_name(r, kSiluName);
    return r;
}

ggml_tensor* flash_attn(ggml_context* ctx,
                        ggml_tensor* q,
                        ggml_tensor* k,
                        ggml_tensor* v,
                        float scale) {
    ggml_tensor* r =
            ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
    ggml_set_name(r, kFlashAttnName);
    return r;
}

// Same op decomposition as ggml_conv_2d (im2col -> mul_mat -> permute); the
// only addition is the name on the inner mul_mat, which ggml_conv_2d does not
// expose.
ggml_tensor* mapany_conv_2d(ggml_context* ctx,
                            ggml_tensor* a,
                            ggml_tensor* b,
                            int s0,
                            int s1,
                            int p0,
                            int p1,
                            int d0,
                            int d1) {
    ggml_tensor* im2col =
            ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true,
                        a->type == GGML_TYPE_BF16
                                ? GGML_TYPE_F32
                                : GGML_TYPE_F16);  // [N, OH, OW, IC*KH*KW]
    ggml_tensor* result = ggml_mul_mat(
            ctx,
            ggml_reshape_2d(ctx, im2col, im2col->ne[0],
                            im2col->ne[3] * im2col->ne[2] * im2col->ne[1]),
            ggml_reshape_2d(ctx, a, a->ne[0] * a->ne[1] * a->ne[2], a->ne[3]));
    ggml_set_name(result, kMatmulName);
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2],
                             im2col->ne[3], a->ne[3]);  // [OC, N, OH, OW]
    return ggml_cont(ctx,
                     ggml_permute(ctx, result, 0, 1, 3, 2));  // [N, OC, OH, OW]
}

ggml_tensor* mapany_conv_transpose_2d_p0(ggml_context* ctx,
                                         ggml_tensor* a,
                                         ggml_tensor* b,
                                         int stride) {
    ggml_tensor* r = ggml_conv_transpose_2d_p0(ctx, a, b, stride);
    ggml_set_name(r, kConvTransposeName);
    return r;
}

}  // namespace

// The vggt-omega builder: one implementation of the IGraphBuilder registry
// interface (see src/graph_builder.hpp). New architectures (vggt original,
// pi3, ...) add their own Impl + RegisterBuilder in a sibling file.
struct VGGTRuntime::Impl : IGraphBuilder {
    static constexpr int kPrefixBB = 5;  // 1 cls + 4 storage tokens

    const GGUFModel* m = nullptr;
    Backend be;
    int S = 0, H = 0, W = 0;
    int Hp = 0, Wp = 0;
    int C = 0, nh = 0, dh = 0;
    int P = 0;     // patch tokens per view
    int Nbb = 0;   // backbone tokens per view (5 + P)
    int Nagg = 0;  // aggregator tokens per view (1 + nreg + P)
    int Ncr = 0;   // camera+register tokens per view (1 + nreg)

    // staged weights (single backend buffer)
    ggml_context* wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    std::map<std::string, ggml_tensor*> wt;

    // single inference graph: backbone -> aggregator -> heads, so the
    // backbone output feeds the aggregator in-graph (no host bridge) and one
    // backend synchronization covers the whole model; dump hooks collect
    // intermediate tensors only in MAP_ANYTHING_ENABLE_DUMP builds
    ggml_context* gctx = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_cgraph* graph = nullptr;           // backbone + aggregator + heads
    ggml_tensor* out_patches = nullptr;     // backbone output (dump point)
    ggml_tensor* out_last = nullptr;        // cached layer 23 output
    std::vector<ggml_tensor*> inter_nodes;  // aggregator cached outputs
    ggml_tensor* in_images = nullptr;       // {W, H, 3, S}
    ggml_tensor* in_mean = nullptr;         // {1,1,3,1}
    ggml_tensor* in_std = nullptr;          // {1,1,3,1}
    ggml_tensor* in_one = nullptr;          // {1,1,1,1}
    ggml_tensor* in_fov_min = nullptr;      // {1,1,1,1}
    ggml_tensor* in_cos_bb = nullptr;       // {dh, 1, Nbb, 1}
    ggml_tensor* in_sin_bb = nullptr;
    ggml_tensor* in_cos_agg = nullptr;  // {dh, 1, Nagg, 1}
    ggml_tensor* in_sin_agg = nullptr;
    ggml_tensor* uv_dp[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};

    ggml_tensor* out_pose = nullptr;    // {9, S}
    ggml_tensor* out_depth = nullptr;   // {W, H, 1, S}
    ggml_tensor* out_depthc = nullptr;  // {W, H, 1, S}
    // original-VGGT point head outputs (VGGTImpl only)
    ggml_tensor* out_pts = nullptr;      // {W, H, 3, S}
    ggml_tensor* out_text = nullptr;     // {2048, 1, 1} text alignment (omega)
    ggml_tensor* dbg_all_tok = nullptr;  // dump-only copy of backbone tokens
    ggml_tensor* out_ptsc = nullptr;     // {W, H, 1, S}

    // host caches for the shape-derived graph inputs
    std::vector<float> host_cos_bb, host_sin_bb, host_cos_agg, host_sin_agg;
    std::vector<float> host_uv_dp[5];
    double build_ms = 0.0;  // weights staging + graph build (set by ctor)
    // explicit runtime switches (dump_dir / dot_path — see RuntimeOptions);
    // read by the MAP_DUMP_ACTIVE/MAP_DUMP_DIR macros below, so every dump
    // site must stay inside an Impl member function
    RuntimeOptions opts;
#ifdef MAP_ANYTHING_ENABLE_DUMP
    mutable std::vector<ggml_tensor*> dbg_nodes;  // dumped after stage 1
    mutable std::vector<std::string> dbg_names;
    mutable std::vector<ggml_tensor*> dbg_nodes_late;  // dumped after stage 3
    mutable std::vector<std::string> dbg_names_late;
    // first attention of this graph only; per-Impl so one context's dump
    // state never leaks into another context built later in the process
    mutable bool dbg_attn_done = false;
#endif

    ggml_tensor* w(const char* name) const {
        auto it = wt.find(name);
        if (it == wt.end()) {
            log_error("missing weight tensor: %s", name);
            return nullptr;
        }
        return it->second;
    }

    ggml_tensor* ln(ggml_context* ctx,
                    ggml_tensor* x,
                    ggml_tensor* nw,
                    ggml_tensor* nb,
                    float eps = kEpsLN) const {
        return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), nw), nb);
    }

    ggml_tensor* lin(ggml_context* ctx,
                     ggml_tensor* x,
                     ggml_tensor* w_,
                     ggml_tensor* b) const {
        // CPU mul_mat with an F16 weight against an F32 activation produced
        // wrong values in this shape family, so F16 weights are cast to F32
        // there — ONCE PER CALL, which on GPUs would re-copy the whole
        // weight matrix for every lin() (~350 casts per inference). GPU
        // backends handle F16-weight x F32-activation mul_mat natively
        // (verified vs f32 reference below 0.02 rel), so no cast there.
        // K-quant weights keep their native type everywhere (CUDA's cpy
        // kernel cannot cast K-quants; mul_mat dispatches to the quantized
        // vec_dot kernels).
        ggml_tensor* w32 = w_;
        if (w_->type == GGML_TYPE_F16 && be.name == "CPU") {
            w32 = ggml_cast(ctx, w_, GGML_TYPE_F32);
        }
        // flatten B to 2D before mul_mat (the f16-weight x 3D-f32 path
        // computes wrong values on the CPU backend)
        const int64_t kn = x->ne[1] * x->ne[2] * x->ne[3];
        ggml_tensor* x2 = ggml_reshape_2d(ctx, x, x->ne[0], kn);
        // NOTE: an F16-activation variant (cast x2 to F16 for f16xf16 tensor
        // cores) measured SLOWER on RTX 4090 (176 -> 189 ms): the per-call
        // activation casts outweigh the matmul gain at these shapes. Kept
        // F32 activations; only the weights stay F16 on GPU backends.
        ggml_tensor* r = ggml_mul_mat(ctx, w32, x2);  // {Cout, kn}
        ggml_set_name(r, kMatmulName);
        ggml_tensor* out = ggml_reshape_4d(ctx, r, w32->ne[1], x->ne[1],
                                           x->ne[2], x->ne[3]);
        if (b) {
            // bias {Cout} broadcasts over token/batch axes via {Cout,1,1,1}
            out = ggml_add(ctx, out,
                           ggml_reshape_4d(ctx, b, w32->ne[1], 1, 1, 1));
        }
        return out;
    }

    // conv outputs keep channels in ne2, so 1-D biases must become
    // {1, 1, oc, 1} for ggml's broadcast rule
    ggml_tensor* bias4(ggml_context* ctx, ggml_tensor* b, int64_t oc) const {
        return ggml_reshape_4d(ctx, b, 1, 1, oc, 1);
    }

    // 2D rope on {dh, nh, T, B} with per-token rows {dh, 1, T, 1}:
    // out = x*cos + rotate_half(x)*sin, pairs (i, i+dh/2), rows already
    // carry the axial [Ah|Aw|Ah|Aw] layout (identity rows for special tokens).
    ggml_tensor* rope_apply(ggml_context* ctx,
                            ggml_tensor* x,
                            int dh_l,
                            ggml_tensor* cos_rows,
                            ggml_tensor* sin_rows) const {
        // 2D rope, fully flattened to 2D: ggml {dh, nh*T*S} rows.
        // cos_rows/sin_rows inputs are {dh, nh, T, S} (same trailing dims).
        // per-row layout in the flattened tail: row = (s*nh + t)*nh + h,
        // row content = [Ah(half/2)|Aw(half/2)] x2  (identity for specials).
        const int64_t half = dh_l / 2;
        // flatten x to {dh, R} where R = nh*T*S (same memory order)
        ggml_tensor* x2d =
                ggml_reshape_2d(ctx, x, dh_l, x->ne[1] * x->ne[2] * x->ne[3]);
        ggml_tensor* c2d = ggml_reshape_2d(
                ctx, cos_rows, dh_l,
                cos_rows->ne[1] * cos_rows->ne[2] * cos_rows->ne[3]);
        ggml_tensor* s2d = ggml_reshape_2d(
                ctx, sin_rows, dh_l,
                sin_rows->ne[1] * sin_rows->ne[2] * sin_rows->ne[3]);
        // per-row halves (contiguous views over ne0)
        ggml_tensor* x1 = ggml_cont(
                ctx, ggml_view_2d(ctx, x2d, half, x2d->ne[1], x2d->nb[1], 0));
        ggml_tensor* x2 =
                ggml_cont(ctx, ggml_view_2d(ctx, x2d, half, x2d->ne[1],
                                            x2d->nb[1], half * sizeof(float)));
        ggml_tensor* c1 = ggml_cont(
                ctx, ggml_view_2d(ctx, c2d, half, c2d->ne[1], c2d->nb[1], 0));
        ggml_tensor* s1 = ggml_cont(
                ctx, ggml_view_2d(ctx, s2d, half, s2d->ne[1], s2d->nb[1], 0));
        // out rows: [x1*c1 - x2*s1 | x2*c1 + x1*s1]
        ggml_tensor* a =
                ggml_sub(ctx, ggml_mul(ctx, x1, c1), ggml_mul(ctx, x2, s1));
        ggml_tensor* b =
                ggml_add(ctx, ggml_mul(ctx, x2, c1), ggml_mul(ctx, x1, s1));
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) {
            const char* st = MAP_DUMP_DIR();
            auto dumpn = [&](ggml_tensor* t, const char* nm) {
                ggml_set_output(t);
                dbg_nodes.push_back(t);
                dbg_names.push_back(std::string(st) + "/" + nm + ".bin");
            };
            dumpn(x, "cpp_rope_x");
            dumpn(x1, "cpp_x1c");
            dumpn(x2, "cpp_x2c");
            dumpn(c1, "cpp_c1c");
            dumpn(s1, "cpp_s1c");
            dumpn(a, "cpp_a2d");
            dumpn(b, "cpp_b2d");
        }
#endif
        ggml_tensor* o = ggml_concat(ctx, a, b, 0);  // {dh, R}
        return ggml_cont(ctx, ggml_reshape_4d(ctx, o, dh_l, x->ne[1], x->ne[2],
                                              x->ne[3]));
    }

    // Attention core: x {C, T, B} -> proj out {C, T, B}.
    ggml_tensor* attention(ggml_context* ctx,
                           ggml_tensor* x,
                           int dim,
                           ggml_tensor* wq,
                           ggml_tensor* bq,
                           ggml_tensor* wk,
                           ggml_tensor* bk,
                           ggml_tensor* wv,
                           ggml_tensor* bv,
                           ggml_tensor* wq_n,
                           ggml_tensor* bq_n,
                           ggml_tensor* wk_n,
                           ggml_tensor* bk_n,
                           ggml_tensor* wproj,
                           ggml_tensor* bproj,
                           ggml_tensor* cos_rows,
                           ggml_tensor* sin_rows,
                           int T,
                           int B) const {
        const int dh_l =
                dim /
                nh;  // per-block head dim (64 backbone/agg, 128 cam trunk)
        ggml_tensor* q_raw = lin(ctx, x, wq, bq);
        ggml_tensor* v_raw = lin(ctx, x, wv, bv);
        ggml_tensor* q = ggml_reshape_4d(ctx, q_raw, dh_l, nh, T, B);
        ggml_tensor* k =
                ggml_reshape_4d(ctx, lin(ctx, x, wk, bk), dh_l, nh, T, B);
        ggml_tensor* v = ggml_reshape_4d(ctx, v_raw, dh_l, nh, T, B);
        if (wq_n) {  // per-head LayerNorm(dh), eps 1e-5, before rope
            q = ln(ctx, q, wq_n, bq_n);
            k = ln(ctx, k, wk_n, bk_n);
        }
        if (cos_rows) {  // rope rows match this block's head dim
            q = rope_apply(ctx, q, dh_l, cos_rows, sin_rows);
            k = rope_apply(ctx, k, dh_l, cos_rows, sin_rows);
        }
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // {dh, T, nh, B}
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
#ifdef MAP_ANYTHING_ENABLE_DUMP
        ggml_tensor* v_p = v;
        ggml_tensor* q_p = q;
        ggml_tensor* k_p = k;
#endif
        ggml_tensor* o =
                flash_attn(ctx, q, k, v, 1.0f / std::sqrt((float)dh_l));
        // FA result ne is {dh, nh, T, B} (ggml returns it "permuted":
        // ne1=n_head, ne2=n_tokens, see ggml.c). Reshaping straight to {dim, T,
        // B} yields the head-major token vector (c = h*dh + d) — no extra
        // permute needed.
        o = ggml_reshape_3d(ctx, o, dim, T, B);
        ggml_tensor* attn_core [[maybe_unused]] = o;
        o = lin(ctx, o, wproj, bproj);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE() && !dbg_attn_done) {
            dbg_attn_done = true;
            const char* st = MAP_DUMP_DIR();
            auto dumpa = [&](ggml_tensor* t, const char* nm) {
                ggml_set_output(t);
                dbg_nodes.push_back(t);
                dbg_names.push_back(std::string(st) + "/" + nm + ".bin");
            };
            dumpa(v_p, "cpp_b0_vperm");
            dumpa(v_raw, "cpp_b0_vraw");
            dumpa(q_raw, "cpp_b0_qraw");
            dumpa(q_p, "cpp_b0_qperm");
            dumpa(k_p, "cpp_b0_kperm");
            dumpa(attn_core, "cpp_b0_fattn");
            dumpa(o, "cpp_b0_proj");
        }
#endif
        return o;
    }

    // Pre-norm residual block (SelfAttentionBlock). x {C, T, B}.
    ggml_tensor* block(ggml_context* ctx,
                       ggml_tensor* x,
                       const std::string& p,
                       int dim,
                       bool with_qk,
                       ggml_tensor* cos_rows,
                       ggml_tensor* sin_rows,
                       int T,
                       int B,
                       const char* dbg_prefix = nullptr) const {
        ggml_tensor* h = ln(ctx, x, w((p + ".norm1.weight").c_str()),
                            w((p + ".norm1.bias").c_str()));
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (dbg_prefix) {
            ggml_set_output(h);
            dbg_nodes.push_back(h);
            dbg_names.push_back(std::string(dbg_prefix) + "_norm1.bin");
        }
#endif
        h = attention(
                ctx, h, dim, w((p + ".attn.q.weight").c_str()),
                w((p + ".attn.q.bias").c_str()),
                w((p + ".attn.k.weight").c_str()),
                w((p + ".attn.k.bias").c_str()),
                w((p + ".attn.v.weight").c_str()),
                w((p + ".attn.v.bias").c_str()),
                with_qk ? w((p + ".attn.q_norm.weight").c_str()) : nullptr,
                with_qk ? w((p + ".attn.q_norm.bias").c_str()) : nullptr,
                with_qk ? w((p + ".attn.k_norm.weight").c_str()) : nullptr,
                with_qk ? w((p + ".attn.k_norm.bias").c_str()) : nullptr,
                w((p + ".attn.proj.weight").c_str()),
                w((p + ".attn.proj.bias").c_str()), cos_rows, sin_rows, T, B);
        h = ggml_mul(ctx, h, w((p + ".ls1.gamma").c_str()));
        h = ggml_add(ctx, x, h);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (dbg_prefix) {
            ggml_set_output(h);
            dbg_nodes.push_back(h);
            dbg_names.push_back(std::string(dbg_prefix) + "_mid.bin");
        }
#endif

        ggml_tensor* f = ln(ctx, h, w((p + ".norm2.weight").c_str()),
                            w((p + ".norm2.bias").c_str()));
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (dbg_prefix) {
            ggml_set_output(f);
            dbg_nodes.push_back(f);
            dbg_names.push_back(std::string(dbg_prefix) + "_n2.bin");
        }
#endif
        f = lin(ctx, f, w((p + ".mlp.fc1.weight").c_str()),
                w((p + ".mlp.fc1.bias").c_str()));
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (dbg_prefix) {
            ggml_set_output(f);
            dbg_nodes.push_back(f);
            dbg_names.push_back(std::string(dbg_prefix) + "_f1.bin");
        }
#endif
        f = ggml_gelu_erf(ctx, f);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (dbg_prefix) {
            ggml_set_output(f);
            dbg_nodes.push_back(f);
            dbg_names.push_back(std::string(dbg_prefix) + "_ge.bin");
        }
#endif
        f = lin(ctx, f, w((p + ".mlp.fc2.weight").c_str()),
                w((p + ".mlp.fc2.bias").c_str()));
        f = ggml_mul(ctx, f, w((p + ".ls2.gamma").c_str()));
        return ggml_add(ctx, h, f);
    }

    // ---------- backbone (DinoVisionTransformer) ----------
    // Returns patch tokens {C, P, S} (x_norm_patchtokens).
    ggml_tensor* backbone(ggml_context* ctx, ggml_tensor* img) {
        const int ps = m->meta.patch_size;
        ggml_tensor* x =
                mapany_conv_2d(ctx, w("bb.patch_embed.proj.weight"), img, ps,
                               ps, 0, 0, 1, 1);  // {Wp, Hp, C, S}
        x = ggml_add(ctx, x, bias4(ctx, w("bb.patch_embed.proj.bias"), C));
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));  // {C, Wp, Hp, S}
        x = ggml_reshape_3d(ctx, x, C, P, S);

        ggml_tensor* cls =
                ggml_repeat(ctx, w("bb.cls_token"),
                            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, 1, S));
        ggml_tensor* stor =
                ggml_repeat(ctx, w("bb.storage_tokens"),
                            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, 4, S));
        ggml_tensor* tokens = ggml_concat(ctx, ggml_concat(ctx, cls, stor, 1),
                                          x, 1);  // {C,Nbb,S}
#ifdef MAP_ANYTHING_ENABLE_DUMP
        const bool dbg = MAP_DUMP_ACTIVE();
        if (dbg) {
            ggml_set_output(x);
            dbg_nodes.push_back(x);
            dbg_names.push_back(std::string(MAP_DUMP_DIR()) + "/cpp_conv4.bin");
            ggml_set_output(tokens);
            dbg_nodes.push_back(tokens);
            dbg_names.push_back(std::string(MAP_DUMP_DIR()) +
                                "/cpp_tokens0.bin");
        }
#else
        [[maybe_unused]] const bool dbg = false;
#endif
        for (int i = 0; i < m->meta.depth; i++) {
            tokens = block(
                    ctx, tokens, "bb.blocks." + std::to_string(i), C, false,
                    in_cos_bb, in_sin_bb, Nbb, S,
#ifdef MAP_ANYTHING_ENABLE_DUMP
                    (dbg && i == 0)
                            ? (std::string(MAP_DUMP_DIR()) + "/cpp_b0").c_str()
                            : nullptr
#else
                    nullptr
#endif
            );
#ifdef MAP_ANYTHING_ENABLE_DUMP
            if (dbg && (i == 0 || i == 1 || i == 3 || i == 7 || i == 11 ||
                        i == 15 || i == 19 || i == 23)) {
                char f[256];
                snprintf(f, sizeof(f), "%s/cpp_blk%d.bin", MAP_DUMP_DIR(), i);
                ggml_set_output(tokens);  // gallocr never reuses outputs
                dbg_nodes.push_back(tokens);
                dbg_names.push_back(f);
            }
#endif
        }
        tokens = ln(ctx, tokens, w("bb.norm.weight"), w("bb.norm.bias"));
        // slice patch rows [5, Nbb). ggml_cont materializes the slice as a
        // dense {C, P, S} tensor: run() reads it with
        // ggml_backend_tensor_get, which memcpy's linearly and would
        // mis-align frames >= 1 on the raw strided view (frame s starts at
        // s * Nbb * C, not s * P * C).
        return ggml_cont(
                ctx,
                ggml_view_3d(ctx, tokens, C, P, S, tokens->nb[1], tokens->nb[2],
                             (size_t)kPrefixBB * C * sizeof(float)));
    }

    // ---------- aggregator (frame + inter-frame[global|register]) ----------
    // Fills `inter` with the cached-layer outputs {2C, Nagg, S} (4 items).
    void aggregator(ggml_context* ctx,
                    ggml_tensor* patches,
                    std::vector<ggml_tensor*>& inter) {
        ggml_tensor* cam = w("agg.cam_token");  // {C, 1, 2, 1}
        ggml_tensor* reg = w("agg.reg_token");  // {C, nreg, 2, 1}
        const int nreg = m->meta.num_register_tokens;
        auto pick = [&](ggml_tensor* t, int idx) {
            return ggml_view_3d(ctx, t, t->ne[0], t->ne[1], 1, t->nb[1],
                                t->nb[2], idx * t->nb[2]);
        };
        ggml_tensor* cam_all;
        ggml_tensor* reg_all;
        if (S > 1) {
            // frame 0 uses token slice 0, frames 1..S-1 use slice 1
            // (official slice_expand_and_flatten). Build both parts with
            // matching ne3 (=1) and concat along the FRAME axis (dim 2):
            // concat on dim 1 would require equal ne3, which only holds for
            // S=2 (S-1 == 1) and crashes for S>=3.
            cam_all = ggml_concat(
                    ctx, pick(cam, 0),
                    ggml_repeat(ctx, pick(cam, 1),
                                ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, 1,
                                                   S - 1)),
                    2);
            reg_all = ggml_concat(
                    ctx, pick(reg, 0),
                    ggml_repeat(ctx, pick(reg, 1),
                                ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, nreg,
                                                   S - 1)),
                    2);
        } else {
            cam_all = pick(cam, 0);
            reg_all = pick(reg, 0);
        }
        cam_all = ggml_reshape_3d(ctx, cam_all, C, 1, S);
        reg_all = ggml_reshape_3d(ctx, reg_all, C, nreg, S);
        ggml_tensor* tokens = ggml_concat(
                ctx, ggml_concat(ctx, cam_all, reg_all, 1), patches, 1);

        auto is_register_block = [&](int i) {
            for (int r : m->meta.register_attn_block_idx)
                if (r == i) return true;
            return false;
        };
        auto is_cached = [&](int i) {
            for (int r : m->meta.cached_layer_idx)
                if (r == i) return true;
            return false;
        };

        for (int i = 0; i < m->meta.aa_depth; i++) {
            // frame-level attention within each view (with rope)
            ggml_tensor* f =
                    block(ctx, tokens, "agg.frame." + std::to_string(i), C,
                          true, in_cos_agg, in_sin_agg, Nagg, S);
            // inter-frame attention across views (no rope)
            if (is_register_block(i)) {
                // only camera+register tokens attend across frames
                ggml_tensor* camreg = ggml_cont(
                        ctx,
                        ggml_view_3d(ctx, f, C, Ncr, S, f->nb[1], f->nb[2], 0));
                camreg = ggml_reshape_3d(ctx, camreg, C, Ncr * S, 1);
                camreg = block(ctx, camreg, "agg.inter." + std::to_string(i), C,
                               true, nullptr, nullptr, Ncr * S, 1);
                camreg = ggml_reshape_3d(ctx, camreg, C, Ncr, S);
                ggml_tensor* pat = ggml_cont(
                        ctx, ggml_view_3d(ctx, f, C, P, S, f->nb[1], f->nb[2],
                                          (size_t)Ncr * C * sizeof(float)));
                tokens = ggml_concat(ctx, camreg, pat, 1);
            } else {
                ggml_tensor* g_in = ggml_reshape_3d(ctx, f, C, Nagg * S, 1);
                tokens = block(ctx, g_in, "agg.inter." + std::to_string(i), C,
                               true, nullptr, nullptr, Nagg * S, 1);
                tokens = ggml_reshape_3d(ctx, tokens, C, Nagg, S);
            }
            if (is_cached(i)) {
                inter.push_back(ggml_concat(ctx, f, tokens, 0));  // {2C,Nagg,S}
            }
        }
    }

    // ---------- text alignment head (vggt-omega enable_alignment) ----------
    // Reads the language-aligned embedding out of the FINAL cached
    // aggregator layer: camera+register tokens (Ncr per view) get LayerNorm'd,
    // flattened frame-major (B, S*Ncr, 2C), joined by one learnable language
    // token, run through 4 rope-free SelfAttentionBlocks (same weight layout
    // as the aggregator blocks, so block() is reused with with_qk=false),
    // and the language slot goes through the projector (Linear-GELU-LN-Linear).
    // The trailing L2 normalize happens on the host in run().
    ggml_tensor* text_head(ggml_context* ctx, ggml_tensor* last) {
        const int C2 = (int)last->ne[0];  // 2 * embed_dim
        ggml_tensor* x = ggml_view_3d(ctx, last, C2, Ncr, S, last->nb[1],
                                      last->nb[2], 0);
        x = ln(ctx, x, w("text_alignment_head.token_norm.weight"),
               w("text_alignment_head.token_norm.bias"));
        // torch (B, S*Ncr, C) is frame-major: {C,Ncr,S} -> permute -> {C,S,Ncr}
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
        x = ggml_reshape_3d(ctx, x, C2, S * Ncr, 1);
        x = ggml_concat(ctx, w("text_alignment_head.language_token"), x, 1);
        const int T = S * Ncr + 1;
        for (int i = 0; i < 4; i++)
            x = block(ctx, x,
                      "text_alignment_head.readout_blocks." + std::to_string(i),
                      C2, false, nullptr, nullptr, T, 1);
        x = ggml_view_3d(ctx, x, C2, 1, 1, x->nb[1], x->nb[2], 0);
        x = ln(ctx, x, w("text_alignment_head.language_token_norm.weight"),
               w("text_alignment_head.language_token_norm.bias"));
        x = lin(ctx, x, w("text_alignment_head.embedding_projector.0.weight"),
                w("text_alignment_head.embedding_projector.0.bias"));
        x = ggml_gelu_erf(ctx, x);
        x = ln(ctx, x, w("text_alignment_head.embedding_projector.2.weight"),
               w("text_alignment_head.embedding_projector.2.bias"));
        x = lin(ctx, x, w("text_alignment_head.embedding_projector.3.weight"),
                w("text_alignment_head.embedding_projector.3.bias"));
        return x;  // {2048, 1, 1}
    }

    // ---------- camera head (single pass) ----------
    ggml_tensor* camera_head(ggml_context* ctx, ggml_tensor* last) {
        const int D2 = 2 * C;  // 2048
#ifdef MAP_ANYTHING_ENABLE_DUMP
        auto dump_cam = [&](ggml_tensor* t, const char* nm) {
            if (!MAP_DUMP_ACTIVE()) return;
            ggml_set_output(t);
            dbg_nodes_late.push_back(t);
            dbg_names_late.push_back(std::string(MAP_DUMP_DIR()) + "/" + nm +
                                     ".bin");
        };
#endif
        ggml_tensor* camreg =
                ggml_cont(ctx, ggml_view_3d(ctx, last, D2, Ncr, S, last->nb[1],
                                            last->nb[2], 0));
        camreg = ggml_reshape_3d(ctx, camreg, D2, Ncr * S, 1);
        camreg = ln(ctx, camreg, w("cam.token_norm.weight"),
                    w("cam.token_norm.bias"));
#ifdef MAP_ANYTHING_ENABLE_DUMP
        dump_cam(camreg, "cpp_cam_norm");
#endif
        for (int k = 0; k < m->meta.trunk_depth; k++) {
            camreg = block(ctx, camreg, "cam.trunk." + std::to_string(k), D2,
                           false, nullptr, nullptr, Ncr * S, 1);
#ifdef MAP_ANYTHING_ENABLE_DUMP
            dump_cam(camreg, ("cpp_cam_trunk" + std::to_string(k)).c_str());
#endif
        }
        camreg = ln(ctx, camreg, w("cam.trunk_norm.weight"),
                    w("cam.trunk_norm.bias"));
#ifdef MAP_ANYTHING_ENABLE_DUMP
        dump_cam(camreg, "cpp_cam_trunknorm");
#endif
        camreg = ggml_reshape_3d(ctx, camreg, D2, Ncr, S);
        // camera token is row 0 of each frame
        ggml_tensor* cam_tok = ggml_view_3d(ctx, camreg, D2, 1, S,
                                            camreg->nb[1], camreg->nb[2], 0);
        cam_tok = ggml_cont(ctx, cam_tok);
        cam_tok = ggml_reshape_3d(ctx, cam_tok, D2, S, 1);
        ggml_tensor* raw = lin(ctx, cam_tok, w("cam.camera_branch.0.weight"),
                               w("cam.camera_branch.0.bias"));
        raw = ggml_gelu_erf(ctx, raw);
        raw = lin(ctx, raw, w("cam.camera_branch.2.weight"),
                  w("cam.camera_branch.2.bias"));
#ifdef MAP_ANYTHING_ENABLE_DUMP
        dump_cam(cam_tok, "cpp_cam_tok");
        dump_cam(raw, "cpp_cam_raw");
#endif
        // activate: T linear (rows 0-2), quat linear (3-6), fov relu+0.01 (7-8)
        // ggml_cont: CUDA unary/concat kernels require contiguous inputs
        // (CPU handles strided views; the values are identical either way)
        ggml_tensor* tq =
                ggml_cont(ctx, ggml_view_2d(ctx, raw, 7, S, raw->nb[1], 0));
        ggml_tensor* fov = ggml_cont(
                ctx,
                ggml_view_2d(ctx, raw, 2, S, raw->nb[1], 7 * sizeof(float)));
        return ggml_concat(ctx, tq,
                           ggml_add(ctx, ggml_relu(ctx, fov), in_fov_min), 0);
    }

    // ---------- dense head (DPT + pixel shuffle) ----------
    // ResidualConvUnit. The original VGGT constructs these with
    // nn.ReLU(inplace=True): the first activation(x) overwrites the module
    // input tensor, so the residual add sees relu(x), not x. vggt-omega
    // uses inplace=False and keeps x as the residual base (relu_residual
    // = false, the regression-locked default).
    ggml_tensor* rcu(ggml_context* ctx,
                     const std::string& p,
                     ggml_tensor* x,
                     const char* dump_tag = nullptr,
                     bool relu_residual = false) const {
        ggml_tensor* xr =
                ggml_relu(ctx, x);  // conv-path input AND residual base
        ggml_tensor* a = xr;
        a = ggml_add(ctx,
                     mapany_conv_2d(ctx, w((p + ".conv1.weight").c_str()), a, 1,
                                    1, 1, 1, 1, 1),
                     bias4(ctx, w((p + ".conv1.bias").c_str()), a->ne[2]));
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (dump_tag && MAP_DUMP_ACTIVE()) {
            ggml_set_output(a);
            dbg_nodes.push_back(a);
            dbg_names.push_back(std::string(MAP_DUMP_DIR()) + "/" + dump_tag +
                                "_c1.bin");
        }
#endif
        a = ggml_relu(ctx, a);
        a = ggml_add(ctx,
                     mapany_conv_2d(ctx, w((p + ".conv2.weight").c_str()), a, 1,
                                    1, 1, 1, 1, 1),
                     bias4(ctx, w((p + ".conv2.bias").c_str()), a->ne[2]));
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (dump_tag && MAP_DUMP_ACTIVE()) {
            ggml_set_output(a);
            dbg_nodes.push_back(a);
            dbg_names.push_back(std::string(MAP_DUMP_DIR()) + "/" + dump_tag +
                                "_c2.bin");
        }
#endif
        return ggml_add(ctx, a, relu_residual ? xr : x);
    }

    ggml_tensor* refinenet(ggml_context* ctx,
                           const std::string& p,
                           ggml_tensor* x,
                           ggml_tensor* skip,
                           int64_t tw,
                           int64_t th,
                           const char* dump_tag = nullptr,
                           bool relu_residual = false) const {
        if (skip)
            x = ggml_add(ctx, x,
                         rcu(ctx, p + ".resConfUnit1", skip, nullptr,
                             relu_residual));
        x = rcu(ctx, p + ".resConfUnit2", x,
                p == "dp.scratch.refinenet4" ? "vg_dp_ref4_rcu" : nullptr,
                relu_residual);
        if (tw > 0 && (tw != x->ne[0] || th != x->ne[1])) {
            x = ggml_interpolate(
                    ctx, x, tw, th, x->ne[2], x->ne[3],
                    GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
#ifdef MAP_ANYTHING_ENABLE_DUMP
            if (dump_tag && MAP_DUMP_ACTIVE()) {
                ggml_set_output(x);
                dbg_nodes.push_back(x);
                dbg_names.push_back(std::string(MAP_DUMP_DIR()) + "/" +
                                    dump_tag + "_interp.bin");
            }
#endif
        }
        return ggml_add(
                ctx,
                mapany_conv_2d(ctx, w((p + ".out_conv.weight").c_str()), x, 1,
                               1, 0, 0, 1, 1),
                bias4(ctx, w((p + ".out_conv.bias").c_str()), x->ne[2]));
    }

    ggml_tensor* conv1x1(ggml_context* ctx,
                         const std::string& p,
                         ggml_tensor* x) const {
        ggml_tensor* kw = w((p + ".weight").c_str());
        ggml_tensor* r = mapany_conv_2d(ctx, kw, x, 1, 1, 0, 0, 1, 1);
        return ggml_add(ctx, r, bias4(ctx, w((p + ".bias").c_str()), r->ne[2]));
    }

    ggml_tensor* conv3x3(ggml_context* ctx,
                         const std::string& p,
                         ggml_tensor* x,
                         bool has_bias) const {
        ggml_tensor* r = mapany_conv_2d(ctx, w((p + ".weight").c_str()), x, 1,
                                        1, 1, 1, 1, 1);
        return has_bias ? ggml_add(ctx, r,
                                   bias4(ctx, w((p + ".bias").c_str()),
                                         r->ne[2]))
                        : r;
    }

    // torch pixel_shuffle(r): {C*r*r, H, W} -> {C, H*r, W*r}.
    // Input {W4, H4, C*r2, N} -> output {W, H, C, N}; r2 = r*r.
    // Channel c = i*r + j splits into output offsets (i on y, j on x).
    ggml_tensor* pixel_shuffle(ggml_context* ctx, ggml_tensor* x, int r) const {
        const int64_t W4 = x->ne[0], H4 = x->ne[1], N = x->ne[3];
        ggml_tensor* a = ggml_cont(
                ctx, ggml_permute(ctx, x, 1, 2, 0, 3));  // {c2, W4, H4, N}
        ggml_tensor* b =
                ggml_reshape_4d(ctx, a, r, r, W4, H4 * N);  // j, i, w4, (h4,s)
        ggml_tensor* c = ggml_cont(
                ctx, ggml_permute(ctx, b, 0, 2, 1, 3));  // j, w4, i, (h4,s)
        ggml_tensor* d =
                ggml_reshape_4d(ctx, c, W4 * r, r, H4, N);     // w, i, h4, s
        return ggml_reshape_4d(ctx, d, W4 * r, r * H4, 1, N);  // w, h, 1, s
    }

    void dense_head(ggml_context* ctx,
                    const std::vector<ggml_tensor*>& inter,
                    ggml_tensor** out_depth,
                    ggml_tensor** out_conf) {
        const int D2 = 2 * C;
        const std::string prefix = "dp";
        ggml_tensor* taps[4];
#ifdef MAP_ANYTHING_ENABLE_DUMP
        // gate on the active predicate, not the pointer: an empty dump_dir
        // still yields a non-null c_str(), which would register outputs and
        // force GPU readback + writes to "/<name>.bin"
        auto dmpn = [&](ggml_tensor* t, const char* nm) {
            if (!MAP_DUMP_ACTIVE()) return;
            ggml_set_output(t);
            dbg_nodes_late.push_back(t);
            dbg_names_late.push_back(std::string(MAP_DUMP_DIR()) + "/" + nm +
                                     ".bin");
        };
#else
        auto dmpn = [](ggml_tensor*, const char*) {};
#endif
        for (int i = 0; i < 4; i++) {
            ggml_tensor* x = ggml_cont(
                    ctx, ggml_view_3d(ctx, inter[i], D2, P, S, inter[i]->nb[1],
                                      inter[i]->nb[2],
                                      (size_t)Ncr * D2 * sizeof(float)));
            x = ln(ctx, x, w((prefix + ".norm.weight").c_str()),
                   w((prefix + ".norm.bias").c_str()));
            dmpn(x, ("cpp_dpt" + std::to_string(i) + "_ln").c_str());
            // projects is a 1x1 conv == per-pixel GEMM. The token layout
            // {D2, P, S} is already (C_in, N), so a plain mul_mat against the
            // reshaped weight is exact and skips the direct-conv kernel's
            // per-output-thread scalar loop entirely.
            static const int kTapChLocal[4] = {256, 512, 1024, 1024};
            const int oc_tap = kTapChLocal[i];
            ggml_tensor* w2 = ggml_reshape_2d(
                    ctx,
                    w((prefix + ".projects." + std::to_string(i) + ".weight")
                              .c_str()),
                    D2, oc_tap);           // gguf ne {1,1,ic,oc} -> (ic, oc)
            x = ggml_mul_mat(ctx, w2, x);  // {oc, P, S} (oc on ne[0])
            ggml_set_name(x, kMatmulName);
            x = ggml_add(ctx, x,
                         ggml_reshape_4d(ctx,
                                         w((prefix + ".projects." +
                                            std::to_string(i) + ".bias")
                                                   .c_str()),
                                         oc_tap, 1, 1, 1));
            dmpn(x, ("cpp_dpt" + std::to_string(i) + "_proj").c_str());
            // y ne {oc, P, S} has p = h*Wp + w (w fastest). Reshaping it
            // directly to {oc,Hp,Wp,S} would swap h/w (C-order); route via
            // permute {P, oc, S} so the p split lands as {Wp, Hp} w-fast.
            x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));  // {P, oc, S}
            // {Wp, Hp, oc, S} (w-fast) for the resize conv
            x = ggml_reshape_4d(ctx, x, Wp, Hp, oc_tap, S);
            x = ggml_add(ctx, x, uv_dp[i]);
            dmpn(x, ("cpp_dpt" + std::to_string(i) + "_uv").c_str());
            const std::string rl =
                    prefix + ".resize_layers." + std::to_string(i);
            if (i == 0) {
                // NOTE: ggml's conv_transpose_2d_p0 kernel is the single
                // largest GPU cost (~33%). A GEMM+pixel_shuffle equivalent
                // (k==s case) was prototyped in tests/test_convt_gemm.cpp —
                // the math checks out there, but folding it into the graph
                // needs a 6-D intermediate (o,i,j x w,h interleaves) that
                // ggml's 4-D views cannot express; revisit with a conversion-
                // time weight repack + dedicated op (TODO P2).
                x = ggml_add(ctx,
                             mapany_conv_transpose_2d_p0(
                                     ctx, w((rl + ".weight").c_str()), x, 4),
                             bias4(ctx, w((rl + ".bias").c_str()), x->ne[2]));
            } else if (i == 1) {
                x = ggml_add(ctx,
                             mapany_conv_transpose_2d_p0(
                                     ctx, w((rl + ".weight").c_str()), x, 2),
                             bias4(ctx, w((rl + ".bias").c_str()), x->ne[2]));
            } else if (i == 3) {
                x = mapany_conv_2d(ctx, w((rl + ".weight").c_str()), x, 2, 2, 1,
                                   1, 1, 1);
                x = ggml_add(ctx, x,
                             bias4(ctx, w((rl + ".bias").c_str()), x->ne[2]));
            }  // i == 2: Identity
            dmpn(x, ("cpp_dpt" + std::to_string(i) + "_resize").c_str());
            // layer_rn convs have no bias
            taps[i] = conv3x3(
                    ctx,
                    prefix + ".scratch.layer" + std::to_string(i + 1) + "_rn",
                    x, false);
            dmpn(taps[i], ("cpp_dpt" + std::to_string(i) + "_rn").c_str());
        }
        ggml_tensor* o = refinenet(ctx, prefix + ".scratch.refinenet4", taps[3],
                                   nullptr, taps[2]->ne[0], taps[2]->ne[1]);
        o = refinenet(ctx, prefix + ".scratch.refinenet3", o, taps[2],
                      taps[1]->ne[0], taps[1]->ne[1]);
        o = refinenet(ctx, prefix + ".scratch.refinenet2", o, taps[1],
                      taps[0]->ne[0], taps[0]->ne[1]);
        // refinenet1 keeps the same size (no-op interpolate)
        o = refinenet(ctx, prefix + ".scratch.refinenet1", o, taps[0],
                      taps[0]->ne[0], taps[0]->ne[1]);
        o = ggml_add(ctx, o, uv_dp[4]);  // fused uv embed at 1/4 resolution
        dmpn(o, "cpp_dpt_fused_uv");
        // prediction heads: 1x1 conv to 16 channels, pixel-shuffle x4
        ggml_tensor* logits = conv1x1(ctx, prefix + ".proj", o);
        dmpn(logits, "cpp_dpt_logits");
        logits = pixel_shuffle(ctx, logits, 4);  // {W, H, 1, S}
        *out_depth = ggml_exp(ctx, logits);
        ggml_tensor* clog = conv1x1(ctx, prefix + ".proj_conf", o);
        clog = pixel_shuffle(ctx, clog, 4);
        *out_conf = ggml_add(ctx, ggml_exp(ctx, clog), in_one);  // 1 + exp(x)
    }

    // ---------- host-side inputs (shape-derived, cached per runtime)
    // ----------
    void prepare_host_inputs() {
        const int D_head = dh;
        const HostTensor* per_bb = m->tensor("bb.rope_embed.periods");
        const HostTensor* per_agg = m->tensor("agg.rope_periods");
        if (!per_bb || !per_agg) {
            log_error("rope periods buffers missing from GGUF");
            return;
        }
        const float* pbb = (const float*)per_bb->data.data();
        const float* pagg = (const float*)per_agg->data.data();
        // backbone rope rows: identity for cls + storage, grid for patches
        std::vector<float> rows_cos_bb, rows_sin_bb, rows_cos_agg, rows_sin_agg;
        build_rope_rows(Hp, Wp, D_head, kPrefixBB, false, pbb, rows_cos_bb);
        build_rope_rows(Hp, Wp, D_head, kPrefixBB, true, pbb, rows_sin_bb);
        build_rope_rows(Hp, Wp, D_head, Ncr, false, pagg, rows_cos_agg);
        build_rope_rows(Hp, Wp, D_head, Ncr, true, pagg, rows_sin_agg);
        // expand to full {dh, nh, T, S}: every (head, view) repeats the row set
        auto expand_rows = [&](const std::vector<float>& rows, int T,
                               std::vector<float>& out) {
            out.resize((size_t)D_head * nh * T * S);
            // ggml {dh, nh, T, S}: linear = d + dh*h + dh*nh*t + dh*nh*T*s
            for (int s = 0; s < S; s++)
                for (int t = 0; t < T; t++)
                    for (int h = 0; h < nh; h++)
                        for (int d = 0; d < D_head; d++)
                            out[((size_t)s * nh * T + (size_t)t * nh + h) *
                                        D_head +
                                d] = rows[(size_t)t * D_head + d];
        };
        expand_rows(rows_cos_bb, Nbb, host_cos_bb);
        expand_rows(rows_sin_bb, Nbb, host_sin_bb);
        expand_rows(rows_cos_agg, Nagg, host_cos_agg);
        expand_rows(rows_sin_agg, Nagg, host_sin_agg);
        // dense-head uv embeddings: 4 taps at patch grid + fused at 1/4 res
        static const int kTapCh[4] = {256, 512, 1024, 1024};
        for (int i = 0; i < 4; i++) {
            build_dpt_uv_embed(Hp, Wp, kTapCh[i], host_uv_dp[i]);
        }
        build_dpt_uv_embed(Hp * 4, Wp * 4, m->meta.dpt_features, host_uv_dp[4]);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) {
            std::string d(MAP_DUMP_DIR());
            auto save = [&](const std::string& f, const std::vector<float>& v) {
                FILE* fp = fopen((d + "/" + f).c_str(), "wb");
                if (!fp) return;
                fwrite(v.data(), 4, v.size(), fp);
                fclose(fp);
            };
            save("cpp_rope_cos_bb.bin", host_cos_bb);
            save("cpp_rope_sin_bb.bin", host_sin_bb);
        }
#endif
    }

    // ---------- weights ----------
    bool stage_weights() {
        const size_t n = m->tensors.size();
        ggml_init_params wp{ggml_tensor_overhead() * n + (1u << 20), nullptr,
                            true};
        wctx = ggml_init(wp);
        if (!wctx) return false;
        for (const auto& [name, ht] : m->tensors) {
            ggml_tensor* t = ggml_new_tensor(wctx, ht.type, 4, ht.ne);
            ggml_set_name(t, name.c_str());
            wt[name] = t;
        }
        wbuf = ggml_backend_alloc_ctx_tensors(wctx, be.handle);
        if (!wbuf) return false;
        // WEIGHTS usage tag: sched offload routing requires it (sam3d loader
        // precedent); without it ops silently stay on the CPU secondary.
        ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (const auto& [name, ht] : m->tensors) {
            ggml_backend_tensor_set(wt[name], ht.data.data(), 0,
                                    nbytes_of(ht.type, ht.ne));
        }
        log_info("weights staged: %zu tensors, buffer %.2f GB", n,
                 ggml_backend_buffer_get_size(wbuf) / 1e9);
        return true;
    }

    // ---------- graph ----------
    // virtual: the original-VGGT builder (VGGTImpl, below) overrides the
    // graph-construction stage while reusing weight staging and run()
    virtual bool build_graph() {
        prepare_host_inputs();
        // single merged graph: backbone (~600 nodes) + aggregator+heads;
        // sized with headroom so both fit in one ggml context
        const size_t kNodes = 20480;
        ggml_init_params gp{ggml_tensor_overhead() * kNodes +
                                    ggml_graph_overhead_custom(kNodes, false),
                            nullptr, true};
        gctx = ggml_init(gp);
        if (!gctx) return false;
        return build_stage1();
    }

    virtual bool build_stage1() {
        // static inputs get their OWN backend buffer (see build_stage1_vggt
        // for why: the gallocr pool recycles their space mid-graph after the
        // last use, so "upload once" would not survive a second run())
        ictx = ggml_init({ggml_tensor_overhead() * 64, nullptr, true});
        if (!ictx) return false;
        // ---- graph inputs (in_images is the only per-run input) ----
        in_images = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, W, H, 3, S);
        ggml_set_input(in_images);
        in_mean = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, 1, 1, 3, 1);
        in_std = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, 1, 1, 3, 1);
        in_cos_bb = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, dh, nh, Nbb, S);
        in_sin_bb = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, dh, nh, Nbb, S);
        ggml_set_input(in_cos_bb);
        ggml_set_input(in_sin_bb);
        in_cos_agg = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, dh, nh, Nagg, S);
        in_sin_agg = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, dh, nh, Nagg, S);
        ggml_set_input(in_cos_agg);
        ggml_set_input(in_sin_agg);
        in_one = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, 1, 1, 1, 1);
        in_fov_min = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, 1, 1, 1, 1);
        ggml_set_input(in_one);
        ggml_set_input(in_fov_min);
        static const int kTapChDpt[4] = {256, 512, 1024, 1024};
        for (int i = 0; i < 4; i++) {
            uv_dp[i] = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, Wp, Hp,
                                          kTapChDpt[i], 1);
            ggml_set_input(uv_dp[i]);
        }
        uv_dp[4] =
                ggml_new_tensor_4d(ictx, GGML_TYPE_F32, Wp * 4, Hp * 4, 256, 1);
        ggml_set_input(uv_dp[4]);

        // ---- backbone -> patch tokens ----
        ggml_tensor* img =
                ggml_div(gctx, ggml_sub(gctx, in_images, in_mean), in_std);
        out_patches = backbone(gctx, img);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        // dump point; set_output pins the buffer against pool reuse, which
        // would otherwise cost ~51 MB of live working set in production
        ggml_set_output(out_patches);
#endif

        // ---- aggregator + heads in the SAME graph: the backbone output
        // feeds the aggregator in-graph, so there is no host bridge and one
        // backend synchronization covers the whole model
        std::vector<ggml_tensor*> inter;
        aggregator(gctx, out_patches, inter);
        if (inter.size() != 4) {
            log_error("expected 4 cached layers, got %zu", inter.size());
            return false;
        }
        for (int i = 0; i < 4; i++) {
#ifdef MAP_ANYTHING_ENABLE_DUMP
            ggml_set_output(inter[i]);  // keep alive (dump reads)
#endif
            inter_nodes.push_back(inter[i]);
        }
        out_last = inter.back();
#ifdef MAP_ANYTHING_ENABLE_DUMP
        ggml_set_output(out_last);
#endif
        if (m->meta.enable_text_alignment) {
            out_text = text_head(gctx, out_last);
            ggml_set_output(out_text);
        }
        out_pose = camera_head(gctx, inter[3]);
        dense_head(gctx, inter, &out_depth, &out_depthc);
        ggml_set_output(out_pose);
        ggml_set_output(out_depth);
        ggml_set_output(out_depthc);

        graph = ggml_new_graph_custom(gctx, 20480, false);
        ggml_build_forward_expand(graph, out_patches);
        for (int i = 0; i < 4; i++)
            ggml_build_forward_expand(graph, inter_nodes[i]);
        ggml_build_forward_expand(graph, out_last);
        if (out_text) ggml_build_forward_expand(graph, out_text);
        ggml_build_forward_expand(graph, out_pose);
        ggml_build_forward_expand(graph, out_depth);
        ggml_build_forward_expand(graph, out_depthc);
        galloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(be.handle));
        // give the static inputs their own buffer BEFORE the graph pool is
        // allocated, so gallocr skips them (they must survive across runs)
        ibuf = ggml_backend_alloc_ctx_tensors(ictx, be.handle);
        if (!ibuf) return false;
        if (!alloc_graph("vggt_omega")) return false;

        // static inputs are uploaded ONCE here (the gallocr-assigned device
        // addresses are reused every run); only in_images changes per run
        ggml_backend_tensor_set(in_mean, m->meta.img_mean, 0,
                                3 * sizeof(float));
        ggml_backend_tensor_set(in_std, m->meta.img_std, 0, 3 * sizeof(float));
        ggml_backend_tensor_set(in_cos_bb, host_cos_bb.data(), 0,
                                host_cos_bb.size() * sizeof(float));
        ggml_backend_tensor_set(in_sin_bb, host_sin_bb.data(), 0,
                                host_sin_bb.size() * sizeof(float));
        ggml_backend_tensor_set(in_cos_agg, host_cos_agg.data(), 0,
                                host_cos_agg.size() * sizeof(float));
        ggml_backend_tensor_set(in_sin_agg, host_sin_agg.data(), 0,
                                host_sin_agg.size() * sizeof(float));
        const float one = 1.0f, fov_min = kFovMin;
        ggml_backend_tensor_set(in_one, &one, 0, sizeof(float));
        ggml_backend_tensor_set(in_fov_min, &fov_min, 0, sizeof(float));
        for (int i = 0; i < 5; i++) {
            ggml_backend_tensor_set(uv_dp[i], host_uv_dp[i].data(), 0,
                                    host_uv_dp[i].size() * sizeof(float));
#ifdef MAP_ANYTHING_ENABLE_DUMP
            if (MAP_DUMP_ACTIVE()) {
                std::string f = std::string(MAP_DUMP_DIR()) + "/cpp_uv" +
                                std::to_string(i) + ".bin";
                FILE* fp = fopen(f.c_str(), "wb");
                fwrite(host_uv_dp[i].data(), 4, host_uv_dp[i].size(), fp);
                fclose(fp);
            }
#endif
        }
        log_info("graph built: %d nodes (backbone+aggregator+heads)",
                 ggml_graph_n_nodes(graph));
        return true;
    }

    // ---------- inference ----------
    bool run(const float* imgs, VGGTOutputs& out) {
        if (!graph) return false;
        auto now = []() { return std::chrono::steady_clock::now(); };
        auto ms_between = [](std::chrono::steady_clock::time_point a,
                             std::chrono::steady_clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        out.timing_ms["build_graphs"] = build_ms;
        const auto t0 = now();
        // the only per-run input; every other input is static and was
        // uploaded once at build time. imgs goes straight to the backend:
        // normalization is the graph's first op (GGUF mean/std), so no
        // host-side intermediate buffer is needed.
        ggml_backend_tensor_set(in_images, imgs, 0,
                                (size_t)S * 3 * H * W * sizeof(float));
        const auto t1 = now();
        ggml_backend_graph_compute(be.handle, graph);
        const auto t2 = now();
        out.timing_ms["input_upload"] = ms_between(t0, t1);
        out.timing_ms["graph_compute"] = ms_between(t1, t2);
        out.timing_ms["inference_total"] = ms_between(t0, t2);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        // all dump reads happen after the single compute; the same active
        // predicate gates registration above and every readback/write here
        if (MAP_DUMP_ACTIVE()) {
            auto write_bin = [](const std::string& path, const float* data,
                                size_t n) {
                FILE* fp = fopen(path.c_str(), "wb");
                if (!fp) {
                    log_info("dump skipped (cannot open) %s", path.c_str());
                    return;
                }
                fwrite(data, 4, n, fp);
                fclose(fp);
                log_info("dumped %s", path.c_str());
            };
            auto dump_nodes = [&](const std::vector<ggml_tensor*>& nodes,
                                  const std::vector<std::string>& names) {
                for (size_t i = 0; i < nodes.size(); i++) {
                    std::vector<float> tmp(ggml_nelements(nodes[i]));
                    ggml_backend_tensor_get(nodes[i], tmp.data(), 0,
                                            tmp.size() * sizeof(float));
                    write_bin(names[i], tmp.data(), tmp.size());
                }
            };
            dump_nodes(dbg_nodes, dbg_names);
            dump_nodes(dbg_nodes_late, dbg_names_late);
            const std::string dump_dir = MAP_DUMP_DIR();
            std::vector<float> tmpP((size_t)C * P * S);
            ggml_backend_tensor_get(out_patches, tmpP.data(), 0,
                                    tmpP.size() * sizeof(float));
            write_bin(dump_dir + "/cpp_patches.bin", tmpP.data(), tmpP.size());
            for (int i = 0; i < 4; i++) {
                std::vector<float> tmpI((size_t)2 * C * Nagg * S);
                ggml_backend_tensor_get(inter_nodes[i], tmpI.data(), 0,
                                        tmpI.size() * sizeof(float));
                write_bin(dump_dir + "/cpp_inter" +
                                  std::to_string(m->meta.cached_layer_idx[i]) +
                                  ".bin",
                          tmpI.data(), tmpI.size());
            }
        }
#endif

        // pose {9, S} -> torch (S, 9)
        out.pose_enc.resize((size_t)9 * S);
        ggml_backend_tensor_get(out_pose, out.pose_enc.data(), 0,
                                (size_t)9 * S * sizeof(float));

        // {W, H, ch, S} -> torch (S, H, W, ch)
        auto fetch_perm = [&](ggml_tensor* t, int ch, std::vector<float>& dst) {
            std::vector<float> tmp((size_t)W * H * ch * S);
            ggml_backend_tensor_get(t, tmp.data(), 0,
                                    tmp.size() * sizeof(float));
            dst.resize((size_t)S * H * W * ch);
            for (int s = 0; s < S; s++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        const size_t src = (size_t)w + (size_t)W * h +
                                           (size_t)W * H * ch * s;
                        for (int c = 0; c < ch; c++) {
                            dst[(((size_t)s * H + h) * W + w) * ch + c] =
                                    tmp[src + (size_t)W * H * c];
                        }
                    }
                }
            }
        };
        out.out_h = H;  // omega's pixel-shuffle head returns the input res
        out.out_w = W;
        fetch_perm(out_depth, 1, out.depth);
        fetch_perm(out_depthc, 1, out.depth_conf);

        if (out_text) {
            constexpr int kTextDim = 2048;  // 2 * embed_dim(1024)
            std::vector<float> tmp(kTextDim);
            ggml_backend_tensor_get(out_text, tmp.data(), 0,
                                    kTextDim * sizeof(float));
            // official head ends with F.normalize(dim=-1)
            double n2 = 0.0;
            for (float v : tmp) n2 += double(v) * v;
            const float inv = n2 > 0.0 ? float(1.0 / std::sqrt(n2)) : 0.0f;
            for (float& v : tmp) v *= inv;
            out.text_embedding = std::move(tmp);
        }
        return true;
    }

    // ---------- IGraphBuilder ----------
    // Load-time: shapes/derived constants, weight staging, graph build,
    // one-time upload of static inputs.
    bool init(const GGUFModel& model,
              Backend backend,
              int n_views,
              int height,
              int width,
              const RuntimeOptions& runtime_opts) override {
        m = &model;
        be = backend;
        opts = runtime_opts;
        S = n_views;
        H = height;
        W = width;
        C = model.meta.embed_dim;
        nh = model.meta.num_heads;
        dh = C / nh;
        const int ps = model.meta.patch_size;
        if (height < ps || width < ps) {
            log_error("H and W must be at least the patch size %d, got %dx%d",
                      ps, height, width);
            return false;
        }
        // Non-multiple sizes are fine: the patch conv truncates
        // (floor((H-ps)/ps)+1 == H/ps integer division), matching torch.
        Hp = height / ps;
        Wp = width / ps;
        P = Hp * Wp;
        Ncr = 1 + model.meta.num_register_tokens;
        Nbb = kPrefixBB + P;
        Nagg = Ncr + P;
        const auto build_t0 = std::chrono::steady_clock::now();
        if (!stage_weights()) {
            log_error("vggt: weight staging failed");
            return false;
        }
        log_info("vggt: weights staged, building graph");
        if (!build_graph()) {
            log_error("builder init failed");
            return false;
        }
        if (!admit_device_scratch()) return false;
        build_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - build_t0)
                           .count();
        // graph-shape debugging aid, valid in every build for every
        // registered builder (moved here from the pi3 builder so the
        // switch is uniform; unlike the dumps it has no compile-time
        // dependency)
        if (!opts.dot_path.empty())
            ggml_graph_dump_dot(graph, nullptr, opts.dot_path.c_str());
        return true;
    }

    // own backend buffer for the static graph inputs (see build_stage1_vggt)
    ggml_context* ictx = nullptr;
    ggml_backend_buffer_t ibuf = nullptr;

    // Set when init failed because the device cannot hold this geometry
    // (graph buffer allocation or the scratch admission below), so callers
    // can shrink the view batch instead of treating it as a model error.
    bool out_of_memory = false;
    std::string init_error;

    // Every builder allocates its graph through here: a failed gallocr
    // reservation on a GPU is a capacity fact, not a malformed graph.
    bool alloc_graph(const char* tag) {
        if (ggml_gallocr_alloc_graph(galloc, graph)) return true;
        out_of_memory = true;
        char buf[192];
        std::snprintf(buf, sizeof buf,
                      "%s: compute graph for %d views at %dx%d does not fit "
                      "in device memory",
                      tag, S, H, W);
        init_error = buf;
        log_error("%s", buf);
        return false;
    }

    // Device memory the CUDA backend takes from its op-scratch pool OUTSIDE
    // the gallocr buffer, upper-bounded per node: an F16/quantized GEMM
    // converts src1 (<= 2 B/elem: F16 or Q8_1), may stage an F16 dst and,
    // on the cuBLAS route, an F16 dequantized src0; flash attention converts
    // F32 Q/K/V to F16 and may keep F32 split-K partials of dst. Pool
    // buffers are released when their op returns, so the requirement is
    // the max over nodes, not the sum. O(nodes).
    static size_t op_scratch_upper_bound(ggml_cgraph* g) {
        size_t peak = 0;
        for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
            const ggml_tensor* n = ggml_graph_node(g, i);
            size_t s = 0;
            if (n->op == GGML_OP_MUL_MAT && n->src[0]->type != GGML_TYPE_F32) {
                s = 2 * ggml_nelements(n->src[1]) + 2 * ggml_nelements(n);
                if (ggml_is_quantized(n->src[0]->type))
                    s += 2 * ggml_nelements(n->src[0]);
            } else if (n->op == GGML_OP_FLASH_ATTN_EXT) {
                s = 2 * (ggml_nelements(n->src[0]) + ggml_nelements(n->src[1]) +
                         ggml_nelements(n->src[2])) +
                    8 * ggml_nelements(n);
            }
            peak = std::max(peak, s);
        }
        return peak;
    }

    // Admission for the first compute: the gallocr buffer is already
    // resident, but the backend's scratch pool grows during compute and the
    // CUDA pool aborts the process when a growth fails. Refusing the
    // geometry here turns that abort into a recoverable error. The fixed
    // part covers cuBLAS handle/workspace and lazily loaded kernels.
    bool admit_device_scratch() {
        ggml_backend_dev_t dev = ggml_backend_get_device(be.handle);
        if (dev == nullptr) return true;
        const auto type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            return true;
        }
        size_t free_b = 0, total_b = 0;
        ggml_backend_dev_memory(dev, &free_b, &total_b);
        if (total_b == 0) return true;  // backend reports no memory info
        constexpr size_t kFixed = size_t{512} << 20;
        const size_t scratch = op_scratch_upper_bound(graph);
        const size_t need = scratch + kFixed;
        const size_t graph_b = ggml_gallocr_get_buffer_size(galloc, 0);
        log_info(
                "device memory: weights %zu MiB, graph %zu MiB, op scratch "
                "<= %zu MiB, free %zu MiB",
                (wbuf ? ggml_backend_buffer_get_size(wbuf) : 0) >> 20,
                graph_b >> 20, scratch >> 20, free_b >> 20);
        if (free_b >= need) return true;
        out_of_memory = true;
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "%d views at %dx%d need ~%zu MiB of device scratch "
                      "beyond the %zu MiB graph buffer, %zu MiB free",
                      S, H, W, need >> 20, graph_b >> 20, free_b >> 20);
        init_error = buf;
        log_error("%s", buf);
        return false;
    }

    ~Impl() {
        if (galloc) ggml_gallocr_free(galloc);
        if (gctx) ggml_free(gctx);
        if (ibuf) ggml_backend_buffer_free(ibuf);
        if (ictx) ggml_free(ictx);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
    }
};

// -----------------------------------------------------------------
// The original-VGGT builder (facebook/VGGT-1B, general.architecture="vggt").
// Differences vs vggt-omega, grounded in the official sources:
//   * patch_embed is DINOv2: no rope; a learnable pos_embed (1, 1+P0, C)
//     added over [cls | patches] BEFORE the register tokens are inserted
//     (registers carry no positional embedding); identity at the official
//     518x518 resolution — other sizes need bicubic interpolation (TODO);
//   * BOTH frame and global attention apply the formula-based
//     RotaryPositionEmbedding2D (freq 100, patch coords +1, special tokens
//     at 0); omega applies rope only on frame attention via a periods
//     buffer, with a different channel layout;
//   * cross-frame blocks are named global_blocks (internals identical to
//     omega's inter_frame_blocks: qk-norm + LayerScale); frame blocks use a
//     plain qkv bias (no LinearKMaskedBias), and there are no
//     register-attention blocks;
//   * register count 4 (omega 16);
//   * camera head: 4-step adaLN iterative refinement (embed_pose /
//     poseLN_modulation / parameter-free adaln_norm / pose_branch),
//     producing pose_enc = activate(prev + delta) with fov = relu (no
//     +0.01 clamp);
//   * two classic-DPT heads (depth output_dim=2 activation=exp; point
//     output_dim=4 activation=inv_log; conf expp1 -> 1+exp), final
//     upsampling by bilinear interpolate to (Hp*patch, Wp*patch) instead of
//     omega's pixel_shuffle.
struct VGGTRuntime::VGGTImpl : VGGTRuntime::Impl {
    // host-side formula rope rows {dh, nh, Nagg, S} and DPT uv embeddings
    std::vector<float> host_cos_v, host_sin_v;
    std::vector<float> host_pos_embed;  // {1+P, C} rows (cls row first)
    std::vector<float> host_uv_tap[4];
    std::vector<float> host_uv_fused;  // {(Hp*ps), (Wp*ps), 128}
    ggml_tensor* in_cos_v = nullptr;
    ggml_tensor* in_sin_v = nullptr;
    ggml_tensor* in_pos_embed = nullptr;
    ggml_tensor* uv_tap[4] = {nullptr, nullptr, nullptr, nullptr};
    ggml_tensor* uv_fused = nullptr;

    // RotaryPositionEmbedding2D(frequency=100): per-axis frequency
    // f_j = 100^(-2j/32), j in [0,16), applied to the two dh/2 halves;
    // patch coords are (h+1, w+1) and special tokens sit at 0.
    void build_rope_rows_vggt() {
        const int dh_l = C / nh;
        const int half = dh_l / 4;  // 16: frequency count per axis
        host_cos_v.assign((size_t)dh_l * nh * Nagg * S, 0.0f);
        host_sin_v.assign((size_t)dh_l * nh * Nagg * S, 0.0f);
        for (int s = 0; s < S; s++) {
            for (int h = 0; h < nh; h++) {
                for (int t = 0; t < Nagg; t++) {
                    double py = 0.0, px = 0.0;
                    if (t >= Ncr) {
                        const int i = t - Ncr;
                        py = double(i / Wp) + 1.0;
                        px = double(i % Wp) + 1.0;
                    }
                    // ggml {dh_l, nh, Nagg, S} row-major: linear =
                    // d + dh_l*h + dh_l*nh*t + dh_l*nh*Nagg*s (matches the
                    // omega builder's expand_rows indexing)
                    float* crow =
                            &host_cos_v[(((size_t)s * Nagg + t) * nh + h) *
                                        dh_l];
                    float* srow =
                            &host_sin_v[(((size_t)s * Nagg + t) * nh + h) *
                                        dh_l];
                    for (int j = 0; j < half; j++) {
                        const double f =
                                std::pow(100.0, -2.0 * j / (double)(half * 2));
                        const float cy = (float)std::cos(py * f);
                        const float sy = (float)std::sin(py * f);
                        const float cx = (float)std::cos(px * f);
                        const float sx = (float)std::sin(px * f);
                        crow[j] = cy;
                        crow[j + half] = cy;
                        crow[j + 2 * half] = cx;
                        crow[j + 3 * half] = cx;
                        srow[j] = sy;
                        srow[j + half] = sy;
                        srow[j + 2 * half] = sx;
                        srow[j + 3 * half] = sx;
                    }
                }
            }
        }
    }

    // DPT pos-embed tables: one per tap at the PATCH-grid resolution (the
    // official head applies pos_embed right after projects, before resize),
    // plus the fused map at full resolution; same utils as omega (verified
    // identical), so build_dpt_uv_embed is reused verbatim.
    void build_uv_tables_vggt() {
        for (int i = 0; i < 4; i++) {
            build_dpt_uv_embed(Hp, Wp, kTapChV[i], host_uv_tap[i]);
        }
        build_dpt_uv_embed(Hp * m->meta.patch_size, Wp * m->meta.patch_size,
                           128, host_uv_fused);
    }

    static constexpr int kTapChV[4] = {256, 512, 1024, 1024};

#ifdef MAP_ANYTHING_ENABLE_DUMP
    // blocks instrumented for the stage-by-stage parity sweep
    static bool is_dump_block(const std::string& p) {
        return p == "bb.blocks.0" || p == "agg.frame.0" || p == "agg.inter.0" ||
               p == "dec.1" || p == "dec.2";
    }
    static std::string dump_tag(const std::string& p) {
        std::string s = p;
        for (auto& c : s)
            if (c == '.') c = '_';
        return s;
    }
    void dump_node(ggml_tensor* t, const std::string& name) const {
        ggml_set_output(t);
        dbg_nodes.push_back(t);
        dbg_names.push_back(std::string(MAP_DUMP_DIR()) + "/" + name + ".bin");
    }
#endif

    // x (64, R) with rows [y(32) | x(32)]: out = x*cos + rot32(x)*sin where
    // rot32 halves each 32-d segment (pairs (j, j+16)).
    ggml_tensor* rope_apply_vggt(ggml_context* ctx,
                                 ggml_tensor* x,
                                 int dh_l,
                                 ggml_tensor* cos_rows,
                                 ggml_tensor* sin_rows) const {
        auto seg = [&](ggml_tensor* t2d, int off) {
            // off is a 16-float segment index within each 64-float row
            return ggml_cont(
                    ctx,
                    ggml_view_2d(ctx, t2d, dh_l / 4, t2d->ne[1], t2d->nb[1],
                                 off * (dh_l / 4) * sizeof(float)));
        };
        ggml_tensor* x2d =
                ggml_reshape_2d(ctx, x, dh_l, x->ne[1] * x->ne[2] * x->ne[3]);
        ggml_tensor* c2d = ggml_reshape_2d(
                ctx, cos_rows, dh_l,
                cos_rows->ne[1] * cos_rows->ne[2] * cos_rows->ne[3]);
        ggml_tensor* s2d = ggml_reshape_2d(
                ctx, sin_rows, dh_l,
                sin_rows->ne[1] * sin_rows->ne[2] * sin_rows->ne[3]);
        ggml_tensor *v1 = seg(x2d, 0), *v2 = seg(x2d, 1);
        ggml_tensor *h1 = seg(x2d, 2), *h2 = seg(x2d, 3);
        ggml_tensor *cy1 = seg(c2d, 0), *cy2 = seg(c2d, 1);
        ggml_tensor *sy1 = seg(s2d, 0), *sy2 = seg(s2d, 1);
        ggml_tensor *cx1 = seg(c2d, 2), *cx2 = seg(c2d, 3);
        ggml_tensor *sx1 = seg(s2d, 2), *sx2 = seg(s2d, 3);
        ggml_tensor* o =
                ggml_concat(ctx,
                            ggml_concat(ctx,
                                        ggml_sub(ctx, ggml_mul(ctx, v1, cy1),
                                                 ggml_mul(ctx, v2, sy1)),
                                        ggml_add(ctx, ggml_mul(ctx, v2, cy2),
                                                 ggml_mul(ctx, v1, sy2)),
                                        0),
                            ggml_concat(ctx,
                                        ggml_sub(ctx, ggml_mul(ctx, h1, cx1),
                                                 ggml_mul(ctx, h2, sx1)),
                                        ggml_add(ctx, ggml_mul(ctx, h2, cx2),
                                                 ggml_mul(ctx, h1, sx2)),
                                        0),
                            0);
        return ggml_cont(ctx, ggml_reshape_4d(ctx, o, dh_l, x->ne[1], x->ne[2],
                                              x->ne[3]));
    }

    // Attention with the VGGT rope layout (otherwise identical to
    // Impl::attention).
    ggml_tensor* attention_v(ggml_context* ctx,
                             ggml_tensor* x,
                             int dim,
                             const std::string& p,
                             bool with_qk,
                             ggml_tensor* cos_rows,
                             ggml_tensor* sin_rows,
                             int T,
                             int B) const {
        const int dh_l = dim / nh;
        ggml_tensor* q_raw = lin(ctx, x, w((p + ".attn.q.weight").c_str()),
                                 w((p + ".attn.q.bias").c_str()));
        ggml_tensor* v_raw = lin(ctx, x, w((p + ".attn.v.weight").c_str()),
                                 w((p + ".attn.v.bias").c_str()));
        ggml_tensor* k_raw = lin(ctx, x, w((p + ".attn.k.weight").c_str()),
                                 w((p + ".attn.k.bias").c_str()));
        ggml_tensor* q = ggml_reshape_4d(ctx, q_raw, dh_l, nh, T, B);
        ggml_tensor* k = ggml_reshape_4d(ctx, k_raw, dh_l, nh, T, B);
        ggml_tensor* v = ggml_reshape_4d(ctx, v_raw, dh_l, nh, T, B);
        if (with_qk) {
            q = ln(ctx, q, w((p + ".attn.q_norm.weight").c_str()),
                   w((p + ".attn.q_norm.bias").c_str()));
            k = ln(ctx, k, w((p + ".attn.k_norm.weight").c_str()),
                   w((p + ".attn.k_norm.bias").c_str()));
        }
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE() && is_dump_block(p)) {
            dump_node(q, "vg_" + dump_tag(p) + "_q_norope");
            dump_node(k, "vg_" + dump_tag(p) + "_k_norope");
        }
        if (MAP_DUMP_ACTIVE() && p == "bb.blocks.0") {
            dump_node(q, "vg_bb0_q_raw");
            dump_node(k, "vg_bb0_k_raw");
            dump_node(v, "vg_bb0_v_raw");
        }
#endif
        if (cos_rows) {
            q = rope_apply_vggt(ctx, q, dh_l, cos_rows, sin_rows);
            k = rope_apply_vggt(ctx, k, dh_l, cos_rows, sin_rows);
        }
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
#ifdef MAP_ANYTHING_ENABLE_DUMP
        // dump AFTER the permute-cont: these are main-chain nodes, and the
        // {dh,T,nh,B} layout equals numpy (B,nh,T,dh) directly
        if (MAP_DUMP_ACTIVE() && is_dump_block(p)) {
            dump_node(q, "vg_" + dump_tag(p) + "_q_rope");
            dump_node(k, "vg_" + dump_tag(p) + "_k_rope");
        }
#endif
        ggml_tensor* o =
                flash_attn(ctx, q, k, v, 1.0f / std::sqrt((float)dh_l));
        o = ggml_reshape_3d(ctx, o, dim, T, B);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE() && is_dump_block(p)) {
            dump_node(o, "vg_" + dump_tag(p) + "_sdpa");
        }
#endif
        return lin(ctx, o, w((p + ".attn.proj.weight").c_str()),
                   w((p + ".attn.proj.bias").c_str()));
    }

    // Pre-norm block with LayerScale and the VGGT rope (no bias-mask).
    // eps: LayerNorm epsilon — the DINOv2 backbone uses 1e-6 (see
    // vision_transformer.py norm_layer), the aggregator blocks 1e-5.
    // mlp_swiglu: MapAnything's DINOv2-G/AAT blocks use SwiGLUFFNFused
    // (w12 2*hidden + w3) instead of the GELU fc1/fc2 pair.
    ggml_tensor* block_v(ggml_context* ctx,
                         ggml_tensor* x,
                         const std::string& p,
                         int dim,
                         bool with_qk,
                         ggml_tensor* cos_rows,
                         ggml_tensor* sin_rows,
                         int T,
                         int B,
                         float eps = kEpsLN,
                         bool with_ls = true,
                         bool mlp_swiglu = false) const {
        ggml_tensor* h = ln(ctx, x, w((p + ".norm1.weight").c_str()),
                            w((p + ".norm1.bias").c_str()), eps);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE() && p == "agg.frame.0") {
            dump_node(h, "vg_agg_frame_0_norm1");
        }
#endif
        h = attention_v(ctx, h, dim, p, with_qk, cos_rows, sin_rows, T, B);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE() && is_dump_block(p)) {
            dump_node(h, "vg_" + dump_tag(p) + "_attn");
        }
#endif
        if (with_ls) h = ggml_mul(ctx, h, w((p + ".ls1.gamma").c_str()));
        h = ggml_add(ctx, x, h);
        ggml_tensor* f = ln(ctx, h, w((p + ".norm2.weight").c_str()),
                            w((p + ".norm2.bias").c_str()), eps);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE() && is_dump_block(p)) {
            dump_node(f, "vg_" + dump_tag(p) + "_mlpin");
        }
#endif
        if (mlp_swiglu) {
            // SwiGLUFFNFused: w12 (dim -> 2*hidden), silu(x1)*x2, w3.
            f = lin(ctx, f, w((p + ".mlp.w12.weight").c_str()),
                    w((p + ".mlp.w12.bias").c_str()));  // {2*Hid, T, B}
            const int64_t hid = f->ne[0] / 2;
            ggml_tensor* x1 =
                    ggml_cont(ctx, ggml_view_3d(ctx, f, hid, f->ne[1], f->ne[2],
                                                f->nb[1], f->nb[2], 0));
            ggml_tensor* x2 = ggml_cont(
                    ctx, ggml_view_3d(ctx, f, hid, f->ne[1], f->ne[2], f->nb[1],
                                      f->nb[2], hid * sizeof(float)));
            f = ggml_mul(ctx, silu(ctx, x1), x2);
            f = lin(ctx, f, w((p + ".mlp.w3.weight").c_str()),
                    w((p + ".mlp.w3.bias").c_str()));
        } else {
            f = lin(ctx, f, w((p + ".mlp.fc1.weight").c_str()),
                    w((p + ".mlp.fc1.bias").c_str()));
            f = ggml_gelu_erf(ctx, f);
            f = lin(ctx, f, w((p + ".mlp.fc2.weight").c_str()),
                    w((p + ".mlp.fc2.bias").c_str()));
        }
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE() && is_dump_block(p)) {
            dump_node(f, "vg_" + dump_tag(p) + "_mlpout");
        }
#endif
        if (with_ls) f = ggml_mul(ctx, f, w((p + ".ls2.gamma").c_str()));
        ggml_tensor* out = ggml_add(ctx, h, f);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE() && is_dump_block(p)) {
            dump_node(out, "vg_" + dump_tag(p) + "_blk");
        }
#endif
        return out;
    }

    // DINOv2 patch embed: conv proj -> [cls | patches] + pos_embed ->
    // insert registers -> 24 blocks (no rope, no qk-norm) -> final norm ->
    // slice patch tokens {C, P, S}.
    ggml_tensor* backbone_vggt(ggml_context* ctx, ggml_tensor* img) {
        const int ps = m->meta.patch_size;
        ggml_tensor* x = mapany_conv_2d(ctx, w("bb.patch_embed.proj.weight"),
                                        img, ps, ps, 0, 0, 1, 1);
        x = ggml_add(ctx, x, bias4(ctx, w("bb.patch_embed.proj.bias"), C));
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
        x = ggml_reshape_3d(ctx, x, C, P, S);
        ggml_tensor* cls =
                ggml_repeat(ctx, w("bb.cls_token"),
                            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, 1, S));
        ggml_tensor* pc = ggml_concat(ctx, cls, x, 1);  // {C, 1+P, S}
        pc = ggml_add(ctx, pc, in_pos_embed);
        ggml_tensor* regs =
                ggml_repeat(ctx, w("bb.register_tokens"),
                            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, 4, S));
        ggml_tensor* cls_row = ggml_cont(
                ctx, ggml_view_3d(ctx, pc, C, 1, S, pc->nb[1], pc->nb[2], 0));
        ggml_tensor* pat =
                ggml_cont(ctx, ggml_view_3d(ctx, pc, C, P, S, pc->nb[1],
                                            pc->nb[2], C * sizeof(float)));
        ggml_tensor* tokens =
                ggml_concat(ctx, ggml_concat(ctx, cls_row, regs, 1), pat,
                            1);  // {C, Nbb, S}
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) {
            ggml_set_output(tokens);
            dbg_nodes.push_back(tokens);
            dbg_names.push_back(std::string(MAP_DUMP_DIR()) +
                                "/vg_tokens0.bin");
        }
#endif
        for (int i = 0; i < m->meta.depth; i++) {
            tokens = block_v(ctx, tokens, "bb.blocks." + std::to_string(i), C,
                             false, nullptr, nullptr, Nbb, S, 1e-6f);
        }
        tokens = ln(ctx, tokens, w("bb.norm.weight"), w("bb.norm.bias"), 1e-6f);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) {
            // a cont COPY as an independent output; it must also be reachable
            // from the graph outputs (expanded explicitly in build), since
            // isolated tensors are never computed/allocated by the gallocr
            dbg_all_tok = ggml_cont(ctx, tokens);
            ggml_set_output(dbg_all_tok);
        }
#endif
        return ggml_cont(ctx, ggml_view_3d(ctx, tokens, C, P, S, tokens->nb[1],
                                           tokens->nb[2],
                                           (size_t)Ncr * C * sizeof(float)));
    }

    // Aggregator: same cam/register expansion and alternating frame/global
    // pattern as omega; BOTH attentions carry the formula rope; no
    // register-attention blocks.
    void aggregator_vggt(ggml_context* ctx,
                         ggml_tensor* patches,
                         std::vector<ggml_tensor*>& inter) {
        ggml_tensor* cam = w("agg.cam_token");
        ggml_tensor* reg = w("agg.reg_token");
        const int nreg = m->meta.num_register_tokens;
        auto pick = [&](ggml_tensor* t, int idx) {
            return ggml_view_3d(ctx, t, t->ne[0], t->ne[1], 1, t->nb[1],
                                t->nb[2], idx * t->nb[2]);
        };
        ggml_tensor* cam_all;
        ggml_tensor* reg_all;
        if (S > 1) {
            cam_all = ggml_concat(
                    ctx, pick(cam, 0),
                    ggml_repeat(ctx, pick(cam, 1),
                                ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, 1,
                                                   S - 1)),
                    2);
            reg_all = ggml_concat(
                    ctx, pick(reg, 0),
                    ggml_repeat(ctx, pick(reg, 1),
                                ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, nreg,
                                                   S - 1)),
                    2);
        } else {
            cam_all = pick(cam, 0);
            reg_all = pick(reg, 0);
        }
        cam_all = ggml_reshape_3d(ctx, cam_all, C, 1, S);
        reg_all = ggml_reshape_3d(ctx, reg_all, C, nreg, S);
        ggml_tensor* tokens = ggml_concat(
                ctx, ggml_concat(ctx, cam_all, reg_all, 1), patches, 1);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) {
            dump_node(tokens, "vg_agg_in");
        }
#endif
        auto is_cached = [&](int i) {
            for (int r : m->meta.cached_layer_idx)
                if (r == i) return true;
            return false;
        };
        for (int i = 0; i < m->meta.aa_depth; i++) {
            // NOTE: the official Aggregator defaults qk_norm=True, so BOTH
            // the frame and the global blocks apply q_norm/k_norm.
            ggml_tensor* f =
                    block_v(ctx, tokens, "agg.frame." + std::to_string(i), C,
                            true, in_cos_v, in_sin_v, Nagg, S);
            ggml_tensor* g_in = ggml_reshape_3d(ctx, f, C, Nagg * S, 1);
            tokens = block_v(ctx, g_in, "agg.inter." + std::to_string(i), C,
                             true, in_cos_v, in_sin_v, Nagg * S, 1);
            tokens = ggml_reshape_3d(ctx, tokens, C, Nagg, S);
            if (is_cached(i)) {
                inter.push_back(ggml_concat(ctx, f, tokens, 0));
            }
        }
    }

    // Camera head: 4-step adaLN iterative refinement on the camera token
    // (index 0 of the last cached layer).
    ggml_tensor* camera_head_vggt(ggml_context* ctx, ggml_tensor* last) {
        const int D2 = 2 * C;
        ggml_tensor* pose_tokens = ggml_cont(
                ctx,
                ggml_view_3d(ctx, last, D2, 1, S, last->nb[1], last->nb[2], 0));
        pose_tokens = ggml_reshape_3d(ctx, pose_tokens, D2, S, 1);
        pose_tokens = ln(ctx, pose_tokens, w("cam.token_norm.weight"),
                         w("cam.token_norm.bias"));
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) {
            dump_node(pose_tokens, "vg_cam_in");
        }
#endif
        ggml_tensor* pred = nullptr;
        for (int k = 0; k < 4; k++) {
            ggml_tensor* module_input;
            if (k == 0) {
                ggml_tensor* empty = ggml_repeat(
                        ctx, w("cam.empty_pose_tokens"),
                        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 9, S, 1));
                module_input = lin(ctx, empty, w("cam.embed_pose.weight"),
                                   w("cam.embed_pose.bias"));
            } else {
                module_input = lin(ctx, pred, w("cam.embed_pose.weight"),
                                   w("cam.embed_pose.bias"));
            }
            // poseLN_modulation = Sequential(SiLU, Linear): the modulation
            // input passes through SiLU before the Linear (index 1)
            ggml_tensor* mod3 = lin(ctx, silu(ctx, module_input),
                                    w("cam.poseLN_modulation.1.weight"),
                                    w("cam.poseLN_modulation.1.bias"));
            // chunk(3) on ne[0]: each of shift/scale/gate is {D2, S}
            ggml_tensor* shift =
                    ggml_cont(ctx, ggml_view_3d(ctx, mod3, D2, S, 1,
                                                mod3->nb[1], mod3->nb[2], 0));
            ggml_tensor* scale = ggml_cont(
                    ctx, ggml_view_3d(ctx, mod3, D2, S, 1, mod3->nb[1],
                                      mod3->nb[2], D2 * sizeof(float)));
            ggml_tensor* gate = ggml_cont(
                    ctx, ggml_view_3d(ctx, mod3, D2, S, 1, mod3->nb[1],
                                      mod3->nb[2], 2 * D2 * sizeof(float)));
            // adaln_norm: parameter-free LayerNorm eps 1e-6; modulate:
            // x * (1 + scale) + shift (in_one {1,1,1,1} broadcasts)
            ggml_tensor* nrm = ggml_norm(ctx, pose_tokens, 1e-6f);
            nrm = ggml_add(ctx,
                           ggml_mul(ctx, nrm, ggml_add(ctx, scale, in_one)),
                           shift);
            ggml_tensor* modulated =
                    ggml_add(ctx, ggml_mul(ctx, gate, nrm), pose_tokens);
            ggml_tensor* h = modulated;
            for (int b = 0; b < 4; b++) {
                h = block_v(ctx, h, "cam.trunk." + std::to_string(b), D2, false,
                            nullptr, nullptr, S, 1);
            }
            h = ln(ctx, h, w("cam.trunk_norm.weight"),
                   w("cam.trunk_norm.bias"));
            ggml_tensor* delta = lin(ctx, h, w("cam.pose_branch.fc1.weight"),
                                     w("cam.pose_branch.fc1.bias"));
            delta = ggml_gelu_erf(ctx, delta);
            delta = lin(ctx, delta, w("cam.pose_branch.fc2.weight"),
                        w("cam.pose_branch.fc2.bias"));
            pred = (k == 0) ? delta : ggml_add(ctx, pred, delta);
        }
        // activate: T linear, quat linear, fov relu (VGGT has no +0.01 clamp)
        ggml_tensor* tq =
                ggml_cont(ctx, ggml_view_2d(ctx, pred, 7, S, pred->nb[1], 0));
        ggml_tensor* fov = ggml_cont(
                ctx,
                ggml_view_2d(ctx, pred, 2, S, pred->nb[1], 7 * sizeof(float)));
        return ggml_concat(ctx, tq, ggml_relu(ctx, fov), 0);
    }

    // Classic DPT head (depth: output_dim 2, activation exp; point:
    // output_dim 4, activation inv_log). Returns the raw activated outputs
    // {W, H, (dim-1), S} and {W, H, 1, S} confidence.
    void dpt_vggt(ggml_context* ctx,
                  const std::vector<ggml_tensor*>& inter,
                  const std::string& prefix,
                  int output_dim,
                  ggml_tensor** out_main,
                  ggml_tensor** out_conf) {
        const int D2 = 2 * C;
        ggml_tensor* taps[4];
        for (int i = 0; i < 4; i++) {
            ggml_tensor* x = ggml_cont(
                    ctx, ggml_view_3d(ctx, inter[i], D2, P, S, inter[i]->nb[1],
                                      inter[i]->nb[2],
                                      (size_t)Ncr * D2 * sizeof(float)));
            x = ln(ctx, x, w((prefix + ".norm.weight").c_str()),
                   w((prefix + ".norm.bias").c_str()));
            // tokens are row-major over the (Hp, Wp) grid; the conv input
            // needs ne {Wp, Hp, D2, S} (permute: old axis 0 -> new 2, etc.)
            x = ggml_reshape_4d(ctx, x, D2, Wp, Hp, S);
            x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));
            x = conv1x1(ctx, prefix + ".projects." + std::to_string(i), x);
            x = ggml_add(ctx, x, uv_tap[i]);
            const std::string rl =
                    prefix + ".resize_layers." + std::to_string(i);
            if (i == 0) {
                x = ggml_add(ctx,
                             mapany_conv_transpose_2d_p0(
                                     ctx, w((rl + ".weight").c_str()), x, 4),
                             bias4(ctx, w((rl + ".bias").c_str()), x->ne[2]));
            } else if (i == 1) {
                x = ggml_add(ctx,
                             mapany_conv_transpose_2d_p0(
                                     ctx, w((rl + ".weight").c_str()), x, 2),
                             bias4(ctx, w((rl + ".bias").c_str()), x->ne[2]));
            } else if (i == 3) {
                x = mapany_conv_2d(ctx, w((rl + ".weight").c_str()), x, 2, 2, 1,
                                   1, 1, 1);
                x = ggml_add(ctx, x,
                             bias4(ctx, w((rl + ".bias").c_str()), x->ne[2]));
            }  // i == 2: Identity
            taps[i] = x;
#ifdef MAP_ANYTHING_ENABLE_DUMP
            if (MAP_DUMP_ACTIVE() && prefix == "dp") {
                dump_node(x, "vg_dp_tap" + std::to_string(i));
            }
#endif
        }
        ggml_tensor* rn[4];
        for (int i = 0; i < 4; i++) {
            rn[i] = conv3x3(
                    ctx,
                    prefix + ".scratch.layer" + std::to_string(i + 1) + "_rn",
                    taps[i], false);
        }
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE() && prefix == "dp") {
            for (int i = 0; i < 4; i++) {
                dump_node(rn[i], "vg_dp_rn_in" + std::to_string(i + 1));
            }
        }
#endif
        // refinenet chain: 4 (upsampled to tap2 size) -> 3 (tap1) -> 2 (tap0)
        // -> 1 (2x upsample, see below)
        ggml_tensor* o =
                refinenet(ctx, prefix + ".scratch.refinenet4", rn[3], nullptr,
                          rn[2]->ne[0], rn[2]->ne[1],
                          prefix == "dp" ? "vg_dp_ref4" : nullptr, true);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE() && prefix == "dp") dump_node(o, "vg_dp_ref4");
#endif
        o = refinenet(ctx, prefix + ".scratch.refinenet3", o, rn[2],
                      rn[1]->ne[0], rn[1]->ne[1], nullptr, true);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE() && prefix == "dp") dump_node(o, "vg_dp_ref3");
#endif
        o = refinenet(ctx, prefix + ".scratch.refinenet2", o, rn[1],
                      rn[0]->ne[0], rn[0]->ne[1], nullptr, true);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE() && prefix == "dp") dump_node(o, "vg_dp_ref2");
#endif
        // The official head calls refinenet1 WITHOUT a size argument, and
        // FeatureFusionBlock then defaults to scale_factor=2 (bilinear,
        // align_corners=True) — NOT a no-op: the fused map must reach 2x the
        // finest tap grid (296 for 518 input) before the full-res
        // interpolate. (vggt-omega passes size= explicitly, so the shared
        // refinenet() no-ops there — same helper, different call convention.)
        o = refinenet(ctx, prefix + ".scratch.refinenet1", o, rn[0],
                      2 * rn[0]->ne[0], 2 * rn[0]->ne[1], nullptr, true);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE() && prefix == "dp") dump_node(o, "vg_dp_ref1");
#endif
        o = conv3x3(ctx, prefix + ".scratch.output_conv1", o,
                    true);  // 256->128
        // upsample to the full image resolution and add the pos embed
        o = ggml_interpolate(
                ctx, o, Wp * m->meta.patch_size, Hp * m->meta.patch_size,
                o->ne[2], o->ne[3],
                GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
        o = ggml_add(ctx, o, uv_fused);
        o = conv3x3(ctx, prefix + ".scratch.output_conv2.0", o,
                    true);  // 128->32
        o = ggml_relu(ctx, o);
        o = conv1x1(ctx, prefix + ".scratch.output_conv2.2", o);  // 32->dim
        // split channels: main = [0, output_dim-1), conf = last channel.
        // o is {W, H, output_dim, S} (conv layout, channels on ne[2]); the
        // view strides are the DIM-1/2/3 strides, so h steps by o->nb[1],
        // the (size-1) channel dim by o->nb[2], frames by o->nb[3].
        ggml_tensor* main_c = ggml_cont(
                ctx, ggml_view_4d(ctx, o, o->ne[0], o->ne[1], output_dim - 1,
                                  o->ne[3], o->nb[1], o->nb[2], o->nb[3], 0));
        ggml_tensor* conf_c = ggml_cont(
                ctx,
                ggml_view_4d(ctx, o, o->ne[0], o->ne[1], 1, o->ne[3], o->nb[1],
                             o->nb[2], o->nb[3], (output_dim - 1) * o->nb[2]));
        *out_main = main_c;
        *out_conf = conf_c;
    }

    bool build_graph() override {
        prepare_host_inputs_vggt();
        const size_t kNodes = 45000;  // backbone + agg + camera + two DPT heads
        ggml_init_params gp{ggml_tensor_overhead() * kNodes +
                                    ggml_graph_overhead_custom(kNodes, false),
                            nullptr, true};
        gctx = ggml_init(gp);
        if (!gctx) return false;
        return build_stage1_vggt();
    }

    // Host-side tables: formula rope rows, DINOv2 pos-embed rows, DPT uv
    // embeddings. The pos-embed rows support any resolution via the official
    // interpolate_pos_encoding (antialias bicubic at 518 identity).

    // Pillow-style cubic kernel (A=-0.5). torch's ANTIALIAS bicubic
    // evaluates THIS kernel at distance/scale — not the A=-0.75 Keys kernel
    // used by the non-antialias path (verified against F.interpolate on the
    // real pos-embed to 2e-7, CPU and CUDA consistent).
    static float cubic_w_aa(float t) {
        const float at = std::fabs(t);
        if (at <= 1.0f) return (1.5f * at - 2.5f) * at * at + 1.0f;
        if (at < 2.0f) return ((2.5f - 0.5f * at) * at - 4.0f) * at + 2.0f;
        return 0.0f;
    }

    // Keys cubic kernel (A=-0.75): torch's NON-antialias bicubic (cubic
    // convolution). Fixed 4-tap support regardless of the scale, tap indices
    // clamped (replicate borders), and NO renormalization — verified against
    // F.interpolate(antialias=False) on the real pi3 pos-embed to 2.4e-7 at
    // both down- and up-scaled non-square targets.
    static float cubic_w_nonaa(float t) {
        const float at = std::fabs(t);
        if (at <= 1.0f) return (1.25f * at - 2.25f) * at * at + 1.0f;
        if (at < 2.0f) return ((-0.75f * at + 3.75f) * at - 6.0f) * at + 3.0f;
        return 0.0f;
    }

    // DINOv2 interpolate_pos_encoding (official vision_transformer.py).
    // antialias=true / offset=0 (default): the VGGT aggregator's settings.
    // antialias=false / offset=0.1: the pi3 + mapanything encoder defaults
    // (both DINOv2 copies keep the torch-hub defaults).
    // NOTE: the official code reads "B, nc, w, h = x.shape" on a (B,C,H,W)
    // tensor, so its w/h names are swapped; the net effect is the NATURAL
    // alignment: the (M,M) patch grid is resized to (Hp, Wp) (h axis -> Hp,
    // w axis -> Wp), and row 1 + h*Wp + w is added to patch token (h, w).
    // Verified numerically against the executed model (implied-pos match
    // 0.0) at 392x518.
    bool build_pos_embed_rows(std::vector<float>& pe_rows,
                              bool antialias = true,
                              float offset = 0.0f) {
        const HostTensor& pe = m->tensors.at("bb.pos_embed");
        const int64_t Psrc = pe.ne[1] - 1;  // stored patch rows (1369)
        const int M = (int)std::lround(std::sqrt((double)Psrc));
        if (M * M != Psrc) {
            log_error("pos_embed patch rows %lld are not a square grid",
                      (long long)Psrc);
            return false;
        }
        const size_t nsrc = (size_t)(1 + Psrc) * C;
        std::vector<float> pef(nsrc);
        if (pe.type == GGML_TYPE_F32) {
            std::memcpy(pef.data(), pe.data.data(), nsrc * sizeof(float));
        } else if (pe.type == GGML_TYPE_F16) {
            for (size_t i = 0; i < nsrc; i++) {
                ggml_fp16_t h16;
                std::memcpy(&h16, pe.data.data() + i * 2, 2);
                pef[i] = ggml_fp16_to_fp32(h16);
            }
        } else {
            log_error("unsupported pos_embed type %d", (int)pe.type);
            return false;
        }
        pe_rows.assign((size_t)(1 + P) * C, 0.0f);
        // cls row is carried over unchanged
        std::copy(pef.begin(), pef.begin() + C, pe_rows.begin());
        if (P == Psrc) {
            // identity at the trained square grid (official early return)
            std::copy(pef.begin() + C, pef.end(), pe_rows.begin() + C);
            return true;
        }
        const int Ho = Hp, Wo = Wp;  // h axis -> Hp rows, w axis -> Wp cols
        if (!antialias) {
            // torch non-antialias bicubic with the interpolate_offset
            // kludge: the scale_factor path makes rheight = 1/scales_h =
            // M / (Ho + offset) (the offset only shifts the sampling grid,
            // the output size stays floor(M * scales_h) = Ho); the source
            // index max(src, 0) clamp does NOT apply to cubic (negative
            // indices are floored and the TAP indices are clamped instead,
            // replicate); weights are NOT renormalized.
            const float shy = (float)M / (Ho + offset);
            const float shx = (float)M / (Wo + offset);
            for (int a = 0; a < Ho; a++) {
                const float cy = shy * (a + 0.5f) - 0.5f;
                const int iy = (int)std::floor(cy);
                const float ty = cy - (float)iy;
                const float wy[4] = {
                        cubic_w_nonaa(ty + 1.0f), cubic_w_nonaa(ty),
                        cubic_w_nonaa(1.0f - ty), cubic_w_nonaa(2.0f - ty)};
                for (int b = 0; b < Wo; b++) {
                    const float cx = shx * (b + 0.5f) - 0.5f;
                    const int ix = (int)std::floor(cx);
                    const float tx = cx - (float)ix;
                    const float wx[4] = {
                            cubic_w_nonaa(tx + 1.0f), cubic_w_nonaa(tx),
                            cubic_w_nonaa(1.0f - tx), cubic_w_nonaa(2.0f - tx)};
                    // row 1 + h*Wp + w aligns with patch token (h, w)
                    float* dst = &pe_rows[(size_t)(1 + (a * Wo + b)) * C];
                    for (int i = 0; i < 4; i++) {
                        const float wv = wy[i];
                        if (wv == 0.0f) continue;
                        const int sy = std::min(std::max(iy - 1 + i, 0), M - 1);
                        for (int j = 0; j < 4; j++) {
                            const float wgt = wv * wx[j];
                            if (wgt == 0.0f) continue;
                            const int sx =
                                    std::min(std::max(ix - 1 + j, 0), M - 1);
                            const float* src_row =
                                    &pef[(size_t)(1 + sy * M + sx) * C];
                            for (int c = 0; c < C; c++)
                                dst[c] += wgt * src_row[c];
                        }
                    }
                }
            }
            return true;
        }
        const float sh = (float)M / Ho, sw = (float)M / Wo;
        std::vector<float> wy, wx;
        for (int a = 0; a < Ho; a++) {
            const float cy = sh * (a + 0.5f) - 0.5f;
            const float supy = sh > 1.0f ? 2.0f * sh : 2.0f;
            const int sy0 = std::max(0, (int)std::ceil(cy - supy));
            const int sy1 = std::min(M - 1, (int)std::floor(cy + supy));
            const float invy = sh > 1.0f ? 1.0f / sh : 1.0f;
            wy.assign(sy1 - sy0 + 1, 0.0f);
            for (int sy = sy0; sy <= sy1; sy++)
                wy[sy - sy0] = cubic_w_aa((float)(sy - cy) * invy);
            for (int b = 0; b < Wo; b++) {
                const float cx = sw * (b + 0.5f) - 0.5f;
                const float supx = sw > 1.0f ? 2.0f * sw : 2.0f;
                const int sx0 = std::max(0, (int)std::ceil(cx - supx));
                const int sx1 = std::min(M - 1, (int)std::floor(cx + supx));
                const float invx = sw > 1.0f ? 1.0f / sw : 1.0f;
                wx.assign(sx1 - sx0 + 1, 0.0f);
                for (int sx = sx0; sx <= sx1; sx++)
                    wx[sx - sx0] = cubic_w_aa((float)(sx - cx) * invx);
                float wy_sum = 0.0f, wx_sum = 0.0f;
                for (float v : wy) wy_sum += v;
                for (float v : wx) wx_sum += v;
                const float wsum = wy_sum * wx_sum;  // joint 2D normalization
                // row 1 + h*Wp + w aligns with patch token (h, w)
                float* dst = &pe_rows[(size_t)(1 + (a * Wo + b)) * C];
                for (int sy = sy0; sy <= sy1; sy++) {
                    const float wv = wy[sy - sy0];
                    for (int sx = sx0; sx <= sx1; sx++) {
                        const float w = wv * wx[sx - sx0];
                        if (w == 0.0f) continue;
                        const float* src_row =
                                &pef[(size_t)(1 + sy * M + sx) * C];
                        for (int c = 0; c < C; c++) dst[c] += w * src_row[c];
                    }
                }
                for (int c = 0; c < C; c++) dst[c] /= wsum;
            }
        }
        return true;
    }

    bool prepare_host_inputs_vggt(bool pe_antialias = true,
                                  float pe_offset = 0.0f) {
        build_rope_rows_vggt();
        build_uv_tables_vggt();
        // pos-embed rows: row 0 = cls, rows 1..P = the (Hp, Wp) grid via the
        // official interpolation (identity at 518x518); registers carry no
        // positional embedding and get zero rows in the tensor (they are
        // inserted AFTER the pos-embed add in the graph).
        if (!build_pos_embed_rows(host_pos_embed, pe_antialias, pe_offset))
            return false;
        return true;
    }

    bool build_stage1_vggt() {
        log_info("vggt: creating graph inputs");
        // Static inputs get their OWN backend buffer, not the gallocr compute
        // pool: the pool recycles memory across ops and a stray out-of-range
        // write from a later op clobbered a static table that the NEXT
        // run() would read (measured: in_cos_v drifted 47.8 after one
        // compute). ggml_gallocr skips tensors that already carry a buffer,
        // so pre-allocating them here removes them from the pool.
        ictx = ggml_init({ggml_tensor_overhead() * 64, nullptr, true});
        if (!ictx) return false;
        // ---- graph inputs (in_images is the only per-run input) ----
        in_images = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, W, H, 3, S);
        ggml_set_input(in_images);
        in_mean = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, 1, 1, 3, 1);
        in_std = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, 1, 1, 3, 1);
        const int dh_l = C / nh;
        in_cos_v = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, dh_l, nh, Nagg, S);
        in_sin_v = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, dh_l, nh, Nagg, S);
        ggml_set_input(in_cos_v);
        ggml_set_input(in_sin_v);
        in_pos_embed = ggml_new_tensor_3d(ictx, GGML_TYPE_F32, C, 1 + P, 1);
        ggml_set_input(in_pos_embed);
        in_one = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, 1, 1, 1, 1);
        ggml_set_input(in_one);
        for (int i = 0; i < 4; i++) {
            uv_tap[i] = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, Wp, Hp,
                                           kTapChV[i], 1);
            ggml_set_input(uv_tap[i]);
        }
        uv_fused =
                ggml_new_tensor_4d(ictx, GGML_TYPE_F32, Wp * m->meta.patch_size,
                                   Hp * m->meta.patch_size, 128, 1);
        ggml_set_input(uv_fused);

        // ---- backbone (DINOv2) -> patch tokens ----
        ggml_tensor* img =
                ggml_div(gctx, ggml_sub(gctx, in_images, in_mean), in_std);
        out_patches = backbone_vggt(gctx, img);
        ggml_set_output(out_patches);

        // ---- aggregator + camera/depth/point heads in ONE graph ----
        std::vector<ggml_tensor*> inter;
        aggregator_vggt(gctx, out_patches, inter);
        if (inter.size() != 4) {
            log_error("expected 4 cached layers, got %zu", inter.size());
            return false;
        }
        for (int i = 0; i < 4; i++) {
#ifdef MAP_ANYTHING_ENABLE_DUMP
            ggml_set_output(inter[i]);
            if (MAP_DUMP_ACTIVE()) {
                dump_node(inter[i],
                          "vg_inter" +
                                  std::to_string(m->meta.cached_layer_idx[i]));
            }
#endif
            inter_nodes.push_back(inter[i]);
        }
        out_last = inter.back();
#ifdef MAP_ANYTHING_ENABLE_DUMP
        ggml_set_output(out_last);
#endif
        out_pose = camera_head_vggt(gctx, inter[3]);
        ggml_tensor *pts = nullptr, *ptsc = nullptr;
        dpt_vggt(gctx, inter, "dp", 2, &out_depth, &out_depthc);
        dpt_vggt(gctx, inter, "pt", 4, &pts, &ptsc);
        // activations: depth = exp(raw), conf = 1 + exp(raw); points =
        // sign(y)*(expm1(|y|)), conf = 1 + exp(raw)
        out_depth = ggml_exp(gctx, out_depth);
        out_depthc = ggml_add(gctx, ggml_exp(gctx, out_depthc), in_one);
        out_pts = ggml_mul(gctx, ggml_sgn(gctx, pts),
                           ggml_expm1(gctx, ggml_abs(gctx, pts)));
        out_ptsc = ggml_add(gctx, ggml_exp(gctx, ptsc), in_one);
        ggml_set_output(out_pose);
        ggml_set_output(out_depth);
        ggml_set_output(out_depthc);
        ggml_set_output(out_pts);
        ggml_set_output(out_ptsc);

        const size_t kNodesV = 45000;  // matches the build_graph() budget
        graph = ggml_new_graph_custom(gctx, kNodesV, false);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (dbg_all_tok) ggml_build_forward_expand(graph, dbg_all_tok);
#endif
        ggml_build_forward_expand(graph, out_patches);
        for (int i = 0; i < 4; i++)
            ggml_build_forward_expand(graph, inter_nodes[i]);
        ggml_build_forward_expand(graph, out_last);
        ggml_build_forward_expand(graph, out_pose);
        ggml_build_forward_expand(graph, out_depth);
        ggml_build_forward_expand(graph, out_depthc);
        ggml_build_forward_expand(graph, out_pts);
        ggml_build_forward_expand(graph, out_ptsc);
        galloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(be.handle));
        // give the static inputs their own buffer BEFORE the graph pool is
        // allocated, so gallocr skips them (they must survive across runs)
        ibuf = ggml_backend_alloc_ctx_tensors(ictx, be.handle);
        if (!ibuf) return false;
        if (!alloc_graph("vggt")) return false;
        log_info("vggt: graph allocation done, uploading static inputs");
        ggml_backend_tensor_set(in_mean, m->meta.img_mean, 0,
                                3 * sizeof(float));
        ggml_backend_tensor_set(in_std, m->meta.img_std, 0, 3 * sizeof(float));
        ggml_backend_tensor_set(in_cos_v, host_cos_v.data(), 0,
                                host_cos_v.size() * sizeof(float));
        ggml_backend_tensor_set(in_sin_v, host_sin_v.data(), 0,
                                host_sin_v.size() * sizeof(float));
        ggml_backend_tensor_set(in_pos_embed, host_pos_embed.data(), 0,
                                host_pos_embed.size() * sizeof(float));
        const float one = 1.0f;
        ggml_backend_tensor_set(in_one, &one, 0, sizeof(float));
        for (int i = 0; i < 4; i++) {
            ggml_backend_tensor_set(uv_tap[i], host_uv_tap[i].data(), 0,
                                    host_uv_tap[i].size() * sizeof(float));
        }
        ggml_backend_tensor_set(uv_fused, host_uv_fused.data(), 0,
                                host_uv_fused.size() * sizeof(float));
        log_info("graph built: %d nodes (vggt: backbone+aggregator+3 heads)",
                 ggml_graph_n_nodes(graph));
        return true;
    }

    bool run(const float* imgs, VGGTOutputs& out) override {
        if (!graph) return false;
        auto now = []() { return std::chrono::steady_clock::now(); };
        auto ms_between = [](std::chrono::steady_clock::time_point a,
                             std::chrono::steady_clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        out.timing_ms["build_graphs"] = build_ms;
        const auto t0 = now();
        ggml_backend_tensor_set(in_images, imgs, 0,
                                (size_t)S * 3 * H * W * sizeof(float));
        ggml_backend_graph_compute(be.handle, graph);
        const auto t2 = now();
        out.timing_ms["inference_total"] = ms_between(t0, t2);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        for (size_t i = 0; i < dbg_nodes.size(); i++) {
            std::vector<float> tmp(ggml_nelements(dbg_nodes[i]));
            ggml_backend_tensor_get(dbg_nodes[i], tmp.data(), 0,
                                    tmp.size() * sizeof(float));
            FILE* fp = fopen(dbg_names[i].c_str(), "wb");
            if (fp) {
                fwrite(tmp.data(), 4, tmp.size(), fp);
                fclose(fp);
            }
        }
        for (size_t i = 0; i < dbg_nodes_late.size(); i++) {
            std::vector<float> tmp(ggml_nelements(dbg_nodes_late[i]));
            ggml_backend_tensor_get(dbg_nodes_late[i], tmp.data(), 0,
                                    tmp.size() * sizeof(float));
            FILE* fp = fopen(dbg_names_late[i].c_str(), "wb");
            if (fp) {
                fwrite(tmp.data(), 4, tmp.size(), fp);
                fclose(fp);
            }
        }
        if (dbg_all_tok) {
            std::vector<float> tmp(ggml_nelements(dbg_all_tok));
            ggml_backend_tensor_get(dbg_all_tok, tmp.data(), 0,
                                    tmp.size() * sizeof(float));
            FILE* fp =
                    fopen((std::string(MAP_DUMP_DIR()) + "/vg_all_tokens.bin")
                                  .c_str(),
                          "wb");
            if (fp) {
                fwrite(tmp.data(), 4, tmp.size(), fp);
                fclose(fp);
            }
        }
#endif
        out.pose_enc.resize((size_t)9 * S);
        ggml_backend_tensor_get(out_pose, out.pose_enc.data(), 0,
                                (size_t)9 * S * sizeof(float));
        auto fetch_perm = [&](ggml_tensor* t, int ch, std::vector<float>& dst) {
            // dense outputs live at the PATCH-ALIGNED resolution
            // (Hp*ps, Wp*ps), not the input resolution
            const int Ho = Hp * m->meta.patch_size;
            const int Wo = Wp * m->meta.patch_size;
            out.out_h = Ho;
            out.out_w = Wo;
            std::vector<float> tmp((size_t)Wo * Ho * ch * S);
            ggml_backend_tensor_get(t, tmp.data(), 0,
                                    tmp.size() * sizeof(float));
            dst.resize((size_t)S * Ho * Wo * ch);
            for (int s = 0; s < S; s++) {
                for (int h = 0; h < Ho; h++) {
                    for (int w = 0; w < Wo; w++) {
                        const size_t src = (size_t)w + (size_t)Wo * h +
                                           (size_t)Wo * Ho * ch * s;
                        for (int c = 0; c < ch; c++) {
                            dst[(((size_t)s * Ho + h) * Wo + w) * ch + c] =
                                    tmp[src + (size_t)Wo * Ho * c];
                        }
                    }
                }
            }
        };
        fetch_perm(out_depth, 1, out.depth);
        fetch_perm(out_depthc, 1, out.depth_conf);
        fetch_perm(out_pts, 3, out.world_points);
        fetch_perm(out_ptsc, 1, out.world_points_conf);
        return true;
    }
};

// -----------------------------------------------------------------
// The Pi3 builder (yyfz233/Pi3, general.architecture="pi3").
// Reuses the VGGT builder's DINOv2 encoder path (backbone_vggt: identical
// architecture — cls + 4 registers + patches, no rope, no qk-norm, eps
// 1e-6, LayerScale) and its rope machinery (rope100, (y+1, x+1) patches,
// 0 for specials). Differences, grounded in the official sources:
//   * no camera token: 5 register tokens are prepended at the DECODER
//     entry (dec.reg_token); the 36 decoder blocks alternate per-frame
//     (even) and global (odd) attention, all with qk-norm + LayerScale;
//   * three parallel TransformerDecoder heads (project + 5 blocks, no
//     qk-norm, no LayerScale) reading the concat of the LAST TWO decoder
//     outputs: local points (LinearPts3d: linear + pixel-shuffle 14,
//     z = exp(z), local = [xy*z, z]), conf (raw logits; sigmoid gives
//     probabilities), camera (fc_t/fc_rot; the host does the marepo-style
//     SVD orthogonalization and the global-point unprojection);
//   * ImageNet mean/std normalization (buffers, applied first).
static void pi3_svd_pose(const float* raw12, float* pose16) {
    // marepo svd_orthogonalize: rows of the 3x3 are L2-normalized,
    // transposed, SVD'd; R = V diag(1, 1, det(V U^T)) U^T.
    // The SVD uses one-sided (Hestenes) Jacobi on the columns of A = m_t:
    // numerically robust and matches torch.svd to fp precision.
    double a[3][3];  // A = m_t: a[r][c], rows r = 0..2
    for (int i = 0; i < 3; i++) {
        double n = 0.0;
        for (int j = 0; j < 3; j++)
            n += (double)raw12[3 + i * 3 + j] * raw12[3 + i * 3 + j];
        n = std::sqrt(n);
        for (int j = 0; j < 3; j++) a[j][i] = raw12[3 + i * 3 + j] / n;
    }
    double v[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int sweep = 0; sweep < 30; sweep++) {
        double off = 0.0;
        for (int p = 0; p < 3; p++) {
            for (int q = p + 1; q < 3; q++) {
                const double ap = a[0][p] * a[0][p] + a[1][p] * a[1][p] +
                                  a[2][p] * a[2][p];
                const double aq = a[0][q] * a[0][q] + a[1][q] * a[1][q] +
                                  a[2][q] * a[2][q];
                const double alpha = a[0][p] * a[0][q] + a[1][p] * a[1][q] +
                                     a[2][p] * a[2][q];
                if (ap == 0.0 || aq == 0.0) continue;
                off = std::max(off, std::fabs(alpha) / std::sqrt(ap * aq));
                if (std::fabs(alpha) <= 1e-15 * std::sqrt(ap * aq)) continue;
                const double zeta = (aq - ap) / (2.0 * alpha);
                const double t =
                        (zeta >= 0 ? 1.0 : -1.0) /
                        (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double sn = c * t;
                for (int i = 0; i < 3; i++) {
                    const double acp = a[i][p], acq = a[i][q];
                    a[i][p] = c * acp - sn * acq;
                    a[i][q] = sn * acp + c * acq;
                    const double vp = v[i][p], vq = v[i][q];
                    v[i][p] = c * vp - sn * vq;
                    v[i][q] = sn * vp + c * vq;
                }
            }
        }
        if (off < 1e-14) break;
    }
    // sigma = column norms of a; sort descending (v columns follow)
    double sig[3];
    for (int j = 0; j < 3; j++)
        sig[j] = std::sqrt(a[0][j] * a[0][j] + a[1][j] * a[1][j] +
                           a[2][j] * a[2][j]);
    for (int i = 0; i < 3; i++)
        for (int j = i + 1; j < 3; j++)
            if (sig[j] > sig[i]) {
                std::swap(sig[i], sig[j]);
                for (int k = 0; k < 3; k++) std::swap(a[k][i], a[k][j]);
                for (int k = 0; k < 3; k++) std::swap(v[k][i], v[k][j]);
            }
    // U = (A V) Sigma^-1: after the sweeps the rotated a already IS A*V,
    // so u is just the normalized columns of a (no extra V factor).
    double u[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) u[i][j] = a[i][j] / std::max(sig[j], 1e-30);
    // M = V U^T; det via cofactors
    auto mm = [&](int i, int j) {
        double s2 = 0.0;
        for (int k = 0; k < 3; k++) s2 += v[k][i] * u[j][k];
        return s2;
    };
    const double det = mm(0, 0) * (mm(1, 1) * mm(2, 2) - mm(1, 2) * mm(2, 1)) -
                       mm(0, 1) * (mm(1, 0) * mm(2, 2) - mm(1, 2) * mm(2, 0)) +
                       mm(0, 2) * (mm(1, 0) * mm(2, 1) - mm(1, 1) * mm(2, 0));
    // R = V' U^T with the det-corrected last column of V:
    // R[i][j] = sum_k V'[i][k] * U[j][k]
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double s2 = 0.0;
            for (int k = 0; k < 3; k++) {
                const double vik = v[i][k] * (k == 2 ? det : 1.0);
                s2 += vik * u[j][k];
            }
            pose16[i * 4 + j] = (float)s2;
        }
    pose16[0 * 4 + 3] = raw12[0];
    pose16[1 * 4 + 3] = raw12[1];
    pose16[2 * 4 + 3] = raw12[2];
    pose16[3 * 4 + 0] = pose16[3 * 4 + 1] = pose16[3 * 4 + 2] = 0.0f;
    pose16[3 * 4 + 3] = 1.0f;
}

// -----------------------------------------------------------------
// The Pi3 builder (yyfz233/Pi3, general.architecture="pi3").
// Reuses the VGGT builder's DINOv2 encoder path (backbone_vggt: identical
// architecture — cls + 4 registers + patches, no rope, no qk-norm, eps
// 1e-6, LayerScale) and its rope machinery (rope100, (y+1, x+1) patches,
// 0 for specials). Differences, grounded in the official sources:
//   * no camera token: 5 register tokens are prepended at the DECODER
//     entry (dec.reg_token); the 36 decoder blocks alternate per-frame
//     (even) and global (odd) attention, all with qk-norm + LayerScale;
//   * three parallel TransformerDecoder heads (project + 5 blocks, no
//     qk-norm, no LayerScale) reading the concat of the LAST TWO decoder
//     outputs: local points (LinearPts3d: linear + pixel-shuffle 14,
//     z = exp(z), local = [xy*z, z]), conf (raw logits; sigmoid gives
//     probabilities), camera (fc_t/fc_rot; the host does the marepo-style
//     SVD orthogonalization and the global-point unprojection);
//   * ImageNet mean/std normalization (buffers, applied first).
static void pi3_svd_pose(const float* raw12, float* pose16);
struct VGGTRuntime::Pi3Impl : VGGTRuntime::VGGTImpl {
    ggml_tensor* out_local = nullptr;     // {W, H, 3, S} local point map
    ggml_tensor* out_confp = nullptr;     // {W, H, 1, S} raw conf logits
    ggml_tensor* out_pose_raw = nullptr;  // {12, 1, S} fc_t | fc_rot

    // TransformerDecoder: Linear project + head_depth BlockRope + Linear out.
    // Per-frame layout ({D, N, S}, T = N, B = S).
    ggml_tensor* transformer_decoder(ggml_context* ctx,
                                     ggml_tensor* x,
                                     const std::string& p,
                                     int out_dim) const {
        x = lin(ctx, x, w((p + ".projects.weight").c_str()),
                w((p + ".projects.bias").c_str()));
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) dump_node(x, "vg_" + p + "_proj");
#endif
        for (int i = 0; i < m->meta.head_depth; i++) {
            x = block_v(ctx, x, p + ".blocks." + std::to_string(i),
                        m->meta.embed_dim, false, in_cos_v, in_sin_v, Nagg, S,
                        kEpsLN, false);
#ifdef MAP_ANYTHING_ENABLE_DUMP
            if (MAP_DUMP_ACTIVE() && p == "pdec" && i == 0)
                dump_node(x, "vg_pdec_blk0");
            if (MAP_DUMP_ACTIVE() && p == "pdec" && i == 4)
                dump_node(x, "vg_pdec_blk4");
#endif
        }
        return lin(ctx, x, w((p + ".linear_out.weight").c_str()),
                   w((p + ".linear_out.bias").c_str()));
    }

    // LinearPts3d: slice the patch tokens (skip the 5 registers), linear
    // proj, pixel-shuffle 14. x {D, N, S} -> {W, H, output_dim, S}.
    ggml_tensor* linear_pts3d(ggml_context* ctx,
                              ggml_tensor* x,
                              const std::string& p,
                              int output_dim) const {
        const int D = m->meta.embed_dim;
        // slice the 5 DECODER register tokens (Ncr = 1 + encoder regs = 5,
        // which equals the decoder register count for this architecture)
        x = ggml_cont(gctx, ggml_view_3d(gctx, x, D, P, S, x->nb[1], x->nb[2],
                                         (size_t)Ncr * D * sizeof(float)));
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE() && output_dim == 3) dump_node(x, "vg_pts3d_view");
#endif
        x = lin(ctx, x, w((p + ".proj.weight").c_str()),
                w((p + ".proj.bias").c_str()));  // {od*196, P, S}
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE() && output_dim == 3) dump_node(x, "vg_pts3d_lin");
#endif
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));  // {P, od*196, S}
        // p = h*Wp + w is the fastest axis, so the reshape splits it as
        // (w, h) — matching torch's view(B, od*196, Hp, Wp) before the
        // shuffle (channel slowest per pixel).
        x = ggml_reshape_4d(ctx, x, Wp, Hp, output_dim * 196, S);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE() && output_dim == 3)
            dump_node(x, "vg_pts3d_reshaped");
#endif
        // pixel_shuffle handles a single output channel (c2 == r*r); split
        // the od output channels and shuffle each independently. x is
        // {Wp, Hp, od*196, S}, so the {Wp, Hp, 196, S} view takes the
        // (nb1, nb2, nb3) stride slots from x's nb[1..3] (ne[0] strides are
        // always 4B and ne[2] steps one pixel-block).
        ggml_tensor* out = nullptr;
        for (int ch = 0; ch < output_dim; ch++) {
            ggml_tensor* g =
                    ggml_view_4d(ctx, x, Wp, Hp, 196, S, x->nb[1], x->nb[2],
                                 x->nb[3], ch * 196 * x->nb[2]);
            g = pixel_shuffle(ctx, ggml_cont(ctx, g), 14);  // {W, H, 1, S}
            out = (ch == 0) ? g : ggml_concat(ctx, out, g, 2);
        }
        return out;  // {W, H, od, S}
    }

    // CameraHead: 2 ResConvBlocks (linear relu x3 + skip), spatial average
    // pool, 2-layer MLP, then the raw fc_t (3) | fc_rot (9) stack. The
    // official call site slices the register tokens off BEFORE the head, so
    // the average pool covers the patch grid only.
    ggml_tensor* camera_head_pi3(ggml_context* ctx, ggml_tensor* x) {
        x = ggml_cont(gctx,
                      ggml_view_3d(gctx, x, x->ne[0], P, S, x->nb[1], x->nb[2],
                                   (size_t)Ncr * x->ne[0] * sizeof(float)));
        for (int i = 0; i < 2; i++) {
            const std::string p =
                    "mhead.res_conv." + std::to_string(i) + ".res_conv";
            ggml_tensor* res = x;
            ggml_tensor* a =
                    ggml_relu(ctx, lin(ctx, x, w((p + "1.weight").c_str()),
                                       w((p + "1.bias").c_str())));
            a = ggml_relu(ctx, lin(ctx, a, w((p + "2.weight").c_str()),
                                   w((p + "2.bias").c_str())));
            a = ggml_relu(ctx, lin(ctx, a, w((p + "3.weight").c_str()),
                                   w((p + "3.bias").c_str())));
            x = ggml_add(ctx, res, a);
        }
        // AdaptiveAvgPool2d(1) over the (Hp, Wp) grid. {512, N, S} already
        // has n = h*Wp + w as the fastest axis, so the reshape splits it
        // as (w, h); the pool needs (w, h) on ne0/ne1 -> permute.
        x = ggml_reshape_4d(ctx, x, 512, Wp, Hp, S);
        // ggml permute axes = DESTINATION positions: old0 (512) -> ne2,
        // old1 (Wp) -> ne0, old2 (Hp) -> ne1
        x = ggml_cont(ctx,
                      ggml_permute(ctx, x, 2, 0, 1, 3));  // {Wp, Hp, 512, S}
        x = ggml_pool_2d(ctx, x, GGML_OP_POOL_AVG, Wp, Hp, Wp, Hp, 0.0f, 0.0f);
        x = ggml_reshape_3d(ctx, x, 512, 1, S);
        x = ggml_relu(ctx, lin(ctx, x, w("mhead.more_mlps.0.weight"),
                               w("mhead.more_mlps.0.bias")));
        x = ggml_relu(ctx, lin(ctx, x, w("mhead.more_mlps.2.weight"),
                               w("mhead.more_mlps.2.bias")));
        ggml_tensor* t = lin(ctx, x, w("mhead.fc_t.weight"),
                             w("mhead.fc_t.bias"));  // {3, 1, S}
        ggml_tensor* r = lin(ctx, x, w("mhead.fc_rot.weight"),
                             w("mhead.fc_rot.bias"));  // {9, 1, S}
        return ggml_concat(ctx, t, r, 0);              // {12, 1, S}
    }

    bool build_graph() override {
        // rope rows (y+1, x+1) reused verbatim; pi3's DINOv2 keeps the
        // torch-hub interpolation defaults (antialias=false, offset=0.1)
        prepare_host_inputs_vggt(false, 0.1f);
        const size_t kNodes = 60000;
        ggml_init_params gp{ggml_tensor_overhead() * kNodes +
                                    ggml_graph_overhead_custom(kNodes, false),
                            nullptr, true};
        gctx = ggml_init(gp);
        if (!gctx) return false;
        return build_stage1_pi3();
    }

    bool build_stage1_pi3() {
        log_info("pi3: creating graph inputs");
        // static inputs get their OWN backend buffer (see build_stage1_vggt
        // for why: the gallocr pool recycles their space mid-graph after the
        // last use, so "upload once" would not survive a second run())
        ictx = ggml_init({ggml_tensor_overhead() * 64, nullptr, true});
        if (!ictx) return false;
        in_images = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, W, H, 3, S);
        ggml_set_input(in_images);
        in_mean = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, 1, 1, 3, 1);
        in_std = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, 1, 1, 3, 1);
        const int dh_l = C / nh;
        in_cos_v = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, dh_l, nh, Nagg, S);
        in_sin_v = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, dh_l, nh, Nagg, S);
        ggml_set_input(in_cos_v);
        ggml_set_input(in_sin_v);
        in_pos_embed = ggml_new_tensor_3d(ictx, GGML_TYPE_F32, C, 1 + P, 1);
        ggml_set_input(in_pos_embed);
        // (no in_one: pi3's conf head outputs raw logits, nothing adds 1)

        // ---- encoder (DINOv2, reused from the VGGT builder) ----
        ggml_tensor* img =
                ggml_div(gctx, ggml_sub(gctx, in_images, in_mean), in_std);
        out_patches = backbone_vggt(gctx, img);  // {C, P, S}
        ggml_set_output(out_patches);

        // ---- decoder entry: [reg(5) | patches] ----
        ggml_tensor* reg = w("dec.reg_token");  // {C, 5, 1, 1}
        ggml_tensor* reg_all = ggml_repeat(
                gctx,
                ggml_view_3d(gctx, reg, C, 5, 1, reg->nb[1], reg->nb[2], 0),
                ggml_new_tensor_3d(gctx, GGML_TYPE_F32, C, 5, S));
        ggml_tensor* tokens = ggml_concat(gctx, reg_all, out_patches, 1);

        // ---- 36 alternating blocks (even per-frame, odd global) ----
        ggml_tensor* h_penult = nullptr;
        for (int i = 0; i < m->meta.aa_depth; i++) {
            ggml_tensor* x;
            int T, B;
            if (i % 2 == 0) {
                x = tokens;
                T = Nagg;
                B = S;
            } else {
                x = ggml_reshape_3d(gctx, tokens, C, Nagg * S, 1);
                T = Nagg * S;
                B = 1;
            }
            x = block_v(gctx, x, "dec." + std::to_string(i), C, true, in_cos_v,
                        in_sin_v, T, B);
            if (i % 2 == 1) x = ggml_reshape_3d(gctx, x, C, Nagg, S);
            tokens = x;
            if (i == m->meta.aa_depth - 2) h_penult = tokens;
#ifdef MAP_ANYTHING_ENABLE_DUMP
            if (MAP_DUMP_ACTIVE() && i <= 4)
                dump_node(tokens, ("vg_dec" + std::to_string(i)).c_str());
#endif
        }
        ggml_tensor* hidden2 = ggml_concat(gctx, h_penult, tokens, 0);
        ggml_set_output(hidden2);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) dump_node(hidden2, "vg_hidden2");
#endif

        // ---- three parallel heads ----
        ggml_tensor* pt = transformer_decoder(gctx, hidden2, "pdec", 1024);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) {
            dump_node(pt, "vg_pdec_out");
            dump_node(pt, "vg_pdec_out2");
        }
#endif
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) {
            dump_node(pt, "vg_pdec_out");
            dump_node(pt, "vg_pdec_out2");
        }
#endif
        ggml_tensor* local_raw = linear_pts3d(gctx, pt, "phead", 3);
        // z = exp(z); local_points = [x*z, y*z, z]. local_raw is
        // {W, H, 3, S}; the {W, H, 1, S} channel views take their stride
        // slots from nb[1..3] (nb[0] is the 4B inner stride).
        ggml_tensor* zc = ggml_cont(
                gctx, ggml_view_4d(gctx, local_raw, local_raw->ne[0],
                                   local_raw->ne[1], 1, S, local_raw->nb[1],
                                   local_raw->nb[2], local_raw->nb[3],
                                   2 * local_raw->nb[2]));
        zc = ggml_exp(gctx, zc);
        ggml_tensor* xc = ggml_cont(
                gctx, ggml_view_4d(gctx, local_raw, local_raw->ne[0],
                                   local_raw->ne[1], 1, S, local_raw->nb[1],
                                   local_raw->nb[2], local_raw->nb[3], 0));
        ggml_tensor* yc = ggml_cont(
                gctx, ggml_view_4d(gctx, local_raw, local_raw->ne[0],
                                   local_raw->ne[1], 1, S, local_raw->nb[1],
                                   local_raw->nb[2], local_raw->nb[3],
                                   local_raw->nb[2]));
        out_local = ggml_concat(gctx,
                                ggml_concat(gctx, ggml_mul(gctx, xc, zc),
                                            ggml_mul(gctx, yc, zc), 2),
                                zc, 2);  // {W, H, 3, S}

        ggml_tensor* cf = transformer_decoder(gctx, hidden2, "cdec", 1024);
        out_confp = linear_pts3d(gctx, cf, "chead", 1);  // {W, H, 1, S}

        ggml_tensor* cm = transformer_decoder(gctx, hidden2, "mdec", 512);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) dump_node(cm, "pi3x_mdec_out");
#endif
        out_pose_raw = camera_head_pi3(gctx, cm);  // {12, 1, S}

        ggml_set_output(out_local);
        ggml_set_output(out_confp);
        ggml_set_output(out_pose_raw);

        const size_t kNodesV = 60000;
        graph = ggml_new_graph_custom(gctx, kNodesV, false);
        ggml_build_forward_expand(graph, out_patches);
        ggml_build_forward_expand(graph, hidden2);
        ggml_build_forward_expand(graph, out_local);
        ggml_build_forward_expand(graph, out_confp);
        ggml_build_forward_expand(graph, out_pose_raw);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        // keep the parity-dump tensors alive: without an expand the gallocr
        // treats them as reusable intermediates and the dumps read garbage
        for (size_t i = 0; i < dbg_nodes.size(); i++)
            ggml_build_forward_expand(graph, dbg_nodes[i]);
#endif
        galloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(be.handle));
        // give the static inputs their own buffer BEFORE the graph pool is
        // allocated, so gallocr skips them (they must survive across runs)
        ibuf = ggml_backend_alloc_ctx_tensors(ictx, be.handle);
        if (!ibuf) return false;
        if (!alloc_graph("pi3")) return false;
        log_info("pi3: graph allocation done, uploading static inputs");
        ggml_backend_tensor_set(in_mean, m->meta.img_mean, 0,
                                3 * sizeof(float));
        ggml_backend_tensor_set(in_std, m->meta.img_std, 0, 3 * sizeof(float));
        ggml_backend_tensor_set(in_cos_v, host_cos_v.data(), 0,
                                host_cos_v.size() * sizeof(float));
        ggml_backend_tensor_set(in_sin_v, host_sin_v.data(), 0,
                                host_sin_v.size() * sizeof(float));
        ggml_backend_tensor_set(in_pos_embed, host_pos_embed.data(), 0,
                                host_pos_embed.size() * sizeof(float));
        log_info("graph built: %d nodes (pi3: encoder+decoder+3 heads)",
                 ggml_graph_n_nodes(graph));
        return true;
    }

    bool run(const float* imgs, VGGTOutputs& out) override {
        if (!graph) return false;
        auto now = []() { return std::chrono::steady_clock::now(); };
        auto ms_between = [](std::chrono::steady_clock::time_point a,
                             std::chrono::steady_clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        out.timing_ms["build_graphs"] = build_ms;
        const auto t0 = now();
        ggml_backend_tensor_set(in_images, imgs, 0,
                                (size_t)S * 3 * H * W * sizeof(float));
        ggml_backend_graph_compute(be.handle, graph);
        const auto t2 = now();
        out.timing_ms["inference_total"] = ms_between(t0, t2);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        for (size_t i = 0; i < dbg_nodes.size(); i++) {
            std::vector<float> tmp(ggml_nelements(dbg_nodes[i]));
            ggml_backend_tensor_get(dbg_nodes[i], tmp.data(), 0,
                                    tmp.size() * sizeof(float));
            FILE* fp = fopen(dbg_names[i].c_str(), "wb");
            if (fp) {
                fwrite(tmp.data(), 4, tmp.size(), fp);
                fclose(fp);
            }
        }
#endif
        // dense outputs are patch-aligned (Wp*14, Hp*14)
        const int Wo = Wp * m->meta.patch_size;
        const int Ho = Hp * m->meta.patch_size;
        out.out_h = Ho;
        out.out_w = Wo;
        auto fetch_perm = [&](ggml_tensor* t, int ch, std::vector<float>& dst) {
            std::vector<float> tmp((size_t)Wo * Ho * ch * S);
            ggml_backend_tensor_get(t, tmp.data(), 0,
                                    tmp.size() * sizeof(float));
            dst.resize((size_t)S * Ho * Wo * ch);
            for (int s = 0; s < S; s++) {
                for (int h = 0; h < Ho; h++) {
                    for (int w = 0; w < Wo; w++) {
                        const size_t src = (size_t)w + (size_t)Wo * h +
                                           (size_t)Wo * Ho * ch * s;
                        for (int c = 0; c < ch; c++) {
                            dst[(((size_t)s * Ho + h) * Wo + w) * ch + c] =
                                    tmp[src + (size_t)Wo * Ho * c];
                        }
                    }
                }
            }
        };
        fetch_perm(out_local, 3, out.local_points);
        fetch_perm(out_confp, 1, out.conf);

        // pi3 has no dedicated depth head: the C API contract (result_depth,
        // aicore/map_anything_capi.h) defines the dense depth as the
        // camera-frame z of the local pointmap. Synthesize it so every
        // architecture surfaces the same typed output shape.
        out.depth.resize((size_t)S * Ho * Wo);
        for (size_t i = 0; i < out.depth.size(); ++i)
            out.depth[i] = out.local_points[3 * i + 2];

        // raw camera params -> host SVD orthogonalization + 4x4 poses
        out.pose_raw.resize((size_t)12 * S);
        ggml_backend_tensor_get(out_pose_raw, out.pose_raw.data(), 0,
                                (size_t)12 * S * sizeof(float));
        out.camera_poses.assign((size_t)16 * S, 0.0f);
        for (int s = 0; s < S; s++) {
            float pose16[16];
            pi3_svd_pose(&out.pose_raw[(size_t)12 * s], pose16);
            for (int i = 0; i < 16; i++)
                out.camera_poses[(size_t)16 * s + i] = pose16[i];
        }
        // global points: camera_poses @ homogenize(local_points)
        out.world_points.assign((size_t)S * Ho * Wo * 3, 0.0f);
        for (int s = 0; s < S; s++) {
            const float* R = &out.camera_poses[(size_t)16 * s];
            for (int h = 0; h < Ho; h++) {
                for (int w = 0; w < Wo; w++) {
                    const size_t o = (((size_t)s * Ho + h) * Wo + w) * 3;
                    const float x = out.local_points[o];
                    const float y = out.local_points[o + 1];
                    const float z = out.local_points[o + 2];
                    for (int r = 0; r < 3; r++) {
                        out.world_points[o + r] =
                                R[r * 4 + 0] * x + R[r * 4 + 1] * y +
                                R[r * 4 + 2] * z + R[r * 4 + 3];
                    }
                }
            }
        }
        return true;
    }
};

static RegisterBuilder g_reg_pi3("pi3", [] {
    return std::unique_ptr<IGraphBuilder>(new VGGTRuntime::Pi3Impl());
});

// -----------------------------------------------------------------
// The MapAnything builder (facebook/map-anything,
// general.architecture="mapanything"). Grounded in the official sources:
//   * encoder: DINOv2 ViT-G/14 (1536, 24 heads), FIRST 24 blocks, swiglu
//     FFN, LayerScale 1e-5, LN eps 1e-6, NO final norm (the model sets
//     norm_returned_features=false -> norm=Identity), registers=None ->
//     the CLS token (1 per view) becomes the per-view additional token;
//   * AAT: 16 SelfAttentionBlocks (swiglu, ls 1e-5, qk_norm=False, NO
//     rope), EVEN = global attention over [view-major tokens | scale
//     token], ODD = frame attention with the scale token REMOVED (it only
//     participates in global); reference view (idx 0) tokens (incl. cls)
//     get is_view_pe row 0 added (a deterministic sin/cos buffer);
//   * IFR at blocks 7/11 (post-block, through the final LN) + final LN:
//     DPT consumes [enc_final, IFR0, IFR1, final]; pose head consumes
//     final; scale head consumes the final scale-token feature;
//   * DPT: dust3r-style input_process (1x1 conv + ConvTranspose 4x/2x +
//     stride-2 conv + 3x3 rn), refinenet chain with a FIXED 2x bilinear
//     (align_corners) after every stage, regressor conv3x3 -> bilinear
//     resize to (W,H) -> conv3x3+relu+conv1x1 -> 6 channels
//     [raydirs(3) | depth(1) | conf(1) | mask(1)];
//   * activations: raydirs / ||v||.clip(1e-8) (unit sphere), depth = exp,
//     conf = 1 + exp (vmin=1), mask = sigmoid; pose head: proj 1x1 -> 2
//     ResConvBlocks (1x1, residual base = input) -> spatial avgpool ->
//     more_mlps -> fc_t(3) + fc_rot(4 quaternion, L2-normalized on host);
//     scale head: MLPHead(1536->196->196->196->1) -> exp (host).
struct VGGTRuntime::MapAnythingImpl final : VGGTRuntime::VGGTImpl {
    int Nv = 0;  // tokens per view = P + 1 (the cls register)

    ggml_tensor* out_rays = nullptr;  // {W, H, 3, S} raw (unit-sphere on host)
    ggml_tensor* out_dense =
            nullptr;  // {W, H, 3, S} [depth_log, conf_log, mask_log]
    ggml_tensor* out_pose_raw = nullptr;   // {7, 1, S} t(3) | quat(4)
    ggml_tensor* out_scale_raw = nullptr;  // {1, 1, 1}

    // DINOv2 encoder: [cls | patches] + pos_embed -> 24 swiglu blocks ->
    // (no final norm). Returns the full {C, 1+P, S} sequence.
    ggml_tensor* backbone_ma(ggml_context* ctx, ggml_tensor* img) {
        const int ps = m->meta.patch_size;
        ggml_tensor* x = mapany_conv_2d(ctx, w("bb.patch_embed.proj.weight"),
                                        img, ps, ps, 0, 0, 1, 1);
        x = ggml_add(ctx, x, bias4(ctx, w("bb.patch_embed.proj.bias"), C));
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
        x = ggml_reshape_3d(ctx, x, C, P, S);
        ggml_tensor* cls =
                ggml_repeat(ctx, w("bb.cls_token"),
                            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, 1, S));
        ggml_tensor* pc = ggml_concat(ctx, cls, x, 1);  // {C, 1+P, S}
        pc = ggml_add(ctx, pc, in_pos_embed);
        const int Tbb = 1 + P;  // no encoder registers (unlike vggt's Nbb)
        for (int i = 0; i < m->meta.depth; i++) {
            pc = block_v(ctx, pc, "bb.blocks." + std::to_string(i), C, false,
                         nullptr, nullptr, Tbb, S, 1e-6f, true, true);
        }
        return pc;  // {C, 1+P, S} — cls row 0, patches rows 1..P
    }

    // DPT input_process chain for one hook (torch key indices differ per
    // branch: 0=[conv1x1,convT4,rn3x3] 1=[conv1x1,convT2,rn3x3]
    // 2=[conv1x1,rn3x3] 3=[conv1x1,conv_s2,rn3x3]).
    ggml_tensor* dpt_input_process(ggml_context* ctx,
                                   int branch,
                                   ggml_tensor* x) const {
        const std::string p =
                "dp.input_process." + std::to_string(branch) + ".";
        if (branch == 0 || branch == 1) {
            x = conv1x1(ctx, p + "0.0", x);
            x = ggml_add(ctx,
                         mapany_conv_transpose_2d_p0(
                                 ctx, w((p + "0.1.weight").c_str()), x,
                                 branch == 0 ? 4 : 2),
                         bias4(ctx, w((p + "0.1.bias").c_str()), x->ne[2]));
        } else if (branch == 2) {
            // act_3 is a single-element Sequential -> key 2.0.0
            x = conv1x1(ctx, p + "0.0", x);
        } else {
            // act_4 = Sequential(Conv2d, Conv2d) -> keys 3.0.0 / 3.0.1
            x = conv1x1(ctx, p + "0.0", x);
            x = mapany_conv_2d(ctx, w((p + "0.1.weight").c_str()), x, 2, 2, 1,
                               1, 1, 1);
            x = ggml_add(ctx, x,
                         bias4(ctx, w((p + "0.1.bias").c_str()), x->ne[2]));
        }
        // layer_rn: 3x3 conv, no bias (rn is always the LAST index:
        // 0.1 / 1.1 / 2.1 / 3.1)
        return conv3x3(ctx, p + "1", x, false);
    }

    // tokens {C, P, S} (token-major, p = w + Wp*h) -> conv grid
    // {Wp, Hp, C, S}.
    ggml_tensor* tokens_to_grid(ggml_context* ctx, ggml_tensor* x) const {
        ggml_tensor* g = ggml_reshape_4d(ctx, x, C, Wp, Hp, S);
        return ggml_cont(ctx, ggml_permute(ctx, g, 2, 0, 1, 3));
    }

    bool build_graph() override {
        // pos-embed rows only (no rope, no uv tables); mapanything's DINOv2
        // keeps the torch-hub interpolation defaults (antialias=false,
        // offset=0.1)
        if (!build_pos_embed_rows(host_pos_embed, false, 0.1f)) return false;
        Nv = Ncr + P;       // P + 1 tokens per view (cls register)
        Nagg = S * Nv + 1;  // + the global scale token
        const size_t kNodes = 120000;
        ggml_init_params gp{ggml_tensor_overhead() * kNodes +
                                    ggml_graph_overhead_custom(kNodes, false),
                            nullptr, true};
        gctx = ggml_init(gp);
        if (!gctx) return false;
        return build_stage1_mapanything();
    }

    bool build_stage1_mapanything() {
        log_info("mapanything: creating graph inputs");
        // static inputs get their OWN backend buffer (see build_stage1_vggt
        // for why: the gallocr pool recycles their space mid-graph after the
        // last use, so "upload once" would not survive a second run())
        ictx = ggml_init({ggml_tensor_overhead() * 64, nullptr, true});
        if (!ictx) return false;
        in_images = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, W, H, 3, S);
        ggml_set_input(in_images);
        in_mean = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, 1, 1, 3, 1);
        in_std = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, 1, 1, 3, 1);
        in_pos_embed = ggml_new_tensor_3d(ictx, GGML_TYPE_F32, C, 1 + P, 1);
        ggml_set_input(in_pos_embed);

        // ---- encoder (DINOv2 ViT-G, first 24 blocks, no final norm) ----
        ggml_tensor* img =
                ggml_div(gctx, ggml_sub(gctx, in_images, in_mean), in_std);
        ggml_tensor* enc = backbone_ma(gctx, img);  // {C, 1+P, S}
        ggml_set_output(enc);

        // ---- AAT input assembly (view-major) ----
        // per-view [patch | cls] {C, P+1, S}; torch concatenates the
        // per-view register AFTER the flattened spatial tokens.
        ggml_tensor* cls_row = ggml_cont(
                gctx,
                ggml_view_3d(gctx, enc, C, 1, S, enc->nb[1], enc->nb[2], 0));
        ggml_tensor* pat =
                ggml_cont(gctx, ggml_view_3d(gctx, enc, C, P, S, enc->nb[1],
                                             enc->nb[2], C * sizeof(float)));
        // The official forward runs fusion_norm_layer on the encoder features
        // UNCONDITIONALLY (channel-last LN over C), even with images_only
        // inputs and before the cls registers are attached.  pat is {C,P,S}
        // with C on ne0, so ggml_norm normalises exactly that axis.
        pat = ln(gctx, pat, w("fusion_norm_layer.weight"),
                 w("fusion_norm_layer.bias"), 1e-6f);
        ggml_tensor* per_view =
                ggml_concat(gctx, pat, cls_row, 1);  // {C,P+1,S}
        // view-major flatten: {C, S*(P+1), 1} (ne2 is the slow axis)
        ggml_tensor* tokens = ggml_reshape_3d(gctx, per_view, C, S * Nv, 1);
        // reference-view PE: add is_view_pos_table row 0 (the buffer holds
        // only the reference row, torch (1, C) -> ggml ne {C, 1})
        ggml_tensor* ref_seg =
                ggml_cont(gctx, ggml_view_3d(gctx, tokens, C, Nv, 1,
                                             tokens->nb[1], tokens->nb[2], 0));
        ref_seg = ggml_add(gctx, ref_seg,
                           ggml_reshape_3d(gctx, w("is_view_pos_table"), C, 1,
                                           1));  // broadcast over Nv
        ggml_tensor* nonref_seg =
                ggml_cont(gctx, ggml_view_3d(gctx, tokens, C, (S - 1) * Nv, 1,
                                             tokens->nb[1], tokens->nb[2],
                                             (size_t)Nv * C * sizeof(float)));
        ggml_tensor* scale_tok =
                ggml_reshape_3d(gctx, w("is_scale_token"), C, 1, 1);
        tokens = ggml_concat(gctx, ggml_concat(gctx, ref_seg, nonref_seg, 1),
                             scale_tok, 1);  // {C, Nagg, 1}
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) dump_node(tokens, "vg_ma_assembled");
#endif
        // NOTE: AAT block order is the REVERSE of vggt: even = global.
        ggml_tensor* h_penult0 = nullptr;
        ggml_tensor* h_penult1 = nullptr;
        for (int i = 0; i < m->meta.aa_depth; i++) {
            if (i % 2 == 0) {
                tokens = block_v(gctx, tokens, "is." + std::to_string(i), C,
                                 false, nullptr, nullptr, Nagg, 1, 1e-6f, true,
                                 true);
            } else {
                ggml_tensor* scale_now = ggml_view_3d(
                        gctx, tokens, C, 1, 1, tokens->nb[1], tokens->nb[2],
                        (size_t)S * Nv * C * sizeof(float));
                ggml_tensor* f =
                        ggml_view_3d(gctx, tokens, C, S * Nv, 1, tokens->nb[1],
                                     (size_t)S * Nv * C * sizeof(float),
                                     0);  // contiguous prefix
                f = ggml_reshape_3d(gctx, f, C, Nv, S);
                f = block_v(gctx, f, "is." + std::to_string(i), C, false,
                            nullptr, nullptr, Nv, S, 1e-6f, true, true);
                tokens = ggml_reshape_3d(gctx, f, C, S * Nv, 1);
                tokens = ggml_concat(gctx, tokens, scale_now, 1);
            }
#ifdef MAP_ANYTHING_ENABLE_DUMP
            if (MAP_DUMP_ACTIVE() && (i <= 1 || i == 15))
                dump_node(tokens, "vg_ma_is" + std::to_string(i));
#endif
            if (i == m->meta.cached_layer_idx[0]) h_penult0 = tokens;
            if (i == m->meta.cached_layer_idx[1]) h_penult1 = tokens;
        }
        // IFR/final features: LN over the full sequence, then slice the
        // spatial part and drop each view's cls register -> {C, P, S}.
        // Fully-physical implementation: every slice is materialized with
        // cont() so no implicit stride survives across view/reshape chains.
        auto spatial_feat = [&](ggml_tensor* t) {
            t = ln(gctx, t, w("is_norm.weight"), w("is_norm.bias"), 1e-6f);
            // drop the scale token (last of the S*Nv+1 sequence)
            t = ggml_cont(gctx, ggml_view_3d(gctx, t, C, S * Nv, 1, t->nb[1],
                                             t->nb[2], 0));
            // per view: take the first P tokens (drop the cls register)
            std::vector<ggml_tensor*> parts;
            for (int s = 0; s < S; s++) {
                ggml_tensor* vs = ggml_cont(
                        gctx,
                        ggml_view_3d(gctx, t, C, Nv, 1, t->nb[1], t->nb[2],
                                     (size_t)s * Nv * C * sizeof(float)));
                parts.push_back(
                        ggml_cont(gctx, ggml_view_3d(gctx, vs, C, P, 1,
                                                     vs->nb[1], vs->nb[2], 0)));
            }
            ggml_tensor* acc = parts[0];
            for (int s = 1; s < S; s++)
                acc = ggml_concat(gctx, acc, parts[s], 1);  // {C, S*P, 1}
            // cont: the reshape is a VIEW of acc's buffer; without an
            // independent buffer the gallocr overwrites it during the DPT
            // and every downstream consumer (and the parity dump) reads
            // clobbered values.
            return ggml_cont(gctx, ggml_reshape_3d(gctx, acc, C, P, S));
        };
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) dump_node(tokens, "vg_ma_ln_in");
#endif
        // one shared is_norm LayerNorm: the full map feeds the parity dump,
        // the tail view is the final scale-token feature
        ggml_tensor* ln_final =
                ln(gctx, tokens, w("is_norm.weight"), w("is_norm.bias"), 1e-6f);
        // final scale-token feature {C, 1, 1}
        ggml_tensor* scale_feat = ggml_cont(
                gctx, ggml_view_3d(gctx, ln_final, C, 1, 1, C * sizeof(float),
                                   C * sizeof(float),
                                   (size_t)S * Nv * C * sizeof(float)));
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) dump_node(ln_final, "vg_ma_ln_out");
#endif
        ggml_tensor* f_ifr0 = spatial_feat(h_penult0);
        ggml_tensor* f_ifr1 = spatial_feat(h_penult1);
        ggml_tensor* f_final = spatial_feat(tokens);
        ggml_tensor* enc_final = ggml_cont(gctx, pat);  // no norm
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) {
            dump_node(enc, "vg_ma_enc");
            dump_node(f_final, "vg_ma_final");
        }
#endif

        // ---- DPT: [enc_final, ifr0, ifr1, final] (hooks 0..3) ----
        ggml_tensor* taps[4];
        const ggml_tensor* feats[4] = {enc_final, f_ifr0, f_ifr1, f_final};
        for (int i = 0; i < 4; i++) {
            ggml_tensor* x =
                    tokens_to_grid(gctx, const_cast<ggml_tensor*>(feats[i]));
            x = dpt_input_process(gctx, i, x);
            taps[i] = x;
#ifdef MAP_ANYTHING_ENABLE_DUMP
            if (MAP_DUMP_ACTIVE())
                dump_node(taps[i], "vg_ma_tap" + std::to_string(i));
#endif
        }
        // refinenet chain: fixed 2x bilinear (align_corners) after every
        // stage; rn4 output is cropped to the tap2 grid first (torch slice).
        ggml_tensor* o = refinenet(gctx, "dp.scratch.refinenet4", taps[3],
                                   nullptr, 2 * taps[3]->ne[0],
                                   2 * taps[3]->ne[1], nullptr, false);
        o = ggml_cont(gctx, ggml_view_4d(gctx, o, taps[2]->ne[0],
                                         taps[2]->ne[1], o->ne[2], o->ne[3],
                                         o->nb[1], o->nb[2], o->nb[3], 0));
        o = refinenet(gctx, "dp.scratch.refinenet3", o, taps[2], 2 * o->ne[0],
                      2 * o->ne[1], nullptr, false);
        o = refinenet(gctx, "dp.scratch.refinenet2", o, taps[1], 2 * o->ne[0],
                      2 * o->ne[1], nullptr, false);
        o = refinenet(gctx, "dp.scratch.refinenet1", o, taps[0], 2 * o->ne[0],
                      2 * o->ne[1], nullptr, false);
        // regressor: conv3x3 -> bilinear (W,H) -> conv3x3+relu+conv1x1
        o = conv3x3(gctx, "dr.conv1", o, true);
        o = ggml_interpolate(
                gctx, o, W, H, o->ne[2], o->ne[3],
                GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
        o = conv3x3(gctx, "dr.conv2.0", o, true);
        o = ggml_relu(gctx, o);
        o = conv1x1(gctx, "dr.conv2.2", o);  // {W, H, 6, S}
        ggml_set_output(o);
        // channel split: rays(3) | depth_log(1) | conf_log(1) | mask_log(1)
        out_rays =
                ggml_cont(gctx, ggml_view_4d(gctx, o, o->ne[0], o->ne[1], 3, S,
                                             o->nb[1], o->nb[2], o->nb[3], 0));
        out_dense = ggml_cont(
                gctx, ggml_view_4d(gctx, o, o->ne[0], o->ne[1], 3, S, o->nb[1],
                                   o->nb[2], o->nb[3], 3 * o->nb[2]));

        // ---- pose head: proj 1x1 -> 2 ResConv -> avgpool -> MLPs ----
        ggml_tensor* ph = tokens_to_grid(gctx, f_final);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) dump_node(ph, "vg_ma_pgin");
#endif
        ph = conv1x1(gctx, "cam.proj", ph);  // {Wp, Hp, 784, S}
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) dump_node(ph, "vg_ma_pj");
#endif
        for (int i = 0; i < 2; i++) {
            const std::string p = "cam.res_conv." + std::to_string(i);
            ggml_tensor* res = ph;
            ggml_tensor* a =
                    ggml_relu(gctx, conv1x1(gctx, p + ".res_conv1", ph));
            a = ggml_relu(gctx, conv1x1(gctx, p + ".res_conv2", a));
            a = ggml_relu(gctx, conv1x1(gctx, p + ".res_conv3", a));
            ph = ggml_add(gctx, res, a);
        }
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) dump_node(ph, "vg_ma_rc");
#endif
        ph = ggml_pool_2d(gctx, ph, GGML_OP_POOL_AVG, Wp, Hp, Wp, Hp, 0.0f,
                          0.0f);  // {1, 1, 784, S}
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) dump_node(ph, "vg_ma_pl");
#endif
        ph = ggml_reshape_3d(gctx, ph, 784, 1, S);
        ph = ggml_relu(gctx, lin(gctx, ph, w("cam.more_mlps.0.weight"),
                                 w("cam.more_mlps.0.bias")));
        ph = ggml_relu(gctx, lin(gctx, ph, w("cam.more_mlps.2.weight"),
                                 w("cam.more_mlps.2.bias")));
        ggml_tensor* ft = lin(gctx, ph, w("cam.fc_t.weight"),
                              w("cam.fc_t.bias"));  // {3, 1, S}
        ggml_tensor* fr = lin(gctx, ph, w("cam.fc_rot.weight"),
                              w("cam.fc_rot.bias"));  // {4, 1, S}
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) {
            dump_node(ft, "vg_ma_ft");
            dump_node(fr, "vg_ma_fr");
            dump_node(ph, "vg_ma_mlpout2");
        }
#endif
        out_pose_raw = ggml_concat(gctx, ft, fr, 0);  // {7, 1, S}
        ggml_set_output(out_pose_raw);

        // ---- scale head (MLPHead on the scale token) ----
        ggml_tensor* sc = scale_feat;
        sc = lin(gctx, sc, w("scl.proj.weight"), w("scl.proj.bias"));
        sc = ggml_relu(gctx, lin(gctx, sc, w("scl.mlp.0.0.weight"),
                                 w("scl.mlp.0.0.bias")));
        sc = ggml_relu(gctx, lin(gctx, sc, w("scl.mlp.1.0.weight"),
                                 w("scl.mlp.1.0.bias")));
        out_scale_raw = lin(gctx, sc, w("scl.output_proj.weight"),
                            w("scl.output_proj.bias"));  // {1, 1, 1}
        ggml_set_output(out_scale_raw);

        ggml_set_output(out_rays);
        ggml_set_output(out_dense);

        const size_t kNodesV = 120000;
        graph = ggml_new_graph_custom(gctx, kNodesV, false);
        ggml_build_forward_expand(graph, out_rays);
        ggml_build_forward_expand(graph, out_dense);
        ggml_build_forward_expand(graph, out_pose_raw);
        ggml_build_forward_expand(graph, out_scale_raw);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        for (size_t i = 0; i < dbg_nodes.size(); i++)
            ggml_build_forward_expand(graph, dbg_nodes[i]);
#endif
        galloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(be.handle));
        // give the static inputs their own buffer BEFORE the graph pool is
        // allocated, so gallocr skips them (they must survive across runs)
        ibuf = ggml_backend_alloc_ctx_tensors(ictx, be.handle);
        if (!ibuf) return false;
        if (!alloc_graph("mapanything")) return false;
        log_info("mapanything: graph allocation done, uploading static inputs");
        ggml_backend_tensor_set(in_mean, m->meta.img_mean, 0,
                                3 * sizeof(float));
        ggml_backend_tensor_set(in_std, m->meta.img_std, 0, 3 * sizeof(float));
        ggml_backend_tensor_set(in_pos_embed, host_pos_embed.data(), 0,
                                host_pos_embed.size() * sizeof(float));
        log_info("graph built: %d nodes (mapanything)",
                 ggml_graph_n_nodes(graph));
        return true;
    }

    bool run(const float* imgs, VGGTOutputs& out) override {
        if (!graph) return false;
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_tensor_set(in_images, imgs, 0,
                                (size_t)S * 3 * H * W * sizeof(float));
        ggml_backend_graph_compute(be.handle, graph);
        out.timing_ms["inference_total"] =
                std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
#ifdef MAP_ANYTHING_ENABLE_DUMP
        for (size_t i = 0; i < dbg_nodes.size(); i++) {
            std::vector<float> tmp(ggml_nelements(dbg_nodes[i]));
            ggml_backend_tensor_get(dbg_nodes[i], tmp.data(), 0,
                                    tmp.size() * sizeof(float));
            FILE* fp = fopen(dbg_names[i].c_str(), "wb");
            if (fp) {
                fwrite(tmp.data(), 4, tmp.size(), fp);
                fclose(fp);
            }
        }
#endif
        // dense outputs {W, H, ch, S} -> (S, H, W, ch) row-major
        auto fetch = [&](ggml_tensor* t, int ch, std::vector<float>& dst) {
            std::vector<float> tmp((size_t)W * H * ch * S);
            ggml_backend_tensor_get(t, tmp.data(), 0,
                                    tmp.size() * sizeof(float));
            dst.resize((size_t)S * H * W * ch);
            for (int s = 0; s < S; s++)
                for (int hh = 0; hh < H; hh++)
                    for (int ww = 0; ww < W; ww++)
                        for (int c = 0; c < ch; c++)
                            dst[(((size_t)s * H + hh) * W + ww) * ch + c] =
                                    tmp[(size_t)ww + (size_t)W * hh +
                                        (size_t)W * H * c +
                                        (size_t)W * H * ch * s];
        };
        // rays: unit-sphere normalize on the fly
        std::vector<float> rays;
        fetch(out_rays, 3, rays);
        out.local_points.assign(rays.size(), 0.0f);  // reused as ray dirs
        for (size_t i = 0; i < rays.size() / 3; i++) {
            float nx = rays[3 * i], ny = rays[3 * i + 1], nz = rays[3 * i + 2];
            float n = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (n < 1e-8f) n = 1e-8f;
            out.local_points[3 * i] = nx / n;
            out.local_points[3 * i + 1] = ny / n;
            out.local_points[3 * i + 2] = nz / n;
        }
        std::vector<float> dense;
        fetch(out_dense, 3, dense);  // [depth_log, conf_log, mask_log]
        const size_t npix = (size_t)S * H * W;
        out.conf.resize(npix);   // official "conf" = 1 + exp (vmin=1):
                                 // the single dense-head confidence channel
        out.depth.resize(npix);  // depth = exp(log)
        out.depth_conf.resize(npix);  // same channel under the vggt key name
        out.mask.resize(npix);        // mask = sigmoid(log)
        for (size_t i = 0; i < npix; i++) {
            out.depth[i] = std::exp(dense[3 * i]);
            out.depth_conf[i] = 1.0f + std::exp(dense[3 * i + 1]);
            out.conf[i] = out.depth_conf[i];
            out.mask[i] = 1.0f / (1.0f + std::exp(-dense[3 * i + 2]));
        }
        // scale = exp(raw); the official forward multiplies cam_trans,
        // depth_along_ray and both pointmaps by it (rays and quats unscaled)
        float sraw = 0.0f;
        ggml_backend_tensor_get(out_scale_raw, &sraw, 0, sizeof(float));
        out.metric_scale = std::exp(sraw);
        // pose: t | quat(xyzw, scalar-last) -> 4x4 c2w with L2-normalized quat
        out.pose_raw.resize((size_t)7 * S);
        ggml_backend_tensor_get(out_pose_raw, out.pose_raw.data(), 0,
                                (size_t)7 * S * sizeof(float));
        out.camera_poses.assign((size_t)16 * S, 0.0f);
        for (int s = 0; s < S; s++) {
            const float* p = &out.pose_raw[(size_t)7 * s];
            float qx = p[3], qy = p[4], qz = p[5], qw = p[6];
            float qn = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
            if (qn < 1e-8f) qn = 1e-8f;
            qx /= qn;
            qy /= qn;
            qz /= qn;
            qw /= qn;
            float* R = &out.camera_poses[(size_t)16 * s];
            R[0] = 1 - 2 * (qy * qy + qz * qz);
            R[1] = 2 * (qx * qy - qz * qw);
            R[2] = 2 * (qx * qz + qy * qw);
            R[4] = 2 * (qx * qy + qz * qw);
            R[5] = 1 - 2 * (qx * qx + qz * qz);
            R[6] = 2 * (qy * qz - qx * qw);
            R[8] = 2 * (qx * qz - qy * qw);
            R[9] = 2 * (qy * qz + qx * qw);
            R[10] = 1 - 2 * (qx * qx + qy * qy);
            // ROW-MAJOR camera-to-world: translation in the LAST COLUMN
            // (indices 3/7/11), matching pi3_svd_pose and the evaluator
            // convention (ext[..., :3, 3]).  The world_points unprojection
            // below reads exactly these slots.
            R[3] = p[0] * out.metric_scale;
            R[7] = p[1] * out.metric_scale;
            R[11] = p[2] * out.metric_scale;
            R[15] = 1.0f;
        }
        for (size_t i = 0; i < npix; i++) out.depth[i] *= out.metric_scale;
        // camera-frame points (world = c2w @ p, host side)
        out.world_points.assign(npix * 3, 0.0f);
        for (int s = 0; s < S; s++) {
            const float* R = &out.camera_poses[(size_t)16 * s];
            for (int hh = 0; hh < H; hh++) {
                for (int ww = 0; ww < W; ww++) {
                    const size_t o3 = (((size_t)s * H + hh) * W + ww) * 3;
                    const float dx = out.local_points[o3];
                    const float dy = out.local_points[o3 + 1];
                    const float dz = out.local_points[o3 + 2];
                    const float d = out.depth[o3 / 3];
                    const float px = dx * d, py = dy * d, pz = dz * d;
                    for (int r = 0; r < 3; r++)
                        out.world_points[o3 + r] =
                                R[r * 4] * px + R[r * 4 + 1] * py +
                                R[r * 4 + 2] * pz + R[r * 4 + 3];
                }
            }
        }
        return true;
    }
};

static RegisterBuilder g_reg_mapanything("mapanything", [] {
    return std::unique_ptr<IGraphBuilder>(new VGGTRuntime::MapAnythingImpl());
});

// -----------------------------------------------------------------
// The DUSt3R builder (naver/dust3r, general.architecture="dust3r") — M5,
// the pair-wise third branch. Grounded in third_party/dust3r-src:
//   * siamese CroCo ViT-L/16 encoder (both views of the pair batched — the
//     official _encode_image_pairs behaviour for same-shape pairs):
//     patch conv k16 s16 (PatchEmbed norm is Identity), fused qkv split by
//     the converter, RoPE2D(freq 100) on q AND k with 0-based (y, x)
//     positions, no LayerScale, no qk-norm, LN eps 1e-6, GELU-erf MLP;
//   * decoder_embed Linear(1024->768) per view; 12 block PAIRS with
//     separate weights per side (dec_blocks -> dec.0.N, dec_blocks2 ->
//     dec.1.N); each side: self-attn + cross-attn into the OTHER side's
//     PREVIOUS output + MLP (both sides read the previous-pair state —
//     model.py _decoder). Cross-attn: projq/projk/projv, rope on q (qpos)
//     and k (kpos), norm_y on the memory;
//   * same-size pairs make pos1 == pos2 elementwise, so ONE (y, x) table
//     per side serves every q/k (only the head count differs: enc 16,
//     dec 12 -> two tables, same row content);
//   * DPT taps per view (dpt_head.py hooks [0, 6, 9, 12] into the
//     post-del decout list): [encoder feature (pre-projection, 1024-dim),
//     dec block 5, 8, 11 (768-dim; the last one dec-normed)];
//   * dust3r DPT head: act_{1,2,3,4}_postprocess (same four-branch
//     structure as mapanything's input_process), scratch.layer{1..4}_rn
//     (3x3, no bias), refinenet chain (refinenet4 takes ONE argument —
//     no skip, no crop), regression head conv3x3(256->128) -> bilinear 2x
//     (align_corners) -> conv3x3(128->32) -> ReLU -> conv1x1(32->4);
//     FULL input-resolution output, 4ch = xyz | conf_log;
//   * host postprocess: pts3d = unit(v) * expm1(|v|) (depth_mode
//     ('exp', -inf, inf); the norm clip 1e-8 guards only the divisor),
//     conf = 1 + exp(log) (conf_mode ('exp', 1, inf)).
//   * NO pose head: poses come from the out-of-network global-alignment
//     optimizer (cloud_opt/), which is not part of the network and is
//     not ported here.
struct VGGTRuntime::Dust3rImpl final : VGGTRuntime::VGGTImpl {
    int Cd = 0, nhd = 0, dhd = 0;  // decoder dim / heads / head dim
    std::vector<float> host_cos_dec, host_sin_dec;  // {dhd, nhd, P, S}
    ggml_tensor* in_cos_dec = nullptr;
    ggml_tensor* in_sin_dec = nullptr;
    ggml_tensor* out_raw = nullptr;  // {W, H, 4, S} concat(head1, head2)

    // croco RoPE2D rows (freq 100, 0-based (y, x) grid positions, all
    // tokens, head-independent rows): [cy|cy|cx|cx] segment layout, the
    // same one build_rope_rows_vggt emits, so rope_apply_vggt applies it
    // verbatim.
    void build_rope_rows_dust3r(int heads,
                                int dh_l,
                                std::vector<float>& cos,
                                std::vector<float>& sin) const {
        const int half = dh_l / 4;  // 16 frequencies per axis at dh=64
        cos.assign((size_t)dh_l * heads * P * S, 0.0f);
        sin.assign((size_t)dh_l * heads * P * S, 0.0f);
        for (int s = 0; s < S; s++) {
            for (int h = 0; h < heads; h++) {
                for (int t = 0; t < P; t++) {
                    const double py = double(t / Wp);  // 0-based (y, x)
                    const double px = double(t % Wp);
                    float* crow =
                            &cos[(((size_t)s * P + t) * heads + h) * dh_l];
                    float* srow =
                            &sin[(((size_t)s * P + t) * heads + h) * dh_l];
                    for (int j = 0; j < half; j++) {
                        // croco get_cos_sin: inv_freq = base^(-2j/D), D = dh/2
                        const double f =
                                std::pow(100.0, -2.0 * j / (double)(half * 2));
                        crow[j] = crow[j + half] = (float)std::cos(py * f);
                        crow[j + 2 * half] = crow[j + 3 * half] =
                                (float)std::cos(px * f);
                        srow[j] = srow[j + half] = (float)std::sin(py * f);
                        srow[j + 2 * half] = srow[j + 3 * half] =
                                (float)std::sin(px * f);
                    }
                }
            }
        }
    }

    // tokens {Ct, P, 1} (p = h*Wp + w) -> conv grid {Wp, Hp, Ct, 1}
    // (same reshape+permute as MapAnythingImpl::tokens_to_grid).
    ggml_tensor* tokens_to_grid_d3(ggml_context* ctx, ggml_tensor* x) const {
        ggml_tensor* g = ggml_reshape_4d(ctx, x, x->ne[0], Wp, Hp, x->ne[2]);
        return ggml_cont(ctx, ggml_permute(ctx, g, 2, 0, 1, 3));
    }

    // Self-attention with an explicit head count (attention_v binds the
    // ENCODER's nh member; the dust3r decoder runs 12 heads on Cd=768).
    ggml_tensor* self_attn(ggml_context* ctx,
                           ggml_tensor* x,
                           int dim,
                           int heads,
                           const std::string& p,
                           ggml_tensor* cos_rows,
                           ggml_tensor* sin_rows,
                           int T,
                           int B) const {
        const int dh_l = dim / heads;
        ggml_tensor* q =
                ggml_reshape_4d(ctx,
                                lin(ctx, x, w((p + ".attn.q.weight").c_str()),
                                    w((p + ".attn.q.bias").c_str())),
                                dh_l, heads, T, B);
        ggml_tensor* k =
                ggml_reshape_4d(ctx,
                                lin(ctx, x, w((p + ".attn.k.weight").c_str()),
                                    w((p + ".attn.k.bias").c_str())),
                                dh_l, heads, T, B);
        ggml_tensor* v =
                ggml_reshape_4d(ctx,
                                lin(ctx, x, w((p + ".attn.v.weight").c_str()),
                                    w((p + ".attn.v.bias").c_str())),
                                dh_l, heads, T, B);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE() && p == "dec.0.0") {
            dump_node(q, "vg_d3_b0_qraw");
            dump_node(k, "vg_d3_b0_kraw");
        }
#endif
        q = rope_apply_vggt(ctx, q, dh_l, cos_rows, sin_rows);
        k = rope_apply_vggt(ctx, k, dh_l, cos_rows, sin_rows);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE() && p == "dec.0.0") {
            dump_node(q, "vg_d3_b0_qrope");
            dump_node(k, "vg_d3_b0_krope");
        }
#endif
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        ggml_tensor* o =
                flash_attn(ctx, q, k, v, 1.0f / std::sqrt((float)dh_l));
        o = ggml_reshape_3d(ctx, o, dim, T, B);
        return lin(ctx, o, w((p + ".attn.proj.weight").c_str()),
                   w((p + ".attn.proj.bias").c_str()));
    }

    // croco CrossAttention: q = rope(projq(x), qpos); k/v from the
    // (already norm_y'd) memory, k roped by kpos. Both position tables
    // are token-indexed views of the shared (y, x) table.
    ggml_tensor* cross_attention(ggml_context* ctx,
                                 ggml_tensor* x,
                                 ggml_tensor* y_mem,
                                 const std::string& p,
                                 int heads,
                                 ggml_tensor* cos_q,
                                 ggml_tensor* sin_q,
                                 ggml_tensor* cos_k,
                                 ggml_tensor* sin_k,
                                 int Tq,
                                 int Tk,
                                 int B) const {
        const int dh_l = Cd / heads;
        ggml_tensor* q = ggml_reshape_4d(
                ctx,
                lin(ctx, x, w((p + ".cross_attn.projq.weight").c_str()),
                    w((p + ".cross_attn.projq.bias").c_str())),
                dh_l, heads, Tq, B);
        ggml_tensor* k = ggml_reshape_4d(
                ctx,
                lin(ctx, y_mem, w((p + ".cross_attn.projk.weight").c_str()),
                    w((p + ".cross_attn.projk.bias").c_str())),
                dh_l, heads, Tk, B);
        ggml_tensor* v = ggml_reshape_4d(
                ctx,
                lin(ctx, y_mem, w((p + ".cross_attn.projv.weight").c_str()),
                    w((p + ".cross_attn.projv.bias").c_str())),
                dh_l, heads, Tk, B);
        q = rope_apply_vggt(ctx, q, dh_l, cos_q, sin_q);
        k = rope_apply_vggt(ctx, k, dh_l, cos_k, sin_k);
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        ggml_tensor* o =
                flash_attn(ctx, q, k, v, 1.0f / std::sqrt((float)dh_l));
        o = ggml_reshape_3d(ctx, o, Cd, Tq, B);
        return lin(ctx, o, w((p + ".cross_attn.proj.weight").c_str()),
                   w((p + ".cross_attn.proj.bias").c_str()));
    }

    // croco DecoderBlock: x = x + self_attn(ln1(x), xpos);
    // x = x + cross_attn(ln2(x), norm_y(y), xpos, ypos);
    // x = x + mlp(ln3(x)).  No LayerScale, no qk-norm, LN eps 1e-6.
    ggml_tensor* dec_block(ggml_context* ctx,
                           ggml_tensor* x,
                           ggml_tensor* y,
                           const std::string& p,
                           ggml_tensor* cos_rows,
                           ggml_tensor* sin_rows,
                           int T,
                           int B) {
        ggml_tensor* h = ln(ctx, x, w((p + ".norm1.weight").c_str()),
                            w((p + ".norm1.bias").c_str()), 1e-6f);
        h = self_attn(ctx, h, Cd, nhd, p, cos_rows, sin_rows, T, B);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE() && p == "dec.0.0")
            dump_node(h, "vg_d3_b0_selfattn");
#endif
        x = ggml_add(ctx, x, h);
        ggml_tensor* ym = ln(ctx, y, w((p + ".norm_y.weight").c_str()),
                             w((p + ".norm_y.bias").c_str()), 1e-6f);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE() && p == "dec.0.0") dump_node(ym, "vg_d3_b0_ymem");
#endif
        h = ln(ctx, x, w((p + ".norm2.weight").c_str()),
               w((p + ".norm2.bias").c_str()), 1e-6f);
        h = cross_attention(ctx, h, ym, p, nhd, cos_rows, sin_rows, cos_rows,
                            sin_rows, T, T, B);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE() && p == "dec.0.0") dump_node(h, "vg_d3_b0_cross");
#endif
        x = ggml_add(ctx, x, h);
        ggml_tensor* f = ln(ctx, x, w((p + ".norm3.weight").c_str()),
                            w((p + ".norm3.bias").c_str()), 1e-6f);
        f = lin(ctx, f, w((p + ".mlp.fc1.weight").c_str()),
                w((p + ".mlp.fc1.bias").c_str()));
        f = ggml_gelu_erf(ctx, f);
        f = lin(ctx, f, w((p + ".mlp.fc2.weight").c_str()),
                w((p + ".mlp.fc2.bias").c_str()));
        return ggml_add(ctx, x, f);
    }

    // dust3r DPT for one side (pfx = "head1" | "head2"); taps are token
    // tensors {Ct, P, 1}.  Four-branch act_postprocess (identical structure
    // to MapAnything's dpt_input_process, official key names), layer_rn,
    // refinenet chain (refinenet4 single-input), regression head with the
    // 2x bilinear BETWEEN conv1 and conv2 (dust3r's own convention).
    ggml_tensor* dpt_head_d3(ggml_context* ctx,
                             const std::string& pfx,
                             ggml_tensor* const* taps) {
        // act_postprocess is a ModuleList (the act_N_postprocess names are
        // only python aliases): keys are {i}.0 conv1x1 and {i}.1 convT4 /
        // convT2 / conv-s2 per branch (verified against the HF safetensors).
        ggml_tensor* tg[4];
        for (int i = 0; i < 4; i++) {
            const std::string p = pfx + ".act_postprocess." + std::to_string(i);
            ggml_tensor* x = tokens_to_grid_d3(ctx, taps[i]);
            x = conv1x1(ctx, p + ".0", x);
            if (i == 0 || i == 1) {
                x = ggml_add(ctx,
                             mapany_conv_transpose_2d_p0(
                                     ctx, w((p + ".1.weight").c_str()), x,
                                     i == 0 ? 4 : 2),
                             bias4(ctx, w((p + ".1.bias").c_str()), x->ne[2]));
            } else if (i == 3) {
                x = ggml_add(ctx,
                             mapany_conv_2d(ctx, w((p + ".1.weight").c_str()),
                                            x, 2, 2, 1, 1, 1, 1),
                             bias4(ctx, w((p + ".1.bias").c_str()), x->ne[2]));
            }
            x = conv3x3(ctx,
                        pfx + ".scratch.layer" + std::to_string(i + 1) + "_rn",
                        x, false);
#ifdef MAP_ANYTHING_ENABLE_DUMP
            if (MAP_DUMP_ACTIVE())
                dump_node(x, "vg_d3_" + pfx + "_tap" + std::to_string(i));
#endif
            tg[i] = x;
        }
        ggml_tensor* o = refinenet(ctx, pfx + ".scratch.refinenet4", tg[3],
                                   nullptr, 2 * tg[3]->ne[0], 2 * tg[3]->ne[1]);
        // official DPTOutputAdapter_fix crops refinenet4's output to tap2's
        // grid ([:, :, :H2, :W2]) — mandatory when Hp is ODD (336/16 = 21:
        // 2x upsampling gives 22 vs 21; square 512 was an accidental no-op)
        o = ggml_cont(ctx,
                      ggml_view_4d(ctx, o, tg[2]->ne[0], tg[2]->ne[1], o->ne[2],
                                   o->ne[3], o->nb[1], o->nb[2], o->nb[3], 0));
        o = refinenet(ctx, pfx + ".scratch.refinenet3", o, tg[2], 2 * o->ne[0],
                      2 * o->ne[1]);
        o = refinenet(ctx, pfx + ".scratch.refinenet2", o, tg[1], 2 * o->ne[0],
                      2 * o->ne[1]);
        o = refinenet(ctx, pfx + ".scratch.refinenet1", o, tg[0], 2 * o->ne[0],
                      2 * o->ne[1]);
        o = conv3x3(ctx, pfx + ".head.0", o, true);
        o = ggml_interpolate(
                ctx, o, 2 * o->ne[0], 2 * o->ne[1], o->ne[2], o->ne[3],
                GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
        o = conv3x3(ctx, pfx + ".head.2", o, true);
        o = ggml_relu(ctx, o);
        return conv1x1(ctx, pfx + ".head.4", o);  // {W, H, 4, 1}
    }

    bool build_graph() override {
        if (S != 2) {
            log_error("dust3r is pair-wise: S must be 2, got %d", S);
            return false;
        }
        // the dec_* dims default to 0 in VGGTMeta; a GGUF without the
        // dust3r.dec_* keys fails here instead of running silent garbage
        if (m->meta.dec_embed_dim <= 0 || m->meta.dec_num_heads <= 0 ||
            m->meta.dec_embed_dim % m->meta.dec_num_heads != 0) {
            log_error(
                    "dust3r GGUF missing/bad dust3r.dec_embed_dim (%d) / "
                    "dust3r.dec_num_heads (%d)",
                    m->meta.dec_embed_dim, m->meta.dec_num_heads);
            return false;
        }
        Cd = m->meta.dec_embed_dim;
        nhd = m->meta.dec_num_heads;
        dhd = Cd / nhd;
        const size_t kNodes = 80000;
        ggml_init_params gp{ggml_tensor_overhead() * kNodes +
                                    ggml_graph_overhead_custom(kNodes, false),
                            nullptr, true};
        gctx = ggml_init(gp);
        if (!gctx) return false;
        return build_stage1_dust3r();
    }

    bool build_stage1_dust3r() {
        log_info("dust3r: creating graph inputs");
        // static inputs get their OWN backend buffer (see build_stage1_vggt
        // for why: the gallocr pool recycles their space mid-graph after the
        // last use, so "upload once" would not survive a second run())
        ictx = ggml_init({ggml_tensor_overhead() * 64, nullptr, true});
        if (!ictx) return false;
        in_images = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, W, H, 3, S);
        ggml_set_input(in_images);
        in_mean = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, 1, 1, 3, 1);
        in_std = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, 1, 1, 3, 1);
        in_cos_v = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, dh, nh, P, S);
        in_sin_v = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, dh, nh, P, S);
        in_cos_dec = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, dhd, nhd, P, S);
        in_sin_dec = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, dhd, nhd, P, S);
        ggml_set_input(in_cos_v);
        ggml_set_input(in_sin_v);
        ggml_set_input(in_cos_dec);
        ggml_set_input(in_sin_dec);

        // dust3r ImgNorm: (x - 0.5) / 0.5  (mean/std from the GGUF meta)
        ggml_tensor* img =
                ggml_div(gctx, ggml_sub(gctx, in_images, in_mean), in_std);

        // ---- siamese encoder (pair batched, B = S) ----
        ggml_tensor* x = mapany_conv_2d(gctx, w("bb_patch.proj.weight"), img,
                                        m->meta.patch_size, m->meta.patch_size,
                                        0, 0, 1, 1);
        x = ggml_add(gctx, x, bias4(gctx, w("bb_patch.proj.bias"), C));
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) dump_node(x, "vg_d3_conv");
#endif
        x = ggml_cont(gctx, ggml_permute(gctx, x, 1, 2, 0, 3));
        x = ggml_reshape_3d(gctx, x, C, P, S);  // {C, P, S}
        for (int i = 0; i < m->meta.depth; i++) {
            x = block_v(gctx, x, "enc." + std::to_string(i), C, false, in_cos_v,
                        in_sin_v, P, S, 1e-6f, false);
            if (i == 0) {
                ggml_set_output(x);
#ifdef MAP_ANYTHING_ENABLE_DUMP
                if (MAP_DUMP_ACTIVE()) dump_node(x, "vg_d3_encblk0");
#endif
            }
        }
        x = ln(gctx, x, w("enc_norm.weight"), w("enc_norm.bias"), 1e-6f);
        ggml_set_output(x);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) dump_node(x, "vg_d3_enc");
#endif

        // ---- decoder entry: per-view projection to Cd (the checkpoint
        // DOES carry decoder_embed.bias — trained, max |b| ≈ 11.3) ----
        ggml_tensor* fboth = lin(gctx, x, w("decoder_embed.weight"),
                                 w("decoder_embed.bias"));  // {Cd, P, S}
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) dump_node(fboth, "vg_d3_decproj");
#endif
        ggml_tensor* f1 =
                ggml_cont(gctx, ggml_view_3d(gctx, fboth, Cd, P, 1,
                                             fboth->nb[1], fboth->nb[2], 0));
        ggml_tensor* f2 =
                ggml_cont(gctx, ggml_view_3d(gctx, fboth, Cd, P, 1,
                                             fboth->nb[1], fboth->nb[2],
                                             (size_t)P * Cd * sizeof(float)));
        // per-view rope table views {dh, heads, P, 1} (slices of the
        // S-length tables; nb[3] steps one view)
        ggml_tensor* cos1 = ggml_view_4d(gctx, in_cos_dec, dhd, nhd, P, 1,
                                         in_cos_dec->nb[1], in_cos_dec->nb[2],
                                         in_cos_dec->nb[3], 0);
        ggml_tensor* sin1 = ggml_view_4d(gctx, in_sin_dec, dhd, nhd, P, 1,
                                         in_sin_dec->nb[1], in_sin_dec->nb[2],
                                         in_sin_dec->nb[3], 0);
        ggml_tensor* cos2 = ggml_view_4d(gctx, in_cos_dec, dhd, nhd, P, 1,
                                         in_cos_dec->nb[1], in_cos_dec->nb[2],
                                         in_cos_dec->nb[3],
                                         (size_t)P * nhd * dhd * sizeof(float));
        ggml_tensor* sin2 = ggml_view_4d(gctx, in_sin_dec, dhd, nhd, P, 1,
                                         in_sin_dec->nb[1], in_sin_dec->nb[2],
                                         in_sin_dec->nb[3],
                                         (size_t)P * nhd * dhd * sizeof(float));

        // ---- 12 block pairs (BOTH sides read the previous-pair state) ----
        ggml_tensor* tap1[4];
        ggml_tensor* tap2[4];
        // hook 0 = the pre-projection ENCODER feature, sliced per view like
        // every other tap (head1 consumes view1's, head2 view2's); without
        // the slice the S=2 batch dim would leak into the conv grid.
        tap1[0] = ggml_cont(
                gctx, ggml_view_3d(gctx, x, C, P, 1, x->nb[1], x->nb[2], 0));
        tap2[0] = ggml_cont(gctx,
                            ggml_view_3d(gctx, x, C, P, 1, x->nb[1], x->nb[2],
                                         (size_t)P * C * sizeof(float)));
        int nt = 1;
        for (int i = 0; i < m->meta.aa_depth; i++) {
            ggml_tensor* n1 =
                    dec_block(gctx, f1, f2, "dec.0." + std::to_string(i), cos1,
                              sin1, P, 1);
            ggml_tensor* n2 =
                    dec_block(gctx, f2, f1, "dec.1." + std::to_string(i), cos2,
                              sin2, P, 1);
            f1 = n1;
            f2 = n2;
            if (i == 5 || i == 8) {  // hooks 6/9 of decout
                tap1[nt] = f1;
                tap2[nt] = f2;
                nt++;
            }
#ifdef MAP_ANYTHING_ENABLE_DUMP
            if (MAP_DUMP_ACTIVE() && i <= 1)
                dump_node(f1, "vg_d3_dec1_" + std::to_string(i));
#endif
        }
        // final norm on both sides (hook 12 tap)
        f1 = ln(gctx, f1, w("dec_norm.weight"), w("dec_norm.bias"), 1e-6f);
        f2 = ln(gctx, f2, w("dec_norm.weight"), w("dec_norm.bias"), 1e-6f);
        tap1[3] = f1;
        tap2[3] = f2;

        // ---- per-side DPT heads -> concat on the view axis ----
        ggml_tensor* raw1 = dpt_head_d3(gctx, "head1", tap1);
        ggml_tensor* raw2 = dpt_head_d3(gctx, "head2", tap2);
        out_raw = ggml_concat(gctx, raw1, raw2, 3);  // {W, H, 4, S}
        ggml_set_output(out_raw);

        graph = ggml_new_graph_custom(gctx, 80000, false);
        ggml_build_forward_expand(graph, out_raw);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        for (size_t i = 0; i < dbg_nodes.size(); i++)
            ggml_build_forward_expand(graph, dbg_nodes[i]);
#endif
        galloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(be.handle));
        // give the static inputs their own buffer BEFORE the graph pool is
        // allocated, so gallocr skips them (they must survive across runs)
        ibuf = ggml_backend_alloc_ctx_tensors(ictx, be.handle);
        if (!ibuf) return false;
        if (!alloc_graph("dust3r")) return false;
        log_info("dust3r: graph allocation done, uploading static inputs");
        // static uploads (ggml_set_input only MARKS the tensors — pitfall #2)
        ggml_backend_tensor_set(in_mean, m->meta.img_mean, 0,
                                3 * sizeof(float));
        ggml_backend_tensor_set(in_std, m->meta.img_std, 0, 3 * sizeof(float));
        build_rope_rows_dust3r(nh, dh, host_cos_v, host_sin_v);
        build_rope_rows_dust3r(nhd, dhd, host_cos_dec, host_sin_dec);
        ggml_backend_tensor_set(in_cos_v, host_cos_v.data(), 0,
                                host_cos_v.size() * sizeof(float));
        ggml_backend_tensor_set(in_sin_v, host_sin_v.data(), 0,
                                host_sin_v.size() * sizeof(float));
        ggml_backend_tensor_set(in_cos_dec, host_cos_dec.data(), 0,
                                host_cos_dec.size() * sizeof(float));
        ggml_backend_tensor_set(in_sin_dec, host_sin_dec.data(), 0,
                                host_sin_dec.size() * sizeof(float));
        log_info("graph built: %d nodes (dust3r: enc+dec+2 DPT)",
                 ggml_graph_n_nodes(graph));
        return true;
    }

    bool run(const float* imgs, VGGTOutputs& out) override {
        if (!graph) return false;
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_tensor_set(in_images, imgs, 0,
                                (size_t)S * 3 * H * W * sizeof(float));
        ggml_backend_graph_compute(be.handle, graph);
        out.timing_ms["inference_total"] =
                std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
#ifdef MAP_ANYTHING_ENABLE_DUMP
        for (size_t i = 0; i < dbg_nodes.size(); i++) {
            std::vector<float> tmp(ggml_nelements(dbg_nodes[i]));
            ggml_backend_tensor_get(dbg_nodes[i], tmp.data(), 0,
                                    tmp.size() * sizeof(float));
            FILE* fp = fopen(dbg_names[i].c_str(), "wb");
            if (fp) {
                fwrite(tmp.data(), 4, tmp.size(), fp);
                fclose(fp);
            }
        }
#endif
        out.out_h = H;  // dust3r DPT emits the FULL input resolution
        out.out_w = W;
        // {W, H, 4, S} -> (S, H, W, 4), then the official postprocess:
        // pts3d = unit(v) * expm1(|v|), conf = 1 + exp(log)
        std::vector<float> raw((size_t)W * H * 4 * S);
        ggml_backend_tensor_get(out_raw, raw.data(), 0,
                                raw.size() * sizeof(float));
        const size_t npix = (size_t)S * H * W;
        out.local_points.assign(npix * 3, 0.0f);
        out.conf.assign(npix, 0.0f);
        out.depth.assign(npix, 0.0f);
        for (int s = 0; s < S; s++) {
            for (int h = 0; h < H; h++) {
                for (int w2 = 0; w2 < W; w2++) {
                    const size_t src =
                            (size_t)w2 + (size_t)W * h + (size_t)W * H * 4 * s;
                    const size_t o3 = (((size_t)s * H + h) * W + w2) * 3;
                    const float vx = raw[src];
                    const float vy = raw[src + (size_t)W * H];
                    const float vz = raw[src + 2 * (size_t)W * H];
                    const float cl = raw[src + 3 * (size_t)W * H];
                    const float n = std::sqrt(vx * vx + vy * vy + vz * vz);
                    const float nn = n < 1e-8f ? 1e-8f : n;
                    const float sc = std::expm1(n);
                    out.local_points[o3] = vx / nn * sc;
                    out.local_points[o3 + 1] = vy / nn * sc;
                    out.local_points[o3 + 2] = vz / nn * sc;
                    out.conf[(size_t)s * H * W + (size_t)h * W + w2] =
                            1.0f + std::exp(cl);
                    out.depth[(size_t)s * H * W + (size_t)h * W + w2] =
                            out.local_points[o3 + 2];
                }
            }
        }
        // world frame == view1's camera: s0 = head1 pts3d (view1 frame),
        // s1 = head2 pts3d_in_other_view (view2's points, view1 frame)
        out.world_points = out.local_points;
        return true;
    }
};

static RegisterBuilder g_reg_dust3r("dust3r", [] {
    return std::unique_ptr<IGraphBuilder>(new VGGTRuntime::Dust3rImpl());
});

// -----------------------------------------------------------------
// The Pi3X builder (yyfz233/Pi3X, general.architecture="pi3x").
// Images-only build: the multimodal conditioning branch (depth/ray/pose
// injection) is dropped exactly like the official disable_multimodal()
// path — with no conditions the injected embeddings are zero and the
// pose-inject blocks (1,9,17,25,33) are gated off (use_pose_mask.sum()==0).
// Shared with pi3: encoder, freq-100 rope, 36 BlockRope decoder with the
// last-two-concat, TransformerDecoder heads, CameraHead (fc_t/fc_rot(9),
// host SVD orthogonalization).  New per the blueprint: ConvHead (points
// dim_out=[2,1], conf dim_out=[1], per-level uv concat, replicate-pad
// convs, group-norm ResidualConvBlocks) and the metric branch
// (metric_token -> ContextOnlyTransformerDecoder -> exp).
struct VGGTRuntime::Pi3XImpl final : VGGTRuntime::Pi3Impl {
    ggml_tensor* out_metric = nullptr;  // {1,1,1} raw; host applies exp
    ggml_tensor* in_uv[4] = {nullptr, nullptr, nullptr, nullptr};
    ggml_tensor* in_uv_c[4] = {nullptr, nullptr, nullptr, nullptr};
    std::vector<float> host_uv[4];
    static constexpr int nh_m = 8;  // mtdec dec_num_heads (dh = 64)
    ggml_tensor* in_cos_m = nullptr;
    ggml_tensor* in_sin_m = nullptr;
    std::vector<float> host_cos_m, host_sin_m;

    // metric rope rows: dh=64, nh=8, ONE flattened context row-sequence
    // (token t of view s at row s*Nagg + t); registers keep identity rows.
    void build_rope_rows_metric() {
        const int dh_m = 512 / nh_m;  // 64
        const int half = dh_m / 4;    // 16 frequencies per axis
        host_cos_m.assign((size_t)dh_m * nh_m * Nagg * S, 0.0f);
        host_sin_m.assign((size_t)dh_m * nh_m * Nagg * S, 0.0f);
        for (int s = 0; s < S; s++)
            for (int t = 0; t < Nagg; t++) {
                double py = 0.0, px = 0.0;
                if (t >= Ncr) {
                    const int i = t - Ncr;
                    py = double(i / Wp) + 1.0;
                    px = double(i % Wp) + 1.0;
                }
                float* crow =
                        &host_cos_m[(((size_t)s * Nagg + t) * nh_m) * dh_m];
                float* srow =
                        &host_sin_m[(((size_t)s * Nagg + t) * nh_m) * dh_m];
                // the table is per-(token, head): ALL nh_m head slots must be
                // filled (previously only head 0 was written, leaving heads
                // 1..7 with zero rows -> their rope output was zero)
                for (int h = 0; h < nh_m; h++) {
                    float* ch = crow + (size_t)h * dh_m;
                    float* sh = srow + (size_t)h * dh_m;
                    for (int j = 0; j < half; j++) {
                        const double f =
                                std::pow(100.0, -2.0 * j / (double)(half * 2));
                        ch[j] = ch[j + half] = (float)std::cos(py * f);
                        ch[j + 2 * half] = ch[j + 3 * half] =
                                (float)std::cos(px * f);
                        sh[j] = sh[j + half] = (float)std::sin(py * f);
                        sh[j + 2 * half] = sh[j + 3 * half] =
                                (float)std::sin(px * f);
                    }
                }
            }
    }

    // tokens {D, P, S} -> grid {Wp, Hp, D, S} (same as MapAnythingImpl's)
    ggml_tensor* tokens_to_grid(ggml_context* ctx, ggml_tensor* x) const {
        ggml_tensor* g = ggml_reshape_4d(ctx, x, x->ne[0], Wp, Hp, S);
        return ggml_cont(ctx, ggml_permute(ctx, g, 2, 0, 1, 3));
    }

    // normalized_view_plane_uv (conv_head.py): u = linspace over width,
    // v over height with spans ar/diag and 1/diag.  Table {W, H, 2, 1}.
    static std::vector<float> uv_table(int w, int h, float ar) {
        const float diag = std::sqrt(ar * ar + 1.0f);
        const float sx = ar / diag, sy = 1.0f / diag;
        std::vector<float> t((size_t)w * h * 2);
        // output layout must match the {W,H,2,1} tensor (x fastest, channel
        // slowest): flat = x + W*y + W*H*ch.  The previous (ch-fastest) order
        // scrambled the u/v channels with the pixel coordinates.
        for (int ch = 0; ch < 2; ch++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    t[(size_t)ch * w * h + (size_t)y * w + x] =
                            (ch == 0 ? sx * (2.0f * x - (w - 1)) / w
                                     : sy * (2.0f * y - (h - 1)) / h);
        return t;
    }

    // replicate padding (padding_mode='replicate'): pad 1 ring around
    // {W, H, C, S}.  Rows first (they carry the corners), then columns.
    ggml_tensor* rep_pad2(ggml_context* ctx, ggml_tensor* x) const {
        const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], S = x->ne[3];
        auto row = [&](int64_t y) {
            return ggml_view_4d(ctx, x, W, 1, C, S, x->nb[1], x->nb[2],
                                x->nb[3], y * x->nb[1]);
        };
        ggml_tensor* v = ggml_concat(ctx, ggml_concat(ctx, row(0), x, 1),
                                     row(H - 1), 1);  // {W, H+2}
        auto col = [&](ggml_tensor* t, int64_t xx) {
            // x=xx column of t {W,H,C,S}: view_4d has NO nb0 (innermost dim
            // is implicitly contiguous), so the strides are h/c/s = nb1/nb2/nb3
            // and the byte offset xx*nb0 goes LAST.  The previous version
            // passed (nb2, nb3, offset) as (nb1, nb2, nb3) — garbage columns.
            return ggml_view_4d(ctx, t, 1, t->ne[1], t->ne[2], t->ne[3],
                                t->nb[1], t->nb[2], t->nb[3],
                                (size_t)xx * t->nb[0]);
        };
        return ggml_concat(ctx, ggml_concat(ctx, col(v, 0), v, 0),
                           col(v, W - 1), 0);  // {W+2, H+2}
    }

    // 3x3 conv with replicate padding + bias
    ggml_tensor* conv3x3_rep(ggml_context* ctx,
                             ggml_tensor* x,
                             const std::string& p) {
        ggml_tensor* xp = ggml_cont(ctx, rep_pad2(ctx, x));
        ggml_tensor* o = mapany_conv_2d(ctx, w((p + ".weight").c_str()), xp, 1,
                                        1, 0, 0, 1, 1);
        return ggml_add(ctx, o, bias4(ctx, w((p + ".bias").c_str()), o->ne[2]));
    }

    // ResidualConvBlock (conv_head.py): GN(1,C) -> ReLU -> Conv3x3(C->2C) ->
    // GN(2C/32, 2C) -> ReLU -> Conv3x3(2C->C); residual = skip(x) + layers(x),
    // skip = Identity (in == out for every use here).
    ggml_tensor* res_block_conv(ggml_context* ctx,
                                const std::string& p,
                                ggml_tensor* x,
                                int C) {
        // torch GroupNorm has affine params (weight/bias) — they were
        // silently dropped before, which corrupted every ResConvBlock.
        // Both are per-channel (C,) vectors: reshape to {1,1,C,1} so ggml's
        // ne[0]==1 broadcast applies across W/H/S (same trick as bias4).
        ggml_tensor* a = ggml_group_norm(ctx, x, 1, 1e-5f);
        a = ggml_mul(ctx, a,
                     ggml_reshape_4d(ctx, w((p + ".layers.0.weight").c_str()),
                                     1, 1, C, 1));
        a = ggml_add(ctx, a, bias4(ctx, w((p + ".layers.0.bias").c_str()), C));
        a = ggml_relu(ctx, a);
        a = conv3x3_rep(ctx, a, p + ".layers.2");
        a = ggml_group_norm(ctx, a, C * 2 / 32, 1e-5f);
        // layers.3 affine lives on the HIDDEN channels
        // (dim_times_res_block_hidden=2 -> 2C)
        a = ggml_mul(ctx, a,
                     ggml_reshape_4d(ctx, w((p + ".layers.3.weight").c_str()),
                                     1, 1, 2 * C, 1));
        a = ggml_add(ctx, a,
                     bias4(ctx, w((p + ".layers.3.bias").c_str()), 2 * C));
        a = ggml_relu(ctx, a);
        a = conv3x3_rep(ctx, a, p + ".layers.5");
        return ggml_add(ctx, x, a);
    }

    // ConvHead: tokens {D, Nagg, S} -> slice [5:] -> grid -> 3 upsample
    // levels (uv concat before each; ConvT2 + Conv3x3 + 2 ResBlocks) ->
    // bilinear to (H, W) -> uv concat -> per-dim output blocks
    // {Conv3x3(->32), ReLU, Conv1x1(->dim_out)}.  Returns the per-dim maps.
    std::vector<ggml_tensor*> conv_head(ggml_context* ctx,
                                        ggml_tensor* tokens,
                                        const std::string& p,
                                        const int* dim_out,
                                        int n_out) {
        ggml_tensor* x = ggml_cont(
                gctx,
                ggml_view_3d(ctx, tokens, tokens->ne[0], P, S, tokens->nb[1],
                             tokens->nb[2],
                             (size_t)Ncr * tokens->ne[0] * sizeof(float)));
        x = tokens_to_grid(ctx, x);  // {Wp, Hp, D, S}
        const int dims[3] = {Wp, 2 * Wp, 4 * Wp};
        const int dys[3] = {Hp, 2 * Hp, 4 * Hp};
        for (int i = 0; i < 3; i++) {
            x = ggml_concat(
                    ctx, x,
                    ggml_repeat(ctx, in_uv[i],
                                ggml_new_tensor_4d(gctx, GGML_TYPE_F32, dims[i],
                                                   dys[i], 2, S)),
                    2);
#ifdef MAP_ANYTHING_ENABLE_DUMP
            if (MAP_DUMP_ACTIVE()) dump_node(x, p + "_uv" + std::to_string(i));
#endif
            x = ggml_add(ctx,
                         mapany_conv_transpose_2d_p0(
                                 ctx,
                                 w((p + ".upsample_blocks." +
                                    std::to_string(i) + ".0.0.weight")
                                           .c_str()),
                                 x, 2),
                         bias4(ctx,
                               w((p + ".upsample_blocks." + std::to_string(i) +
                                  ".0.0.bias")
                                         .c_str()),
                               w((p + ".upsample_blocks." + std::to_string(i) +
                                  ".0.0.bias")
                                         .c_str())
                                       ->ne[0]));
#ifdef MAP_ANYTHING_ENABLE_DUMP
            if (MAP_DUMP_ACTIVE())
                dump_node(x, p + "_convt" + std::to_string(i));
#endif
            x = conv3x3_rep(
                    ctx, x,
                    p + ".upsample_blocks." + std::to_string(i) + ".0.1");
#ifdef MAP_ANYTHING_ENABLE_DUMP
            if (MAP_DUMP_ACTIVE()) dump_node(x, p + "_c3" + std::to_string(i));
#endif
            x = res_block_conv(
                    ctx, p + ".upsample_blocks." + std::to_string(i) + ".1", x,
                    (int)x->ne[2]);
            x = res_block_conv(
                    ctx, p + ".upsample_blocks." + std::to_string(i) + ".2", x,
                    (int)x->ne[2]);
#ifdef MAP_ANYTHING_ENABLE_DUMP
            if (MAP_DUMP_ACTIVE()) dump_node(x, p + "_lvl" + std::to_string(i));
#endif
        }
        x = ggml_interpolate(ctx, x, W, H, x->ne[2], x->ne[3],
                             GGML_SCALE_MODE_BILINEAR);
        x = ggml_concat(ctx, x,
                        ggml_repeat(ctx, in_uv[3],
                                    ggml_new_tensor_4d(gctx, GGML_TYPE_F32, W,
                                                       H, 2, S)),
                        2);
        std::vector<ggml_tensor*> outs;
        for (int j = 0; j < n_out; j++) {
            ggml_tensor* o = conv3x3_rep(
                    ctx, x, p + ".output_block." + std::to_string(j) + ".0");
            o = ggml_relu(ctx, o);
            o = mapany_conv_2d(
                    ctx,
                    w((p + ".output_block." + std::to_string(j) + ".2.weight")
                              .c_str()),
                    o, 1, 1, 0, 0, 1, 1);
            o = ggml_add(ctx, o,
                         bias4(ctx,
                               w((p + ".output_block." + std::to_string(j) +
                                  ".2.bias")
                                         .c_str()),
                               o->ne[2]));
            outs.push_back(o);
        }
        return outs;
    }

    // CrossOnlyBlockRope: x = x + cross_attn(ln(x), ln_y(y), xpos, ypos);
    // x = x + mlp(ln3(x)).  x is the single metric token {512,1,1} (its
    // position is the first register slot, pos 0 -> NO rope rotation on
    // q); the k rope uses the same per-view rope tables flattened to
    // {dh, nh, Nagg*S, 1} (flat order identical: t + Nagg*s).
    ggml_tensor* cross_block(ggml_context* ctx,
                             const std::string& p,
                             ggml_tensor* x,
                             ggml_tensor* y) {
        const int dh_l = 512 / nh_m;  // mtdec heads = 8 -> dh = 64
        const int64_t T = y->ne[1];
        ggml_tensor* q = lin(ctx,
                             ln(ctx, x, w((p + ".norm2.weight").c_str()),
                                w((p + ".norm2.bias").c_str())),
                             w((p + ".cross_attn.q_proj.weight").c_str()),
                             w((p + ".cross_attn.q_proj.bias").c_str()));
        ggml_tensor* yn = ln(ctx, y, w((p + ".norm_y.weight").c_str()),
                             w((p + ".norm_y.bias").c_str()));
        ggml_tensor* k =
                lin(ctx, yn, w((p + ".cross_attn.k_proj.weight").c_str()),
                    w((p + ".cross_attn.k_proj.bias").c_str()));
        ggml_tensor* v =
                lin(ctx, yn, w((p + ".cross_attn.v_proj.weight").c_str()),
                    w((p + ".cross_attn.v_proj.bias").c_str()));
        ggml_tensor* qc = ggml_reshape_4d(ctx, q, dh_l, nh_m, 1, 1);
        ggml_tensor* kc = ggml_reshape_4d(ctx, k, dh_l, nh_m, T, 1);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) dump_node(k, p + "_kn");  // {512,T*S,1}
#endif
        ggml_tensor* vc = ggml_reshape_4d(ctx, v, dh_l, nh_m, T, 1);
        ggml_tensor* cm = ggml_reshape_4d(ctx, in_cos_m, dh_l, nh_m, T, 1);
        ggml_tensor* sm = ggml_reshape_4d(ctx, in_sin_m, dh_l, nh_m, T, 1);
        kc = rope_apply_vggt(ctx, kc, dh_l, cm, sm);
        qc = ggml_cont(ctx,
                       ggml_permute(ctx, qc, 0, 2, 1, 3));  // {dh,1,nh_m,S}
        kc = ggml_cont(ctx, ggml_permute(ctx, kc, 0, 2, 1, 3));
        vc = ggml_cont(ctx, ggml_permute(ctx, vc, 0, 2, 1, 3));
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) {
            dump_node(qc, p + "_q");   // {dh,1,nh,1} == (1,nh,1,dh)
            dump_node(kc, p + "_kr");  // {dh,T,nh,1}
        }
#endif
        ggml_tensor* o =
                flash_attn(ctx, qc, kc, vc, 1.0f / std::sqrt((float)dh_l));
        o = ggml_reshape_3d(ctx, o, 512, 1, 1);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) dump_node(o, p + "_sdpa");
#endif
        o = lin(ctx, o, w((p + ".cross_attn.proj.weight").c_str()),
                w((p + ".cross_attn.proj.bias").c_str()));
        x = ggml_add(ctx, x, o);
        ggml_tensor* h = ln(ctx, x, w((p + ".norm3.weight").c_str()),
                            w((p + ".norm3.bias").c_str()));
        h = ggml_gelu(ctx, lin(ctx, h, w((p + ".mlp.fc1.weight").c_str()),
                               w((p + ".mlp.fc1.bias").c_str())));
        h = lin(ctx, h, w((p + ".mlp.fc2.weight").c_str()),
                w((p + ".mlp.fc2.bias").c_str()));
        return ggml_add(ctx, x, h);
    }

    // metric branch: metric_token -> projects_x -> 5 x CrossOnlyBlockRope
    // (context = the last-two-concat decoder tokens) -> Linear(512,1).
    ggml_tensor* metric_branch(ggml_context* ctx, ggml_tensor* hidden2) {
        ggml_tensor* x = lin(ctx, w("mtok"), w("mtdec.projects_x.weight"),
                             w("mtdec.projects_x.bias"));  // {512,1,1}
        ggml_tensor* y = lin(ctx, hidden2, w("mtdec.projects_y.weight"),
                             w("mtdec.projects_y.bias"));  // {512,T,S}
        // flatten the (token, view) axes to one context sequence (the flat
        // order t + Nagg*s matches the rope tables' layout)
        y = ggml_reshape_3d(gctx, y, y->ne[0], y->ne[1] * y->ne[2], 1);
        for (int i = 0; i < 5; i++) {
            x = cross_block(ctx, "mtdec.blocks." + std::to_string(i), x, y);
#ifdef MAP_ANYTHING_ENABLE_DUMP
            if (MAP_DUMP_ACTIVE())
                dump_node(x, "pi3x_mtblk" + std::to_string(i));
#endif
        }
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) dump_node(x, "pi3x_mtdec_out");
#endif
        // the ContextOnlyTransformerDecoder ends with its own linear_out
        // (512->512) BEFORE the metric head; omitting it gave metric 3.8x
        x = lin(ctx, x, w("mtdec.linear_out.weight"),
                w("mtdec.linear_out.bias"));
        return lin(ctx, x, w("mth.weight"), w("mth.bias"));  // {1,1,1}
    }

    bool build_graph() override {
        prepare_host_inputs_vggt(false, 0.1f);
        const size_t kNodes = 90000;
        ggml_init_params gp{ggml_tensor_overhead() * kNodes +
                                    ggml_graph_overhead_custom(kNodes, false),
                            nullptr, true};
        gctx = ggml_init(gp);
        if (!gctx) return false;
        // static inputs get their OWN backend buffer (see build_stage1_vggt
        // for why: the gallocr pool recycles their space mid-graph after the
        // last use, so "upload once" would not survive a second run())
        ictx = ggml_init({ggml_tensor_overhead() * 64, nullptr, true});
        if (!ictx) return false;
        // static uv tables for the conv heads: level grids and the full
        // resolution; aspect ratio = image w/h (pi3x conv_head convention)
        const int dims[3] = {Wp, 2 * Wp, 4 * Wp};
        const int dys[3] = {Hp, 2 * Hp, 4 * Hp};
        const float ar = (float)W / (float)H;
        for (int i = 0; i < 3; i++) {
            host_uv[i] = uv_table(dims[i], dys[i], ar);
            in_uv[i] = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, dims[i], dys[i],
                                          2, 1);
            ggml_set_input(in_uv[i]);
            in_uv_c[i] = in_uv[i];
        }
        host_uv[3] = uv_table(W, H, ar);
        in_uv[3] = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, W, H, 2, 1);
        ggml_set_input(in_uv[3]);
        in_uv_c[3] = in_uv[3];
        build_rope_rows_metric();
        in_cos_m = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, 512 / nh_m, nh_m,
                                      Nagg * S, 1);
        in_sin_m = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, 512 / nh_m, nh_m,
                                      Nagg * S, 1);
        ggml_set_input(in_cos_m);
        ggml_set_input(in_sin_m);
        return build_stage1_pi3x();
    }

    bool build_stage1_pi3x() {
        log_info("pi3x: creating graph inputs");
        in_images = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, W, H, 3, S);
        ggml_set_input(in_images);
        in_mean = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, 1, 1, 3, 1);
        in_std = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, 1, 1, 3, 1);
        const int dh_l = C / nh;
        in_cos_v = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, dh_l, nh, Nagg, S);
        in_sin_v = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, dh_l, nh, Nagg, S);
        ggml_set_input(in_cos_v);
        ggml_set_input(in_sin_v);
        in_pos_embed = ggml_new_tensor_3d(ictx, GGML_TYPE_F32, C, 1 + P, 1);
        ggml_set_input(in_pos_embed);

        // ---- encoder + decoder: identical to pi3 (register concat, 36
        // alternating blocks, last-two-concat) ----
        ggml_tensor* img =
                ggml_div(gctx, ggml_sub(gctx, in_images, in_mean), in_std);
        ggml_tensor* patches = backbone_vggt(gctx, img);  // {C, P, S}
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) dump_node(patches, "pi3x_patches");
#endif
        ggml_tensor* reg = w("dec.reg_token");
        ggml_tensor* reg_all = ggml_repeat(
                gctx,
                ggml_view_3d(gctx, reg, C, 5, 1, reg->nb[1], reg->nb[2], 0),
                ggml_new_tensor_3d(gctx, GGML_TYPE_F32, C, 5, S));
        ggml_tensor* tokens = ggml_concat(gctx, reg_all, patches, 1);
        ggml_tensor* h_penult = nullptr;
        for (int i = 0; i < m->meta.aa_depth; i++) {
            ggml_tensor* x;
            int T, B;
            if (i % 2 == 0) {
                x = tokens;
                T = Nagg;
                B = S;
            } else {
                x = ggml_reshape_3d(gctx, tokens, C, Nagg * S, 1);
                T = Nagg * S;
                B = 1;
            }
            x = block_v(gctx, x, "dec." + std::to_string(i), C, true, in_cos_v,
                        in_sin_v, T, B);
            if (i % 2 == 1) x = ggml_reshape_3d(gctx, x, C, Nagg, S);
            tokens = x;
#ifdef MAP_ANYTHING_ENABLE_DUMP
            if (MAP_DUMP_ACTIVE() && i <= 4)
                dump_node(tokens, ("pi3x_dec" + std::to_string(i)).c_str());
#endif
            if (i == m->meta.aa_depth - 2) h_penult = tokens;
        }
        ggml_tensor* hidden2 = ggml_concat(gctx, h_penult, tokens, 0);
        ggml_set_output(hidden2);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) dump_node(hidden2, "pi3x_hidden2");
#endif

        // ---- heads ----
        ggml_tensor* pt = transformer_decoder(gctx, hidden2, "pdec", 1024);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) dump_node(pt, "pi3x_pdec_out");
#endif
        auto pxy_z = conv_head(gctx, pt, "phead", nullptr, 2);
        ggml_tensor* xy = pxy_z[0];  // {W, H, 2, S}
        ggml_tensor* z = pxy_z[1];   // {W, H, 1, S}
        ggml_tensor* zc = ggml_exp(gctx, ggml_clamp(gctx, z, -1e30f, 15.0f));
        ggml_tensor* xc = ggml_cont(
                gctx, ggml_view_4d(gctx, xy, xy->ne[0], xy->ne[1], 1, S,
                                   xy->nb[1], xy->nb[2], xy->nb[3], 0));
        ggml_tensor* yc = ggml_cont(
                gctx, ggml_view_4d(gctx, xy, xy->ne[0], xy->ne[1], 1, S,
                                   xy->nb[1], xy->nb[2], xy->nb[3], xy->nb[2]));
        out_local = ggml_concat(gctx,
                                ggml_concat(gctx, ggml_mul(gctx, xc, zc),
                                            ggml_mul(gctx, yc, zc), 2),
                                zc, 2);  // {W, H, 3, S}

        ggml_tensor* cf = transformer_decoder(gctx, hidden2, "cdec", 1024);
        auto couts = conv_head(gctx, cf, "chead", nullptr, 1);
        out_confp = couts[0];  // {W, H, 1, S}

        ggml_tensor* cm = transformer_decoder(gctx, hidden2, "mdec", 512);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        if (MAP_DUMP_ACTIVE()) dump_node(cm, "pi3x_mdec_out");
#endif
        out_pose_raw = camera_head_pi3(gctx, cm);  // {12, 1, S}

        out_metric = metric_branch(gctx, hidden2);  // {1,1,1}
        ggml_set_output(out_local);
        ggml_set_output(out_confp);
        ggml_set_output(out_pose_raw);
        ggml_set_output(out_metric);

        const size_t kNodesV = 90000;
        graph = ggml_new_graph_custom(gctx, kNodesV, false);
        ggml_build_forward_expand(graph, out_local);
        ggml_build_forward_expand(graph, out_confp);
        ggml_build_forward_expand(graph, out_pose_raw);
        ggml_build_forward_expand(graph, out_metric);
#ifdef MAP_ANYTHING_ENABLE_DUMP
        for (size_t i = 0; i < dbg_nodes.size(); i++)
            ggml_build_forward_expand(graph, dbg_nodes[i]);
#endif
        galloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(be.handle));
        // give the static inputs their own buffer BEFORE the graph pool is
        // allocated, so gallocr skips them (they must survive across runs)
        ibuf = ggml_backend_alloc_ctx_tensors(ictx, be.handle);
        if (!ibuf) return false;
        if (!alloc_graph("pi3x")) return false;
        log_info("pi3x: graph allocation done, uploading static inputs");
        ggml_backend_tensor_set(in_mean, m->meta.img_mean, 0,
                                3 * sizeof(float));
        ggml_backend_tensor_set(in_std, m->meta.img_std, 0, 3 * sizeof(float));
        ggml_backend_tensor_set(in_pos_embed, host_pos_embed.data(), 0,
                                host_pos_embed.size() * sizeof(float));
        // decoder rope tables: WITHOUT these the decoder attention runs
        // position-free (q_rope all zeros), which silently degrades every
        // block and explodes by hidden2 (the root cause of the pi3x
        // "rot 24.85deg stable" symptom)
        ggml_backend_tensor_set(in_cos_v, host_cos_v.data(), 0,
                                host_cos_v.size() * sizeof(float));
        ggml_backend_tensor_set(in_sin_v, host_sin_v.data(), 0,
                                host_sin_v.size() * sizeof(float));
        for (int i = 0; i < 4; i++)
            ggml_backend_tensor_set(in_uv_c[i], host_uv[i].data(), 0,
                                    host_uv[i].size() * sizeof(float));
        ggml_backend_tensor_set(in_cos_m, host_cos_m.data(), 0,
                                host_cos_m.size() * sizeof(float));
        ggml_backend_tensor_set(in_sin_m, host_sin_m.data(), 0,
                                host_sin_m.size() * sizeof(float));
        log_info("graph built: %d nodes (pi3x)", ggml_graph_n_nodes(graph));
        return true;
    }

    bool run(const float* imgs, VGGTOutputs& out) override {
        if (!graph) return false;
        ggml_backend_tensor_set(in_images, imgs, 0,
                                (size_t)S * 3 * H * W * sizeof(float));
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(be.handle, graph);
        out.timing_ms["inference_total"] =
                std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
#ifdef MAP_ANYTHING_ENABLE_DUMP
        for (size_t i = 0; i < dbg_nodes.size(); i++) {
            std::vector<float> tmp(ggml_nelements(dbg_nodes[i]));
            ggml_backend_tensor_get(dbg_nodes[i], tmp.data(), 0,
                                    tmp.size() * sizeof(float));
            FILE* fp = fopen(dbg_names[i].c_str(), "wb");
            if (fp) {
                fwrite(tmp.data(), 4, tmp.size(), fp);
                fclose(fp);
            }
        }
#endif
        const int Ho = H, Wo = W;
        auto fetch = [&](ggml_tensor* t, int ch, std::vector<float>& dst) {
            std::vector<float> tmp((size_t)Wo * Ho * ch * S);
            ggml_backend_tensor_get(t, tmp.data(), 0,
                                    tmp.size() * sizeof(float));
            dst.resize((size_t)S * Ho * Wo * ch);
            for (int s = 0; s < S; s++)
                for (int hh = 0; hh < Ho; hh++)
                    for (int ww = 0; ww < Wo; ww++)
                        for (int c = 0; c < ch; c++)
                            dst[(((size_t)s * Ho + hh) * Wo + ww) * ch + c] =
                                    tmp[(size_t)ww + (size_t)Wo * hh +
                                        (size_t)Wo * Ho * c +
                                        (size_t)Wo * Ho * ch * s];
        };
        // metric scale = exp(raw)
        float mraw = 0.0f;
        ggml_backend_tensor_get(out_metric, &mraw, 0, sizeof(float));
        out.metric_scale = std::exp(mraw);

        fetch(out_local, 3, out.local_points);
        for (size_t i = 0; i < out.local_points.size(); i++)
            out.local_points[i] *=
                    out.metric_scale;  // official: local x metric

        // pi3x depth: z of the (metric) local pointmap — same contract as
        // pi3, result_depth = camera-frame z (aicore/map_anything_capi.h).
        out.depth.resize((size_t)S * Ho * Wo);
        for (size_t i = 0; i < out.depth.size(); ++i)
            out.depth[i] = out.local_points[3 * i + 2];

        fetch(out_confp, 1, out.conf);  // continuous quality
        // raw camera params -> host SVD orthogonalization + 4x4 poses;
        // translations are converted to metric like the official forward
        out.pose_raw.resize((size_t)12 * S);
        ggml_backend_tensor_get(out_pose_raw, out.pose_raw.data(), 0,
                                (size_t)12 * S * sizeof(float));
        out.camera_poses.assign((size_t)16 * S, 0.0f);
        for (int s = 0; s < S; s++) {
            float pose16[16];
            pi3_svd_pose(&out.pose_raw[(size_t)12 * s], pose16);
            pose16[3] *= out.metric_scale;
            pose16[7] *= out.metric_scale;
            pose16[11] *= out.metric_scale;
            for (int i = 0; i < 16; i++)
                out.camera_poses[(size_t)16 * s + i] = pose16[i];
        }
        // global points: camera_poses @ homogenize(local_points) (metric)
        out.world_points.assign((size_t)S * Ho * Wo * 3, 0.0f);
        for (int s = 0; s < S; s++) {
            const float* R = &out.camera_poses[(size_t)16 * s];
            for (int h = 0; h < Ho; h++) {
                for (int w2 = 0; w2 < Wo; w2++) {
                    const size_t o = (((size_t)s * Ho + h) * Wo + w2) * 3;
                    const float x = out.local_points[o];
                    const float y = out.local_points[o + 1];
                    const float z = out.local_points[o + 2];
                    for (int r = 0; r < 3; r++)
                        out.world_points[o + r] =
                                R[r * 4 + 0] * x + R[r * 4 + 1] * y +
                                R[r * 4 + 2] * z + R[r * 4 + 3];
                }
            }
        }
        return true;
    }
};

static RegisterBuilder g_reg_pi3x("pi3x", [] {
    return std::unique_ptr<IGraphBuilder>(new VGGTRuntime::Pi3XImpl());
});

static RegisterBuilder g_reg_vggt("vggt", [] {
    return std::unique_ptr<IGraphBuilder>(new VGGTRuntime::VGGTImpl());
});

static RegisterBuilder g_reg_vggt_omega("vggt_omega", [] {
    return std::unique_ptr<IGraphBuilder>(new VGGTRuntime::Impl());
});

std::unique_ptr<IGraphBuilder> create_builder(const std::string& arch) {
    auto& reg = builder_registry();
    auto it = reg.find(arch);
    if (it == reg.end()) {
        std::string known;
        for (const auto& [k, _] : reg) known += (known.empty() ? "" : ", ") + k;
        log_error("unknown architecture '%s' (registered: %s)", arch.c_str(),
                  known.c_str());
        return nullptr;
    }
    return it->second();
}

VGGTRuntime::VGGTRuntime(const GGUFModel& model,
                         Backend backend,
                         int n_views,
                         int height,
                         int width,
                         const RuntimeOptions& opts)
    : s_(n_views), h_(height), w_(width) {
    impl_.reset(dynamic_cast<Impl*>(
            create_builder(model.meta.architecture).release()));
    if (!impl_) {
        // create_builder already logs unknown architectures; this catches a
        // builder registered under the right key but not derived from Impl
        // (the shared-component base every builder must go through)
        log_error("builder for '%s' does not derive VGGTRuntime::Impl",
                  model.meta.architecture.c_str());
        return;
    }
    if (!impl_->init(model, backend, n_views, height, width, opts)) {
        out_of_memory_ = impl_->out_of_memory;
        error_ = impl_->init_error;
        impl_.reset();
    }
}

bool VGGTRuntime::run(const float* images_f32, VGGTOutputs& out) {
    if (!impl_ || !impl_->graph) return false;
    return impl_->run(images_f32, out);
}

// Out-of-line so translation units holding only the hpp (e.g. the test
// binary) can destroy the runtime; ~Impl does the actual cleanup.
VGGTRuntime::~VGGTRuntime() = default;

}  // namespace map_anything
