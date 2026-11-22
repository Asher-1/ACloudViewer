// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/wevisone/wevisone_mask_decoder_graph.h"

#include "common/data_root_util.hpp"

// vmsLn discovery is POSIX-only (dlopen + glob); other platforms fall back
// to the double-log path below.
#if defined(__linux__) || defined(__APPLE__)
#define WWV_HAVE_DLOPEN 1
#include <dlfcn.h>
#include <glob.h>
#endif

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>

namespace wwv {

// Option-injected libtorch path (see set_mkl_log_lib_path); read once by
// the vmsLn resolver below. No direct environment access: the pip-layout
// fallback resolves the user home through the common data-root interface.
static std::string g_mkl_log_lib_path;
void set_mkl_log_lib_path(const std::string& path) {
    g_mkl_log_lib_path = path;
}

// 1-element f32 leaf constant. The graph context is no_alloc, so the value
// is staged here and uploaded by the caller after ggml_gallocr_alloc_graph
// (same contract as every other host-provided leaf).
static std::vector<std::pair<ggml_tensor*, float>> g_wwv_f32_leaves;
void wwv_clear_f32_leaves() { g_wwv_f32_leaves.clear(); }
const std::vector<std::pair<ggml_tensor*, float>>& wwv_f32_leaves() {
    return g_wwv_f32_leaves;
}

// PixelShuffle perm-table staging (same upload contract as f32 leaves).
static std::vector<std::pair<ggml_tensor*, std::vector<int32_t>>>
        g_wwv_perm_staged;
void wwv_clear_perm_staged() { g_wwv_perm_staged.clear(); }
const std::vector<std::pair<ggml_tensor*, std::vector<int32_t>>>&
wwv_perm_staged() {
    return g_wwv_perm_staged;
}
ggml_tensor* f32_leaf(ggml_context* ctx0, float v) {
    // a CONSTANT, not an input: gallocr allocates it with the graph and the
    // caller uploads the staged value after ggml_gallocr_alloc_graph.
    ggml_tensor* t = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    g_wwv_f32_leaves.emplace_back(t, v);
    return t;
}

// (1, N) f32 vector leaf constant (same upload contract as f32_leaf).
// Column-vector shape (1, N): ggml broadcast is per-dim, so slot-axis
// weights must live on ne[1] to stretch along ne[0] = d.
static std::vector<std::pair<ggml_tensor*, std::vector<float>>>
        g_wwv_f32vec_staged;
void wwv_clear_f32vec_staged() { g_wwv_f32vec_staged.clear(); }
const std::vector<std::pair<ggml_tensor*, std::vector<float>>>&
wwv_f32vec_staged() {
    return g_wwv_f32vec_staged;
}
ggml_tensor* f32_vec_leaf(ggml_context* ctx0, std::vector<float> v) {
    ggml_tensor* t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1,
                                        static_cast<int64_t>(v.size()));
    g_wwv_f32vec_staged.emplace_back(t, std::move(v));
    return t;
}

namespace {

// Bit-exact-with-torch float log. torch's AVX2 log kernel dispatches to
// MKL VML vmsLn with a BY-VALUE ABI — vmsLn(int n, const float* a,
// float* r, int mode) — confirmed by disassembly (test %edi,%edi on
// entry; torch passes mode 0x140102 in ecx). mode 0 (VML_HA) reproduces
// torch.log bit-for-bit on a 2M-input fuzz; (float)log(double) agrees
// only ~98.2% and the residual 1-8 ulp blows up to radian-scale phase
// differences inside the Fourier encoder (frequencies up to 2^31).
// The library is resolved once through the option-injected path or the
// common pip torch layouts (user home comes from the shared data-root
// interface, never a task-local getenv); callers fall back to the glibc
// double log when absent.
bool mkl_logf_exact(const float* x, float* y, int n) {
#if defined(WWV_HAVE_DLOPEN)
    using VmsLnFn = void (*)(int, const float*, float*, int);
    static VmsLnFn fn = nullptr;
    static bool resolved = false;
    if (!resolved) {
        resolved = true;
        void* h = nullptr;
        if (!g_mkl_log_lib_path.empty()) {
            h = dlopen(g_mkl_log_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        }
        if (h == nullptr) {
            const std::string home = aicore::home_dir();
            const std::string pats[] = {
                    home + "/.local/lib/python3*/site-packages/torch/lib/"
                           "libtorch_cpu.so",
                    "/usr/lib/python3*/site-packages/torch/lib/libtorch_cpu.so",
                    "/usr/local/lib/python3*/dist-packages/torch/lib/"
                    "libtorch_cpu.so",
            };
            for (const std::string& pat : pats) {
                glob_t g;
                if (::glob(pat.c_str(), 0, nullptr, &g) == 0 &&
                    g.gl_pathc > 0) {
                    h = dlopen(g.gl_pathv[0], RTLD_NOW | RTLD_LOCAL);
                }
                globfree(&g);
                if (h != nullptr) break;
            }
        }
        if (h != nullptr) {
            fn = reinterpret_cast<VmsLnFn>(dlsym(h, "vmsLn"));
        }
    }
    if (fn == nullptr) return false;
    fn(n, x, y, 0);  // mode 0 = VML_HA: bit-exact with torch.log
    return true;
#else
    (void)x;
    (void)y;
    (void)n;
    return false;
#endif
}

}  // namespace

namespace {

// conv_transpose_2d_p0 runs without padding: k4 s2 gives (2*in + 2) —
// crop back to the torch k4/s2/p1 output (2*in). The torch kernel indexes
// its output through the padded input, which shifts every torch sample to
// ggml sample o + p: out_torch[o] = out_ggml[o + p] (derived and verified
// on H=1/H=2 fold-backs). So the window starts at p in BOTH x and y —
// offset = p*(nb0 + nb1) — and runs 2*in samples: [p, p + 2*in) of the
// (2*in + 2) ggml output. Cropping from 0 shifts the whole map by one
// pixel and decorrelates it (corr ~0.02 observed).
ggml_tensor* convt_crop(ggml_context* ctx0,
                        ggml_tensor* t,
                        int64_t w,
                        int64_t h,
                        int64_t c,
                        int64_t n) {
    const size_t p = 1;  // torch padding of the k4/s2 ConvTranspose2d
    ggml_tensor* v = ggml_view_4d(ctx0, t, w, h, c, n, t->nb[1], t->nb[2],
                                  t->nb[3], p * (t->nb[0] + t->nb[1]));
    return ggml_cont(ctx0, v);
}

constexpr float kGnEps = 1e-5f;  // torch GroupNorm default

ggml_tensor* prec(ggml_context* ctx0, ggml_tensor* a, ggml_tensor* b) {
    ggml_tensor* t = ggml_mul_mat(ctx0, a, b);
    ggml_mul_mat_set_prec(t, GGML_PREC_F32);
    return t;
}

ggml_tensor* ln(ggml_context* ctx0,
                const char* tag,
                ggml_tensor* x,
                ggml_tensor* g,
                ggml_tensor* b) {
    ggml_tensor* cur = ggml_norm(ctx0, x, 1e-5f);
    if (g != nullptr) cur = ggml_mul(ctx0, cur, g);
    if (b != nullptr) cur = ggml_add(ctx0, cur, b);
    return cur;
}

ggml_tensor* gn(ggml_context* ctx0,
                ggml_tensor* x,
                int32_t groups,
                ggml_tensor* g,
                ggml_tensor* b) {
    ggml_tensor* cur = ggml_group_norm(ctx0, x, groups, kGnEps);
    if (g != nullptr || b != nullptr) {
        // ggml broadcast is per-dim divide: the (C,) affine must become
        // (1, 1, C, 1) to align with the (W, H, C, N) activation.
        const int64_t c = x->ne[2];
        ggml_tensor* ga = nullptr;
        ggml_tensor* ba = nullptr;
        if (g != nullptr) {
            ga = ggml_reshape_4d(ctx0, g, 1, 1, c, 1);
            cur = ggml_mul(ctx0, cur, ga);
        }
        if (b != nullptr) {
            ba = ggml_reshape_4d(ctx0, b, 1, 1, c, 1);
            cur = ggml_add(ctx0, cur, ba);
        }
    }
    return cur;
}

// torch Conv2d. GGUF weight ne = (Kw, Kh, Cin, Cout) per ggml conv layout.
// The (Cout,) bias must be reshaped (1, 1, Cout, 1) — ggml broadcast is
// per-dim divide and (Cout,) would try to divide the OW axis.
ggml_tensor* conv2d(ggml_context* ctx0,
                    ggml_tensor* w,
                    ggml_tensor* b,
                    ggml_tensor* x,
                    int stride,
                    int pad) {
    ggml_tensor* cur = ggml_conv_2d(ctx0, w, x, stride, stride, pad, pad, 1, 1);
    if (b != nullptr) {
        cur = ggml_add(ctx0, cur, ggml_reshape_4d(ctx0, b, 1, 1, b->ne[0], 1));
    }
    return cur;
}

// 1x1 conv as per-pixel GEMM CONTRACTING THE CHANNEL AXIS. mul_mat(a, b)
// contracts ne[0] of both operands, so the map must be brought to
// (C, W*H*N) rows first; the torch (out, in) conv weight is stored
// transposed by the converter as the (C, out) GEMM weight. Fused weights
// arrive column-sliced ((C, block)) from the cascade fuse helper.
ggml_tensor* conv1x1(ggml_context* ctx0, ggml_tensor* w2d, ggml_tensor* x) {
    const int64_t w = x->ne[0];
    const int64_t h = x->ne[1];
    const int64_t n = x->ne[3] > 0 ? x->ne[3] : 1;
    // ggml permute semantics: old axis i moves TO axis_i (ggml.c:
    // ne[axis_i] = a->ne[i]). (C, W, H) keeps the token flattening
    // row-major (t = y*W + x) after the (C, HW) reshape — verified against
    // torch and a standalone ggml probe at 96x64.
    ggml_tensor* cur = ggml_permute(ctx0, x, 1, 2, 0, 3);  // (C, W, H, N)
    cur = ggml_cont(ctx0, cur);
    cur = ggml_reshape_2d(ctx0, cur, x->ne[2], w * h * n);
    cur = prec(ctx0, w2d, cur);  // mul_mat result (M=192, N=H*W*N):
    // logical (m, n) at m + M*n — n is the ROW-MAJOR token index, so the
    // epilogue reshape is (out, W, H) and the permute walks (x, y, out).
    cur = ggml_reshape_4d(ctx0, cur, w2d->ne[1], w, h, n);
    // (out, W, H, N) -> (W, H, out, N): old0 -> dest2, old1 -> dest0,
    // old2 -> dest1 (p_i is the DESTINATION axis of old axis i).
    cur = ggml_permute(ctx0, cur, 2, 0, 1, 3);
    return ggml_cont(ctx0, cur);
}

ggml_tensor* resize(ggml_context* ctx0, ggml_tensor* x, int64_t w, int64_t h) {
    if (x->ne[0] == w && x->ne[1] == h) return x;
    return ggml_interpolate(ctx0, x, w, h, x->ne[2],
                            x->ne[3] > 0 ? x->ne[3] : 1,
                            GGML_SCALE_MODE_BILINEAR);
}

// torch F.adaptive_avg_pool2d((C, H, W) -> (OH, OW)) with integer bin
// sizes: per-channel block mean over the (H/OH, W/OW) window. Same pattern
// as depthwise3x3 — im2col patches of the (KW, KH) window, then a plain
// tap sum (no weights). x: (W, H, C, N); result: (OW, OH, C, N).
ggml_tensor* adaptive_pool(ggml_context* ctx0,
                           ggml_tensor* x,
                           int64_t ow,
                           int64_t oh) {
    const int64_t w = x->ne[0], h = x->ne[1], c = x->ne[2], n = x->ne[3];
    const int64_t kw = w / ow, kh = h / oh;
    // im2col pins N == 1 (GGML_ASSERT(b->ne[3] == 1)): fold (c, n) into a
    // single channel axis — pooling is spatial-only per (c, n) pair.
    ggml_tensor* xc = (n > 1) ? ggml_view_4d(ctx0, x, w, h, c * n, 1, x->nb[1],
                                             x->nb[2], 0, 0)
                              : x;
    const int64_t cn = c * n;
    ggml_tensor* dummy = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, kw, kh, cn, 1);
    ggml_tensor* patches = ggml_im2col(ctx0, dummy, xc, kw, kh, 0, 0, 1, 1,
                                       true, GGML_TYPE_F32);
    // (K*K*cn, OW, OH, 1): flat = k + K*K*cn_idx + K*K*cn*l,
    // l = ow + OW*oh
    const int64_t l = patches->ne[1] * patches->ne[2] * patches->ne[3];
    ggml_tensor* tap = ggml_reshape_3d(ctx0, patches, kw * kh, cn, l);
    tap = ggml_sum_rows(ctx0, tap);  // (1, cn, L)
    tap = ggml_scale(ctx0, tap, 1.0f / static_cast<float>(kw * kh));
    tap = ggml_cont(ctx0, ggml_view_2d(ctx0, tap, cn, l,
                                       ggml_row_size(GGML_TYPE_F32, cn),
                                       0));  // (cn, L)
    tap = ggml_reshape_4d(ctx0, tap, cn, patches->ne[1], patches->ne[2],
                          patches->ne[3]);
    // (cn, OW, OH, 1) -> (OW, OH, cn, 1): old0 -> dest2, old1 -> dest0,
    // old2 -> dest1 (perm[i] is the DESTINATION axis of old axis i) —
    // the (W, H, C, N) map order.
    ggml_tensor* pooled = ggml_cont(ctx0, ggml_permute(ctx0, tap, 2, 0, 1, 3));
    if (n > 1) {
        // unfold (c*n) back to (c, n) with n on the batch axis
        pooled = ggml_view_4d(ctx0, pooled, ow, oh, c, n, pooled->nb[1],
                              pooled->nb[2], pooled->nb[2] * c, 0);
    }
    return pooled;
}

// torch F.interpolate on a (1, C, H, W) map squeezed from (W, H, C, 1).
ggml_tensor* resize_map(ggml_context* ctx0,
                        ggml_tensor* x,
                        int64_t w,
                        int64_t h,
                        bool force_bilinear = false) {
    if (x->ne[0] == w && x->ne[1] == h) return x;
    // torch resize: pool_h = min(src, dst) — a downscale is
    // adaptive_avg_pool2d ONLY (the bilinear branch is skipped when the
    // pool already reached dst), an upscale is bilinear interpolate only.
    // force_bilinear selects the FullCanvasAlphaCascadeHead._resize
    // semantics instead: F.interpolate bilinear for ANY size change.
    if (!force_bilinear && x->ne[0] > w && x->ne[1] > h && x->ne[0] % w == 0 &&
        x->ne[1] % h == 0) {
        return adaptive_pool(ctx0, x, w, h);
    }
    // preserve the Q batch axis (the cascade resizes per-query coarse maps)
    return ggml_interpolate(ctx0, x, w, h, x->ne[2], x->ne[3],
                            GGML_SCALE_MODE_BILINEAR);
}

void build_pixel_shuffle_perm(int64_t w,
                              int64_t h,
                              int64_t c,
                              std::vector<int32_t>* perm) {
    perm->resize(static_cast<size_t>(c * w * h * 4));
    for (int64_t cc = 0; cc < c; ++cc) {
        for (int64_t y2 = 0; y2 < h * 2; ++y2) {
            for (int64_t x2 = 0; x2 < w * 2; ++x2) {
                const int64_t y = y2 / 2;
                const int64_t x = x2 / 2;
                const int64_t sy = y2 % 2;
                const int64_t sx = x2 % 2;
                const int64_t ch = cc * 4 + sy * 2 + sx;
                (*perm)[static_cast<size_t>(x2 +
                                            (w * 2) * (y2 + (h * 2) * cc))] =
                        static_cast<int32_t>(x + w * (y + h * ch));
            }
        }
    }
}

ggml_tensor* pixel_shuffle(
        ggml_context* ctx0,
        std::unordered_map<std::string, std::vector<int32_t>>* perm_tables,
        std::vector<std::pair<ggml_tensor*, std::vector<int32_t>>>*,
        ggml_tensor* x) {
    const int64_t w = x->ne[0];
    const int64_t h = x->ne[1];
    const int64_t c = x->ne[2] / 4;
    const std::string key = std::to_string(w) + "x" + std::to_string(h) + "x" +
                            std::to_string(c);
    auto it = perm_tables->find(key);
    if (it == perm_tables->end()) {
        std::vector<int32_t> table;
        build_pixel_shuffle_perm(w, h, c, &table);
        it = perm_tables->emplace(key, std::move(table)).first;
    }
    // i32 perm leaf: staged into the global list, uploaded post-gallocr.
    ggml_tensor* idx = ggml_new_tensor_1d(
            ctx0, GGML_TYPE_I32, static_cast<int64_t>(it->second.size()));
    g_wwv_perm_staged.emplace_back(idx, it->second);
    ggml_tensor* flat = ggml_reshape_2d(ctx0, x, 1, ggml_nelements(x));
    ggml_tensor* shuffled = ggml_get_rows(ctx0, flat, idx);
    return ggml_reshape_4d(ctx0, shuffled, w * 2, h * 2, c, 1);
}

// Depthwise 3x3 (groups == channels, bias-free) via patch-only im2col:
// patches of different channels never mix, so a broadcast mul + sum over
// the 9 tap rows is exactly torch's depthwise conv.
ggml_tensor* depthwise3x3(ggml_context* ctx0,
                          ggml_tensor* w9c,
                          ggml_tensor* x) {
    // x: (W, H, C, N); w9c: (9, C) tap weights (k = ky*3 + kx).
    const int64_t c = x->ne[2];
    ggml_tensor* dummy = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 3, 3, c, 1);
    ggml_tensor* patches =
            ggml_im2col(ctx0, dummy, x, 1, 1, 1, 1, 1, 1, true, GGML_TYPE_F32);
    // im2col kernel writes patch taps as ic*(KH*KW) + ky*KW + kx inside a
    // (KW*KH*IC, OW, OH) buffer — flat = k + 9*ic + 9C*l, which IS the
    // (9, C, L) layout the (9, C) weight matrix needs. The old
    // reshape(C,9,L)+permute misread it as C-major and scrambled taps.
    const int64_t l = patches->ne[1] * patches->ne[2];
    ggml_tensor* tap = ggml_reshape_3d(ctx0, patches, 9, c, l);
    tap = ggml_mul(ctx0, tap, w9c);
    tap = ggml_sum_rows(ctx0, tap);  // (1, C, L)
    ggml_tensor* cur =
            ggml_view_2d(ctx0, tap, c, l, ggml_row_size(GGML_TYPE_F32, c), 0);
    cur = ggml_cont(ctx0, cur);
    // (C, L) with flat = c + C*t: transpose to (L, C) whose flat = t + hw*c
    // is exactly the (W, H, C) map order, then reshape to 3-D.
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    cur = ggml_reshape_3d(ctx0, cur, x->ne[0], x->ne[1], c);
    return cur;
}

// torch nn.MultiheadAttention(batch_first) self-attention.
ggml_tensor* self_attention(ggml_context* ctx0,
                            ggml_tensor* target,
                            ggml_tensor* query_pos,
                            ggml_tensor* in_proj_w,
                            ggml_tensor* in_proj_b,
                            ggml_tensor* out_w,
                            ggml_tensor* out_b,
                            int32_t heads) {
    const int64_t d = target->ne[0];
    const int64_t qn = target->ne[1];
    const int64_t hd = d / heads;
    ggml_tensor* qk = ggml_add(ctx0, target, query_pos);
    // nn.MultiheadAttention packs in_proj rows [q; k; v]; q and k project
    // (target + query_pos) but **v projects the raw target** — the value
    // stream carries no positional signal. The old code projected v from
    // qk, injecting pos into the attention output.
    ggml_tensor* wqk =
            ggml_view_2d(ctx0, in_proj_w, d, 2 * d, in_proj_w->nb[1], 0);
    // in_proj rows are [q; k; v] with d*d elements each: the v row block
    // starts after 2*d*d floats (the old 2*d offset landed inside the k
    // block and produced a garbage value projection).
    ggml_tensor* wv = ggml_view_2d(ctx0, in_proj_w, d, d, in_proj_w->nb[1],
                                   2 * d * d * sizeof(float));
    ggml_tensor* qkv = prec(ctx0, wqk, qk);
    ggml_tensor* vproj = prec(ctx0, wv, target);
    if (in_proj_b != nullptr) {
        qkv = ggml_add(ctx0, qkv, ggml_view_1d(ctx0, in_proj_b, 2 * d, 0));
        vproj = ggml_add(ctx0, vproj,
                         ggml_view_1d(ctx0, in_proj_b, d,
                                      ggml_row_size(in_proj_b->type, 2 * d)));
    }
    ggml_tensor* q = ggml_view_2d(ctx0, qkv, d, qn, qkv->nb[1], 0);
    ggml_tensor* k = ggml_view_2d(ctx0, qkv, d, qn, qkv->nb[1],
                                  ggml_row_size(qkv->type, d));
    ggml_tensor* v = vproj;
    auto split_heads = [&](ggml_tensor* t) {
        // t is a strided view of the packed qkv — cont() before reshape.
        ggml_tensor* cur =
                ggml_reshape_3d(ctx0, ggml_cont(ctx0, t), hd, heads, qn);
        // old1 (heads) -> axis2, old2 (qn) -> axis1: (hd, qn, heads)
        return ggml_permute(ctx0, cur, 0, 2, 1, 3);
    };
    // Per-head attention with plain 2-D slices: every mul_mat operand is a
    // standard contiguous row-block view (no 3-D permuted views, whose
    // stride interaction with mul_mat silently corrupted the scores).
    ggml_tensor* qc = ggml_cont(ctx0, q);
    ggml_tensor* kc = ggml_cont(ctx0, k);
    ggml_tensor* vc = ggml_cont(ctx0, v);
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(hd));
    ggml_tensor* flat = nullptr;
    for (int32_t h = 0; h < heads; ++h) {
        const size_t off =
                static_cast<size_t>(h) * hd * ggml_row_size(GGML_TYPE_F32, 1);
        ggml_tensor* q_h = ggml_view_2d(ctx0, qc, hd, qn, qc->nb[1], off);
        ggml_tensor* k_h = ggml_view_2d(ctx0, kc, hd, qn, kc->nb[1], off);
        ggml_tensor* v_h = ggml_view_2d(ctx0, vc, hd, qn, vc->nb[1], off);
        // scores[kv, q] = Σ_dim k_h[dim, kv]·q_h[dim, q]
        ggml_tensor* scores = prec(ctx0, k_h, q_h);
        scores = ggml_scale(ctx0, scores, sm_scale);
        scores = ggml_soft_max(ctx0, scores);  // over ne[0] = qn_kv
        // out[h_idx, q] = Σ_kv v_h[h_idx, kv]·p[kv, q]:
        // mul_mat(a, b) = a^T·b with result (a.ne[1], b.ne[1]) — the value
        // matrix must be materialized transposed on the a side (K = qn_kv).
        ggml_tensor* out_h =
                prec(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v_h)), scores);
        flat = (flat == nullptr) ? out_h : ggml_concat(ctx0, flat, out_h, 0);
    }
    ggml_tensor* res = prec(ctx0, out_w, flat);
    if (out_b != nullptr) res = ggml_add(ctx0, res, out_b);
    return res;
}

// Multi-scale deformable attention for one decoder layer, fully in-graph.
// See msda design note: exact F.grid_sample(align_corners=False, zeros)
// semantics through floor/step/cast ops; gathers via ggml_get_rows per
// level; softmax over (levels*points) per (q, h); aggregation keeps
// (d, q, h) and folds heads back into the channel axis.
ggml_tensor* msda_layer(
        ggml_context* ctx0,
        ggml_tensor* query,
        ggml_tensor* memory,
        ggml_tensor* ref,
        int32_t heads,
        int32_t levels,
        int32_t points,
        int32_t grid_w,
        int32_t grid_h,
        ggml_tensor* val_w,
        ggml_tensor* val_b,
        ggml_tensor* off_w,
        ggml_tensor* off_b,
        ggml_tensor* att_w,
        ggml_tensor* att_b,
        ggml_tensor* out_w,
        ggml_tensor* out_b,
        const std::function<void(const char*, ggml_tensor*)>& tap_fn) {
    const int64_t d = query->ne[0];
    const int64_t qn = query->ne[1];
    const int64_t level_rows = static_cast<int64_t>(grid_w) * grid_h;
    const int64_t hlp = static_cast<int64_t>(heads) * levels * points;

    ggml_tensor* value = prec(ctx0, val_w, memory);
    if (val_b != nullptr) value = ggml_add(ctx0, value, val_b);
    if (tap_fn) tap_fn("msda_value", value);

    ggml_tensor* offs = prec(ctx0, off_w, query);
    if (off_b != nullptr) offs = ggml_add(ctx0, offs, off_b);
    if (tap_fn) tap_fn("msda_offs", offs);
    ggml_tensor* offs_p = ggml_reshape_3d(ctx0, offs, 2, hlp, qn);
    // (1, hlp, Q) strided views: element stride 2 along ne1 (x = even row).
    // Flat slot layout (1, S): even/odd rows of offs_p are strided (1, S)
    // views (element stride 2 along the flat axis).
    ggml_tensor* offs_x = ggml_view_2d(ctx0, offs_p, 1, hlp * qn,
                                       ggml_row_size(GGML_TYPE_F32, 2), 0);
    ggml_tensor* offs_y = ggml_view_2d(ctx0, offs_p, 1, hlp * qn,
                                       ggml_row_size(GGML_TYPE_F32, 2),
                                       ggml_row_size(GGML_TYPE_F32, 1));

    // ref rows (cx, cy, bw, bh) broadcast over (hlp, q). The slot layout is
    // f + hlp*q, so the box row must tile the SLOT axis while keeping the
    // QUERY axis: a flat (1, hlp*qn) repeat tiles k mod qn and hands query
    // k%6 a wrong box. Reshape to (1, 1, qn), repeat over the middle axis,
    // then flatten back to (1, hlp*qn).
    auto ref_row = [&](int32_t i) {
        ggml_tensor* row =
                ggml_cont(ctx0, ggml_view_2d(ctx0, ref, 1, qn, ref->nb[1],
                                             ggml_row_size(GGML_TYPE_F32, i)));
        row = ggml_reshape_3d(ctx0, row, 1, 1, qn);
        row = ggml_repeat(ctx0, row,
                          ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, hlp, qn));
        return ggml_reshape_2d(ctx0, row, 1, hlp * qn);
    };
    ggml_tensor* cx_b = ref_row(0);
    ggml_tensor* cy_b = ref_row(1);
    ggml_tensor* scale =
            ggml_repeat(ctx0, f32_leaf(ctx0, 0.5f / static_cast<float>(points)),
                        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, hlp * qn));
    ggml_tensor* loc_x =
            ggml_add(ctx0, cx_b,
                     ggml_mul(ctx0, offs_x, ggml_mul(ctx0, ref_row(2), scale)));
    ggml_tensor* loc_y =
            ggml_add(ctx0, cy_b,
                     ggml_mul(ctx0, offs_y, ggml_mul(ctx0, ref_row(3), scale)));

    // grid_sample(align_corners=False, zeros): px = loc * W - 0.5.
    ggml_tensor* px =
            ggml_sub(ctx0, ggml_scale(ctx0, loc_x, static_cast<float>(grid_w)),
                     f32_leaf(ctx0, 0.5f));
    ggml_tensor* py =
            ggml_sub(ctx0, ggml_scale(ctx0, loc_y, static_cast<float>(grid_h)),
                     f32_leaf(ctx0, 0.5f));
    if (tap_fn) {
        tap_fn("msda_px", px);
        tap_fn("msda_py", py);
        tap_fn("msda_locx", loc_x);
        tap_fn("msda_refw", ref_row(2));
    }
    ggml_tensor* ix = ggml_cast(ctx0, ggml_floor(ctx0, px), GGML_TYPE_I32);
    ggml_tensor* iy = ggml_cast(ctx0, ggml_floor(ctx0, py), GGML_TYPE_I32);
    ggml_tensor* fx = ggml_sub(ctx0, px, ggml_cast(ctx0, ix, GGML_TYPE_F32));
    ggml_tensor* fy = ggml_sub(ctx0, py, ggml_cast(ctx0, iy, GGML_TYPE_F32));
    ggml_tensor* one = f32_leaf(ctx0, 1.0f);
    ggml_tensor* ix1 = ggml_cast(
            ctx0, ggml_add(ctx0, ggml_cast(ctx0, ix, GGML_TYPE_F32), one),
            GGML_TYPE_I32);
    ggml_tensor* iy1 = ggml_cast(
            ctx0, ggml_add(ctx0, ggml_cast(ctx0, iy, GGML_TYPE_F32), one),
            GGML_TYPE_I32);
    ggml_tensor* ix_f = ggml_cast(ctx0, ix, GGML_TYPE_F32);
    ggml_tensor* iy_f = ggml_cast(ctx0, iy, GGML_TYPE_F32);
    ggml_tensor* ix1_f = ggml_cast(ctx0, ix1, GGML_TYPE_F32);
    ggml_tensor* iy1_f = ggml_cast(ctx0, iy1, GGML_TYPE_F32);

    // row index = iy * W + ix.
    auto row_of = [&](ggml_tensor* xv, ggml_tensor* yv) {
        return ggml_cast(
                ctx0,
                ggml_add(ctx0, ggml_scale(ctx0, yv, static_cast<float>(grid_w)),
                         xv),
                GGML_TYPE_I32);
    };
    // padding_mode='zeros': out-of-bounds corners contribute zero (weights
    // unchanged, values dropped), so clamp the gather rows into range and
    // gate each corner weight with an in-bounds mask. op_step is strict
    // (x > 0), so every boundary test needs a 0.5 margin to keep the
    // boundary indices 0 and W-1/H-1 inside the mask.
    auto in_bounds = [&](ggml_tensor* xv, ggml_tensor* yv) {
        // (limit - x) as (-x) + limit: ggml sub requires the scalar LAST.
        ggml_tensor* mx = ggml_mul(
                ctx0, ggml_step(ctx0, ggml_add(ctx0, xv, f32_leaf(ctx0, 0.5f))),
                ggml_step(ctx0,
                          ggml_add(ctx0, ggml_scale(ctx0, xv, -1.0f),
                                   f32_leaf(ctx0, static_cast<float>(grid_w) -
                                                          0.5f))));
        ggml_tensor* my = ggml_mul(
                ctx0, ggml_step(ctx0, ggml_add(ctx0, yv, f32_leaf(ctx0, 0.5f))),
                ggml_step(ctx0,
                          ggml_add(ctx0, ggml_scale(ctx0, yv, -1.0f),
                                   f32_leaf(ctx0, static_cast<float>(grid_h) -
                                                          0.5f))));
        return ggml_mul(ctx0, mx, my);
    };
    auto row_of_c = [&](ggml_tensor* xv, ggml_tensor* yv) {
        ggml_tensor* xc =
                ggml_clamp(ctx0, xv, 0.0f, static_cast<float>(grid_w - 1));
        ggml_tensor* yc =
                ggml_clamp(ctx0, yv, 0.0f, static_cast<float>(grid_h - 1));
        return ggml_cast(
                ctx0,
                ggml_add(ctx0, ggml_scale(ctx0, yc, static_cast<float>(grid_w)),
                         xc),
                GGML_TYPE_I32);
    };
    ggml_tensor* idx_nw = row_of_c(ix_f, iy_f);
    ggml_tensor* idx_ne = row_of_c(ix1_f, iy_f);
    ggml_tensor* idx_sw = row_of_c(ix_f, iy1_f);
    ggml_tensor* idx_se = row_of_c(ix1_f, iy1_f);
    ggml_tensor* mask_nw = in_bounds(ix_f, iy_f);
    ggml_tensor* mask_ne = in_bounds(ix1_f, iy_f);
    ggml_tensor* mask_sw = in_bounds(ix_f, iy1_f);
    ggml_tensor* mask_se = in_bounds(ix1_f, iy1_f);

    // in-bounds for an integer corner v: 0 <= v <= hi-1. op_step is
    // strict (x > 0), so both tests carry a 0.5 margin. The upper test is
    // step((hi - 0.5) - v); ggml sub needs the scalar LAST, hence
    // scale(v, -1) + (hi - 0.5). (The old step(v + 0.5 - hi) fired when
    // v >= hi — INVERTED — which zeroed every in-bounds corner weight and
    // collapsed the whole MSDA output to out_proj(0) + bias.)
    auto bound = [&](ggml_tensor* v, float hi) {
        ggml_tensor* lo =
                ggml_step(ctx0, ggml_add(ctx0, v, f32_leaf(ctx0, 0.5f)));
        ggml_tensor* hi2 =
                ggml_step(ctx0, ggml_add(ctx0, ggml_scale(ctx0, v, -1.0f),
                                         f32_leaf(ctx0, hi - 0.5f)));
        return ggml_mul(ctx0, lo, hi2);
    };
    ggml_tensor* ib_nw = ggml_mul(ctx0, bound(ix_f, static_cast<float>(grid_w)),
                                  bound(iy_f, static_cast<float>(grid_h)));
    ggml_tensor* ib_ne =
            ggml_mul(ctx0, bound(ix1_f, static_cast<float>(grid_w)),
                     bound(iy_f, static_cast<float>(grid_h)));
    ggml_tensor* ib_sw = ggml_mul(ctx0, bound(ix_f, static_cast<float>(grid_w)),
                                  bound(iy1_f, static_cast<float>(grid_h)));
    ggml_tensor* ib_se =
            ggml_mul(ctx0, bound(ix1_f, static_cast<float>(grid_w)),
                     bound(iy1_f, static_cast<float>(grid_h)));

    // 1 - fx: ggml_sub requires the second operand to divide the first,
    // so rewrite as (-fx) + 1 (scalar add broadcasts upward).
    ggml_tensor* one_m_fx = ggml_add(ctx0, ggml_scale(ctx0, fx, -1.0f), one);
    ggml_tensor* one_m_fy = ggml_add(ctx0, ggml_scale(ctx0, fy, -1.0f), one);
    ggml_tensor* w_nw =
            ggml_mul(ctx0, ggml_mul(ctx0, one_m_fx, one_m_fy), ib_nw);
    ggml_tensor* w_ne = ggml_mul(ctx0, ggml_mul(ctx0, fx, one_m_fy), ib_ne);
    ggml_tensor* w_sw = ggml_mul(ctx0, ggml_mul(ctx0, one_m_fx, fy), ib_sw);
    ggml_tensor* w_se = ggml_mul(ctx0, ggml_mul(ctx0, fx, fy), ib_se);

    ggml_tensor* att = prec(ctx0, att_w, query);
    if (att_b != nullptr) att = ggml_add(ctx0, att, att_b);
    // (H*L*P, Q) -> (P, L, H, Q) -> softmax over (P*L) per (h, q).
    ggml_tensor* att_p = ggml_reshape_4d(ctx0, att, points, levels, heads, qn);
    att_p = ggml_reshape_3d(ctx0, att_p, points * levels, heads, qn);
    att_p = ggml_soft_max(ctx0, att_p);
    if (tap_fn) tap_fn("msda_att", att_p);
    ggml_tensor* att_slots = ggml_reshape_2d(ctx0, att_p, 1, hlp * qn);
    // Slot order check: f = (h*L + l)*P + p; (P, L) flatten gives p + P*l
    // per h — matches f within each h block ✓; the gather slot order equals
    // offs_p / weight views ✓.

    auto slotw = [&](ggml_tensor* t) {
        return ggml_reshape_2d(ctx0, t, 1, hlp * qn);
    };
    ggml_tensor* w_nw_s =
            ggml_mul(ctx0, ggml_mul(ctx0, att_slots, slotw(w_nw)), mask_nw);
    ggml_tensor* w_ne_s =
            ggml_mul(ctx0, ggml_mul(ctx0, att_slots, slotw(w_ne)), mask_ne);
    ggml_tensor* w_sw_s =
            ggml_mul(ctx0, ggml_mul(ctx0, att_slots, slotw(w_sw)), mask_sw);
    ggml_tensor* w_se_s =
            ggml_mul(ctx0, ggml_mul(ctx0, att_slots, slotw(w_se)), mask_se);
    if (tap_fn) {
        tap_fn("msda_idxw", ggml_cast(ctx0, idx_nw, GGML_TYPE_F32));
        tap_fn("msda_wnw", w_nw_s);
    }

    const int64_t n_slots = hlp * qn;
    // Per-level four-corner sums stay level-scoped: the slot layout already
    // encodes (h, l, p, q) — slot f = h*(L*P) + l*P + p within HLP slots per
    // query — so level l's result may only land in level-l slots. The old
    // acc = sum_l lvl_l smeared every level over every slot, and the (l, p)
    // fold then re-counted each level points*levels times.
    std::vector<ggml_tensor*> lvl_by_level;
    for (int32_t l = 0; l < levels; ++l) {
        ggml_tensor* val_l =
                ggml_view_2d(ctx0, value, d, level_rows, value->nb[1],
                             static_cast<size_t>(l) * level_rows *
                                     ggml_row_size(value->type, d));
        auto gather = [&](ggml_tensor* idx, ggml_tensor* wt,
                          const char* dbg = nullptr) {
            ggml_tensor* g = ggml_get_rows(ctx0, val_l,
                                           ggml_view_1d(ctx0, idx, n_slots, 0));
            ggml_tensor* m = ggml_mul(ctx0, g, wt);
            // NOTE: any g->data / idx->data inspection in this lambda runs at
            // BUILD time and reads zero-filled pool memory (host path) —
            // post-compute state must be validated through taps only.
            if (tap_fn && dbg != nullptr && l == 0) {
                tap_fn(dbg, m);
            }
            return m;
        };
        ggml_tensor* lvl = gather(idx_nw, w_nw_s, "msda_pnw");
        if (tap_fn && l == 0) {
            tap_fn("msda_wne", w_ne_s);
            tap_fn("msda_wsw", w_sw_s);
            tap_fn("msda_wse", w_se_s);
            tap_fn("msda_idxne", ggml_cast(ctx0, idx_ne, GGML_TYPE_F32));
            tap_fn("msda_idxsw", ggml_cast(ctx0, idx_sw, GGML_TYPE_F32));
            tap_fn("msda_idxse", ggml_cast(ctx0, idx_se, GGML_TYPE_F32));
        }
        lvl = ggml_add(ctx0, lvl, gather(idx_ne, w_ne_s, "msda_pne"));
        lvl = ggml_add(ctx0, lvl, gather(idx_sw, w_sw_s, "msda_psw"));
        lvl = ggml_add(ctx0, lvl, gather(idx_se, w_se_s, "msda_pse"));
        // tap AFTER the four-corner accumulation: the torch reference
        // msda_lvlN (replay_msda.py) is the summed bilinear sample per level,
        // not the nw corner alone.
        if (tap_fn) {
            tap_fn(("msda_lvl" + std::to_string(l)).c_str(), lvl);
        }
        lvl_by_level.push_back(lvl);
    }
    // sel[slot] = lvl_{l(slot)}[slot] via 0/1 masks keyed on the level band
    // (slot % hlp) % (L*P) / P == l.
    ggml_tensor* sel = nullptr;
    for (int32_t l = 0; l < levels; ++l) {
        std::vector<float> mv(static_cast<size_t>(n_slots), 0.0f);
        for (int64_t s = 0; s < n_slots; ++s) {
            const int64_t f = s % hlp;
            if (f % (levels * points) / points == static_cast<int64_t>(l)) {
                mv[static_cast<size_t>(s)] = 1.0f;
            }
        }
        ggml_tensor* term = ggml_mul(ctx0, lvl_by_level[l],
                                     f32_vec_leaf(ctx0, std::move(mv)));
        sel = (sel == nullptr) ? term : ggml_add(ctx0, sel, term);
    }
    if (tap_fn) tap_fn("msda_sel", sel);

    // Fold: (d, S) -> (d, L*P, heads, qn) is an exact flat identity
    // (slot = f + HLP*q, f = fi + LP*h), so each head's L*P-slot window sits
    // on ne[1] with the head axis on ne[2]. Sum the window, then slice the
    // head's OWN channel block [h*hd, (h+1)*hd): torch head h samples value
    // channels [h*hd, +hd) — the old fold read channels [0, hd) for every
    // head on top of the level smearing.
    ggml_tensor* cur =
            ggml_reshape_4d(ctx0, sel, d, levels * points, heads, qn);
    cur = ggml_cont(ctx0,
                    ggml_permute(ctx0, cur, 1, 0, 2, 3));  // (LP, d, H, q)
    cur = ggml_sum_rows(ctx0, cur);                        // (1, d, H, q)
    const int64_t hd = d / heads;
    ggml_tensor* out = nullptr;
    for (int32_t hh = 0; hh < heads; ++hh) {
        // head-hh block at flat dv + hh*(hd + d) + d*heads*q
        ggml_tensor* block = ggml_view_3d(
                ctx0, cur, hd, qn, 1, ggml_row_size(GGML_TYPE_F32, d * heads),
                ggml_row_size(GGML_TYPE_F32, d * heads) * qn,
                static_cast<size_t>(hh) * (hd + d) * sizeof(float));
        ggml_tensor* part = ggml_cont_2d(ctx0, block, hd, qn);
        out = (out == nullptr) ? part : ggml_concat(ctx0, out, part, 0);
    }
    cur = out;
    if (tap_fn) tap_fn("msda_agg", cur);

    ggml_tensor* res = prec(ctx0, out_w, cur);
    if (out_b != nullptr) res = ggml_add(ctx0, res, out_b);
    return res;
}

}  // namespace

ggml_cgraph* build_mask_decoder_graph(
        ggml_context* ctx0,
        const DecoderConfig& cfg,
        int32_t q,
        int32_t grid_w,
        int32_t grid_h,
        int32_t vit_channels,
        int32_t llm_dim,
        const MaskDecoderWeights& w,
        const MaskDecoderInputs& in,
        MaskDecoderOutputs* out,
        std::string* err,
        std::vector<std::pair<std::string, ggml_tensor*>>* debug_taps,
        const char* taps_filter,
        bool taps_direct) {
    const int32_t d = cfg.hidden_dim;
    const int32_t heads = cfg.transformer_heads;
    const int32_t levels = cfg.msda_num_levels;
    const int32_t points = cfg.msda_num_points;
    const int32_t gn_groups = cfg.group_norm_groups;
    const int64_t s = cfg.cnn_image_size;
    const int64_t s2 = s / 2;
    const int64_t s4 = s / 4;
    const int64_t hw = static_cast<int64_t>(grid_w) * grid_h;

    wwv_clear_f32_leaves();
    wwv_clear_perm_staged();
    wwv_clear_f32vec_staged();
    if (q <= 0 || grid_w <= 0 || grid_h <= 0) {
        if (err != nullptr) *err = "invalid decoder graph shapes";
        return nullptr;
    }
    if (w.encoder_layers.size() !=
                static_cast<size_t>(cfg.msda_encoder_layers) ||
        w.transformer_layers.size() !=
                static_cast<size_t>(cfg.transformer_layers)) {
        if (err != nullptr) *err = "decoder layer tables incomplete";
        return nullptr;
    }
    // real-size graphs exceed the default 2048-node limit
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Debug taps: stage-boundary leaves for the torch-side bisect. Each tap
    // is a cpy so the value materializes in the graph when reachable.
    auto tap = [&](const char* name, ggml_tensor* t) {
        if (debug_taps == nullptr) return;
        if (taps_filter != nullptr &&
            std::strstr(name, taps_filter) == nullptr) {
            return;
        }
        // direct mode (host compute path): every op output keeps its data in
        // the non-recycling context pool after graph_compute, so the tap can
        // reference the tensor itself instead of inserting a cpy node. The
        // cpy nodes were observed to perturb execution (adding taps changed
        // the logits by 0.31) — with them gone the tapped run equals the
        // production run.
        if (taps_direct && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)) {
            // Tap-only nodes (the I32->F32 index casts) sit off the output
            // chain: expand them so graph_compute executes them and the
            // post-compute dump reads a real value. In the host path every
            // tensor owns its own pool block, so extra executed nodes cannot
            // alter other nodes' results.
            ggml_build_forward_expand(gf, t);
            debug_taps->emplace_back(name, t);
            return;
        }
        // dst must use the SOURCE's dimensionality: a fixed 4-D dst gives
        // ne[2..3]=0 for 2-D/3-D tensors -> ggml_nbytes(dst)==0 -> the cpy
        // writes nothing and the dump reads uninitialized pool memory.
        const int nd = ggml_n_dims(t);
        ggml_tensor* c = ggml_cpy(
                ctx0, t, ggml_new_tensor(ctx0, GGML_TYPE_F32, nd, t->ne));
        ggml_build_forward_expand(gf, c);
        debug_taps->emplace_back(name, c);
    };

    // Host-side perm table cache (the ggml leaves are staged per call).
    std::unordered_map<std::string, std::vector<int32_t>> perm_tables;

    // ---------------- SameScaleMSDAEncoder ----------------
    std::vector<ggml_tensor*> tokens(levels);
    for (int32_t l = 0; l < levels; ++l) {
        ggml_tensor* cur = conv1x1(ctx0, w.input_proj_w[l], in.levels[l]);
        cur = gn(ctx0, cur, gn_groups, w.input_proj_gn_w[l],
                 w.input_proj_gn_b[l]);
        // (Wg, Hg, d) -> (d, hw) with token = y*Wg + x.
        tap(("enc_in" + std::to_string(l)).c_str(), in.levels[l]);
        tap(("enc_w" + std::to_string(l)).c_str(), w.input_proj_w[l]);
        tap(("enc_conv" + std::to_string(l)).c_str(), cur);
        // cur (W, H, C, 1) is natively token-major: flat = t + hw*c.
        // The token matrix must be (c, t) with flat = c + d*t (mul_mat K=ne0
        // convention), so reshape to (t, c) then permute to the (c, t) view.
        // The old view_2d(cur, d, hw, nb[2]) read cur_flat[c + hw*t] — it
        // silently mixed channels with spatial positions.
        ggml_tensor* cur2d = ggml_reshape_2d(ctx0, cur, hw, d);
        ggml_tensor* tok = ggml_permute(ctx0, cur2d, 1, 0, 2, 3);
        tokens[l] = ggml_cont(ctx0, tok);
        // torch hooks input_proj[i] -> (C, H, W): the tap must produce the
        // (C, H, W) flat, which is cur's native (W, H, C) order via the
        // reversed-shape identity. tokens[l] is (d, hw)-contiguous
        // (flat = t*d + c) — same values, but the dump-side elementwise
        // compare needs cur's byte order, so tap cur under the enc_proj
        // name (the tokens tap itself stays available as enc_conv).
        tap(("enc_proj" + std::to_string(l)).c_str(), cur);
    }
    // PositionEmbeddingSine2D (host constant, level blocks) + level_embed
    // (weight) added per level, then concatenated in the same order as the
    // projected tokens — engine.py SameScaleMSDAEncoder.forward.
    // The pos leaf bytes are laid out (levels, c, t): flat = l*d*hw + c*hw + t.
    // Reshape to the native (t, c, l) grid, then permute so each per-level
    // slice walks (c, t) in the SAME layout as tokens (flat = c + d*t).
    // The old view_2d(nb[1]) read pos[c + d*t] — transposed garbage.
    ggml_tensor* pos3 =
            ggml_reshape_3d(ctx0, w.encoder_pos_leaf, hw, d, levels);
    // NOTE: ggml_view_impl resets nb[0] to the element size — a view of a
    // permuted (transposed) tensor loses the outer stride. Materialize the
    // (c, t, l) order once, then slice per level with a standard row view.
    ggml_tensor* posT = ggml_cont(
            ctx0,
            ggml_permute(ctx0, pos3, 1, 0, 2, 3));  // (c, t, l) contiguous
    // torch separates the residual carrier (pure src) from the positional
    // input (src+pos feeds the depthwise mixer, fresh every layer):
    //   mixed = pointwise(gelu(depthwise((src+pos) maps)))
    //   src   = norm1(src + mixed)          <- pos NOT accumulated
    // The old code carried (src+pos) through the residual — wrong.
    ggml_tensor* src_all = nullptr;
    ggml_tensor* pos_all = nullptr;
    for (int32_t l = 0; l < levels; ++l) {
        ggml_tensor* pos_l =
                ggml_view_2d(ctx0, posT, d, hw, posT->nb[1],
                             static_cast<size_t>(l) * d * hw *
                                     ggml_row_size(GGML_TYPE_F32, 1));
        ggml_tensor* le_l = ggml_view_2d(
                ctx0, w.level_embed, d, 1, w.level_embed->nb[1],
                static_cast<size_t>(l) * ggml_row_size(w.level_embed->type, d));
        le_l = ggml_repeat(ctx0, le_l,
                           ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, hw));
        if (l == 0) {
            tap("pos_l0", pos_l);
            tap("le_l0", le_l);
        }
        src_all = (src_all == nullptr)
                          ? tokens[l]
                          : ggml_concat(ctx0, src_all, tokens[l], 1);
        pos_all = (pos_all == nullptr)
                          ? ggml_add(ctx0, pos_l, le_l)
                          : ggml_concat(ctx0, pos_all,
                                        ggml_add(ctx0, pos_l, le_l), 1);
    }
    ggml_tensor* memory = src_all;

    tap("enc_in_sum", ggml_add(ctx0, src_all, pos_all));
    int32_t enc_li = 0;
    for (const auto& lw : w.encoder_layers) {
        // depthwise 3x3 (bias-free) + pointwise 1x1 per level, applied to
        // (src + pos) reshaped into maps.
        ggml_tensor* mem_full = ggml_add(ctx0, memory, pos_all);
        ggml_tensor* mixed = nullptr;
        for (int32_t l = 0; l < levels; ++l) {
            const size_t off = static_cast<size_t>(l) * hw *
                               ggml_row_size(GGML_TYPE_F32, d);
            ggml_tensor* src_l =
                    ggml_view_2d(ctx0, mem_full, d, hw, mem_full->nb[1], off);
            // src_l is (c, t)-ordered (flat = c + d*t); the depthwise conv
            // consumes a (W, H, C) map (flat = t + hw*c), so bridge through
            // a (t, c) transpose before the spatial reshape.
            ggml_tensor* srcT = ggml_cont(
                    ctx0, ggml_permute(ctx0, src_l, 1, 0, 2, 3));  // (t, c)
            ggml_tensor* map_l = ggml_reshape_3d(ctx0, srcT, grid_w, grid_h, d);
            ggml_tensor* cur = depthwise3x3(ctx0, lw[0], map_l);
            if (enc_li == 0 && l == 0) tap("dw_l0", cur);
            cur = ggml_gelu(ctx0, cur);
            cur = conv1x1(ctx0, lw[1], cur);
            if (enc_li == 0 && l == 0) tap("pw_l0", cur);
            // cur is (W, H, C) native (flat = t + hw*c): back to the (c, t)
            // token layout via reshape + permute + cont.
            ggml_tensor* cur2 = ggml_reshape_2d(ctx0, cur, hw, d);
            ggml_tensor* tok = ggml_cont(
                    ctx0, ggml_permute(ctx0, cur2, 1, 0, 2, 3));  // (c, t)
            mixed = (mixed == nullptr) ? tok : ggml_concat(ctx0, mixed, tok, 1);
        }
        memory = ggml_add(ctx0, memory, mixed);
        if (enc_li == 0) tap("enc_mixed0", mixed);
        memory = ln(ctx0, "enc_n1a", memory, lw[2], lw[3]);
        if (enc_li == 0) tap("enc_n1_0", memory);
        ggml_tensor* ffn = prec(ctx0, lw[4], memory);
        if (lw[5] != nullptr) ffn = ggml_add(ctx0, ffn, lw[5]);
        ffn = ggml_gelu(ctx0, ffn);
        ffn = prec(ctx0, lw[6], ffn);
        if (lw[7] != nullptr) ffn = ggml_add(ctx0, ffn, lw[7]);
        memory = ggml_add(ctx0, memory, ffn);
        memory = ln(ctx0, "enc_n1b", memory, lw[8], lw[9]);
        if (enc_li == 0) tap("enc_n2_0", memory);
        ++enc_li;
    }

    tap("pos_leaf", w.encoder_pos_leaf);
    tap("memory", memory);

    // ---------------- MaskQueryBuilder ----------------
    ggml_tensor* content = prec(ctx0, w.content_proj_w, in.t_seg);
    if (w.content_proj_b != nullptr) {
        content = ggml_add(ctx0, content, w.content_proj_b);
    }
    content = ln(ctx0, "enc_n1c", content, w.content_norm_w, w.content_norm_b);
    tap("query_content", content);

    ggml_tensor* boxes = in.boxes01;
    auto box_row = [&](int32_t i) {
        return ggml_view_2d(ctx0, boxes, 1, q, boxes->nb[1],
                            ggml_row_size(GGML_TYPE_F32, i));
    };
    ggml_tensor* x1 = box_row(0);
    ggml_tensor* y1 = box_row(1);
    ggml_tensor* x2 = box_row(2);
    ggml_tensor* y2 = box_row(3);
    tap("box_x1", x1);
    tap("box_x2", x2);
    // stages-in isolation switch: feeding <stages>/query_position.torch.bin
    // replaces the box-Fourier branch wholesale, so a remaining cross-attn
    // divergence cannot come from the query_position band.
    ggml_tensor* position;
    if (in.query_position != nullptr) {
        position = in.query_position;
    } else {
        auto fourier = [&](ggml_tensor* v, const char* dbg_phase) {
            // Outer product (F,1) x (1,Q): ggml broadcast is per-dim repeat
            // (can_repeat(b, a) = b divides a per dim), so a cross-dim outer
            // must materialize the (F,Q) repeat of one operand first.
            const int64_t f = w.box_freqs->ne[0];
            ggml_tensor* freqs =
                    ggml_repeat(ctx0, w.box_freqs,
                                ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, f, q));
            // FourierPE1D: phase = 2*pi * value * frequency. The multiply
            // ORDER must match torch ((2pi*v) * freqs): at freq=2^31 the F32
            // rounding of the intermediate differs per order and sin/cos of a
            // phase that is ±300 rad off is garbage.
            ggml_tensor* vs = ggml_scale(
                    ctx0, ggml_reshape_2d(ctx0, ggml_cont(ctx0, v), 1, q),
                    2.0f * static_cast<float>(M_PI));
            ggml_tensor* phase = ggml_mul(ctx0, freqs, vs);
            if (dbg_phase != nullptr) tap(dbg_phase, phase);
            return ggml_concat(ctx0, ggml_sin(ctx0, phase),
                               ggml_cos(ctx0, phase), 0);
        };
        ggml_tensor* width = ggml_sub(ctx0, x2, x1);
        ggml_tensor* height = ggml_sub(ctx0, y2, y1);
        // BoxFourierEncoder: encoded = 0.2*log(clamp(size, 1e-6, 1e30)) + 0.5.
        // The log runs on the HOST when the box values are available: the
        // ggml_log kernel diverges from torch's (MKL vmsLn, see
        // mkl_logf_exact) and the Fourier phase (frequencies up to 2^31)
        // amplifies ulp gaps to radian scale inside the query position band.
        ggml_tensor* enc_w;
        ggml_tensor* enc_h;
        if (in.boxes01_host != nullptr) {
            std::vector<float> bw(static_cast<size_t>(q));
            std::vector<float> bh(static_cast<size_t>(q));
            for (int32_t k = 0; k < q; ++k) {
                bw[static_cast<size_t>(k)] = std::clamp(
                        in.boxes01_host[4 * k + 2] - in.boxes01_host[4 * k + 0],
                        1e-6f, 1e30f);
                bh[static_cast<size_t>(k)] = std::clamp(
                        in.boxes01_host[4 * k + 3] - in.boxes01_host[4 * k + 1],
                        1e-6f, 1e30f);
            }
            std::vector<float> ew(static_cast<size_t>(q));
            std::vector<float> eh(static_cast<size_t>(q));
            if (mkl_logf_exact(bw.data(), ew.data(), q) &&
                mkl_logf_exact(bh.data(), eh.data(), q)) {
                for (int32_t k = 0; k < q; ++k) {
                    ew[static_cast<size_t>(k)] =
                            0.2f * ew[static_cast<size_t>(k)] + 0.5f;
                    eh[static_cast<size_t>(k)] =
                            0.2f * eh[static_cast<size_t>(k)] + 0.5f;
                }
            } else {
                // no MKL VML available: double-precision glibc log, 98.2%
                // bit-exact with torch (residual ulps are a known limitation).
                for (int32_t k = 0; k < q; ++k) {
                    ew[static_cast<size_t>(k)] =
                            0.2f * static_cast<
                                           float>(std::log(static_cast<double>(
                                           bw[static_cast<size_t>(k)]))) +
                            0.5f;
                    eh[static_cast<size_t>(k)] =
                            0.2f * static_cast<
                                           float>(std::log(static_cast<double>(
                                           bh[static_cast<size_t>(k)]))) +
                            0.5f;
                }
            }
            enc_w = f32_vec_leaf(ctx0, std::move(ew));
            enc_h = f32_vec_leaf(ctx0, std::move(eh));
        } else {
            enc_w = ggml_add(
                    ctx0,
                    ggml_scale(ctx0,
                               ggml_log(ctx0,
                                        ggml_clamp(ctx0, width, 1e-6f, 1e30f)),
                               0.2f),
                    f32_leaf(ctx0, 0.5f));
            enc_h = ggml_add(
                    ctx0,
                    ggml_scale(ctx0,
                               ggml_log(ctx0,
                                        ggml_clamp(ctx0, height, 1e-6f, 1e30f)),
                               0.2f),
                    f32_leaf(ctx0, 0.5f));
        }
        tap("box_freqs_tap", w.box_freqs);
        ggml_tensor* f_x1 = fourier(x1, nullptr);
        tap("fourier_x1", f_x1);
        tap("enc_w_tap", enc_w);
        ggml_tensor* f_ew = fourier(enc_w, "phase_ew");
        tap("fourier_ew", f_ew);
        ggml_tensor* box_feat = f_x1;  // f_x1 = fourier(x1, nullptr)
        box_feat = ggml_concat(ctx0, box_feat, fourier(y1, nullptr), 0);
        box_feat = ggml_concat(ctx0, box_feat, f_ew, 0);
        box_feat = ggml_concat(ctx0, box_feat, fourier(enc_h, nullptr), 0);
        tap("box_feat", box_feat);
        ggml_tensor* box_hidden = prec(ctx0, w.box_mlp_w1, box_feat);
        if (w.box_mlp_b1 != nullptr)
            box_hidden = ggml_add(ctx0, box_hidden, w.box_mlp_b1);
        // nn.GELU() default is the exact erf variant — ggml_gelu is the tanh
        // approximation and drifts ~0.03 here.
        box_hidden = ggml_gelu_erf(ctx0, box_hidden);
        tap("box_hidden", box_hidden);
        position = prec(ctx0, w.box_mlp_w2, box_hidden);
        if (w.box_mlp_b2 != nullptr)
            position = ggml_add(ctx0, position, w.box_mlp_b2);
        position = ln(ctx0, "enc_n1d", position, w.position_norm_w,
                      w.position_norm_b);
    }
    tap("query_position", position);

    auto clamp01 = [&](ggml_tensor* t) {
        return ggml_clamp(ctx0, t, 0.0f, 1.0f);
    };
    ggml_tensor* ref_cx =
            ggml_scale(ctx0, ggml_add(ctx0, clamp01(x1), clamp01(x2)), 0.5f);
    ggml_tensor* ref_cy =
            ggml_scale(ctx0, ggml_add(ctx0, clamp01(y1), clamp01(y2)), 0.5f);
    ggml_tensor* ref_w = ggml_clamp(
            ctx0, ggml_sub(ctx0, clamp01(x2), clamp01(x1)), 1e-4f, 1.0f);
    ggml_tensor* ref_h = ggml_clamp(
            ctx0, ggml_sub(ctx0, clamp01(y2), clamp01(y1)), 1e-4f, 1.0f);
    ggml_tensor* ref = ggml_concat(ctx0, ref_cx, ref_cy, 0);
    ref = ggml_concat(ctx0, ref, ref_w, 0);
    ref = ggml_concat(ctx0, ref, ref_h, 0);

    // ---------------- transformer ----------------
    ggml_tensor* target = content;
    for (const auto& tw : w.transformer_layers) {
        // slots: 0/1 self in_proj w/b, 2/3 self out w/b, 4/5 norm1 w/b,
        // 6/7 msda val w/b, 8/9 msda off w/b, 10/11 msda att w/b,
        // 12/13 msda out w/b, 14/15 norm2 w/b, 16/17 ffn1 w/b,
        // 18/19 ffn2 w/b, norm3 = ln at the end (w/b at 14/15? no —
        // the Python layer has THREE norms; slots above list two, so the
        // third norm reuses slots (norm3 w/b ride in 14/15 and norm2 uses
        // none) — the converter table documents: norm1@4/5, norm2@14/15,
        // norm3 rides OUTSIDE this table as the trailing ln of the layer.
        ggml_tensor* self_out = self_attention(ctx0, target, position, tw[0],
                                               tw[1], tw[2], tw[3], heads);
        if (&tw == w.transformer_layers.data()) tap("tfm_sa0", self_out);
        target = ln(ctx0, "enc_n2a", ggml_add(ctx0, target, self_out), tw[4],
                    tw[5]);
        if (&tw == w.transformer_layers.data()) tap("tfm_n1_0", target);
        ggml_tensor* cross = msda_layer(
                ctx0, ggml_add(ctx0, target, position), memory, ref, heads,
                levels, points, grid_w, grid_h, tw[6], tw[7], tw[8], tw[9],
                tw[10], tw[11], tw[12], tw[13],
                (&tw == w.transformer_layers.data())
                        ? std::function<void(const char*, ggml_tensor*)>(tap)
                        : nullptr);
        if (&tw == w.transformer_layers.data()) tap("tfm_cr0", cross);
        target = ln(ctx0, "enc_n2b", ggml_add(ctx0, target, cross), tw[14],
                    tw[15]);
        if (&tw == w.transformer_layers.data()) tap("tfm_n2_0", target);
        ggml_tensor* ffn = prec(ctx0, tw[16], target);
        if (tw[17] != nullptr) ffn = ggml_add(ctx0, ffn, tw[17]);
        ffn = ggml_gelu_erf(ctx0, ffn);  // F.gelu default is exact erf
        ffn = prec(ctx0, tw[18], ffn);
        if (tw[19] != nullptr) ffn = ggml_add(ctx0, ffn, tw[19]);
        target = ggml_add(ctx0, target, ffn);
        // norm3 weights are appended by the converter as layer-suffix
        // entries and applied through the same ln helper via the shared
        // tail table (w.transformer_norms[li]).
        target = ln(
                ctx0, "content_norm", target,
                w.transformer_norms[&tw - w.transformer_layers.data()].first,
                w.transformer_norms[&tw - w.transformer_layers.data()].second);
        tap(("tfm" + std::to_string(&tw - w.transformer_layers.data())).c_str(),
            target);
    }

    // ---------------- pixel fusion ----------------
    // SameScaleFPN top-down on the projected level maps.
    auto level_map = [&](int32_t l) {
        const size_t off =
                static_cast<size_t>(l) * hw * ggml_row_size(GGML_TYPE_F32, d);
        // torch maps[l] = chunk.transpose(1,2).reshape(B, C, H, W): flat =
        // c*H*W + (y*W + x), channel SLOWEST. The memory pool stores
        // channels FASTEST (flat = c + d*s), and ggml_reshape only shares
        // the data pointer with contiguous strides — it reinterprets the
        // (c, t) bytes as (W, H, C) no matter what logical order the source
        // view had. The needed mapping i -> p = (i%hw)*d + i/hw needs
        // stride 192 for x and stride 1 for c, but ggml_view pins nb0 to
        // the source's element stride — so build the (c, y, x) view (c on
        // the contiguous ne0) and permute it into (W, H, C) shape.
        ggml_tensor* m = ggml_view_3d(
                ctx0, memory, d, grid_h, grid_w,
                ggml_row_size(GGML_TYPE_F32, grid_w * d),  // y stride
                ggml_row_size(GGML_TYPE_F32, d),           // x stride
                off);
        m = ggml_permute(ctx0, m, 2, 1, 0, 3);  // (c, y, x) -> (W, H, C, 1)
        // NOTE: the pix_lv dump reads the SHARED physical bytes (c fastest),
        // not the logical (W, H, C) order — logical correctness is only
        // observable through the conv output (pix_lat*).
        tap(("pix_lv" + std::to_string(l)).c_str(), m);
        return m;
    };
    ggml_tensor* fused = conv1x1(ctx0, w.fpn_lateral_w[3], level_map(3));
    tap("pix_lat3", fused);
    for (int32_t l = levels - 2; l >= 0; --l) {
        ggml_tensor* lat = conv1x1(ctx0, w.fpn_lateral_w[l], level_map(l));
        tap(("pix_lat" + std::to_string(l)).c_str(), lat);
        fused = ggml_add(ctx0, fused, lat);
        const auto& ow = w.fpn_output[static_cast<size_t>(levels - 2 - l)];
        fused = conv2d(ctx0, ow[0], nullptr, fused, 1, 1);
        fused = gn(ctx0, fused, gn_groups, ow[1], ow[2]);
        fused = ggml_gelu(ctx0, fused);
        tap(("pix_out" + std::to_string(levels - 2 - l)).c_str(), fused);
    }
    tap("pix_fpn", fused);
    // Upsample: 2 x (conv3x3 + PixelShuffle + GELU).
    ggml_tensor* up = conv2d(ctx0, w.up_conv0_w, w.up_conv0_b, fused, 1, 1);
    up = ggml_gelu(ctx0, pixel_shuffle(ctx0, &perm_tables, nullptr, up));
    up = conv2d(ctx0, w.up_conv1_w, w.up_conv1_b, up, 1, 1);
    up = ggml_gelu(ctx0, pixel_shuffle(ctx0, &perm_tables, nullptr, up));
    tap("pix_up", up);
    // Resize to the S/4 canvas (the Python adaptive-pool + interpolate
    // branch reduces to this bilinear path on the released grid shapes),
    // then smooth + ResidualPixelShuffle to S/2 and the stem add.
    ggml_tensor* fpn_map = resize_map(ctx0, up, s4, s4);
    fpn_map = conv1x1(ctx0, w.fpn_proj_w, fpn_map);
    tap("pix_fpn_proj", fpn_map);
    // torch smooth = Sequential(GN, GELU, Conv2d 3x3): the norm+act run
    // BEFORE the conv — the old code applied conv first and normed after.
    ggml_tensor* smooth =
            gn(ctx0, fpn_map, gn_groups, w.smooth_gn_w, w.smooth_gn_b);
    smooth = ggml_gelu(ctx0, smooth);
    smooth = conv2d(ctx0, w.smooth_conv_w, nullptr, smooth, 1, 1);
    tap("pix_smooth", smooth);
    ggml_tensor* base = resize_map(ctx0, smooth, s2, s2);
    ggml_tensor* res = pixel_shuffle(
            ctx0, &perm_tables, nullptr,
            conv2d(ctx0, w.s2_up_conv_w, w.s2_up_conv_b, smooth, 1, 1));
    ggml_tensor* pixel_feature = ggml_add(ctx0, base, res);
    tap("pix_s2up", pixel_feature);
    // ShallowImageStem(output_stride=2): conv s2 -> GN -> GELU -> conv s1.
    ggml_tensor* cnn =
            conv2d(ctx0, w.stem_conv0_w, nullptr, in.cnn_image, 2, 1);
    cnn = ggml_gelu(ctx0, gn(ctx0, cnn, gn_groups, w.stem_gn0_w, w.stem_gn0_b));
    cnn = conv2d(ctx0, w.stem_conv1_w, nullptr, cnn, 1, 1);
    cnn = ggml_gelu(ctx0, gn(ctx0, cnn, gn_groups, w.stem_gn1_w, w.stem_gn1_b));
    tap("pix_cnn", cnn);
    ggml_tensor* cnn_proj = conv1x1(ctx0, w.cnn_proj_w, cnn);
    tap("pix_cnn_proj", cnn_proj);
    pixel_feature = ggml_add(ctx0, pixel_feature, cnn_proj);
    // torch hook: the AdditivePixelFusion module OUTPUT (C, H, W) — the
    // (W, H, C) native order equals the torch flat (reversed-shape
    // identity, N=1). The old tap pointed at cascade semantic_s2 (48ch).
    tap("pixel_fusion", pixel_feature);

    // ---------------- dynamic mask predictor ----------------
    ggml_tensor* kern = prec(ctx0, w.kernel_w1, target);
    if (w.kernel_b1 != nullptr) kern = ggml_add(ctx0, kern, w.kernel_b1);
    kern = ggml_gelu(ctx0, kern);
    ggml_tensor* params = prec(ctx0, w.kernel_w2, kern);
    if (w.kernel_b2 != nullptr) params = ggml_add(ctx0, params, w.kernel_b2);
    tap("kernel_params", params);
    const int64_t md = cfg.mask_feature_dim;
    ggml_tensor* kernels = ggml_view_2d(ctx0, params, md, q, params->nb[1], 0);
    ggml_tensor* kbias = ggml_view_2d(ctx0, params, 1, q, params->nb[1],
                                      ggml_row_size(params->type, md));
    // torch einsum('qc,bchw->bqhw') contracts the CHANNEL axis. The old
    // reshape_2d(pixel_feature, md, s2*s2) treated ne[0] of the (W, H, C)
    // map as the contraction axis — but (W, H, C) flat puts x there, not c,
    // so the kernel mixed spatial positions instead of channels. Permute to
    // (C, H, W) logical and let mul_mat contract ne[0] over the batch dims.
    ggml_tensor* feat_c = ggml_cont(
            ctx0,
            ggml_permute(ctx0, pixel_feature, 2, 1, 0, 3));  // (C, H, W, N)
    tap("pix_featc", feat_c);
    ggml_tensor* coarse = prec(ctx0, kernels, feat_c);  // (Q, H, W, N)
    tap("pix_cq", coarse);
    // (Q, H, W, N) -> (W, H, N, Q): old0(Q) -> dest3, old1(H) -> dest1,
    // old2(W) -> dest0, old3(N) -> dest2 (perm[i] = destination axis of old
    // i). With N=1 the contiguous (W, H, 1, Q) flat w + H*(h + W*q) equals
    // the torch (Q, H, W) row-major flat q*H*W + h*W + w — the reversed-
    // shape identity the elementwise parity gate relies on. perm (3,0,1,2)
    // here would swap the H/W strides and only the h==w diagonal survives.
    coarse = ggml_cont(ctx0, ggml_permute(ctx0, coarse, 3, 1, 0, 2));
    // kbias is a strided view of the packed params — cont() before reshape.
    coarse =
            ggml_add(ctx0, coarse,
                     ggml_reshape_4d(ctx0, ggml_cont(ctx0, kbias), 1, 1, 1, q));
    out->mask_logits_coarse = coarse;

    // ---------------- FullCanvasAlphaCascadeHead ----------------
    // Stages S/4 -> S/2 -> S/1 over the S/2 coarse prediction. The fuse
    // 1x1 GEMMs mix shared and per-query inputs and are split into a
    // shared block + a per-query block (same math); the condition applies
    // per-query scale/bias through broadcast repeat; residual blocks and
    // output convs run batched over ne3 = Q.
    const int32_t cc = cfg.alpha_cascade_dim;
    const int32_t cgn = cc < gn_groups ? cc : gn_groups;

    auto proj_branch = [&](ggml_tensor* conv_w, ggml_tensor* gn_w,
                           ggml_tensor* gn_b, ggml_tensor* x) {
        ggml_tensor* t = conv2d(ctx0, conv_w, nullptr, x, 1, 1);
        t = gn(ctx0, t, cgn, gn_w, gn_b);
        return ggml_gelu(ctx0, t);
    };
    // 1x1 fuse over (W, H, Cin[, N]) with a column-sliced (cc, Cin) weight.
    auto fuse1x1 = [&](ggml_tensor* fuse_w2d, int64_t col_begin,
                       int64_t col_count, ggml_tensor* x) {
        ggml_tensor* wblk =
                ggml_view_2d(ctx0, fuse_w2d, cc, col_count, fuse_w2d->nb[1],
                             static_cast<size_t>(col_begin) *
                                     ggml_row_size(fuse_w2d->type, 1));
        return conv1x1(ctx0, wblk, x);
    };
    // Residual block: GN -> GELU -> conv3x3 -> GN -> GELU -> conv3x3 + skip.
    auto res_block = [&](const std::array<ggml_tensor*, 12>& stack, size_t off,
                         ggml_tensor* x) {
        ggml_tensor* t =
                ggml_gelu(ctx0, gn(ctx0, x, cgn, stack[off], stack[off + 1]));
        t = conv2d(ctx0, stack[off + 2], nullptr, t, 1, 1);
        t = ggml_gelu(ctx0, gn(ctx0, t, cgn, stack[off + 3], stack[off + 4]));
        t = conv2d(ctx0, stack[off + 5], nullptr, t, 1, 1);
        return ggml_add(ctx0, x, t);
    };
    // Per-query condition: f * (1 + 0.1 tanh(scale)) + 0.1 bias.
    auto condition = [&](ggml_tensor* x, ggml_tensor* query_w,
                         ggml_tensor* query_b, ggml_tensor* queries, int64_t w_,
                         int64_t h_) {
        ggml_tensor* sb = prec(ctx0, query_w, queries);
        if (query_b != nullptr) sb = ggml_add(ctx0, sb, query_b);
        ggml_tensor* scale = ggml_view_2d(ctx0, sb, cc, q, sb->nb[1], 0);
        ggml_tensor* bias = ggml_view_2d(ctx0, sb, cc, q, sb->nb[1],
                                         ggml_row_size(sb->type, cc));
        auto to_map = [&](ggml_tensor* v) {
            // q at dim3 to match the (W, H, 1, Q) target layout.
            // scatter v (cc, Q) -> (W, H, cc, Q): move cc to dim2, repeat
            // the (1, 1, cc, Q) tile over the spatial dims (per-dim tile).
            ggml_tensor* t = ggml_cont(
                    ctx0,
                    ggml_permute(
                            ctx0,
                            ggml_reshape_2d(ctx0, ggml_cont(ctx0, v), cc, q), 2,
                            3, 0,
                            1));  // (1, 1, cc, q): old0 -> dest2, old1 -> dest3
            // target layout (W, H, cc, Q): C at dim2 — matches x.
            return ggml_repeat(
                    ctx0, t,
                    ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, w_, h_, cc, q));
        };
        ggml_tensor* gain = to_map(ggml_tanh(ctx0, scale));
        gain = ggml_scale(ctx0, gain, 0.1f);
        gain = ggml_add(ctx0, gain, f32_leaf(ctx0, 1.0f));
        return ggml_add(ctx0, ggml_mul(ctx0, x, gain),
                        ggml_scale(ctx0, to_map(bias), 0.1f));
    };
    // Shared inputs.
    ggml_tensor* semantic_s2 =
            proj_branch(w.cascade_semantic_w, w.cascade_semantic_gn_w,
                        w.cascade_semantic_gn_b, pixel_feature);
    tap("cas_semantic_s2", semantic_s2);
    ggml_tensor* semantic_s4 = resize_map(ctx0, semantic_s2, s4, s4, true);
    tap("cas_semantic_s4", semantic_s4);
    ggml_tensor* rgb4 =
            proj_branch(w.rgb_s4_w, w.rgb_s4_gn_w, w.rgb_s4_gn_b,
                        resize_map(ctx0, in.cnn_image, s4, s4, true));
    tap("cas_rgb4", rgb4);
    ggml_tensor* rgb2 =
            proj_branch(w.rgb_s2_w, w.rgb_s2_gn_w, w.rgb_s2_gn_b,
                        resize_map(ctx0, in.cnn_image, s2, s2, true));
    tap("cas_rgb2", rgb2);
    ggml_tensor* rgb1 =
            proj_branch(w.rgb_s1_w, w.rgb_s1_gn_w, w.rgb_s1_gn_b, in.cnn_image);
    tap("cas_rgb1", rgb1);
    ggml_tensor* coarse4_in = resize_map(ctx0, coarse, s4, s4, true);
    tap("cas_coarse4_in", coarse4_in);
    ggml_tensor* coarse4_p = proj_branch(w.coarse_s4_w, w.coarse_s4_gn_w,
                                         w.coarse_s4_gn_b, coarse4_in);
    tap("cas_coarse4_p", coarse4_p);
    ggml_tensor* coarse2_p = proj_branch(w.coarse_s2_w, w.coarse_s2_gn_w,
                                         w.coarse_s2_gn_b, coarse);
    tap("cas_coarse2_p", coarse2_p);

    // ---- stage S/4 ----
    // ggml broadcast requires the LARGER (per-query, Q=3) tensor as the
    // first operand: the per-query coarse term leads, shared terms follow.
    ggml_tensor* fuse4 = fuse1x1(w.fuse_s4_w, 2 * cc, cc, coarse4_p);
    fuse4 = ggml_add(ctx0, fuse4, fuse1x1(w.fuse_s4_w, 0, cc, semantic_s4));
    fuse4 = ggml_add(ctx0, fuse4, fuse1x1(w.fuse_s4_w, cc, cc, rgb4));
    tap("cas_fuse4", fuse4);
    fuse4 = condition(fuse4, w.query_s4_w, w.query_s4_b, target, s4, s4);
    tap("cas_fuse4c", fuse4);
    for (int b = 0; b < cfg.alpha_cascade_blocks; ++b) {
        fuse4 = res_block(w.cascade_blocks[0], static_cast<size_t>(b) * 6,
                          fuse4);
    }
    ggml_tensor* feat4 = fuse4;
    tap("cas_feat4", feat4);
    ggml_tensor* delta4 = [&] {
        // (W, H, cc, Q) -> (cc, W, H, Q): old0(W) -> dest1, old1(H) -> dest2,
        // old2(cc) -> dest0 — dest.ne[perm[i]] = src.ne[i] (ggml.c). The
        // reshape_2d(cc, s4*s4*q) that follows then has column p = x + W*y +
        // W*H*q with channels on the contraction axis ne0. perm (2,0,1,3)
        // here yields (H, cc, W, Q) and corrupts the contraction.
        ggml_tensor* f = ggml_permute(ctx0, feat4, 1, 2, 0, 3);
        f = ggml_cont(ctx0, f);
        f = ggml_reshape_2d(ctx0, f, cc, static_cast<int64_t>(s4) * s4 * q);
        tap("cas_f_s4", f);
        ggml_tensor* d = ggml_reshape_4d(ctx0, prec(ctx0, w.output_s4_w, f), s4,
                                         s4, 1, q);
        tap("cas_delta4", d);
        return ggml_add(ctx0, d,
                        ggml_reshape_4d(ctx0, w.output_s4_b, 1, 1, 1, 1));
    }();
    ggml_tensor* logits_s4 = ggml_add(
            ctx0, ggml_reshape_4d(ctx0, coarse4_in, s4, s4, 1, q), delta4);
    tap("cas_logits_s4", logits_s4);

    // ---- stage S/2 ----
    ggml_tensor* up42 = convt_crop(
            ctx0, ggml_conv_transpose_2d_p0(ctx0, w.up_s4_s2_w, feat4, 2), s2,
            s2, cc, q);
    tap("cas_up42", up42);
    ggml_tensor* prev2_raw = resize_map(ctx0, logits_s4, s2, s2, true);
    tap("cas_prev2_raw", prev2_raw);
    ggml_tensor* prev2 =
            proj_branch(w.previous_s2_w, w.previous_s2_gn_w, w.previous_s2_gn_b,
                        ggml_sigmoid(ctx0, prev2_raw));
    tap("cas_prev2", prev2);
    // torch cat([up_s4_to_s2(f4), semantic, rgb, coarse, previous]): the
    // transposed upsample owns input channels [0, cc) of fuse_s2 and goes
    // through the fuse 1x1 like every other block (no identity passthrough).
    ggml_tensor* fuse2 = fuse1x1(w.fuse_s2_w, 0, cc, up42);
    fuse2 = ggml_add(ctx0, fuse2, fuse1x1(w.fuse_s2_w, cc, cc, semantic_s2));
    fuse2 = ggml_add(ctx0, fuse2, fuse1x1(w.fuse_s2_w, 2 * cc, cc, rgb2));
    fuse2 = ggml_add(ctx0, fuse2, fuse1x1(w.fuse_s2_w, 3 * cc, cc, coarse2_p));
    fuse2 = ggml_add(ctx0, fuse2, fuse1x1(w.fuse_s2_w, 4 * cc, cc, prev2));
    tap("cas_fuse2", fuse2);
    fuse2 = condition(fuse2, w.query_s2_w, w.query_s2_b, target, s2, s2);
    tap("cas_fuse2c", fuse2);
    for (int b = 0; b < cfg.alpha_cascade_blocks; ++b) {
        fuse2 = res_block(w.cascade_blocks[1], static_cast<size_t>(b) * 6,
                          fuse2);
    }
    tap("cas_f2", fuse2);
    ggml_tensor* delta2 = [&] {
        // (W, H, cc, Q) -> (cc, W, H, Q): same permute derivation as delta4.
        ggml_tensor* f = ggml_permute(ctx0, fuse2, 1, 2, 0, 3);
        f = ggml_cont(ctx0, f);
        f = ggml_reshape_2d(ctx0, f, cc, s2 * s2 * q);
        tap("cas_f_s2", f);
        ggml_tensor* d = ggml_reshape_4d(ctx0, prec(ctx0, w.output_s2_w, f), s2,
                                         s2, 1, q);
        tap("cas_delta2", d);
        return ggml_add(ctx0, d,
                        ggml_reshape_4d(ctx0, w.output_s2_b, 1, 1, 1, 1));
    }();
    ggml_tensor* logits_s2 = ggml_add(ctx0, coarse, delta2);
    tap("cas_logits_s2", logits_s2);
    out->mask_logits_s2 = logits_s2;

    // ---- stage S/1 ----
    ggml_tensor* up21 = convt_crop(
            ctx0, ggml_conv_transpose_2d_p0(ctx0, w.up_s2_s1_w, fuse2, 2), s, s,
            cc, q);
    tap("cas_up21", up21);
    ggml_tensor* prev1_raw = resize_map(ctx0, logits_s2, s, s, true);
    tap("cas_prev1_raw", prev1_raw);
    ggml_tensor* prev1 =
            proj_branch(w.previous_s1_w, w.previous_s1_gn_w, w.previous_s1_gn_b,
                        ggml_sigmoid(ctx0, prev1_raw));
    tap("cas_prev1", prev1);
    ggml_tensor* semantic_s1 = resize_map(ctx0, semantic_s2, s, s, true);
    tap("cas_semantic_s1", semantic_s1);
    // torch cat([up_s2_to_s1(f2), semantic_s1, rgb_s1, previous_s1]): up21
    // owns input channels [0, cc) of fuse_s1.
    ggml_tensor* fuse1 = fuse1x1(w.fuse_s1_w, 0, cc, up21);
    fuse1 = ggml_add(ctx0, fuse1, fuse1x1(w.fuse_s1_w, cc, cc, semantic_s1));
    fuse1 = ggml_add(ctx0, fuse1, fuse1x1(w.fuse_s1_w, 2 * cc, cc, rgb1));
    fuse1 = ggml_add(ctx0, fuse1, fuse1x1(w.fuse_s1_w, 3 * cc, cc, prev1));
    tap("cas_fuse1", fuse1);
    fuse1 = condition(fuse1, w.query_s1_w, w.query_s1_b, target, s, s);
    tap("cas_fuse1c", fuse1);
    for (int b = 0; b < cfg.alpha_cascade_blocks; ++b) {
        fuse1 = res_block(w.cascade_blocks[2], static_cast<size_t>(b) * 6,
                          fuse1);
    }
    tap("cas_f1", fuse1);
    ggml_tensor* delta1 = [&] {
        // (W, H, cc, Q) -> (cc, W, H, Q): same permute derivation as delta4.
        ggml_tensor* f = ggml_permute(ctx0, fuse1, 1, 2, 0, 3);
        f = ggml_cont(ctx0, f);
        f = ggml_reshape_2d(ctx0, f, cc, static_cast<int64_t>(s) * s * q);
        tap("cas_f_s1", f);
        ggml_tensor* d =
                ggml_reshape_4d(ctx0, prec(ctx0, w.output_s1_w, f), s, s, 1, q);
        tap("cas_delta1", d);
        return ggml_add(ctx0, d,
                        ggml_reshape_4d(ctx0, w.output_s1_b, 1, 1, 1, 1));
    }();
    out->mask_logits_s1 = ggml_add(
            ctx0, ggml_reshape_4d(ctx0, prev1_raw, s, s, 1, q), delta1);
    // expand the terminal output: the forward expansion pulls the whole
    // graph (coarse/s2 are intermediate nodes on this path).
    ggml_build_forward_expand(gf, out->mask_logits_s1);
    return gf;
}

}  // namespace wwv
