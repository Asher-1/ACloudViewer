// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/yolo/yolo_graph.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "common/debug_dump.hpp"
#include "ggml-alloc.h"
#include "ggml.h"
#include "gguf.h"

namespace yolo {

namespace {

std::atomic<uint64_t> g_next_plan_owner_id{1};

// Q8_0 block layout (binary-compatible with ggml's block_q8_0): a per-32-
// element fp16 scale followed by 32 int8 deltas. Defined locally so the host
// can dequantize weights for backends with no Q8 conv path (vulkan) without
// pulling ggml-common.h out of the ggml src tree.
constexpr int QK8_0 = 32;
struct block_q8_0 {
    ggml_fp16_t d;
    int8_t qs[QK8_0];
};
static_assert(sizeof(block_q8_0) == sizeof(ggml_fp16_t) + QK8_0,
              "block_q8_0 layout mismatch");

/* Builds the op-graph tensor chain inside a fresh gctx. Weight tensors are
 * created in wctx on the first build and looked up by name on every rebuild
 * (identical ops => identical weight set, so rebuilds always hit). */
struct GraphBuilder {
    ggml_context* gctx;  // graph tensors (rebuilt per canvas)
    ggml_context* wctx;  // weight tensors (session-persistent, data in wbuf)
    const ModelDef& model;
    // CUDA/Vulkan: every quantized weight conforms to the igemm Q8_0 path
    // (K 32-aligned), so quantized convs join the f16 direct flow.
    // vulkan: the same flag triggers a one-shot host dequant of Q8_0 weights
    // to f16 at load time, since vulkan has no Q8 conv shader.
    bool q8_direct = false;
    // Resolved backend family (Runtime, never a compile-time macro) selects
    // the DIRECT conv ops (ggml_conv_2d_direct family, added by the
    // yolo_merged ggml patch): CUDA always runs the f16 direct flow, Vulkan
    // when weights/activations share an f16/f32 dtype. CPU/Metal sessions
    // keep the generic im2col vocabulary.
    bool use_direct_conv = false;
    // CUDA f16 flow: conv_transpose needs an F32 detour (see conv_transpose).
    bool cuda_backend = false;

    ggml_tensor* w(const std::string& prefix, const char* suffix) {
        const std::string name = prefix + "." + suffix;
        // Rebuilds reuse the wctx tensor structs created by the first build;
        // ggml_backend_alloc_ctx_tensors_from_buft() only ran once, so a
        // rebuild must never create a new weight tensor (it would have no
        // buffer backing). The op set is model-static, hence the lookup
        // always succeeds after the first build.
        if (ggml_tensor* t = ggml_get_tensor(wctx, name.c_str())) {
            return t;
        }
        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) return nullptr;
        ggml_tensor* t =
                ggml_new_tensor(wctx, it->second.type, 4, it->second.ne);
        ggml_set_name(t, name.c_str());
        return t;
    }

    // Restore the 4D conv kernel view for quantized 2D-stored weights.
    // NOTE: only F32 kernels return as-is — native f16/f32 model weights are
    // 4D-stored, but the Vulkan q8->f16 host expansion keeps the quantized
    // 2D [K, OC] layout, and those MUST be reshaped (upstream parity:
    // ggml_conv_2d_direct asserts a->ne[2] == b->ne[2] on a 2D kernel).
    ggml_tensor* kernel4d(const OpDef& op, ggml_tensor* wT) {
        if (wT->ne[2] != 1 || wT->ne[3] != 1 || wT->type == GGML_TYPE_F32) {
            return wT;
        }
        const int64_t kh = op.ai("k", 0), kw = op.ai("k", 1);
        const int64_t out = wT->ne[1];
        const int64_t in = wT->ne[0] / (kh * kw);
        return ggml_reshape_4d(gctx, wT, kw, kh, in, out);
    }

    ggml_tensor* add_bias_act(const OpDef& op,
                              const std::string& prefix,
                              ggml_tensor* out) {
        if (ggml_tensor* b = w(prefix, "b")) {
            if (b->type != out->type) b = ggml_cast(gctx, b, out->type);
            out = ggml_add(gctx, out,
                           ggml_reshape_4d(gctx, b, 1, 1, b->ne[0], 1));
        }
        auto act = op.sparams.find("act");
        if (act != op.sparams.end() && act->second == "silu") {
            out = ggml_silu(gctx, out);
        }
        return out;
    }

    ggml_tensor* conv2d(const OpDef& op,
                        const std::string& prefix,
                        ggml_tensor* x) {
        ggml_tensor* wT = w(prefix, "w");
        if (!wT) {
            YOLO_LOG_ERROR("conv '%s' has no weight tensor '.w'",
                           prefix.c_str());
            return nullptr;
        }
        const bool depthwise = op.type == "dwconv";
        ggml_tensor* bias = w(prefix, "b");
        const auto act = op.sparams.find("act");
        const bool silu = act != op.sparams.end() && act->second == "silu";
        const bool direct_types =
                wT->type == x->type &&
                (wT->type == GGML_TYPE_F32 || wT->type == GGML_TYPE_F16);
        ggml_tensor* out;
        if (use_direct_conv && direct_types) {
            // CUDA/Vulkan f16 (or f32/f32) fast path: the patched direct conv
            // op skips the im2col materialization entirely (CUDA: tensor-core
            // mma throughput; Vulkan: fused shader).
            ggml_tensor* w4d = kernel4d(op, wT);
            if (!depthwise && bias && silu) {
                return ggml_conv_2d_direct_bias_silu(
                        gctx, w4d, x, bias, (int)op.ai("s", 0),
                        (int)op.ai("s", 1), (int)op.ai("p", 0),
                        (int)op.ai("p", 1), (int)op.ai("d", 0),
                        (int)op.ai("d", 1));
            }
            out = depthwise ? ggml_conv_2d_dw_direct(
                                      gctx, w4d, x, (int)op.ai("s", 0),
                                      (int)op.ai("s", 1), (int)op.ai("p", 0),
                                      (int)op.ai("p", 1), (int)op.ai("d", 0),
                                      (int)op.ai("d", 1))
                            : ggml_conv_2d_direct(
                                      gctx, w4d, x, (int)op.ai("s", 0),
                                      (int)op.ai("s", 1), (int)op.ai("p", 0),
                                      (int)op.ai("p", 1), (int)op.ai("d", 0),
                                      (int)op.ai("d", 1));
        } else if (!depthwise && ggml_is_quantized(wT->type)) {
            if (q8_direct && use_direct_conv) {
                // CUDA Q8 flow: the igemm Q8 path takes F16 activations. The
                // PSA attention proj conv feeds F32 (mul_mat/softmax chain),
                // hence the per-conv cast.
                if (x->type != GGML_TYPE_F16)
                    x = ggml_cast(gctx, x, GGML_TYPE_F16);
                ggml_tensor* w4d = kernel4d(op, wT);
                if (bias && silu) {
                    return ggml_conv_2d_direct_bias_silu(
                            gctx, w4d, x, bias, (int)op.ai("s", 0),
                            (int)op.ai("s", 1), (int)op.ai("p", 0),
                            (int)op.ai("p", 1), (int)op.ai("d", 0),
                            (int)op.ai("d", 1));
                }
                out = ggml_conv_2d_direct(
                        gctx, w4d, x, (int)op.ai("s", 0), (int)op.ai("s", 1),
                        (int)op.ai("p", 0), (int)op.ai("p", 1),
                        (int)op.ai("d", 0), (int)op.ai("d", 1));
            } else {
                out = conv2d_q(wT, kernel4d(op, wT), x, (int)op.ai("s", 0),
                               (int)op.ai("s", 1), (int)op.ai("p", 0),
                               (int)op.ai("p", 1), (int)op.ai("d", 0),
                               (int)op.ai("d", 1));
            }
        } else {
            ggml_tensor* w4d = kernel4d(op, wT);
            if (!depthwise && w4d->ne[2] != x->ne[2]) {
                YOLO_LOG_ERROR(
                        "conv %s type=%s: kernel ne=[%lld,%lld,%lld,%lld] vs "
                        "input ne=[%lld,%lld,%lld,%lld]",
                        prefix.c_str(), op.type.c_str(), (long long)w4d->ne[0],
                        (long long)w4d->ne[1], (long long)w4d->ne[2],
                        (long long)w4d->ne[3], (long long)x->ne[0],
                        (long long)x->ne[1], (long long)x->ne[2],
                        (long long)x->ne[3]);
            }
            if (depthwise) w4d = dw_kernel(w4d);
            out = depthwise
                          ? ggml_conv_2d_dw(
                                    gctx, w4d, x, (int)op.ai("s", 0),
                                    (int)op.ai("s", 1), (int)op.ai("p", 0),
                                    (int)op.ai("p", 1), (int)op.ai("d", 0),
                                    (int)op.ai("d", 1))
                          : ggml_conv_2d(gctx, w4d, x, (int)op.ai("s", 0),
                                         (int)op.ai("s", 1), (int)op.ai("p", 0),
                                         (int)op.ai("p", 1), (int)op.ai("d", 0),
                                         (int)op.ai("d", 1));
        }
        return add_bias_act(op, prefix, out);
    }

    ggml_tensor* conv_transpose(const OpDef& op,
                                const std::string& prefix,
                                ggml_tensor* x) {
        ggml_tensor* wT = w(prefix, "w");
        if (!wT) {
            YOLO_LOG_ERROR("transpose conv '%s' has no weight tensor '.w'",
                           prefix.c_str());
            return nullptr;
        }
        // CUDA f16 flow: ggml-cuda's conv_transpose kernel (p0) is
        // F32-only for input/dst (conv2d-transpose.cu asserts it and reads
        // both as float*); its reorder kernel consumes F16 weights natively.
        // Cast activations around the op — the same runtime-by-backend
        // pattern as interpolate below. Vulkan has a native F16 pipeline
        // (yolo_merged patch) and CPU consumes F16 directly, so both keep
        // the zero-copy path.
        const bool f32_detour = cuda_backend && x->type == GGML_TYPE_F16;
        if (f32_detour) x = ggml_cast(gctx, x, GGML_TYPE_F32);
        ggml_tensor* out =
                ggml_conv_transpose_2d_p0(gctx, wT, x, (int)op.ip("s"));
        if (f32_detour) out = ggml_cast(gctx, out, GGML_TYPE_F16);
        return add_bias_act(op, prefix, out);
    }

    // ggml's generic CPU depthwise lowering emits an F16 im2col even for F32
    // input, so its kernel must match. GPU sessions use direct convolution and
    // never enter this helper.
    ggml_tensor* dw_kernel(ggml_tensor* wT) {
        return wT->type == GGML_TYPE_F32 ? ggml_cast(gctx, wT, GGML_TYPE_F16)
                                         : wT;
    }

    // Quantized conv: ggml_conv_2d would build mul_mat(F16 im2col, Q8 kernel)
    // which asserts on CPU (src1 must be F32 or the kernel dtype). Mirror
    // llama.cpp instead: mul_mat(Q8 kernel [K,OC], F32 im2col) — src1 F32
    // gets dynamically quantized to the kernel's vec_dot type. w4d only
    // lends KH/KW/IC shape metadata to im2col; mul_mat consumes wT itself so
    // quant blocks stay contiguous.
    ggml_tensor* conv2d_q(ggml_tensor* wT,
                          ggml_tensor* w4d,
                          ggml_tensor* x,
                          int s0,
                          int s1,
                          int p0,
                          int p1,
                          int d0,
                          int d1) {
        ggml_tensor* im2 =
                ggml_im2col(gctx, w4d, x, s0, s1, p0, p1, d0, d1, true,
                            GGML_TYPE_F32);  // [K, OW, OH, N]
        const int64_t P = im2->ne[1] * im2->ne[2] * im2->ne[3];
        ggml_tensor* dst = ggml_mul_mat(
                gctx, wT, ggml_reshape_2d(gctx, im2, im2->ne[0], P));
        dst = ggml_reshape_4d(gctx, dst, wT->ne[1], im2->ne[1], im2->ne[2],
                              im2->ne[3]);  // (N, OH, OW, OC)
        // permute semantics: ne[axis_i] = old ne[i] — send OC to slot 2,
        // W/H to 0/1.
        return ggml_cont(gctx,
                         ggml_permute(gctx, dst, 2, 0, 1, 3));  // [W,H,OC,N]
    }

    // 1x1 / depthwise convs inside psa_attention (no act). Quantized weights
    // are stored 2D [K, out]; the 4D view only lends shape metadata for
    // im2col.
    ggml_tensor* attention_conv(const std::string& prefix,
                                const char* tag,
                                ggml_tensor* x,
                                int64_t k = 1) {
        ggml_tensor* wT = w(prefix, (std::string(tag) + "_w").c_str());
        if (!wT) {
            YOLO_LOG_ERROR("attention '%s' has no weight tensor '%s_w'",
                           prefix.c_str(), tag);
            return nullptr;
        }
        ggml_tensor* out;
        if (ggml_is_quantized(wT->type) && k == 1) {
            ggml_tensor* w4d =
                    ggml_reshape_4d(gctx, wT, 1, 1, wT->ne[0], wT->ne[1]);
            if (q8_direct && use_direct_conv) {
                // PSA attention feeds its proj conv F32 (mul_mat/softmax
                // chain); the igemm Q8 path takes F16 activations.
                if (x->type != GGML_TYPE_F16)
                    x = ggml_cast(gctx, x, GGML_TYPE_F16);
                out = ggml_conv_2d_direct(gctx, w4d, x, 1, 1, 0, 0, 1, 1);
            } else {
                out = conv2d_q(wT, w4d, x, 1, 1, 0, 0, 1, 1);
            }
        } else {
            if (wT->ne[2] == 1 && wT->ne[3] == 1) {
                wT = ggml_reshape_4d(gctx, wT, k, k, wT->ne[0] / (k * k),
                                     wT->ne[1]);
            }
            const bool direct_types =
                    wT->type == x->type &&
                    (wT->type == GGML_TYPE_F32 || wT->type == GGML_TYPE_F16);
            if (use_direct_conv && direct_types) {
                out = k > 1 ? ggml_conv_2d_dw_direct(gctx, wT, x, 1, 1,
                                                     (int)(k / 2), (int)(k / 2),
                                                     1, 1)
                            : ggml_conv_2d_direct(gctx, wT, x, 1, 1, 0, 0, 1,
                                                  1);
            } else {
                out = k > 1 ? ggml_conv_2d_dw(gctx, dw_kernel(wT), x, 1, 1,
                                              (int)(k / 2), (int)(k / 2), 1, 1)
                            : ggml_conv_2d(gctx, wT, x, 1, 1, 0, 0, 1, 1);
            }
        }
        if (ggml_tensor* b = w(prefix, (std::string(tag) + "_b").c_str())) {
            out = ggml_add(gctx, out,
                           ggml_reshape_4d(gctx, b, 1, 1, b->ne[0], 1));
        }
        return out;
    }

    ggml_tensor* psa_attention(const OpDef& op,
                               const std::string& prefix,
                               ggml_tensor* x) {
        const int64_t nh = op.ip("nh"), kd = op.ip("kd"), hd = op.ip("hd");
        const float scale = op.fparams.count("scale")
                                    ? (float)op.fparams.at("scale")
                                    : 1.0f;
        const int64_t W = x->ne[0], H = x->ne[1], N = x->ne[3];
        const int64_t HW = W * H, k2d = 2 * kd + hd, C = nh * hd;

        ggml_tensor* qkv = attention_conv(prefix, "qkv", x);  // [W,H,nh*k2d,N]
        // torch: qkv.view(B, nh, k2d, N) — token dim innermost, channel outer.
        qkv = ggml_reshape_4d(gctx, qkv, HW, k2d, nh, N);  // [tokens,k2d,nh,N]
        auto view = [&](int64_t start, int64_t len) {
            return ggml_cont(
                    gctx,
                    ggml_view_4d(gctx, qkv, HW, len, nh, N, qkv->nb[1],
                                 qkv->nb[2], qkv->nb[3], start * qkv->nb[1]));
        };
        ggml_tensor* q = ggml_scale(gctx, view(0, kd), scale);  // [HW,kd,nh,N]
        ggml_tensor* k = view(kd, kd);
        ggml_tensor* v = view(2 * kd, hd);  // [HW,hd,nh,N]

        // torch: attn = softmax((q*scale)^T @ k, dim=-1); x = v @ attn^T.
        // ggml dst[m,n] = sum_k A[k,m]B[k,n] with ne0 from A->ne1, so
        // mul_mat(kT, qT) puts k-tokens on ne0 — ggml_soft_max then
        // normalizes over keys exactly like torch dim=-1 (llama.cpp KQ
        // pattern). mul_mat needs non-transposed contiguous operands, hence
        // the cont(permute)s.
        ggml_tensor* qT = ggml_cont(gctx, ggml_permute(gctx, q, 1, 0, 2, 3));
        ggml_tensor* kT = ggml_cont(gctx, ggml_permute(gctx, k, 1, 0, 2, 3));
        ggml_tensor* attn = ggml_soft_max(gctx, ggml_mul_mat(gctx, kT, qT));
        ggml_tensor* out = ggml_mul_mat(gctx, v, attn);  // [hd,q_tok,nh,N]
        out = ggml_reshape_4d(
                gctx, ggml_cont(gctx, ggml_permute(gctx, out, 1, 0, 2, 3)), W,
                H, C, N);

        // pe: depthwise 3x3 on v, residual, proj 1x1
        ggml_tensor* v_img =
                ggml_reshape_4d(gctx, ggml_cont(gctx, v), W, H, C, N);
        ggml_tensor* pe = attention_conv(prefix, "pe", v_img, 3);
        ggml_tensor* sum = ggml_add(gctx, out, pe);
        return attention_conv(prefix, "proj", sum);
    }

    // ------------------------------------------------------------------
    // YOLO-World / YOLOE ops (open-vocabulary heads, op-graph v3/v4).
    // In-tree port of upstream cpp_ggml yolo_graph.cpp, with missing-weight
    // paths returning nullptr + a logged error instead of asserting.
    // ------------------------------------------------------------------

    // Element-wise max without a native ggml op: max(a, b) == b + relu(a-b).
    ggml_tensor* max2(ggml_tensor* a, ggml_tensor* b) {
        return ggml_add(gctx, b, ggml_relu(gctx, ggml_sub(gctx, a, b)));
    }

    // LayerNorm(ct) + Linear(ct -> ec) over the ne0 (column) axis of `src`.
    // ggml_norm/scale are F32-only on the CPU backend, so F16 inputs are
    // cast up for the norm+linear math and cast back on exit.
    ggml_tensor* ln_linear(const std::string& prefix,
                           const char* tag,
                           ggml_tensor* src,
                           float eps) {
        const enum ggml_type in_type = src->type;
        if (src->type != GGML_TYPE_F32)
            src = ggml_cast(gctx, src, GGML_TYPE_F32);
        ggml_tensor* y = ggml_norm(gctx, src, eps);
        if (ggml_tensor* w_ = w(prefix, (std::string(tag) + "_ln_w").c_str())) {
            if (w_->type != GGML_TYPE_F32)
                w_ = ggml_cast(gctx, w_, GGML_TYPE_F32);
            y = ggml_mul(gctx, y, w_);
        }
        if (ggml_tensor* b_ = w(prefix, (std::string(tag) + "_ln_b").c_str())) {
            if (b_->type != GGML_TYPE_F32)
                b_ = ggml_cast(gctx, b_, GGML_TYPE_F32);
            y = ggml_add(gctx, y, b_);
        }
        if (ggml_tensor* w_ = w(prefix, (std::string(tag) + "_w").c_str())) {
            if (w_->type != GGML_TYPE_F32)
                w_ = ggml_cast(gctx, w_, GGML_TYPE_F32);
            y = ggml_mul_mat(gctx, w_, y);
        }
        if (ggml_tensor* b_ = w(prefix, (std::string(tag) + "_b").c_str())) {
            y = ggml_add(gctx, y, ggml_reshape_2d(gctx, b_, b_->ne[0], 1));
        }
        if (in_type != GGML_TYPE_F32) y = ggml_cast(gctx, y, in_type);
        return y;
    }

    // Exact AdaptiveMaxPool2d(k, k): ATen window edges (start = floor(i*dim/k),
    // end = ceil((i+1)*dim/k)); windows overlap when dim % k != 0, so each
    // output cell is a separate view_4d + pool_2d pair assembled with
    // concat/permute into a [C, k*k] (channel, patch) tensor.
    ggml_tensor* adaptive_max_pool2d(ggml_tensor* x, int k) {
        const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], N = x->ne[3];
        GGML_ASSERT(N == 1);
        auto win = [&](int i, int64_t dim) -> std::pair<int, int> {
            const int64_t s = ((int64_t)i * dim) / k;
            const int64_t e = ((int64_t)(i + 1) * dim + k - 1) / k;
            return {(int)s, (int)e};
        };
        std::vector<ggml_tensor*> rows;
        for (int i = 0; i < k; i++) {
            auto [h0, h1] = win(i, H);
            std::vector<ggml_tensor*> cols;
            for (int j = 0; j < k; j++) {
                auto [w0, w1] = win(j, W);
                ggml_tensor* v = ggml_cont(
                        gctx, ggml_view_4d(gctx, x, w1 - w0, h1 - h0, C, 1,
                                           x->nb[1], x->nb[2], x->nb[3],
                                           ((size_t)h0 * W + w0) * x->nb[0]));
                // cont() above is REQUIRED, not cosmetic: the GPU pool_2d
                // kernels assume a contiguous source, while the ATen-window
                // view is a strided sub-block of x (CPU handles strides
                // correctly, CUDA/Vulkan silently misread → corrupted
                // text-update → detections dropped). The window is at most
                // k×k cells of one level, so the copy is negligible.
                // ggml_pool_2d inherits the input type, but the CPU pool
                // kernel unconditionally writes F32; an F16 dst would
                // overflow its buffer by 2x. Cast up so the pool output is
                // F32-sized.
                if (v->type != GGML_TYPE_F32)
                    v = ggml_cast(gctx, v, GGML_TYPE_F32);
                cols.push_back(ggml_pool_2d(gctx, v, GGML_OP_POOL_MAX, w1 - w0,
                                            h1 - h0, w1 - w0, h1 - h0, 0, 0));
            }
            ggml_tensor* row = cols[0];
            for (int j = 1; j < k; j++)
                row = ggml_concat(gctx, row, cols[j], 0);  // [k, 1, C]
            rows.push_back(ggml_cont(
                    gctx, ggml_permute(gctx, row, 1, 0, 2, 3)));  // [1,k,C]
        }
        ggml_tensor* grid = rows[0];
        for (int i = 1; i < k; i++)
            grid = ggml_concat(gctx, grid, rows[i], 0);  // [k, k, C]
        // permute(2,1,0,3) -> [C, k, k]; flatten (j, i) row-major to match
        // torch adaptive_max_pool2d(...).view(B, C, -1).
        grid = ggml_cont(gctx, ggml_permute(gctx, grid, 2, 1, 0, 3));
        return ggml_reshape_2d(gctx, grid, C, k * k);
    }

    // MaxSigmoidAttnBlock gate:
    // out = proj_conv(x) * sigmoid(max_n(embed . guide_n)/sqrt(hc) + bias).
    // embed [w,h,ec] and proj [w,h,c2] come from plain conv ops; text is
    // [512, nc] (nc rows of 512-d CLIP text embeddings). The max over nc
    // classes is a static tree of max2 nodes because nc is fixed at session
    // creation. Head bias scalars come from the load-time scalar_params
    // extraction (never from HostTensor.data — release-safe rebuilds).
    ggml_tensor* max_sigmoid_attn(const OpDef& op,
                                  const std::string& prefix,
                                  ggml_tensor* embed,
                                  ggml_tensor* proj,
                                  ggml_tensor* text) {
        const int64_t nh = op.ip("nh"), hc = op.ip("hc");
        const int64_t W = embed->ne[0], H = embed->ne[1];
        const int64_t HW = W * H, c2 = proj->ne[2], nc = text->ne[1];

        ggml_tensor* gl_w = w(prefix, "gl_w");
        if (!gl_w) {
            YOLO_LOG_ERROR("max_sigmoid_attn '%s' has no gl_w", prefix.c_str());
            return nullptr;
        }
        ggml_tensor* guide = ggml_mul_mat(gctx, gl_w, text);  // [ec, nc]
        // mul_mat returns F32. Keep F32 reference graphs in F32, while F16
        // deployment graphs use the native F16 CUDA GEMM contract.
        if (text->type == GGML_TYPE_F16)
            guide = ggml_cast(gctx, guide, GGML_TYPE_F16);
        if (ggml_tensor* b = w(prefix, "gl_b"))
            guide = ggml_add(gctx, guide,
                             ggml_reshape_2d(gctx, b, b->ne[0], 1));

        // embed -> [ec, HW] with the channel axis on ne0 (mul_mat weight
        // side). permute(1,2,0,3): [W, H, ec] -> [ec, W, H]. The 2D is a
        // RESHAPE, not a view: eT is contiguous, so (ec, W*H) is a free
        // re-label — and every GPU kernel below (mul_mat / sum_rows / the
        // elementwise max tree) requires contiguous inputs (non-contiguous
        // views segfault or silently misread on CUDA/Vulkan).
        ggml_tensor* eT =
                ggml_cont(gctx, ggml_permute(gctx, embed, 1, 2, 0, 3));
        ggml_tensor* e2 = ggml_reshape_2d(gctx, eT, embed->ne[2],
                                          HW);  // [ec, HW] contiguous
        ggml_tensor* pT = ggml_cont(gctx, ggml_permute(gctx, proj, 1, 2, 0, 3));
        ggml_tensor* p2 = ggml_reshape_2d(gctx, pT, c2,
                                          HW);  // [c2, HW] contiguous

        // Head bias is a per-head scalar constant baked into the graph from
        // the load-time scalar_params extraction.
        auto bias_it = model.scalar_params.find(prefix + ".bias");
        const float* bias_data = bias_it != model.scalar_params.end()
                                         ? bias_it->second.data()
                                         : nullptr;
        std::vector<ggml_tensor*> head_outs;
        for (int64_t m = 0; m < nh; m++) {
            // Per-head slices are strided views of contiguous matrices; the
            // GPU kernels they feed (mul_mat / scale_bias) require
            // contiguous inputs, so materialize each slice (tiny: hc rows).
            ggml_tensor* g_m = ggml_cont(
                    gctx, ggml_view_2d(gctx, guide, hc, nc, guide->nb[1],
                                       m * hc * ggml_element_size(guide)));
            ggml_tensor* e_m = ggml_cont(
                    gctx, ggml_view_2d(gctx, e2, hc, HW, e2->nb[1],
                                       m * hc * ggml_element_size(e2)));
            // CUDA requires matching F32/F16 operands for mul_mat. Choose
            // the activation type from the actual guide tensor rather than
            // the model's nominal dtype (upstream parity).
            if (e_m->type != guide->type)
                e_m = ggml_cast(gctx, e_m, guide->type);
            ggml_tensor* aw = ggml_mul_mat(gctx, g_m, e_m);  // [nc, HW]
            // Tree-max over the nc rows (torch aw.max(dim=-1)). Transpose
            // once to [HW, nc] so each class row is a CONTIGUOUS 1 x HW
            // view (a column view of aw would be strided — see the note
            // above on GPU kernels and non-contiguous inputs).
            ggml_tensor* awT =
                    ggml_cont(gctx, ggml_permute(gctx, aw, 1, 0, 2, 3));
            std::vector<ggml_tensor*> rows;
            for (int64_t n = 0; n < nc; n++) {
                rows.push_back(ggml_view_2d(gctx, awT, HW, 1, awT->nb[0],
                                            n * awT->nb[1]));
            }
            while (rows.size() > 1) {
                std::vector<ggml_tensor*> nxt;
                for (size_t j = 0; j + 1 < rows.size(); j += 2)
                    nxt.push_back(max2(rows[j], rows[j + 1]));
                if (rows.size() % 2) nxt.push_back(rows.back());
                rows.swap(nxt);
            }
            ggml_tensor* p_m = ggml_cont(
                    gctx, ggml_view_2d(gctx, p2, hc, HW, p2->nb[1],
                                       m * hc * ggml_element_size(p2)));
            // torch: aw / sqrt(hc) + bias[m] -> sigmoid (scale_bias fold).
            // ggml scale_bias is F32-only on CPU: cast up and back for F16.
            ggml_tensor* aw1 = ggml_scale_bias(
                    gctx, ggml_cast(gctx, rows[0], GGML_TYPE_F32),
                    1.0f / std::sqrt((float)hc),
                    bias_data ? bias_data[m] : 0.0f);
            aw1 = ggml_sigmoid(gctx, aw1);
            if (p_m->type != GGML_TYPE_F32)
                aw1 = ggml_cast(gctx, aw1, p_m->type);
            aw1 = ggml_reshape_2d(gctx, aw1, 1, HW);
            head_outs.push_back(
                    ggml_mul(gctx, p_m, aw1));  // [hc,HW] x [1,HW] broadcast
        }
        ggml_tensor* out = head_outs[0];
        for (size_t m = 1; m < head_outs.size(); m++)
            out = ggml_concat(gctx, out, head_outs[m], 0);  // [c2, HW]
        return ggml_reshape_3d(
                gctx, ggml_cont(gctx, ggml_permute(gctx, out, 1, 0, 2, 3)), W,
                H, c2);
    }

    // ImagePoolingAttn: image tokens attend into the text embedding
    // (residual). Each input is a [w,h,ec] 1x1-projected feature map; text
    // is [512, nc].
    ggml_tensor* image_pooling_attn(const OpDef& op,
                                    const std::string& prefix,
                                    const std::vector<ggml_tensor*>& feats,
                                    ggml_tensor* text) {
        const int64_t nh = op.ip("nh"), hc = op.ip("hc"), k = op.ip("k", 3);
        const int64_t nc = text->ne[1];
        // ggml_pool_2d outputs F32, so the attention math runs in F32 and
        // the result is cast back to the text type for the residual.
        ggml_tensor* text32 = text->type == GGML_TYPE_F32
                                      ? text
                                      : ggml_cast(gctx, text, GGML_TYPE_F32);
        // 1. adaptive max pool each level -> [ec, k*k] patches, concat over
        // patches.
        ggml_tensor* xcat = adaptive_max_pool2d(feats[0], (int)k);
        for (size_t f = 1; f < feats.size(); f++) {
            xcat = ggml_concat(gctx, xcat,
                               adaptive_max_pool2d(feats[f], (int)k), 1);
        }
        // 2. q = query(text), k/v = key/value(x); LayerNorm over channels.
        ggml_tensor* xT = xcat;  // [ec, P]
        ggml_tensor* kT = ln_linear(prefix, "key", xT, 1e-5f);
        ggml_tensor* vT = ln_linear(prefix, "value", xT, 1e-5f);
        ggml_tensor* q = ln_linear(prefix, "query", text32, 1e-5f);
        const int64_t P = xT->ne[1];
        // 3. per-head scaled dot-product attention (llama.cpp KQ pattern).
        std::vector<ggml_tensor*> head_outs;
        for (int64_t m = 0; m < nh; m++) {
            // Per-head slices are strided views of contiguous matrices;
            // materialize them (tiny) for the GPU mul_mat/softmax kernels.
            ggml_tensor* q_m = ggml_cont(
                    gctx, ggml_view_2d(gctx, q, hc, nc, q->nb[1],
                                       m * hc * ggml_element_size(q)));
            ggml_tensor* k_m = ggml_cont(
                    gctx, ggml_view_2d(gctx, kT, hc, P, kT->nb[1],
                                       m * hc * ggml_element_size(kT)));
            ggml_tensor* v_m = ggml_cont(
                    gctx, ggml_view_2d(gctx, vT, hc, P, vT->nb[1],
                                       m * hc * ggml_element_size(vT)));
            ggml_tensor* aw = ggml_mul_mat(gctx, k_m, q_m);  // [P, nc]
            aw = ggml_scale(gctx, aw, 1.0f / std::sqrt((float)hc));
            aw = ggml_soft_max(gctx, aw);  // over keys (torch dim=-1)
            ggml_tensor* vT_m = ggml_cont(
                    gctx, ggml_permute(gctx, v_m, 1, 0, 2, 3));  // [P, hc]
            head_outs.push_back(ggml_mul_mat(gctx, vT_m, aw));   // [hc, nc]
        }
        ggml_tensor* out = head_outs[0];
        for (size_t m = 1; m < head_outs.size(); m++)
            out = ggml_concat(gctx, out, head_outs[m], 0);  // [ec, nc]
        ggml_tensor* pw = w(prefix, "proj_w");
        if (!pw) {
            YOLO_LOG_ERROR("image_pooling_attn '%s' has no proj_w",
                           prefix.c_str());
            return nullptr;
        }
        if (pw->type != GGML_TYPE_F32) pw = ggml_cast(gctx, pw, GGML_TYPE_F32);
        out = ggml_mul_mat(gctx, pw, out);  // [ct, nc]
        if (ggml_tensor* b = w(prefix, "proj_b"))
            out = ggml_add(gctx, out, ggml_reshape_2d(gctx, b, b->ne[0], 1));
        out = ggml_add(gctx, out, text32);  // residual (scale 1.0 in World)
        if (text->type != GGML_TYPE_F32) out = ggml_cast(gctx, out, text->type);
        return out;  // [512, nc]
    }

    // WorldDetect and YOLOE: contrastive embedding branch + plain detect
    // decode. feats alternate [box0, emb0, (mask0,) box1, ...]; text is
    // [512, nc].
    ggml_tensor* world_detect(const OpDef& op,
                              const std::string& prefix,
                              const std::vector<ggml_tensor*>& feats,
                              ggml_tensor* text,
                              int64_t nc) {
        const int64_t rm = op.ip("reg_max", 16);
        // L2-normalisation needs F32 (ggml_sum_rows is F32-only); cast the
        // F16 graph text/embedding back for the contrastive head math.
        ggml_tensor* text32 = text->type == GGML_TYPE_F32
                                      ? text
                                      : ggml_cast(gctx, text, GGML_TYPE_F32);
        ggml_tensor* sq = ggml_sqr(gctx, text32);
        ggml_tensor* t_norm = ggml_div(
                gctx, text32, ggml_sqrt(gctx, ggml_sum_rows(gctx, sq)));
        ggml_tensor* out = nullptr;
        const bool has_masks = op.ip("has_masks", 0) != 0;
        const bool bn_contrastive = op.ip("bn_contrastive", 0) != 0;
        const size_t stride = has_masks ? 3 : 2;
        const size_t n_levels = feats.size() / stride;
        for (size_t l = 0; l < n_levels; l++) {
            ggml_tensor* box = feats[stride * l];      // [w, h, 4*rm]
            ggml_tensor* emb = feats[stride * l + 1];  // [w, h, embed]
            const int64_t W = box->ne[0], H = box->ne[1], HW = W * H;
            // World normalizes image embeddings. YOLOE's BNContrastiveHead
            // instead applies its folded BatchNorm affine transform. The
            // [embed, HW] matrix is a RESHAPE of the contiguous [embed, W,
            // H] tensor (not a view): the L2-normalise chain below
            // (sum_rows/sqr/sqrt/div) and the final mul_mat require
            // contiguous inputs on the GPU backends.
            ggml_tensor* eT = ggml_cont(
                    gctx, ggml_permute(gctx, emb, 1, 2, 0, 3));  // [em,W,H]
            ggml_tensor* eT32 = eT->type == GGML_TYPE_F32
                                        ? eT
                                        : ggml_cast(gctx, eT, GGML_TYPE_F32);
            ggml_tensor* e2 = ggml_reshape_2d(gctx, eT32, emb->ne[2],
                                              HW);  // contiguous
            if (bn_contrastive) {
                ggml_tensor* scale =
                        w(prefix,
                          ("cv4_" + std::to_string(l) + "_bn_scale").c_str());
                ggml_tensor* shift =
                        w(prefix,
                          ("cv4_" + std::to_string(l) + "_bn_shift").c_str());
                if (!scale || !shift) {
                    YOLO_LOG_ERROR(
                            "world head '%s' missing BN contrastive affine "
                            "tensors",
                            prefix.c_str());
                    return nullptr;
                }
                if (scale->type != GGML_TYPE_F32)
                    scale = ggml_cast(gctx, scale, GGML_TYPE_F32);
                if (shift->type != GGML_TYPE_F32)
                    shift = ggml_cast(gctx, shift, GGML_TYPE_F32);
                e2 = ggml_mul(gctx, e2,
                              ggml_reshape_2d(gctx, scale, scale->ne[0], 1));
                e2 = ggml_add(gctx, e2,
                              ggml_reshape_2d(gctx, shift, shift->ne[0], 1));
            } else {
                e2 = ggml_div(
                        gctx, e2,
                        ggml_sqrt(gctx,
                                  ggml_sum_rows(gctx, ggml_sqr(gctx, e2))));
            }
            ggml_tensor* scores = ggml_mul_mat(gctx, t_norm, e2);  // [nc,HW]
            // scores = scores * logit_scale.exp() + bias (per level), from
            // the load-time scalar_params extraction.
            const std::string lv = std::to_string(l);
            auto ls_it = model.scalar_params.find(prefix + ".cv4_" + lv +
                                                  "_logit_scale");
            auto bs_it =
                    model.scalar_params.find(prefix + ".cv4_" + lv + "_bias");
            const float ls = ls_it != model.scalar_params.end()
                                     ? ls_it->second[0]
                                     : 1.0f;
            const float bs = bs_it != model.scalar_params.end()
                                     ? bs_it->second[0]
                                     : 0.0f;
            scores = ggml_scale_bias(gctx, scores, ls, bs);
            ggml_tensor* s4 = ggml_reshape_3d(
                    gctx,
                    ggml_cont(gctx, ggml_permute(gctx, scores, 1, 0, 2, 3)), W,
                    H, nc);
            if (s4->type != box->type)
                s4 = ggml_cast(gctx, s4, box->type);  // concat type match
            ggml_tensor* level = ggml_concat(gctx, box, s4, 2);
            if (has_masks) {
                ggml_tensor* mask = feats[stride * l + 2];
                level = ggml_concat(gctx, level, mask, 2);
            }
            const int64_t level_no =
                    4 * rm + nc + (has_masks ? op.ip("nm", 0) : 0);
            ggml_tensor* r = ggml_reshape_2d(gctx, level, HW, level_no);
            out = out ? ggml_concat(gctx, out, r, 0) : r;
        }
        return out;  // [A, 4*rm + nc (+ nm)]
    }

    // YOLOE v4: the graph text input is the normalised pre-reprta MobileCLIP
    // feature; apply the checkpoint's reprta residual (Residual(SwiGLUFFN))
    // so the head sees what torch's get_tpe produces. The trailing L2
    // normalise of get_tpe is free: world_detect L2-normalises its text
    // input regardless.
    ggml_tensor* reprta(const std::string& prefix, ggml_tensor* x) {
        ggml_tensor* w12 = w(prefix, "reprta_w12_w");
        ggml_tensor* b12 = w(prefix, "reprta_w12_b");
        ggml_tensor* w3 = w(prefix, "reprta_w3_w");
        ggml_tensor* b3 = w(prefix, "reprta_w3_b");
        if (!w12 || !b12 || !w3 || !b3) {
            YOLO_LOG_ERROR("YOLOE reprta tensors incomplete at %s",
                           prefix.c_str());
            return nullptr;
        }
        if (w12->type != GGML_TYPE_F32)
            w12 = ggml_cast(gctx, w12, GGML_TYPE_F32);
        if (w3->type != GGML_TYPE_F32) w3 = ggml_cast(gctx, w3, GGML_TYPE_F32);
        ggml_tensor* x12 = ggml_mul_mat(gctx, w12, x);  // [2*hidden, nc]
        x12 = ggml_add(gctx, x12, ggml_reshape_2d(gctx, b12, b12->ne[0], 1));
        const int64_t hh = x12->ne[0] / 2;
        ggml_tensor* x1 =
                ggml_view_2d(gctx, x12, hh, x12->ne[1], x12->nb[1], 0);
        ggml_tensor* x2 = ggml_view_2d(gctx, x12, hh, x12->ne[1], x12->nb[1],
                                       (size_t)hh * x12->nb[0]);
        // ggml-cuda kernels reject strided views, so both SwiGLU halves must
        // be made contiguous or the text tower falls back to CPU and syncs
        // the graph every frame.
        x1 = ggml_cont(gctx, x1);
        x2 = ggml_cont(gctx, x2);
        ggml_tensor* hidden = ggml_mul(gctx, ggml_silu(gctx, x1), x2);
        ggml_tensor* y = ggml_mul_mat(gctx, w3, hidden);  // [512, nc]
        y = ggml_add(gctx, y, ggml_reshape_2d(gctx, b3, b3->ne[0], 1));
        return ggml_add(gctx, x, y);  // Residual: x + SwiGLUFFN(x)
    }

    // ------------------------------------------------------------------
    // YOLOE SAVPE (Small Adoptable Visual Prompt Encoder) — the official
    // visual-prompt path: example boxes on the image are rasterized into
    // P3-resolution binary masks, and the encoder turns them into [Q, 512]
    // class embeddings that REPLACE the MobileCLIP text embeddings as the
    // head's cls_pe (no reprta residual, no text tower; in-tree port of
    // ultralytics nn/modules/block.py SAVPE + YOLOEDetect.get_vpe). All
    // convs ride the shared conv2d path (dtype rules + direct-conv fast
    // paths) through synthetic OpDefs; weights follow the GraphBuilder
    // naming convention written by
    // core/AICore/src/tasks/yolo/tools/convert_yoloe_savpe_gguf.py
    // ("savpe.cv1_0_0.w" = GraphBuilder w("savpe.cv1_0_0", "w")).
    // ------------------------------------------------------------------
    ggml_tensor* savpe_conv(
            const char* tag, ggml_tensor* x, int k, int s, int p, bool silu) {
        OpDef op;
        op.type = "conv";
        op.aparams["s"] = {(int64_t)s, (int64_t)s};
        op.aparams["p"] = {(int64_t)p, (int64_t)p};
        op.aparams["d"] = {1, 1};
        op.aparams["k"] = {(int64_t)k, (int64_t)k};
        if (silu) op.sparams["act"] = "silu";
        return conv2d(op, std::string("savpe.") + tag, x);
    }

    ggml_tensor* savpe(const std::vector<ggml_tensor*>& fpn, ggml_tensor* vp) {
        // vp: external [W3, H3, Q, 1] F32 binary masks on the P3 grid.
        const int64_t W3 = fpn[0]->ne[0], H3 = fpn[0]->ne[1];
        const int64_t HW3 = W3 * H3;
        const int64_t Q = vp->ne[2];

        // Shared branches: activation (cv2: Conv1x1 -> Upsample) and
        // semantic (cv1: Conv3x3 -> Conv3x3 -> Upsample) per level — the
        // convs run at the level's OWN resolution and the upsample comes
        // LAST (Sequential order), landing every level on the P3 grid.
        ggml_tensor* act = nullptr;  // [W3, H3, c3]
        ggml_tensor* sem = nullptr;  // [W3, H3, c3]
        for (int l = 0; l < 3; l++) {
            const std::string lv = std::to_string(l);
            dbg_savpe_fpn[l] = fpn[l];
            ggml_tensor* a =
                    savpe_conv(("cv2_" + lv).c_str(), fpn[l], 1, 1, 0, true);
            dbg_savpe_cv2[l] = a;
            ggml_tensor* s0 = savpe_conv(("cv1_" + lv + "_0").c_str(), fpn[l],
                                         3, 1, 1, true);
            ggml_tensor* s1 =
                    savpe_conv(("cv1_" + lv + "_1").c_str(), s0, 3, 1, 1, true);
            if (l > 0) {
                a = ggml_upscale(gctx, a, 1 << l, GGML_SCALE_MODE_NEAREST);
                s1 = ggml_upscale(gctx, s1, 1 << l, GGML_SCALE_MODE_NEAREST);
            }
            act = act ? ggml_concat(gctx, act, a, 2) : a;
            sem = sem ? ggml_concat(gctx, sem, s1, 2) : s1;
        }
        // cv4 (3x3, plain) -> [W3, H3, 16]; cv3 (1x1, plain) -> [W3, H3, 512]
        ggml_tensor* y = savpe_conv("cv4", act, 3, 1, 1, false);
        ggml_tensor* x = savpe_conv("cv3", sem, 1, 1, 0, false);
        if (!y || !x) return nullptr;
        dbg_savpe_x = x;
        dbg_savpe_y = y;

        // Channels-first staging of the semantic map: [HW3, C] F32 rows,
        // p = h*W + w — the SAME grid order the mask leaf and s_all use
        // (torch's [C,H,W].flatten(2) layout). x's own memory already IS
        // that layout ([W, H, C] with p innermost), so a plain reshape —
        // no permute — reinterprets it; the per-group views below index it
        // row-major ((p, ch) at p + ch*nb[1]). A permute-based staging here
        // scrambles channels against grid positions and silently zeroes
        // visual-prompt detections: permute maps a's dim i into slot axis_i
        // (ne[axis_i] = a->ne[i]), so cont(permute(x, 2,0,1,3)) yields the
        // flat order h + H3*c + H3*C*w, which no clean [C,HW3]/[HW3,C]
        // reinterpretation of the [W3,H3,C] source can reproduce — only the
        // raw memory order matches flatten(2).
        ggml_tensor* xT = x;
        if (xT->type != GGML_TYPE_F32) xT = ggml_cast(gctx, xT, GGML_TYPE_F32);
        xT = ggml_reshape_2d(gctx, xT, HW3, x->ne[2]);

        // Per prompt q: cv5(mask) -> cat(y, .) -> cv6 -> masked softmax over
        // the P3 grid -> one [HW3, 16] score column block.
        ggml_tensor* s_all = nullptr;  // [HW3, 16*Q], prompt blocks in order
        for (int64_t q = 0; q < Q; q++) {
            ggml_tensor* vp_q = ggml_cont(
                    gctx, ggml_view_4d(gctx, vp, W3, H3, 1, 1, vp->nb[1],
                                       vp->nb[2], vp->nb[3], q * vp->nb[2]));
            ggml_tensor* m = savpe_conv("cv5", vp_q, 3, 1, 1, false);
            // Concat type match (same pattern as world_detect's s4): the
            // mask branch consumes the F32 external leaf, so on the GPU
            // f16 activation flows cv5 lands on F32 while cv4's `y` is F16
            // — cast the mask column to the activation flow's dtype before
            // the concat (the following cv6 convs then see one dtype).
            if (m->type != y->type) m = ggml_cast(gctx, m, y->type);
            ggml_tensor* yq = ggml_concat(gctx, y, m, 2);  // [W3,H3,32]
            yq = savpe_conv("cv6_0", yq, 3, 1, 1, true);
            yq = savpe_conv("cv6_1", yq, 3, 1, 1, false);

            // Stage per-group scores: [16, HW3] F32, rows = groups, p = h*W+w
            // grid order (torch's [16,H,W].flatten(2)). yq's memory is that
            // order transposed ([HW3, 16] rows, p innermost), so reshape to
            // [HW3, 16], transpose-view, then cont materializes the
            // group-major layout the masked-mul broadcast (src1 [1, HW3]
            // must tile src0) and the s_all column offset below require.
            ggml_tensor* yT = yq;
            if (yT->type != GGML_TYPE_F32)
                yT = ggml_cast(gctx, yT, GGML_TYPE_F32);
            yT = ggml_cont(gctx, ggml_permute(gctx,
                                              ggml_reshape_2d(gctx, yT, HW3,
                                                              yq->ne[2]),
                                              1, 0, 2, 3));  // [16, HW3]

            // score = y * vp + (1 - vp) * finfo.min, softmax over the grid.
            // For the binary mask leaf, (1 - vp) == relu(1 - vp) via one
            // scale_bias. The previous relu(-vp) identity is identically
            // ZERO for non-negative vp, so the -1e30 outside gate never
            // fired and the softmax leaked uniform mass over the whole P3
            // grid — the aggregated vpe collapsed into the global image
            // average and the contrastive head scored nothing.
            //
            // The mask column broadcasts over the 16 group rows: ggml binary
            // ops require src1 to tile src0 (each src0 dim a multiple of the
            // src1 dim), so the mask must be [1, HW3] next to yT's [16, HW3].
            ggml_tensor* vp_col = ggml_reshape_2d(
                    gctx, ggml_reshape_1d(gctx, vp_q, HW3), 1, HW3);
            ggml_tensor* outside = ggml_scale(
                    gctx,
                    ggml_relu(gctx, ggml_scale_bias(gctx, vp_col, -1.0f, 1.0f)),
                    -1e30f);
            ggml_tensor* masked =
                    ggml_add(gctx, ggml_mul(gctx, yT, vp_col), outside);
            // Softmax must run over the grid (ne0): transpose to [HW3, 16]
            // (ne[axis_i] = a->ne[i], so (1,0,2,3) swaps the first two dims).
            ggml_tensor* s_q =
                    ggml_cont(gctx, ggml_permute(gctx, masked, 1, 0, 2, 3));
            s_q = ggml_soft_max(gctx, s_q);
            s_all = s_all ? ggml_concat(gctx, s_all, s_q, 1) : s_q;
        }

        // Grouped aggregation: for each of the 16 channel groups,
        // agg_g = X_g @ S_g with X_g = channels [g*32, g*32+32) of the
        // semantic map and S_g = the prompt-q score columns of that group;
        // concat over groups -> [512, Q] (the head's cls_pe layout).
        const int64_t c16 = y->ne[2];
        const int64_t d = x->ne[2] / c16;  // 32 for the shipped 512-d models
        ggml_tensor* out = nullptr;
        for (int64_t g = 0; g < c16; g++) {
            ggml_tensor* x_g =
                    ggml_cont(gctx, ggml_view_2d(gctx, xT, HW3, d, xT->nb[1],
                                                 (size_t)(g * d) * xT->nb[1]));
            // Column-offset into s_all: S_g[p, q] = s_all[p, q*16 + g] — the
            // softmax weight of GROUP g at grid row p for prompt q (official
            // SAVPE aggregation). The previous row-offset (g * nb[0]) read
            // group 0 at grid g+p: mathematically wrong AND out of bounds
            // for g+p >= HW3 (observed as m-scale savpe failures on every
            // backend, while the loose test threshold masked it on n).
            ggml_tensor* s_g = ggml_cont(
                    gctx, ggml_view_2d(gctx, s_all, HW3, Q, s_all->nb[1] * c16,
                                       (size_t)g * s_all->nb[1]));
            ggml_tensor* agg = ggml_mul_mat(gctx, x_g, s_g);  // [d, Q]
            out = out ? ggml_concat(gctx, out, agg, 0) : agg;
        }
        // The official encoder L2-normalizes here; world_detect re-normalizes
        // its cls_pe input regardless, so the extra chain is skipped.
        return out;  // [512, Q]
    }

    // Debug readback nodes (AICORE_SAVPE_DUMP): the savpe branch outputs.
    ggml_tensor* dbg_savpe_x = nullptr;
    ggml_tensor* dbg_savpe_y = nullptr;
    ggml_tensor* dbg_savpe_fpn[3] = {nullptr, nullptr, nullptr};
    ggml_tensor* dbg_savpe_cv2[3] = {nullptr, nullptr, nullptr};
};

/* Drop the current run plan. Called on rebuild failure (the plan is unusable
 * then) and from free_session(). */
void clear_run_plan(Session* s) {
    if (s->gctx) ggml_free(s->gctx);  // cgraph + tensor structs live in gctx
    s->gctx = nullptr;
    s->input = nullptr;
    s->output = nullptr;
    s->text_input = nullptr;  // leaf lives in gctx; text_pending survives
    s->vp_input = nullptr;    // leaf lives in gctx; vp_pending survives
    s->savpe_out = nullptr;
    s->savpe_x = nullptr;
    s->savpe_y = nullptr;
    s->graph = nullptr;
    s->input_w = s->input_h = 0;
    s->anchors.clear();
    s->anchor_strides.clear();
    s->anchor_total = 0;
    s->dfl_proj.clear();
    s->output_f16.clear();
    s->output_proto_f16.clear();
}

/* Build the tensor chain + cgraph for a canvas into a FRESH gctx. THE single
 * graph builder: create_session() builds the initial plan through it and
 * session_ensure_canvas() rebuilds through it, so both paths always produce
 * the same op chain (CUDA/Vulkan f16 input flow, segment protos, GPU output
 * cast, anchor grid). On success the old gctx is released and the session
 * fields are swapped atomically; on failure the old plan is dropped (see
 * clear_run_plan) because the graph alloc below may have already reset the
 * sched state. */
bool build_run_plan(Session* s, int input_w, int input_h) {
    const ModelMeta& meta = s->model.meta;
    // Per-anchor output channels: detect=4*rm+nc, segment=+nm, pose=+nk,
    // obb=+ne.
    const int no = 4 * meta.reg_max + meta.nc + meta.nm + meta.nk + meta.ne;
    // Visual-prompt mode (YOLOE savpe): the head's cls_pe comes from the
    // savpe encoder instead of the text leaf. world_nc == prompt count Q.
    const bool visual = s->visual_mode() && s->model.has_savpe &&
                        s->model.detect_op_index >= 0;

    // ggml node budget: every op expands to a few nodes, and a
    // text-conditioned head adds one row view plus a 3-node max2 merge per
    // class and reduction site (measured 72..140 nodes per class on the
    // upstream yolov8-world family, so 256 bounds the shipped models). The
    // savpe branch adds ~60 base nodes plus ~15 per prompt and ~3 per
    // channel group (bounded by the 4096 slab + per-prompt allowance).
    const size_t node_budget =
            s->model.ops.size() * 12 + 512 +
            (s->model.has_text_input ? (size_t)s->world_nc * 256 : 0) +
            (visual ? 4096 + (size_t)s->opts.visual_count * 256 : 0);

    // Graph context: intermediate tensor structs (data lives in galloc/sched).
    const size_t g_size = node_budget * ggml_tensor_overhead() + (32u << 20);
    ggml_context* gctx = ggml_init({g_size, nullptr, /*no_alloc*/ true});
    if (!gctx) {
        YOLO_LOG_ERROR("graph ggml context allocation failed");
        return false;
    }

    const bool use_direct_conv = s->backend.is_cuda || s->backend.is_vulkan;
    GraphBuilder gb{gctx,         s->wctx,         s->model,
                    s->q8_direct, use_direct_conv, s->backend.is_cuda};
    std::vector<ggml_tensor*> values(s->model.ops.size(), nullptr);

    ggml_tensor* input =
            ggml_new_tensor_4d(gctx, GGML_TYPE_F32, input_w, input_h, 3, 1);
    ggml_set_input(input);  // allocated before compute nodes
    ggml_set_name(input, "image");

    // Open-vocabulary text leaf: external [512, nc] F32 embedding (CLIP /
    // MobileCLIP text encoder output). Recreated with every canvas; the
    // host copy (text_pending) survives and is re-uploaded on the next run.
    // Visual-prompt sessions take a [W3, H3, Q] binary mask leaf instead;
    // the savpe chain itself is built lazily on the first in_text() call
    // (it consumes FPN features that only exist once the op loop reaches
    // them), so no reprta residual applies on that path.
    ggml_tensor* text_input = nullptr;
    ggml_tensor* graph_text = nullptr;
    ggml_tensor* vp_input = nullptr;
    bool savpe_failed = false;
    if (s->model.has_text_input && !visual) {
        text_input = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 512, s->world_nc);
        ggml_set_input(text_input);
        ggml_set_name(text_input, "text");
        graph_text = text_input;
        // v4 YOLOE: the reprta residual rides on the detect op; apply it to
        // the text input before the dtype cast below.
        if (s->model.detect_op_index >= 0) {
            const std::string dp =
                    "op." + std::to_string(s->model.detect_op_index);
            if (s->model.tensors.count(dp + ".reprta_w12_w")) {
                graph_text = gb.reprta(dp, graph_text);
                if (!graph_text) {
                    ggml_free(gctx);
                    return false;
                }
            }
        }
    }
    if (visual) {
        const int stride0 = meta.strides.empty() ? 8 : (int)meta.strides[0];
        vp_input =
                ggml_new_tensor_4d(gctx, GGML_TYPE_F32, input_w / stride0,
                                   input_h / stride0, s->opts.visual_count, 1);
        ggml_set_input(vp_input);
        ggml_set_name(vp_input, "vp_masks");
    }

    // The input tensor is always F32; GPU f16 flows insert the cast node.
    // The flow is selected by the RESOLVED backend family (BackendCtx::
    // is_cuda / is_vulkan), never by compile-time macros: in a dynamic-
    // backend build the user may run a CUDA-enabled binary on CPU (or vice
    // versa) and the data flow must follow the actual device.
    ggml_tensor* graph_input = input;
    if (s->backend.is_cuda) {
        // CUDA f16 flow: the whole backbone runs F16 (igemm fast path).
        graph_input = ggml_cast(gctx, input, GGML_TYPE_F16);
        if (graph_text) graph_text = ggml_cast(gctx, graph_text, GGML_TYPE_F16);
    } else if (s->backend.is_vulkan && (meta.dtype == "f16" || s->q8_direct)) {
        graph_input = ggml_cast(gctx, input, GGML_TYPE_F16);
        if (graph_text) graph_text = ggml_cast(gctx, graph_text, GGML_TYPE_F16);
    }

    auto in0 = [&](const OpDef& op) {
        const int idx = op.inputs.empty() ? -1 : op.inputs[0];
        return idx < 0 ? graph_input : values[idx];
    };
    auto in_text = [&]() -> ggml_tensor* {
        if (visual && graph_text == nullptr && !savpe_failed) {
            std::vector<ggml_tensor*> fpn;
            for (int op_idx : s->model.savpe_fpn_ops) {
                fpn.push_back(values[op_idx]);
            }
            graph_text = gb.savpe(fpn, vp_input);
            if (!graph_text) savpe_failed = true;
        }
        return graph_text;
    };

    ggml_tensor* output_proto = nullptr;
    for (size_t i = 0; i < s->model.ops.size(); i++) {
        const OpDef& op = s->model.ops[i];
        const std::string prefix = "op." + std::to_string(i);
        ggml_tensor* out = nullptr;

        if (op.type == "max_sigmoid_attn") {
            out = gb.max_sigmoid_attn(op, prefix, values[op.inputs[0]],
                                      values[op.inputs[1]], in_text());
        } else if (op.type == "image_pooling_attn") {
            std::vector<ggml_tensor*> feats;
            for (int j : op.inputs) feats.push_back(values[j]);
            out = gb.image_pooling_attn(op, prefix, feats, in_text());
        } else if (op.type == "world_detect" || op.type == "world_segment") {
            std::vector<ggml_tensor*> feats;
            const bool has_masks = op.type == "world_segment";
            const size_t n =
                    has_masks ? op.inputs.size() - 1 : op.inputs.size();
            for (size_t j = 0; j < n; j++)
                feats.push_back(values[op.inputs[j]]);
            if (has_masks) output_proto = values[op.inputs.back()];
            out = gb.world_detect(op, prefix, feats, in_text(), meta.nc);
        } else if (op.type == "conv" || op.type == "dwconv") {
            out = gb.conv2d(op, prefix, in0(op));
        } else if (op.type == "maxpool") {
            const int k = (int)op.ip("k"), st = (int)op.ip("s"),
                      p = (int)op.ip("p");
            out = ggml_pool_2d(gctx, in0(op), GGML_OP_POOL_MAX, k, k, st, st,
                               (float)p, (float)p);
        } else if (op.type == "concat") {
            out = values[op.inputs[0]];
            for (size_t j = 1; j < op.inputs.size(); j++) {
                out = ggml_concat(gctx, out, values[op.inputs[j]], 2);
            }
        } else if (op.type == "upsample") {
            out = ggml_upscale(gctx, in0(op), (int)op.ip("sf"),
                               GGML_SCALE_MODE_NEAREST);
        } else if (op.type == "interpolate") {
            ggml_tensor* x = in0(op);
            const int64_t sf = op.ip("sf", 1);
            const uint32_t mode =
                    GGML_SCALE_MODE_BILINEAR |
                    (op.ip("align_corners") ? GGML_SCALE_FLAG_ALIGN_CORNERS
                                            : 0);
            // CUDA f16 flow: interpolate runs F32; cast around it. Selected
            // at runtime by the resolved backend family (see graph_input).
            if (s->backend.is_cuda && x->type == GGML_TYPE_F16) {
                x = ggml_cast(gctx, x, GGML_TYPE_F32);
                out = ggml_interpolate(gctx, x, x->ne[0] * sf, x->ne[1] * sf,
                                       x->ne[2], x->ne[3], mode);
                out = ggml_cast(gctx, out, GGML_TYPE_F16);
            } else {
                out = ggml_interpolate(gctx, x, x->ne[0] * sf, x->ne[1] * sf,
                                       x->ne[2], x->ne[3], mode);
            }
        } else if (op.type == "conv_transpose") {
            out = gb.conv_transpose(op, prefix, in0(op));
        } else if (op.type == "add") {
            out = ggml_add(gctx, values[op.inputs[0]], values[op.inputs[1]]);
        } else if (op.type == "slice") {
            // The channel slice is a contiguous sub-block view: the nb[0..2]
            // chain matches a dense tensor and ne[3]==1 skips the nb[3]
            // check, so ggml_is_contiguous(view) holds. Every consumer
            // (concat, conv) addresses it exactly like a dense tensor; the
            // cont copy would be a redundant kernel per C2f block (upstream
            // semantics).
            ggml_tensor* x = in0(op);
            const int64_t start = op.ip("start"), end = op.ip("end");
            out = ggml_view_4d(gctx, x, x->ne[0], x->ne[1], end - start,
                               x->ne[3], x->nb[1], x->nb[2], x->nb[3],
                               start * x->nb[2]);
        } else if (op.type == "psa_attention") {
            out = gb.psa_attention(op, prefix, in0(op));
        } else if (op.type == "detect" || op.type == "segment" ||
                   op.type == "pose" || op.type == "obb") {
            // Per-level conv output ne=[W,H,no,N] is already CHW-ordered in
            // memory (c outer, h middle, w inner); a plain reshape_2d matches
            // torch's x.view(B, no, H*W); concat along the anchor dim. No
            // permute needed. segment's last input is the proto map, kept as
            // the second graph output.
            const size_t n_feats =
                    op.inputs.size() - (op.type == "segment" ? 1 : 0);
            for (size_t j = 0; j < n_feats; j++) {
                ggml_tensor* t = values[op.inputs[j]];
                const int64_t HW = t->ne[0] * t->ne[1];
                ggml_tensor* r = ggml_reshape_2d(gctx, t, HW, no);
                out = out ? ggml_concat(gctx, out, r, 0) : r;
            }
            if (op.type == "segment") output_proto = values[op.inputs.back()];
        } else if (op.type == "semantic") {
            // Identity marker: the head convs already emitted
            // [W/8, H/8, nc, 1] logits; the task just declares the readback
            // layout for argmax.
            out = in0(op);
        } else if (op.type == "avgpool") {
            // Classify: AdaptiveAvgPool2d(1) — a global average pool whose
            // kernel equals the input extent (imgsz/32), so k0/k1 are
            // runtime values.
            ggml_tensor* x = in0(op);
            out = ggml_pool_2d(gctx, x, GGML_OP_POOL_AVG, x->ne[0], x->ne[1], 1,
                               1, 0, 0);
        } else if (op.type == "linear") {
            // y = x @ W^T + b with W stored [in, out]. A pooled [1,1,C,1]
            // vector (classify) and a [W,H,C,1] feature map (prompt-free
            // YOLOE vocabulary) are the same matmul; the map only needs
            // channel-first staging, exactly like the world_detect
            // contrastive branch. Keeping the head a matmul rather than a
            // 1x1 conv matters: ggml-cuda's IGEMM conv path rejects non-8-
            // aligned output channels and reserves per-plan buffers, neither
            // of which a vocabulary-sized classifier should pay.
            ggml_tensor* x = in0(op);
            ggml_tensor* wT = gb.w(prefix, "w");
            if (!wT) {
                YOLO_LOG_ERROR("linear '%s' has no weight tensor '.w'",
                               prefix.c_str());
                ggml_free(gctx);
                return false;
            }
            const int64_t C = x->ne[2], W = x->ne[0], H = x->ne[1];
            const int64_t HW = W * H;
            const bool spatial = HW > 1;
            ggml_tensor* feats = nullptr;
            if (spatial) {
                // A [W, H, C] map keeps channels slowest, so [C, HW] is not
                // a view of it: stage it channels-first once.
                ggml_tensor* cf =
                        ggml_cont(gctx, ggml_permute(gctx, x, 1, 2, 0, 3));
                GGML_ASSERT(ggml_is_contiguous(cf) && cf->ne[0] == C &&
                            cf->ne[1] == W && cf->ne[2] == H);
                feats = ggml_view_2d(gctx, cf, C, HW, C * ggml_element_size(cf),
                                     0);  // [C,HW]
            } else {
                feats = ggml_reshape_2d(gctx, x, C, 1);  // pooled [1,1,C]
            }
            out = ggml_mul_mat(gctx, wT, feats);  // [out, HW], always F32
            if (ggml_tensor* b = gb.w(prefix, "b")) {
                out = ggml_add(gctx, out,
                               ggml_reshape_2d(gctx, b, b->ne[0], 1));
            }
            if (spatial && out->type != x->type) {
                out = ggml_cast(gctx, out, x->type);  // head concat dtype
            }
            out = spatial ? ggml_reshape_3d(
                                    gctx,
                                    ggml_cont(gctx, ggml_permute(gctx, out, 1,
                                                                 0, 2, 3)),
                                    W, H, out->ne[0])
                          : ggml_reshape_1d(gctx, out, out->ne[0]);
        } else if (op.type == "classify") {
            // Identity marker on the [nc] logits; softmax/topk run in
            // postprocess.
            out = in0(op);
        } else if (op.type == "depth") {
            const float cal_a =
                    (float)(op.fparams.count("cal_a") ? op.fparams.at("cal_a")
                                                      : 1.0);
            const float cal_b =
                    (float)(op.fparams.count("cal_b") ? op.fparams.at("cal_b")
                                                      : 0.0);
            out = ggml_exp(gctx,
                           ggml_scale_bias(
                                   gctx, ggml_clamp(gctx, in0(op), -4.0f, 5.0f),
                                   cal_a, cal_b));
        } else {
            YOLO_LOG_ERROR("unknown op type '%s' at index %zu", op.type.c_str(),
                           i);
            ggml_free(gctx);
            return false;
        }
        if (!out) {
            YOLO_LOG_ERROR("op %zu ('%s') produced no output; missing weight?",
                           i, op.type.c_str());
            ggml_free(gctx);
            return false;
        }
        values[i] = out;
        if (s->opts.keep_all_ops && i < s->op_values.size()) {
            // Diagnostic view for the optrace bisection tool; gctx outlives
            // the plan so the pointers stay valid until the next rebuild.
            s->op_values[i] = out;
        }
    }
    ggml_tensor* output = values.back();
    if (!output || savpe_failed) {
        YOLO_LOG_ERROR("graph produced no output (last op returned null)");
        ggml_free(gctx);
        return false;
    }

    // GPU backends: cast F16 head outputs to F32 on-device.
    if (output->type == GGML_TYPE_F16 && s->backend.gpu) {
        output = ggml_cast(gctx, output, GGML_TYPE_F32);
    }
    if (output_proto && output_proto->type == GGML_TYPE_F16 && s->backend.gpu) {
        output_proto = ggml_cast(gctx, output_proto, GGML_TYPE_F32);
    }

    ggml_cgraph* graph =
            ggml_new_graph_custom(gctx, node_budget, /*grads*/ false);
    if (s->opts.keep_all_ops) {
        // Keep every op output alive for debugging.
        for (size_t i = 0; i < s->model.ops.size(); i++) {
            if (values[i]) {
                ggml_set_output(values[i]);
                ggml_build_forward_expand(graph, values[i]);
            }
        }
        // savpe debug nodes (visual mode): the dump hook reads them after
        // the run, so they must be kept alive too — without ggml_set_output
        // the gallocr reuses their buffers as soon as their last consumer
        // runs, and the dump sees garbage (observed on the savpe bisection).
        for (int l = 0; l < 3; l++) {
            ggml_tensor* dbg[2] = {gb.dbg_savpe_fpn[l], gb.dbg_savpe_cv2[l]};
            for (ggml_tensor* t : dbg) {
                if (t) {
                    ggml_set_output(t);
                    ggml_build_forward_expand(graph, t);
                }
            }
        }
        for (ggml_tensor* t : {gb.dbg_savpe_x, gb.dbg_savpe_y, graph_text}) {
            if (t) {
                ggml_set_output(t);
                ggml_build_forward_expand(graph, t);
            }
        }
        // Keep the C-API-visible outputs alive too (the post-cast nodes on
        // GPU flows) so the optrace tool can compare exactly what
        // session_read_proto/session output readback consumes. Duplicates
        // with values.back() are deduplicated by the graph builder.
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        if (output_proto) {
            ggml_set_output(output_proto);
            ggml_build_forward_expand(graph, output_proto);
        }
    } else {
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        if (output_proto) {
            ggml_set_output(output_proto);
            ggml_build_forward_expand(graph, output_proto);
        }
    }

    // Direct-conv backends cache transformed weights. Tensor addresses and
    // shapes may be reused after a model is freed, so neither is a sufficient
    // cache identity. Stamp every direct convolution with this session's
    // monotonic generation; the CUDA patch validates it before reusing a plan.
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor* node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_CONV_2D) {
            std::memcpy(&node->op_params[8], &s->plan_owner_id,
                        sizeof(s->plan_owner_id));
        }
    }

    // ---- commit the new plan ----
    clear_run_plan(s);  // frees the old gctx (if any) and resets the fields
    if (s->opts.keep_all_ops) {
        // Take the freshly built per-op outputs from the local values vector:
        // s->op_values still holds the PREVIOUS plan's pointers here, so a
        // plain resize/assign(nullptr) would silently discard everything the
        // store loop above just wrote (observed as a 0/N op comparison in
        // the optrace tool).
        s->op_values.assign(values.begin(), values.end());
    } else {
        s->op_values.clear();
    }
    s->gctx = gctx;
    s->input = input;
    s->output = output;
    s->output_proto = output_proto;
    s->text_input = text_input;
    s->vp_input = vp_input;
    s->savpe_out = visual ? graph_text : nullptr;  // vpe node for the dump hook
    s->savpe_x = visual ? gb.dbg_savpe_x : nullptr;
    s->savpe_y = visual ? gb.dbg_savpe_y : nullptr;
    for (int l = 0; l < 3; l++) {
        s->savpe_fpn_dbg[l] = visual ? gb.dbg_savpe_fpn[l] : nullptr;
        s->savpe_cv2_dbg[l] = visual ? gb.dbg_savpe_cv2[l] : nullptr;
    }
    s->graph = graph;
    s->input_w = input_w;
    s->input_h = input_h;
    s->output_f16.resize(
            output->type == GGML_TYPE_F16 ? (size_t)ggml_nelements(output) : 0);
    s->output_proto_f16.resize(output_proto && output_proto->type ==
                                                       GGML_TYPE_F16
                                       ? (size_t)ggml_nelements(output_proto)
                                       : 0);

    if (meta.task == "detect" || meta.task == "segment" ||
        meta.task == "pose" || meta.task == "obb") {
        // Box-anchored tasks share the anchor grid; depth/semantic/classify
        // decode dense or vector outputs and never touch it.
        // Postprocess constants (mirrors ultralytics make_anchors with 0.5
        // offset). Segment models share the detect anchor grid.
        for (int l = 0; l < meta.nl; l++) {
            const int stride = (int)meta.strides[l];
            const int fw = input_w / stride, fh = input_h / stride;
            for (int y = 0; y < fh; y++)
                for (int x = 0; x < fw; x++) {
                    s->anchors.push_back(x + 0.5f);
                    s->anchors.push_back(y + 0.5f);
                    s->anchor_strides.push_back((float)stride);
                }
        }
        s->anchor_total = (int)s->anchor_strides.size();
        for (int i = 0; i < meta.reg_max; i++) s->dfl_proj.push_back((float)i);
    }

    // Allocate the weight buffer on the primary backend and upload the host
    // copies BEFORE the scheduler allocates the graph: split_graph infers a
    // node's backend from its weight sources' buffers, so weights without a
    // buffer leave views unassigned (backend id -1) and the gallocr aborts on
    // GGML_ASSERT(buffer_id >= 0) instead of returning an error. Rebuilds
    // reuse these tensor structs and never touch the buffer again.
    if (!s->wbuf) {
        ggml_backend_buffer_type_t buft = backend_ctx_weight_buft(s->backend);
        s->wbuf = ggml_backend_alloc_ctx_tensors_from_buft(s->wctx, buft);
        if (!s->wbuf) {
            YOLO_LOG_ERROR("weight allocation failed");
            ggml_free(gctx);
            return false;
        }
        for (ggml_tensor* t = ggml_get_first_tensor(s->wctx); t;
             t = ggml_get_next_tensor(s->wctx, t)) {
            const HostTensor& ht = s->model.tensors.at(t->name);
            ggml_backend_tensor_set(t, ht.data.data(), 0, ht.data.size());
        }
    }

    // Allocate the graph: backend_ctx_graph_alloc resets the scheduler
    // internally (clearing ALL tensor→backend assignments), so the external
    // leaves must be passed HERE as pins — pre-assignments made above would
    // be lost. input/output/text(/proto) end up on the GPU so upload and
    // readback do not bounce through host memory.
    if (!backend_ctx_graph_alloc(s->backend, s->graph, s->input, s->output,
                                 s->text_input, s->output_proto, s->vp_input)) {
        clear_run_plan(
                s);  // sched/galloc state no longer matches the old graph
        return false;
    }

    YOLO_LOG_INFO("run plan ready: %dx%d, %d ops, anchors=%d", input_w, input_h,
                  (int)s->model.ops.size(), s->anchor_total);
    return true;
}

}  // namespace

// One-shot load-time weight preprocessing. The flow is selected by the
// RESOLVED backend family (BackendCtx::is_cuda / is_vulkan), never by
// compile-time macros — a CUDA-enabled build must still run the plain
// F32 flow when the lease resolved to CPU, and a CPU-only build that
// dlopen-ed a GPU backend module gets the GPU flow.
//   CPU    — semantic q8 models: expand to f16 once. The generic Q8 path
//            dynamically quantizes activations and fell below the external
//            semantic reference gate; other CPU tasks retain native Q8.
//   CUDA   — f32 models: cast weights to f16 once for the igemm flow;
//            q8 models expand to f16 once on the host. The CUDA Q8 conv path
//            also expands every weight into a per-conv f16 plan, so retaining
//            the compressed device tensor only adds device memory and a
//            failure-only code path without doing quantized arithmetic.
//   Vulkan — q8 models: expand Q8_0 weights to f16 on the host once
//            (vulkan has no Q8 conv shader).
// Idempotent: tensors already preprocessed (type != file_type) are skipped,
// so it can run again after an on-demand reload of released host weights.
static void prepare_host_weights(Session* s) {
    const bool expand_q8 = s->backend.is_cuda || s->backend.is_vulkan ||
                           s->model.meta.task == "semantic";
    if (expand_q8) {
        // Route quantized convs through the direct flow only when every
        // quantized tensor conforms; K 32-alignment is the hard constraint.
        for (const auto& [name, ht] : s->model.tensors) {
            if (!ggml_is_quantized(ht.type)) continue;
            if (ht.type != GGML_TYPE_Q8_0 || ht.ne[0] % 32 != 0) {
                s->q8_direct = false;
                break;
            }
            s->q8_direct = true;
        }
    }
    if (expand_q8 && s->q8_direct) {
        // Direct GPU backends and the precision-gated CPU semantic path consume
        // f16 weights. Expand Q8_0 once before upload instead of retaining a
        // compressed copy and repeatedly converting during convolution.
        for (auto& [name, ht] : s->model.tensors) {
            if (ht.type != GGML_TYPE_Q8_0) continue;
            const int64_t n = ht.ne[0] * ht.ne[1] * ht.ne[2] * ht.ne[3];
            std::vector<uint8_t> f16(n * sizeof(ggml_fp16_t));
            const block_q8_0* src =
                    reinterpret_cast<const block_q8_0*>(ht.data.data());
            ggml_fp16_t* dst = reinterpret_cast<ggml_fp16_t*>(f16.data());
            for (int64_t i = 0; i < n; ++i) {
                const block_q8_0* blk = src + i / QK8_0;
                dst[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(blk->d) *
                                           (float)blk->qs[i % QK8_0]);
            }
            ht.data = std::move(f16);
            ht.type = GGML_TYPE_F16;
        }
    }
    if (s->backend.is_cuda && s->model.meta.dtype == "f32") {
        // CUDA f32 models: cast weights to f16 once on the host for igemm.
        for (auto& [name, ht] : s->model.tensors) {
            if (ht.type != GGML_TYPE_F32) continue;
            if (name.size() > 2 && name.compare(name.size() - 2, 2, ".b") == 0)
                continue;  // biases stay F32
            const int64_t n = ht.ne[0] * ht.ne[1] * ht.ne[2] * ht.ne[3];
            std::vector<uint8_t> f16(n * sizeof(ggml_fp16_t));
            const float* src = reinterpret_cast<const float*>(ht.data.data());
            ggml_fp16_t* dst = reinterpret_cast<ggml_fp16_t*>(f16.data());
            for (int64_t i = 0; i < n; ++i) dst[i] = ggml_fp32_to_fp16(src[i]);
            ht.data = std::move(f16);
            ht.type = GGML_TYPE_F16;
        }
    }
}

Session* create_session(const std::string& gguf_path,
                        const std::string& device_request,
                        const SessionOptions& opts) {
    auto model = load_gguf(gguf_path);
    if (!model) return nullptr;

    yolo::set_log_level(opts.log_level);

    Session* s = new Session();
    s->plan_owner_id =
            g_next_plan_owner_id.fetch_add(1, std::memory_order_relaxed);
    s->model = std::move(*model);
    s->opts = opts;
    const int threads = opts.threads > 0 ? opts.threads : 1;
    s->backend = init_backend_ctx(threads, device_request);
    if (!s->backend.cpu) {
        free_session(s);
        return nullptr;
    }
    if (opts.profile_ops) {
        backend_enable_op_profile(s->backend);
    }

    // Determine canvas size: explicit opts override GGUF metadata.
    s->input_w = opts.input_w > 0 ? opts.input_w : s->model.meta.imgsz;
    s->input_h = opts.input_h > 0 ? opts.input_h : s->model.meta.imgsz;

    const ModelMeta& meta = s->model.meta;

    // Open-vocabulary session state: the class count is a runtime knob that
    // fixes the text-input shape and every nc-dependent tensor in the
    // graph, so it must be resolved before the first graph build. opts
    // world_nc wins; otherwise the GGUF yolo.nc default. A text-conditioned
    // head cannot run without a vocabulary: with no explicit class list the
    // caller means "use whatever the checkpoint shipped" (vocab_txt),
    // mirroring the Python default of predicting before any set_classes
    // call. Visual-prompt sessions (YOLOE savpe) instead fix nc to the
    // prompt count Q: the head's cls_pe comes from the image-derived
    // embeddings, so neither a class list nor a stored vocabulary applies.
    const bool visual = opts.visual_count > 0;
    if (visual && !s->model.has_savpe) {
        YOLO_LOG_ERROR(
                "visual prompts need a YOLOE GGUF converted with savpe "
                "weights (yolo.savpe=1); this checkpoint ships none");
        free_session(s);
        return nullptr;
    }
    const int world_nc = visual ? opts.visual_count
                                : (opts.world_nc > 0 ? opts.world_nc : meta.nc);
    s->world_nc = world_nc;
    if (s->model.has_text_input || visual) {
        s->model.meta.nc = world_nc;
        if (!visual && opts.world_nc <= 0 &&
            s->model.vocab_txt.size() != (size_t)world_nc * 512) {
            YOLO_LOG_ERROR(
                    "no stored vocabulary for %zu classes: convert a "
                    "checkpoint carrying txt_feats, or pass a class list",
                    (size_t)world_nc);
            free_session(s);
            return nullptr;
        }
        if (!visual && opts.world_nc <= 0) {
            s->text_pending = s->model.vocab_txt;
        }
    }

    // Weight context: tensor structs only; data goes to the backend buffer
    s->wctx = ggml_init(
            {(size_t)(s->model.tensors.size() * ggml_tensor_overhead() +
                      1024 * 1024),
             nullptr, /*no_alloc*/ true});
    if (!s->wctx) {
        YOLO_LOG_ERROR("ggml weight context allocation failed");
        free_session(s);
        return nullptr;
    }

    // One-shot load-time weight preprocessing (idempotent; also run by
    // session_ensure_host_weights after an on-demand reload).
    prepare_host_weights(s);

    // Build the initial run plan through THE single graph builder — the
    // same path session_ensure_canvas() uses for canvas rebuilds, so the
    // initial graph and every rebuild produce an identical op chain.
    if (!build_run_plan(s, s->input_w, s->input_h)) {
        free_session(s);
        return nullptr;
    }

    YOLO_LOG_INFO(
            "session ready: backend=%s, task=%s, %d ops, input=%dx%d, "
            "anchors=%d",
            s->backend.device_name.c_str(), meta.task.c_str(),
            (int)s->model.ops.size(), s->input_w, s->input_h, s->anchor_total);
    return s;
}

bool session_ensure_canvas(Session* s, int input_w, int input_h) {
    if (!s || input_w <= 0 || input_h <= 0) return false;
    if (s->gctx && s->input_w == input_w && s->input_h == input_h) return true;
    /* Canvas changed (letterbox keeps the source aspect ratio, so a
     * non-square frame lands on a non-square canvas): rebuild the graph
     * context through the single run-plan builder. The weight tensor
     * structs (wctx), the uploaded weight buffer (wbuf), the backend bundle
     * and the scheduler are all reused; only the graph context and the
     * derived anchor grid are rebuilt. On failure the old plan is already
     * dropped (build_run_plan clears it before committing) and the session
     * is left without a usable plan. */
    return build_run_plan(s, input_w, input_h);
}

bool session_release_host_weights(Session* s) {
    if (!s) return false;
    for (auto& [name, ht] : s->model.tensors) {
        ht.data.clear();
        ht.data.shrink_to_fit();
    }
    return true;
}

bool session_ensure_host_weights(Session* s) {
    if (!s || s->model.gguf_path.empty()) return false;
    bool any_missing = false;
    for (const auto& [name, ht] : s->model.tensors) {
        if (ht.data.empty()) {
            any_missing = true;
            break;
        }
    }
    if (!any_missing) return true;  // nothing to do

    // Re-read the raw tensor bytes straight from the GGUF file (metadata
    // only, no tensor mapping) using the offsets recorded at load time.
    gguf_init_params ip{};  // no_alloc: header only
    gguf_context* g = gguf_init_from_file(s->model.gguf_path.c_str(), ip);
    if (!g) {
        YOLO_LOG_ERROR("ensure_host_weights: failed to reopen %s",
                       s->model.gguf_path.c_str());
        return false;
    }
    FILE* f = std::fopen(s->model.gguf_path.c_str(), "rb");
    if (!f) {
        gguf_free(g);
        YOLO_LOG_ERROR("ensure_host_weights: cannot open %s",
                       s->model.gguf_path.c_str());
        return false;
    }
    bool ok = true;
    for (auto& [name, ht] : s->model.tensors) {
        if (!ht.data.empty()) continue;
        const int64_t tid = gguf_find_tensor(g, name.c_str());
        if (tid < 0 || std::fseek(f, (long)ht.file_offset, SEEK_SET) != 0) {
            ok = false;
            break;
        }
        const size_t nbytes = gguf_get_tensor_size(g, tid);
        std::vector<uint8_t> raw(nbytes);
        if (std::fread(raw.data(), 1, nbytes, f) != nbytes) {
            ok = false;
            break;
        }
        ht.data = std::move(raw);
        ht.type = ht.file_type;  // restore the original GGUF type
    }
    std::fclose(f);
    gguf_free(g);
    if (!ok) {
        YOLO_LOG_ERROR("ensure_host_weights: reload failed for %s",
                       s->model.gguf_path.c_str());
        return false;
    }
    // Re-run the idempotent backend preprocessing (Vulkan Q8->F16 / CUDA
    // F32->F16) on the restored original data.
    prepare_host_weights(s);
    return true;
}

bool session_run(Session* s, const float* chw_image) {
    const auto t0 = std::chrono::steady_clock::now();
    const size_t input_elements = (size_t)ggml_nelements(s->input);
    const size_t bytes = input_elements * sizeof(float);
    if (s->backend.gpu) {
        ggml_backend_tensor_set_async(s->backend.gpu, s->input, chw_image, 0,
                                      bytes);
    } else {
        ggml_backend_tensor_set(s->input, chw_image, 0, bytes);
    }

    // Graph allocation reuses the text leaf's backing storage between runs.
    // Keep the host embedding for the session and upload it before every
    // graph execution, just as the image input is uploaded on every frame
    // (also re-uploads after a canvas rebuild recreated the leaf). Visual
    // prompts mirror the same pattern with the rasterized mask buffer.
    if (s->text_input && !s->text_pending.empty()) {
        const size_t text_bytes = s->text_pending.size() * sizeof(float);
        if (s->backend.gpu) {
            ggml_backend_tensor_set_async(s->backend.gpu, s->text_input,
                                          s->text_pending.data(), 0,
                                          text_bytes);
        } else {
            ggml_backend_tensor_set(s->text_input, s->text_pending.data(), 0,
                                    text_bytes);
        }
    }
    if (s->vp_input && !s->vp_pending.empty()) {
        const size_t vp_bytes = s->vp_pending.size() * sizeof(float);
        if (s->backend.gpu) {
            ggml_backend_tensor_set_async(s->backend.gpu, s->vp_input,
                                          s->vp_pending.data(), 0, vp_bytes);
        } else {
            ggml_backend_tensor_set(s->vp_input, s->vp_pending.data(), 0,
                                    vp_bytes);
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int st = backend_ctx_graph_compute(s->backend, s->graph);
    if (s->opts.profile_gaps) {
        s->gap_comp_ms += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t1)
                                  .count();
        s->gap_up_ms +=
                std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (++s->gap_frames % 30 == 0) {
            std::fprintf(stderr,
                         "[gap-prof] upload=%.3fms compute=%.3fms "
                         "(frames=%d)\n",
                         s->gap_up_ms / s->gap_frames,
                         s->gap_comp_ms / s->gap_frames, s->gap_frames);
        }
    }
    if (st != GGML_STATUS_SUCCESS) {
        YOLO_LOG_ERROR("graph compute failed: %d", st);
        return false;
    }
    return true;
}

bool session_read_output(Session* s,
                         std::vector<float>& out,
                         int& no,
                         int& na) {
    if (s->model.meta.task != "detect" && s->model.meta.task != "segment" &&
        s->model.meta.task != "pose" && s->model.meta.task != "obb") {
        YOLO_LOG_ERROR(
                "session_read_output requires a detect, segment, pose or obb "
                "model, got %s",
                s->model.meta.task.c_str());
        return false;
    }
    // output layout: ne[0] = anchors, ne[1] = channels; element (a, c) at
    // a + c*na.
    na = (int)s->output->ne[0];
    no = (int)s->output->ne[1];
    out.resize((size_t)na * no);
    const auto tr0 = std::chrono::steady_clock::now();
    if (s->output->type == GGML_TYPE_F16) {
        ggml_backend_tensor_get(s->output, s->output_f16.data(), 0,
                                s->output_f16.size() * sizeof(ggml_fp16_t));
        const auto trc = std::chrono::steady_clock::now();
        ggml_fp16_to_fp32_row(s->output_f16.data(), out.data(), out.size());
        if (s->opts.profile_gaps) {
            s->gap_cast_ms += std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - trc)
                                      .count();
        }
    } else {
        ggml_backend_tensor_get(s->output, out.data(), 0,
                                out.size() * sizeof(float));
    }
    if (s->opts.profile_gaps) {
        s->gap_get_ms += std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - tr0)
                                 .count();
        if (++s->gap_rframes % 30 == 0) {
            std::fprintf(stderr,
                         "[gap-prof] tensor_get=%.3fms cast_out=%.3fms "
                         "(frames=%d)\n",
                         s->gap_get_ms / s->gap_rframes,
                         s->gap_cast_ms / s->gap_rframes, s->gap_rframes);
        }
    }
    return true;
}

bool session_read_proto(
        Session* s, std::vector<float>& out, int& nm, int& w, int& h) {
    if (!s->output_proto) {
        YOLO_LOG_ERROR("session_read_proto requires a segment model");
        return false;
    }
    w = (int)s->output_proto->ne[0];
    h = (int)s->output_proto->ne[1];
    nm = (int)s->output_proto->ne[2];
    out.resize((size_t)w * h * nm);
    if (s->output_proto->type == GGML_TYPE_F16) {
        ggml_backend_tensor_get(
                s->output_proto, s->output_proto_f16.data(), 0,
                s->output_proto_f16.size() * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(s->output_proto_f16.data(), out.data(),
                              out.size());
    } else {
        ggml_backend_tensor_get(s->output_proto, out.data(), 0,
                                out.size() * sizeof(float));
    }
    return true;
}

bool session_read_depth(Session* s,
                        std::vector<float>& out,
                        int& width,
                        int& height) {
    if (s->model.meta.task != "depth" || s->output->ne[2] != 1 ||
        s->output->ne[3] != 1) {
        YOLO_LOG_ERROR(
                "session_read_depth requires a single-channel depth model");
        return false;
    }
    width = (int)s->output->ne[0];
    height = (int)s->output->ne[1];
    out.resize((size_t)width * height);
    if (s->output->type == GGML_TYPE_F16) {
        ggml_backend_tensor_get(s->output, s->output_f16.data(), 0,
                                s->output_f16.size() * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(s->output_f16.data(), out.data(), out.size());
    } else {
        ggml_backend_tensor_get(s->output, out.data(), 0,
                                out.size() * sizeof(float));
    }
    return true;
}

bool session_set_text(Session* s, const float* text_embed) {
    if (!s || !s->text_input) {
        YOLO_LOG_ERROR("session_set_text requires a YOLO-World/YOLOE model");
        return false;
    }
    // Host input is row-major [nc, 512]; ggml stores [512, nc] column-major,
    // i.e. the exact same memory layout, so a plain copy suffices. Queue
    // the update and upload it at the start of the next graph run — this
    // keeps CPU and GPU backends on the same input-buffer lifecycle and
    // avoids writing through a tensor before its allocator attached a
    // buffer.
    s->text_pending.assign(text_embed,
                           text_embed + ggml_nelements(s->text_input));
    return true;
}

bool session_prepare_visual_masks(Session* s, const LetterboxInfo& info) {
    if (!s || !s->visual_mode() || !s->model.has_savpe) return false;
    const int q = s->opts.visual_count;
    if (s->opts.visual_boxes.size() != (size_t)q * 4) return false;
    const int stride0 =
            s->model.meta.strides.empty() ? 8 : (int)s->model.meta.strides[0];
    const int w3 = s->input_w / stride0;
    const int h3 = s->input_h / stride0;
    if (w3 <= 0 || h3 <= 0) return false;

    // Binary P3 masks [W3, H3, Q]: 1 inside the prompted box, 0 outside.
    // Boxes arrive in original-image pixels; map them through the letterbox
    // (scale + pad) and down to the P3 grid, matching the official
    // LoadVisualPrompt(scale_factor=1/8) rasterization.
    s->vp_pending.assign((size_t)w3 * h3 * q, 0.0f);
    float* masks = s->vp_pending.data();
    const bool dbg_log = aicore::debug::savpe_debug_enabled();
    if (dbg_log)
        std::fprintf(stderr, "[savpe-dbg] masks %dx%d q=%d boxes=%zu\n", w3, h3,
                     q, s->opts.visual_boxes.size());
    for (int i = 0; i < q; i++) {
        const float* box = &s->opts.visual_boxes[(size_t)i * 4];
        const float sx1 = (box[0] * info.scale + info.pad_w) / stride0;
        const float sy1 = (box[1] * info.scale + info.pad_h) / stride0;
        const float sx2 = (box[2] * info.scale + info.pad_w) / stride0;
        const float sy2 = (box[3] * info.scale + info.pad_h) / stride0;
        int x0 = std::max(0, std::min((int)std::lround(sx1), w3));
        int y0 = std::max(0, std::min((int)std::lround(sy1), h3));
        int x1 = std::max(x0, std::min((int)std::lround(sx2), w3));
        int y1 = std::max(y0, std::min((int)std::lround(sy2), h3));
        float* plane = masks + (size_t)i * w3 * h3;
        for (int y = y0; y < y1; ++y) {
            float* row = plane + (size_t)y * w3;
            for (int x = x0; x < x1; ++x) row[x] = 1.0f;
        }
        size_t nz = 0;
        if (dbg_log) {
            for (size_t k = 0; k < (size_t)w3 * h3; ++k)
                nz += masks[(size_t)i * w3 * h3 + k] > 0.f;
            std::fprintf(
                    stderr,
                    "[savpe-dbg] plane %d nonzero=%zu rect=[%d,%d)-[%d,%d)\n",
                    i, nz, x0, y0, x1, y1);
        }
    }
    if (const char* mdump = aicore::debug::savpe_mask_dump_path()) {
        FILE* f = std::fopen(mdump, "wb");
        if (f != nullptr) {
            std::fwrite(masks, sizeof(float), s->vp_pending.size(), f);
            std::fclose(f);
        }
    }
    return true;
}

bool session_read_semantic(
        Session* s, std::vector<float>& out, int& nc, int& w, int& h) {
    if (s->model.meta.task != "semantic") {
        YOLO_LOG_ERROR(
                "session_read_semantic requires a semantic model, got %s",
                s->model.meta.task.c_str());
        return false;
    }
    // logits layout: ne[0]=W, ne[1]=H, ne[2]=nc on the canvas/8 grid.
    w = (int)s->output->ne[0];
    h = (int)s->output->ne[1];
    nc = (int)s->output->ne[2];
    out.resize((size_t)w * h * nc);
    if (s->output->type == GGML_TYPE_F16) {
        ggml_backend_tensor_get(s->output, s->output_f16.data(), 0,
                                s->output_f16.size() * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(s->output_f16.data(), out.data(), out.size());
    } else {
        ggml_backend_tensor_get(s->output, out.data(), 0,
                                out.size() * sizeof(float));
    }
    return true;
}

bool session_read_logits(Session* s, std::vector<float>& out) {
    if (s->model.meta.task != "classify") {
        YOLO_LOG_ERROR("session_read_logits requires a classify model, got %s",
                       s->model.meta.task.c_str());
        return false;
    }
    const int64_t n = ggml_nelements(s->output);
    out.resize(n);
    if (s->output->type == GGML_TYPE_F16) {
        ggml_backend_tensor_get(s->output, s->output_f16.data(), 0,
                                s->output_f16.size() * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(s->output_f16.data(), out.data(), out.size());
    } else {
        ggml_backend_tensor_get(s->output, out.data(), 0,
                                out.size() * sizeof(float));
    }
    return true;
}

void free_session(Session* s) {
    if (!s) return;
    backend_print_op_profile();
    if (s->wbuf) ggml_backend_buffer_free(s->wbuf);
    free_backend_ctx(s->backend);
    if (s->wctx) ggml_free(s->wctx);
    if (s->gctx) ggml_free(s->gctx);
    delete s;
}

}  // namespace yolo
