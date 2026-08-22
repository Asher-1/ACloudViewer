// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/yolo/yolo_graph.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml.h"
#include "gguf.h"

namespace yolo {

namespace {

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
        ggml_tensor* out =
                ggml_conv_transpose_2d_p0(gctx, wT, x, (int)op.ip("s"));
        return add_bias_act(op, prefix, out);
    }

    // ggml 0.18.1 conv_2d_dw builds mul_mat(F32 kernel, F16 im2col) which
    // asserts on CPU (src1 must be F32 when src0 is F32). Cast F32 kernels
    // to F16 — matching the im2col F16 dtype — so both operands are F16 like
    // every other conv.
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
};

/* Drop the current run plan. Called on rebuild failure (the plan is unusable
 * then) and from free_session(). */
void clear_run_plan(Session* s) {
    if (s->gctx) ggml_free(s->gctx);  // cgraph + tensor structs live in gctx
    s->gctx = nullptr;
    s->input = nullptr;
    s->output = nullptr;
    s->graph = nullptr;
    s->input_w = s->input_h = 0;
    s->anchors.clear();
    s->anchor_strides.clear();
    s->anchor_total = 0;
    s->dfl_proj.clear();
    s->output_f16.clear();
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
    const int no = 4 * meta.reg_max + meta.nc + meta.nm;

    // Graph context: intermediate tensor structs (data lives in galloc/sched).
    const size_t g_size =
            (size_t)(s->model.ops.size() * 12 + 512) * ggml_tensor_overhead() +
            (32u << 20);
    ggml_context* gctx = ggml_init({g_size, nullptr, /*no_alloc*/ true});
    if (!gctx) {
        YOLO_LOG_ERROR("graph ggml context allocation failed");
        return false;
    }

    GraphBuilder gb{gctx, s->wctx, s->model, s->q8_direct,
                    s->backend.is_cuda || s->backend.is_vulkan};
    std::vector<ggml_tensor*> values(s->model.ops.size(), nullptr);

    ggml_tensor* input =
            ggml_new_tensor_4d(gctx, GGML_TYPE_F32, input_w, input_h, 3, 1);
    ggml_set_input(input);  // allocated before compute nodes
    ggml_set_name(input, "image");

    // The input tensor is always F32; GPU f16 flows insert the cast node.
    // The flow is selected by the RESOLVED backend family (BackendCtx::
    // is_cuda / is_vulkan), never by compile-time macros: in a dynamic-
    // backend build the user may run a CUDA-enabled binary on CPU (or vice
    // versa) and the data flow must follow the actual device.
    ggml_tensor* graph_input = input;
    if (s->backend.is_cuda) {
        // CUDA f16 flow: the whole backbone runs F16 (igemm fast path).
        graph_input = ggml_cast(gctx, input, GGML_TYPE_F16);
    } else if (s->backend.is_vulkan && (meta.dtype == "f16" || s->q8_direct)) {
        graph_input = ggml_cast(gctx, input, GGML_TYPE_F16);
    }

    auto in0 = [&](const OpDef& op) {
        const int idx = op.inputs.empty() ? -1 : op.inputs[0];
        return idx < 0 ? graph_input : values[idx];
    };

    ggml_tensor* output_proto = nullptr;
    for (size_t i = 0; i < s->model.ops.size(); i++) {
        const OpDef& op = s->model.ops[i];
        const std::string prefix = "op." + std::to_string(i);
        ggml_tensor* out = nullptr;

        if (op.type == "conv" || op.type == "dwconv") {
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
        } else if (op.type == "detect" || op.type == "segment") {
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
    }
    ggml_tensor* output = values.back();
    if (!output) {
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

    ggml_cgraph* graph = ggml_new_graph_custom(
            gctx, s->model.ops.size() * 12 + 512, /*grads*/ false);
    if (s->opts.keep_all_ops) {
        // Keep every op output alive for debugging.
        for (size_t i = 0; i < s->model.ops.size(); i++) {
            if (values[i]) {
                ggml_set_output(values[i]);
                ggml_build_forward_expand(graph, values[i]);
            }
        }
    } else {
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        if (output_proto) {
            ggml_set_output(output_proto);
            ggml_build_forward_expand(graph, output_proto);
        }
    }

    // ---- commit the new plan ----
    clear_run_plan(s);  // frees the old gctx (if any) and resets the fields
    s->gctx = gctx;
    s->input = input;
    s->output = output;
    s->output_proto = output_proto;
    s->graph = graph;
    s->input_w = input_w;
    s->input_h = input_h;
    s->output_f16.resize(
            output->type == GGML_TYPE_F16 ? (size_t)ggml_nelements(output) : 0);
    s->output_proto_f16.resize(output_proto && output_proto->type ==
                                                       GGML_TYPE_F16
                                       ? (size_t)ggml_nelements(output_proto)
                                       : 0);

    if (meta.task != "depth") {
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

    // Pin input/output(/proto) to the GPU backend when the scheduler is
    // active so upload and readback do not bounce through host memory.
    // backend_ctx_graph_alloc resets the sched first; the tensor-backend
    // assignments survive that reset and are consumed by alloc_graph.
    if (s->backend.sched && s->backend.gpu) {
        ggml_backend_sched_reset(s->backend.sched);
        ggml_backend_sched_set_tensor_backend(s->backend.sched, s->input,
                                              s->backend.gpu);
        ggml_backend_sched_set_tensor_backend(s->backend.sched, s->output,
                                              s->backend.gpu);
        if (s->output_proto) {
            ggml_backend_sched_set_tensor_backend(
                    s->backend.sched, s->output_proto, s->backend.gpu);
        }
    }
    if (!backend_ctx_graph_alloc(s->backend, s->graph, s->input, s->output)) {
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
//   CUDA   — f32 models: cast weights to f16 once for the igemm flow;
//            q8 models keep Q8_0 for the direct igemm path.
//   Vulkan — q8 models: expand Q8_0 weights to f16 on the host once
//            (vulkan has no Q8 conv shader).
// Idempotent: tensors already preprocessed (type != file_type) are skipped,
// so it can run again after an on-demand reload of released host weights.
static void prepare_host_weights(Session* s) {
    if (s->backend.is_cuda || s->backend.is_vulkan) {
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
    if (s->backend.is_vulkan && s->q8_direct) {
        // Vulkan: expand Q8_0 weights to f16 on the host once.
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
    if (s->model.meta.task != "detect" && s->model.meta.task != "segment") {
        YOLO_LOG_ERROR(
                "session_read_output requires a detect or segment model, "
                "got %s",
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
