// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// AnyCalib static ggml graph: DINOv2 ViT-L/14 backbone + light DPT decoder
// + convex tangent head, translated from the pinned upstream ONNX export
// (opset 18, node dump archived in TEST_ALIGNMENT_TODO.md batch AF). The
// graph ends at the convex-combination accumulator {46, 46, 49, 2}; the
// final output layout (322x322 tangent map) and the unit-ray normalization
// are exact f32 host postprocess (TEST_ALIGNMENT_TODO.md batch AF spec).
#include "tasks/anycalib/anycalib_graph.hpp"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/gguf_file_io.hpp"
#include "tasks/lightglue/backend.hpp"

namespace aicore::anycalib {
namespace {

constexpr int kImageSize = 322;
constexpr int kGrid = 23;  // 322 / 14 patch grid
constexpr int kMap = 46;   // reassembled 2x grid
constexpr int kTokens = 1 + kGrid * kGrid;
constexpr int kDim = 1024;
constexpr int kHeads = 16;
constexpr int kHeadDim = kDim / kHeads;
constexpr int kDepth = 24;
constexpr int kSub = 7;   // convex upsampling group
constexpr int kTaps = 9;  // 3x3 neighbourhood
constexpr float kLnEps = 1e-6f;
constexpr float kNormEps = 1.1920929e-07f;  // ONNX LessOrEqual bound
constexpr size_t kMaxGraphNodes = 4096;

// ---------------------------------------------------------------- GGUF file
class File {
public:
    ~File() { Close(); }
    bool Open(const std::string& path) {
        Close();
        std::string open_error;
        gguf = ggml_common::open_gguf_file(path, /*no_alloc=*/false, &ctx,
                                           "anycalib", &open_error);
        if (!gguf || !ctx) {
            error = open_error;
            Close();
            return false;
        }
        if (Str("general.architecture") != "anycalib") {
            if (error.empty()) error = "GGUF is not an anycalib model";
            Close();
            return false;
        }
        const int pretransposed =
                gguf_find_key(gguf, "anycalib.quant.matmul_pretransposed");
        pretransposed_matmul =
                pretransposed >= 0 &&
                gguf_get_kv_type(gguf, pretransposed) == GGUF_TYPE_UINT32 &&
                gguf_get_val_u32(gguf, pretransposed) == 1;
        return error.empty();
    }
    void Close() {
        device_tensors.clear();
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        if (gguf) {
            gguf_free(gguf);
            gguf = nullptr;
        }
    }
    ggml_tensor* Req(const std::string& n) {
        auto device = device_tensors.find(n);
        if (device != device_tensors.end()) return device->second;
        auto* t = ctx ? ggml_get_tensor(ctx, n.c_str()) : nullptr;
        if (!t && error.empty()) error = "anycalib GGUF missing tensor: " + n;
        return t;
    }
    int64_t N() const { return gguf_get_n_tensors(gguf); }
    // Q8_0 tier: matmul weights are stored pre-transposed (ne[0] == K). The
    // flag travels in the GGUF metadata because the dtype signal disappears
    // when non-CPU backends expand the weights to F32 at load.
    bool pretransposed_matmul = false;
    std::string Str(const char* key) {
        auto i = gguf_find_key(gguf, key);
        if (i < 0 || gguf_get_kv_type(gguf, i) != GGUF_TYPE_STRING) {
            if (error.empty())
                error = std::string("anycalib GGUF missing string key: ") + key;
            return {};
        }
        return gguf_get_val_str(gguf, i);
    }
    ggml_tensor* Tensor(int64_t i) const {
        return ctx ? ggml_get_tensor(ctx, gguf_get_tensor_name(gguf, i))
                   : nullptr;
    }
    gguf_context* gguf = nullptr;
    ggml_context* ctx = nullptr;
    std::unordered_map<std::string, ggml_tensor*> device_tensors;
    std::string error;
};

// ------------------------------------------------------------- graph helpers
ggml_tensor* LayerNorm(ggml_context* c,
                       ggml_tensor* x,
                       ggml_tensor* w,
                       ggml_tensor* b) {
    ggml_tensor* y = ggml_norm(c, x, kLnEps);
    y = ggml_mul(c, y, w);
    return ggml_add(c, y, b);
}

// Dense 3x3 convolution with symmetric padding on a {w, h, c, 1} map,
// following the proven yolo_graph im2col sequence (kernel ggml layout
// {kw, kh, cin, out} = numpy [out, cin, kh, kw] reversed).
ggml_tensor* Conv3x3(ggml_context* c,
                     ggml_tensor* x,
                     ggml_tensor* w4,
                     ggml_tensor* b,
                     int in_map) {
    ggml_tensor* im2 =
            ggml_im2col(c, w4, x, 1, 1, 1, 1, 1, 1, true, GGML_TYPE_F32);
    const int64_t P = im2->ne[1] * im2->ne[2] * im2->ne[3];
    ggml_tensor* w2 = ggml_reshape_2d(c, w4, w4->ne[0] * w4->ne[1] * w4->ne[2],
                                      w4->ne[3]);
    ggml_tensor* d =
            ggml_mul_mat(c, w2, ggml_reshape_2d(c, im2, im2->ne[0], P));
    d = ggml_add(c, d, b);
    // {OC, P} -> {OC, OW, OH, 1} -> permute -> {W, H, OC, 1} (yolo tail).
    d = ggml_reshape_4d(c, d, w4->ne[3], in_map, in_map, 1);
    return ggml_cont(c, ggml_permute(c, d, 2, 0, 1, 3));
}

// 1x1 convolution == per-position linear on a {w, h, c, 1} map.
ggml_tensor* Conv1x1(ggml_context* c,
                     ggml_tensor* x,
                     ggml_tensor* w4,
                     ggml_tensor* b) {
    // x is {W, H, C, 1} HWC; gather per-position channel vectors via a CHW
    // materialization, then restore HWC on the way out.
    const int64_t hw = x->ne[0] * x->ne[1];
    ggml_tensor* x2d = ggml_reshape_2d(
            c, ggml_cont(c, ggml_permute(c, x, 1, 2, 0, 3)), x->ne[2], hw);
    ggml_tensor* w2 = ggml_reshape_2d(c, w4, w4->ne[2], w4->ne[3]);
    ggml_tensor* d = ggml_add(c, ggml_mul_mat(c, w2, x2d), b);
    // {out, P} -> {out, W, H, 1} -> permute -> {W, H, out, 1} HWC.
    d = ggml_reshape_4d(c, d, w4->ne[3], x->ne[0], x->ne[1], 1);
    return ggml_cont(c, ggml_permute(c, d, 2, 0, 1, 3));
}

// Exact half-pixel bilinear 2x resample as a constant matmul: at this fixed
// scale every output pixel is a border-clamped 4-tap combination with
// constant 0.25/0.75 weights, identical to the traced ONNX Resize (linear,
// half_pixel, antialias=0). The matrix data is uploaded after graph
// allocation (the build context is no_alloc); the caller owns the vector.
ggml_tensor* Resize2xMatrix(ggml_context* c,
                            int in_n,
                            int out_n,
                            std::vector<float>& host) {
    const float s = float(in_n) / float(out_n);
    host.assign((size_t)out_n * out_n * in_n * in_n, 0.0f);
    auto taps = [&](int d) {
        const float src = (float(d) + 0.5f) * s - 0.5f;
        const int i0 = (int)std::floor(src);
        const float f = src - float(i0);
        const int a = std::clamp(i0, 0, in_n - 1);
        const int b = std::clamp(i0 + 1, 0, in_n - 1);
        return std::make_pair(std::make_pair(a, b),
                              std::make_pair(1.0f - f, f));
    };
    for (int ho = 0; ho < out_n; ++ho) {
        const auto th = taps(ho);
        for (int wo = 0; wo < out_n; ++wo) {
            const auto tw = taps(wo);
            const int row = ho * out_n + wo;
            float* dst = host.data() + (size_t)row * in_n * in_n;
            dst[(size_t)th.first.first * in_n + tw.first.first] +=
                    th.second.first * tw.second.first;
            dst[(size_t)th.first.first * in_n + tw.first.second] +=
                    th.second.first * tw.second.second;
            dst[(size_t)th.first.second * in_n + tw.first.first] +=
                    th.second.second * tw.second.first;
            dst[(size_t)th.first.second * in_n + tw.first.second] +=
                    th.second.second * tw.second.second;
        }
    }
    return ggml_new_tensor_2d(c, GGML_TYPE_F32, in_n * in_n, out_n * out_n);
}

}  // namespace

// ----------------------------------------------------------------- Session
struct Session::Impl {
    File file;
    lightglue::engine_backend be;
    ggml_backend_buffer_t weights = nullptr;
    ggml_context* device_ctx = nullptr;  // non-CPU weight context
    std::vector<float> r_host_;  // resize constant matrix upload staging

    void Release() {
        if (weights) {
            ggml_backend_buffer_free(weights);
            weights = nullptr;
        }
        if (device_ctx) {
            ggml_free(device_ctx);
            device_ctx = nullptr;
        }
        be.release();
        file.Close();
    }
};

Session::~Session() {
    Release();
    if (impl_) {
        delete impl_;
        impl_ = nullptr;
    }
}

void Session::Release() {
    loaded_ = false;
    if (impl_) impl_->Release();
}

bool Session::Load(const std::string& gguf_path,
                   const SessionOptions& options) {
    Release();
    error_.clear();
    if (!impl_) impl_ = new Impl();
    if (!impl_->file.Open(gguf_path)) {
        error_ = impl_->file.error;
        return false;
    }
    if (!impl_->be.init(options.device, options.num_threads)) {
        error_ = impl_->be.error;
        return false;
    }
    if (impl_->be.is_cpu()) {
        // Bind the GGUF memory block directly: f16 stays f16 (elementwise
        // operands cast at graph build) and Q8_0 matmul weights run
        // quantized on the CPU mul_mat.
        impl_->weights = ggml_backend_cpu_buffer_from_ptr(
                ggml_get_mem_buffer(impl_->file.ctx),
                ggml_get_mem_size(impl_->file.ctx));
        if (!impl_->weights) {
            error_ = "failed to bind anycalib weights";
            return false;
        }
        for (int64_t i = 0; i < impl_->file.N(); ++i)
            impl_->file.Tensor(i)->buffer = impl_->weights;
    } else {
        // Non-CPU routing (lightglue policy): the Vulkan backend has no
        // stable f16 path and quantized kernels are CPU-only, so every
        // weight is uploaded as F32. The graph then runs the plain f32
        // path; F16/Q8_0 storage rounding of the weights is preserved
        // (expansion changes storage, not values).
        ggml_init_params dp{};
        dp.mem_size = ggml_tensor_overhead() * (impl_->file.N() + 8);
        dp.no_alloc = true;
        impl_->device_ctx = ggml_init(dp);
        if (!impl_->device_ctx) {
            error_ = "failed to create the anycalib device weight context";
            return false;
        }
        for (int64_t i = 0; i < impl_->file.N(); ++i) {
            const ggml_tensor* src = impl_->file.Tensor(i);
            ggml_tensor* dst = ggml_new_tensor(impl_->device_ctx, GGML_TYPE_F32,
                                               GGML_MAX_DIMS, src->ne);
            ggml_set_name(dst, ggml_get_name(src));
            impl_->file.device_tensors.emplace(ggml_get_name(src), dst);
        }
        impl_->weights =
                ggml_backend_alloc_ctx_tensors(impl_->device_ctx, impl_->be.be);
        if (!impl_->weights) {
            error_ = "failed to allocate anycalib device weights";
            return false;
        }
        // WEIGHTS usage tag: sched offload routing requires it, otherwise
        // every op silently stays on the CPU secondary.
        ggml_backend_buffer_set_usage(impl_->weights,
                                      GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int64_t i = 0; i < impl_->file.N(); ++i) {
            const ggml_tensor* src = impl_->file.Tensor(i);
            ggml_tensor* dst =
                    impl_->file.device_tensors.at(ggml_get_name(src));
            if (src->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
            } else if (src->type == GGML_TYPE_F16) {
                std::vector<float> values((size_t)ggml_nelements(src));
                ggml_fp16_to_fp32_row(
                        static_cast<const ggml_fp16_t*>(src->data),
                        values.data(), static_cast<int64_t>(values.size()));
                ggml_backend_tensor_set(dst, values.data(), 0,
                                        values.size() * sizeof(float));
            } else if (src->type == GGML_TYPE_Q8_0) {
                // Q8_0 block layout: { ggml_fp16_t d (scale), int8_t qs[32] }.
                std::vector<float> values((size_t)ggml_nelements(src));
                const size_t block_bytes = ggml_type_size(GGML_TYPE_Q8_0);
                const int64_t block_elems = ggml_blck_size(GGML_TYPE_Q8_0);
                const size_t nblocks = values.size() / (size_t)block_elems;
                const uint8_t* raw = static_cast<const uint8_t*>(src->data);
                for (size_t b = 0; b < nblocks; ++b) {
                    const uint8_t* bp = raw + b * block_bytes;
                    ggml_fp16_t d_half;
                    std::memcpy(&d_half, bp, sizeof(d_half));
                    const float scale = ggml_fp16_to_fp32(d_half);
                    const int8_t* qs = reinterpret_cast<const int8_t*>(
                            bp + sizeof(d_half));
                    const size_t base = b * (size_t)block_elems;
                    for (int64_t j = 0; j < block_elems; ++j) {
                        values[base + (size_t)j] = scale * qs[j];
                    }
                }
                ggml_backend_tensor_set(dst, values.data(), 0,
                                        values.size() * sizeof(float));
            } else {
                error_ = "unsupported anycalib weight type for device upload";
                return false;
            }
        }
    }
    loaded_ = true;
    return true;
}

bool Session::Run(const float* image_rgb_chw, RunResult& out) {
    out = RunResult{};
    if (!loaded_ || !impl_) {
        error_ = "anycalib session is not ready";
        return false;
    }
    File& f = impl_->file;
    const auto t0 = std::chrono::steady_clock::now();

    ggml_init_params bp{};
    bp.mem_size = ggml_tensor_overhead() * kMaxGraphNodes + 256 * 1024;
    bp.no_alloc = true;
    ggml_context* bctx = ggml_init(bp);
    if (!bctx) {
        error_ = "anycalib graph context init failed";
        return false;
    }
    struct CtxGuard {
        ggml_context* c;
        ~CtxGuard() {
            if (c) ggml_free(c);
        }
    } guard{bctx};
    // The f16 GGUF stores biases/scales/pos-embed in f16 while activations
    // are f32: cast those elementwise operands at graph build (mul_mat would
    // accept f16, add/mul would not). Q8_0 matmul weights stay native so the
    // CPU mul_mat runs quantized (q8 x f32, f32 accumulation); the Q8_0 GGUF
    // stores only 2-D matmul weights in Q8_0 — every elementwise operand is
    // f16/f32 by construction.
    auto W = [&](const std::string& n) -> ggml_tensor* {
        ggml_tensor* t = f.Req(n);
        if (t == nullptr || t->type == GGML_TYPE_F32 ||
            t->type == GGML_TYPE_Q8_0) {
            return t;
        }
        return ggml_cast(bctx, t, GGML_TYPE_F32);
    };
    // Traced ONNX MatMul weights are [K, M] (unlike nn.Linear [M, K]); the
    // ggml mul_mat contract needs ne[0] == K, so f32/f16 fused matmul
    // weights transpose at graph build (free stride view + cont). The Q8_0
    // GGUF stores matmul weights pre-transposed (ne[0] == K, quantization
    // blocks along the contraction axis), so they pass through untouched.
    auto WT = [&](const std::string& n) -> ggml_tensor* {
        ggml_tensor* t = W(n);
        if (t != nullptr && f.pretransposed_matmul) {
            return t;
        }
        return ggml_cont(bctx, ggml_transpose(bctx, t));
    };

    // Constants uploaded after graph allocation (no_alloc build context).
    std::vector<std::pair<ggml_tensor*, const void*>> constant_sets;
    std::vector<ggml_tensor*> constant_zeros;

    ggml_tensor* input =
            ggml_new_tensor_3d(bctx, GGML_TYPE_F32, kImageSize, kImageSize, 3);
    ggml_set_name(input, "image");

    // In-graph ImageNet normalization (DINOv2 wrapper buffers).
    ggml_tensor* mean_cast = W("backbone.image_mean");
    ggml_tensor* std_cast = W("backbone.image_std");
    ggml_tensor* mean = ggml_reshape_3d(bctx, mean_cast, 1, 1, 3);
    ggml_tensor* stdt = ggml_reshape_3d(bctx, std_cast, 1, 1, 3);
    ggml_tensor* sub_normed = ggml_sub(bctx, input, mean);
    ggml_tensor* x = ggml_div(bctx, sub_normed, stdt);

    // Patch embed: conv 14x14 stride 14 -> {1024, 529} tokens; prepend the
    // cls token (pos-embed row 0) and add the resolved pos-embed.
    {
        ggml_tensor* w4 = W("backbone.patch_embed.proj.weight");
        ggml_tensor* im2 = ggml_im2col(bctx, w4, x, 14, 14, 0, 0, 1, 1, true,
                                       GGML_TYPE_F32);
        const int64_t P = im2->ne[1] * im2->ne[2] * im2->ne[3];
        ggml_tensor* w2 = ggml_reshape_2d(bctx, w4, 14 * 14 * 3, kDim);
        ggml_tensor* tok = ggml_mul_mat(
                bctx, w2, ggml_reshape_2d(bctx, im2, im2->ne[0], P));
        tok = ggml_add(bctx, tok, W("backbone.patch_embed.proj.bias"));
        ggml_tensor* pos = W("backbone.pos_embed");
        ggml_tensor* cls = ggml_view_2d(bctx, pos, kDim, 1, pos->nb[1], 0);
        x = ggml_add(bctx, ggml_concat(bctx, cls, tok, 1), pos);
    }

    // Transformer blocks.
    std::vector<ggml_tensor*> taps;
    for (int k = 0; k < kDepth; ++k) {
        const std::string p = "backbone.blocks." + std::to_string(k) + ".";
        ggml_tensor* h =
                LayerNorm(bctx, x, W(p + "norm1.weight"), W(p + "norm1.bias"));
        ggml_tensor* qkv =
                ggml_add(bctx, ggml_mul_mat(bctx, WT(p + "attn.qkv.weight"), h),
                         W(p + "attn.qkv.bias"));
        ggml_tensor* q = ggml_view_2d(bctx, qkv, kDim, kTokens, qkv->nb[1], 0);
        ggml_tensor* k_t = ggml_view_2d(bctx, qkv, kDim, kTokens, qkv->nb[1],
                                        kDim * sizeof(float));
        ggml_tensor* v = ggml_view_2d(bctx, qkv, kDim, kTokens, qkv->nb[1],
                                      2 * kDim * sizeof(float));
        q = ggml_cont(bctx, ggml_permute(bctx,
                                         ggml_reshape_4d(
                                                 bctx, ggml_cont(bctx, q),
                                                 kHeadDim, kHeads, kTokens, 1),
                                         0, 2, 1, 3));
        k_t = ggml_cont(
                bctx,
                ggml_permute(bctx,
                             ggml_reshape_4d(bctx, ggml_cont(bctx, k_t),
                                             kHeadDim, kHeads, kTokens, 1),
                             0, 2, 1, 3));
        v = ggml_cont(bctx, ggml_permute(bctx,
                                         ggml_reshape_4d(
                                                 bctx, ggml_cont(bctx, v),
                                                 kHeadDim, kHeads, kTokens, 1),
                                         0, 2, 1, 3));
        q = ggml_scale(bctx, q, 1.0f / std::sqrt(float(kHeadDim)));
        ggml_tensor* logits = ggml_soft_max(bctx, ggml_mul_mat(bctx, k_t, q));
        ggml_tensor* v_t = ggml_cont(bctx, ggml_permute(bctx, v, 1, 0, 2, 3));
        ggml_tensor* attn = ggml_reshape_2d(
                bctx,
                ggml_cont(bctx,
                          ggml_permute(bctx, ggml_mul_mat(bctx, v_t, logits), 0,
                                       2, 1, 3)),
                kDim, kTokens);
        attn = ggml_add(bctx,
                        ggml_mul_mat(bctx, WT(p + "attn.proj.weight"), attn),
                        W(p + "attn.proj.bias"));
        x = ggml_add(bctx, x, ggml_mul(bctx, attn, W(p + "ls1.gamma")));
        h = LayerNorm(bctx, x, W(p + "norm2.weight"), W(p + "norm2.bias"));
        h = ggml_add(bctx, ggml_mul_mat(bctx, WT(p + "mlp.fc1.weight"), h),
                     W(p + "mlp.fc1.bias"));
        h = ggml_gelu_erf(bctx, h);
        h = ggml_add(bctx, ggml_mul_mat(bctx, WT(p + "mlp.fc2.weight"), h),
                     W(p + "mlp.fc2.bias"));
        x = ggml_add(bctx, x, ggml_mul(bctx, h, W(p + "ls2.gamma")));
        // DPT taps = the traced Slices add_10/24/36/48 = 0-based blocks
        // {4, 11, 17, 23} (the add numbering is 2 per block from add_1).
        if (k == 4 || k == 11 || k == 17 || k == 23) taps.push_back(x);
    }
    if (taps.size() != 4) {
        error_ = "anycalib backbone tap collection failed";
        return false;
    }

    // Light DPT decoder. Tap tokens -> {23, 23, 1024} maps (cls dropped).
    auto tap_map = [&](ggml_tensor* t) {
        ggml_tensor* patches = ggml_view_2d(bctx, t, kDim, kGrid * kGrid,
                                            t->nb[1], kDim * sizeof(float));
        return ggml_reshape_3d(bctx, ggml_cont(bctx, patches), kGrid, kGrid,
                               kDim);
    };
    // Reassemble: 1x1 project, 2x bilinear (constant matmul), 3x3 conv.
    ggml_tensor* convs[4];
    const int kProjectOut[4] = {128, 256, 512, 1024};
    for (int i = 0; i < 4; ++i) {
        const std::string p =
                "decoder.reassemble_blocks.projects." + std::to_string(i) + ".";
        ggml_tensor* m = tap_map(taps[i]);
        ggml_tensor* w2 =
                ggml_reshape_2d(bctx, W(p + "weight"), kDim, kProjectOut[i]);
        ggml_tensor* pr = ggml_add(
                bctx,
                ggml_mul_mat(bctx, w2,
                             ggml_reshape_2d(bctx, m, kDim, kGrid * kGrid)),
                W(p + "bias"));
        pr = ggml_cont(bctx,
                       ggml_permute(bctx,
                                    ggml_reshape_4d(bctx, pr, kProjectOut[i],
                                                    kGrid, kGrid, 1),
                                    2, 0, 1, 3));
        ggml_tensor* r2 = Resize2xMatrix(bctx, kGrid, kMap, impl_->r_host_);
        constant_sets.emplace_back(r2, impl_->r_host_.data());
        pr = ggml_reshape_3d(
                bctx,
                ggml_mul_mat(bctx, r2,
                             ggml_reshape_2d(bctx, pr, kGrid * kGrid,
                                             kProjectOut[i])),
                kMap, kMap, kProjectOut[i]);
        const std::string cp = "decoder.convs." + std::to_string(i) + ".";
        convs[i] = Conv3x3(bctx, pr, W(cp + "weight"), W(cp + "bias"), kMap);
    }
    // Residual unit: conv-relu-conv-relu + skip.
    auto unit = [&](int block, const char* kind, ggml_tensor* in) {
        const std::string base = "decoder.fusion_blocks." +
                                 std::to_string(block) + "." + kind + ".conv.";
        ggml_tensor* h = ggml_relu(bctx, Conv3x3(bctx, in, W(base + "0.weight"),
                                                 W(base + "0.bias"), kMap));
        h = ggml_relu(bctx, Conv3x3(bctx, h, W(base + "2.weight"),
                                    W(base + "2.bias"), kMap));
        return ggml_add(bctx, h, in);
    };
    ggml_tensor* a49 = unit(0, "res_conv_unit2", convs[3]);
    ggml_tensor* a51 = ggml_add(bctx, unit(1, "res_conv_unit1", convs[2]), a49);
    ggml_tensor* a52 = unit(1, "res_conv_unit2", a51);
    ggml_tensor* a54 = ggml_add(bctx, unit(2, "res_conv_unit1", convs[1]), a52);
    ggml_tensor* a55 = unit(2, "res_conv_unit2", a54);
    ggml_tensor* a57 = ggml_add(bctx, unit(3, "res_conv_unit1", convs[0]), a55);
    ggml_tensor* a58 = unit(3, "res_conv_unit2", a57);
    ggml_tensor* relu14 =
            ggml_relu(bctx, Conv3x3(bctx, a58, W("decoder.project.0.weight"),
                                    W("decoder.project.0.bias"), kMap));

    // Head: tangent branch (2ch) and convex-weights branch (441ch).
    ggml_tensor* tan128 = ggml_relu(
            bctx, Conv3x3(bctx, relu14, W("head.tangent_head.0.weight"),
                          W("head.tangent_head.0.bias"), kMap));
    ggml_tensor* tan46 = Conv1x1(bctx, tan128, W("head.tangent_head.2.weight"),
                                 W("head.tangent_head.2.bias"));
    ggml_tensor* up128 = ggml_relu(
            bctx,
            Conv3x3(bctx, relu14, W("head.upsampling_weights_head.0.weight"),
                    W("head.upsampling_weights_head.0.bias"), kMap));
    ggml_tensor* w441 =
            Conv1x1(bctx, up128, W("head.upsampling_weights_head.2.weight"),
                    W("head.upsampling_weights_head.2.bias"));

    // Convex combination. Traced semantics (node 831): weights softmax over
    // the 9-tap axis; output pixel = sum_tap w[p_in, sub, tap] *
    // neighbor_tap[p_in]. Weights {46,46,441} -> {46,46,49,9} where the 49
    // axis is (h_sub*7 + w_sub); softmax over the 9 axis via permute.
    ggml_tensor* w4 =
            ggml_reshape_4d(bctx, w441, kMap, kMap, kSub * kSub, kTaps);
    ggml_tensor* w9 = ggml_soft_max(
            bctx, ggml_cont(bctx, ggml_permute(bctx, w4, 1, 2, 3, 0)));
    // Per-tap weight slices need {W, H, i49} as the leading contiguous dims,
    // but the softmax output leads with the tap axis (and W is not dim0
    // contiguous, so no view can slice it directly) — materialize the slice
    // layout once ({W, H, i49, tap}, 3.7 MB copy).
    ggml_tensor* w9p = ggml_cont(bctx, ggml_permute(bctx, w9, 3, 0, 1, 2));
    // Symmetric 1 px zero frame on W and H (traced Pad mode=constant).
    ggml_tensor* zero_col = ggml_new_tensor_3d(bctx, GGML_TYPE_F32, 1, kMap, 2);
    ggml_tensor* zero_row =
            ggml_new_tensor_3d(bctx, GGML_TYPE_F32, kMap + 2, 1, 2);
    constant_zeros.push_back(zero_col);
    constant_zeros.push_back(zero_row);
    ggml_tensor* padded = ggml_concat(bctx, zero_col, tan46, 0);
    padded = ggml_concat(bctx, padded, zero_col, 0);
    padded = ggml_concat(bctx, zero_row, padded, 1);
    padded = ggml_concat(bctx, padded, zero_row, 1);
    // Accumulate the nine taps with per-tap weight slices (avoids >4D
    // tensors): acc ne = {46, 46, 49, 2}.
    ggml_tensor* acc = nullptr;
    // Broadcast target {46,46,49,2}: repeat requires every source dim to be
    // 1 or equal to the target, so the c=2 neighbor plane is the result
    // carrier and the per-tap weight slice stays {46,46,49,1}.
    ggml_tensor* full_shape =
            ggml_new_tensor_4d(bctx, GGML_TYPE_F32, kMap, kMap, kSub * kSub, 2);
    for (int i = 0; i < kTaps; ++i) {
        const int dy = i / 3, dx = i % 3;
        ggml_tensor* wi =
                ggml_view_4d(bctx, w9p, kMap, kMap, kSub * kSub, 1, w9p->nb[1],
                             w9p->nb[2], w9p->nb[3], (size_t)i * w9p->nb[3]);
        ggml_tensor* nb = ggml_view_3d(
                bctx, padded, kMap, kMap, 2, padded->nb[1], padded->nb[2],
                (size_t)(dy * (kMap + 2) + dx) * sizeof(float));
        nb = ggml_reshape_4d(bctx, ggml_cont(bctx, nb), kMap, kMap, 1, 2);
        ggml_tensor* term =
                ggml_mul(bctx, ggml_repeat(bctx, nb, full_shape), wi);
        acc = (acc == nullptr) ? term : ggml_add(bctx, acc, term);
    }

    ggml_cgraph* gf = ggml_new_graph_custom(bctx, kMaxGraphNodes, false);
    ggml_build_forward_expand(gf, acc);
    if (!ggml_gallocr_alloc_graph(impl_->be.galloc, gf)) {
        error_ = "failed to allocate anycalib graph";
        return false;
    }
    // Constants are uploaded after graph allocation (no_alloc build
    // context); the buffer guard keeps non-galloc'd constants (e.g. under
    // future backend paths that skip constant allocation) safe.
    for (auto* t : constant_zeros)
        if (t->buffer != nullptr) ggml_set_zero(t);
    for (const auto& cs : constant_sets)
        if (cs.first->buffer != nullptr)
            ggml_backend_tensor_set(cs.first, cs.second, 0,
                                    ggml_nbytes(cs.first));
    ggml_backend_tensor_set(input, image_rgb_chw, 0, ggml_nbytes(input));
    const auto t1 = std::chrono::steady_clock::now();
    if (ggml_backend_graph_compute(impl_->be.be, gf) != GGML_STATUS_SUCCESS) {
        error_ = "anycalib graph compute failed";
        return false;
    }
    const auto t2 = std::chrono::steady_clock::now();
    std::vector<float> acc_host((size_t)kMap * kMap * kSub * kSub * 2);
    ggml_backend_tensor_get(acc, acc_host.data(), 0,
                            acc_host.size() * sizeof(float));
    ggml_backend_synchronize(impl_->be.be);
    const auto t3 = std::chrono::steady_clock::now();

    // Host postprocess: exact f32 replication of traced nodes 840-852.
    // acc ne = {W, H, (h_sub*7 + w_sub), 2}; flat index
    // w46 + 46*h46 + 2116*i49 + 103684*c (numpy C-order of the traced
    // sum_1 [2,7,7,46,46]); output pixel H = h46*7 + hs, W = w46*7 + ws
    // (traced permute [0,1,4,2,5,3]).
    const int out_n = kImageSize;
    const size_t kPlane = (size_t)kMap * kMap;
    const size_t kAccCh = kPlane * (size_t)(kSub * kSub);
    out.tangent.resize((size_t)out_n * out_n * 2);
    for (int h46 = 0; h46 < kMap; ++h46) {
        for (int w46 = 0; w46 < kMap; ++w46) {
            for (int hs = 0; hs < kSub; ++hs) {
                for (int ws = 0; ws < kSub; ++ws) {
                    const size_t src = ((size_t)hs * kSub + ws) * kPlane +
                                       (size_t)h46 * kMap + w46;
                    const int H = h46 * kSub + hs;
                    const int W = w46 * kSub + ws;
                    out.tangent[((size_t)H * out_n + W) * 2 + 0] =
                            acc_host[src];
                    out.tangent[((size_t)H * out_n + W) * 2 + 1] =
                            acc_host[src + kAccCh];
                }
            }
        }
    }
    out.rays.resize((size_t)out_n * out_n * 3);
    for (int i = 0; i < out_n * out_n; ++i) {
        const float tx = out.tangent[2 * (size_t)i + 0];
        const float ty = out.tangent[2 * (size_t)i + 1];
        const float norm = std::sqrt(tx * tx + ty * ty);
        float xy0;
        float xy1;
        if (norm <= kNormEps) {
            xy0 = tx;
            xy1 = ty;
        } else {
            const float s = std::sin(norm);
            xy0 = s * tx / norm;
            xy1 = s * ty / norm;
        }
        out.rays[3 * (size_t)i + 0] = xy0;
        out.rays[3 * (size_t)i + 1] = xy1;
        out.rays[3 * (size_t)i + 2] = std::cos(norm);
    }
    const auto t4 = std::chrono::steady_clock::now();
    out.preprocess_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
    out.inference_ms =
            std::chrono::duration<double, std::milli>(t2 - t1).count();
    out.postprocess_ms =
            std::chrono::duration<double, std::milli>(t3 - t2).count() +
            std::chrono::duration<double, std::milli>(t4 - t3).count();
    out.e2e_ms = std::chrono::duration<double, std::milli>(t4 - t0).count();
    return true;
}

}  // namespace aicore::anycalib
