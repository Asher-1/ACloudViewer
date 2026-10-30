// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "relateanything/full_model.hpp"

#include <json/json.h>

#include <QRegularExpression>
#include <QString>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "clip_tokenizer.hpp"
#include "corpus_type_map.hpp"
#include "ggml.h"
#include "gguf.h"
#include "relateanything/backend.hpp"
#include "relateanything/gguf_model.hpp"

namespace relateanything {
namespace {

constexpr char kImageMagic[] = "RAIMv1\0";
constexpr char kImageMagic2[] = "RAIMv2\0";
constexpr char kImageOutputMagic[] = "RAFOv1\0";
constexpr float kNegInf = -1.0e9f;

template <typename T>
void read_exact(std::ifstream &stream, T *dst, std::size_t count = 1) {
    const std::size_t bytes = sizeof(T) * count;
    stream.read(reinterpret_cast<char *>(dst),
                static_cast<std::streamsize>(bytes));
    if (!stream || static_cast<std::size_t>(stream.gcount()) != bytes) {
        throw std::runtime_error("truncated RAIM input");
    }
}

std::size_t product(std::size_t a, std::size_t b, const char *what) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::runtime_error(std::string(what) + " is too large");
    }
    return a * b;
}

struct TensorLoader {
    ModelOptions options;
    std::unique_ptr<Backend> backend;
    gguf_context *gguf = nullptr;
    ggml_context *weights_ctx = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
    std::map<std::string, std::string> original_to_emitted;
    std::filesystem::path path;

    explicit TensorLoader(const ModelOptions &opts,
                          const char *graph_kind = "relateanything_full_v1")
        : options(opts),
          backend(std::make_unique<Backend>(opts)),
          path(opts.model) {
        try {
            if (path.empty())
                throw std::runtime_error("full graph model path is required");
            gguf_init_params params{};
            params.no_alloc = true;
            params.ctx = &weights_ctx;
            gguf = gguf_init_from_file(path.string().c_str(), params);
            if (!gguf || !weights_ctx)
                throw std::runtime_error("cannot load full graph GGUF: " +
                                         path.string());
            const auto graph_id = gguf_find_key(gguf, "ra.graph_kind");
            if (graph_id < 0 ||
                gguf_get_kv_type(gguf, graph_id) != GGUF_TYPE_STRING ||
                std::string(gguf_get_val_str(gguf, graph_id)) != graph_kind)
                throw std::runtime_error(std::string("inference requires a ") +
                                         graph_kind + " model");

            const int64_t map_id = gguf_find_key(gguf, "ra.tensor_name_map");
            if (map_id >= 0 &&
                gguf_get_arr_type(gguf, map_id) == GGUF_TYPE_STRING) {
                const std::size_t n = gguf_get_arr_n(gguf, map_id);
                for (std::size_t i = 0; i + 1 < n; i += 2) {
                    original_to_emitted.emplace(
                            gguf_get_arr_str(gguf, map_id, i),
                            gguf_get_arr_str(gguf, map_id, i + 1));
                }
            }
            for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i) {
                const char *emitted = gguf_get_tensor_name(gguf, i);
                tensors.emplace(emitted, ggml_get_tensor(weights_ctx, emitted));
            }
            if (options.strict_f32_auto) {
                // Match upstream: precise CUDA arithmetic for all-F32 matrices;
                // F16/Q8 models retain the optimized default arithmetic.
                bool f32_matrices = true;
                for (const auto &[name, tensor] : tensors)
                    if (ggml_n_dims(tensor) >= 2 &&
                        tensor->type != GGML_TYPE_F32)
                        f32_matrices = false;
                options.strict_f32 = options.backend == "cuda" && f32_matrices;
            }
            weights_buffer = ggml_backend_alloc_ctx_tensors(
                    weights_ctx, backend->accelerator);
            if (!weights_buffer)
                throw std::runtime_error("cannot allocate full graph weights");
            ggml_backend_buffer_set_usage(weights_buffer,
                                          GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            std::ifstream stream(path, std::ios::binary);
            if (!stream)
                throw std::runtime_error("cannot read full graph GGUF: " +
                                         path.string());
            for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i) {
                const char *emitted = gguf_get_tensor_name(gguf, i);
                ggml_tensor *tensor = ggml_get_tensor(weights_ctx, emitted);
                const std::size_t offset = gguf_get_data_offset(gguf) +
                                           gguf_get_tensor_offset(gguf, i);
                const std::size_t bytes = ggml_nbytes(tensor);
                std::vector<std::uint8_t> data(bytes);
                stream.seekg(static_cast<std::streamoff>(offset),
                             std::ios::beg);
                read_exact(stream, data.data(), bytes);
                ggml_backend_tensor_set(tensor, data.data(), 0, bytes);
            }
        } catch (...) {
            if (weights_buffer) ggml_backend_buffer_free(weights_buffer);
            if (weights_ctx) ggml_free(weights_ctx);
            if (gguf) gguf_free(gguf);
            throw;
        }
    }

    ~TensorLoader() {
        if (weights_buffer) ggml_backend_buffer_free(weights_buffer);
        if (weights_ctx) ggml_free(weights_ctx);
        if (gguf) gguf_free(gguf);
    }

    ggml_tensor *tensor(const std::string &original) const {
        std::string emitted = original;
        if (const auto it = original_to_emitted.find(original);
            it != original_to_emitted.end())
            emitted = it->second;
        const auto it = tensors.find(emitted);
        if (it == tensors.end() || !it->second)
            throw std::runtime_error("missing full graph tensor: " + original);
        return it->second;
    }

    std::size_t meta_u32(const char *key, std::size_t fallback = 0) const {
        const int64_t id = gguf_find_key(gguf, key);
        if (id < 0) return fallback;
        switch (gguf_get_kv_type(gguf, id)) {
            case GGUF_TYPE_UINT32:
                return gguf_get_val_u32(gguf, id);
            case GGUF_TYPE_INT32:
                return static_cast<std::size_t>(gguf_get_val_i32(gguf, id));
            default:
                return fallback;
        }
    }

    float meta_float(const char *key, float fallback) const {
        const int64_t id = gguf_find_key(gguf, key);
        if (id < 0) return fallback;
        if (gguf_get_kv_type(gguf, id) == GGUF_TYPE_FLOAT32)
            return gguf_get_val_f32(gguf, id);
        return fallback;
    }

    std::vector<float> host_f32(ggml_tensor *tensor) const {
        if (!tensor) throw std::runtime_error("cannot read null tensor");
        if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
            throw std::runtime_error(
                    "host_f32 is only used for F32/F16 auxiliary tensors: " +
                    std::string(tensor->name));
        }
        const std::size_t n = ggml_nelements(tensor);
        std::vector<std::uint8_t> raw(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
        std::vector<float> out(n);
        if (tensor->type == GGML_TYPE_F32)
            std::memcpy(out.data(), raw.data(), n * sizeof(float));
        else {
            const auto *p = reinterpret_cast<const std::uint16_t *>(raw.data());
            for (std::size_t i = 0; i < n; ++i)
                out[i] = ggml_fp16_to_fp32(p[i]);
        }
        return out;
    }
};

struct GraphRun {
    ggml_context *ctx = nullptr;
    ggml_cgraph *graph = nullptr;
    std::vector<std::pair<ggml_tensor *, std::vector<std::uint8_t>>> inputs;
    std::size_t input_bytes = 0;
    GraphRun() = default;
    GraphRun(const GraphRun &) = delete;
    GraphRun &operator=(const GraphRun &) = delete;
    GraphRun(GraphRun &&other) noexcept
        : ctx(other.ctx), graph(other.graph), inputs(std::move(other.inputs)) {
        other.ctx = nullptr;
    }
    ~GraphRun() {
        if (ctx) ggml_free(ctx);
    }
};

GraphRun make_run() {
    GraphRun run;
    const std::size_t bytes = ggml_tensor_overhead() * 8192 +
                              ggml_graph_overhead_custom(8192, false);
    run.ctx = ggml_init({bytes, nullptr, true});
    if (!run.ctx) throw std::runtime_error("cannot create full graph context");
    return run;
}

ggml_tensor *input_tensor(GraphRun &run,
                          ggml_context *ctx,
                          enum ggml_type type,
                          int64_t ne0,
                          int64_t ne1,
                          int64_t ne2 = 1,
                          int64_t ne3 = 1,
                          const void *data = nullptr) {
    ggml_tensor *t = nullptr;
    if (ne3 != 1)
        t = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
    else if (ne2 != 1)
        t = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
    else if (ne1 != 1)
        t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    else
        t = ggml_new_tensor_1d(ctx, type, ne0);
    ggml_set_input(t);
    if (data) {
        const auto *ptr = static_cast<const std::uint8_t *>(data);
        run.inputs.emplace_back(
                t, std::vector<std::uint8_t>(ptr, ptr + ggml_nbytes(t)));
    }
    return t;
}

ggml_tensor *as_f32(ggml_context *ctx, ggml_tensor *tensor) {
    return tensor && tensor->type != GGML_TYPE_F32
                   ? ggml_cast(ctx, tensor, GGML_TYPE_F32)
                   : tensor;
}

ggml_tensor *matmul(ggml_context *ctx, ggml_tensor *weight, ggml_tensor *x) {
    auto *y = ggml_mul_mat(ctx, weight, x);
    // Keep the owner visible to shared backend dispatch. CPU uses upstream's
    // dot arithmetic (GGML_LLAMAFILE=OFF); CUDA retains upstream's F16
    // temporary/conversion instead of RMBG's F32-output optimization.
    ggml_set_name(y, "relateanything_matmul");
    // Accumulate in F32 for all storage types. Otherwise Vulkan may accumulate
    // attention and linear layers in F16, compounding rounding at every layer.
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    return y;
}

ggml_tensor *linear(ggml_context *ctx,
                    ggml_tensor *x,
                    ggml_tensor *weight,
                    ggml_tensor *bias) {
    ggml_tensor *y = matmul(ctx, weight, x);
    return bias ? ggml_add(ctx, y, as_f32(ctx, bias)) : y;
}

ggml_tensor *norm(ggml_context *ctx,
                  ggml_tensor *x,
                  ggml_tensor *weight,
                  ggml_tensor *bias,
                  float eps) {
    ggml_tensor *y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, as_f32(ctx, weight));
    return bias ? ggml_add(ctx, y, as_f32(ctx, bias)) : y;
}

ggml_tensor *activation(ggml_context *ctx, ggml_tensor *x, bool silu) {
    return silu ? ggml_silu(ctx, x) : ggml_gelu_erf(ctx, x);
}

ggml_tensor *concat_axis1(ggml_context *ctx,
                          const std::vector<ggml_tensor *> &xs) {
    ggml_tensor *out = as_f32(ctx, xs.front());
    for (std::size_t i = 1; i < xs.size(); ++i)
        out = ggml_concat(ctx, out, as_f32(ctx, xs[i]), 1);
    return out;
}

ggml_tensor *rope(ggml_context *ctx,
                  ggml_tensor *x,
                  ggml_tensor *cos,
                  ggml_tensor *sin,
                  int64_t dh) {
    // x is [dh, tokens, heads, batch], contiguous after the permutation.
    const int64_t half = dh / 2;
    ggml_tensor *a = ggml_view_4d(ctx, x, half, x->ne[1], x->ne[2], x->ne[3],
                                  x->nb[1], x->nb[2], x->nb[3], 0);
    ggml_tensor *b = ggml_view_4d(
            ctx, x, half, x->ne[1], x->ne[2], x->ne[3], x->nb[1], x->nb[2],
            x->nb[3], static_cast<std::size_t>(half) * sizeof(float));
    ggml_tensor *neg_b = ggml_neg(ctx, ggml_cont(ctx, b));
    ggml_tensor *rotated = ggml_concat(ctx, neg_b, a, 0);
    return ggml_add(ctx, ggml_mul(ctx, x, cos), ggml_mul(ctx, rotated, sin));
}

void build_mask(std::vector<float> &mask,
                std::size_t keys,
                std::size_t queries,
                const std::vector<std::uint8_t> &valid_keys) {
    mask.assign(keys * queries, 0.0f);
    for (std::size_t k = 0; k < keys; ++k)
        if (!valid_keys[k])
            for (std::size_t q = 0; q < queries; ++q)
                mask[k + q * keys] = kNegInf;
}

// torch.linspace(0, 7, 16) on CUDA: start + step*i below the midpoint and
// end - step*(n-1-i) (one FMA) above it, with step = 7.0f/15.0f.
std::array<float, 16> fourier_octaves() {
    std::array<float, 16> lin{};
    const float step = 7.0f / 15.0f;
    for (int i = 0; i < 16; ++i)
        lin[i] = i < 8 ? step * static_cast<float>(i)
                       : std::fma(-step, static_cast<float>(15 - i), 7.0f);
    return lin;
}

std::vector<float> fourier_point(float x, float y, int freqs = 16) {
    static const std::array<float, 16> frequencies = [] {
        auto f = fourier_octaves();
        for (auto &v : f) v = std::pow(2.0f, v);
        return f;
    }();
    if (freqs != 16)
        throw std::runtime_error("unsupported Fourier frequency count");
    std::vector<float> out;
    out.reserve(4 * freqs);
    const float pi = static_cast<float>(M_PI);
    for (int axis = 0; axis < 2; ++axis) {
        const float value = axis == 0 ? x : y;
        for (int i = 0; i < freqs; ++i)
            out.push_back(std::sin((value * frequencies[i]) * pi));
        for (int i = 0; i < freqs; ++i)
            out.push_back(std::cos((value * frequencies[i]) * pi));
    }
    return out;
}

std::array<float, 4> xyxy(const float *b) {
    return {std::clamp(b[0] - b[2] * 0.5f, 0.0f, 1.0f),
            std::clamp(b[1] - b[3] * 0.5f, 0.0f, 1.0f),
            std::clamp(b[0] + b[2] * 0.5f, 0.0f, 1.0f),
            std::clamp(b[1] + b[3] * 0.5f, 0.0f, 1.0f)};
}

// RelGeomEncoder.features before its logs: entries 2-6 and 10-11 are log(arg)
// (lin = 0), every other entry is lin (arg = 1, log(1) = 0). IEEE + - * / sqrt
// on the host match CUDA; log/tanh stay on the device under strict CUDA.
struct GeometryTerms {
    std::array<float, 19> arg, lin;
};

GeometryTerms geometry_terms(const float *s,
                             const float *o,
                             const float *region = nullptr) {
    const float eps = 1e-6f;
    const float dx = (o[0] - s[0]) / (s[2] + eps);
    const float dy = (o[1] - s[1]) / (s[3] + eps);
    const float sa_raw = s[2] * s[3], oa_raw = o[2] * o[3];
    const float sa = std::max(sa_raw, eps), oa = std::max(oa_raw, eps);
    const std::array<float, 4> sb = {s[0] - s[2] * 0.5f, s[1] - s[3] * 0.5f,
                                     s[0] + s[2] * 0.5f, s[1] + s[3] * 0.5f};
    const std::array<float, 4> ob = {o[0] - o[2] * 0.5f, o[1] - o[3] * 0.5f,
                                     o[0] + o[2] * 0.5f, o[1] + o[3] * 0.5f};
    const float ix =
            std::max(0.0f, std::min(sb[2], ob[2]) - std::max(sb[0], ob[0]));
    const float iy =
            std::max(0.0f, std::min(sb[3], ob[3]) - std::max(sb[1], ob[1]));
    const float inter = ix * iy, uni = sa + oa - inter + eps;
    const float dist = std::sqrt(std::max(
            (o[0] - s[0]) * (o[0] - s[0]) + (o[1] - s[1]) * (o[1] - s[1]),
            eps));
    GeometryTerms g{};
    g.lin = {dx,
             dy,
             0,
             0,
             0,
             0,
             0,
             inter / uni,
             inter / sa,
             inter / oa,
             0,
             0,
             (o[0] - s[0]) / dist,
             (o[1] - s[1]) / dist,
             o[1] - s[1],
             1.0f,
             1.0f,
             inter / uni,
             inter / std::min(sa, oa)};
    if (region) std::copy(region, region + 4, g.lin.begin() + 15);
    g.arg.fill(1.0f);
    g.arg[2] = (o[2] + eps) / (s[2] + eps);
    g.arg[3] = (o[3] + eps) / (s[3] + eps);
    g.arg[4] = (oa_raw + eps) / (sa_raw + eps);
    g.arg[5] = sa_raw + eps;
    g.arg[6] = oa_raw + eps;
    g.arg[10] = std::max(s[2] / (s[3] + eps), eps);
    g.arg[11] = std::max(o[2] / (o[3] + eps), eps);
    return g;
}

// 10 * tanh(feats / 10): CUDA turns the scalar division into feats * 0.1f.
std::array<float, 19> geometry(const GeometryTerms &g) {
    std::array<float, 19> out{};
    for (int i = 0; i < 19; ++i)
        out[i] = 10.0f *
                 std::tanh((std::log(g.arg[i]) + g.lin[i]) * (1.0f / 10.0f));
    return out;
}

std::vector<float> read_f32(ggml_tensor *tensor) {
    std::vector<float> out(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, out.data(), 0, out.size() * sizeof(float));
    return out;
}

ggml_tensor *packed_view(ggml_context *ctx,
                         ggml_tensor *weight,
                         int segment,
                         int64_t width) {
    return ggml_view_2d(
            ctx, weight, weight->ne[0], width, weight->nb[1],
            static_cast<std::size_t>(segment) * width * weight->nb[1]);
}

ggml_tensor *packed_bias(ggml_context *ctx,
                         ggml_tensor *bias,
                         int segment,
                         int64_t width) {
    return ggml_view_1d(ctx, bias, width,
                        static_cast<std::size_t>(segment) * width *
                                ggml_element_size(bias));
}

ggml_tensor *packed_attention(
        ggml_context *ctx,
        ggml_tensor *query,
        ggml_tensor *key,
        ggml_tensor *value,
        ggml_tensor *in_weight,
        ggml_tensor *in_bias,
        ggml_tensor *out_weight,
        ggml_tensor *out_bias,
        int64_t dim,
        int64_t nheads,
        ggml_tensor *mask = nullptr,
        bool flash = false,
        bool prescale = false,
        std::vector<std::pair<std::string, ggml_tensor *>> *trace = nullptr) {
    const int64_t dh = dim / nheads;
    ggml_tensor *q = nullptr, *k = nullptr, *v = nullptr;
    if (query == key && key == value) {
        auto *qkv = linear(ctx, query, in_weight, in_bias);
        q = ggml_view_2d(ctx, qkv, dim, query->ne[1], qkv->nb[1], 0);
        k = ggml_view_2d(ctx, qkv, dim, key->ne[1], qkv->nb[1],
                         dim * sizeof(float));
        v = ggml_view_2d(ctx, qkv, dim, value->ne[1], qkv->nb[1],
                         2 * dim * sizeof(float));
    } else {
        q = linear(ctx, query, packed_view(ctx, in_weight, 0, dim),
                   packed_bias(ctx, in_bias, 0, dim));
        auto *kv_weight =
                ggml_view_2d(ctx, in_weight, dim, 2 * dim, in_weight->nb[1],
                             dim * in_weight->nb[1]);
        auto *kv_bias = ggml_view_1d(ctx, in_bias, 2 * dim,
                                     dim * ggml_element_size(in_bias));
        auto *kv = linear(ctx, key, kv_weight, kv_bias);
        k = ggml_view_2d(ctx, kv, dim, key->ne[1], kv->nb[1], 0);
        v = ggml_view_2d(ctx, kv, dim, value->ne[1], kv->nb[1],
                         dim * sizeof(float));
    }
    const int64_t nq = query->ne[1], nk = key->ne[1];
    // nn.MultiheadAttention with need_weights scales q by float(sqrt(1/dh))
    // before bmm and applies an unscaled softmax.
    const float softmax_scale =
            prescale ? 1.0f : 1.0f / std::sqrt(static_cast<float>(dh));
    if (prescale)
        q = ggml_scale(
                ctx, q,
                static_cast<float>(std::sqrt(1.0 / static_cast<double>(dh))));
    if (trace) {
        trace->emplace_back("pool_q", q);
        trace->emplace_back("pool_kv", k->view_src ? k->view_src : k);
    }
    auto split = [&](ggml_tensor *z) {
        return ggml_permute(ctx,
                            ggml_reshape_4d(ctx, ggml_cont(ctx, z), dh, nheads,
                                            z->ne[1], 1),
                            0, 2, 1, 3);
    };
    q = split(q);
    k = split(k);
    v = split(v);
    if (flash && dh % 32 == 0) {
        if (mask) {
            const int pad = (64 - mask->ne[1] % 64) % 64;
            mask = ggml_cast(ctx,
                             pad ? ggml_pad(ctx, mask, 0, pad, 0, 0) : mask,
                             GGML_TYPE_F16);
        }
        auto *attn = ggml_flash_attn_ext(ctx, ggml_cont(ctx, q),
                                         ggml_cast(ctx, k, GGML_TYPE_F16),
                                         ggml_cast(ctx, v, GGML_TYPE_F16), mask,
                                         softmax_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        return linear(ctx, ggml_reshape_3d(ctx, attn, dim, nq, 1), out_weight,
                      out_bias);
    }
    ggml_tensor *scores = matmul(ctx, k, q);
    ggml_tensor *probs =
            ggml_soft_max_ext(ctx, scores, mask, softmax_scale, 0.0f);
    // MultiheadAttention's bmm(probs, v) is a non-transposed cuBLAS product
    // over v's [dh, hw] head slices; OUT_PROD with transposed probs issues it.
    ggml_tensor *attn =
            prescale ? ggml_out_prod(ctx, v, ggml_transpose(ctx, probs))
                     : matmul(ctx,
                              ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3)),
                              probs);
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    attn = ggml_reshape_3d(ctx, attn, dim, nq, 1);
    if (trace) {
        trace->emplace_back("pool_scores", scores);
        trace->emplace_back("pool_probs", probs);
        trace->emplace_back("pool_attn", attn);
    }
    (void)nk;
    return linear(ctx, attn, out_weight, out_bias);
}

std::vector<float> make_fourier_matrix(const std::vector<float> &points,
                                       std::size_t count,
                                       int freqs = 16) {
    std::vector<float> out(4 * freqs * count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto f = fourier_point(points[2 * i], points[2 * i + 1], freqs);
        for (std::size_t j = 0; j < f.size(); ++j) out[j + f.size() * i] = f[j];
    }
    return out;
}

}  // namespace

struct FullModel::Impl : TensorLoader {
    struct PackedAttention {
        ggml_context *ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        std::vector<ggml_tensor *> weights, biases;
        ~PackedAttention() {
            if (buffer) ggml_backend_buffer_free(buffer);
            if (ctx) ggml_free(ctx);
        }
    } packed_qkv;
    struct DeviceTensors {
        ggml_context *ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        std::map<std::string, ggml_tensor *> tensors;
        ~DeviceTensors() {
            if (buffer) ggml_backend_buffer_free(buffer);
            if (ctx) ggml_free(ctx);
        }
    } cached_positions, retained_features;
    // ConvNeXt convolution weights, dequantized once at load and stored in
    // the packed F32 layouts ggml_im2col / ggml_conv_2d_dw_direct expect.
    struct PackedConvs {
        ggml_context *ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        std::map<std::string, ggml_tensor *> kernels;
        ~PackedConvs() {
            if (buffer) ggml_backend_buffer_free(buffer);
            if (ctx) ggml_free(ctx);
        }
    } packed_convs;
    std::string model_family = "dinov3_vit";
    std::vector<std::size_t> stage_channels, stage_depths;
    std::size_t c = 0, d = 0, depth = 0, heads = 0, intermediate = 0,
                patch = 16, regs = 4, predicates = 0;
    float eps = 1e-5f;
    bool swiglu = false;
    std::vector<float> vocabulary;
    std::vector<float> tap_weights, compose_gates;
    std::vector<std::string> predicate_names;
    std::map<std::string, bool> corpus_types;
    float calibration_a = 1, calibration_b = 0;
    std::vector<float> rope_angle, rope_cos, rope_sin;
    int rope_height = 0, rope_width = 0;
    std::size_t backend_buffer_bytes = 0;
    ggml_backend_buffer_t readback_buffer = nullptr;
    std::size_t readback_capacity = 0;

    ~Impl() {
        if (readback_buffer) ggml_backend_buffer_free(readback_buffer);
    }

    void *readback(std::size_t bytes) {
        if (bytes == 0) return nullptr;
        if (readback_capacity < bytes) {
            if (readback_buffer) ggml_backend_buffer_free(readback_buffer);
            readback_buffer = nullptr;
            readback_capacity = 0;
            auto *device = ggml_backend_get_device(backend->accelerator);
            auto *buffer_type = ggml_backend_dev_host_buffer_type(device);
            if (buffer_type)
                readback_buffer =
                        ggml_backend_buft_alloc_buffer(buffer_type, bytes);
            if (readback_buffer)
                readback_capacity =
                        ggml_backend_buffer_get_size(readback_buffer);
        }
        return readback_buffer ? ggml_backend_buffer_get_base(readback_buffer)
                               : nullptr;
    }

    explicit Impl(const ModelOptions &opts) : TensorLoader(opts) {
        c = meta_u32("ra.hidden_size");
        d = meta_u32("ra.relation_dim");
        depth = meta_u32("ra.depth");
        heads = meta_u32("ra.num_heads");
        intermediate = meta_u32("ra.intermediate_size");
        patch = meta_u32("ra.patch_size", 16);
        regs = meta_u32("ra.num_register_tokens", 4);
        predicates = meta_u32("ra.predicate_count");
        eps = meta_float("ra.layer_norm_eps", 1e-5f);
        const int64_t ffn = gguf_find_key(gguf, "ra.ffn_type");
        swiglu = ffn >= 0 &&
                 std::string(gguf_get_val_str(gguf, ffn)) == "swiglu";
        const int64_t family_id = gguf_find_key(gguf, "ra.model_family");
        model_family = family_id >= 0 ? gguf_get_val_str(gguf, family_id)
                                      : "dinov3_vit";
        if (model_family == "dinov3_convnext") {
            // The hierarchical tower carries per-stage widths and depths
            // instead of the ViT scalar fields.
            const auto hs_id = gguf_find_key(gguf, "ra.hidden_sizes");
            const auto dp_id = gguf_find_key(gguf, "ra.depths");
            if (hs_id < 0 || dp_id < 0 ||
                gguf_get_arr_type(gguf, hs_id) != GGUF_TYPE_INT32 ||
                gguf_get_arr_type(gguf, dp_id) != GGUF_TYPE_INT32)
                throw std::runtime_error(
                        "convnext metadata is missing hidden_sizes/depths");
            const auto *hs = static_cast<const std::int32_t *>(
                    gguf_get_arr_data(gguf, hs_id));
            const auto *dp = static_cast<const std::int32_t *>(
                    gguf_get_arr_data(gguf, dp_id));
            for (std::size_t i = 0; i < gguf_get_arr_n(gguf, hs_id); ++i) {
                if (hs[i] <= 0)
                    throw std::runtime_error(
                            "convnext hidden size must be positive");
                stage_channels.push_back(static_cast<std::size_t>(hs[i]));
            }
            for (std::size_t i = 0; i < gguf_get_arr_n(gguf, dp_id); ++i) {
                if (dp[i] <= 0)
                    throw std::runtime_error("convnext depth must be positive");
                stage_depths.push_back(static_cast<std::size_t>(dp[i]));
            }
            if (stage_channels.size() != 4 || stage_depths.size() != 4)
                throw std::runtime_error(
                        "convnext metadata requires exactly 4 stages");
            if (stage_channels.back() != c)
                throw std::runtime_error(
                        "convnext hidden_size disagrees with the deepest "
                        "stage");
            depth = 0;
            for (auto v : stage_depths) depth += v;
            if (!c || !d || !predicates)
                throw std::runtime_error("incomplete full graph metadata");
            if (patch != 16)
                throw std::runtime_error(
                        "convnext fusion requires patch size 16");
        } else {
            if (!c || !d || !depth || !heads || !intermediate || !predicates)
                throw std::runtime_error("incomplete full graph metadata");
            if (c % heads)
                throw std::runtime_error("invalid full graph dimensions");
        }
        if (d % 8) throw std::runtime_error("invalid full graph dimensions");
        tap_weights = host_f32(t("ra.backbone.layer_weights"));
        compose_gates = host_f32(t("ra.compose_gate"));
        calibration_a = meta_float("ra.calibration_a", 1.0f);
        calibration_b = meta_float("ra.calibration_b", 0.0f);
        const auto names_id = gguf_find_key(gguf, "ra.predicate_names");
        if (names_id >= 0)
            for (std::size_t i = 0; i < gguf_get_arr_n(gguf, names_id); ++i)
                predicate_names.emplace_back(
                        gguf_get_arr_str(gguf, names_id, i));
        if (!options.predicate_texts.empty()) {
            predicate_names = options.predicate_texts;
            vocabulary = encode_predicates(options, predicate_names);
            predicates = predicate_names.size();
        } else if (!options.texts.empty()) {
            Json::Value names;
            std::ifstream file(options.texts);
            if (!file || !(file >> names) || !names.isArray() || names.empty())
                throw std::runtime_error("texts must be a nonempty JSON array");
            predicate_names.clear();
            for (const auto &name : names) {
                if (!name.isString())
                    throw std::runtime_error("text predicates must be strings");
                predicate_names.push_back(name.asString());
            }
            vocabulary = encode_predicates(options, predicate_names);
            predicates = predicate_names.size();
        } else if (!options.vocabulary.empty()) {
            const auto bytes = std::filesystem::file_size(options.vocabulary);
            const std::size_t text_dim = meta_u32("ra.text_dim");
            if (!text_dim || !bytes || bytes % (text_dim * sizeof(float)))
                throw std::runtime_error(
                        "vocabulary must be a nonempty row-major [V,text_dim] "
                        "F32 file");
            predicates = bytes / (text_dim * sizeof(float));
            predicate_names.clear();
            vocabulary.resize(bytes / sizeof(float));
            std::ifstream stream(options.vocabulary, std::ios::binary);
            read_exact(stream, vocabulary.data(), vocabulary.size());
            for (std::size_t row = 0; row < predicates; ++row) {
                double norm2 = 0;
                for (std::size_t j = 0; j < text_dim; ++j) {
                    const float v = vocabulary[row * text_dim + j];
                    if (!std::isfinite(v))
                        throw std::runtime_error(
                                "vocabulary contains a nonfinite value");
                    norm2 += v * v;
                }
                const float denom =
                        std::max(1e-12f, static_cast<float>(std::sqrt(norm2)));
                for (std::size_t j = 0; j < text_dim; ++j)
                    vocabulary[row * text_dim + j] /= denom;
            }
        }
        if (predicate_names.empty())
            for (std::size_t i = 0; i < predicates; ++i)
                predicate_names.push_back(std::to_string(i));
        if (!options.vocabulary_names.empty()) {
            Json::Value names;
            std::ifstream file(options.vocabulary_names);
            if (!file || !(file >> names) || !names.isArray())
                throw std::runtime_error(
                        "vocabulary names must be a JSON array");
            predicate_names.clear();
            for (const auto &name : names) {
                if (!name.isString())
                    throw std::runtime_error("predicate names must be strings");
                predicate_names.push_back(name.asString());
            }
        }
        if (options.decompose) {
            Json::Value types;
            if (options.predicate_type_map.empty()) {
                Json::CharReaderBuilder reader;
                std::istringstream file(ra_corpus_type_map);
                std::string error;
                if (!Json::parseFromStream(reader, file, &types, &error))
                    throw std::runtime_error(
                            "invalid embedded predicate types");
            } else {
                std::ifstream file(options.predicate_type_map);
                if (!file || !(file >> types))
                    throw std::runtime_error("cannot read predicate types");
            }
            if (!types.isObject())
                throw std::runtime_error(
                        "predicate type map must be a JSON object");
            for (const auto &name : types.getMemberNames()) {
                if (!types[name].isBool())
                    throw std::runtime_error(
                            "predicate types must be booleans");
                corpus_types.emplace(name, types[name].asBool());
            }
        }
        if (predicate_names.size() != predicates)
            throw std::runtime_error("predicate names and weights disagree");
        if (predicates > 131072)
            throw std::runtime_error(
                    "vocabulary exceeds the exact gather-index range");
        // Pack once at model load. Q/K/V have the same input and their output
        // rows are independent, so one matrix product preserves the graph's
        // math while avoiding two dispatches per DINOv3 block. Strict F32 still
        // selects precise F32 accumulation. ConvNeXt has no attention and
        // packs its convolution kernels instead.
        if (model_family != "dinov3_convnext") {
            packed_qkv.ctx = ggml_init(
                    {2 * depth * ggml_tensor_overhead(), nullptr, true});
            if (!packed_qkv.ctx)
                throw std::runtime_error(
                        "cannot create packed attention weights");
            for (std::size_t layer = 0; layer < depth; ++layer) {
                const auto prefix = "ra.backbone.model.model.layer." +
                                    std::to_string(layer) + ".attention.";
                auto *weight = t(prefix + "q_proj.weight");
                packed_qkv.weights.push_back(ggml_new_tensor_2d(
                        packed_qkv.ctx, weight->type, c, 3 * c));
                packed_qkv.biases.push_back(ggml_new_tensor_1d(
                        packed_qkv.ctx, GGML_TYPE_F32, 3 * c));
            }
            packed_qkv.buffer = ggml_backend_alloc_ctx_tensors(
                    packed_qkv.ctx, backend->accelerator);
            if (!packed_qkv.buffer)
                throw std::runtime_error(
                        "cannot allocate packed attention weights");
            ggml_backend_buffer_set_usage(packed_qkv.buffer,
                                          GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            for (std::size_t layer = 0; layer < depth; ++layer) {
                const auto prefix = "ra.backbone.model.model.layer." +
                                    std::to_string(layer) + ".attention.";
                const std::array<std::string, 3> projections = {
                        "q_proj", "k_proj", "v_proj"};
                const auto block_bytes =
                        ggml_nbytes(t(prefix + "q_proj.weight"));
                std::vector<std::uint8_t> bytes(3 * block_bytes);
                std::vector<float> bias(3 * c, 0);
                for (std::size_t part = 0; part < 3; ++part) {
                    auto *source = t(prefix + projections[part] + ".weight");
                    if (source->type != packed_qkv.weights[layer]->type ||
                        ggml_nbytes(source) != block_bytes)
                        throw std::runtime_error(
                                "incompatible Q/K/V projection layout");
                    ggml_backend_tensor_get(source,
                                            bytes.data() + part * block_bytes,
                                            0, block_bytes);
                    if (part != 1) {
                        auto values = host_f32(
                                t(prefix + projections[part] + ".bias"));
                        std::copy(values.begin(), values.end(),
                                  bias.begin() + part * c);
                    }
                }
                ggml_backend_tensor_set(packed_qkv.weights[layer], bytes.data(),
                                        0, bytes.size());
                ggml_backend_tensor_set(packed_qkv.biases[layer], bias.data(),
                                        0, bias.size() * sizeof(float));
            }
        } else {
            build_convnext_convs();
        }
        if (options.cache_positions) {
            const int grid = 448 / patch;
            const std::array<std::string, 3> names = {
                    "spatial_pool.scene_pe", "rel_transformer.scene_pe",
                    "rel_interaction.scene_pe"};
            auto run = make_run();
            run.graph = ggml_new_graph_custom(run.ctx, 8192, false);
            std::vector<ggml_tensor *> values;
            for (const auto &name : names) {
                auto *value = scene_pe(run, name, grid, grid);
                ggml_set_output(value);
                ggml_build_forward_expand(run.graph, value);
                values.push_back(value);
            }
            cached_positions.ctx =
                    ggml_init({3 * ggml_tensor_overhead(), nullptr, true});
            if (!cached_positions.ctx)
                throw std::runtime_error(
                        "cannot create cached scene positions");
            for (std::size_t i = 0; i < names.size(); ++i)
                cached_positions.tensors.emplace(
                        names[i],
                        ggml_new_tensor_2d(cached_positions.ctx, GGML_TYPE_F32,
                                           values[i]->ne[0], values[i]->ne[1]));
            cached_positions.buffer = ggml_backend_alloc_ctx_tensors(
                    cached_positions.ctx, backend->accelerator);
            if (!cached_positions.buffer)
                throw std::runtime_error(
                        "cannot allocate cached scene positions");
            ggml_backend_buffer_set_usage(cached_positions.buffer,
                                          GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            // Use the inference backend to preserve its rounding and avoid host
            // recomputation.
            execute(run);
            for (std::size_t i = 0; i < names.size(); ++i)
                ggml_backend_tensor_copy(values[i],
                                         cached_positions.tensors.at(names[i]));
            ggml_backend_sched_reset(backend->scheduler);
        }
        if (options.retain_features) {
            retained_features.ctx =
                    ggml_init({ggml_tensor_overhead(), nullptr, true});
            if (!retained_features.ctx)
                throw std::runtime_error("cannot create retained patch map");
            retained_features.tensors.emplace(
                    "fused_patch_map",
                    ggml_new_tensor_2d(retained_features.ctx, GGML_TYPE_F32, c,
                                       (448 / patch) * (448 / patch)));
            retained_features.buffer = ggml_backend_alloc_ctx_tensors(
                    retained_features.ctx, backend->accelerator);
            if (!retained_features.buffer)
                throw std::runtime_error("cannot allocate retained patch map");
        }
    }

    ggml_tensor *t(const std::string &n) { return tensor(n); }

    ggml_tensor *affine(ggml_context *ctx,
                        ggml_tensor *x,
                        const std::string &p,
                        bool bias = true) {
        return linear(ctx, x, t("ra." + p + ".weight"),
                      bias ? t("ra." + p + ".bias") : nullptr);
    }

    ggml_tensor *ln(ggml_context *ctx, ggml_tensor *x, const std::string &p) {
        return norm(ctx, x, t("ra." + p + ".weight"), t("ra." + p + ".bias"),
                    1e-5f);
    }

    ggml_tensor *mha(ggml_context *ctx,
                     ggml_tensor *q,
                     ggml_tensor *kv,
                     const std::string &p,
                     ggml_tensor *mask = nullptr,
                     bool need_weights = false,
                     std::vector<std::pair<std::string, ggml_tensor *>> *trace =
                             nullptr) {
        return packed_attention(
                ctx, q, kv, kv, t("ra." + p + ".in_proj_weight"),
                t("ra." + p + ".in_proj_bias"),
                t("ra." + p + ".out_proj.weight"),
                t("ra." + p + ".out_proj.bias"), q->ne[0], 8, mask,
                options.flash_attention && !strict_cuda() &&
                        options.backend != "cpu",
                need_weights && strict_cuda(), trace);
    }

    // Strict CUDA: nn.MultiheadAttention's fused inference path, which PyTorch
    // takes for self-attention with a boolean key-padding mask
    // (_native_multi_head_attention): bias-free QKV GEMM, (qkv + bias) scaled
    // by dh^-1/2 per head, bmm(q, k^T), masked softmax, bmm(p, v) and the
    // out_proj addmm. mask is the [keys, 1] 0/-inf row; adding -inf before
    // the softmax equals its masked softmax since exp(-inf) adds exactly 0.
    ggml_tensor *native_self_attention(ggml_context *ctx,
                                       ggml_tensor *x,
                                       const std::string &p,
                                       ggml_tensor *mask) {
        const int64_t dim = x->ne[0], tokens = x->ne[1], nheads = 8,
                      dh = dim / nheads;
        auto *qkv = matmul(ctx, t("ra." + p + ".in_proj_weight"), x);
        // The bias joins a reshaped view, so it stays out of the GEMM epilogue.
        qkv = ggml_add(ctx, ggml_reshape_3d(ctx, qkv, dh, 3 * nheads, tokens),
                       ggml_reshape_3d(
                               ctx, as_f32(ctx, t("ra." + p + ".in_proj_bias")),
                               dh, 3 * nheads, 1));
        const auto head = [&](int segment) {
            return ggml_cont(
                    ctx,
                    ggml_permute(ctx,
                                 ggml_view_3d(ctx, qkv, dh, nheads, tokens,
                                              qkv->nb[1], qkv->nb[2],
                                              segment * nheads * qkv->nb[1]),
                                 0, 2, 1, 3));
        };
        auto *q = ggml_scale(ctx, head(0),
                             1.0f / std::sqrt(static_cast<float>(dh)));
        ggml_tensor *scores = matmul(ctx, head(1), q);
        if (mask) scores = ggml_add(ctx, scores, mask);
        auto *attn = ggml_out_prod(
                ctx, head(2), ggml_transpose(ctx, ggml_soft_max(ctx, scores)));
        attn = ggml_reshape_2d(
                ctx, ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3)), dim,
                tokens);
        return linear(ctx, attn, t("ra." + p + ".out_proj.weight"),
                      t("ra." + p + ".out_proj.bias"));
    }

    // Upstream strict CUDA cross-attention uses explicit F32 score, softmax
    // and value-product ops. The additive mask is broadcast over queries
    // and heads; packed k/v projections remain contiguous.
    ggml_tensor *strict_cross_attention(ggml_context *ctx,
                                        ggml_tensor *x,
                                        ggml_tensor *memory,
                                        const std::string &p,
                                        ggml_tensor *mask) {
        const int64_t dim = x->ne[0], nheads = 8, dh = dim / nheads;
        auto *w = t("ra." + p + ".in_proj_weight");
        auto *b = t("ra." + p + ".in_proj_bias");
        auto *q = linear(ctx, x, packed_view(ctx, w, 0, dim),
                         packed_bias(ctx, b, 0, dim));
        auto *kv = linear(
                ctx, memory,
                ggml_view_2d(ctx, w, dim, 2 * dim, w->nb[1], dim * w->nb[1]),
                ggml_view_1d(ctx, b, 2 * dim, dim * ggml_element_size(b)));
        auto *k = ggml_cont(
                ctx, ggml_view_2d(ctx, kv, dim, memory->ne[1], kv->nb[1], 0));
        auto *v = ggml_cont(ctx, ggml_view_2d(ctx, kv, dim, memory->ne[1],
                                              kv->nb[1], dim * sizeof(float)));
        const auto heads = [&](ggml_tensor *z) {
            return ggml_cont(
                    ctx,
                    ggml_permute(ctx,
                                 ggml_reshape_3d(ctx, z, dh, nheads, z->ne[1]),
                                 0, 2, 1, 3));
        };
        auto *scores = ggml_scale(ctx, matmul(ctx, heads(k), heads(q)),
                                  1.0f / std::sqrt(static_cast<float>(dh)));
        if (mask) scores = ggml_add(ctx, scores, mask);
        auto *attn = ggml_out_prod(
                ctx, heads(v), ggml_transpose(ctx, ggml_soft_max(ctx, scores)));
        attn = ggml_reshape_2d(
                ctx, ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3)), dim,
                x->ne[1]);
        return linear(ctx, attn, t("ra." + p + ".out_proj.weight"),
                      t("ra." + p + ".out_proj.bias"));
    }

    ggml_tensor *encoder(ggml_context *ctx,
                         ggml_tensor *x,
                         const std::string &p,
                         ggml_tensor *mask) {
        auto *y = ln(ctx, x, p + ".norm1");
        if (strict_cuda()) {
            // TransformerEncoderLayer's fused inference path: its FFN is
            // _addmm_activation(use_gelu=True), the cublasLt GELU_BIAS
            // epilogue.
            x = ggml_add(ctx, x,
                         native_self_attention(ctx, y, p + ".self_attn", mask));
            y = affine(ctx,
                       ggml_gelu(ctx, affine(ctx, ln(ctx, x, p + ".norm2"),
                                             p + ".linear1")),
                       p + ".linear2");
            return ggml_add(ctx, x, y);
        }
        x = ggml_add(ctx, x, mha(ctx, y, y, p + ".self_attn", mask));
        y = affine(ctx,
                   ggml_gelu_erf(ctx, affine(ctx, ln(ctx, x, p + ".norm2"),
                                             p + ".linear1")),
                   p + ".linear2");
        return ggml_add(ctx, x, y);
    }

    ggml_tensor *decoder(ggml_context *ctx,
                         ggml_tensor *x,
                         ggml_tensor *memory,
                         const std::string &p,
                         ggml_tensor *pair_mask,
                         ggml_tensor *memory_mask,
                         bool last = false) {
        auto *y = ln(ctx, x, p + ".norm1");
        const std::string cross =
                p + (last ? ".cross_attn" : ".multihead_attn");
        if (strict_cuda()) {
            x = ggml_add(
                    ctx, x,
                    native_self_attention(ctx, y, p + ".self_attn", pair_mask));
            x = ggml_add(ctx, x,
                         strict_cross_attention(ctx, ln(ctx, x, p + ".norm2"),
                                                memory, cross, memory_mask));
        } else {
            x = ggml_add(ctx, x, mha(ctx, y, y, p + ".self_attn", pair_mask));
            x = ggml_add(ctx, x,
                         mha(ctx, ln(ctx, x, p + ".norm2"), memory, cross,
                             memory_mask));
        }
        y = ln(ctx, x, p + ".norm3");
        auto *ffn1 = affine(ctx, y, p + (last ? ".ffn.0" : ".linear1"));
        // 显式标记 bias ADD；后端只在完整、安全的域中合并写出。
        ggml_elementwise_set_epilogue_hint(ffn1, GGML_EPILOGUE_GELU);
        y = affine(ctx, ggml_gelu_erf(ctx, ffn1),
                   p + (last ? ".ffn.3" : ".linear2"));
        return ggml_add(ctx, x, y);
    }

    bool strict_cuda() const {
        return options.strict_f32 && options.backend == "cuda";
    }
    // Candidate identities are fixed before the relation head; dropping invalid
    // rows preserves the public 128-slot wire format while reducing graph work.
    bool compact_pairs() const { return options.compact_pairs; }

    // Fourier features of (x, y) points as [sin x, cos x, sin y, cos y] with
    // 16 bands each: freqs = 2**linspace(0, 7, 16), angles =
    // (v*freqs)*float(pi). Strict CUDA evaluates powf/sinf/cosf on the device
    // like PyTorch.
    ggml_tensor *fourier(GraphRun &run, const std::vector<float> &points) {
        const int64_t count = static_cast<int64_t>(points.size() / 2);
        if (!strict_cuda()) {
            auto values = make_fourier_matrix(points, count);
            return input_tensor(run, run.ctx, GGML_TYPE_F32, 64, count, 1, 1,
                                values.data());
        }
        auto *ctx = run.ctx;
        const auto octaves = fourier_octaves();
        std::array<float, 16> two{};
        two.fill(2.0f);
        std::vector<float> coords(16 * points.size());
        for (std::size_t i = 0; i < points.size(); ++i)
            std::fill_n(coords.begin() + 16 * i, 16, points[i]);
        auto *freqs = ggml_pow(
                ctx,
                input_tensor(run, ctx, GGML_TYPE_F32, 16, 1, 1, 1, two.data()),
                input_tensor(run, ctx, GGML_TYPE_F32, 16, 1, 1, 1,
                             octaves.data()));
        auto *angles =
                ggml_scale(ctx,
                           ggml_mul(ctx,
                                    input_tensor(run, ctx, GGML_TYPE_F32, 16, 2,
                                                 count, 1, coords.data()),
                                    freqs),
                           static_cast<float>(M_PI));
        return ggml_reshape_2d(ctx,
                               ggml_concat(ctx, ggml_sin(ctx, angles),
                                           ggml_cos(ctx, angles), 0),
                               64, count);
    }

    // [19, count] RelGeomEncoder features; strict CUDA evaluates logf/tanhf on
    // the device.
    ggml_tensor *geometry_features(GraphRun &run,
                                   const std::vector<GeometryTerms> &terms) {
        auto *ctx = run.ctx;
        const int64_t count = static_cast<int64_t>(terms.size());
        std::vector<float> arg, lin;
        for (const auto &g : terms) {
            if (strict_cuda()) {
                arg.insert(arg.end(), g.arg.begin(), g.arg.end());
                lin.insert(lin.end(), g.lin.begin(), g.lin.end());
            } else {
                const auto f = geometry(g);
                lin.insert(lin.end(), f.begin(), f.end());
            }
        }
        if (!strict_cuda())
            return input_tensor(run, ctx, GGML_TYPE_F32, 19, count, 1, 1,
                                lin.data());
        auto *logs = ggml_log(ctx, input_tensor(run, ctx, GGML_TYPE_F32, 19,
                                                count, 1, 1, arg.data()));
        auto *feats = ggml_add(ctx, logs,
                               input_tensor(run, ctx, GGML_TYPE_F32, 19, count,
                                            1, 1, lin.data()));
        return ggml_scale(ctx,
                          ggml_tanh(ctx, ggml_scale(ctx, feats, 1.0f / 10.0f)),
                          10.0f);
    }

    ggml_tensor *scene_pe(GraphRun &run, const std::string &p, int h, int w) {
        const auto cached = cached_positions.tensors.find(p);
        if (cached != cached_positions.tensors.end()) {
            if (cached->second->ne[1] != h * w)
                throw std::runtime_error("cached position grid mismatch");
            return cached->second;
        }
        // Patch centres (i + 0.5) * (1/n): CUDA turns the scalar division into
        // a reciprocal multiply.
        std::vector<float> points;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                points.push_back((static_cast<float>(x) + 0.5f) *
                                 (1.0f / static_cast<float>(w)));
                points.push_back((static_cast<float>(y) + 0.5f) *
                                 (1.0f / static_cast<float>(h)));
            }
        auto *pe = fourier(run, points);
        return ggml_mul(run.ctx, affine(run.ctx, pe, p + ".proj"),
                        as_f32(run.ctx, t("ra." + p + ".gamma")));
    }

    // Corner tokens of `boxes` ([rows][stride] cxcywh boxes), one tensor per
    // (box slot, corner) group in PyTorch's projection order: each group is
    // its own rows x 64 GEMM plus corner_bias(TL = 0, BR = 1).
    std::vector<ggml_tensor *> corners(GraphRun &run,
                                       const std::vector<float> &boxes,
                                       const std::string &p,
                                       int stride = 1) {
        auto *ctx = run.ctx;
        const std::size_t rows = boxes.size() / (4 * stride);
        std::vector<float> points;
        points.reserve(4 * stride * rows);
        for (int slot = 0; slot < stride; ++slot)
            for (int corner = 0; corner < 2; ++corner)
                for (std::size_t i = 0; i < rows; ++i) {
                    const auto b = xyxy(boxes.data() + 4 * (i * stride + slot));
                    points.push_back(b[2 * corner]);
                    points.push_back(b[2 * corner + 1]);
                }
        auto *pe = fourier(run, points);
        std::vector<ggml_tensor *> groups;
        for (int g = 0; g < 2 * stride; ++g) {
            const int32_t kind = g % 2;
            auto *ids =
                    input_tensor(run, ctx, GGML_TYPE_I32, 1, 1, 1, 1, &kind);
            auto *bias = as_f32(
                    ctx,
                    ggml_get_rows(ctx, t("ra." + p + ".corner_bias.weight"),
                                  ids));
            auto *rows_pe = ggml_view_2d(ctx, pe, 64, rows, pe->nb[1],
                                         g * rows * pe->nb[1]);
            groups.push_back(
                    ggml_add(ctx, affine(ctx, rows_pe, p + ".proj"), bias));
        }
        return groups;
    }

    ggml_tensor *
    pool(GraphRun &run,
         ggml_tensor *fmap,
         const std::vector<float> &boxes,
         int h,
         int w,
         const std::vector<float> &coverage = {},
         std::vector<std::pair<std::string, ggml_tensor *>> *trace = nullptr) {
        auto *ctx = run.ctx;
        const int64_t n = boxes.size() / 4;
        auto corner = corners(run, boxes, "spatial_pool.box_pe");
        // corner_tokens.mean(dim=2) is (TL + BR) * 0.5.
        auto *query = ggml_add(
                ctx, ggml_scale(ctx, ggml_add(ctx, corner[0], corner[1]), .5f),
                ggml_reshape_1d(
                        ctx, as_f32(ctx, t("ra.spatial_pool.base_query")), c));
        auto *pe = scene_pe(run, "spatial_pool.scene_pe", h, w);
        auto *memory = ggml_add(ctx, fmap, pe);
        if (trace) {
            trace->emplace_back("pool_scene_pe", pe);
            trace->emplace_back("pool_tl", corner[0]);
            trace->emplace_back("pool_br", corner[1]);
            trace->emplace_back("pool_query", query);
        }
        ggml_tensor *mask = nullptr;
        if (!coverage.empty()) {
            auto lambda = host_f32(t("ra.spatial_pool.cov_lambda"));
            std::vector<float> bias(8 * n * h * w);
            for (int head = 0; head < 8; ++head)
                for (int64_t i = 0; i < n * h * w; ++i)
                    bias[head * n * h * w + i] =
                            lambda[head] *
                            std::log(std::clamp(coverage[i], 0.0f, 1.0f) +
                                     1e-4f);
            mask = input_tensor(run, ctx, GGML_TYPE_F32, h * w, n, 8, 1,
                                bias.data());
        }
        auto *attn = mha(ctx, query, memory, "spatial_pool.cross_attn", mask,
                         true, trace);
        if (trace) trace->emplace_back("pool_out_proj", attn);
        return ln(ctx, attn, "spatial_pool.norm");
    }

    ggml_tensor *key_mask(GraphRun &run,
                          const std::vector<uint8_t> &valid,
                          int queries) {
        if (compact_pairs() &&
            std::all_of(valid.begin(), valid.end(),
                        [](uint8_t value) { return value != 0; }))
            return nullptr;
        // Strict CUDA attention broadcasts one [keys, 1] row over queries and
        // heads.
        if (strict_cuda()) queries = 1;
        std::vector<float> mask;
        build_mask(mask, valid.size(), queries, valid);
        return input_tensor(run, run.ctx, GGML_TYPE_F32, valid.size(), queries,
                            1, 1, mask.data());
    }

    ggml_tensor *deformable(GraphRun &run,
                            ggml_tensor *scene,
                            ggml_tensor *offsets,
                            ggml_tensor *weights,
                            const std::vector<float> &anchors,
                            int h,
                            int w,
                            int count) {
        auto *ctx = run.ctx;
        constexpr int points = 16, nh = 8;
        const int dh = d / nh;
        std::vector<float> centers(count * points * 2), half(centers.size());
        for (int i = 0; i < count; ++i)
            for (int p = 0; p < points; ++p)
                for (int axis = 0; axis < 2; ++axis) {
                    centers[(i * points + p) * 2 + axis] =
                            anchors[i * 16 + (p / 4) * 4 + axis];
                    half[(i * points + p) * 2 + axis] =
                            .5f *
                            std::max(.05f,
                                     anchors[i * 16 + (p / 4) * 4 + axis + 2]);
                }
        auto *cp = input_tensor(run, ctx, GGML_TYPE_F32, 2, points, 1, count,
                                centers.data());
        auto *hp = input_tensor(run, ctx, GGML_TYPE_F32, 2, points, 1, count,
                                half.data());
        auto *pos =
                ggml_clamp(ctx,
                           ggml_add(ctx,
                                    ggml_mul(ctx,
                                             ggml_reshape_4d(ctx, offsets, 2,
                                                             points, nh, count),
                                             hp),
                                    cp),
                           0, 1);
        if (strict_cuda()) {
            // PyTorch: grid_sample over [H, dh, h, w], cat null slots, then
            // einsum("bdkp,bkp->bkd") == bmm([H*K, dh, 24], [H*K, 24, 1]).
            auto *image = ggml_reshape_4d(
                    ctx, ggml_cont(ctx, ggml_transpose(ctx, scene)), w, h, dh,
                    nh);
            auto *coords = ggml_scale_bias(
                    ctx, ggml_cont(ctx, ggml_permute(ctx, pos, 0, 1, 3, 2)), 2,
                    -1);
            auto *vals =
                    ggml_reshape_4d(ctx, ggml_grid_sample(ctx, image, coords),
                                    4, 4, count, dh * nh);
            auto *nulls = ggml_cont(
                    ctx,
                    ggml_permute(
                            ctx,
                            ggml_reshape_4d(
                                    ctx,
                                    as_f32(ctx,
                                           t("ra.deformable_read.null_vec")),
                                    dh, 2, 4, nh),
                            2, 0, 1, 3));
            nulls = ggml_repeat_4d(
                    ctx, ggml_reshape_4d(ctx, nulls, 2, 4, 1, dh * nh), 2, 4,
                    count, dh * nh);
            vals = ggml_reshape_4d(ctx, ggml_concat(ctx, vals, nulls, 0), 24,
                                   count, dh, nh);
            vals = ggml_reshape_3d(
                    ctx, ggml_cont(ctx, ggml_permute(ctx, vals, 0, 2, 1, 3)),
                    24, dh, count * nh);
            auto *wh = ggml_reshape_3d(
                    ctx, ggml_cont(ctx, ggml_permute(ctx, weights, 0, 2, 1, 3)),
                    24, 1, count * nh);
            auto *read =
                    ggml_reshape_3d(ctx, matmul(ctx, vals, wh), dh, count, nh);
            return ggml_reshape_2d(
                    ctx, ggml_cont(ctx, ggml_permute(ctx, read, 0, 2, 1, 3)), d,
                    count);
        }
        // [xy, point, head, pair] -> [point, pair, head, xy]. Keeping the
        // channel vector contiguous makes GET_ROWS a native GPU gather.
        pos = ggml_cont(ctx, ggml_permute(ctx, pos, 3, 0, 2, 1));
        auto *px = ggml_view_3d(ctx, pos, points, count, nh, pos->nb[1],
                                pos->nb[2], 0);
        auto *py = ggml_view_3d(ctx, pos, points, count, nh, pos->nb[1],
                                pos->nb[2], pos->nb[3]);
        px = ggml_scale_bias(ctx, ggml_reshape_2d(ctx, px, points * count, nh),
                             w, .5f);
        py = ggml_scale_bias(ctx, ggml_reshape_2d(ctx, py, points * count, nh),
                             h, .5f);
        auto *x0 = ggml_floor(ctx, px), *y0 = ggml_floor(ctx, py);
        auto *dx = ggml_sub(ctx, px, x0), *dy = ggml_sub(ctx, py, y0);
        auto *base = ggml_add(ctx, x0, ggml_scale(ctx, y0, w + 2));
        auto *grid = ggml_cont(
                ctx,
                ggml_permute(ctx, ggml_reshape_4d(ctx, scene, dh, nh, w, h), 0,
                             3, 1, 2));
        // Zero border reproduces grid_sample(..., padding_mode="zeros",
        // align_corners=False), including half-covered image boundary pixels.
        grid = ggml_pad_ext(ctx, grid, 0, 0, 1, 1, 1, 1, 0, 0);
        grid = ggml_reshape_3d(ctx, grid, dh, (w + 2) * (h + 2), nh);
        ggml_tensor *sampled = nullptr;
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x) {
                auto *ids = ggml_cast(
                        ctx, ggml_scale_bias(ctx, base, 1, x + (w + 2) * y),
                        GGML_TYPE_I32);
                auto *wx = x ? dx : ggml_scale_bias(ctx, dx, -1, 1);
                auto *wy = y ? dy : ggml_scale_bias(ctx, dy, -1, 1);
                auto *weighted =
                        ggml_mul(ctx, ggml_get_rows(ctx, grid, ids),
                                 ggml_reshape_3d(ctx, ggml_mul(ctx, wx, wy), 1,
                                                 points * count, nh));
                sampled = sampled ? ggml_add(ctx, sampled, weighted) : weighted;
            }
        auto *wh = ggml_cont(ctx, ggml_permute(ctx,
                                               ggml_reshape_4d(ctx, weights, 6,
                                                               4, nh, count),
                                               0, 1, 3, 2));
        auto *real_weights =
                ggml_cont(ctx, ggml_view_4d(ctx, wh, 4, 4, count, nh, wh->nb[1],
                                            wh->nb[2], wh->nb[3], 0));
        sampled = ggml_mul(
                ctx, sampled,
                ggml_reshape_3d(ctx, real_weights, 1, points * count, nh));
        sampled =
                ggml_cont(ctx, ggml_permute(ctx,
                                            ggml_reshape_4d(ctx, sampled, dh,
                                                            points, count, nh),
                                            1, 0, 2, 3));
        auto *read = ggml_reshape_3d(ctx, ggml_sum_rows(ctx, sampled), dh,
                                     count, nh);
        auto *null_weights = ggml_cont(
                ctx, ggml_view_4d(ctx, wh, 2, 4, count, nh, wh->nb[1],
                                  wh->nb[2], wh->nb[3], 4 * sizeof(float)));
        auto *nulls = ggml_cont(
                ctx,
                ggml_permute(
                        ctx,
                        ggml_reshape_3d(
                                ctx,
                                as_f32(ctx, t("ra.deformable_read.null_vec")),
                                dh, 8, nh),
                        1, 0, 2, 3));
        read = ggml_add(
                ctx, read,
                matmul(ctx, nulls,
                       ggml_reshape_3d(ctx, null_weights, 8, count, nh)));
        return ggml_reshape_2d(
                ctx, ggml_cont(ctx, ggml_permute(ctx, read, 0, 2, 1, 3)), d,
                count);
    }

    void dump(const std::string &name, const std::vector<float> &values) const {
        if (options.dump_directory.empty()) return;
        std::filesystem::create_directories(options.dump_directory);
        std::ofstream file(options.dump_directory / (name + ".f32"),
                           std::ios::binary);
        file.write(reinterpret_cast<const char *>(values.data()),
                   values.size() * sizeof(float));
    }

    std::vector<std::vector<float>> outputs(
            GraphRun &run,
            const std::vector<std::pair<std::string, ggml_tensor *>>
                    &requested) {
        const auto started = std::chrono::steady_clock::now();
        run.graph = ggml_new_graph_custom(run.ctx, 8192, false);
        for (const auto &item : requested) {
            ggml_set_output(item.second);
            ggml_build_forward_expand(run.graph, item.second);
        }
        execute(run);
        const auto executed = std::chrono::steady_clock::now();
        std::vector<std::vector<float>> values(requested.size());
        std::vector<std::size_t> staging_offsets(
                requested.size(), std::numeric_limits<std::size_t>::max());
        std::size_t staging_bytes = 0;
        for (std::size_t i = 0; i < requested.size(); ++i) {
            if (retained_features.tensors.count(requested[i].first) == 0 &&
                requested[i].second->type == GGML_TYPE_F32) {
                staging_bytes = (staging_bytes + 63) & ~std::size_t(63);
                staging_offsets[i] = staging_bytes;
                staging_bytes += ggml_nbytes(requested[i].second);
            }
        }
        auto *staging = static_cast<std::uint8_t *>(readback(staging_bytes));
        std::vector<ggml_backend_t> pending_backends;
        for (std::size_t i = 0; i < requested.size(); ++i) {
            const auto &item = requested[i];
            const auto retained = retained_features.tensors.find(item.first);
            if (retained != retained_features.tensors.end()) {
                ggml_backend_tensor_copy(item.second, retained->second);
                continue;
            }
            if (item.second->type != GGML_TYPE_F32) {
                values[i] = read_f32(item.second);
                continue;
            }
            values[i].resize(ggml_nelements(item.second));
            void *destination =
                    staging ? static_cast<void *>(staging + staging_offsets[i])
                            : static_cast<void *>(values[i].data());
            auto *output_backend = ggml_backend_sched_get_tensor_backend(
                    backend->scheduler, item.second);
            ggml_backend_tensor_get_async(output_backend, item.second,
                                          destination, 0,
                                          ggml_nbytes(item.second));
            if (std::find(pending_backends.begin(), pending_backends.end(),
                          output_backend) == pending_backends.end())
                pending_backends.push_back(output_backend);
        }
        for (auto *output_backend : pending_backends)
            ggml_backend_synchronize(output_backend);
        if (staging)
            for (std::size_t i = 0; i < requested.size(); ++i) {
                if (staging_offsets[i] !=
                    std::numeric_limits<std::size_t>::max())
                    std::memcpy(values[i].data(), staging + staging_offsets[i],
                                ggml_nbytes(requested[i].second));
            }
        for (std::size_t i = 0; i < requested.size(); ++i) {
            if (!options.dump_directory.empty()) {
                const auto retained =
                        retained_features.tensors.find(requested[i].first);
                dump(requested[i].first,
                     retained == retained_features.tensors.end()
                             ? values[i]
                             : read_f32(requested[i].second));
            }
        }
        ggml_backend_sched_reset(backend->scheduler);
        if (options.profile)
            std::cerr << "outputs read_ms="
                      << std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - executed)
                                 .count()
                      << " total_ms="
                      << std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - started)
                                 .count()
                      << '\n';
        return values;
    }

    void execute(GraphRun &run) {
        const auto started = std::chrono::steady_clock::now();
        for (int i = 0; i < ggml_graph_n_nodes(run.graph); ++i) {
            auto *node = ggml_graph_node(run.graph, i);
            // Fused kernels dispatch using the final node's owner, including
            // ADD epilogues and views. Keep ownership across the whole graph.
            if (std::strncmp(node->name, "relateanything_", 15) != 0)
                ggml_format_name(node, "relateanything_%s",
                                 ggml_op_name(node->op));
        }
        if (options.quant_f32_input || options.strict_f32) {
            for (int i = 0; i < ggml_graph_n_nodes(run.graph); ++i) {
                auto *node = ggml_graph_node(run.graph, i);
                if (node->op == GGML_OP_SOFT_MAX && options.strict_f32)
                    ggml_soft_max_set_hint(node, GGML_HINT_STRICT_F32);
                if (node->op == GGML_OP_NORM && options.strict_f32)
                    ggml_norm_set_hint(node, GGML_HINT_STRICT_F32);
                if (node->op == GGML_OP_L2_NORM && options.strict_f32)
                    ggml_l2_norm_set_hint(node, GGML_HINT_STRICT_F32);
                // PyTorch evaluates these with precise libdevice
                // sinf/cosf/logf/tanhf/expf.
                const bool elementwise =
                        node->op == GGML_OP_SIN || node->op == GGML_OP_COS ||
                        node->op == GGML_OP_LOG ||
                        (node->op == GGML_OP_UNARY &&
                         (ggml_get_unary_op(node) == GGML_UNARY_OP_SILU ||
                          ggml_get_unary_op(node) == GGML_UNARY_OP_TANH ||
                          ggml_get_unary_op(node) == GGML_UNARY_OP_EXP));
                // GELU is strict only when it is the epilogue of a strict F32
                // addmm. Quantized/F16 GEMMs use the regular CUDA GELU instead
                // of falling back to CPU.
                bool strict_gelu = false;
                if (node->op == GGML_OP_UNARY &&
                    ggml_get_unary_op(node) == GGML_UNARY_OP_GELU &&
                    node->src[0]->op == GGML_OP_ADD) {
                    for (int source = 0; source < 2; ++source) {
                        const auto *mm = node->src[0]->src[source];
                        strict_gelu =
                                strict_gelu || (mm->op == GGML_OP_MUL_MAT &&
                                                ggml_mul_mat_get_hint(mm) ==
                                                        GGML_HINT_STRICT_F32);
                    }
                }
                if ((elementwise || strict_gelu) && options.strict_f32)
                    ggml_elementwise_set_hint(node, GGML_HINT_STRICT_F32);
                if (node->op == GGML_OP_OUT_PROD && options.strict_f32 &&
                    node->src[0]->type == GGML_TYPE_F32 &&
                    node->src[1]->type == GGML_TYPE_F32)
                    ggml_mul_mat_set_hint(node, GGML_HINT_STRICT_F32);
                if (node->op == GGML_OP_MUL_MAT && options.strict_f32 &&
                    node->src[0]->type == GGML_TYPE_F32 &&
                    node->src[1]->type == GGML_TYPE_F32 &&
                    ggml_mul_mat_get_hint(node) != GGML_HINT_SERIAL_F32)
                    ggml_mul_mat_set_hint(node, GGML_HINT_STRICT_F32);
                else if (node->op == GGML_OP_MUL_MAT &&
                         options.quant_f32_input &&
                         ggml_is_quantized(node->src[0]->type))
                    ggml_mul_mat_set_hint(node, GGML_HINT_QUANT_F32_INPUT);
            }
        }
        ggml_backend_sched_reset(backend->scheduler);
        for (const auto &item : run.inputs)
            ggml_backend_sched_set_tensor_backend(
                    backend->scheduler, item.first, backend->accelerator);
        if (!ggml_backend_sched_alloc_graph(backend->scheduler, run.graph))
            throw std::runtime_error("cannot allocate full graph");
        backend_buffer_bytes = std::max(
                backend_buffer_bytes,
                ggml_backend_buffer_get_size(weights_buffer) +
                        (packed_qkv.buffer ? ggml_backend_buffer_get_size(
                                                     packed_qkv.buffer)
                                           : 0) +
                        (cached_positions.buffer
                                 ? ggml_backend_buffer_get_size(
                                           cached_positions.buffer)
                                 : 0) +
                        (retained_features.buffer
                                 ? ggml_backend_buffer_get_size(
                                           retained_features.buffer)
                                 : 0) +
                        ggml_backend_sched_get_buffer_size(
                                backend->scheduler, backend->accelerator));
        const auto allocated = std::chrono::steady_clock::now();
        for (const auto &item : run.inputs)
            ggml_backend_tensor_set(item.first, item.second.data(), 0,
                                    ggml_nbytes(item.first));
        if (ggml_backend_sched_graph_compute(backend->scheduler, run.graph) !=
            GGML_STATUS_SUCCESS)
            throw std::runtime_error("full graph execution failed");
        ggml_backend_sched_synchronize(backend->scheduler);
        if (options.profile) {
            const auto done = std::chrono::steady_clock::now();
            std::map<std::string, int> cpu_ops;
            std::vector<std::string> cpu_nodes;
            for (int i = 0; i < ggml_graph_n_nodes(run.graph); ++i) {
                auto *node = ggml_graph_node(run.graph, i);
                if (ggml_backend_sched_get_tensor_backend(
                            backend->scheduler, node) == backend->cpu) {
                    ++cpu_ops[ggml_op_name(node->op)];
                    cpu_nodes.emplace_back(std::string(node->name) + ':' +
                                           ggml_op_desc(node));
                }
            }
            std::cerr << "graph nodes=" << ggml_graph_n_nodes(run.graph)
                      << " allocate_ms="
                      << std::chrono::duration<double, std::milli>(allocated -
                                                                   started)
                                 .count()
                      << " compute_ms="
                      << std::chrono::duration<double, std::milli>(done -
                                                                   allocated)
                                 .count();
            for (const auto &op : cpu_ops)
                std::cerr << " cpu_" << op.first << '=' << op.second;
            for (const auto &name : cpu_nodes)
                std::cerr << " cpu_node=" << name;
            std::cerr << '\n';
        }
    }

    float stage_ratio(std::size_t stage) const {
        // Stage stride is 4*2^i, so the ratio to the patch grid is fixed:
        // 4, 2, 1 and 0.5 at patch size 16.
        return static_cast<float>(patch) /
               (4.f * static_cast<float>(1 << stage));
    }

    // Quantized or F16 sources are cast once on the inference backend; the
    // returned host F32 array keeps the source physical order, which already
    // matches the packed GGML convolution layout.
    std::vector<float> dequant_f32(ggml_tensor *source) {
        const std::size_t n = ggml_nelements(source);
        std::vector<float> out(n);
        if (source->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(source, out.data(), 0, n * sizeof(float));
            return out;
        }
        auto run = make_run();
        run.graph = ggml_new_graph_custom(run.ctx, 8, false);
        std::vector<std::uint8_t> raw(ggml_nbytes(source));
        ggml_backend_tensor_get(source, raw.data(), 0, raw.size());
        auto *src = input_tensor(run, run.ctx, source->type, source->ne[0],
                                 source->ne[1], source->ne[2], source->ne[3],
                                 raw.data());
        auto *cast = ggml_cast(run.ctx, src, GGML_TYPE_F32);
        ggml_set_output(cast);
        ggml_build_forward_expand(run.graph, cast);
        execute(run);
        ggml_backend_tensor_get(cast, out.data(), 0, n * sizeof(float));
        ggml_backend_sched_reset(backend->scheduler);
        return out;
    }

    void build_convnext_convs() {
        // Every convolution weight is dequantized once at load into the F32
        // layouts the graph consumes. LayerNorm affines are folded into the
        // following linear map where the residual path allows it: a
        // LayerNorm affine y*w+b composed with a linear map equals the same
        // map with w folded into its input columns and W*b into its bias.
        // Biases themselves stay explicit: the norms normalize per position
        // over the channel axis, which only cancels scalars, not the
        // per-channel constant a bias adds.
        enum class Kind {
            kDownsampleStem,
            kDownsampleConv,
            kDownsampleBias,
            kDepthwise,
            kPointwisePacked,
            kStageProj
        };
        struct Spec {
            Kind kind;
            std::string name;
            std::array<int64_t, 4> ne;
            std::size_t s = 0, j = 0;
        };
        std::vector<Spec> specs;
        const auto stage_prefix = [](std::size_t s) {
            return "ra.backbone.model.model.stages." + std::to_string(s);
        };
        const auto ch = [&](std::size_t s) {
            return static_cast<int64_t>(stage_channels[s]);
        };
        specs.push_back({Kind::kDownsampleStem,
                         stage_prefix(0) + ".downsample_layers.0.weight",
                         {4, 4, 3, ch(0)}});
        for (std::size_t s = 1; s < 4; ++s) {
            specs.push_back({Kind::kDownsampleConv,
                             stage_prefix(s) + ".downsample_layers.1.weight",
                             {2, 2, ch(s - 1), ch(s)},
                             s});
            // The stage's downsample LayerNorm affine is folded into this
            // convolution: its scale multiplies the input channels and its
            // shift lands in the bias through the original kernel.
            specs.push_back(
                    {Kind::kDownsampleBias,
                     stage_prefix(s) + ".downsample_layers.1.bias.packed",
                     {ch(s), 1, 1, 1},
                     s});
        }
        for (std::size_t s = 0; s < 4; ++s)
            for (std::size_t j = 0; j < stage_depths[s]; ++j)
                specs.push_back({Kind::kDepthwise,
                                 stage_prefix(s) + ".layers." +
                                         std::to_string(j) +
                                         ".depthwise_conv.weight",
                                 {7, 7, 1, ch(s)},
                                 s,
                                 j});
        for (std::size_t s = 0; s < 4; ++s)
            for (std::size_t j = 0; j < stage_depths[s]; ++j) {
                const auto p = stage_prefix(s) + ".layers." + std::to_string(j);
                // The LayerScale gamma composes with the second pointwise map
                // as a row scale, so both weights and bias are folded here.
                specs.push_back({Kind::kPointwisePacked,
                                 p + ".pointwise_conv1.weight.packed",
                                 {ch(s), 4 * ch(s), 1, 1},
                                 s,
                                 j});
                specs.push_back({Kind::kPointwisePacked,
                                 p + ".pointwise_conv1.bias.packed",
                                 {4 * ch(s), 1, 1, 1},
                                 s,
                                 j});
                specs.push_back({Kind::kPointwisePacked,
                                 p + ".pointwise_conv2.weight.packed",
                                 {4 * ch(s), ch(s), 1, 1},
                                 s,
                                 j});
                specs.push_back({Kind::kPointwisePacked,
                                 p + ".pointwise_conv2.bias.packed",
                                 {ch(s), 1, 1, 1},
                                 s,
                                 j});
            }
        for (std::size_t s = 0; s < 4; ++s) {
            const float r = stage_ratio(s);
            if (r > 1.f)
                specs.push_back(
                        {Kind::kStageProj,
                         "ra.backbone.stage_proj." + std::to_string(s) +
                                 ".weight",
                         {static_cast<int64_t>(r), static_cast<int64_t>(r),
                          ch(s), static_cast<int64_t>(c)}});
        }
        packed_convs.ctx = ggml_init(
                {specs.size() * ggml_tensor_overhead(), nullptr, true});
        if (!packed_convs.ctx)
            throw std::runtime_error("cannot create packed conv kernels");
        for (const auto &spec : specs) {
            auto *kernel = ggml_new_tensor(packed_convs.ctx, GGML_TYPE_F32, 4,
                                           spec.ne.data());
            packed_convs.kernels.emplace(spec.name, kernel);
        }
        // Depthwise kernels switch to the channels-innermost (CWHN) memory
        // order every backend's conv_2d_dw dispatch expects for token-major
        // input: the logical shape stays [KW, KH, 1, C] but the strides place
        // the channel axis innermost, so the physical order is
        // (ky*KW + kx)*C + c.
        for (const auto &spec : specs)
            if (spec.kind == Kind::kDepthwise) {
                auto *k = packed_convs.kernels.at(spec.name);
                k->nb[0] = k->ne[3] * sizeof(float);
                k->nb[1] = k->ne[0] * k->nb[0];
                k->nb[2] = k->nb[0];
                k->nb[3] = sizeof(float);
            }
        packed_convs.buffer = ggml_backend_alloc_ctx_tensors(
                packed_convs.ctx, backend->accelerator);
        if (!packed_convs.buffer)
            throw std::runtime_error("cannot allocate packed conv kernels");
        ggml_backend_buffer_set_usage(packed_convs.buffer,
                                      GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        // Host-side normalization data consumed by the folds below: stage
        // s>0 folds its downsample LayerNorm affine into the downsample
        // convolution.
        std::array<std::vector<float>, 4> stage_ln_w, stage_ln_b;
        for (std::size_t s = 1; s < 4; ++s) {
            stage_ln_w[s] = dequant_f32(
                    t(stage_prefix(s) + ".downsample_layers.0.weight"));
            stage_ln_b[s] = dequant_f32(
                    t(stage_prefix(s) + ".downsample_layers.0.bias"));
        }
        for (const auto &spec : specs) {
            auto *kernel = packed_convs.kernels.at(spec.name);
            std::vector<float> host;
            switch (spec.kind) {
                case Kind::kDepthwise: {
                    const int64_t chn = ch(spec.s);
                    // [C,1,7,7] source: c*49 + ky*7 + kx
                    auto src = dequant_f32(t(spec.name));
                    host.resize(src.size());
                    for (int64_t ci = 0; ci < chn; ++ci)
                        for (int64_t ky = 0; ky < 7; ++ky)
                            for (int64_t kx = 0; kx < 7; ++kx)
                                host[(ky * 7 + kx) * chn + ci] =
                                        src[ci * 49 + ky * 7 + kx];
                    break;
                }
                case Kind::kDownsampleConv: {
                    // [KW,KH,IC,OC] physical order
                    host = dequant_f32(t(spec.name));
                    const int64_t ic_n = ch(spec.s - 1);
                    const auto &ln = stage_ln_w[spec.s];
                    for (std::size_t i = 0; i < host.size(); ++i)
                        host[i] *= ln[(i / 4) % ic_n];
                    break;
                }
                case Kind::kDownsampleBias: {
                    // bias' = bias + kernel_orig * ln_shift, summed over the
                    // input channels of the [KW,KH,IC,OC] layout (KW*KH = 4).
                    auto k = dequant_f32(t(stage_prefix(spec.s) +
                                           ".downsample_layers.1.weight"));
                    auto b = dequant_f32(t(stage_prefix(spec.s) +
                                           ".downsample_layers.1.bias"));
                    const auto &shift = stage_ln_b[spec.s];
                    const int64_t oc_n = ch(spec.s);
                    const int64_t ic_n = ch(spec.s - 1);
                    host.resize(oc_n);
                    for (int64_t oc = 0; oc < oc_n; ++oc) {
                        const int64_t base = oc * 4 * ic_n;
                        float acc = b[oc];
                        for (int64_t ic = 0; ic < ic_n; ++ic)
                            acc += shift[ic] *
                                   (k[base + ic * 4] + k[base + ic * 4 + 1] +
                                    k[base + ic * 4 + 2] +
                                    k[base + ic * 4 + 3]);
                        host[oc] = acc;
                    }
                    break;
                }
                case Kind::kPointwisePacked: {
                    const auto p = stage_prefix(spec.s) + ".layers." +
                                   std::to_string(spec.j);
                    const bool pw1 = spec.name.find("pointwise_conv1") !=
                                     std::string::npos;
                    auto w = dequant_f32(
                            t(p + (pw1 ? ".pointwise_conv1.weight"
                                       : ".pointwise_conv2.weight")));
                    auto b =
                            dequant_f32(t(p + (pw1 ? ".pointwise_conv1.bias"
                                                   : ".pointwise_conv2.bias")));
                    const int64_t dd = ch(spec.s), d4 = 4 * dd;
                    const bool is_weight =
                            spec.name.compare(spec.name.size() - 14, 14,
                                              ".weight.packed") == 0;
                    if (pw1) {
                        // Block LayerNorm affine folded into the first map.
                        auto ln_w = dequant_f32(t(p + ".layer_norm.weight"));
                        auto ln_b = dequant_f32(t(p + ".layer_norm.bias"));
                        if (is_weight) {
                            for (int64_t oc = 0; oc < d4; ++oc)
                                for (int64_t ic = 0; ic < dd; ++ic)
                                    w[oc * dd + ic] *= ln_w[ic];
                            host = std::move(w);
                        } else {
                            host.resize(d4);
                            for (int64_t oc = 0; oc < d4; ++oc) {
                                float acc = b[oc];
                                for (int64_t ic = 0; ic < dd; ++ic)
                                    acc += w[oc * dd + ic] * ln_b[ic];
                                host[oc] = acc;
                            }
                        }
                    } else {
                        // LayerScale gamma folded into the second map's rows.
                        auto gamma = dequant_f32(t(p + ".gamma"));
                        if (is_weight) {
                            for (int64_t oc = 0; oc < dd; ++oc)
                                for (int64_t ic = 0; ic < d4; ++ic)
                                    w[oc * d4 + ic] *= gamma[oc];
                            host = std::move(w);
                        } else {
                            for (int64_t oc = 0; oc < dd; ++oc)
                                b[oc] *= gamma[oc];
                            host = std::move(b);
                        }
                    }
                    break;
                }
                default:
                    host = dequant_f32(t(spec.name));
                    break;
            }
            if (host.size() != static_cast<std::size_t>(ggml_nelements(kernel)))
                throw std::runtime_error("conv kernel size mismatch: " +
                                         spec.name);
            ggml_backend_tensor_set(kernel, host.data(), 0,
                                    host.size() * sizeof(float));
        }
    }

    // ConvNeXt block: depthwise 7x7, channel LayerNorm, gated FFN. The block
    // keeps the token-major [C, h*w] layout end to end: it is
    // channels-contiguous, so every backend's depthwise dispatch takes its
    // CWHN variant and returns token-major, and the channel LayerNorm is the
    // bare normalize. The norm affine is folded into pointwise_conv1 at load
    // (see build_convnext_convs); the depthwise bias stays explicit because
    // the per-position norm only cancels scalars, not per-channel constants.
    ggml_tensor *convnext_block(ggml_context *ctx,
                                ggml_tensor *x,
                                int w,
                                int h,
                                std::size_t stage,
                                std::size_t j) {
        const auto p = "ra.backbone.model.model.stages." +
                       std::to_string(stage) + ".layers." + std::to_string(j);
        const int64_t C = x->ne[0], hw = x->ne[1];
        // [C, w, h] permuted to [W, H, C] semantics: a zero-copy
        // channels-contiguous view of the token-major input.
        auto *spatial =
                ggml_permute(ctx, ggml_reshape_3d(ctx, x, C, w, h), 2, 0, 1, 3);
        // bias 仍在卷积写出舍入后相加，减少一次全激活广播 ADD。
        auto *dw = ggml_conv_2d_dw_direct_bias(
                ctx, packed_convs.kernels.at(p + ".depthwise_conv.weight"),
                spatial, as_f32(ctx, t(p + ".depthwise_conv.bias")),
                1, 1, 3, 3, 1, 1);
        auto *y = ggml_view_2d(ctx, dw, C, hw, C * sizeof(float), 0);
        y = ggml_norm(ctx, y, eps);
        y = linear(
                ctx, y,
                packed_convs.kernels.at(p + ".pointwise_conv1.weight.packed"),
                packed_convs.kernels.at(p + ".pointwise_conv1.bias.packed"));
        ggml_elementwise_set_epilogue_hint(y, GGML_EPILOGUE_GELU);
        y = ggml_gelu_erf(ctx, y);
        // pointwise_conv2 carries the folded LayerScale gamma, so the branch
        // result feeds the residual directly.
        y = linear(
                ctx, y,
                packed_convs.kernels.at(p + ".pointwise_conv2.weight.packed"),
                packed_convs.kernels.at(p + ".pointwise_conv2.bias.packed"));
        return ggml_add(ctx, x, y);
    }

    // token-major [C_in, h*w] -> im2col+matmul -> token-major [C_out, h'*w'].
    // The r x r stride-r projection also realizes pixel_unshuffle: reshaping
    // [d, C*r^2] to the [KW, KH, IC, OC] convolution layout composes the
    // phase folding with the 1x1 projection in one product. Biases are added
    // by the callers: each consumer applies its own (possibly folded) shift.
    ggml_tensor *convnext_conv(ggml_context *ctx,
                               ggml_tensor *x,
                               int w,
                               int h,
                               const std::string &kernel_name,
                               int64_t c_out,
                               int stride,
                               ggml_tensor *spatial_input = nullptr) {
        auto *kernel = packed_convs.kernels.at(kernel_name);
        const int64_t K = kernel->ne[0] * kernel->ne[1] * kernel->ne[2];
        auto *view =
                spatial_input
                        ? spatial_input
                        : ggml_cont(ctx, ggml_permute(ctx,
                                                      ggml_reshape_3d(ctx, x,
                                                                      x->ne[0],
                                                                      w, h),
                                                      2, 0, 1, 3));
        auto *columns = ggml_im2col(ctx, kernel, view, stride, stride, 0, 0, 1,
                                    1, true, GGML_TYPE_F32);
        const int64_t w_out = w / stride, h_out = h / stride;
        return matmul(ctx, ggml_reshape_2d(ctx, kernel, K, c_out),
                      ggml_reshape_2d(ctx, columns, K, w_out * h_out));
    }

    ggml_tensor *backbone_convnext(
            GraphRun &run,
            ggml_context *ctx,
            ggml_tensor *image,
            std::vector<ggml_tensor *> &taps,
            int height,
            int width,
            std::vector<std::pair<std::string, ggml_tensor *>> *trace) {
        (void)run;
        auto tokenize = [&](ggml_tensor *map, int64_t channels, int64_t hw) {
            // [W, H, C] -> token-major [C, hw].
            return ggml_reshape_2d(
                    ctx, ggml_cont(ctx, ggml_permute(ctx, map, 1, 2, 0, 3)),
                    channels, hw);
        };
        ggml_tensor *x = nullptr;
        int w = width, h = height;
        for (std::size_t s = 0; s < 4; ++s) {
            const auto prefix =
                    "ra.backbone.model.model.stages." + std::to_string(s);
            if (s == 0) {
                // The stem LayerNorm keeps its full affine: the residual path
                // branches right after it, so the affine cannot move past the
                // branch point.
                x = convnext_conv(ctx, nullptr, w, h,
                                  prefix + ".downsample_layers.0.weight",
                                  stage_channels[0], 4, image);
                x = ggml_add(
                        ctx, x,
                        as_f32(ctx, t(prefix + ".downsample_layers.0.bias")));
                x = norm(ctx, x, t(prefix + ".downsample_layers.1.weight"),
                         t(prefix + ".downsample_layers.1.bias"), eps);
            } else {
                // The downsample LayerNorm affine is folded into the
                // convolution that follows (scale into the input channels,
                // shift into its packed bias); the residual branches at that
                // convolution's output, so the fold is exact.
                x = ggml_norm(ctx, x, eps);
                x = convnext_conv(ctx, x, w, h,
                                  prefix + ".downsample_layers.1.weight",
                                  stage_channels[s], 2);
                x = ggml_add(
                        ctx, x,
                        packed_convs.kernels.at(
                                prefix + ".downsample_layers.1.bias.packed"));
            }
            w /= s == 0 ? 4 : 2;
            h /= s == 0 ? 4 : 2;
            for (std::size_t j = 0; j < stage_depths[s]; ++j)
                x = convnext_block(ctx, x, w, h, s, j);
            if (trace)
                trace->emplace_back("convnext_stage_" + std::to_string(s), x);
            taps.push_back(x);
        }
        // Stage fusion: the finer stages fold r x r phases into the projection
        // convolution, the stride-32 map is bilinearly resized, and every
        // branch shares projection+norm and the softmax combiner.
        const int grid = height / patch;
        std::vector<ggml_tensor *> projected;
        for (std::size_t s = 0; s < 4; ++s) {
            const std::string p = "ra.backbone.stage_proj." + std::to_string(s);
            const std::string n = "ra.backbone.stage_norm." + std::to_string(s);
            const float r = stage_ratio(s);
            const int64_t channels = stage_channels[s];
            ggml_tensor *f = taps[s];
            if (r > 1.f) {
                const int sw = grid * static_cast<int>(r),
                          sh = grid * static_cast<int>(r);
                f = convnext_conv(ctx, f, sw, sh, p + ".weight", c,
                                  static_cast<int>(r));
                f = ggml_add(ctx, f, as_f32(ctx, t(p + ".bias")));
            } else {
                if (r < 1.f) {
                    // Bilinear resize of the [W, H, C] view to the patch grid.
                    auto *view = ggml_cont(
                            ctx,
                            ggml_permute(ctx,
                                         ggml_reshape_3d(ctx, f, channels,
                                                         grid / 2, grid / 2),
                                         2, 0, 1, 3));
                    auto *up = ggml_interpolate(ctx, view, grid, grid, channels,
                                                1, GGML_SCALE_MODE_BILINEAR);
                    f = tokenize(up, channels,
                                 static_cast<int64_t>(grid) * grid);
                }
                // The 1x1 projection stores a 4D Conv2d weight [d, C, 1, 1];
                // flatten it to [C, d].
                f = linear(ctx, f,
                           ggml_reshape_2d(ctx, t(p + ".weight"), channels,
                                           static_cast<int64_t>(c)),
                           t(p + ".bias"));
            }
            f = norm(ctx, f, t(n + ".weight"), t(n + ".bias"), eps);
            projected.push_back(f);
        }
        if (options.strict_f32 && options.backend == "cuda") {
            // Official F.softmax(layer_weights) runs on the device; its expf
            // and warp-sum order differ from host std::exp by 1 ulp on some
            // checkpoints.
            auto *sw = ggml_soft_max(
                    ctx, as_f32(ctx, t("ra.backbone.layer_weights")));
            ggml_tensor *fused =
                    ggml_mul(ctx, projected[0], ggml_view_1d(ctx, sw, 1, 0));
            for (std::size_t i = 1; i < projected.size(); ++i)
                fused = ggml_add(
                        ctx, fused,
                        ggml_mul(ctx, projected[i],
                                 ggml_view_1d(ctx, sw, 1, i * sizeof(float))));
            if (trace) trace->emplace_back("backbone_patch", fused);
            return fused;
        }
        auto weights = tap_weights;
        const float mx = *std::max_element(weights.begin(), weights.end());
        float sum = 0;
        for (auto &v : weights) {
            v = std::exp(v - mx);
            sum += v;
        }
        ggml_tensor *fused = ggml_scale(ctx, projected[0], weights[0] / sum);
        for (std::size_t i = 1; i < projected.size(); ++i)
            fused = ggml_add(ctx, fused,
                             ggml_scale(ctx, projected[i], weights[i] / sum));
        if (trace) trace->emplace_back("backbone_patch", fused);
        return fused;
    }

    ggml_tensor *backbone(GraphRun &run,
                          ggml_context *ctx,
                          ggml_tensor *image,
                          std::vector<ggml_tensor *> &taps,
                          int height,
                          int width,
                          std::vector<std::pair<std::string, ggml_tensor *>>
                                  *trace = nullptr) {
        if (model_family == "dinov3_convnext")
            return backbone_convnext(run, ctx, image, taps, height, width,
                                     trace);
        auto *kernel =
                t("ra.backbone.model.embeddings.patch_embeddings.weight");
        auto *columns = ggml_im2col(ctx, kernel, image, patch, patch, 0, 0, 1,
                                    1, true, GGML_TYPE_F32);
        const int64_t patch_tokens = (height / patch) * (width / patch);
        auto *patches = matmul(
                ctx, ggml_reshape_2d(ctx, kernel, 3 * patch * patch, c),
                ggml_reshape_2d(ctx, columns, 3 * patch * patch, patch_tokens));
        if (options.strict_f32 && options.backend == "cuda" &&
            kernel->type == GGML_TYPE_F32)
            ggml_mul_mat_set_hint(patches, GGML_HINT_SERIAL_F32);
        // GGML [C,N] has the same contiguous storage as PyTorch [N,C].
        patches = ggml_cont(ctx, patches);
        patches = ggml_add(ctx, patches,
                           as_f32(ctx, t("ra.backbone.model.embeddings.patch_"
                                         "embeddings.bias")));
        if (trace) trace->emplace_back("backbone_patch", patches);
        ggml_tensor *cls = ggml_reshape_2d(
                ctx, t("ra.backbone.model.embeddings.cls_token"), c, 1);
        ggml_tensor *reg = ggml_reshape_2d(
                ctx, t("ra.backbone.model.embeddings.register_tokens"), c,
                regs);
        ggml_tensor *x = concat_axis1(ctx, {cls, reg, patches});
        const int64_t tokens = x->ne[1];
        const int64_t dh = static_cast<int64_t>(c / heads);
        // Strict CUDA evaluates trig and attention with PyTorch's kernels.
        const bool strict_cuda =
                options.strict_f32 && options.backend == "cuda";
        if (rope_height != height || rope_width != width) {
            // Angles follow HF DINOv3 rounding step by step: inv_freq =
            // 1/powf(theta, i*4/dh); centers (i+0.5)*(1/n) (CUDA turns the
            // scalar division into a reciprocal multiply), then 2c-1; angle =
            // (float(2pi)*coord)*inv_freq. Layout per token: 16 y, 16 x, tiled.
            rope_angle.assign(dh * tokens, 0.0f);
            const float theta = meta_float("ra.rope_theta", 100.0f);
            const int quarter = static_cast<int>(dh / 4);
            std::vector<float> inv(quarter);
            for (int i = 0; i < quarter; ++i)
                inv[i] = 1.0f /
                         std::pow(theta,
                                  static_cast<float>(i) *
                                          (4.0f / static_cast<float>(dh)));
            const float two_pi = static_cast<float>(2.0 * M_PI);
            const int64_t grid_h = height / patch, grid_w = width / patch;
            const float step_h = 1.0f / static_cast<float>(grid_h),
                        step_w = 1.0f / static_cast<float>(grid_w);
            for (int64_t token = regs + 1; token < tokens; ++token) {
                const int64_t p = token - regs - 1, py = p / grid_w,
                              px = p % grid_w;
                for (int64_t j = 0; j < dh; ++j) {
                    const int fi = static_cast<int>(j % quarter);
                    const bool y_axis = ((j / quarter) % 2) == 0;
                    const float unit =
                            y_axis ? (static_cast<float>(py) + 0.5f) * step_h
                                   : (static_cast<float>(px) + 0.5f) * step_w;
                    // DINOv3 defines patch centers in [-1, 1], not [0, 1].
                    const float coord = 2.0f * unit - 1.0f;
                    const float scaled = two_pi * coord;
                    rope_angle[j + token * dh] = scaled * inv[fi];
                }
            }
            rope_cos.resize(rope_angle.size());
            rope_sin.resize(rope_angle.size());
            // Prefix tokens have angle 0, so cos = 1 and sin = 0 leave them
            // unrotated.
            for (std::size_t i = 0; i < rope_angle.size(); ++i) {
                rope_cos[i] = std::cos(rope_angle[i]);
                rope_sin[i] = std::sin(rope_angle[i]);
            }
            rope_height = height;
            rope_width = width;
        }
        ggml_tensor *cos, *sin;
        if (strict_cuda) {
            // PyTorch computes cos/sin on the device with precise libdevice
            // sinf/cosf.
            ggml_tensor *angle = input_tensor(run, ctx, GGML_TYPE_F32, dh,
                                              tokens, 1, 1, rope_angle.data());
            cos = ggml_cos(ctx, angle);
            sin = ggml_sin(ctx, angle);
        } else {
            cos = input_tensor(run, ctx, GGML_TYPE_F32, dh, tokens, 1, 1,
                               rope_cos.data());
            sin = input_tensor(run, ctx, GGML_TYPE_F32, dh, tokens, 1, 1,
                               rope_sin.data());
        }
        for (std::size_t layer = 0; layer < depth; ++layer) {
            const std::string pfx =
                    "ra.backbone.model.model.layer." + std::to_string(layer);
            ggml_tensor *y = norm(ctx, x, t(pfx + ".norm1.weight"),
                                  t(pfx + ".norm1.bias"), eps);
            if (trace && layer == 0) trace->emplace_back("backbone_norm1", y);
            const bool traced = trace && layer == 0;
            auto mark = [&](const char *name, ggml_tensor *value) {
                if (traced) trace->emplace_back(name, value);
                return value;
            };
            auto *qkv = linear(ctx, y, packed_qkv.weights[layer],
                               packed_qkv.biases[layer]);
            auto *q = mark("backbone_q",
                           ggml_cont(ctx, ggml_view_2d(ctx, qkv, c, tokens,
                                                       qkv->nb[1], 0)));
            auto *k = mark(
                    "backbone_k",
                    ggml_cont(ctx, ggml_view_2d(ctx, qkv, c, tokens, qkv->nb[1],
                                                c * sizeof(float))));
            auto *v = mark(
                    "backbone_v",
                    ggml_cont(ctx, ggml_view_2d(ctx, qkv, c, tokens, qkv->nb[1],
                                                2 * c * sizeof(float))));
            q = ggml_permute(ctx, ggml_reshape_4d(ctx, q, dh, heads, tokens, 1),
                             0, 2, 1, 3);
            k = ggml_permute(ctx, ggml_reshape_4d(ctx, k, dh, heads, tokens, 1),
                             0, 2, 1, 3);
            v = ggml_permute(ctx, ggml_reshape_4d(ctx, v, dh, heads, tokens, 1),
                             0, 2, 1, 3);
            q = rope(ctx, q, cos, sin, dh);
            k = rope(ctx, k, cos, sin, dh);
            ggml_tensor *attn;
            if (options.flash_attention && !options.strict_f32 &&
                options.backend != "cpu") {
                attn = ggml_flash_attn_ext(
                        ctx, ggml_cont(ctx, q),
                        ggml_cast(ctx, k, GGML_TYPE_F16),
                        ggml_cast(ctx, v, GGML_TYPE_F16), nullptr,
                        1.0f / std::sqrt(static_cast<float>(dh)), 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
            } else {
                ggml_tensor *scores = matmul(ctx, k, q);
                ggml_tensor *probs = ggml_soft_max_ext(
                        ctx, scores, nullptr,
                        1.0f / std::sqrt(static_cast<float>(dh)), 0.0f);
                ggml_tensor *vt =
                        ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
                attn = matmul(ctx, vt, probs);
                attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
            }
            attn = mark("backbone_attn",
                        ggml_reshape_3d(ctx, attn, c, tokens, 1));
            attn = mark("backbone_o_proj",
                        linear(ctx, attn, t(pfx + ".attention.o_proj.weight"),
                               t(pfx + ".attention.o_proj.bias")));
            attn = mark(
                    "backbone_ls1",
                    ggml_mul(ctx, attn,
                             as_f32(ctx, t(pfx + ".layer_scale1.lambda1"))));
            x = ggml_add(ctx, x, attn);
            y = mark("backbone_norm2", norm(ctx, x, t(pfx + ".norm2.weight"),
                                            t(pfx + ".norm2.bias"), eps));
            ggml_tensor *mlp;
            if (swiglu) {
                auto *up = mark("backbone_up",
                                linear(ctx, y, t(pfx + ".mlp.up_proj.weight"),
                                       t(pfx + ".mlp.up_proj.bias")));
                mlp = ggml_mul(
                        ctx,
                        activation(
                                ctx,
                                linear(ctx, y, t(pfx + ".mlp.gate_proj.weight"),
                                       t(pfx + ".mlp.gate_proj.bias")),
                                true),
                        up);
            } else {
                mlp = activation(
                        ctx,
                        mark("backbone_up",
                             linear(ctx, y, t(pfx + ".mlp.up_proj.weight"),
                                    t(pfx + ".mlp.up_proj.bias"))),
                        false);
            }
            mlp = mark("backbone_down", linear(ctx, mark("backbone_act", mlp),
                                               t(pfx + ".mlp.down_proj.weight"),
                                               t(pfx + ".mlp.down_proj.bias")));
            x = ggml_add(
                    ctx, x,
                    mark("backbone_ls2",
                         ggml_mul(ctx, mlp,
                                  as_f32(ctx,
                                         t(pfx + ".layer_scale2.lambda1")))));
            if (trace && layer == 0) trace->emplace_back("backbone_block0", x);
            if (layer == 6 || layer == 9 || layer + 1 == depth)
                taps.push_back(ggml_norm(
                        ctx,
                        ggml_cont(ctx,
                                  ggml_view_2d(
                                          ctx, x, c, patch_tokens, x->nb[1],
                                          static_cast<std::size_t>(regs + 1) *
                                                  x->nb[1])),
                        1e-5f));
        }
        if (strict_cuda) {
            // Official F.softmax(layer_weights) runs on the device; its expf
            // and warp-sum order differ from host std::exp by 1 ulp on some
            // checkpoints.
            auto *w = ggml_soft_max(
                    ctx, as_f32(ctx, t("ra.backbone.layer_weights")));
            ggml_tensor *fused =
                    ggml_mul(ctx, taps[0], ggml_view_1d(ctx, w, 1, 0));
            for (std::size_t i = 1; i < taps.size(); ++i)
                fused = ggml_add(
                        ctx, fused,
                        ggml_mul(ctx, taps[i],
                                 ggml_view_1d(ctx, w, 1, i * sizeof(float))));
            return fused;
        }
        auto weights = tap_weights;
        const float mx = *std::max_element(weights.begin(), weights.end());
        float sum = 0;
        for (auto &w : weights) {
            w = std::exp(w - mx);
            sum += w;
        }
        ggml_tensor *fused = ggml_scale(ctx, taps[0], weights[0] / sum);
        for (std::size_t i = 1; i < taps.size(); ++i)
            fused = ggml_add(ctx, fused,
                             ggml_scale(ctx, taps[i], weights[i] / sum));
        return fused;
    }

    ImageOutput infer(const ImageInput &input) {
        if (input.width != 448 || input.height != 448)
            throw std::runtime_error(
                    "full graph currently requires 448x448 input");
        if (input.object_count > options.max_boxes)
            throw std::runtime_error(
                    "untrimmed input exceeds Options.max_boxes");
        if (input.rgb_chw.size() !=
                    product(3, input.width * input.height, "image") ||
            input.boxes_cxcywh.size() != input.object_count * 4)
            throw std::runtime_error("invalid RAIM input buffers");
        for (float value : input.rgb_chw)
            if (!std::isfinite(value))
                throw std::runtime_error("image has nonfinite values");
        for (float value : input.boxes_cxcywh)
            if (!std::isfinite(value))
                throw std::runtime_error("boxes have nonfinite values");
        if ((!input.coverage.empty() &&
             input.coverage.size() != input.object_count * 1024) ||
            (!input.fill.empty() &&
             (input.coverage.empty() ||
              input.fill.size() != input.object_count)) ||
            (!input.box_scores.empty() &&
             input.box_scores.size() != input.object_count))
            throw std::runtime_error("invalid optional region buffers");
        for (const auto *buffer :
             {&input.coverage, &input.fill, &input.box_scores})
            for (float value : *buffer)
                if (!std::isfinite(value) || value < 0 || value > 1)
                    throw std::runtime_error(
                            "region coverage, fill and scores must be finite "
                            "in [0,1]");
        for (std::size_t i = 0; i < input.object_count; ++i)
            if (input.boxes_cxcywh[4 * i + 2] < 0 ||
                input.boxes_cxcywh[4 * i + 3] < 0)
                throw std::runtime_error("boxes have negative width or height");
        if (input.object_count < 2) {
            ImageOutput out;
            out.predicate_count = predicates;
            out.decomposed = options.decompose;
            return out;
        }
        const auto begin = std::chrono::steady_clock::now();
        const std::size_t n = input.object_count;
        const int h = input.height / patch, w = input.width / patch, hw = h * w;
        std::vector<float> cov_grid;
        if (!input.coverage.empty()) {
            cov_grid.resize(n * hw);
            // Adaptive average pooling uses overlapping floor/ceil bins.
            for (std::size_t i = 0; i < n; ++i)
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x) {
                        const int y0 = y * 32 / h,
                                  y1 = ((y + 1) * 32 + h - 1) / h;
                        const int x0 = x * 32 / w,
                                  x1 = ((x + 1) * 32 + w - 1) / w;
                        float sum = 0;
                        for (int py = y0; py < y1; ++py)
                            for (int px = x0; px < x1; ++px)
                                sum += input.coverage[i * 1024 + py * 32 + px];
                        cov_grid[i * hw + y * w + x] =
                                sum / ((y1 - y0) * (x1 - x0));
                    }
        }
        constexpr int output_slots = 128;
        std::vector<float> patch_data, object_features, geo_scores, rel_scores;
        {
            auto run = make_run();
            auto *image =
                    input_tensor(run, run.ctx, GGML_TYPE_F32, input.width,
                                 input.height, 3, 1, input.rgb_chw.data());
            std::vector<ggml_tensor *> taps;
            std::vector<std::pair<std::string, ggml_tensor *>> traced;
            auto *fmap = backbone(
                    run, run.ctx, image, taps, input.height, input.width,
                    options.dump_directory.empty() ? nullptr : &traced);
            auto *objects =
                    pool(run, fmap, input.boxes_cxcywh, h, w, cov_grid,
                         options.dump_directory.empty() ? nullptr : &traced);
            std::vector<GeometryTerms> all_geo;
            for (std::size_t s = 0; s < n; ++s)
                for (std::size_t o = 0; o < n; ++o)
                    all_geo.push_back(
                            geometry_terms(input.boxes_cxcywh.data() + 4 * s,
                                           input.boxes_cxcywh.data() + 4 * o));
            auto *gf = geometry_features(run, all_geo);
            if (!options.dump_directory.empty())
                traced.emplace_back("geo_feats", gf);
            auto geometry_affine = [&](ggml_tensor *x,
                                       const std::string &name) {
                if (options.backend != "vulkan" || options.matmul_geometry)
                    return affine(run.ctx, x, name);
                // Preserve F32 products in this short TopK-controlling MLP.
                auto *weight = as_f32(run.ctx, t("ra." + name + ".weight"));
                const int64_t inner = weight->ne[0], width = weight->ne[1],
                              count = x->ne[1];
                auto *weights = ggml_repeat_4d(
                        run.ctx,
                        ggml_reshape_3d(run.ctx, weight, inner, width, 1),
                        inner, width, count, 1);
                auto *inputs = ggml_repeat_4d(
                        run.ctx, ggml_reshape_3d(run.ctx, x, inner, 1, count),
                        inner, width, count, 1);
                auto *result = ggml_reshape_2d(
                        run.ctx,
                        ggml_sum_rows(run.ctx,
                                      ggml_mul(run.ctx, weights, inputs)),
                        width, count);
                return ggml_add(run.ctx, result,
                                as_f32(run.ctx, t("ra." + name + ".bias")));
            };
            auto *gs = geometry_affine(
                    ggml_relu(run.ctx,
                              geometry_affine(gf, "sampler.geo_scorer.0")),
                    "sampler.geo_scorer.2");
            auto scorer = [&](const std::string &p, const std::string &trace) {
                auto *norm = ln(run.ctx, objects, p + ".0");
                auto *hidden = affine(run.ctx, norm, p + ".1");
                auto *activated = ggml_gelu_erf(run.ctx, hidden);
                auto *result = affine(run.ctx, activated, p + ".3");
                if (!options.dump_directory.empty()) {
                    traced.emplace_back(trace + "_0", norm);
                    traced.emplace_back(trace + "_1", hidden);
                    traced.emplace_back(trace + "_2", activated);
                    traced.emplace_back(trace + "_3", result);
                }
                return result;
            };
            auto *zs = scorer("sampler.f_sub", "sampler_f_sub");
            auto *zo = scorer("sampler.f_obj", "sampler_f_obj");
            // [object,subject] storage is flattened as subject*N + object.
            ggml_tensor *rs;
            if (options.matmul_sampler) {
                rs = matmul(run.ctx, zo, zs);
            } else {
                // Upstream rounds each product before reduction; a GEMM can
                // fuse multiply/add and change candidates near the TopK cutoff.
                const int64_t dim = zs->ne[0];
                auto *subjects = ggml_repeat_4d(
                        run.ctx, ggml_reshape_3d(run.ctx, zs, dim, 1, n), dim,
                        n, n, 1);
                auto *objects = ggml_repeat_4d(run.ctx, zo, dim, n, n, 1);
                auto *products = ggml_mul(run.ctx, subjects, objects);
                auto *sums = ggml_sum_rows(run.ctx, products);
                if (strict_cuda()) {
                    ggml_sum_rows_set_hint(sums, GGML_HINT_STRICT_F32);
                }
                rs = ggml_reshape_2d(run.ctx, sums, n, n);
            }
            rs = ggml_scale(run.ctx, rs,
                            1.0f / std::sqrt(static_cast<float>(zs->ne[0])));
            std::vector<std::pair<std::string, ggml_tensor *>> requested = {
                    {"fused_patch_map", fmap},
                    {"spatial_pool_0", objects},
                    {"geo_scores", gs},
                    {"rel_scores", rs}};
            requested.insert(requested.end(), traced.begin(), traced.end());
            auto values = outputs(run, requested);
            patch_data = std::move(values[0]);
            object_features = std::move(values[1]);
            geo_scores = std::move(values[2]);
            rel_scores = std::move(values[3]);
        }
        std::vector<int> candidates(n * n);
        std::iota(candidates.begin(), candidates.end(), 0);
        auto valid_pair = [&](int i) { return i / n != i % n; };
        auto sort_by = [&](const std::vector<float> &scores) {
            std::stable_sort(candidates.begin(), candidates.end(),
                             [&](int a, int b) {
                                 if (valid_pair(a) != valid_pair(b))
                                     return valid_pair(a);
                                 return valid_pair(a) && scores[a] > scores[b];
                             });
        };
        sort_by(geo_scores);
        candidates.resize(std::min<std::size_t>(400, candidates.size()));
        sort_by(rel_scores);
        candidates.resize(
                std::min<std::size_t>(output_slots, candidates.size()));
        ImageOutput out;
        out.pair_count = output_slots;
        out.predicate_count = predicates;
        out.subject_indices.assign(output_slots, 0);
        out.object_indices.assign(output_slots, 0);
        out.valid_mask.assign(output_slots, 0);
        out.pair_logits.assign(output_slots,
                               -std::numeric_limits<float>::infinity());
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            const int ix = candidates[i];
            out.subject_indices[i] = ix / n;
            out.object_indices[i] = ix % n;
            out.valid_mask[i] = valid_pair(ix);
            out.pair_logits[i] = rel_scores[ix];
        }
        // Invalid rows are excluded as attention keys in every relation block.
        // Removing their queries preserves valid-row semantics and the wire
        // format.
        const int K = compact_pairs() ? std::count(out.valid_mask.begin(),
                                                   out.valid_mask.end(), 1)
                                      : output_slots;
        const std::vector<uint8_t> valid(out.valid_mask.begin(),
                                         out.valid_mask.begin() + K);
        std::vector<float> sub(K * 4), obj(K * 4), paired_boxes(K * 8),
                regions(K * 8), anchors(K * 16);
        std::vector<GeometryTerms> geof(K);
        std::vector<float> region_cov(cov_grid.empty() ? 0 : 2 * K * hw,
                                      1.0f - 1e-4f);
        for (int i = 0; i < K; ++i) {
            const float *s =
                    input.boxes_cxcywh.data() + 4 * out.subject_indices[i];
            const float *o =
                    input.boxes_cxcywh.data() + 4 * out.object_indices[i];
            std::copy(s, s + 4, sub.begin() + 4 * i);
            std::copy(o, o + 4, obj.begin() + 4 * i);
            std::copy(s, s + 4, paired_boxes.begin() + 8 * i);
            std::copy(o, o + 4, paired_boxes.begin() + 8 * i + 4);
            float region_features[4] = {1, 1, 0, 0};
            if (!cov_grid.empty()) {
                const auto si = out.subject_indices[i],
                           oi = out.object_indices[i];
                const auto *cs = input.coverage.data() + si * 1024;
                const auto *co = input.coverage.data() + oi * 1024;
                float area_s = 0, area_o = 0, intersection = 0;
                for (int j = 0; j < 1024; ++j) {
                    area_s += cs[j];
                    area_o += co[j];
                    intersection += cs[j] * co[j];
                }
                region_features[0] = input.fill.empty() ? 1 : input.fill[si];
                region_features[1] = input.fill.empty() ? 1 : input.fill[oi];
                region_features[2] =
                        intersection /
                        std::max(area_s + area_o - intersection, 1e-6f);
                region_features[3] = intersection /
                                     std::max(std::min(area_s, area_o), 1e-6f);
                for (int j = 0; j < hw; ++j)
                    region_cov[i * hw + j] = std::max(cov_grid[si * hw + j],
                                                      cov_grid[oi * hw + j]);
            }
            geof[i] = geometry_terms(
                    s, o, cov_grid.empty() ? nullptr : region_features);
            float u[4], contact[4];
            const auto sb = xyxy(s), ob = xyxy(o);
            for (int axis = 0; axis < 2; ++axis) {
                const float lo = std::min(s[axis] - .5f * s[axis + 2],
                                          o[axis] - .5f * o[axis + 2]);
                const float hi = std::max(s[axis] + .5f * s[axis + 2],
                                          o[axis] + .5f * o[axis + 2]);
                u[axis] = .5f * (lo + hi);
                u[axis + 2] = hi - lo;
                const float inner1 = std::max(sb[axis], ob[axis]),
                            inner2 = std::min(sb[axis + 2], ob[axis + 2]);
                contact[axis] = .5f * (inner1 + inner2);
                contact[axis + 2] = std::fabs(inner2 - inner1);
            }
            std::copy(u, u + 4, regions.begin() + 4 * i);
            std::copy(contact, contact + 4, regions.begin() + 4 * (K + i));
            std::copy(s, s + 4, anchors.begin() + 16 * i);
            std::copy(o, o + 4, anchors.begin() + 16 * i + 4);
            std::copy(u, u + 4, anchors.begin() + 16 * i + 8);
            std::copy(contact, contact + 4, anchors.begin() + 16 * i + 12);
        }
        dump("subject_indices", std::vector<float>(out.subject_indices.begin(),
                                                   out.subject_indices.end()));
        dump("object_indices", std::vector<float>(out.object_indices.begin(),
                                                  out.object_indices.end()));
        dump("valid_mask",
             std::vector<float>(out.valid_mask.begin(), out.valid_mask.end()));
        dump("pair_logits", out.pair_logits);
        {
            auto run = make_run();
            auto *ctx = run.ctx;
            auto *fmap =
                    options.retain_features
                            ? retained_features.tensors.at("fused_patch_map")
                            : input_tensor(run, ctx, GGML_TYPE_F32, c, hw, 1, 1,
                                           patch_data.data());
            auto *objects = input_tensor(run, ctx, GGML_TYPE_F32, c, n, 1, 1,
                                         object_features.data());
            auto *si = input_tensor(run, ctx, GGML_TYPE_I32, K, 1, 1, 1,
                                    out.subject_indices.data());
            auto *oi = input_tensor(run, ctx, GGML_TYPE_I32, K, 1, 1, 1,
                                    out.object_indices.data());
            auto *vs = ggml_get_rows(ctx, objects, si),
                 *vo = ggml_get_rows(ctx, objects, oi);
            auto *gp = geometry_features(run, geof);
            auto *geo_h0 = affine(ctx, gp, "geo_encoder.mlp.0");
            auto *geo_h1 = ggml_gelu_erf(ctx, geo_h0);
            auto *geo_h2 = affine(ctx, geo_h1, "geo_encoder.mlp.2");
            auto *geo = ln(ctx, geo_h2, "geo_encoder.mlp.3");
            // encode_pairs stacks [sub TL, sub BR, obj TL, obj BR] per pair:
            // [d, 4, K].
            ggml_tensor *bt = nullptr;
            for (auto *group :
                 corners(run, paired_boxes, "box_prompt_encoder", 2)) {
                auto *token = ggml_reshape_3d(ctx, group, d, 1, K);
                bt = bt ? ggml_concat(ctx, bt, token, 1) : token;
            }
            bt = ggml_reshape_2d(ctx, bt, d, 4 * K);
            auto *pooled = pool(run, fmap, regions, h, w, region_cov);
            auto *vu = ggml_view_2d(ctx, pooled, c, K, pooled->nb[1], 0);
            auto *vc = ggml_view_2d(ctx, pooled, c, K, pooled->nb[1],
                                    K * pooled->nb[1]);
            auto *pair = ggml_concat(
                    ctx,
                    ggml_concat(ctx,
                                ggml_concat(ctx, ggml_concat(ctx, vs, vo, 0),
                                            vu, 0),
                                vc, 0),
                    geo, 0);
            pair = affine(ctx, pair, "pair_proj");
            auto *scene = affine(ctx, fmap, "rel_transformer.scene_proj");
            auto *memory = ggml_concat(
                    ctx,
                    ggml_add(ctx, scene,
                             scene_pe(run, "rel_transformer.scene_pe", h, w)),
                    bt, 1);
            auto *pm = key_mask(run, valid, K);
            std::vector<uint8_t> mv(hw, 1);
            for (int i = 0; i < K; ++i)
                for (int j = 0; j < 4; ++j) mv.push_back(valid[i]);
            auto *mm = key_mask(run, mv, K);
            auto *rel = encoder(ctx, pair, "rel_transformer.self_layers.0", pm);
            auto *self0 = rel;
            rel = encoder(ctx, rel, "rel_transformer.self_layers.1", pm);
            auto *self1 = rel;
            rel = decoder(ctx, rel, memory, "rel_transformer.cross_layers.0",
                          pm, mm);
            auto *cross0 = rel;
            rel = decoder(ctx, rel, memory, "rel_transformer.last_cross", pm,
                          mm, true);
            auto *qn = ln(ctx, rel, "deformable_read.norm");
            auto *offsets = affine(ctx, qn, "deformable_read.offset_mlp");
            auto *weights = affine(ctx, qn, "deformable_read.weight_mlp");
            // The PyTorch view is [K, H, A*(P+S)].  GGML stores the same
            // buffer as [A*(P+S), H, K]; softmax over ne0 therefore performs
            // the per-head/per-pair normalization without a host transpose.
            weights =
                    ggml_soft_max(ctx, ggml_reshape_3d(ctx, weights, 24, 8, K));
            auto *transformer = rel;
            auto *dr =
                    deformable(run, scene, offsets, weights, anchors, h, w, K);
            auto *deform = ggml_add(
                    ctx, rel,
                    ggml_mul(ctx, affine(ctx, dr, "deformable_read.out_proj"),
                             as_f32(ctx, t("ra.deformable_read.gamma"))));
            rel = encoder(ctx, deform, "rel_interaction.dep_layers.0", pm);
            rel = encoder(ctx, rel, "rel_interaction.dep_layers.1", pm);
            memory = ggml_add(
                    ctx, affine(ctx, fmap, "rel_interaction.scene_proj", false),
                    scene_pe(run, "rel_interaction.scene_pe", h, w));
            mv = valid;
            mv.resize(K + hw, 1);
            rel = encoder(ctx, ggml_concat(ctx, rel, memory, 1),
                          "rel_interaction.gnd_layers.0",
                          key_mask(run, mv, K + hw));
            rel = ggml_cont(ctx, ggml_view_2d(ctx, rel, d, K, rel->nb[1], 0));
            auto *q = affine(ctx,
                             ln(ctx,
                                ggml_gelu_erf(ctx, affine(ctx, rel,
                                                          "vocab_head.proj.0")),
                                "vocab_head.proj.2"),
                             "vocab_head.proj.3", false);
            const auto &gates = compose_gates;
            q = ln(ctx,
                   ggml_add(ctx,
                            ggml_add(ctx, q,
                                     ggml_scale(ctx,
                                                affine(ctx, vs, "sub_text_proj",
                                                       false),
                                                gates[0])),
                            ggml_scale(ctx,
                                       affine(ctx, vo, "obj_text_proj", false),
                                       gates[1])),
                   "compose_norm");
            auto *qs = affine(
                    ctx,
                    ln(ctx,
                       ggml_gelu_erf(ctx,
                                     affine(ctx, ggml_concat(ctx, rel, geo, 0),
                                            "spa_proj.0")),
                       "spa_proj.2"),
                    "spa_proj.3", false);
            auto *bank =
                    vocabulary.empty()
                            ? t("ra.vocab_head.W")
                            : input_tensor(run, ctx, GGML_TYPE_F32,
                                           meta_u32("ra.text_dim"), predicates,
                                           1, 1, vocabulary.data());
            auto *semantic_query = ggml_l2_norm(ctx, q, 1e-12f);
            auto *spatial_query = ggml_l2_norm(ctx, qs, 1e-12f);
            ggml_tensor *sem, *spa;
            // PyTorch issues two independent [K, text_dim] @ W.T GEMMs.
            // Combining them changes cuBLAS kernel selection and therefore F32
            // roundoff.
            if (options.fused_vocabulary && !strict_cuda()) {
                auto *projected = matmul(
                        ctx, bank,
                        ggml_concat(ctx, semantic_query, spatial_query, 1));
                sem = ggml_view_2d(ctx, projected, predicates, K,
                                   projected->nb[1], 0);
                spa = ggml_view_2d(ctx, projected, predicates, K,
                                   projected->nb[1], K * projected->nb[1]);
            } else {
                sem = matmul(ctx, bank, semantic_query);
                spa = matmul(ctx, bank, spatial_query);
            }
            auto *av =
                    vocabulary.empty()
                            ? as_f32(ctx, t("ra.vocab_head.alpha"))
                            : ggml_reshape_1d(
                                      ctx,
                                      ggml_sigmoid(
                                              ctx,
                                              affine(ctx,
                                                     ggml_gelu_erf(
                                                             ctx,
                                                             affine(ctx, bank,
                                                                    "vocab_"
                                                                    "head.gate_"
                                                                    "mlp.0")),
                                                     "vocab_head.gate_mlp.2")),
                                      predicates);
            auto *ia = ggml_scale_bias(ctx, av, -1, 1);
            auto *logits = ggml_add(ctx, ggml_mul(ctx, sem, ia),
                                    ggml_mul(ctx, spa, av));
            auto *scale = ggml_clamp(
                    ctx,
                    ggml_exp(ctx, as_f32(ctx, t("ra.vocab_head.logit_scale"))),
                    0, 100);
            logits = ggml_add(ctx, ggml_mul(ctx, logits, scale),
                              as_f32(ctx, t("ra.vocab_head.logit_bias")));
            auto *best_id =
                    ggml_cast(ctx, ggml_argmax(ctx, logits), GGML_TYPE_F32);
            std::vector<float> row_offsets(K);
            for (int i = 0; i < K; ++i)
                row_offsets[i] = static_cast<float>(i * predicates);
            auto *base = input_tensor(run, ctx, GGML_TYPE_F32, K, 1, 1, 1,
                                      row_offsets.data());
            auto *indices =
                    ggml_cast(ctx, ggml_add(ctx, best_id, base), GGML_TYPE_I32);
            auto *best_logit = ggml_get_rows(
                    ctx, ggml_reshape_2d(ctx, logits, 1, K * predicates),
                    indices);
            std::vector<std::pair<std::string, ggml_tensor *>> requested = {
                    {"best_predicate", best_id}, {"best_logit", best_logit}};
            if (options.decompose)
                requested.emplace_back("predicate_alpha", av);
            if (options.raw_logits || !options.dump_directory.empty())
                requested.emplace_back("predicate_logits", logits);
            else if (options.decompose)
                requested.emplace_back("predicate_logits", logits);
            if (!options.dump_directory.empty()) {
                requested.insert(requested.end(),
                                 {{"deformable_read", deform},
                                  {"rel_interaction", rel},
                                  {"rel_self0", self0},
                                  {"rel_self1", self1},
                                  {"rel_cross0", cross0},
                                  {"vocab_query", q},
                                  {"spa_proj", qs},
                                  {"vocab_semantic_query", semantic_query},
                                  {"vocab_spatial_query", spatial_query},
                                  {"vocab_semantic", sem},
                                  {"vocab_spatial", spa},
                                  {"rel_transformer", transformer},
                                  {"scene_projected", scene},
                                  {"deformable_offsets", offsets},
                                  {"deformable_weights", weights},
                                  {"geo_encoder", geo},
                                  {"vsub", vs},
                                  {"vobj", vo},
                                  {"box_prompt_tokens", bt},
                                  {"spatial_pool_1", pooled},
                                  {"pair_proj", pair},
                                  {"deformable_sample", dr},
                                  {"geo_feats_k", gp},
                                  {"geo_h0", geo_h0},
                                  {"geo_h1", geo_h1},
                                  {"geo_h2", geo_h2}});
            }
            auto values = outputs(run, requested);
            for (int i = 0; i < K; ++i)
                if (out.valid_mask[i]) {
                    const float z = calibration_a * (values[1][i] +
                                                     out.pair_logits[i]) +
                                    calibration_b;
                    out.relations.push_back(
                            {static_cast<std::size_t>(out.subject_indices[i]),
                             static_cast<std::size_t>(out.object_indices[i]),
                             static_cast<std::size_t>(values[0][i]),
                             (1.0f / (1.0f + std::exp(-std::clamp(z, -80.0f,
                                                                  80.0f)))) *
                                     (input.box_scores.empty()
                                              ? 1.0f
                                              : input.box_scores[out.subject_indices
                                                                         [i]] *
                                                        input.box_scores
                                                                [out.object_indices
                                                                         [i]])});
                }
            std::stable_sort(out.relations.begin(), out.relations.end(),
                             [](const Relation &a, const Relation &b) {
                                 return a.score > b.score;
                             });
            if (options.decompose) {
                out.decomposed = true;
                const auto &alpha = values[2];
                const auto &all_logits = values[3];
                out.predicate_spatial.assign(predicates, 0);
                for (std::size_t j = 0; j < predicates; ++j) {
                    const auto known = corpus_types.find(predicate_names[j]);
                    out.predicate_spatial[j] =
                            (known == corpus_types.end() ? alpha[j] >= .5f
                                                         : known->second)
                                    ? 1
                                    : 0;
                }
                for (bool spatial : {true, false}) {
                    std::vector<std::pair<float, Relation>> ranked;
                    for (int i = 0; i < K; ++i)
                        if (out.valid_mask[i]) {
                            float best =
                                    -std::numeric_limits<float>::infinity();
                            std::size_t pred = 0;
                            for (std::size_t j = 0; j < predicates; ++j) {
                                const auto known =
                                        corpus_types.find(predicate_names[j]);
                                const bool is_spatial =
                                        known == corpus_types.end()
                                                ? alpha[j] >= .5f
                                                : known->second;
                                const float fused =
                                        all_logits[i * predicates + j] +
                                        out.pair_logits[i];
                                if (is_spatial == spatial && fused > best) {
                                    best = fused;
                                    pred = j;
                                }
                            }
                            if (std::isfinite(best))
                                ranked.push_back(
                                        {best,
                                         {static_cast<std::size_t>(
                                                  out.subject_indices[i]),
                                          static_cast<std::size_t>(
                                                  out.object_indices[i]),
                                          pred,
                                          1.0f / (1.0f + std::exp(-best))}});
                        }
                    std::stable_sort(ranked.begin(), ranked.end(),
                                     [](const auto &a, const auto &b) {
                                         return a.first > b.first;
                                     });
                    auto &stream = spatial ? out.spatial_relations
                                           : out.semantic_relations;
                    for (std::size_t j = 0;
                         j < std::min(options.topk, ranked.size()); ++j) {
                        auto r = ranked[j].second;
                        if (!input.box_scores.empty())
                            r.score *= input.box_scores[r.subject] *
                                       input.box_scores[r.object];
                        stream.push_back(r);
                    }
                }
            }
            out.relations.resize(std::min(options.topk, out.relations.size()));
            if (options.raw_logits || !options.dump_directory.empty()) {
                out.predicate_logits =
                        std::move(values[options.decompose ? 3 : 2]);
                out.predicate_logits.resize(output_slots * predicates, 0.0f);
            }
        }
        out.milliseconds = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - begin)
                                   .count();
        out.backend_buffer_bytes = backend_buffer_bytes;
        return out;
    }
};

FullModel::FullModel(const ModelOptions &options)
    : impl_(std::make_unique<Impl>(options)) {}
FullModel::~FullModel() = default;
struct PredicateEncoder::Impl : TensorLoader {
    const int dim, length, depth, heads, output_dim;
    clip_bpe_tokenizer tokenizer;
    QRegularExpression regex;
    std::vector<std::string> templates;

    static ModelOptions student_options(const ModelOptions &options) {
        if (options.text_student.empty() || options.tokenizer.empty())
            throw std::runtime_error(
                    "text encoding requires student and tokenizer paths");
        auto opts = options;
        opts.model = options.text_student;
        return opts;
    }

    explicit Impl(const ModelOptions &options)
        : TensorLoader(student_options(options),
                       "relateanything_text_student_v1"),
          dim(meta_u32("ra.dim")),
          length(meta_u32("ra.max_len")),
          depth(meta_u32("ra.depth")),
          heads(meta_u32("ra.heads")),
          output_dim(meta_u32("ra.out_dim")) {
        if (dim != 256 || length != 32 || depth != 6 || heads != 4 ||
            output_dim != 512)
            throw std::runtime_error("unsupported text student dimensions");
        Json::Value config;
        std::ifstream token_file(options.tokenizer);
        if (!token_file || !(token_file >> config))
            throw std::runtime_error("cannot read tokenizer.json");
        const auto &normalizers = config["normalizer"]["normalizers"];
        const auto &splitters = config["pre_tokenizer"]["pretokenizers"];
        const std::string clip_pattern =
                "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|"
                "["
                "\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+";
        if (config["model"]["type"].asString() != "BPE" ||
            config["model"]["end_of_word_suffix"].asString() != "</w>" ||
            config["normalizer"]["type"].asString() != "Sequence" ||
            normalizers.size() != 3 ||
            normalizers[0]["type"].asString() != "NFC" ||
            normalizers[1]["type"].asString() != "Replace" ||
            normalizers[1]["pattern"]["Regex"].asString() != "\\s+" ||
            normalizers[1]["content"].asString() != " " ||
            normalizers[2]["type"].asString() != "Lowercase" ||
            config["pre_tokenizer"]["type"].asString() != "Sequence" ||
            splitters.size() != 2 ||
            splitters[0]["type"].asString() != "Split" ||
            splitters[0]["pattern"]["Regex"].asString() != clip_pattern ||
            !splitters[0]["invert"].asBool() ||
            splitters[1]["type"].asString() != "ByteLevel" ||
            splitters[1]["add_prefix_space"].asBool() ||
            config["post_processor"]["cls"][1].asInt() != 49406 ||
            config["post_processor"]["sep"][1].asInt() != 49407)
            throw std::runtime_error(
                    "unsupported tokenizer contract; expected official CLIP "
                    "BPE "
                    "configuration");
        for (const auto &name : config["model"]["vocab"].getMemberNames())
            tokenizer.encoder.emplace(name,
                                      config["model"]["vocab"][name].asInt());
        int rank = 0;
        for (const auto &merge : config["model"]["merges"]) {
            std::string a, b;
            if (merge.isArray() && merge.size() == 2) {
                a = merge[0].asString();
                b = merge[1].asString();
            } else if (merge.isString()) {
                std::istringstream row(merge.asString());
                row >> a >> b;
            } else
                throw std::runtime_error("invalid tokenizer merge");
            tokenizer.merge_ranks.emplace(clip_merge_key(a, b), rank++);
        }
        if (tokenizer.encoder.size() != 49408)
            throw std::runtime_error("invalid CLIP vocabulary");
        clip_init_byte_encoder(tokenizer.byte_encoder);
        regex = QRegularExpression(
                QString::fromStdString(clip_pattern),
                QRegularExpression::UseUnicodePropertiesOption);
        if (!regex.isValid())
            throw std::runtime_error("cannot initialize Unicode tokenizer");
        templates = options.text_templates;
        if (templates.empty())
            templates = {"{p}", "one object is {p} another object",
                         "a photo of something {p} something"};
    }

    std::vector<float> encode(const std::vector<std::string> &texts) {
        // Keep request-dependent strings bounded by the current call, while
        // the immutable encoder/merge tables and weights stay context-owned.
        tokenizer.cache.clear();
        auto &loader = *this;
        Json::Value token_rows(Json::arrayValue);
        std::vector<float> result(texts.size() * output_dim, 0);
        for (std::size_t row = 0; row < texts.size(); ++row)
            for (const auto &format : templates) {
                std::string text = format;
                auto slot = text.find("{p}");
                if (slot == std::string::npos)
                    throw std::runtime_error("text template requires {p}");
                while (slot != std::string::npos) {
                    text.replace(slot, 3, texts[row]);
                    slot = text.find("{p}", slot + texts[row].size());
                }
                // QString lowercasing, like tokenizers Lowercase, uses Unicode
                // mappings without the locale-dependent final-sigma rule.
                const QString normalized =
                        QString::fromUtf8(text.data(), int(text.size()))
                                .normalized(QString::NormalizationForm_C)
                                .toLower();
                auto matches = regex.globalMatch(normalized);
                std::vector<int32_t> ids{49406};
                while (matches.hasNext()) {
                    std::string word =
                            matches.next().captured().toUtf8().toStdString();
                    if (word == "<|startoftext|>" || word == "<|endoftext|>") {
                        ids.push_back(tokenizer.encoder.at(word));
                        continue;
                    }
                    std::string encoded;
                    for (uint8_t byte : word)
                        encoded += tokenizer.byte_encoder.at(byte);
                    std::istringstream pieces(
                            clip_bpe_encode(tokenizer, encoded));
                    while (pieces >> word)
                        ids.push_back(tokenizer.encoder.at(word));
                }
                ids.push_back(49407);
                if (ids.size() > static_cast<std::size_t>(length)) {
                    ids.resize(length);
                    ids.back() = 49407;
                }
                const int valid = ids.size();
                Json::Value token_row(Json::arrayValue);
                for (auto id : ids) token_row.append(id);
                token_rows.append(token_row);
                ids.resize(length, 0);
                auto run = make_run();
                auto *ctx = run.ctx;
                auto t = [&](const std::string &name) {
                    return loader.tensor("ts." + name);
                };
                auto affine = [&](ggml_tensor *x, const std::string &p,
                                  bool bias = true) {
                    return linear(ctx, x, t(p + ".weight"),
                                  bias ? t(p + ".bias") : nullptr);
                };
                auto ln = [&](ggml_tensor *x, const std::string &p) {
                    return norm(ctx, x, t(p + ".weight"), t(p + ".bias"),
                                1e-5f);
                };
                auto *tokens = input_tensor(run, ctx, GGML_TYPE_I32, length, 1,
                                            1, 1, ids.data());
                auto *x =
                        ggml_get_rows(ctx, t("token_embedding.weight"), tokens);
                x = affine(x, "tok_proj", false);
                x = ggml_add(ctx, x, t("positional"));
                std::vector<float> padding(length * length, 0), keep(length, 0);
                for (int k = 0; k < length; ++k) {
                    keep[k] = k < valid ? 1.0f / valid : 0;
                    if (k >= valid)
                        for (int q = 0; q < length; ++q)
                            padding[q * length + k] =
                                    -std::numeric_limits<float>::infinity();
                }
                auto *mask = input_tensor(run, ctx, GGML_TYPE_F32, length,
                                          length, 1, 1, padding.data());
                for (int block = 0; block < depth; ++block) {
                    const auto p = "blocks." + std::to_string(block);
                    auto *y = ln(x, p + ".norm1");
                    y = packed_attention(ctx, y, y, y,
                                         t(p + ".attn.in_proj_weight"),
                                         t(p + ".attn.in_proj_bias"),
                                         t(p + ".attn.out_proj.weight"),
                                         t(p + ".attn.out_proj.bias"), dim,
                                         heads, mask, false);
                    x = ggml_add(ctx, x, y);
                    y = affine(ggml_gelu_erf(ctx, affine(ln(x, p + ".norm2"),
                                                         p + ".ffn.0")),
                               p + ".ffn.2");
                    x = ggml_add(ctx, x, y);
                }
                x = ln(x, "ln_final");
                auto *valid_weights = input_tensor(
                        run, ctx, GGML_TYPE_F32, length, 1, 1, 1, keep.data());
                auto *pooled =
                        matmul(ctx, ggml_cont(ctx, ggml_transpose(ctx, x)),
                               valid_weights);
                auto *embedding = ggml_l2_norm(
                        ctx, affine(pooled, "head", false), 1e-12f);
                run.graph = ggml_new_graph_custom(ctx, 8192, false);
                ggml_set_output(embedding);
                ggml_build_forward_expand(run.graph, embedding);
                for (int i = 0; i < ggml_graph_n_nodes(run.graph); ++i) {
                    auto *node = ggml_graph_node(run.graph, i);
                    if (std::strncmp(node->name, "relateanything_", 15) != 0)
                        ggml_format_name(node, "relateanything_%s",
                                         ggml_op_name(node->op));
                }
                if (loader.options.strict_f32)
                    for (int i = 0; i < ggml_graph_n_nodes(run.graph); ++i) {
                        auto *node = ggml_graph_node(run.graph, i);
                        if (node->op == GGML_OP_SOFT_MAX)
                            ggml_soft_max_set_hint(node, GGML_HINT_STRICT_F32);
                        if (node->op == GGML_OP_MUL_MAT)
                            ggml_mul_mat_set_hint(node, GGML_HINT_STRICT_F32);
                    }
                ggml_backend_sched_reset(loader.backend->scheduler);
                for (const auto &input : run.inputs)
                    ggml_backend_sched_set_tensor_backend(
                            loader.backend->scheduler, input.first,
                            loader.backend->accelerator);
                if (!ggml_backend_sched_alloc_graph(loader.backend->scheduler,
                                                    run.graph))
                    throw std::runtime_error("cannot allocate text graph");
                for (const auto &input : run.inputs)
                    ggml_backend_tensor_set(input.first, input.second.data(), 0,
                                            input.second.size());
                if (ggml_backend_sched_graph_compute(loader.backend->scheduler,
                                                     run.graph) !=
                    GGML_STATUS_SUCCESS)
                    throw std::runtime_error("text graph failed");
                ggml_backend_sched_synchronize(loader.backend->scheduler);
                auto values = read_f32(embedding);
                for (int j = 0; j < output_dim; ++j)
                    result[row * output_dim + j] += values[j];
                ggml_backend_sched_reset(loader.backend->scheduler);
            }
        for (std::size_t row = 0; row < texts.size(); ++row) {
            float sum = 0;
            for (int j = 0; j < output_dim; ++j)
                sum += result[row * output_dim + j] *
                       result[row * output_dim + j];
            const float denominator = std::max(std::sqrt(sum), 1e-12f);
            for (int j = 0; j < output_dim; ++j)
                result[row * output_dim + j] /= denominator;
        }
        if (!options.token_output.empty()) {
            std::ofstream file(options.token_output);
            file << token_rows;
            if (!file) throw std::runtime_error("cannot write token IDs");
        }
        return result;
    }
};

PredicateEncoder::PredicateEncoder(const ModelOptions &options)
    : impl_(std::make_unique<Impl>(options)) {}
PredicateEncoder::~PredicateEncoder() = default;
std::vector<float> PredicateEncoder::encode(
        const std::vector<std::string> &texts) {
    return impl_->encode(texts);
}
std::vector<float> encode_predicates(const ModelOptions &options,
                                     const std::vector<std::string> &texts) {
    return PredicateEncoder(options).encode(texts);
}

ImageOutput FullModel::infer(const ImageInput &input) {
    if (input.object_count <= impl_->options.max_boxes)
        return impl_->infer(input);
    const auto n = input.object_count;
    if (n > 10000 || input.boxes_cxcywh.size() != 4 * n ||
        (!input.box_scores.empty() && input.box_scores.size() != n) ||
        (!input.coverage.empty() && input.coverage.size() != 1024 * n) ||
        (!input.fill.empty() && input.fill.size() != n))
        throw std::runtime_error("invalid input for box selection");
    for (float value : input.box_scores)
        if (!std::isfinite(value) || value < 0 || value > 1)
            throw std::runtime_error("box scores must be finite in [0,1]");
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    if (!input.box_scores.empty())
        std::stable_sort(order.begin(), order.end(),
                         [&](std::size_t a, std::size_t b) {
                             return input.box_scores[a] > input.box_scores[b];
                         });
    order.resize(impl_->options.max_boxes);
    ImageInput selected = input;
    selected.object_count = order.size();
    auto gather = [&](const std::vector<float> &source,
                      std::vector<float> &dest, std::size_t width) {
        dest.clear();
        if (!source.empty())
            for (auto i : order)
                dest.insert(dest.end(), source.begin() + i * width,
                            source.begin() + (i + 1) * width);
    };
    gather(input.boxes_cxcywh, selected.boxes_cxcywh, 4);
    gather(input.coverage, selected.coverage, 1024);
    gather(input.fill, selected.fill, 1);
    gather(input.box_scores, selected.box_scores, 1);
    return impl_->infer(selected);
}
bool FullModel::strict_f32() const { return impl_->options.strict_f32; }
void FullModel::calibration(float &a, float &b) const {
    a = impl_->calibration_a;
    b = impl_->calibration_b;
}
const std::vector<std::string> &FullModel::predicates() const {
    return impl_->predicate_names;
}

bool is_image_input(const std::filesystem::path &path) {
    return path.extension() == ".raim" || path.extension() == ".raip";
}

ImageInput read_image_input(const std::filesystem::path &path) {
    if (path.extension() == ".raip") return read_image_request(path);
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open RAIM input: " + path.string());
    char magic[sizeof(kImageMagic)]{};
    read_exact(stream, magic, sizeof(magic));
    const bool extended = std::memcmp(magic, kImageMagic2, sizeof(magic)) == 0;
    if (!extended && std::memcmp(magic, kImageMagic, sizeof(magic)) != 0)
        throw std::runtime_error("invalid RAIM magic");
    std::uint32_t width = 0, height = 0, objects = 0;
    read_exact(stream, &width);
    read_exact(stream, &height);
    read_exact(stream, &objects);
    if (width != 448 || height != 448 || objects > 10000)
        throw std::runtime_error("invalid RAIM dimensions");
    std::uint32_t flags = 0;
    if (extended) read_exact(stream, &flags);
    if (flags & ~7u || ((flags & 2u) && !(flags & 1u)))
        throw std::runtime_error("invalid RAIM region flags");
    const std::uintmax_t expected =
            20 + (extended ? 4 : 0) +
            (std::uintmax_t(width) * height * 3 +
             std::uintmax_t(objects) *
                     (4 + ((flags & 1u) ? 1024 : 0) + ((flags & 2u) ? 1 : 0) +
                      ((flags & 4u) ? 1 : 0))) *
                    4;
    if (std::filesystem::file_size(path) != expected)
        throw std::runtime_error("invalid RAIM byte count");
    ImageInput input;
    input.width = width;
    input.height = height;
    input.object_count = objects;
    input.rgb_chw.resize(
            product(3, product(width, height, "RAIM image"), "RAIM image"));
    input.boxes_cxcywh.resize(product(objects, std::size_t(4), "RAIM boxes"));
    read_exact(stream, input.rgb_chw.data(), input.rgb_chw.size());
    read_exact(stream, input.boxes_cxcywh.data(), input.boxes_cxcywh.size());
    if (flags & 1u) {
        input.coverage.resize(objects * 1024);
        read_exact(stream, input.coverage.data(), input.coverage.size());
    }
    if (flags & 2u) {
        input.fill.resize(objects);
        read_exact(stream, input.fill.data(), input.fill.size());
    }
    if (flags & 4u) {
        input.box_scores.resize(objects);
        read_exact(stream, input.box_scores.data(), input.box_scores.size());
    }
    return input;
}

void write_image_output(const std::filesystem::path &path,
                        const ImageOutput &output) {
    if (output.predicate_logits.size() !=
                output.pair_count * output.predicate_count ||
        output.valid_mask.size() != output.pair_count)
        throw std::runtime_error("invalid full graph output");
    std::ofstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot create output: " + path.string());
    stream.write(kImageOutputMagic, sizeof(kImageOutputMagic));
    const auto p = static_cast<std::uint32_t>(output.pair_count),
               v = static_cast<std::uint32_t>(output.predicate_count);
    stream.write(reinterpret_cast<const char *>(&p), sizeof(p));
    stream.write(reinterpret_cast<const char *>(&v), sizeof(v));
    stream.write(reinterpret_cast<const char *>(output.valid_mask.data()),
                 static_cast<std::streamsize>(output.valid_mask.size() *
                                              sizeof(std::int32_t)));
    stream.write(reinterpret_cast<const char *>(output.subject_indices.data()),
                 static_cast<std::streamsize>(output.subject_indices.size() *
                                              sizeof(std::int32_t)));
    stream.write(reinterpret_cast<const char *>(output.object_indices.data()),
                 static_cast<std::streamsize>(output.object_indices.size() *
                                              sizeof(std::int32_t)));
    stream.write(reinterpret_cast<const char *>(output.pair_logits.data()),
                 static_cast<std::streamsize>(output.pair_logits.size() *
                                              sizeof(float)));
    stream.write(reinterpret_cast<const char *>(output.predicate_logits.data()),
                 static_cast<std::streamsize>(output.predicate_logits.size() *
                                              sizeof(float)));
    if (!stream) throw std::runtime_error("failed writing full graph output");
}

void write_image_input(const std::filesystem::path &path,
                       const ImageInput &input) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot write RAIM: " + path.string());
    const std::uint32_t flags = (!input.coverage.empty() ? 1u : 0u) |
                                (!input.fill.empty() ? 2u : 0u) |
                                (!input.box_scores.empty() ? 4u : 0u);
    stream.write(flags ? kImageMagic2 : kImageMagic, sizeof(kImageMagic));
    const std::uint32_t dims[] = {
            static_cast<std::uint32_t>(input.width),
            static_cast<std::uint32_t>(input.height),
            static_cast<std::uint32_t>(input.object_count)};
    stream.write(reinterpret_cast<const char *>(dims), sizeof(dims));
    if (flags)
        stream.write(reinterpret_cast<const char *>(&flags), sizeof(flags));
    stream.write(reinterpret_cast<const char *>(input.rgb_chw.data()),
                 input.rgb_chw.size() * sizeof(float));
    stream.write(reinterpret_cast<const char *>(input.boxes_cxcywh.data()),
                 input.boxes_cxcywh.size() * sizeof(float));
    for (const auto *buffer : {&input.coverage, &input.fill, &input.box_scores})
        stream.write(reinterpret_cast<const char *>(buffer->data()),
                     buffer->size() * sizeof(float));
    if (!stream) throw std::runtime_error("failed writing RAIM");
}

}  // namespace relateanything
