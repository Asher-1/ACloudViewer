// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ggml_gpu_ops.hpp"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <functional>
#include <string>
#include <unordered_map>

#include "tensor_ops.hpp"

namespace lightglue::aliked_internal {
namespace {

constexpr int64_t kMaxGraphNodes = 512;
constexpr float kSeluScale = 1.050700987f;
constexpr float kSeluAlpha = 1.67326324f;

struct CachedOneInputGraph {
    ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_tensor *in = nullptr;
    ggml_tensor *out = nullptr;
    int32_t w = 0;
    int32_t h = 0;
    int32_t c = 0;
};

struct CachedUnaryInPlaceGraph {
    ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_tensor *in = nullptr;
    ggml_tensor *out = nullptr;
    int32_t w = 0;
    int32_t h = 0;
    int32_t c = 0;
};

std::unordered_map<std::string, CachedOneInputGraph> g_one_input_graphs;
std::unordered_map<std::string, CachedUnaryInPlaceGraph> g_unary_inplace_graphs;

ggml_tensor *GgmlSelu(ggml_context *ctx, ggml_tensor *x) {
    ggml_tensor *pos = ggml_scale(ctx, ggml_relu(ctx, x), kSeluScale);
    ggml_tensor *neg =
            ggml_scale(ctx, ggml_expm1(ctx, x), kSeluScale * kSeluAlpha);
    ggml_tensor *step = ggml_step(ctx, x);
    ggml_tensor *ones =
            ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, x->ne);
    ones = ggml_fill(ctx, ones, 1.0f);
    ggml_tensor *neg_mask = ggml_sub(ctx, ones, step);
    return ggml_add(ctx, ggml_mul(ctx, pos, step),
                    ggml_mul(ctx, neg, neg_mask));
}

ggml_tensor *NewInputLike(ggml_context *ctx, const GpuTensor &tensor) {
    return ggml_new_tensor_4d(ctx, GGML_TYPE_F32, tensor.w, tensor.h, tensor.c,
                              1);
}

bool RunGraphCopyOut(internal::Backend *backend,
                     ggml_context *ctx,
                     ggml_cgraph *graph,
                     ggml_tensor *out,
                     GpuTensor *output,
                     std::string *error) {
    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(ctx, backend->handle);
    if (buffer == nullptr) {
        if (error) {
            *error = "failed to allocate ggml op buffer";
        }
        return false;
    }

    if (!ggml_gallocr_alloc_graph(backend->allocator, graph)) {
        if (error) {
            *error = "failed to allocate ggml op graph";
        }
        ggml_backend_buffer_free(buffer);
        return false;
    }

    if (ggml_backend_graph_compute(backend->handle, graph) !=
        GGML_STATUS_SUCCESS) {
        if (error) {
            *error = "ggml op graph compute failed";
        }
        ggml_backend_buffer_free(buffer);
        return false;
    }

    if (!GpuTensor::Allocate(backend, static_cast<int32_t>(out->ne[0]),
                             static_cast<int32_t>(out->ne[1]),
                             static_cast<int32_t>(out->ne[2]), output, error)) {
        ggml_backend_buffer_free(buffer);
        return false;
    }

    ggml_backend_tensor_copy(out, output->tensor);
    ggml_backend_buffer_free(buffer);
    return true;
}

bool RunUnaryInPlaceOnGpuTensor(
        internal::Backend *backend,
        GpuTensor *tensor,
        const std::function<ggml_tensor *(ggml_context *, ggml_tensor *)> &op,
        std::string *error,
        const char *cache_key = nullptr) {
    if (cache_key != nullptr && cache_key[0] != '\0') {
        CachedUnaryInPlaceGraph &entry = g_unary_inplace_graphs[cache_key];
        if (entry.graph == nullptr || entry.w != tensor->w ||
            entry.h != tensor->h || entry.c != tensor->c) {
            if (entry.buffer != nullptr) {
                ggml_backend_buffer_free(entry.buffer);
            }
            if (entry.ctx != nullptr) {
                ggml_free(entry.ctx);
            }
            entry = CachedUnaryInPlaceGraph{};

            const size_t graph_overhead =
                    ggml_graph_overhead_custom(kMaxGraphNodes, false);
            const size_t ctx_size =
                    graph_overhead + ggml_tensor_overhead() * 16 + 1024 * 1024;
            ggml_init_params params{ctx_size, nullptr, true};
            entry.ctx = ggml_init(params);
            if (entry.ctx == nullptr) {
                if (error) {
                    *error = "failed to create cached unary in-place context";
                }
                return false;
            }
            entry.in = NewInputLike(entry.ctx, *tensor);
            entry.out = op(entry.ctx, entry.in);
            entry.graph =
                    ggml_new_graph_custom(entry.ctx, kMaxGraphNodes, false);
            ggml_build_forward_expand(entry.graph, entry.out);
            entry.buffer =
                    ggml_backend_alloc_ctx_tensors(entry.ctx, backend->handle);
            if (entry.buffer == nullptr ||
                !ggml_gallocr_alloc_graph(backend->allocator, entry.graph)) {
                if (error) {
                    *error = "failed to allocate cached unary in-place graph";
                }
                return false;
            }
            entry.w = tensor->w;
            entry.h = tensor->h;
            entry.c = tensor->c;
        }
        ggml_backend_tensor_copy(tensor->tensor, entry.in);
        if (ggml_backend_graph_compute(backend->handle, entry.graph) !=
            GGML_STATUS_SUCCESS) {
            if (error) {
                *error = "cached unary in-place compute failed";
            }
            return false;
        }
        ggml_backend_tensor_copy(entry.out, tensor->tensor);
        return true;
    }

    const size_t graph_overhead =
            ggml_graph_overhead_custom(kMaxGraphNodes, false);
    const size_t ctx_size =
            graph_overhead + ggml_tensor_overhead() * 16 + 1024 * 1024;
    ggml_init_params params{ctx_size, nullptr, true};
    ggml_context *ctx = ggml_init(params);
    if (ctx == nullptr) {
        if (error) {
            *error = "failed to create ggml unary in-place context";
        }
        return false;
    }

    ggml_tensor *input = NewInputLike(ctx, *tensor);
    ggml_tensor *out = op(ctx, input);
    ggml_cgraph *graph = ggml_new_graph_custom(ctx, kMaxGraphNodes, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(ctx, backend->handle);
    if (buffer == nullptr) {
        if (error) {
            *error = "failed to allocate ggml unary in-place buffer";
        }
        ggml_free(ctx);
        return false;
    }

    if (!ggml_gallocr_alloc_graph(backend->allocator, graph)) {
        if (error) {
            *error = "failed to allocate ggml unary in-place graph";
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_copy(tensor->tensor, input);
    if (ggml_backend_graph_compute(backend->handle, graph) !=
        GGML_STATUS_SUCCESS) {
        if (error) {
            *error = "ggml unary in-place compute failed";
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_copy(out, tensor->tensor);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

bool RunBinaryInPlaceOnGpuTensor(
        internal::Backend *backend,
        GpuTensor *accum,
        const GpuTensor &other,
        const std::function<ggml_tensor *(
                ggml_context *, ggml_tensor *, ggml_tensor *)> &op,
        std::string *error) {
    const size_t graph_overhead =
            ggml_graph_overhead_custom(kMaxGraphNodes, false);
    const size_t ctx_size =
            graph_overhead + ggml_tensor_overhead() * 16 + 1024 * 1024;
    ggml_init_params params{ctx_size, nullptr, true};
    ggml_context *ctx = ggml_init(params);
    if (ctx == nullptr) {
        if (error) {
            *error = "failed to create ggml binary in-place context";
        }
        return false;
    }

    ggml_tensor *lhs = NewInputLike(ctx, *accum);
    ggml_tensor *rhs = NewInputLike(ctx, other);
    ggml_tensor *out = op(ctx, lhs, rhs);
    ggml_cgraph *graph = ggml_new_graph_custom(ctx, kMaxGraphNodes, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(ctx, backend->handle);
    if (buffer == nullptr) {
        if (error) {
            *error = "failed to allocate ggml binary in-place buffer";
        }
        ggml_free(ctx);
        return false;
    }

    if (!ggml_gallocr_alloc_graph(backend->allocator, graph)) {
        if (error) {
            *error = "failed to allocate ggml binary in-place graph";
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_copy(accum->tensor, lhs);
    ggml_backend_tensor_copy(other.tensor, rhs);
    if (ggml_backend_graph_compute(backend->handle, graph) !=
        GGML_STATUS_SUCCESS) {
        if (error) {
            *error = "ggml binary in-place compute failed";
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_copy(out, accum->tensor);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

bool RunGraphWithInputs(
        internal::Backend *backend,
        const std::function<ggml_tensor *(
                ggml_context *, ggml_tensor *, ggml_tensor *)> &build,
        const GpuTensor &a,
        const GpuTensor &b,
        GpuTensor *output,
        std::string *error) {
    const size_t graph_overhead =
            ggml_graph_overhead_custom(kMaxGraphNodes, false);
    const size_t ctx_size =
            graph_overhead + ggml_tensor_overhead() * 16 + 1024 * 1024;
    ggml_init_params params{ctx_size, nullptr, true};
    ggml_context *ctx = ggml_init(params);
    if (ctx == nullptr) {
        if (error) {
            *error = "failed to create ggml two-input context";
        }
        return false;
    }

    ggml_tensor *in_a = NewInputLike(ctx, a);
    ggml_tensor *in_b = NewInputLike(ctx, b);
    ggml_tensor *out = build(ctx, in_a, in_b);
    ggml_cgraph *graph = ggml_new_graph_custom(ctx, kMaxGraphNodes, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(ctx, backend->handle);
    if (buffer == nullptr) {
        if (error) {
            *error = "failed to allocate ggml two-input buffer";
        }
        ggml_free(ctx);
        return false;
    }

    if (!ggml_gallocr_alloc_graph(backend->allocator, graph)) {
        if (error) {
            *error = "failed to allocate ggml two-input graph";
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_copy(a.tensor, in_a);
    ggml_backend_tensor_copy(b.tensor, in_b);
    if (ggml_backend_graph_compute(backend->handle, graph) !=
        GGML_STATUS_SUCCESS) {
        if (error) {
            *error = "ggml two-input compute failed";
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    if (!GpuTensor::Allocate(backend, static_cast<int32_t>(out->ne[0]),
                             static_cast<int32_t>(out->ne[1]),
                             static_cast<int32_t>(out->ne[2]), output, error)) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_copy(out, output->tensor);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

bool RunGraphWithInput(internal::Backend *backend,
                       const std::function<ggml_tensor *(ggml_context *,
                                                         ggml_tensor *)> &build,
                       const GpuTensor &input,
                       GpuTensor *output,
                       std::string *error,
                       const char *cache_key = nullptr) {
    if (cache_key != nullptr && cache_key[0] != '\0') {
        CachedOneInputGraph &entry = g_one_input_graphs[cache_key];
        if (entry.graph == nullptr || entry.w != input.w ||
            entry.h != input.h || entry.c != input.c) {
            if (entry.buffer != nullptr) {
                ggml_backend_buffer_free(entry.buffer);
            }
            if (entry.ctx != nullptr) {
                ggml_free(entry.ctx);
            }
            entry = CachedOneInputGraph{};

            const size_t graph_overhead =
                    ggml_graph_overhead_custom(kMaxGraphNodes, false);
            const size_t ctx_size =
                    graph_overhead + ggml_tensor_overhead() * 16 + 1024 * 1024;
            ggml_init_params params{ctx_size, nullptr, true};
            entry.ctx = ggml_init(params);
            if (entry.ctx == nullptr) {
                if (error) {
                    *error = "failed to create cached one-input context";
                }
                return false;
            }
            entry.in = NewInputLike(entry.ctx, input);
            entry.out = build(entry.ctx, entry.in);
            entry.graph =
                    ggml_new_graph_custom(entry.ctx, kMaxGraphNodes, false);
            ggml_build_forward_expand(entry.graph, entry.out);
            entry.buffer =
                    ggml_backend_alloc_ctx_tensors(entry.ctx, backend->handle);
            if (entry.buffer == nullptr ||
                !ggml_gallocr_alloc_graph(backend->allocator, entry.graph)) {
                if (error) {
                    *error = "failed to allocate cached one-input graph";
                }
                return false;
            }
            entry.w = input.w;
            entry.h = input.h;
            entry.c = input.c;
        }

        if (!GpuTensor::Allocate(
                    backend, static_cast<int32_t>(entry.out->ne[0]),
                    static_cast<int32_t>(entry.out->ne[1]),
                    static_cast<int32_t>(entry.out->ne[2]), output, error)) {
            return false;
        }
        ggml_backend_tensor_copy(input.tensor, entry.in);
        if (ggml_backend_graph_compute(backend->handle, entry.graph) !=
            GGML_STATUS_SUCCESS) {
            if (error) {
                *error = "cached one-input compute failed";
            }
            return false;
        }
        ggml_backend_tensor_copy(entry.out, output->tensor);
        return true;
    }

    const size_t graph_overhead =
            ggml_graph_overhead_custom(kMaxGraphNodes, false);
    const size_t ctx_size =
            graph_overhead + ggml_tensor_overhead() * 16 + 1024 * 1024;
    ggml_init_params params{ctx_size, nullptr, true};
    ggml_context *ctx = ggml_init(params);
    if (ctx == nullptr) {
        if (error) {
            *error = "failed to create ggml one-input context";
        }
        return false;
    }

    ggml_tensor *in = NewInputLike(ctx, input);
    ggml_tensor *out = build(ctx, in);
    ggml_cgraph *graph = ggml_new_graph_custom(ctx, kMaxGraphNodes, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(ctx, backend->handle);
    if (buffer == nullptr) {
        if (error) {
            *error = "failed to allocate ggml one-input buffer";
        }
        ggml_free(ctx);
        return false;
    }

    if (!ggml_gallocr_alloc_graph(backend->allocator, graph)) {
        if (error) {
            *error = "failed to allocate ggml one-input graph";
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_copy(input.tensor, in);
    if (ggml_backend_graph_compute(backend->handle, graph) !=
        GGML_STATUS_SUCCESS) {
        if (error) {
            *error = "ggml one-input compute failed";
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    if (!GpuTensor::Allocate(backend, static_cast<int32_t>(out->ne[0]),
                             static_cast<int32_t>(out->ne[1]),
                             static_cast<int32_t>(out->ne[2]), output, error)) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_copy(out, output->tensor);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

}  // namespace

bool RunAvgPool2dGpu(internal::Backend *backend,
                     const GpuTensor &input,
                     int32_t kh,
                     int32_t kw,
                     int32_t stride,
                     GpuTensor *output,
                     std::string *error) {
    if (input.tensor == nullptr) {
        if (error) {
            *error = "avg pool input is null";
        }
        return false;
    }

    const int32_t oh = (input.h - kh) / stride + 1;
    const int32_t ow = (input.w - kw) / stride + 1;
    if (oh <= 0 || ow <= 0) {
        if (error) {
            *error = "invalid avg pool output shape";
        }
        return false;
    }

    return RunGraphWithInput(
            backend,
            [kh, kw, stride](ggml_context *ctx, ggml_tensor *in) {
                return ggml_pool_2d(ctx, in, GGML_OP_POOL_AVG, kw, kh, stride,
                                    stride, 0, 0);
            },
            input, output, error,
            (std::string("avgpool_") + std::to_string(kh) + "x" +
             std::to_string(kw) + "_s" + std::to_string(stride))
                    .c_str());
}

bool RunInterpolateGpu(internal::Backend *backend,
                       const GpuTensor &input,
                       int32_t out_w,
                       int32_t out_h,
                       GpuTensor *output,
                       std::string *error) {
    if (input.tensor == nullptr) {
        if (error) {
            *error = "interpolate input is null";
        }
        return false;
    }

    const int32_t channels = input.c;
    const std::string key = "interp_" + std::to_string(out_w) + "x" +
                            std::to_string(out_h) + "_c" +
                            std::to_string(channels);
    return RunGraphWithInput(
            backend,
            [out_w, out_h, channels](ggml_context *ctx, ggml_tensor *in) {
                return ggml_interpolate(ctx, in, out_w, out_h, channels, 1,
                                        GGML_SCALE_MODE_BILINEAR);
            },
            input, output, error, key.c_str());
}

bool RunSeluGpu(internal::Backend *backend,
                GpuTensor *tensor,
                std::string *error) {
    if (tensor == nullptr || tensor->tensor == nullptr) {
        if (error) {
            *error = "SELU input tensor is null";
        }
        return false;
    }
    return RunUnaryInPlaceOnGpuTensor(
            backend, tensor,
            [](ggml_context *ctx, ggml_tensor *x) { return GgmlSelu(ctx, x); },
            error, "selu_inplace");
}

bool RunConvBnSeluGpu(GgmlConvRunner *runner,
                      internal::Backend *backend,
                      const FusedConv2d &weights,
                      const GpuTensor &input,
                      int32_t pad,
                      int32_t stride,
                      GpuTensor *output,
                      const char *cache_key,
                      std::string *error) {
    if (runner == nullptr || input.tensor == nullptr) {
        if (error) {
            *error = "invalid conv+SELU input";
        }
        return false;
    }
    if (cache_key == nullptr || cache_key[0] == '\0') {
        if (error) {
            *error = "conv+SELU requires cache key";
        }
        return false;
    }
    if (!runner->EnsureCachedPublic(cache_key, weights, error)) {
        return false;
    }

    const int32_t oh = (input.h + 2 * pad - weights.kh) / stride + 1;
    const int32_t ow = (input.w + 2 * pad - weights.kw) / stride + 1;
    if (oh <= 0 || ow <= 0) {
        if (error) {
            *error = "invalid conv+SELU output shape";
        }
        return false;
    }

    const GgmlConvRunner::CachedWeight &cached = runner->CachedEntry(cache_key);
    const size_t graph_overhead =
            ggml_graph_overhead_custom(kMaxGraphNodes, false);
    const size_t ctx_size =
            graph_overhead + ggml_tensor_overhead() * 32 + 4 * 1024 * 1024;
    ggml_init_params params{ctx_size, nullptr, true};
    ggml_context *ctx = ggml_init(params);
    if (ctx == nullptr) {
        if (error) {
            *error = "failed to create conv+SELU context";
        }
        return false;
    }

    ggml_tensor *in = NewInputLike(ctx, input);
    ggml_tensor *conv = ggml_conv_2d_direct(ctx, cached.kernel, in, stride,
                                            stride, pad, pad, 1, 1);
    ggml_tensor *added = ggml_add(ctx, conv, cached.bias);
    ggml_tensor *out = GgmlSelu(ctx, added);
    ggml_cgraph *graph = ggml_new_graph_custom(ctx, kMaxGraphNodes, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(ctx, backend->handle);
    if (buffer == nullptr) {
        if (error) {
            *error = "failed to allocate conv+SELU buffer";
        }
        ggml_free(ctx);
        return false;
    }

    if (!ggml_gallocr_alloc_graph(backend->allocator, graph)) {
        if (error) {
            *error = "failed to allocate conv+SELU graph";
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_copy(input.tensor, in);
    if (ggml_backend_graph_compute(backend->handle, graph) !=
        GGML_STATUS_SUCCESS) {
        if (error) {
            *error = "conv+SELU compute failed";
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    if (!GpuTensor::Allocate(backend, static_cast<int32_t>(out->ne[0]),
                             static_cast<int32_t>(out->ne[1]),
                             static_cast<int32_t>(out->ne[2]), output, error)) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_copy(out, output->tensor);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

bool RunAddGpu(internal::Backend *backend,
               GpuTensor *accum,
               const GpuTensor &other,
               std::string *error) {
    if (accum == nullptr || accum->tensor == nullptr ||
        other.tensor == nullptr) {
        if (error) {
            *error = "add inputs are null";
        }
        return false;
    }
    return RunBinaryInPlaceOnGpuTensor(
            backend, accum, other,
            [](ggml_context *ctx, ggml_tensor *a, ggml_tensor *b) {
                return ggml_add(ctx, a, b);
            },
            error);
}

bool RunConcatChannelGpu(internal::Backend *backend,
                         const GpuTensor &a,
                         const GpuTensor &b,
                         GpuTensor *output,
                         std::string *error) {
    if (a.tensor == nullptr || b.tensor == nullptr) {
        if (error) {
            *error = "concat inputs are null";
        }
        return false;
    }
    if (a.h != b.h || a.w != b.w) {
        if (error) {
            *error = "concat spatial mismatch";
        }
        return false;
    }

    return RunGraphWithInputs(
            backend,
            [](ggml_context *ctx, ggml_tensor *left, ggml_tensor *right) {
                return ggml_concat(ctx, left, right, 2);
            },
            a, b, output, error);
}

bool RunClampGpu(internal::Backend *backend,
                 GpuTensor *tensor,
                 float min_val,
                 float max_val,
                 std::string *error) {
    if (tensor == nullptr || tensor->tensor == nullptr) {
        if (error) {
            *error = "clamp input is null";
        }
        return false;
    }
    return RunUnaryInPlaceOnGpuTensor(
            backend, tensor,
            [=](ggml_context *ctx, ggml_tensor *x) {
                return ggml_clamp(ctx, x, min_val, max_val);
            },
            error);
}

bool RunSigmoidInPlaceGpu(internal::Backend *backend,
                          GpuTensor *tensor,
                          std::string *error) {
    if (tensor == nullptr || tensor->tensor == nullptr) {
        if (error) {
            *error = "sigmoid input is null";
        }
        return false;
    }
    return RunUnaryInPlaceOnGpuTensor(
            backend, tensor,
            [](ggml_context *ctx, ggml_tensor *x) {
                return ggml_sigmoid(ctx, x);
            },
            error);
}

bool RunL2NormalizeChannelsGpu(internal::Backend *backend,
                               GpuTensor *tensor,
                               int32_t channels,
                               int32_t h,
                               int32_t w,
                               std::string *error) {
    if (tensor == nullptr || tensor->tensor == nullptr) {
        if (error) {
            *error = "L2 normalize input is null";
        }
        return false;
    }

    std::vector<float> nchw;
    if (!tensor->DownloadNchw(backend, &nchw, channels, h, w, error)) {
        return false;
    }
    L2NormalizeChannels(&nchw, channels, h, w);
    return tensor->UploadNchw(backend, nchw, channels, h, w, error);
}

bool RunCropWhcnGpu(internal::Backend *backend,
                    const GpuTensor &input,
                    int32_t pad_top,
                    int32_t pad_left,
                    int32_t out_h,
                    int32_t out_w,
                    GpuTensor *output,
                    std::string *error) {
    if (input.tensor == nullptr) {
        if (error) {
            *error = "crop input is null";
        }
        return false;
    }

    std::vector<float> nchw;
    if (!input.DownloadNchw(backend, &nchw, input.c, input.h, input.w, error)) {
        return false;
    }

    std::vector<float> cropped(static_cast<size_t>(input.c) * out_h * out_w,
                               0.0f);
    for (int32_t ch = 0; ch < input.c; ++ch) {
        for (int32_t y = 0; y < out_h; ++y) {
            for (int32_t x = 0; x < out_w; ++x) {
                cropped[static_cast<size_t>(ch) * out_h * out_w + y * out_w +
                        x] = nchw[static_cast<size_t>(ch) * input.h * input.w +
                                  (y + pad_top) * input.w + (x + pad_left)];
            }
        }
    }

    if (!GpuTensor::Allocate(backend, out_w, out_h, input.c, output, error)) {
        return false;
    }
    return output->UploadNchw(backend, cropped, input.c, out_h, out_w, error);
}

}  // namespace lightglue::aliked_internal
