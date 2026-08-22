// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/aliked/ggml_gpu_ops.hpp"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

#include "tasks/aliked/gpu_pipeline_cache.hpp"
#include "tasks/aliked/gpu_sync.hpp"
#include "tasks/aliked/gpu_tensor.hpp"
#include "tasks/aliked/tensor_ops.hpp"
#include "tasks/aliked/vulkan/vulkan_aliked_dispatch.hpp"

namespace lightglue::aliked_internal {
namespace {

constexpr int64_t kMaxGraphNodes = 512;
constexpr float kSeluScale = 1.050700987f;
constexpr float kSeluAlpha = 1.67326324f;

struct CachedOneInputGraph {
    ggml_backend_t backend = nullptr;
    ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_tensor *in = nullptr;
    ggml_tensor *out = nullptr;
    int32_t w = 0;
    int32_t h = 0;
    int32_t c = 0;
};

struct CachedUnaryInPlaceGraph {
    ggml_backend_t backend = nullptr;
    ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_tensor *in = nullptr;
    ggml_tensor *out = nullptr;
    int32_t w = 0;
    int32_t h = 0;
    int32_t c = 0;
};

struct CachedBinaryInPlaceGraph {
    ggml_backend_t backend = nullptr;
    ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_tensor *lhs = nullptr;
    ggml_tensor *rhs = nullptr;
    ggml_tensor *out = nullptr;
    int32_t w = 0;
    int32_t h = 0;
    int32_t c = 0;
};

std::unordered_map<std::string, CachedOneInputGraph> g_one_input_graphs;
std::unordered_map<std::string, CachedUnaryInPlaceGraph> g_unary_inplace_graphs;
std::unordered_map<std::string, CachedBinaryInPlaceGraph>
        g_binary_inplace_graphs;
std::mutex g_gpu_op_cache_mutex;

std::string BackendCacheKey(const char *cache_key, internal::Backend *backend) {
    const auto handle = reinterpret_cast<std::uintptr_t>(
            backend != nullptr ? backend->handle : nullptr);
    return std::string(cache_key) + "@" + std::to_string(handle);
}

ggml_gallocr_t NewGraphGallocr(internal::Backend *backend) {
    if (backend == nullptr || backend->handle == nullptr) {
        return nullptr;
    }
    return ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend->handle));
}

void FreeGraphGallocr(ggml_gallocr_t *gallocr) {
    if (gallocr == nullptr || *gallocr == nullptr) {
        return;
    }
    ggml_gallocr_free(*gallocr);
    *gallocr = nullptr;
}

bool RunBoundGraphCompute(internal::Backend *backend,
                          ggml_cgraph *graph,
                          std::string *error) {
    if (backend == nullptr || backend->handle == nullptr || graph == nullptr) {
        if (error) {
            *error = "invalid backend or graph for bound compute";
        }
        return false;
    }
    if (ggml_backend_graph_compute(backend->handle, graph) !=
        GGML_STATUS_SUCCESS) {
        if (error) {
            *error = "cached ggml graph compute failed";
        }
        return false;
    }
    SyncGpuPipeline(backend);
    FlushGpuPipeline(backend);
    return true;
}

bool CachedGraphCompute(internal::Backend *backend,
                        ggml_cgraph *graph,
                        ggml_gallocr_t graph_gallocr,
                        std::string *error) {
    if (backend == nullptr || backend->handle == nullptr || graph == nullptr) {
        if (error) {
            *error = "invalid backend or graph for cached compute";
        }
        return false;
    }
    ggml_gallocr_t gallocr =
            graph_gallocr != nullptr ? graph_gallocr : backend->allocator;
    if (gallocr != nullptr && !ggml_gallocr_alloc_graph(gallocr, graph)) {
        if (error) {
            *error = "failed to bind cached ggml graph allocator";
        }
        return false;
    }
    return RunBoundGraphCompute(backend, graph, error);
}

bool EphemeralGallocrCompute(internal::Backend *backend,
                             ggml_cgraph *graph,
                             std::string *error) {
    if (backend == nullptr || backend->allocator == nullptr ||
        graph == nullptr) {
        if (error) {
            *error = "invalid backend or graph for ephemeral compute";
        }
        return false;
    }
    if (!ggml_gallocr_alloc_graph(backend->allocator, graph)) {
        if (error) {
            *error = "failed to bind ephemeral ggml graph allocator";
        }
        return false;
    }
    return CachedGraphCompute(backend, graph, nullptr, error);
}

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

void FinishEphemeralGpuCopy(internal::Backend *backend, bool hard_idle) {
    SyncGpuPipeline(backend);
    FlushGpuPipeline(backend);
#if defined(AICORE_VULKAN_ALIKED)
    if (hard_idle && backend != nullptr && backend->IsVulkan()) {
        VkAlikedQueueIdle(backend->handle);
    }
#endif
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

    if (!EphemeralGallocrCompute(backend, graph, error)) {
        ggml_backend_buffer_free(buffer);
        return false;
    }

    if (!GpuTensor::Allocate(backend, static_cast<int32_t>(out->ne[0]),
                             static_cast<int32_t>(out->ne[1]),
                             static_cast<int32_t>(out->ne[2]), output, error)) {
        ggml_backend_buffer_free(buffer);
        return false;
    }

    BackendTensorCopyCompat(backend, out, output->tensor);
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
        std::lock_guard<std::mutex> lock(g_gpu_op_cache_mutex);
        CachedUnaryInPlaceGraph &entry =
                g_unary_inplace_graphs[BackendCacheKey(cache_key, backend)];
        if (entry.graph == nullptr || entry.w != tensor->w ||
            entry.h != tensor->h || entry.c != tensor->c) {
            FreeGraphGallocr(&entry.gallocr);
            if (entry.buffer != nullptr) {
                ggml_backend_buffer_free(entry.buffer);
            }
            if (entry.ctx != nullptr) {
                ggml_free(entry.ctx);
            }
            entry = CachedUnaryInPlaceGraph{};
            entry.backend = backend->handle;

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
            entry.gallocr = NewGraphGallocr(backend);
            if (entry.buffer == nullptr || entry.gallocr == nullptr ||
                !ggml_gallocr_alloc_graph(entry.gallocr, entry.graph)) {
                if (error) {
                    *error = "failed to allocate cached unary in-place graph";
                }
                return false;
            }
            entry.w = tensor->w;
            entry.h = tensor->h;
            entry.c = tensor->c;
        }
        BackendTensorCopyCompat(backend, tensor->tensor, entry.in);
        if (!CachedGraphCompute(backend, entry.graph, entry.gallocr, error)) {
            return false;
        }
        BackendTensorCopyCompat(backend, entry.out, tensor->tensor);
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

    BackendTensorCopyCompat(backend, tensor->tensor, input);
    if (!EphemeralGallocrCompute(backend, graph, error)) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    BackendTensorCopyCompat(backend, out, tensor->tensor);
    FinishEphemeralGpuCopy(backend, true);
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
        std::string *error,
        const char *cache_key = nullptr) {
    if (cache_key != nullptr && cache_key[0] != '\0') {
        std::lock_guard<std::mutex> lock(g_gpu_op_cache_mutex);
        CachedBinaryInPlaceGraph &entry =
                g_binary_inplace_graphs[BackendCacheKey(cache_key, backend)];
        if (entry.graph == nullptr || entry.w != accum->w ||
            entry.h != accum->h || entry.c != accum->c) {
            FreeGraphGallocr(&entry.gallocr);
            if (entry.buffer != nullptr) {
                ggml_backend_buffer_free(entry.buffer);
            }
            if (entry.ctx != nullptr) {
                ggml_free(entry.ctx);
            }
            entry = CachedBinaryInPlaceGraph{};
            entry.backend = backend->handle;

            const size_t graph_overhead =
                    ggml_graph_overhead_custom(kMaxGraphNodes, false);
            const size_t ctx_size =
                    graph_overhead + ggml_tensor_overhead() * 16 + 1024 * 1024;
            ggml_init_params params{ctx_size, nullptr, true};
            entry.ctx = ggml_init(params);
            if (entry.ctx == nullptr) {
                if (error) {
                    *error = "failed to create cached binary in-place context";
                }
                return false;
            }
            entry.lhs = NewInputLike(entry.ctx, *accum);
            entry.rhs = NewInputLike(entry.ctx, other);
            entry.out = op(entry.ctx, entry.lhs, entry.rhs);
            entry.graph =
                    ggml_new_graph_custom(entry.ctx, kMaxGraphNodes, false);
            ggml_build_forward_expand(entry.graph, entry.out);
            entry.buffer =
                    ggml_backend_alloc_ctx_tensors(entry.ctx, backend->handle);
            entry.gallocr = NewGraphGallocr(backend);
            if (entry.buffer == nullptr || entry.gallocr == nullptr ||
                !ggml_gallocr_alloc_graph(entry.gallocr, entry.graph)) {
                if (error) {
                    *error = "failed to allocate cached binary in-place graph";
                }
                return false;
            }
            entry.w = accum->w;
            entry.h = accum->h;
            entry.c = accum->c;
        }
        // Bind graph slots before uploading inputs — re-alloc after copy zeros
        // lhs/rhs.
        if (!ggml_gallocr_alloc_graph(entry.gallocr, entry.graph)) {
            if (error) {
                *error = "failed to bind cached binary in-place graph";
            }
            return false;
        }
        BackendTensorCopyCompat(backend, accum->tensor, entry.lhs);
        BackendTensorCopyCompat(backend, other.tensor, entry.rhs);
        if (!RunBoundGraphCompute(backend, entry.graph, error)) {
            return false;
        }
        GpuTensor result;
        if (!GpuTensor::Allocate(backend, accum->w, accum->h, accum->c, &result,
                                 error)) {
            return false;
        }
        BackendTensorCopyCompat(backend, entry.out, result.tensor);
        FinishEphemeralGpuCopy(backend, true);
        *accum = std::move(result);
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

    BackendTensorCopyCompat(backend, accum->tensor, lhs);
    BackendTensorCopyCompat(backend, other.tensor, rhs);
    if (!EphemeralGallocrCompute(backend, graph, error)) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    BackendTensorCopyCompat(backend, out, accum->tensor);
    FinishEphemeralGpuCopy(backend, true);
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

    BackendTensorCopyCompat(backend, a.tensor, in_a);
    BackendTensorCopyCompat(backend, b.tensor, in_b);
    if (!EphemeralGallocrCompute(backend, graph, error)) {
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
        std::lock_guard<std::mutex> lock(g_gpu_op_cache_mutex);
        CachedOneInputGraph &entry =
                g_one_input_graphs[BackendCacheKey(cache_key, backend)];
        if (entry.graph == nullptr || entry.w != input.w ||
            entry.h != input.h || entry.c != input.c) {
            FreeGraphGallocr(&entry.gallocr);
            if (entry.buffer != nullptr) {
                ggml_backend_buffer_free(entry.buffer);
            }
            if (entry.ctx != nullptr) {
                ggml_free(entry.ctx);
            }
            entry = CachedOneInputGraph{};
            entry.backend = backend->handle;

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
            entry.gallocr = NewGraphGallocr(backend);
            if (entry.buffer == nullptr || entry.gallocr == nullptr ||
                !ggml_gallocr_alloc_graph(entry.gallocr, entry.graph)) {
                if (error) {
                    *error = "failed to allocate cached one-input graph";
                }
                return false;
            }
            entry.w = input.w;
            entry.h = input.h;
            entry.c = input.c;
        }

        GpuTensor tmp_output;
        GpuTensor *dst = (output == &input) ? &tmp_output : output;
        if (!GpuTensor::Allocate(
                    backend, static_cast<int32_t>(entry.out->ne[0]),
                    static_cast<int32_t>(entry.out->ne[1]),
                    static_cast<int32_t>(entry.out->ne[2]), dst, error)) {
            return false;
        }
        BackendTensorCopyCompat(backend, input.tensor, entry.in);
        if (!CachedGraphCompute(backend, entry.graph, entry.gallocr, error)) {
            return false;
        }
        BackendTensorCopyCompat(backend, entry.out, dst->tensor);
        if (dst != output) {
            *output = std::move(tmp_output);
        }
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

    BackendTensorCopyCompat(backend, input.tensor, in);
    if (!EphemeralGallocrCompute(backend, graph, error)) {
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

void ClearCachedGpuOpGraphs(internal::Backend *backend) {
    std::lock_guard<std::mutex> lock(g_gpu_op_cache_mutex);
    const ggml_backend_t handle =
            backend != nullptr ? backend->handle : nullptr;
    const auto clear = [handle](auto *entries) {
        for (auto it = entries->begin(); it != entries->end();) {
            auto &entry = it->second;
            if (handle != nullptr && entry.backend != handle) {
                ++it;
                continue;
            }
            FreeGraphGallocr(&entry.gallocr);
            if (entry.buffer != nullptr) {
                ggml_backend_buffer_free(entry.buffer);
            }
            if (entry.ctx != nullptr) {
                ggml_free(entry.ctx);
            }
            it = entries->erase(it);
        }
    };
    clear(&g_one_input_graphs);
    clear(&g_unary_inplace_graphs);
    clear(&g_binary_inplace_graphs);
}

void RebindAllCachedGgmlOpGraphs(internal::Backend *backend) {
    if (backend == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_gpu_op_cache_mutex);
    for (auto &entry : g_one_input_graphs) {
        if (entry.second.backend == backend->handle &&
            entry.second.graph != nullptr && entry.second.gallocr != nullptr) {
            ggml_gallocr_alloc_graph(entry.second.gallocr, entry.second.graph);
        }
    }
    for (auto &entry : g_unary_inplace_graphs) {
        if (entry.second.backend == backend->handle &&
            entry.second.graph != nullptr && entry.second.gallocr != nullptr) {
            ggml_gallocr_alloc_graph(entry.second.gallocr, entry.second.graph);
        }
    }
}

void BeginVulkanExtract(internal::Backend *backend) {
#if defined(AICORE_VULKAN_ALIKED)
    if (backend != nullptr && backend->IsVulkan()) {
        // Drop SELU/unary graphs from a prior ctx (e.g. CPU-then-Vulkan
        // parity).
        ClearCachedGpuOpGraphs(backend);
        VkAlikedQueueIdle(backend->handle);
        FlushGpuPipeline(backend);
    }
#else
    (void)backend;
#endif
}

void EndVulkanExtract(internal::Backend *backend, GpuPipelineCache *cache) {
#if defined(AICORE_VULKAN_ALIKED)
    if (backend == nullptr || !backend->IsVulkan()) {
        return;
    }
    VkAlikedQueueIdle(backend->handle);
    FlushGpuPipeline(backend);
    if (cache != nullptr) {
        cache->ggml()->runner()->InvalidateDeviceGraphs();
        cache->ComputeGgml()->runner()->InvalidateDeviceGraphs();
    }
    ClearCachedGpuOpGraphs(backend);
#else
    (void)backend;
    (void)cache;
#endif
}

void ResetVulkanExtractPipeline(internal::Backend *backend,
                                GpuPipelineCache *cache) {
    EndVulkanExtract(backend, cache);
}

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

    BackendTensorCopyCompat(backend, input.tensor, in);
    if (!EphemeralGallocrCompute(backend, graph, error)) {
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
               std::string *error,
               const char *cache_key) {
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
            error, cache_key);
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

    if (backend != nullptr && backend->IsCuda()) {
        SyncGpuTensorMeta(tensor);
        return RunUnaryInPlaceOnGpuTensor(
                backend, tensor,
                [](ggml_context *ctx, ggml_tensor *in) {
                    ggml_tensor *sq = ggml_sqr(ctx, in);
                    ggml_tensor *sq_p = ggml_permute(ctx, sq, 2, 0, 1, 3);
                    ggml_tensor *sum_c = ggml_sum_rows(ctx, sq_p);
                    ggml_tensor *clamped =
                            ggml_clamp(ctx, sum_c, 1e-12f, 1e30f);
                    ggml_tensor *norm = ggml_sqrt(ctx, clamped);
                    ggml_tensor *norm_bc = ggml_repeat(ctx, norm, in);
                    return ggml_div(ctx, in, norm_bc);
                },
                error, "l2norm_channels");
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

#if defined(AICORE_VULKAN_ALIKED)
    // The custom dense-copy shader is qualified for the small single-channel
    // score map only. Large multi-channel feature crops have intermittently
    // returned zero-filled regions on NVIDIA Vulkan drivers; use the host
    // round-trip below for that correctness-sensitive path.
    if (input.c == 1 && backend != nullptr && backend->IsVulkan() &&
        VkAlikedAvailable(backend->handle)) {
        if (!GpuTensor::Allocate(backend, out_w, out_h, input.c, output,
                                 error)) {
            return false;
        }

        // The custom dense-copy shader already handles arbitrary WHCN strides
        // and source offsets. A metadata-only tensor view avoids the Vulkan
        // ggml_view + cpy path, which can return zero-filled large crops.
        ggml_tensor cropped_view = *input.tensor;
        cropped_view.ne[0] = out_w;
        cropped_view.ne[1] = out_h;
        const size_t byte_offset =
                static_cast<size_t>(pad_left) * input.tensor->nb[0] +
                static_cast<size_t>(pad_top) * input.tensor->nb[1];
        cropped_view.data =
                static_cast<uint8_t *>(input.tensor->data) + byte_offset;
        if (VkAlikedDenseCopyWhcn(backend->handle, &cropped_view,
                                  output->tensor, out_w, out_h, input.c)) {
            VkAlikedQueueIdle(backend->handle);
            return true;
        }
        output->Release();
    }
#endif

    if (backend != nullptr && backend->IsGpu() &&
        !(backend->IsVulkan() && input.c > 1)) {
        const int32_t ic = input.c;
        const char *cache_key = "crop_whcn";
        return RunGraphWithInput(
                backend,
                [pad_top, pad_left, out_h, out_w, ic](ggml_context *ctx,
                                                      ggml_tensor *in) {
                    const size_t es = ggml_element_size(in);
                    const size_t offset = pad_left * es + pad_top * in->nb[1];
                    ggml_tensor *view = ggml_view_4d(ctx, in, out_w, out_h, ic,
                                                     1, in->nb[1], in->nb[2],
                                                     in->nb[3], offset);
                    ggml_tensor *out = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                          out_w, out_h, ic, 1);
                    return ggml_cpy(ctx, view, out);
                },
                input, output, error, cache_key);
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
