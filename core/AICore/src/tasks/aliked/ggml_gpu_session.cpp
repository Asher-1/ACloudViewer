// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/aliked/ggml_gpu_session.hpp"

#include "tasks/aliked/ggml_gpu_ops.hpp"
#include "tasks/aliked/gpu_sync.hpp"
#include "tasks/aliked/gpu_tensor.hpp"

#if defined(AICORE_VULKAN_ALIKED)
#include "tasks/aliked/vulkan/vulkan_aliked_dispatch.hpp"

#endif
#if defined(AICORE_CUDA_ALIKED)
#include "tasks/aliked/cuda/aliked_cuda.hpp"

#endif

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <cstring>
#include <vector>

namespace lightglue::aliked_internal {
namespace {

constexpr int64_t kMaxGraphNodes = 512;
constexpr float kSeluScale = 1.050700987f;
constexpr float kSeluAlpha = 1.67326324f;

void PinTensorToGpu(internal::Backend *backend, ggml_tensor *tensor) {
    if (backend == nullptr || !backend->HasSched() || tensor == nullptr) {
        return;
    }
    ggml_backend_sched_set_tensor_backend(backend->sched, tensor,
                                          backend->handle);
}

void PinSchedScoreHeadGraph(internal::Backend *backend,
                            ggml_cgraph *graph,
                            ggml_tensor *graph_in,
                            const std::vector<ggml_tensor *> &weights) {
    if (backend == nullptr || !backend->HasSched() || graph == nullptr) {
        return;
    }
    PinTensorToGpu(backend, graph_in);
    for (ggml_tensor *weight : weights) {
        PinTensorToGpu(backend, weight);
    }
    const int n_nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < n_nodes; ++i) {
        PinTensorToGpu(backend, ggml_graph_node(graph, i));
    }
}

float *DevPtr(const GpuTensor &tensor) {
    return reinterpret_cast<float *>(tensor.tensor->data);
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

}  // namespace

GgmlGpuSession::GgmlGpuSession(internal::Backend *backend)
    : backend_(backend), runner_(backend) {}

GgmlGpuSession::~GgmlGpuSession() = default;

bool GgmlGpuSession::RunConv(const FusedConv2d &weights,
                             const GpuTensor &input,
                             GpuTensor *output,
                             int32_t pad,
                             int32_t stride,
                             const char *cache_key,
                             std::string *error) {
    return runner_.RunDevice(weights, input, output, pad, stride, error,
                             cache_key);
}

bool GgmlGpuSession::RunSeluConvChain(const std::vector<SeluConvSpec> &layers,
                                      const GpuTensor &input,
                                      GpuTensor *output,
                                      std::string *error) {
    if (layers.empty()) {
        if (error) {
            *error = "empty SELU conv chain";
        }
        return false;
    }

    GpuTensor ping;
    GpuTensor pong;
    const GpuTensor *next_input = &input;

    for (size_t i = 0; i < layers.size(); ++i) {
        const SeluConvSpec &layer = layers[i];
        const bool is_last = i + 1 == layers.size();
        GpuTensor *dst = is_last ? output : ((i % 2 == 0) ? &ping : &pong);
        if (!runner_.RunDevice(layer.weights, *next_input, dst, layer.pad,
                               layer.stride, error, layer.cache_key)) {
            return false;
        }
        if (layer.apply_selu) {
            SyncGpuPipeline(backend_);
#if defined(AICORE_CUDA_ALIKED)
            if (backend_->IsCuda()) {
                if (!AlikedCudaApplySelu(backend_->handle, DevPtr(*dst),
                                         dst->ElementCount())) {
                    if (error) {
                        *error = "SELU CUDA failed in conv chain";
                    }
                    return false;
                }
            } else
#endif
                    if (!RunSeluGpu(backend_, dst, error)) {
                return false;
            }
            SyncGpuPipeline(backend_);
        }
        next_input = dst;
    }
    return true;
}

bool GgmlGpuSession::RunFusedConvChainGraph(
        const std::vector<ConvChainSpec> &layers,
        const GpuTensor &input,
        GpuTensor *output,
        std::string *error) {
    if (layers.empty() || input.tensor == nullptr) {
        if (error) {
            *error = "invalid fused conv chain input";
        }
        return false;
    }

    for (const ConvChainSpec &layer : layers) {
        if (layer.cache_key == nullptr || layer.cache_key[0] == '\0') {
            if (error) {
                *error = "fused conv chain layer missing cache key";
            }
            return false;
        }
        if (!runner_.EnsureCachedPublic(layer.cache_key, layer.weights,
                                        error)) {
            return false;
        }
    }

    const size_t graph_overhead =
            ggml_graph_overhead_custom(kMaxGraphNodes, false);
    const size_t ctx_size =
            graph_overhead + ggml_tensor_overhead() * 64 + 4 * 1024 * 1024;
    ggml_init_params params{ctx_size, nullptr, true};
    ggml_context *ctx = ggml_init(params);
    if (ctx == nullptr) {
        if (error) {
            *error = "failed to create fused conv chain context";
        }
        output->Release();
        return false;
    }

    ggml_tensor *graph_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, input.w,
                                               input.h, input.c, 1);
    ggml_tensor *x = graph_in;
    for (const ConvChainSpec &layer : layers) {
        const GgmlConvRunner::CachedWeight &cached =
                runner_.CachedEntry(layer.cache_key);
        ggml_tensor *conv =
                ggml_conv_2d_direct(ctx, cached.kernel, x, layer.stride,
                                    layer.stride, layer.pad, layer.pad, 1, 1);
        ggml_tensor *added = ggml_add(ctx, conv, cached.bias);
        x = layer.apply_selu ? GgmlSelu(ctx, added) : added;
    }

    ggml_cgraph *graph = ggml_new_graph_custom(ctx, kMaxGraphNodes, false);
    ggml_build_forward_expand(graph, x);

    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(ctx, backend_->handle);
    if (buffer == nullptr) {
        if (error) {
            *error = "failed to allocate fused conv chain buffer";
        }
        ggml_free(ctx);
        output->Release();
        return false;
    }

    ggml_gallocr_t gallocr = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_->handle));
    if (gallocr == nullptr) {
        if (error) {
            *error = "failed to create fused conv chain gallocr";
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        output->Release();
        return false;
    }
    if (!ggml_gallocr_alloc_graph(gallocr, graph)) {
        if (error) {
            *error = "failed to allocate fused conv chain graph";
        }
        ggml_gallocr_free(gallocr);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        output->Release();
        return false;
    }

    BackendTensorCopyCompat(backend_, input.tensor, graph_in);

    if (ggml_backend_graph_compute(backend_->handle, graph) !=
        GGML_STATUS_SUCCESS) {
        if (error) {
            *error = "fused conv chain graph compute failed";
        }
        ggml_gallocr_free(gallocr);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        output->Release();
        return false;
    }

    if (!GpuTensor::Allocate(backend_, static_cast<int32_t>(x->ne[0]),
                             static_cast<int32_t>(x->ne[1]),
                             static_cast<int32_t>(x->ne[2]), output, error)) {
        ggml_gallocr_free(gallocr);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }
    BackendTensorCopyCompat(backend_, x, output->tensor);
#if defined(AICORE_VULKAN_ALIKED)
    if (backend_->IsVulkan()) {
        ggml_backend_synchronize(backend_->handle);
        VkAlikedQueueIdle(backend_->handle);
    } else
#endif
    {
        FlushGpuPipeline(backend_);
    }
    ggml_gallocr_free(gallocr);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

bool GgmlGpuSession::RunScoreHeadSchedGraph(
        const std::vector<ConvChainSpec> &layers,
        const GpuTensor &input,
        GpuTensor *output,
        const ScoreHeadSchedOptions &opts,
        std::string *error) {
    if (!backend_->HasSched()) {
        if (error) {
            *error = "score-head scheduler path requires Vulkan sched backend";
        }
        return false;
    }
    if (backend_->vulkan_config.scheduler_tail_only && layers.size() > 1) {
        std::vector<ConvChainSpec> prefix(layers.begin(), layers.end() - 1);
        const ConvChainSpec &tail = layers.back();
        GpuTensor mid;
        if (!RunFusedConvChainGraph(prefix, input, &mid, error)) {
            return false;
        }
        return RunScoreHeadSchedGraphImpl({tail}, mid, output, opts, error);
    }
    return RunScoreHeadSchedGraphImpl(layers, input, output, opts, error);
}

bool GgmlGpuSession::RunScoreHeadSchedGraphImpl(
        const std::vector<ConvChainSpec> &layers,
        const GpuTensor &input,
        GpuTensor *output,
        const ScoreHeadSchedOptions &opts,
        std::string *error) {
    if (!backend_->HasSched()) {
        if (error) {
            *error = "score-head scheduler path requires Vulkan sched backend";
        }
        return false;
    }
    if (layers.empty() || input.tensor == nullptr) {
        if (error) {
            *error = "invalid score-head sched input";
        }
        return false;
    }

    for (const ConvChainSpec &layer : layers) {
        if (layer.cache_key == nullptr || layer.cache_key[0] == '\0') {
            if (error) {
                *error = "score-head sched layer missing cache key";
            }
            return false;
        }
        if (!runner_.EnsureCachedPublic(layer.cache_key, layer.weights,
                                        error)) {
            return false;
        }
    }

    const size_t graph_overhead =
            ggml_graph_overhead_custom(kMaxGraphNodes, false);
    const size_t ctx_size =
            graph_overhead + ggml_tensor_overhead() * 64 + 4 * 1024 * 1024;
    ggml_init_params params{ctx_size, nullptr, true};
    ggml_context *ctx = ggml_init(params);
    if (ctx == nullptr) {
        if (error) {
            *error = "failed to create score-head sched context";
        }
        output->Release();
        return false;
    }

    ggml_tensor *graph_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, input.w,
                                               input.h, input.c, 1);
    ggml_tensor *x = graph_in;
    std::vector<ggml_tensor *> weight_tensors;
    weight_tensors.reserve(layers.size() * 2);
    for (const ConvChainSpec &layer : layers) {
        const GgmlConvRunner::CachedWeight &cached =
                runner_.CachedEntry(layer.cache_key);
        weight_tensors.push_back(cached.kernel);
        weight_tensors.push_back(cached.bias);
        ggml_tensor *conv =
                ggml_conv_2d_direct(ctx, cached.kernel, x, layer.stride,
                                    layer.stride, layer.pad, layer.pad, 1, 1);
        ggml_tensor *added = ggml_add(ctx, conv, cached.bias);
        x = layer.apply_selu ? GgmlSelu(ctx, added) : added;
    }
    if (opts.apply_sigmoid) {
        x = ggml_sigmoid(ctx, x);
    }
    if (opts.apply_crop) {
        const size_t es = ggml_element_size(x);
        const size_t offset = static_cast<size_t>(opts.crop_pad_left) * es +
                              static_cast<size_t>(opts.crop_pad_top) * x->nb[1];
        ggml_tensor *view =
                ggml_view_4d(ctx, x, opts.crop_out_w, opts.crop_out_h, x->ne[2],
                             1, x->nb[1], x->nb[2], x->nb[3], offset);
        ggml_tensor *cropped =
                ggml_new_tensor_4d(ctx, GGML_TYPE_F32, opts.crop_out_w,
                                   opts.crop_out_h, x->ne[2], 1);
        x = ggml_cpy(ctx, view, cropped);
    }

    ggml_cgraph *graph = ggml_new_graph_custom(ctx, kMaxGraphNodes, false);
    ggml_build_forward_expand(graph, x);

    internal::Backend *backend = backend_;
    ggml_tensor *input_tensor = input.tensor;
    ggml_cgraph *graph_ptr = graph;
    std::vector<ggml_tensor *> pin_weights = weight_tensors;
    ggml_tensor *pin_in = graph_in;
    if (!backend_->SchedRunGraph(
                graph,
                [=]() {
                    BackendTensorCopyCompat(backend, input_tensor, pin_in);
                },
                error,
                [=]() {
                    PinSchedScoreHeadGraph(backend, graph_ptr, pin_in,
                                           pin_weights);
                })) {
        ggml_free(ctx);
        output->Release();
        return false;
    }

    const int32_t out_w = static_cast<int32_t>(x->ne[0]);
    const int32_t out_h = static_cast<int32_t>(x->ne[1]);
    const int32_t out_c = static_cast<int32_t>(x->ne[2]);
    if (!GpuTensor::Allocate(backend_, out_w, out_h, out_c, output, error)) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_synchronize(backend_->handle);
    BackendTensorCopyCompat(backend_, x, output->tensor);
    ggml_free(ctx);
    return true;
}

}  // namespace lightglue::aliked_internal
