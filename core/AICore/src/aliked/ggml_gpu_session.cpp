// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ggml_gpu_session.hpp"

#include "ggml_gpu_ops.hpp"

#if defined(LIGHTGLUE_HAS_CUDA)
#include "aliked_cuda.hpp"
#endif

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>
#if defined(LIGHTGLUE_HAS_CUDA)
#include <cuda_runtime.h>
#endif

namespace lightglue::aliked_internal {
namespace {

constexpr int64_t kMaxGraphNodes = 512;
constexpr float kSeluScale = 1.050700987f;
constexpr float kSeluAlpha = 1.67326324f;

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

void SyncGpuPipeline(internal::Backend *backend) {
    if (backend != nullptr && backend->handle != nullptr) {
        ggml_backend_synchronize(backend->handle);
#if defined(LIGHTGLUE_HAS_CUDA)
        if (backend->IsCuda()) {
            cudaDeviceSynchronize();
        }
#endif
    }
}

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
#if defined(LIGHTGLUE_HAS_CUDA)
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

    int32_t oh = input.h;
    int32_t ow = input.w;
    int32_t oc = input.c;
    for (const ConvChainSpec &layer : layers) {
        oh = (oh + 2 * layer.pad - layer.weights.kh) / layer.stride + 1;
        ow = (ow + 2 * layer.pad - layer.weights.kw) / layer.stride + 1;
        oc = layer.weights.oc;
        if (oh <= 0 || ow <= 0) {
            if (error) {
                *error = "invalid fused conv chain output shape";
            }
            return false;
        }
    }

    if (!GpuTensor::Allocate(backend_, ow, oh, oc, output, error)) {
        return false;
    }

    const size_t graph_overhead =
            ggml_graph_overhead_custom(kMaxGraphNodes, false);
    const size_t ctx_size =
            graph_overhead + ggml_tensor_overhead() * 128 + 8 * 1024 * 1024;
    ggml_init_params params{ctx_size, nullptr, true};
    ggml_context *ctx = ggml_init(params);
    if (ctx == nullptr) {
        if (error) {
            *error = "failed to create fused conv chain context";
        }
        output->Release();
        return false;
    }

    ggml_tensor *x = input.tensor;
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

    if (!ggml_gallocr_alloc_graph(backend_->allocator, graph)) {
        if (error) {
            *error = "failed to allocate fused conv chain graph";
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        output->Release();
        return false;
    }

    if (ggml_backend_graph_compute(backend_->handle, graph) !=
        GGML_STATUS_SUCCESS) {
        if (error) {
            *error = "fused conv chain graph compute failed";
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        output->Release();
        return false;
    }

    ggml_backend_tensor_copy(x, output->tensor);
    SyncGpuPipeline(backend_);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

}  // namespace lightglue::aliked_internal
