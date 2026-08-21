// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/aliked/ggml_cnn.hpp"

#include <ggml-alloc.h>
#include <ggml-backend.h>

#include "tasks/aliked/gpu_sync.hpp"
#include "tasks/aliked/gpu_tensor.hpp"
#include "tasks/aliked/tensor_ops.hpp"

#if defined(AICORE_VULKAN_ALIKED)
#include "tasks/aliked/vulkan/vulkan_aliked_dispatch.hpp"

#endif
#if defined(AICORE_CUDA_ALIKED)
#include <cuda_runtime.h>
#endif

#include <cmath>
#include <memory>

namespace lightglue::aliked_internal {
namespace {

constexpr int64_t kMaxGraphNodes = 4096;

internal::Backend *VulkanCpuConvBackend() {
    static std::unique_ptr<internal::Backend> cpu_backend;
    static bool ready = false;
    if (!ready) {
        cpu_backend = std::make_unique<internal::Backend>();
        ready = cpu_backend->Init("cpu", 0);
    }
    return ready ? cpu_backend.get() : nullptr;
}

int32_t IndexNchw(int32_t c, int32_t y, int32_t x, int32_t h, int32_t w) {
    return c * h * w + y * w + x;
}

}  // namespace

FusedConv2d FuseConvBn(const std::vector<float> &weight_nchw,
                       int32_t oc,
                       int32_t ic,
                       int32_t kh,
                       int32_t kw,
                       const std::vector<float> *conv_bias,
                       const std::vector<float> &gamma,
                       const std::vector<float> &beta,
                       const std::vector<float> &mean,
                       const std::vector<float> &var) {
    FusedConv2d fused;
    fused.ic = ic;
    fused.oc = oc;
    fused.kh = kh;
    fused.kw = kw;
    fused.kernel.assign(static_cast<size_t>(kw) * kh * ic * oc, 0.0f);
    fused.bias.assign(static_cast<size_t>(oc), 0.0f);

    for (int32_t o = 0; o < oc; ++o) {
        const float inv_std =
                1.0f / std::sqrt(var[static_cast<size_t>(o)] + kBnEpsGgml);
        const float scale = gamma[static_cast<size_t>(o)] * inv_std;
        const float shift = beta[static_cast<size_t>(o)] -
                            mean[static_cast<size_t>(o)] * scale;
        const float b = conv_bias ? (*conv_bias)[static_cast<size_t>(o)] : 0.0f;
        fused.bias[static_cast<size_t>(o)] = b * scale + shift;

        for (int32_t i = 0; i < ic; ++i) {
            for (int32_t ky = 0; ky < kh; ++ky) {
                for (int32_t kx = 0; kx < kw; ++kx) {
                    const size_t pt_idx =
                            static_cast<size_t>(o) * ic * kh * kw +
                            static_cast<size_t>(i) * kh * kw + ky * kw + kx;
                    const size_t ggml_idx =
                            static_cast<size_t>(kx) + ky * kw +
                            static_cast<size_t>(i) * kh * kw +
                            static_cast<size_t>(o) * ic * kh * kw;
                    fused.kernel[ggml_idx] = weight_nchw[pt_idx] * scale;
                }
            }
        }
    }
    return fused;
}

FusedConv2dNchw FuseConvBnNchw(const std::vector<float> &weight_nchw,
                               int32_t oc,
                               int32_t ic,
                               int32_t kh,
                               int32_t kw,
                               const std::vector<float> *conv_bias,
                               const std::vector<float> &gamma,
                               const std::vector<float> &beta,
                               const std::vector<float> &mean,
                               const std::vector<float> &var) {
    FusedConv2dNchw fused;
    fused.ic = ic;
    fused.oc = oc;
    fused.kh = kh;
    fused.kw = kw;
    fused.kernel.assign(static_cast<size_t>(oc) * ic * kh * kw, 0.0f);
    fused.bias.assign(static_cast<size_t>(oc), 0.0f);

    for (int32_t o = 0; o < oc; ++o) {
        const float inv_std =
                1.0f / std::sqrt(var[static_cast<size_t>(o)] + kBnEpsGgml);
        const float scale = gamma[static_cast<size_t>(o)] * inv_std;
        const float shift = beta[static_cast<size_t>(o)] -
                            mean[static_cast<size_t>(o)] * scale;
        const float b = conv_bias ? (*conv_bias)[static_cast<size_t>(o)] : 0.0f;
        fused.bias[static_cast<size_t>(o)] = b * scale + shift;

        for (int32_t i = 0; i < ic; ++i) {
            for (int32_t ky = 0; ky < kh; ++ky) {
                for (int32_t kx = 0; kx < kw; ++kx) {
                    const size_t pt_idx =
                            static_cast<size_t>(o) * ic * kh * kw +
                            static_cast<size_t>(i) * kh * kw + ky * kw + kx;
                    fused.kernel[pt_idx] = weight_nchw[pt_idx] * scale;
                }
            }
        }
    }
    return fused;
}

void NchwToWhcn(const std::vector<float> &nchw,
                int32_t c,
                int32_t h,
                int32_t w,
                std::vector<float> *whcn) {
    whcn->assign(static_cast<size_t>(c) * h * w, 0.0f);
    for (int32_t ch = 0; ch < c; ++ch) {
        for (int32_t y = 0; y < h; ++y) {
            for (int32_t x = 0; x < w; ++x) {
                (*whcn)[static_cast<size_t>(x) + static_cast<size_t>(y) * w +
                        static_cast<size_t>(ch) * h * w] =
                        nchw[IndexNchw(ch, y, x, h, w)];
            }
        }
    }
}

void WhcnToNchw(const std::vector<float> &whcn,
                int32_t c,
                int32_t h,
                int32_t w,
                std::vector<float> *nchw) {
    nchw->assign(static_cast<size_t>(c) * h * w, 0.0f);
    for (int32_t ch = 0; ch < c; ++ch) {
        for (int32_t y = 0; y < h; ++y) {
            for (int32_t x = 0; x < w; ++x) {
                (*nchw)[IndexNchw(ch, y, x, h, w)] =
                        whcn[static_cast<size_t>(x) +
                             static_cast<size_t>(y) * w +
                             static_cast<size_t>(ch) * h * w];
            }
        }
    }
}

GgmlConvRunner::GgmlConvRunner(internal::Backend *backend)
    : backend_(backend) {}

GgmlConvRunner::~GgmlConvRunner() {
    InvalidateDeviceGraphs();
    for (auto &entry : cache_) {
        CachedWeight &w = entry.second;
        if (w.owns_buffer) {
            if (w.buffer != nullptr) {
                ggml_backend_buffer_free(w.buffer);
            }
            if (w.ctx != nullptr) {
                ggml_free(w.ctx);
            }
        }
    }
}

void GgmlConvRunner::InvalidateDeviceGraphs() {
    for (auto &entry : cache_) {
        CachedWeight &w = entry.second;
        if (w.graph_gallocr != nullptr) {
            ggml_gallocr_free(w.graph_gallocr);
            w.graph_gallocr = nullptr;
        }
        if (w.graph_buffer != nullptr) {
            ggml_backend_buffer_free(w.graph_buffer);
            w.graph_buffer = nullptr;
        }
        if (w.graph_ctx != nullptr) {
            ggml_free(w.graph_ctx);
            w.graph_ctx = nullptr;
        }
        w.graph = nullptr;
        w.graph_in = nullptr;
        w.graph_out = nullptr;
        w.graph_ih = 0;
        w.graph_iw = 0;
        w.graph_ic = 0;
        w.graph_pad = 0;
        w.graph_stride = 0;
    }
}

void GgmlConvRunner::ImportWeightEntriesFrom(const GgmlConvRunner &other) {
    for (const auto &entry : other.cache_) {
        if (cache_.count(entry.first) > 0) {
            continue;
        }
        CachedWeight shared{};
        shared.ctx = entry.second.ctx;
        shared.buffer = entry.second.buffer;
        shared.kernel = entry.second.kernel;
        shared.bias = entry.second.bias;
        shared.owns_buffer = false;
        cache_[entry.first] = shared;
    }
}

void GgmlConvRunner::RebindAllDeviceGraphs() {
    if (backend_ == nullptr) {
        return;
    }
    for (auto &entry : cache_) {
        CachedWeight &w = entry.second;
        if (w.graph == nullptr || w.graph_gallocr == nullptr) {
            continue;
        }
        ggml_gallocr_alloc_graph(w.graph_gallocr, w.graph);
    }
}

bool GgmlConvRunner::EnsureCached(const char *cache_key,
                                  const FusedConv2d &weights,
                                  std::string *error) {
    if (cache_key == nullptr || cache_key[0] == '\0') {
        return true;
    }
    if (cache_.count(cache_key) > 0) {
        return true;
    }

    const size_t ctx_size = ggml_tensor_overhead() * 4 + 256 * 1024;
    ggml_init_params params{ctx_size, nullptr, true};
    ggml_context *ctx = ggml_init(params);
    if (ctx == nullptr) {
        if (error) {
            *error = "failed to create GGML weight context";
        }
        return false;
    }

    ggml_tensor *kernel = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F32, weights.kw, weights.kh, weights.ic, weights.oc);
    ggml_tensor *bias =
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, weights.oc, 1);

    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(ctx, backend_->handle);
    if (buffer == nullptr) {
        if (error) {
            *error = "failed to allocate persistent GGML conv weights";
        }
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(kernel, weights.kernel.data(), 0,
                            weights.kernel.size() * sizeof(float));
    ggml_backend_tensor_set(bias, weights.bias.data(), 0,
                            weights.bias.size() * sizeof(float));

    cache_[cache_key] = CachedWeight{ctx, buffer, kernel, bias};
    return true;
}

bool GgmlConvRunner::RunGraph(ggml_tensor *kernel,
                              ggml_tensor *bias,
                              const FusedConv2d &weights,
                              int32_t ih,
                              int32_t iw,
                              int32_t pad,
                              int32_t stride,
                              std::vector<float> *output_nchw,
                              int32_t *oh,
                              int32_t *ow,
                              std::string *error) {
    *oh = (ih + 2 * pad - weights.kh) / stride + 1;
    *ow = (iw + 2 * pad - weights.kw) / stride + 1;
    if (*oh <= 0 || *ow <= 0) {
        if (error) {
            *error = "invalid conv output shape";
        }
        return false;
    }

    const size_t graph_overhead =
            ggml_graph_overhead_custom(kMaxGraphNodes, false);
    const size_t ctx_size =
            graph_overhead + ggml_tensor_overhead() * 64 + 4 * 1024 * 1024;
    ggml_init_params params{ctx_size, nullptr, true};
    ggml_context *ctx = ggml_init(params);
    if (ctx == nullptr) {
        if (error) {
            *error = "failed to create ggml context";
        }
        return false;
    }

    ggml_tensor *input =
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, iw, ih, weights.ic, 1);
    ggml_tensor *conv = ggml_conv_2d_direct(ctx, kernel, input, stride, stride,
                                            pad, pad, 1, 1);
    ggml_tensor *out = ggml_add(ctx, conv, bias);

    ggml_cgraph *graph = ggml_new_graph_custom(ctx, kMaxGraphNodes, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(ctx, backend_->handle);
    if (buffer == nullptr) {
        if (error) {
            *error = "failed to allocate GGML tensors";
        }
        ggml_free(ctx);
        return false;
    }

    if (!ggml_gallocr_alloc_graph(backend_->allocator, graph)) {
        if (error) {
            *error = "failed to allocate GGML graph";
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(input, input_whcn_.data(), 0,
                            input_whcn_.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend_->handle, graph) !=
        GGML_STATUS_SUCCESS) {
        if (error) {
            *error = "GGML graph compute failed";
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    const int64_t out_w = out->ne[0];
    const int64_t out_h = out->ne[1];
    const int64_t out_c = out->ne[2];
    output_whcn_.assign(static_cast<size_t>(out_w * out_h * out_c), 0.0f);
    ggml_backend_tensor_get(out, output_whcn_.data(), 0,
                            output_whcn_.size() * sizeof(float));

    WhcnToNchw(output_whcn_, static_cast<int32_t>(out_c),
               static_cast<int32_t>(out_h), static_cast<int32_t>(out_w),
               output_nchw);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

bool GgmlConvRunner::EnsureDeviceGraph(CachedWeight *entry,
                                       const FusedConv2d &weights,
                                       const GpuTensor &input,
                                       int32_t pad,
                                       int32_t stride,
                                       std::string *error) {
    if (entry == nullptr) {
        if (error) {
            *error = "null conv cache entry";
        }
        return false;
    }
    if (entry->graph != nullptr && entry->graph_ih == input.h &&
        entry->graph_iw == input.w && entry->graph_ic == input.c &&
        entry->graph_pad == pad && entry->graph_stride == stride) {
        return true;
    }

    if (entry->graph_buffer != nullptr) {
        ggml_backend_buffer_free(entry->graph_buffer);
        entry->graph_buffer = nullptr;
    }
    if (entry->graph_gallocr != nullptr) {
        ggml_gallocr_free(entry->graph_gallocr);
        entry->graph_gallocr = nullptr;
    }
    if (entry->graph_ctx != nullptr) {
        ggml_free(entry->graph_ctx);
        entry->graph_ctx = nullptr;
    }
    entry->graph = nullptr;
    entry->graph_in = nullptr;
    entry->graph_out = nullptr;

    const size_t graph_overhead =
            ggml_graph_overhead_custom(kMaxGraphNodes, false);
    const size_t ctx_size =
            graph_overhead + ggml_tensor_overhead() * 64 + 4 * 1024 * 1024;
    ggml_init_params params{ctx_size, nullptr, true};
    entry->graph_ctx = ggml_init(params);
    if (entry->graph_ctx == nullptr) {
        if (error) {
            *error = "failed to create cached conv graph context";
        }
        return false;
    }

    entry->graph_in = ggml_new_tensor_4d(entry->graph_ctx, GGML_TYPE_F32,
                                         input.w, input.h, input.c, 1);
    ggml_tensor *graph_kernel =
            ggml_new_tensor_4d(entry->graph_ctx, GGML_TYPE_F32, weights.kw,
                               weights.kh, weights.ic, weights.oc);
    ggml_tensor *graph_bias = ggml_new_tensor_4d(
            entry->graph_ctx, GGML_TYPE_F32, 1, 1, weights.oc, 1);
    ggml_tensor *conv =
            ggml_conv_2d_direct(entry->graph_ctx, graph_kernel, entry->graph_in,
                                stride, stride, pad, pad, 1, 1);
    entry->graph_out = ggml_add(entry->graph_ctx, conv, graph_bias);
    entry->graph =
            ggml_new_graph_custom(entry->graph_ctx, kMaxGraphNodes, false);
    ggml_build_forward_expand(entry->graph, entry->graph_out);

    entry->graph_buffer =
            ggml_backend_alloc_ctx_tensors(entry->graph_ctx, backend_->handle);
    if (entry->graph_buffer == nullptr) {
        if (error) {
            *error = "failed to allocate cached conv graph buffer";
        }
        return false;
    }
    ggml_backend_tensor_set(graph_kernel, weights.kernel.data(), 0,
                            weights.kernel.size() * sizeof(float));
    ggml_backend_tensor_set(graph_bias, weights.bias.data(), 0,
                            weights.bias.size() * sizeof(float));
    entry->graph_gallocr = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_->handle));
    if (entry->graph_gallocr == nullptr ||
        !ggml_gallocr_alloc_graph(entry->graph_gallocr, entry->graph)) {
        if (error) {
            *error = "failed to allocate cached conv graph";
        }
        return false;
    }

    entry->graph_ih = input.h;
    entry->graph_iw = input.w;
    entry->graph_ic = input.c;
    entry->graph_pad = pad;
    entry->graph_stride = stride;
    return true;
}

bool GgmlConvRunner::RunGraphDevice(ggml_tensor *kernel,
                                    ggml_tensor *bias,
                                    const FusedConv2d &weights,
                                    const GpuTensor &input,
                                    GpuTensor *output,
                                    int32_t pad,
                                    int32_t stride,
                                    std::string *error,
                                    const char *cache_key) {
    (void)kernel;
    (void)bias;
    if (input.tensor == nullptr) {
        if (error) {
            *error = "GPU conv input tensor is null";
        }
        return false;
    }

    if (backend_->IsVulkan() && backend_->vulkan_config.force_cpu_conv) {
        internal::Backend *cpu_backend = VulkanCpuConvBackend();
        if (cpu_backend == nullptr) {
            if (error) {
                *error = "failed to init CPU backend for Vulkan conv fallback";
            }
            return false;
        }
        GgmlConvRunner cpu_runner(cpu_backend);
        std::vector<float> in_nchw;
        if (!input.DownloadNchw(backend_, &in_nchw, input.c, input.h, input.w,
                                error)) {
            return false;
        }
        std::vector<float> out_nchw;
        int32_t oh = 0;
        int32_t ow = 0;
        if (!cpu_runner.Run(weights, in_nchw, input.h, input.w, pad, stride,
                            &out_nchw, &oh, &ow, error, cache_key)) {
            return false;
        }
        if (!GpuTensor::Allocate(backend_, ow, oh, weights.oc, output, error)) {
            return false;
        }
        return output->UploadNchw(backend_, out_nchw, weights.oc, oh, ow,
                                  error);
    }

    const int32_t oh = (input.h + 2 * pad - weights.kh) / stride + 1;
    const int32_t ow = (input.w + 2 * pad - weights.kw) / stride + 1;
    if (oh <= 0 || ow <= 0) {
        if (error) {
            *error = "invalid conv output shape";
        }
        return false;
    }

    if (!GpuTensor::Allocate(backend_, ow, oh, weights.oc, output, error)) {
        return false;
    }

    if (cache_key != nullptr && cache_key[0] != '\0' &&
        cache_.count(cache_key) > 0) {
        CachedWeight &entry = cache_[cache_key];
        if (!EnsureDeviceGraph(&entry, weights, input, pad, stride, error)) {
            output->Release();
            return false;
        }
        BackendTensorCopyCompat(backend_, input.tensor, entry.graph_in);
        if (!ggml_gallocr_alloc_graph(entry.graph_gallocr, entry.graph)) {
            if (error) {
                *error = "failed to bind cached GGML conv graph allocator";
            }
            output->Release();
            return false;
        }
        if (ggml_backend_graph_compute(backend_->handle, entry.graph) !=
            GGML_STATUS_SUCCESS) {
            if (error) {
                *error = "cached GGML conv compute failed";
            }
            output->Release();
            return false;
        }
        SyncGpuPipeline(backend_);
        FlushGpuPipeline(backend_);
        if (cache_key != nullptr) {
            LogTensorStrideIfDebug(
                    (std::string(cache_key) + ".graph_out").c_str(),
                    entry.graph_out, ow, oh, weights.oc);
        }
        BackendTensorCopyCompat(backend_, entry.graph_out, output->tensor);
        SyncGpuTensorMeta(output);
        if (cache_key != nullptr) {
            LogTensorStrideIfDebug((std::string(cache_key) + ".dst").c_str(),
                                   output->tensor, ow, oh, weights.oc);
        }
        return true;
    }

    const size_t graph_overhead =
            ggml_graph_overhead_custom(kMaxGraphNodes, false);
    const size_t ctx_size =
            graph_overhead + ggml_tensor_overhead() * 64 + 4 * 1024 * 1024;
    ggml_init_params params{ctx_size, nullptr, true};
    ggml_context *ctx = ggml_init(params);
    if (ctx == nullptr) {
        if (error) {
            *error = "failed to create ggml context";
        }
        output->Release();
        return false;
    }

    ggml_tensor *conv = ggml_conv_2d_direct(ctx, kernel, input.tensor, stride,
                                            stride, pad, pad, 1, 1);
    ggml_tensor *out = ggml_add(ctx, conv, bias);

    ggml_cgraph *graph = ggml_new_graph_custom(ctx, kMaxGraphNodes, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(ctx, backend_->handle);
    if (buffer == nullptr) {
        if (error) {
            *error = "failed to allocate GGML tensors";
        }
        ggml_free(ctx);
        output->Release();
        return false;
    }

    if (!ggml_gallocr_alloc_graph(backend_->allocator, graph)) {
        if (error) {
            *error = "failed to allocate GGML graph";
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        output->Release();
        return false;
    }
    if (ggml_backend_graph_compute(backend_->handle, graph) !=
        GGML_STATUS_SUCCESS) {
        if (error) {
            *error = "GGML graph compute failed";
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        output->Release();
        return false;
    }
    ggml_backend_synchronize(backend_->handle);
    if (cache_key != nullptr) {
        LogTensorStrideIfDebug((std::string(cache_key) + ".graph_out").c_str(),
                               out, ow, oh, weights.oc);
    }
    BackendTensorCopyCompat(backend_, out, output->tensor);
    SyncGpuTensorMeta(output);
    if (cache_key != nullptr) {
        LogTensorStrideIfDebug((std::string(cache_key) + ".dst").c_str(),
                               output->tensor, ow, oh, weights.oc);
    }
    SyncGpuPipeline(backend_);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

bool GgmlConvRunner::RunDevice(const FusedConv2d &weights,
                               const GpuTensor &input,
                               GpuTensor *output,
                               int32_t pad,
                               int32_t stride,
                               std::string *error,
                               const char *cache_key) {
    if (backend_ == nullptr || backend_->handle == nullptr ||
        backend_->allocator == nullptr) {
        if (error) {
            *error = "GGML backend is not initialized";
        }
        return false;
    }

    if (cache_key != nullptr && cache_key[0] != '\0') {
        if (!EnsureCached(cache_key, weights, error)) {
            return false;
        }
        const CachedWeight &cached = cache_.at(cache_key);
        return RunGraphDevice(cached.kernel, cached.bias, weights, input,
                              output, pad, stride, error, cache_key);
    }

    const size_t graph_overhead =
            ggml_graph_overhead_custom(kMaxGraphNodes, false);
    const size_t ctx_size =
            graph_overhead + ggml_tensor_overhead() * 64 + 4 * 1024 * 1024;
    ggml_init_params params{ctx_size, nullptr, true};
    ggml_context *ctx = ggml_init(params);
    if (ctx == nullptr) {
        if (error) {
            *error = "failed to create ggml context";
        }
        return false;
    }

    ggml_tensor *kernel = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F32, weights.kw, weights.kh, weights.ic, weights.oc);
    ggml_tensor *bias =
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, weights.oc, 1);
    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(ctx, backend_->handle);
    if (buffer == nullptr) {
        if (error) {
            *error = "failed to allocate GGML tensors";
        }
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(kernel, weights.kernel.data(), 0,
                            weights.kernel.size() * sizeof(float));
    ggml_backend_tensor_set(bias, weights.bias.data(), 0,
                            weights.bias.size() * sizeof(float));

    const bool ok = RunGraphDevice(kernel, bias, weights, input, output, pad,
                                   stride, error);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

bool GgmlConvRunner::Run(const FusedConv2d &weights,
                         const std::vector<float> &input_nchw,
                         int32_t ih,
                         int32_t iw,
                         int32_t pad,
                         int32_t stride,
                         std::vector<float> *output_nchw,
                         int32_t *oh,
                         int32_t *ow,
                         std::string *error,
                         const char *cache_key) {
    if (backend_ == nullptr || backend_->handle == nullptr ||
        backend_->allocator == nullptr) {
        if (error) {
            *error = "GGML backend is not initialized";
        }
        return false;
    }

    NchwToWhcn(input_nchw, weights.ic, ih, iw, &input_whcn_);

    if (cache_key != nullptr && cache_key[0] != '\0') {
        if (!EnsureCached(cache_key, weights, error)) {
            return false;
        }
        const CachedWeight &cached = cache_.at(cache_key);
        return RunGraph(cached.kernel, cached.bias, weights, ih, iw, pad,
                        stride, output_nchw, oh, ow, error);
    }

    const size_t graph_overhead =
            ggml_graph_overhead_custom(kMaxGraphNodes, false);
    const size_t ctx_size =
            graph_overhead + ggml_tensor_overhead() * 64 + 4 * 1024 * 1024;
    ggml_init_params params{ctx_size, nullptr, true};
    ggml_context *ctx = ggml_init(params);
    if (ctx == nullptr) {
        if (error) {
            *error = "failed to create ggml context";
        }
        return false;
    }

    ggml_tensor *kernel = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F32, weights.kw, weights.kh, weights.ic, weights.oc);
    ggml_tensor *bias =
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, weights.oc, 1);
    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(ctx, backend_->handle);
    if (buffer == nullptr) {
        if (error) {
            *error = "failed to allocate GGML tensors";
        }
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(kernel, weights.kernel.data(), 0,
                            weights.kernel.size() * sizeof(float));
    ggml_backend_tensor_set(bias, weights.bias.data(), 0,
                            weights.bias.size() * sizeof(float));

    const bool ok = RunGraph(kernel, bias, weights, ih, iw, pad, stride,
                             output_nchw, oh, ow, error);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

bool RunFusedConv2dGgml(internal::Backend *backend,
                        const FusedConv2d &weights,
                        const std::vector<float> &input_nchw,
                        int32_t ih,
                        int32_t iw,
                        int32_t pad,
                        int32_t stride,
                        std::vector<float> *output_nchw,
                        int32_t *oh,
                        int32_t *ow,
                        std::string *error) {
    GgmlConvRunner runner(backend);
    return runner.Run(weights, input_nchw, ih, iw, pad, stride, output_nchw, oh,
                      ow, error);
}

}  // namespace lightglue::aliked_internal
