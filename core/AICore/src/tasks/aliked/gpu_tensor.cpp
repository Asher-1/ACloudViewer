// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "gpu_tensor.hpp"

#include <ggml-backend.h>
#include <ggml.h>

#include "gpu_pipeline_cache.hpp"
#include "gpu_sync.hpp"
#include "score_debug.hpp"
#if defined(AICORE_VULKAN_ALIKED)
#include "vulkan/vulkan_aliked_dispatch.hpp"
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ggml_cnn.hpp"

namespace lightglue::aliked_internal {
namespace {

#if defined(AICORE_VULKAN_ALIKED)
bool CopyScoreToVulkanScratch(internal::Backend *backend,
                              GpuTensor *score,
                              int32_t h,
                              int32_t w,
                              GpuPipelineCache *cache,
                              std::string *error) {
    if (backend == nullptr || score == nullptr || cache == nullptr ||
        score->tensor == nullptr) {
        if (error) {
            *error = "CopyScoreToVulkanScratch: invalid arguments";
        }
        return false;
    }
    if (!backend->IsVulkan() || !VkAlikedAvailable(backend->handle)) {
        return false;
    }
    AlikedVulkanDkdScratch *scratch = cache->vulkan_dkd_scratch();
    // Score maps are small, while the custom dense-copy path has exhibited
    // zero-filled output for cropped Vulkan views on NVIDIA drivers. Keep the
    // correctness path as the default and retain the device copy only as an
    // explicit diagnostic opt-in.
    const char *device_pin = std::getenv("LIGHTGLUE_ALIKED_SCORE_DEVICE_PIN");
    if (device_pin == nullptr || device_pin[0] == '0') {
        std::vector<float> nchw;
        if (!score->DownloadNchw(backend, &nchw, 1, h, w, error)) {
            return false;
        }
        GpuTensor pinned;
        if (!GpuTensor::Allocate(backend, w, h, 1, &pinned, error)) {
            return false;
        }
        if (!pinned.UploadNchw(backend, nchw, 1, h, w, error)) {
            return false;
        }
        VkAlikedQueueIdle(backend->handle);
        *score = std::move(pinned);
        scratch->score_h = 0;
        scratch->score_w = 0;
        return true;
    }
    if (scratch->score_h != h || scratch->score_w != w ||
        scratch->score_contig.tensor == nullptr) {
        if (!GpuTensor::Allocate(backend, w, h, 1, &scratch->score_contig,
                                 error)) {
            return false;
        }
        scratch->score_h = h;
        scratch->score_w = w;
    }
    if (!VkAlikedDenseCopyWhcn(backend->handle, score->tensor,
                               scratch->score_contig.tensor, w, h, 1)) {
        if (error) {
            *error = "Vulkan score dense-copy failed";
        }
        return false;
    }
    VkAlikedQueueIdle(backend->handle);
    // Keep score on an independent GpuTensor buffer; never swap the ephemeral
    // gallocr slot back in — subsequent feature densify reuses that pool.
    *score = std::move(scratch->score_contig);
    scratch->score_h = 0;
    scratch->score_w = 0;
    return true;
}
#endif

bool DownloadWhcnDense(internal::Backend *backend,
                       const ggml_tensor *tensor,
                       int32_t w,
                       int32_t h,
                       int32_t c,
                       std::vector<float> *whcn) {
    if (tensor == nullptr || whcn == nullptr || w <= 0 || h <= 0 || c <= 0) {
        return false;
    }
    const size_t count = static_cast<size_t>(w) * static_cast<size_t>(h) *
                         static_cast<size_t>(c);
    whcn->resize(count);
    if (IsContiguousWhcn(tensor, w, h, c)) {
        ggml_backend_tensor_get(tensor, whcn->data(), 0, count * sizeof(float));
        return true;
    }
    for (int32_t ch = 0; ch < c; ++ch) {
        for (int32_t y = 0; y < h; ++y) {
            for (int32_t x = 0; x < w; ++x) {
                const size_t byte_off = static_cast<size_t>(x) * tensor->nb[0] +
                                        static_cast<size_t>(y) * tensor->nb[1] +
                                        static_cast<size_t>(ch) * tensor->nb[2];
                float value = 0.0f;
                ggml_backend_tensor_get(tensor, &value, byte_off,
                                        sizeof(float));
                (*whcn)[static_cast<size_t>(x) + static_cast<size_t>(y) * w +
                        static_cast<size_t>(ch) * w * h] = value;
            }
        }
    }
    (void)backend;
    return true;
}

}  // namespace

void GpuTensor::Release() {
    if (buffer != nullptr) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    if (ctx != nullptr) {
        ggml_free(ctx);
        ctx = nullptr;
    }
    tensor = nullptr;
}

bool GpuTensor::Allocate(internal::Backend *backend,
                         int32_t width,
                         int32_t height,
                         int32_t channels,
                         GpuTensor *out,
                         std::string *error) {
    if (backend == nullptr || backend->handle == nullptr) {
        if (error) {
            *error = "GPU backend is not initialized";
        }
        return false;
    }

    out->Release();
    out->w = width;
    out->h = height;
    out->c = channels;

    const size_t ctx_size = ggml_tensor_overhead() * 4 + 256 * 1024;
    ggml_init_params params{ctx_size, nullptr, true};
    out->ctx = ggml_init(params);
    if (out->ctx == nullptr) {
        if (error) {
            *error = "failed to create GPU tensor context";
        }
        return false;
    }

    out->tensor = ggml_new_tensor_4d(out->ctx, GGML_TYPE_F32, width, height,
                                     channels, 1);
    out->buffer = ggml_backend_alloc_ctx_tensors(out->ctx, backend->handle);
    if (out->buffer == nullptr) {
        if (error) {
            const size_t bytes = static_cast<size_t>(width) *
                                 static_cast<size_t>(height) *
                                 static_cast<size_t>(channels) * sizeof(float);
            *error = "failed to allocate GPU tensor buffer (" +
                     std::to_string(width) + "x" + std::to_string(height) +
                     "x" + std::to_string(channels) + ", " +
                     std::to_string(bytes) + " bytes)";
        }
        out->Release();
        return false;
    }
    return true;
}

bool GpuTensor::UploadNchw(internal::Backend *backend,
                           const std::vector<float> &nchw,
                           int32_t ic,
                           int32_t ih,
                           int32_t iw,
                           std::string *error) {
    if (tensor == nullptr) {
        if (error) {
            *error = "GPU tensor is not allocated";
        }
        return false;
    }
    std::vector<float> whcn;
    NchwToWhcn(nchw, ic, ih, iw, &whcn);
    ggml_backend_tensor_set(tensor, whcn.data(), 0,
                            whcn.size() * sizeof(float));
    (void)backend;
    return true;
}

bool GpuTensor::DownloadNchw(internal::Backend *backend,
                             std::vector<float> *nchw,
                             int32_t ic,
                             int32_t ih,
                             int32_t iw,
                             std::string *error) const {
    if (tensor == nullptr) {
        if (error) {
            *error = "GPU tensor is not allocated";
        }
        return false;
    }
    std::vector<float> whcn(ElementCount());
#if defined(AICORE_VULKAN_ALIKED)
    if (backend != nullptr && backend->IsVulkan()) {
        BarrierGpuPipeline(backend);
    } else
#endif
    {
        FlushGpuPipeline(backend);
    }
    if (!DownloadWhcnDense(backend, tensor, w, h, c, &whcn)) {
        if (error) {
            *error = "failed to download GPU tensor";
        }
        return false;
    }
    WhcnToNchw(whcn, c, h, w, nchw);
    (void)ic;
    (void)ih;
    (void)iw;
    return true;
}

namespace {

bool SameTensorLayout(const ggml_tensor *a, const ggml_tensor *b) {
    if (a->type != b->type) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (a->ne[i] != b->ne[i] || a->nb[i] != b->nb[i]) {
            return false;
        }
    }
    return true;
}

bool TryVulkanDenseCopy(internal::Backend *backend,
                        const ggml_tensor *src,
                        ggml_tensor *dst) {
#if defined(AICORE_VULKAN_ALIKED)
    if (backend == nullptr || !backend->IsVulkan() ||
        !VkAlikedAvailable(backend->handle) || src == nullptr ||
        dst == nullptr || src->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src->ne[3] != 1 || dst->ne[3] != 1) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        if (src->ne[i] != dst->ne[i]) {
            return false;
        }
    }
    const int32_t w = static_cast<int32_t>(src->ne[0]);
    const int32_t h = static_cast<int32_t>(src->ne[1]);
    const int32_t c = static_cast<int32_t>(src->ne[2]);
    return VkAlikedDenseCopyWhcn(backend->handle, src, dst, w, h, c);
#else
    (void)backend;
    (void)src;
    (void)dst;
    return false;
#endif
}

}  // namespace

bool IsContiguousWhcn(const ggml_tensor *tensor,
                      int32_t w,
                      int32_t h,
                      int32_t c) {
    if (tensor == nullptr || w <= 0 || h <= 0 || c <= 0) {
        return false;
    }
    if (tensor->ne[0] != w || tensor->ne[1] != h || tensor->ne[2] != c) {
        return false;
    }
    const size_t es = ggml_element_size(tensor);
    return tensor->nb[0] == es &&
           tensor->nb[1] == static_cast<size_t>(w) * es &&
           tensor->nb[2] ==
                   static_cast<size_t>(w) * static_cast<size_t>(h) * es;
}

bool EnsureDenseWhcn(internal::Backend *backend,
                     GpuTensor *tensor,
                     std::string *error) {
#if defined(AICORE_VULKAN_ALIKED)
    if (backend != nullptr && backend->IsVulkan()) {
        return EnsureDenseWhcnGpu(backend, tensor, error);
    }
#endif
    if (tensor == nullptr || tensor->tensor == nullptr) {
        if (error) {
            *error = "cannot densify null GPU tensor";
        }
        return false;
    }
    if (IsContiguousWhcn(tensor->tensor, tensor->w, tensor->h, tensor->c)) {
        return true;
    }
    std::vector<float> nchw;
    if (!tensor->DownloadNchw(backend, &nchw, tensor->c, tensor->h, tensor->w,
                              error)) {
        return false;
    }
    GpuTensor dense;
    if (!GpuTensor::Allocate(backend, tensor->w, tensor->h, tensor->c, &dense,
                             error)) {
        return false;
    }
    if (!dense.UploadNchw(backend, nchw, tensor->c, tensor->h, tensor->w,
                          error)) {
        return false;
    }
    *tensor = std::move(dense);
    return true;
}

bool EnsureDenseWhcnGpu(internal::Backend *backend,
                        GpuTensor *tensor,
                        std::string *error) {
    if (tensor == nullptr || tensor->tensor == nullptr) {
        if (error) {
            *error = "cannot densify null GPU tensor";
        }
        return false;
    }
    SyncGpuTensorMeta(tensor);
    BarrierGpuPipeline(backend);
#if defined(AICORE_VULKAN_ALIKED)
    if (backend != nullptr && backend->IsVulkan()) {
        VkAlikedQueueIdle(backend->handle);
    }
#endif
    if (IsContiguousWhcn(tensor->tensor, tensor->w, tensor->h, tensor->c)) {
        return true;
    }
    GpuTensor dense;
    if (!GpuTensor::Allocate(backend, tensor->w, tensor->h, tensor->c, &dense,
                             error)) {
        return false;
    }
    if (TryVulkanDenseCopy(backend, tensor->tensor, dense.tensor)) {
#if defined(AICORE_VULKAN_ALIKED)
        VkAlikedQueueIdle(backend->handle);
#endif
        *tensor = std::move(dense);
        return true;
    }
    std::vector<float> nchw;
    if (!tensor->DownloadNchw(backend, &nchw, tensor->c, tensor->h, tensor->w,
                              error)) {
        return false;
    }
    if (!dense.UploadNchw(backend, nchw, tensor->c, tensor->h, tensor->w,
                          error)) {
        return false;
    }
#if defined(AICORE_VULKAN_ALIKED)
    if (backend != nullptr && backend->IsVulkan()) {
        VkAlikedQueueIdle(backend->handle);
    }
#endif
    *tensor = std::move(dense);
    return true;
}

bool PinVulkanScoreMap(internal::Backend *backend,
                       GpuTensor *score,
                       int32_t h,
                       int32_t w,
                       GpuPipelineCache *cache,
                       std::string *error) {
#if defined(AICORE_VULKAN_ALIKED)
    if (backend == nullptr || score == nullptr || cache == nullptr) {
        if (error) {
            *error = "PinVulkanScoreMap: invalid arguments";
        }
        return false;
    }
    if (!backend->IsVulkan()) {
        return true;
    }
    SyncGpuTensorMeta(score);
    if (score->w != w || score->h != h || score->c != 1) {
        if (error) {
            *error = "PinVulkanScoreMap: score map shape mismatch";
        }
        return false;
    }
    BarrierGpuPipeline(backend);
    return CopyScoreToVulkanScratch(backend, score, h, w, cache, error);
#else
    (void)backend;
    (void)score;
    (void)h;
    (void)w;
    (void)cache;
    (void)error;
    return true;
#endif
}

bool PrepareScoreMapForDkd(internal::Backend *backend,
                           GpuTensor *score,
                           int32_t h,
                           int32_t w,
                           GpuPipelineCache *cache,
                           std::string *error) {
    if (backend == nullptr || score == nullptr || cache == nullptr) {
        if (error) {
            *error = "PrepareScoreMapForDkd: invalid arguments";
        }
        return false;
    }
    SyncGpuTensorMeta(score);
    if (score->w != w || score->h != h || score->c != 1) {
        if (error) {
            *error = "PrepareScoreMapForDkd: score map shape mismatch";
        }
        return false;
    }

    BarrierGpuPipeline(backend);

#if defined(AICORE_VULKAN_ALIKED)
    if (backend->IsVulkan() && VkAlikedAvailable(backend->handle)) {
        if (!EnsureDenseWhcnGpu(backend, score, error)) {
            return false;
        }
        if (CopyScoreToVulkanScratch(backend, score, h, w, cache, error)) {
            if (DkdDebugEnabled()) {
                LogScoreMapStage(backend, *score, h, w, "prepare_pin", error);
            }
            return true;
        }
    }
#endif
    return EnsureDenseWhcnGpu(backend, score, error);
}

bool ForceDenseWhcn(internal::Backend *backend,
                    GpuTensor *tensor,
                    std::string *error) {
#if defined(AICORE_VULKAN_ALIKED)
    if (backend != nullptr && backend->IsVulkan()) {
        return EnsureDenseWhcnGpu(backend, tensor, error);
    }
#endif
    if (tensor == nullptr || tensor->tensor == nullptr) {
        if (error) {
            *error = "cannot force-densify null GPU tensor";
        }
        return false;
    }
    SyncGpuTensorMeta(tensor);
    FlushGpuPipeline(backend);
    std::vector<float> nchw;
    if (!tensor->DownloadNchw(backend, &nchw, tensor->c, tensor->h, tensor->w,
                              error)) {
        return false;
    }
    GpuTensor dense;
    if (!GpuTensor::Allocate(backend, tensor->w, tensor->h, tensor->c, &dense,
                             error)) {
        return false;
    }
    if (!dense.UploadNchw(backend, nchw, tensor->c, tensor->h, tensor->w,
                          error)) {
        return false;
    }
    *tensor = std::move(dense);
    return true;
}

void SyncGpuTensorMeta(GpuTensor *tensor) {
    if (tensor == nullptr || tensor->tensor == nullptr) {
        return;
    }
    tensor->w = static_cast<int32_t>(tensor->tensor->ne[0]);
    tensor->h = static_cast<int32_t>(tensor->tensor->ne[1]);
    tensor->c = static_cast<int32_t>(tensor->tensor->ne[2]);
}

void LogTensorStrideIfDebug(const char *label,
                            const ggml_tensor *tensor,
                            int32_t w,
                            int32_t h,
                            int32_t c) {
    if (tensor == nullptr || label == nullptr) {
        return;
    }
    const char *stride_env = std::getenv("LIGHTGLUE_ALIKED_CONV_STRIDE_DEBUG");
    const bool stride_debug = stride_env != nullptr && stride_env[0] != '0';
    const bool offset_dkd =
            DkdDebugEnabled() && std::strstr(label, ".offset") != nullptr;
    if (!stride_debug && !offset_dkd) {
        return;
    }
    const bool contiguous = IsContiguousWhcn(tensor, w, h, c);
    std::fprintf(stderr,
                 "[conv-stride] %s ne=(%lld,%lld,%lld,%lld) "
                 "nb=(%zu,%zu,%zu,%zu) whc=(%d,%d,%d) contiguous=%d\n",
                 label, static_cast<long long>(tensor->ne[0]),
                 static_cast<long long>(tensor->ne[1]),
                 static_cast<long long>(tensor->ne[2]),
                 static_cast<long long>(tensor->ne[3]), tensor->nb[0],
                 tensor->nb[1], tensor->nb[2], tensor->nb[3], w, h, c,
                 contiguous ? 1 : 0);
}

void BackendTensorCopyCompat(internal::Backend *backend,
                             const ggml_tensor *src,
                             ggml_tensor *dst) {
    if (src == nullptr || dst == nullptr || src == dst) {
        return;
    }
    if (ggml_nelements(src) != ggml_nelements(dst)) {
        GGML_ABORT("tensor copy element count mismatch");
    }
    if (SameTensorLayout(src, dst)) {
        ggml_backend_tensor_copy(src, dst);
        return;
    }
    if (TryVulkanDenseCopy(backend, src, dst)) {
        return;
    }

    const int32_t w = static_cast<int32_t>(src->ne[0]);
    const int32_t h = static_cast<int32_t>(src->ne[1]);
    const int32_t c = static_cast<int32_t>(src->ne[2]);
    std::vector<float> whcn;
    if (!DownloadWhcnDense(backend, src, w, h, c, &whcn)) {
        GGML_ABORT("tensor copy strided read failed");
    }
    ggml_backend_tensor_set(dst, whcn.data(), 0, whcn.size() * sizeof(float));
}

}  // namespace lightglue::aliked_internal
