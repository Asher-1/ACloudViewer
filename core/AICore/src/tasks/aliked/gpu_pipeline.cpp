// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/aliked/gpu_pipeline.hpp"

#include <ggml-backend.h>

#include "tasks/aliked/aliked_gpu_ops.hpp"
#include "tasks/aliked/aliked_stage_bench.hpp"
#include "tasks/aliked/deform_conv.hpp"
#include "tasks/aliked/ggml_cnn.hpp"
#include "tasks/aliked/ggml_gpu_ops.hpp"
#include "tasks/aliked/ggml_gpu_session.hpp"
#include "tasks/aliked/gpu_pipeline_cache.hpp"
#include "tasks/aliked/gpu_postprocess.hpp"
#include "tasks/aliked/gpu_sync.hpp"
#include "tasks/aliked/gpu_tensor.hpp"
#include "tasks/aliked/model_weights.hpp"
#include "tasks/aliked/score_debug.hpp"
#include "tasks/aliked/tensor_ops.hpp"

#if defined(AICORE_VULKAN_ALIKED)
#include "tasks/aliked/vulkan/vulkan_aliked_dispatch.hpp"

#endif
#if defined(AICORE_CUDA_ALIKED)
#include <cuda_runtime.h>

#include "tasks/aliked/cuda/aliked_cuda.hpp"

#endif

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace lightglue::aliked_internal {
namespace {

struct InputPadder {
    int32_t pad_left = 0;
    int32_t pad_right = 0;
    int32_t pad_top = 0;
    int32_t pad_bottom = 0;
    int32_t padded_h = 0;
    int32_t padded_w = 0;

    explicit InputPadder(int32_t h, int32_t w, int32_t divisor) {
        const int32_t pad_h = ((h + divisor - 1) / divisor) * divisor - h;
        const int32_t pad_w = ((w + divisor - 1) / divisor) * divisor - w;
        pad_left = pad_w / 2;
        pad_right = pad_w - pad_left;
        pad_top = pad_h / 2;
        pad_bottom = pad_h - pad_top;
        padded_h = h + pad_h;
        padded_w = w + pad_w;
    }

    std::vector<float> Pad(const std::vector<float> &input,
                           int32_t c,
                           int32_t h,
                           int32_t w) const {
        std::vector<float> output(static_cast<size_t>(c) * padded_h * padded_w,
                                  0.0f);
        for (int32_t ch = 0; ch < c; ++ch) {
            for (int32_t y = 0; y < padded_h; ++y) {
                const int32_t src_y = std::min(std::max(y - pad_top, 0), h - 1);
                for (int32_t x = 0; x < padded_w; ++x) {
                    const int32_t src_x =
                            std::min(std::max(x - pad_left, 0), w - 1);
                    output[static_cast<size_t>(ch) * padded_h * padded_w +
                           y * padded_w + x] =
                            input[static_cast<size_t>(ch) * h * w + src_y * w +
                                  src_x];
                }
            }
        }
        return output;
    }
};

float *DevPtr(const GpuTensor &tensor) {
    return reinterpret_cast<float *>(tensor.tensor->data);
}

bool LogBackboneIfDebug(internal::Backend *backend,
                        const GpuTensor &tensor,
                        int32_t c,
                        int32_t h,
                        int32_t w,
                        const char *stage,
                        std::string *error);

bool UseCudaCustomKernels(internal::Backend *backend) {
#if defined(AICORE_CUDA_ALIKED)
    return backend != nullptr && backend->IsCuda();
#else
    (void)backend;
    return false;
#endif
}

bool AvgPoolGpu(internal::Backend *backend,
                const GpuTensor &input,
                int32_t kh,
                int32_t kw,
                int32_t stride,
                GpuTensor *output,
                std::string *error) {
    if (!UseCudaCustomKernels(backend)) {
        return RunAvgPool2dGpu(backend, input, kh, kw, stride, output, error);
    }
#if defined(AICORE_CUDA_ALIKED)
    const int32_t oh = (input.h - kh) / stride + 1;
    const int32_t ow = (input.w - kw) / stride + 1;
    if (!GpuTensor::Allocate(backend, ow, oh, input.c, output, error)) {
        return false;
    }
    SyncGpuPipeline(backend);
    if (!AlikedCudaAvgPool2d(backend->handle, DevPtr(input), input.c, input.h,
                             input.w, kh, kw, stride, DevPtr(*output), oh,
                             ow)) {
        if (error) {
            *error = "CUDA avg pool failed";
        }
        return false;
    }
    return true;
#else
    if (error) {
        *error = "CUDA avg pool requested without CUDA build";
    }
    return false;
#endif
}

bool UpsampleGpu(internal::Backend *backend,
                 const GpuTensor &input,
                 int32_t out_w,
                 int32_t out_h,
                 GpuTensor *output,
                 std::string *error) {
    if (input.h == out_h && input.w == out_w) {
        if (!GpuTensor::Allocate(backend, out_w, out_h, input.c, output,
                                 error)) {
            return false;
        }
        BackendTensorCopyCompat(backend, input.tensor, output->tensor);
        return true;
    }
#if defined(AICORE_VULKAN_ALIKED)
    if (backend != nullptr && backend->IsVulkan()) {
        FlushGpuPipeline(backend);
    }
    if (backend != nullptr && backend->IsVulkan() &&
        UseVulkanGpuUpsample(backend)) {
        GpuTensor src_contig;
        const GpuTensor *src = &input;
        if (!IsContiguousWhcn(input.tensor, input.w, input.h, input.c)) {
            if (!GpuTensor::Allocate(backend, input.w, input.h, input.c,
                                     &src_contig, error)) {
                return false;
            }
            BackendTensorCopyCompat(backend, input.tensor, src_contig.tensor);
            src = &src_contig;
        }
        GpuTensor upsampled;
        if (!GpuTensor::Allocate(backend, out_w, out_h, input.c, &upsampled,
                                 error)) {
            return false;
        }
        if (VkAlikedUpsampleBilinear(backend->handle, src->tensor,
                                     upsampled.tensor, input.c, input.h,
                                     input.w, out_h, out_w)) {
            *output = std::move(upsampled);
            FlushGpuPipeline(backend);
            return true;
        }
    }
    if (backend != nullptr && backend->IsVulkan()) {
        std::vector<float> in_nchw;
        if (!input.DownloadNchw(backend, &in_nchw, input.c, input.h, input.w,
                                error)) {
            return false;
        }
        std::vector<float> out_nchw;
        UpsampleBilinear(in_nchw, input.c, input.h, input.w, out_h, out_w,
                         &out_nchw);
        if (!GpuTensor::Allocate(backend, out_w, out_h, input.c, output,
                                 error)) {
            return false;
        }
        if (!output->UploadNchw(backend, out_nchw, input.c, out_h, out_w,
                                error)) {
            return false;
        }
        FlushGpuPipeline(backend);
        return true;
    }
#endif
    if (!UseCudaCustomKernels(backend)) {
        return RunInterpolateGpu(backend, input, out_w, out_h, output, error);
    }
#if defined(AICORE_CUDA_ALIKED)
    GpuTensor upsampled;
    if (!GpuTensor::Allocate(backend, out_w, out_h, input.c, &upsampled,
                             error)) {
        return false;
    }
    SyncGpuPipeline(backend);
    if (!AlikedCudaUpsampleBilinear(backend->handle, DevPtr(input), input.c,
                                    input.h, input.w, out_h, out_w,
                                    DevPtr(upsampled))) {
        if (error) {
            *error = "CUDA upsample failed";
        }
        return false;
    }
    *output = std::move(upsampled);
    return true;
#else
    if (error) {
        *error = "CUDA upsample requested without CUDA build";
    }
    return false;
#endif
}

bool CropWhcnGpu(internal::Backend *backend,
                 const GpuTensor &input,
                 int32_t pad_top,
                 int32_t pad_left,
                 int32_t out_h,
                 int32_t out_w,
                 GpuTensor *output,
                 std::string *error) {
    if (!UseCudaCustomKernels(backend)) {
        return RunCropWhcnGpu(backend, input, pad_top, pad_left, out_h, out_w,
                              output, error);
    }
#if defined(AICORE_CUDA_ALIKED)
    if (!GpuTensor::Allocate(backend, out_w, out_h, input.c, output, error)) {
        return false;
    }
    SyncGpuPipeline(backend);
    if (!AlikedCudaCropWhcn(backend->handle, DevPtr(input), input.c, input.h,
                            input.w, pad_top, pad_left, out_h, out_w,
                            DevPtr(*output))) {
        if (error) {
            *error = "CUDA crop failed";
        }
        return false;
    }
    return true;
#else
    if (error) {
        *error = "CUDA crop requested without CUDA build";
    }
    return false;
#endif
}

bool SeluGpu(internal::Backend *backend,
             GpuTensor *tensor,
             std::string *error) {
    if (!UseCudaCustomKernels(backend)) {
        return RunSeluGpu(backend, tensor, error);
    }
#if defined(AICORE_CUDA_ALIKED)
    SyncGpuPipeline(backend);
    if (!AlikedCudaApplySelu(backend->handle, DevPtr(*tensor),
                             tensor->ElementCount())) {
        if (error) {
            *error = "SELU CUDA kernel failed";
        }
        return false;
    }
    return true;
#else
    if (error) {
        *error = "CUDA SELU requested without CUDA build";
    }
    return false;
#endif
}

bool ConvBnSeluGpu(GgmlConvRunner *runner,
                   internal::Backend *backend,
                   const GpuTensor &input,
                   const std::vector<float> &weight,
                   int32_t oc,
                   int32_t kh,
                   int32_t kw,
                   const std::vector<float> &gamma,
                   const std::vector<float> &beta,
                   const std::vector<float> &mean,
                   const std::vector<float> &var,
                   int32_t pad,
                   int32_t stride,
                   GpuTensor *output,
                   const char *cache_key,
                   std::string *error) {
    const FusedConv2d fused = FuseConvBn(weight, oc, input.c, kh, kw, nullptr,
                                         gamma, beta, mean, var);
    if (backend->IsGpu()) {
        GpuTensor tmp;
        if (!runner->RunDevice(fused, input, &tmp, pad, stride, error,
                               cache_key)) {
            return false;
        }
        if (!SeluGpu(backend, &tmp, error)) {
            return false;
        }
        *output = std::move(tmp);
        return true;
    }
    return RunConvBnSeluGpu(runner, backend, fused, input, pad, stride, output,
                            cache_key, error);
}

bool ConvSeluGpu(GgmlConvRunner *runner,
                 internal::Backend *backend,
                 const GpuTensor &input,
                 const std::vector<float> &weight,
                 int32_t oc,
                 int32_t kh,
                 int32_t kw,
                 const std::vector<float> *bias,
                 int32_t pad,
                 int32_t stride,
                 GpuTensor *output,
                 const char *cache_key,
                 std::string *error) {
    std::vector<float> ones(static_cast<size_t>(oc), 1.0f);
    std::vector<float> zeros(static_cast<size_t>(oc), 0.0f);
    const FusedConv2d fused = FuseConvBn(weight, oc, input.c, kh, kw, bias,
                                         ones, zeros, zeros, ones);
    if (backend->IsGpu()) {
        GpuTensor tmp;
        if (!runner->RunDevice(fused, input, &tmp, pad, stride, error,
                               cache_key)) {
            return false;
        }
        if (!SeluGpu(backend, &tmp, error)) {
            return false;
        }
        *output = std::move(tmp);
        return true;
    }
    return RunConvBnSeluGpu(runner, backend, fused, input, pad, stride, output,
                            cache_key, error);
}

bool ConvGpu(GgmlConvRunner *runner,
             const GpuTensor &input,
             const std::vector<float> &weight,
             int32_t oc,
             int32_t kh,
             int32_t kw,
             const std::vector<float> *bias,
             int32_t pad,
             int32_t stride,
             GpuTensor *output,
             const char *cache_key,
             std::string *error) {
    std::vector<float> ones(static_cast<size_t>(oc), 1.0f);
    std::vector<float> zeros(static_cast<size_t>(oc), 0.0f);
    const FusedConv2d fused = FuseConvBn(weight, oc, input.c, kh, kw, bias,
                                         ones, zeros, zeros, ones);
    return runner->RunDevice(fused, input, output, pad, stride, error,
                             cache_key);
}

bool ConcatChannelGpu(internal::Backend *backend,
                      const GpuTensor &a,
                      const GpuTensor &b,
                      GpuTensor *output,
                      std::string *error) {
    if (!UseCudaCustomKernels(backend)) {
        return RunConcatChannelGpu(backend, a, b, output, error);
    }
#if defined(AICORE_CUDA_ALIKED)
    if (!GpuTensor::Allocate(backend, a.w, a.h, a.c + b.c, output, error)) {
        return false;
    }
    SyncGpuPipeline(backend);
    if (!AlikedCudaConcatChannel(backend->handle, DevPtr(a), a.c, DevPtr(b),
                                 b.c, a.h, a.w, DevPtr(*output))) {
        if (error) {
            *error = "concat CUDA failed";
        }
        return false;
    }
    return true;
#else
    if (error) {
        *error = "CUDA concat requested without CUDA build";
    }
    return false;
#endif
}

bool SigmoidInPlaceGpu(internal::Backend *backend,
                       GpuTensor *tensor,
                       std::string *error) {
    if (!UseCudaCustomKernels(backend)) {
        return RunSigmoidInPlaceGpu(backend, tensor, error);
    }
#if defined(AICORE_CUDA_ALIKED)
    SyncGpuPipeline(backend);
    if (!AlikedCudaSigmoidInPlace(backend->handle, DevPtr(*tensor),
                                  tensor->ElementCount())) {
        if (error) {
            *error = "sigmoid CUDA failed";
        }
        return false;
    }
    return true;
#else
    if (error) {
        *error = "CUDA sigmoid requested without CUDA build";
    }
    return false;
#endif
}

bool L2NormalizeChannelsGpu(internal::Backend *backend,
                            GpuTensor *tensor,
                            int32_t channels,
                            int32_t h,
                            int32_t w,
                            std::string *error) {
#if defined(AICORE_VULKAN_ALIKED)
    if (backend != nullptr && backend->IsVulkan() &&
        VkAlikedAvailable(backend->handle)) {
        SyncGpuTensorMeta(tensor);
        if (VkAlikedL2NormInplace(backend->handle, tensor->tensor, channels, h,
                                  w)) {
            return true;
        }
    }
#endif
    if (!UseCudaCustomKernels(backend)) {
        return RunL2NormalizeChannelsGpu(backend, tensor, channels, h, w,
                                         error);
    }
#if defined(AICORE_CUDA_ALIKED)
    SyncGpuPipeline(backend);
    if (!AlikedCudaL2NormalizeChannels(backend->handle, DevPtr(*tensor),
                                       channels, h, w)) {
        if (error) {
            *error = "L2 normalize CUDA failed";
        }
        return false;
    }
    return true;
#else
    if (error) {
        *error = "CUDA L2 normalize requested without CUDA build";
    }
    return false;
#endif
}

bool AddInPlaceGpu(internal::Backend *backend,
                   GpuTensor *accum,
                   const GpuTensor &other,
                   std::string *error,
                   const char *cache_key = nullptr) {
#if defined(AICORE_VULKAN_ALIKED)
    // The generic Vulkan add graph is not yet covered for the dynamic shapes
    // used by every ALIKED residual block. Keep the numerically proven host
    // bridge until the explicit Vulkan batch path has strict parity coverage.
    if (backend != nullptr && backend->IsVulkan()) {
        std::vector<float> lhs;
        std::vector<float> rhs;
        if (!accum->DownloadNchw(backend, &lhs, accum->c, accum->h, accum->w,
                                 error) ||
            !other.DownloadNchw(backend, &rhs, other.c, other.h, other.w,
                                error) ||
            lhs.size() != rhs.size()) {
            if (error != nullptr && error->empty()) {
                *error = "Vulkan residual add shape mismatch";
            }
            return false;
        }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int64_t i = 0; i < static_cast<int64_t>(lhs.size()); ++i) {
            lhs[static_cast<size_t>(i)] += rhs[static_cast<size_t>(i)];
        }
        if (!accum->UploadNchw(backend, lhs, accum->c, accum->h, accum->w,
                               error)) {
            return false;
        }
        FlushGpuPipeline(backend);
        return true;
    }
#endif
    if (!UseCudaCustomKernels(backend)) {
        return RunAddGpu(backend, accum, other, error, cache_key);
    }
#if defined(AICORE_CUDA_ALIKED)
    SyncGpuPipeline(backend);
    if (!AlikedCudaAddInPlace(backend->handle, DevPtr(*accum), DevPtr(other),
                              accum->ElementCount())) {
        if (error) {
            *error = "residual add CUDA failed";
        }
        return false;
    }
    return true;
#else
    if (error) {
        *error = "CUDA add requested without CUDA build";
    }
    return false;
#endif
}

bool ResBlockForwardGpu(GpuPipelineCache *compute,
                        GpuPipelineCache *weights,
                        const GpuTensor &input,
                        int32_t ic,
                        int32_t oc,
                        const TensorMap &tensors,
                        const std::string &prefix,
                        bool dcn,
                        GpuTensor *output,
                        std::string *error) {
    if (compute == nullptr) {
        compute = weights;
    }
    if (weights == nullptr) {
        weights = compute;
    }
    GpuPipelineCache *const dcn_cache = weights;
    internal::Backend *const backend = compute->backend();
    GgmlConvRunner *const runner = compute->ggml()->runner();
    GpuTensor conv1;
    if (dcn) {
        if (!DcnConvBnDispatch(
                    dcn_cache, input,
                    RequireTensor(tensors, prefix + "_conv1_offset_conv_weight",
                                  error),
                    RequireTensor(tensors, prefix + "_conv1_offset_conv_bias",
                                  error),
                    RequireTensor(tensors,
                                  prefix + "_conv1_regular_conv_weight", error),
                    oc, RequireTensor(tensors, prefix + "_bn1_weight", error),
                    RequireTensor(tensors, prefix + "_bn1_bias", error),
                    RequireTensor(tensors, prefix + "_bn1_running_mean", error),
                    RequireTensor(tensors, prefix + "_bn1_running_var", error),
                    prefix + ".dcn1", &conv1, error)) {
            return false;
        }
    } else {
        if (!ConvBnSeluGpu(
                    runner, backend, input,
                    RequireTensor(tensors, prefix + "_conv1_weight", error), oc,
                    3, 3, RequireTensor(tensors, prefix + "_bn1_weight", error),
                    RequireTensor(tensors, prefix + "_bn1_bias", error),
                    RequireTensor(tensors, prefix + "_bn1_running_mean", error),
                    RequireTensor(tensors, prefix + "_bn1_running_var", error),
                    1, 1, &conv1, (prefix + ".conv1").c_str(), error)) {
            return false;
        }
    }
    if (dcn && !SeluGpu(backend, &conv1, error)) {
        return false;
    }
    if (dcn && !LogBackboneIfDebug(backend, conv1, oc, conv1.h, conv1.w,
                                   (prefix + ".selu1").c_str(), error)) {
        return false;
    }

    GpuTensor conv2;
    if (dcn) {
        if (!DcnConvBnDispatch(
                    dcn_cache, conv1,
                    RequireTensor(tensors, prefix + "_conv2_offset_conv_weight",
                                  error),
                    RequireTensor(tensors, prefix + "_conv2_offset_conv_bias",
                                  error),
                    RequireTensor(tensors,
                                  prefix + "_conv2_regular_conv_weight", error),
                    oc, RequireTensor(tensors, prefix + "_bn2_weight", error),
                    RequireTensor(tensors, prefix + "_bn2_bias", error),
                    RequireTensor(tensors, prefix + "_bn2_running_mean", error),
                    RequireTensor(tensors, prefix + "_bn2_running_var", error),
                    prefix + ".dcn2", &conv2, error)) {
            return false;
        }
    } else {
        const FusedConv2d fused = FuseConvBn(
                RequireTensor(tensors, prefix + "_conv2_weight", error), oc, oc,
                3, 3, nullptr,
                RequireTensor(tensors, prefix + "_bn2_weight", error),
                RequireTensor(tensors, prefix + "_bn2_bias", error),
                RequireTensor(tensors, prefix + "_bn2_running_mean", error),
                RequireTensor(tensors, prefix + "_bn2_running_var", error));
        if (!runner->RunDevice(fused, conv1, &conv2, 1, 1, error,
                               (prefix + ".conv2").c_str())) {
            return false;
        }
    }
    if (dcn && !LogBackboneIfDebug(backend, conv2, oc, conv2.h, conv2.w,
                                   (prefix + ".dcn2_out").c_str(), error)) {
        return false;
    }

    GpuTensor identity_buf;
    const GpuTensor *identity = &input;
    if (tensors.count(prefix + "_downsample_weight") > 0) {
        const std::vector<float> &down_w =
                RequireTensor(tensors, prefix + "_downsample_weight", error);
        const std::vector<float> &down_b =
                RequireTensor(tensors, prefix + "_downsample_bias", error);
#if defined(AICORE_VULKAN_ALIKED)
        if (backend != nullptr && backend->IsVulkan()) {
            std::vector<float> input_nchw;
            if (!input.DownloadNchw(backend, &input_nchw, ic, input.h, input.w,
                                    error)) {
                return false;
            }
            std::vector<float> identity_nchw;
            int32_t identity_h = 0;
            int32_t identity_w = 0;
            Conv2d(input_nchw, ic, input.h, input.w, down_w, oc, 1, 1, &down_b,
                   0, 1, &identity_nchw, &identity_h, &identity_w);
            if (!GpuTensor::Allocate(backend, identity_w, identity_h, oc,
                                     &identity_buf, error) ||
                !identity_buf.UploadNchw(backend, identity_nchw, oc, identity_h,
                                         identity_w, error)) {
                return false;
            }
        } else
#endif
                if (!ConvGpu(runner, input, down_w, oc, 1, 1, &down_b, 0, 1,
                             &identity_buf, (prefix + ".downsample").c_str(),
                             error)) {
            return false;
        }
#if defined(AICORE_VULKAN_ALIKED)
        if (backend != nullptr && backend->IsVulkan() && DkdDebugEnabled()) {
            BarrierGpuPipeline(backend);
            if (!LogBackboneStage(backend, identity_buf, oc, identity_buf.h,
                                  identity_buf.w,
                                  (prefix + ".identity").c_str(), error)) {
                return false;
            }
        }
#endif
        identity = &identity_buf;
    } else if (ic != oc) {
        error->assign("residual channel mismatch without downsample for " +
                      prefix);
        return false;
    }

    if (!AddInPlaceGpu(backend, &conv2, *identity, error,
                       (prefix + ".add").c_str())) {
        return false;
    }
    if (dcn && !LogBackboneIfDebug(backend, conv2, oc, conv2.h, conv2.w,
                                   (prefix + ".add_out").c_str(), error)) {
        return false;
    }
    if (dcn) {
        FlushGpuPipeline(backend);
    }
    if (!SeluGpu(backend, &conv2, error)) {
        return false;
    }
    if (dcn && !LogBackboneIfDebug(backend, conv2, oc, conv2.h, conv2.w,
                                   (prefix + ".selu2").c_str(), error)) {
        return false;
    }
    *output = std::move(conv2);
    return true;
}

bool LogBackboneIfDebug(internal::Backend *backend,
                        const GpuTensor &tensor,
                        int32_t c,
                        int32_t h,
                        int32_t w,
                        const char *stage,
                        std::string *error) {
#if defined(AICORE_VULKAN_ALIKED)
    if (backend != nullptr && backend->IsVulkan() && DkdDebugEnabled()) {
        BarrierGpuPipeline(backend);
        return LogBackboneStage(backend, tensor, c, h, w, stage, error);
    }
#else
    (void)backend;
    (void)tensor;
    (void)c;
    (void)h;
    (void)w;
    (void)stage;
#endif
    return error == nullptr || error->empty();
}

}  // namespace

bool ExtractDenseMapGpuVram(const TensorMap &tensors,
                            const std::vector<float> &image,
                            int32_t width,
                            int32_t height,
                            int32_t orig_h,
                            int32_t orig_w,
                            GpuDenseMaps *maps,
                            std::string *error,
                            internal::Backend *backend,
                            GpuPipelineCache *cache) {
    if (backend == nullptr || !backend->IsGpu()) {
        if (error) {
            *error = "GPU pipeline requires a CUDA/Vulkan backend";
        }
        return false;
    }

    GpuPipelineCache stack_cache(backend);
    GpuPipelineCache *pipe = cache != nullptr ? cache : &stack_cache;
    {
        StageBench bench("setup");
        if (cache != nullptr && cache->IsWarmedUp()) {
            if (!pipe->EnsureComputeLinked(error)) {
                return false;
            }
        } else if (!pipe->Warmup(tensors, error) ||
                   !pipe->EnsureComputeLinked(error)) {
            return false;
        }
    }

    GgmlGpuSession *compute = pipe->ComputeGgml();
    GgmlConvRunner *runner = compute->runner();
    internal::Backend *const b = pipe->backend();

#if defined(AICORE_VULKAN_ALIKED)
    if (b != nullptr && b->IsVulkan()) {
        BarrierGpuPipeline(b);
    }
#endif

    InputPadder padder(height, width, 32);
    const std::vector<float> padded = padder.Pad(image, 3, height, width);

    if (!pipe->EnsureInput(padder.padded_w, padder.padded_h, 3, error)) {
        return false;
    }
    GpuTensor &x1 = pipe->InputBuffer();
    {
        StageBench bench("upload_input");
        if (!x1.UploadNchw(b, padded, 3, padder.padded_h, padder.padded_w,
                           error)) {
            return false;
        }
    }

    {
        StageBench bench("backbone");
        {
            StageBench stage("backbone.block1");
            if (!ConvBnSeluGpu(
                        runner, b, x1,
                        RequireTensor(tensors, "block1_conv1_weight", error),
                        16, 3, 3,
                        RequireTensor(tensors, "block1_bn1_weight", error),
                        RequireTensor(tensors, "block1_bn1_bias", error),
                        RequireTensor(tensors, "block1_bn1_running_mean",
                                      error),
                        RequireTensor(tensors, "block1_bn1_running_var", error),
                        1, 1, &x1, "block1.conv1", error)) {
                return false;
            }
            if (!ConvBnSeluGpu(
                        runner, b, x1,
                        RequireTensor(tensors, "block1_conv2_weight", error),
                        16, 3, 3,
                        RequireTensor(tensors, "block1_bn2_weight", error),
                        RequireTensor(tensors, "block1_bn2_bias", error),
                        RequireTensor(tensors, "block1_bn2_running_mean",
                                      error),
                        RequireTensor(tensors, "block1_bn2_running_var", error),
                        1, 1, &x1, "block1.conv2", error)) {
                return false;
            }
        }

        GpuTensor x2;
        {
            StageBench stage("backbone.block2");
            if (!AvgPoolGpu(b, x1, 2, 2, 2, &x2, error)) {
                return false;
            }
            if (!ResBlockForwardGpu(pipe, pipe, x2, 16, 32, tensors, "block2",
                                    false, &x2, error)) {
                return false;
            }
        }
        if (!LogBackboneIfDebug(b, x2, 32, x2.h, x2.w, "block2_out", error)) {
            return false;
        }

        GpuTensor x3;
        {
            StageBench stage("backbone.block3");
            if (!AvgPoolGpu(b, x2, 4, 4, 4, &x3, error)) {
                return false;
            }
            if (!ResBlockForwardGpu(pipe, pipe, x3, 32, 64, tensors, "block3",
                                    true, &x3, error)) {
                return false;
            }
        }
        if (!LogBackboneIfDebug(b, x3, 64, x3.h, x3.w, "block3_out", error)) {
            return false;
        }

        GpuTensor x4;
        {
            StageBench stage("backbone.block4");
            if (!AvgPoolGpu(b, x3, 4, 4, 4, &x4, error)) {
                return false;
            }
            if (!LogBackboneIfDebug(b, x4, 64, x4.h, x4.w, "block4.in",
                                    error)) {
                return false;
            }
            if (!ResBlockForwardGpu(pipe, pipe, x4, 64, 128, tensors, "block4",
                                    true, &x4, error)) {
                return false;
            }
        }
        if (!LogBackboneIfDebug(b, x4, 128, x4.h, x4.w, "block4_out", error)) {
            return false;
        }

        auto project = [&](const GpuTensor &src, const char *weight_name,
                           GpuTensor *dst) -> bool {
            return ConvSeluGpu(runner, b, src,
                               RequireTensor(tensors, weight_name, error), 32,
                               1, 1, nullptr, 0, 1, dst, weight_name, error);
        };

        const int32_t fh = x1.h;
        const int32_t fw = x1.w;
        GpuTensor f1;
        GpuTensor f2;
        GpuTensor f3;
        GpuTensor f4;
        {
            StageBench stage("backbone.project");
            if (!project(x1, "conv1_weight", &f1) ||
                !LogBackboneIfDebug(b, f1, 32, fh, fw, "proj_f1", error) ||
                !project(x2, "conv2_weight", &f2) ||
                !LogBackboneIfDebug(b, f2, 32, f2.h, f2.w, "proj_f2", error) ||
                !project(x3, "conv3_weight", &f3) ||
                !project(x4, "conv4_weight", &f4)) {
                return false;
            }
        }
#if defined(AICORE_VULKAN_ALIKED)
        if (b != nullptr && b->IsVulkan()) {
            BarrierGpuPipeline(b);
        }
#endif
        {
            StageBench stage("backbone.upsample");
            if (!UpsampleGpu(b, f2, fw, fh, &f2, error) ||
                !LogBackboneIfDebug(b, f2, 32, fh, fw, "upsample_f2", error) ||
                !UpsampleGpu(b, f3, fw, fh, &f3, error) ||
                !LogBackboneIfDebug(b, f3, 32, fh, fw, "upsample_f3", error) ||
                !UpsampleGpu(b, f4, fw, fh, &f4, error) ||
                !LogBackboneIfDebug(b, f4, 32, fh, fw, "upsample_f4", error)) {
                return false;
            }
        }
#if defined(AICORE_VULKAN_ALIKED)
        if (b != nullptr && b->IsVulkan()) {
            BarrierGpuPipeline(b);
        }
#endif

        GpuTensor fused;
        GpuTensor fused2;
        GpuTensor feature_gpu;
        {
            StageBench stage("backbone.concat");
            if (!ConcatChannelGpu(b, f1, f2, &fused, error) ||
                !LogBackboneIfDebug(b, fused, 64, fh, fw, "concat_f1_f2",
                                    error) ||
                !ConcatChannelGpu(b, fused, f3, &fused2, error) ||
                !LogBackboneIfDebug(b, fused2, 96, fh, fw, "concat_96",
                                    error) ||
                !ConcatChannelGpu(b, fused2, f4, &feature_gpu, error) ||
                !LogBackboneIfDebug(b, feature_gpu, 128, fh, fw, "concat_out",
                                    error)) {
                return false;
            }
        }

        std::vector<GgmlGpuSession::ConvChainSpec> score_layers =
                pipe->ScoreHeadLayers();
        score_layers.push_back(
                {pipe->ScoreHeadFinal(), 1, 1, "score_head_6", false});
        GpuTensor score_gpu;
        BarrierGpuPipeline(b);
#if defined(AICORE_VULKAN_ALIKED)
        if (b != nullptr && b->IsVulkan() && DkdDebugEnabled()) {
            if (!LogFeatureMapStage(b, feature_gpu, 128, fh, fw,
                                    "score_head_in", error)) {
                return false;
            }
        }
#endif
        const bool use_score_sched = b != nullptr && b->HasSched();
        {
            StageBench stage("backbone.score_head");
            GgmlGpuSession::ScoreHeadSchedOptions sched_opts{};
            if (use_score_sched) {
                sched_opts.apply_sigmoid = true;
                if (!compute->RunScoreHeadSchedGraph(score_layers, feature_gpu,
                                                     &score_gpu, sched_opts,
                                                     error)) {
                    return false;
                }
            } else if (!compute->RunFusedConvChainGraph(
                               score_layers, feature_gpu, &score_gpu, error)) {
                return false;
            }
        }
        BarrierGpuPipeline(b);
#if defined(AICORE_VULKAN_ALIKED)
        if (b != nullptr && b->IsVulkan() && DkdDebugEnabled()) {
            if (!LogScoreMapStage(
                        b, score_gpu, score_gpu.h, score_gpu.w,
                        use_score_sched ? "sigmoid" : "score_head_out",
                        error)) {
                return false;
            }
        }
#endif
        if (!use_score_sched) {
            if (!SigmoidInPlaceGpu(b, &score_gpu, error)) {
                return false;
            }
#if defined(AICORE_VULKAN_ALIKED)
            if (b != nullptr && b->IsVulkan()) {
                VkAlikedQueueIdle(b->handle);
                if (DkdDebugEnabled()) {
                    LogScoreMapStage(b, score_gpu, score_gpu.h, score_gpu.w,
                                     "sigmoid", error);
                }
            }
#endif
        }
        {
            StageBench stage("backbone.normalize");
            if (!L2NormalizeChannelsGpu(b, &feature_gpu, 128, fh, fw, error)) {
                return false;
            }
        }

        maps->height = orig_h;
        maps->width = orig_w;
        if (padder.pad_top == 0 && padder.pad_left == 0 &&
            orig_h == feature_gpu.h && orig_w == feature_gpu.w) {
            maps->feature = std::move(feature_gpu);
            maps->score = std::move(score_gpu);
        } else {
            if (!CropWhcnGpu(b, feature_gpu, padder.pad_top, padder.pad_left,
                             orig_h, orig_w, &maps->feature, error)) {
                return false;
            }
            if (!CropWhcnGpu(b, score_gpu, padder.pad_top, padder.pad_left,
                             orig_h, orig_w, &maps->score, error)) {
                return false;
            }
        }
#if defined(AICORE_VULKAN_ALIKED)
        if (b != nullptr && b->IsVulkan()) {
            if (!PinVulkanScoreMap(b, &maps->score, maps->height, maps->width,
                                   pipe, error)) {
                return false;
            }
            if (DkdDebugEnabled()) {
                LogScoreMapStage(b, maps->score, maps->height, maps->width,
                                 "crop_pin", error);
            }
            VkAlikedQueueIdle(b->handle);
        }
#endif
    }

    {
        StageBench bench("dense_tail");
        (void)maps;
    }
    BarrierGpuPipeline(b);
    return error == nullptr || error->empty();
}

bool ExtractDenseMapGpu(const TensorMap &tensors,
                        const std::vector<float> &image,
                        int32_t width,
                        int32_t height,
                        int32_t orig_h,
                        int32_t orig_w,
                        std::vector<float> *feature_map,
                        std::vector<float> *score_map,
                        std::string *error,
                        internal::Backend *backend) {
    if (backend == nullptr || !backend->IsGpu()) {
        if (error) {
            *error = "GPU pipeline requires a CUDA/Vulkan backend";
        }
        return false;
    }

    GpuDenseMaps maps;
    if (!ExtractDenseMapGpuVram(tensors, image, width, height, orig_h, orig_w,
                                &maps, error, backend, nullptr)) {
        return false;
    }
    return maps.feature.DownloadNchw(backend, feature_map, 128, maps.height,
                                     maps.width, error) &&
           maps.score.DownloadNchw(backend, score_map, 1, maps.height,
                                   maps.width, error);
}

}  // namespace lightglue::aliked_internal
