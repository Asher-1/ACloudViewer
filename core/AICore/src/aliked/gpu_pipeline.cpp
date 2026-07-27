// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "gpu_pipeline.hpp"

#include <ggml-backend.h>

#include "aliked_gpu_ops.hpp"
#include "deform_conv.hpp"
#include "ggml_cnn.hpp"
#include "ggml_gpu_ops.hpp"
#include "ggml_gpu_session.hpp"
#include "gpu_pipeline_cache.hpp"
#include "gpu_postprocess.hpp"
#include "gpu_tensor.hpp"
#include "model_weights.hpp"
#include "tensor_ops.hpp"
#if defined(LIGHTGLUE_HAS_CUDA)
#include <cuda_runtime.h>

#include "aliked_cuda.hpp"
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

bool UseCudaCustomKernels(internal::Backend *backend) {
#if defined(LIGHTGLUE_HAS_CUDA)
    return backend != nullptr && backend->IsCuda();
#else
    (void)backend;
    return false;
#endif
}

void SyncGpuPipeline(internal::Backend *backend) {
    if (backend != nullptr && backend->handle != nullptr) {
        ggml_backend_synchronize(backend->handle);
    }
#if defined(LIGHTGLUE_HAS_CUDA)
    if (backend != nullptr && backend->IsCuda()) {
        cudaDeviceSynchronize();
    }
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
#if defined(LIGHTGLUE_HAS_CUDA)
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
        ggml_backend_tensor_copy(input.tensor, output->tensor);
        return true;
    }
    if (!UseCudaCustomKernels(backend)) {
        return RunInterpolateGpu(backend, input, out_w, out_h, output, error);
    }
#if defined(LIGHTGLUE_HAS_CUDA)
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
#if defined(LIGHTGLUE_HAS_CUDA)
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
#if defined(LIGHTGLUE_HAS_CUDA)
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
#if defined(LIGHTGLUE_HAS_CUDA)
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
#if defined(LIGHTGLUE_HAS_CUDA)
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
    if (!UseCudaCustomKernels(backend)) {
        return RunL2NormalizeChannelsGpu(backend, tensor, channels, h, w,
                                         error);
    }
#if defined(LIGHTGLUE_HAS_CUDA)
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
                   std::string *error) {
    if (!UseCudaCustomKernels(backend)) {
        return RunAddGpu(backend, accum, other, error);
    }
#if defined(LIGHTGLUE_HAS_CUDA)
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

bool ResBlockForwardGpu(GpuPipelineCache *cache,
                        const GpuTensor &input,
                        int32_t ic,
                        int32_t oc,
                        const TensorMap &tensors,
                        const std::string &prefix,
                        bool dcn,
                        GpuTensor *output,
                        std::string *error) {
    GpuTensor conv1;
    if (dcn) {
        if (!DcnConvBnDispatch(
                    cache, input,
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
                    cache->ggml()->runner(), cache->backend(), input,
                    RequireTensor(tensors, prefix + "_conv1_weight", error), oc,
                    3, 3, RequireTensor(tensors, prefix + "_bn1_weight", error),
                    RequireTensor(tensors, prefix + "_bn1_bias", error),
                    RequireTensor(tensors, prefix + "_bn1_running_mean", error),
                    RequireTensor(tensors, prefix + "_bn1_running_var", error),
                    1, 1, &conv1, (prefix + ".conv1").c_str(), error)) {
            return false;
        }
    }
    if (dcn && !SeluGpu(cache->backend(), &conv1, error)) {
        return false;
    }

    GpuTensor conv2;
    if (dcn) {
        if (!DcnConvBnDispatch(
                    cache, conv1,
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
        if (!cache->ggml()->runner()->RunDevice(fused, conv1, &conv2, 1, 1,
                                                error,
                                                (prefix + ".conv2").c_str())) {
            return false;
        }
    }

    GpuTensor identity_buf;
    const GpuTensor *identity = &input;
    if (tensors.count(prefix + "_downsample_weight") > 0) {
        const std::vector<float> &down_b =
                RequireTensor(tensors, prefix + "_downsample_bias", error);
        if (!ConvGpu(cache->ggml()->runner(), input,
                     RequireTensor(tensors, prefix + "_downsample_weight",
                                   error),
                     oc, 1, 1, &down_b, 0, 1, &identity_buf,
                     (prefix + ".downsample").c_str(), error)) {
            return false;
        }
        identity = &identity_buf;
    } else if (ic != oc) {
        error->assign("residual channel mismatch without downsample for " +
                      prefix);
        return false;
    }

    if (!AddInPlaceGpu(cache->backend(), &conv2, *identity, error)) {
        return false;
    }
    if (!SeluGpu(cache->backend(), &conv2, error)) {
        return false;
    }
    *output = std::move(conv2);
    return true;
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
    GpuPipelineCache *active = cache != nullptr ? cache : &stack_cache;
    if (!active->Warmup(tensors, error)) {
        return false;
    }

    InputPadder padder(height, width, 32);
    const std::vector<float> padded = padder.Pad(image, 3, height, width);

    if (!active->EnsureInput(padder.padded_w, padder.padded_h, 3, error)) {
        return false;
    }
    GpuTensor &x1 = active->InputBuffer();
    if (!x1.UploadNchw(backend, padded, 3, padder.padded_h, padder.padded_w,
                       error)) {
        return false;
    }

    if (!ConvBnSeluGpu(active->ggml()->runner(), active->backend(), x1,
                       RequireTensor(tensors, "block1_conv1_weight", error), 16,
                       3, 3, RequireTensor(tensors, "block1_bn1_weight", error),
                       RequireTensor(tensors, "block1_bn1_bias", error),
                       RequireTensor(tensors, "block1_bn1_running_mean", error),
                       RequireTensor(tensors, "block1_bn1_running_var", error),
                       1, 1, &x1, "block1.conv1", error)) {
        return false;
    }
    if (!ConvBnSeluGpu(active->ggml()->runner(), active->backend(), x1,
                       RequireTensor(tensors, "block1_conv2_weight", error), 16,
                       3, 3, RequireTensor(tensors, "block1_bn2_weight", error),
                       RequireTensor(tensors, "block1_bn2_bias", error),
                       RequireTensor(tensors, "block1_bn2_running_mean", error),
                       RequireTensor(tensors, "block1_bn2_running_var", error),
                       1, 1, &x1, "block1.conv2", error)) {
        return false;
    }

    GpuTensor x2;
    if (!AvgPoolGpu(backend, x1, 2, 2, 2, &x2, error)) {
        return false;
    }
    if (!ResBlockForwardGpu(active, x2, 16, 32, tensors, "block2", false, &x2,
                            error)) {
        return false;
    }

    GpuTensor x3;
    if (!AvgPoolGpu(backend, x2, 4, 4, 4, &x3, error)) {
        return false;
    }
    if (!ResBlockForwardGpu(active, x3, 32, 64, tensors, "block3", true, &x3,
                            error)) {
        return false;
    }

    GpuTensor x4;
    if (!AvgPoolGpu(backend, x3, 4, 4, 4, &x4, error)) {
        return false;
    }
    if (!ResBlockForwardGpu(active, x4, 64, 128, tensors, "block4", true, &x4,
                            error)) {
        return false;
    }

    auto project = [&](const GpuTensor &src, const char *weight_name,
                       GpuTensor *dst) -> bool {
        return ConvSeluGpu(active->ggml()->runner(), active->backend(), src,
                           RequireTensor(tensors, weight_name, error), 32, 1, 1,
                           nullptr, 0, 1, dst, weight_name, error);
    };

    const int32_t fh = x1.h;
    const int32_t fw = x1.w;
    GpuTensor f1;
    if (!project(x1, "conv1_weight", &f1)) {
        return false;
    }
    GpuTensor f2;
    if (!project(x2, "conv2_weight", &f2)) {
        return false;
    }
    if (!UpsampleGpu(backend, f2, fw, fh, &f2, error)) {
        return false;
    }
    GpuTensor f3;
    if (!project(x3, "conv3_weight", &f3)) {
        return false;
    }
    if (!UpsampleGpu(backend, f3, fw, fh, &f3, error)) {
        return false;
    }
    GpuTensor f4;
    if (!project(x4, "conv4_weight", &f4)) {
        return false;
    }
    if (!UpsampleGpu(backend, f4, fw, fh, &f4, error)) {
        return false;
    }

    GpuTensor fused;
    if (!ConcatChannelGpu(backend, f1, f2, &fused, error)) {
        return false;
    }
    GpuTensor fused2;
    if (!ConcatChannelGpu(backend, fused, f3, &fused2, error)) {
        return false;
    }
    GpuTensor feature_gpu;
    if (!ConcatChannelGpu(backend, fused2, f4, &feature_gpu, error)) {
        return false;
    }

    std::vector<GgmlGpuSession::ConvChainSpec> score_layers =
            active->ScoreHeadLayers();
    score_layers.push_back(
            {active->ScoreHeadFinal(), 1, 1, "score_head_6", false});
    GpuTensor score_gpu;
    SyncGpuPipeline(backend);
    if (!active->ggml()->RunFusedConvChainGraph(score_layers, feature_gpu,
                                                &score_gpu, error)) {
        return false;
    }
    SyncGpuPipeline(backend);
    if (!SigmoidInPlaceGpu(backend, &score_gpu, error)) {
        return false;
    }
    if (!L2NormalizeChannelsGpu(backend, &feature_gpu, 128, fh, fw, error)) {
        return false;
    }

    if (!CropWhcnGpu(backend, feature_gpu, padder.pad_top, padder.pad_left,
                     orig_h, orig_w, &maps->feature, error)) {
        return false;
    }
    if (!CropWhcnGpu(backend, score_gpu, padder.pad_top, padder.pad_left,
                     orig_h, orig_w, &maps->score, error)) {
        return false;
    }
    maps->height = orig_h;
    maps->width = orig_w;
    SyncGpuPipeline(backend);
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
