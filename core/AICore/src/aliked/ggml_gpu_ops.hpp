#pragma once

#include "../backend.h"

#include "ggml_cnn.hpp"
#include "gpu_tensor.hpp"

#include <string>

namespace lightglue::aliked_internal {

class GpuPipelineCache;

bool RunAvgPool2dGpu(internal::Backend *backend, const GpuTensor &input, int32_t kh,
                     int32_t kw, int32_t stride, GpuTensor *output,
                     std::string *error);

bool RunInterpolateGpu(internal::Backend *backend, const GpuTensor &input,
                       int32_t out_w, int32_t out_h, GpuTensor *output,
                       std::string *error);

bool RunSeluGpu(internal::Backend *backend, GpuTensor *tensor, std::string *error);

bool RunConvBnSeluGpu(GgmlConvRunner *runner, internal::Backend *backend,
                      const FusedConv2d &weights, const GpuTensor &input, int32_t pad,
                      int32_t stride, GpuTensor *output, const char *cache_key,
                      std::string *error);

bool RunAddGpu(internal::Backend *backend, GpuTensor *accum,
               const GpuTensor &other, std::string *error,
               const char *cache_key = nullptr);

bool RunConcatChannelGpu(internal::Backend *backend, const GpuTensor &a,
                         const GpuTensor &b, GpuTensor *output, std::string *error);

bool RunClampGpu(internal::Backend *backend, GpuTensor *tensor, float min_val,
                 float max_val, std::string *error);

bool RunSigmoidInPlaceGpu(internal::Backend *backend, GpuTensor *tensor,
                          std::string *error);

bool RunL2NormalizeChannelsGpu(internal::Backend *backend, GpuTensor *tensor,
                               int32_t channels, int32_t h, int32_t w,
                               std::string *error);

bool RunCropWhcnGpu(internal::Backend *backend, const GpuTensor &input,
                    int32_t pad_top, int32_t pad_left, int32_t out_h, int32_t out_w,
                    GpuTensor *output, std::string *error);

void ClearCachedGpuOpGraphs();

// Re-bind shared backend gallocr for all cached ggml op graphs (Vulkan extract).
void RebindAllCachedGgmlOpGraphs(internal::Backend *backend);

// Flush before each Vulkan extract; drop cached graphs after SDDH extracts only.
void BeginVulkanExtract(internal::Backend *backend);
void EndVulkanExtract(internal::Backend *backend, GpuPipelineCache *cache = nullptr);

// Back-compat alias for EndVulkanExtract.
void ResetVulkanExtractPipeline(internal::Backend *backend,
                                GpuPipelineCache *cache = nullptr);

} // namespace lightglue::aliked_internal