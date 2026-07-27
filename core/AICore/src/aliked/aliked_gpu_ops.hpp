#pragma once

#include "../backend.h"

#include "gpu_pipeline_cache.hpp"
#include "gpu_postprocess.hpp"
#include "gpu_tensor.hpp"
#include "model_weights.hpp"

#include <string>
#include <vector>

namespace lightglue::aliked_internal {

// Phase 5d dispatch: CUDA custom kernels today; Vulkan compute (5d) or CPU bridge (5b).
enum class AlikedCustomOpBackend {
  kCpu,
  kCuda,
  kVulkanBridge, // GGML VRAM + CPU custom ops (5b)
  kVulkanCompute // Phase 5d SPIR-V compute (future)
};

AlikedCustomOpBackend DetectCustomOpBackend(internal::Backend *backend);

bool DcnConvBnDispatch(GpuPipelineCache *cache, const GpuTensor &input,
                       const std::vector<float> &offset_w,
                       const std::vector<float> &offset_b,
                       const std::vector<float> &regular_w, int32_t oc,
                       const std::vector<float> &gamma, const std::vector<float> &beta,
                       const std::vector<float> &mean, const std::vector<float> &var,
                       const std::string &cache_prefix, GpuTensor *output,
                       std::string *error);

bool RunDkdDispatch(const GpuTensor &score_map, int32_t h, int32_t w,
                    const DkdOptions &options, internal::Backend *backend,
                    GpuKeypointResult *result, std::string *error,
                    GpuPipelineCache *cache);

bool RunSddhDispatch(const GpuTensor &feature_map, int32_t descriptor_dim, int32_t fh,
                     int32_t fw, const std::vector<float> &keypoints_norm,
                     int32_t keypoint_count, int32_t kernel_size, int32_t n_pos,
                     const std::vector<float> &offset_0_w,
                     const std::vector<float> &offset_0_b,
                     const std::vector<float> &offset_2_w,
                     const std::vector<float> &offset_2_b,
                     const std::vector<float> &sf_conv_w,
                     const std::vector<float> &agg_weights,
                     internal::Backend *backend, GpuTensor *descriptors,
                     std::string *error, GpuPipelineCache *cache);

void ClearAlikedDcnParityEntries();
bool WriteAlikedDcnParityDump(const std::string &path, std::string *error);

} // namespace lightglue::aliked_internal
