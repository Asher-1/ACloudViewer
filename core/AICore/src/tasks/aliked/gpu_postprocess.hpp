#pragma once

#include "backend.h"

#include "gpu_tensor.hpp"
#include "postprocess.hpp"

#if defined(AICORE_CUDA_ALIKED)
#include "cuda/aliked_cuda.hpp"
#endif

#include <string>
#include <vector>

namespace lightglue::aliked_internal {

class GpuPipelineCache;

struct GpuDenseMaps {
  GpuTensor feature;
  GpuTensor score;
  int32_t height = 0;
  int32_t width = 0;
};

struct GpuKeypointResult {
  GpuTensor keypoints_norm; // interleaved [x,y] x count, stored as w=2*count,h=1,c=1
  GpuTensor scores;
  int32_t count = 0;
};

bool RunDkdGpu(const GpuTensor &score_map, int32_t h, int32_t w,
               const DkdOptions &options, internal::Backend *backend,
               GpuKeypointResult *output, std::string *error,
#if defined(AICORE_CUDA_ALIKED)
               AlikedDkdScratch *scratch = nullptr
#else
               void *scratch = nullptr
#endif
);

bool RunSddhGpu(const GpuTensor &feature_map, int32_t dim, int32_t h, int32_t w,
                const GpuTensor &keypoints_norm, int32_t count, int32_t kernel_size,
                int32_t n_pos, const std::vector<float> &offset_0_w,
                const std::vector<float> &offset_0_b,
                const std::vector<float> &offset_2_w,
                const std::vector<float> &offset_2_b,
                const std::vector<float> &sf_conv_w,
                const std::vector<float> &agg_weights, internal::Backend *backend,
                GpuTensor *descriptors, std::string *error,
                GpuPipelineCache *cache = nullptr);

} // namespace lightglue::aliked_internal
