#pragma once

#include "gpu_pipeline_cache.hpp"

#include "gpu_postprocess.hpp"
#include "model_weights.hpp"

#include <string>
#include <vector>

namespace lightglue::aliked_internal {

bool ExtractDenseMapGpu(const TensorMap &tensors, const std::vector<float> &image,
                        int32_t width, int32_t height, int32_t orig_h, int32_t orig_w,
                        std::vector<float> *feature_map, std::vector<float> *score_map,
                        std::string *error, internal::Backend *backend);

bool ExtractDenseMapGpuVram(const TensorMap &tensors, const std::vector<float> &image,
                            int32_t width, int32_t height, int32_t orig_h, int32_t orig_w,
                            GpuDenseMaps *maps, std::string *error,
                            internal::Backend *backend,
                            GpuPipelineCache *cache = nullptr);

} // namespace lightglue::aliked_internal
