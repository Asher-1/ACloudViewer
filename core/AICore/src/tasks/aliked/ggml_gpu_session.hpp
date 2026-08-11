#pragma once

#include "backend.h"

#include "ggml_cnn.hpp"
#include "gpu_tensor.hpp"

#include <string>
#include <vector>

namespace lightglue::aliked_internal {

// Wraps GgmlConvRunner with SELU conv chaining helpers.
class GgmlGpuSession {
public:
  explicit GgmlGpuSession(internal::Backend *backend);
  ~GgmlGpuSession();

  GgmlGpuSession(const GgmlGpuSession &) = delete;
  GgmlGpuSession &operator=(const GgmlGpuSession &) = delete;

  internal::Backend *backend() const { return backend_; }
  GgmlConvRunner *runner() { return &runner_; }
  const GgmlConvRunner *runner() const { return &runner_; }

  void ResetConvGraphCaches() { runner_.InvalidateDeviceGraphs(); }

  bool RunConv(const FusedConv2d &weights, const GpuTensor &input, GpuTensor *output,
               int32_t pad, int32_t stride, const char *cache_key,
               std::string *error);

  struct SeluConvSpec {
    FusedConv2d weights;
    int32_t pad = 0;
    int32_t stride = 1;
    const char *cache_key = nullptr;
    bool apply_selu = true;
  };

  struct ConvChainSpec {
    FusedConv2d weights;
    int32_t pad = 0;
    int32_t stride = 1;
    const char *cache_key = nullptr;
    bool apply_selu = false;
  };

  struct ScoreHeadSchedOptions {
    bool apply_sigmoid = false;
    bool apply_crop = false;
    int32_t crop_pad_top = 0;
    int32_t crop_pad_left = 0;
    int32_t crop_out_h = 0;
    int32_t crop_out_w = 0;
  };

  bool RunSeluConvChain(const std::vector<SeluConvSpec> &layers, const GpuTensor &input,
                        GpuTensor *output, std::string *error);

  // Single GGML graph: conv [+ ggml SELU]* … + optional final conv.
  bool RunFusedConvChainGraph(const std::vector<ConvChainSpec> &layers,
                              const GpuTensor &input, GpuTensor *output,
                              std::string *error);

  // Vulkan scheduler path: fused conv chain + optional sigmoid + optional crop.
  // When LIGHTGLUE_ALIKED_VULKAN_SCHED_TAIL=1, SELU prefix uses legacy gallocr;
  // only the final conv (+ sigmoid) runs through sched.
  bool RunScoreHeadSchedGraph(const std::vector<ConvChainSpec> &layers,
                              const GpuTensor &input, GpuTensor *output,
                              const ScoreHeadSchedOptions &opts,
                              std::string *error);

private:
  bool RunScoreHeadSchedGraphImpl(const std::vector<ConvChainSpec> &layers,
                                  const GpuTensor &input, GpuTensor *output,
                                  const ScoreHeadSchedOptions &opts,
                                  std::string *error);

  internal::Backend *backend_ = nullptr;
  GgmlConvRunner runner_;
};

} // namespace lightglue::aliked_internal
