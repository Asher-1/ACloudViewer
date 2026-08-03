#pragma once

#include "../backend.h"

#include "ggml_gpu_session.hpp"
#include "gpu_tensor.hpp"
#include "model_weights.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(AICORE_CUDA_ALIKED)
#include "cuda/aliked_cuda.hpp"
#endif

namespace lightglue::aliked_internal {

#if defined(AICORE_VULKAN_ALIKED)
struct AlikedVulkanDkdScratch {
  GpuTensor nms;
  GpuTensor tmp_a;
  GpuTensor tmp_b;
  GpuTensor tmp_c;
  GpuTensor block_keys;
  GpuTensor block_indices;
  GpuTensor indices_dev;
  GpuTensor score_contig;
  int32_t map_count = 0;
  int32_t max_kpts = 0;
  int32_t score_h = 0;
  int32_t score_w = 0;
};

struct AlikedVulkanSddhScratch {
  GpuTensor workspace;
  GpuTensor feature_contig;
  int32_t capacity_count = 0;
  int32_t dim = 0;
  int32_t kernel_size = 0;
  int32_t feat_h = 0;
  int32_t feat_w = 0;
};
#endif

// Persistent VRAM caches reused across ExtractFromRgb calls.
class GpuPipelineCache {
public:
  struct CachedDcnWeights {
    GpuTensor weight;
    GpuTensor bias;
  };

  struct CachedSddhWeights {
    GpuTensor offset_0_w;
    GpuTensor offset_0_b;
    GpuTensor offset_2_w;
    GpuTensor offset_2_b;
    GpuTensor sf_conv_w;
    GpuTensor agg_weights;
  };

  explicit GpuPipelineCache(internal::Backend *backend);
  ~GpuPipelineCache();

  GpuPipelineCache(const GpuPipelineCache &) = delete;
  GpuPipelineCache &operator=(const GpuPipelineCache &) = delete;

  internal::Backend *backend() const { return backend_; }
  GgmlGpuSession *ggml() { return &ggml_; }
  const GgmlGpuSession *ggml() const { return &ggml_; }
  GgmlGpuSession *ComputeGgml() { return &compute_ggml_; }
  const GgmlGpuSession *ComputeGgml() const { return &compute_ggml_; }
  bool IsWarmedUp() const { return warmed_up_; }

  bool Warmup(const TensorMap &tensors, std::string *error);

  bool EnsureComputeLinked(std::string *error);

  // Copy persistent weight handles from an already-warmed cache (stack fallback).
  bool ShareWarmStateFrom(const GpuPipelineCache &source, std::string *error);

  bool EnsureInput(int32_t w, int32_t h, int32_t c, std::string *error);
  GpuTensor &InputBuffer() { return input_buffer_; }

  bool EnsureDcnWeight(const std::string &key, const FusedConv2dNchw &fused,
                       std::string *error);
  const float *DcnWeightPtr(const std::string &key) const;
  const float *DcnBiasPtr(const std::string &key) const;

  bool EnsureDcnWorkspace(int32_t w, int32_t h, int32_t in_c, int32_t out_c,
                          std::string *error);
  float *DcnNchwInPtr();
  float *DcnNchwOffsetPtr();
  float *DcnNchwOutPtr();
  ggml_tensor *DcnNchwInTensor() { return dcn_nchw_in_.tensor; }
  ggml_tensor *DcnNchwOffsetTensor() { return dcn_nchw_offset_.tensor; }
  ggml_tensor *DcnNchwOutTensor() { return dcn_nchw_out_.tensor; }
  const CachedDcnWeights *FindDcnWeight(const std::string &key) const;

  const std::vector<GgmlGpuSession::ConvChainSpec> &ScoreHeadLayers() const {
    return score_head_layers_;
  }
  const FusedConv2d &ScoreHeadFinal() const { return score_head_final_; }

  bool EnsureSddhWeights(const std::vector<float> &offset_0_w,
                         const std::vector<float> &offset_0_b,
                         const std::vector<float> &offset_2_w,
                         const std::vector<float> &offset_2_b,
                         const std::vector<float> &sf_conv_w,
                         const std::vector<float> &agg_weights, std::string *error);
  const float *SddhOffset0WPtr() const;
  const float *SddhOffset0BPtr() const;
  const float *SddhOffset2WPtr() const;
  const float *SddhOffset2BPtr() const;
  const float *SddhSfConvWPtr() const;
  const float *SddhAggWeightsPtr() const;
  bool HasSddhWeights() const { return sddh_loaded_; }
  const CachedSddhWeights &SddhWeightTensors() const { return sddh_weights_; }

#if defined(AICORE_CUDA_ALIKED)
  AlikedDkdScratch *dkd_scratch() { return dkd_scratch_.get(); }
  AlikedSddhScratch *sddh_scratch() { return sddh_scratch_.get(); }
#endif

#if defined(AICORE_VULKAN_ALIKED)
  bool EnsureVulkanDkdScratch(int32_t h, int32_t w, int32_t max_kpts, std::string *error);
  AlikedVulkanDkdScratch *vulkan_dkd_scratch() { return &vulkan_dkd_scratch_; }
  bool EnsureVulkanSddhScratch(int32_t count, int32_t dim, int32_t kernel_size,
                               int32_t feat_h, int32_t feat_w, std::string *error);
  AlikedVulkanSddhScratch *vulkan_sddh_scratch() { return &vulkan_sddh_scratch_; }
#endif

private:
  internal::Backend *backend_ = nullptr;
  GgmlGpuSession ggml_;
  GgmlGpuSession compute_ggml_;
  bool compute_linked_ = false;
  std::unordered_map<std::string, CachedDcnWeights> dcn_weights_;
  GpuTensor dcn_nchw_in_;
  GpuTensor dcn_nchw_offset_;
  GpuTensor dcn_nchw_out_;
  int32_t dcn_ws_w_ = 0;
  int32_t dcn_ws_h_ = 0;
  int32_t dcn_ws_in_c_ = 0;
  int32_t dcn_ws_out_c_ = 0;

  std::vector<GgmlGpuSession::ConvChainSpec> score_head_layers_;
  FusedConv2d score_head_final_;
  bool warmed_up_ = false;

  CachedSddhWeights sddh_weights_;
  bool sddh_loaded_ = false;

  GpuTensor input_buffer_;
  int32_t input_w_ = 0;
  int32_t input_h_ = 0;
  int32_t input_c_ = 0;

#if defined(AICORE_CUDA_ALIKED)
  std::unique_ptr<AlikedDkdScratch> dkd_scratch_;
  std::unique_ptr<AlikedSddhScratch> sddh_scratch_;
#endif

#if defined(AICORE_VULKAN_ALIKED)
  AlikedVulkanDkdScratch vulkan_dkd_scratch_;
  AlikedVulkanSddhScratch vulkan_sddh_scratch_;
#endif
};

} // namespace lightglue::aliked_internal
