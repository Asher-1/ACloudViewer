#pragma once

#include "tasks/aliked/backend.h"


#include "tasks/aliked/gpu_tensor.hpp"


#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace lightglue::aliked_internal {

constexpr float kBnEpsGgml = 1e-5f;

struct FusedConv2d {
  std::vector<float> kernel; // GGML [KW, KH, IC, OC]
  std::vector<float> bias;   // [OC]
  int32_t ic = 0;
  int32_t oc = 0;
  int32_t kh = 0;
  int32_t kw = 0;
};

FusedConv2d FuseConvBn(const std::vector<float> &weight_nchw, int32_t oc,
                       int32_t ic, int32_t kh, int32_t kw,
                       const std::vector<float> *conv_bias,
                       const std::vector<float> &gamma,
                       const std::vector<float> &beta,
                       const std::vector<float> &mean,
                       const std::vector<float> &var);

// Fused BN weights in PyTorch NCHW [OC,IC,KH,KW] layout (for DCN CUDA).
struct FusedConv2dNchw {
  std::vector<float> kernel;
  std::vector<float> bias;
  int32_t ic = 0;
  int32_t oc = 0;
  int32_t kh = 0;
  int32_t kw = 0;
};

FusedConv2dNchw FuseConvBnNchw(const std::vector<float> &weight_nchw, int32_t oc,
                               int32_t ic, int32_t kh, int32_t kw,
                               const std::vector<float> *conv_bias,
                               const std::vector<float> &gamma,
                               const std::vector<float> &beta,
                               const std::vector<float> &mean,
                               const std::vector<float> &var);

void NchwToWhcn(const std::vector<float> &nchw, int32_t c, int32_t h, int32_t w,
                std::vector<float> *whcn);

void WhcnToNchw(const std::vector<float> &whcn, int32_t c, int32_t h, int32_t w,
                std::vector<float> *nchw);

// Reusable GGML conv runner with persistent fused weights on the backend.
class GgmlConvRunner {
public:
  struct CachedWeight {
    ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor *kernel = nullptr;
    ggml_tensor *bias = nullptr;
    bool owns_buffer = true;
    // Persistent device conv graph — avoids per-layer Vulkan buffer alloc.
    ggml_context *graph_ctx = nullptr;
    ggml_backend_buffer_t graph_buffer = nullptr;
    ggml_gallocr_t graph_gallocr = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_tensor *graph_in = nullptr;
    ggml_tensor *graph_out = nullptr;
    int32_t graph_ih = 0;
    int32_t graph_iw = 0;
    int32_t graph_ic = 0;
    int32_t graph_pad = 0;
    int32_t graph_stride = 0;
  };

  explicit GgmlConvRunner(internal::Backend *backend);
  ~GgmlConvRunner();

  GgmlConvRunner(const GgmlConvRunner &) = delete;
  GgmlConvRunner &operator=(const GgmlConvRunner &) = delete;

  bool Run(const FusedConv2d &weights, const std::vector<float> &input_nchw,
           int32_t ih, int32_t iw, int32_t pad, int32_t stride,
           std::vector<float> *output_nchw, int32_t *oh, int32_t *ow,
           std::string *error, const char *cache_key = nullptr);

  // Device-resident WHCN conv: input/output stay in VRAM.
  bool RunDevice(const FusedConv2d &weights, const GpuTensor &input, GpuTensor *output,
                 int32_t pad, int32_t stride, std::string *error,
                 const char *cache_key = nullptr);

  bool EnsureCachedPublic(const char *cache_key, const FusedConv2d &weights,
                          std::string *error) {
    return EnsureCached(cache_key, weights, error);
  }

  // Drop cached device conv graphs (keep fused weights) before each extract.
  void InvalidateDeviceGraphs();

  // Re-bind gallocr for all cached graphs (once per extract on Vulkan).
  void RebindAllDeviceGraphs();

  // Share persistent weight buffers from another runner (graph fields omitted).
  void ImportWeightEntriesFrom(const GgmlConvRunner &other);

  const CachedWeight &CachedEntry(const char *cache_key) const {
    return cache_.at(cache_key);
  }

private:
  bool EnsureCached(const char *cache_key, const FusedConv2d &weights,
                    std::string *error);
  bool RunGraph(ggml_tensor *kernel, ggml_tensor *bias, const FusedConv2d &weights,
                int32_t ih, int32_t iw, int32_t pad, int32_t stride,
                std::vector<float> *output_nchw, int32_t *oh, int32_t *ow,
                std::string *error);

  bool RunGraphDevice(ggml_tensor *kernel, ggml_tensor *bias,
                      const FusedConv2d &weights, const GpuTensor &input,
                      GpuTensor *output, int32_t pad, int32_t stride,
                      std::string *error, const char *cache_key = nullptr);

  bool EnsureDeviceGraph(CachedWeight *entry, const FusedConv2d &weights,
                         const GpuTensor &input, int32_t pad, int32_t stride,
                         std::string *error);

  internal::Backend *backend_ = nullptr;
  std::unordered_map<std::string, CachedWeight> cache_;
  std::vector<float> input_whcn_;
  std::vector<float> output_whcn_;
};

bool RunFusedConv2dGgml(internal::Backend *backend, const FusedConv2d &weights,
                        const std::vector<float> &input_nchw, int32_t ih,
                        int32_t iw, int32_t pad, int32_t stride,
                        std::vector<float> *output_nchw, int32_t *oh, int32_t *ow,
                        std::string *error);

} // namespace lightglue::aliked_internal
