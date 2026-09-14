#pragma once

#include "tasks/aliked/backend.h"


#include <ggml.h>

#include <cstdint>
#include <string>
#include <vector>

namespace lightglue::aliked_internal {

class GpuPipelineCache;

// Device-resident WHCN tensor [W,H,C,N] (ggml layout).
struct GpuTensor {
  ggml_context *ctx = nullptr;
  ggml_backend_buffer_t buffer = nullptr;
  ggml_tensor *tensor = nullptr;
  int32_t w = 0;
  int32_t h = 0;
  int32_t c = 0;

  void Release();
  ~GpuTensor() { Release(); }

  GpuTensor() = default;
  GpuTensor(GpuTensor &&other) noexcept { *this = std::move(other); }
  GpuTensor &operator=(GpuTensor &&other) noexcept {
    if (this != &other) {
      Release();
      ctx = other.ctx;
      buffer = other.buffer;
      tensor = other.tensor;
      w = other.w;
      h = other.h;
      c = other.c;
      other.ctx = nullptr;
      other.buffer = nullptr;
      other.tensor = nullptr;
    }
    return *this;
  }
  GpuTensor(const GpuTensor &) = delete;
  GpuTensor &operator=(const GpuTensor &) = delete;

  static bool Allocate(internal::Backend *backend, int32_t w, int32_t h, int32_t c,
                       GpuTensor *out, std::string *error);

  bool UploadNchw(internal::Backend *backend, const std::vector<float> &nchw,
                  int32_t ic, int32_t ih, int32_t iw, std::string *error);

  bool DownloadNchw(internal::Backend *backend, std::vector<float> *nchw,
                    int32_t ic, int32_t ih, int32_t iw, std::string *error) const;

  size_t ElementCount() const {
    return static_cast<size_t>(w) * h * c;
  }
};

bool IsContiguousWhcn(const ggml_tensor *tensor, int32_t w, int32_t h, int32_t c);

// CUDA custom kernels read flat WHCN; re-pack if gallocr left padded strides.
bool EnsureDenseWhcn(internal::Backend *backend, GpuTensor *tensor,
                     std::string *error);

// Vulkan: GPU dense-copy + queue_idle, then host roundtrip fallback.
bool EnsureDenseWhcnGpu(internal::Backend *backend, GpuTensor *tensor,
                        std::string *error);

// Pin score map into DKD scratch before gallocr reuses CNN tensor slots (Vulkan).
bool PinVulkanScoreMap(internal::Backend *backend, GpuTensor *score, int32_t h,
                       int32_t w, GpuPipelineCache *cache, std::string *error);

// DKD boundary: fence CNN score head, always VkAliked dense-copy into scratch.
bool PrepareScoreMapForDkd(internal::Backend *backend, GpuTensor *score,
                           int32_t h, int32_t w, GpuPipelineCache *cache,
                           std::string *error);

// Always download + re-upload to a fresh contiguous WHCN buffer (Vulkan reuse).
bool ForceDenseWhcn(internal::Backend *backend, GpuTensor *tensor,
                    std::string *error);

// Sync GpuTensor::w/h/c from ggml_tensor::ne after backend ops.
void SyncGpuTensorMeta(GpuTensor *tensor);

void BackendTensorCopyCompat(internal::Backend *backend, const ggml_tensor *src,
                             ggml_tensor *dst);

// Log ggml nb[] strides when LIGHTGLUE_ALIKED_CONV_STRIDE_DEBUG=1, or when
// label contains ".offset" and DKD debug is enabled.
void LogTensorStrideIfDebug(const char *label, const ggml_tensor *tensor,
                            int32_t w, int32_t h, int32_t c);

} // namespace lightglue::aliked_internal
