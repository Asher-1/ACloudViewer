#pragma once

#include "../backend.h"

#include <ggml.h>

#include <cstdint>
#include <string>
#include <vector>

namespace lightglue::aliked_internal {

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

} // namespace lightglue::aliked_internal
