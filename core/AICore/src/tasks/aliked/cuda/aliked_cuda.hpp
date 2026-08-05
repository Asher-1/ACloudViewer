#pragma once

#include <ggml-backend.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace lightglue::aliked_internal {

#if defined(AICORE_CUDA_ALIKED)

struct AlikedDkdScratch {
  AlikedDkdScratch();
  ~AlikedDkdScratch();

  AlikedDkdScratch(const AlikedDkdScratch &) = delete;
  AlikedDkdScratch &operator=(const AlikedDkdScratch &) = delete;

  void Ensure(int32_t count);
  void Release();

  struct Impl;
  std::unique_ptr<Impl> impl;

  friend bool AlikedCudaRunDkd(ggml_backend_t backend, const float *score_map, int32_t h,
                               int32_t w, int32_t radius, int32_t top_k, float scores_th,
                               int32_t n_limit, float *keypoints_norm, float *scores,
                               int32_t *out_count, AlikedDkdScratch *scratch);
};

struct AlikedSddhScratch {
  AlikedSddhScratch();
  ~AlikedSddhScratch();

  AlikedSddhScratch(const AlikedSddhScratch &) = delete;
  AlikedSddhScratch &operator=(const AlikedSddhScratch &) = delete;

  void Ensure(int32_t count, int32_t dim, int32_t kernel_size);
  void Release();

  struct Impl;
  std::unique_ptr<Impl> impl;

  friend bool AlikedCudaRunSddh(ggml_backend_t backend, const float *feature_map,
                                int32_t dim, int32_t h, int32_t w,
                                const float *keypoints_norm, int32_t count,
                                int32_t kernel_size, int32_t n_pos,
                                const float *offset_0_w, const float *offset_0_b,
                                const float *offset_2_w, const float *offset_2_b,
                                const float *sf_conv_w, const float *agg_weights,
                                float *descriptors, AlikedSddhScratch *scratch);
};

bool AlikedCudaDeformConv2d(ggml_backend_t backend, const float *input, int32_t ic,
                            int32_t ih, int32_t iw, const float *offset,
                            const float *weight, const float *bias, int32_t oc,
                            int32_t kh, int32_t kw, int32_t pad, float *output);

bool AlikedCudaWhcnToNchw(ggml_backend_t backend, const float *whcn, float *nchw,
                          int32_t c, int32_t h, int32_t w);

bool AlikedCudaNchwToWhcn(ggml_backend_t backend, const float *nchw, float *whcn,
                          int32_t c, int32_t h, int32_t w);

bool AlikedCudaConcatChannel(ggml_backend_t backend, const float *a, int32_t ca,
                             const float *b, int32_t cb, int32_t h, int32_t w,
                             float *output);

bool AlikedCudaAddInPlace(ggml_backend_t backend, float *dst, const float *src,
                          size_t count);

bool AlikedCudaApplySelu(ggml_backend_t backend, float *data, size_t count);

bool AlikedCudaClampInPlace(ggml_backend_t backend, float *data, size_t count,
                            float min_value, float max_value);

bool AlikedCudaSigmoidInPlace(ggml_backend_t backend, float *data, size_t count);

bool AlikedCudaL2NormalizeChannels(ggml_backend_t backend, float *data, int32_t c,
                                   int32_t h, int32_t w);

bool AlikedCudaAvgPool2d(ggml_backend_t backend, const float *input, int32_t ic,
                         int32_t ih, int32_t iw, int32_t kh, int32_t kw,
                         int32_t stride, float *output, int32_t oh, int32_t ow);

bool AlikedCudaUpsampleBilinear(ggml_backend_t backend, const float *input, int32_t ic,
                                int32_t ih, int32_t iw, int32_t out_h, int32_t out_w,
                                float *output);

bool AlikedCudaCropWhcn(ggml_backend_t backend, const float *input, int32_t ic,
                        int32_t padded_h, int32_t padded_w, int32_t pad_top,
                        int32_t pad_left, int32_t out_h, int32_t out_w, float *output);

bool AlikedCudaRunDkd(ggml_backend_t backend, const float *score_map, int32_t h,
                      int32_t w, int32_t radius, int32_t top_k, float scores_th,
                      int32_t n_limit, float *keypoints_norm, float *scores,
                      int32_t *out_count, AlikedDkdScratch *scratch = nullptr);

bool AlikedCudaRunSddh(ggml_backend_t backend, const float *feature_map, int32_t dim,
                       int32_t h, int32_t w, const float *keypoints_norm, int32_t count,
                       int32_t kernel_size, int32_t n_pos, const float *offset_0_w,
                       const float *offset_0_b, const float *offset_2_w,
                       const float *offset_2_b, const float *sf_conv_w,
                       const float *agg_weights, float *descriptors,
                       AlikedSddhScratch *scratch = nullptr);

#else

struct AlikedDkdScratch;
struct AlikedSddhScratch;

inline bool AlikedCudaDeformConv2d(ggml_backend_t, const float *, int32_t, int32_t,
                                   int32_t, const float *, const float *, const float *,
                                   int32_t, int32_t, int32_t, int32_t, float *) {
  return false;
}

inline bool AlikedCudaConcatChannel(ggml_backend_t, const float *, int32_t, const float *,
                                    int32_t, int32_t, int32_t, float *) {
  return false;
}

inline bool AlikedCudaWhcnToNchw(ggml_backend_t, const float *, float *, int32_t, int32_t,
                                 int32_t) {
  return false;
}

inline bool AlikedCudaNchwToWhcn(ggml_backend_t, const float *, float *, int32_t, int32_t,
                                 int32_t) {
  return false;
}

inline bool AlikedCudaAddInPlace(ggml_backend_t, float *, const float *, size_t) {
  return false;
}

inline bool AlikedCudaApplySelu(ggml_backend_t, float *, size_t) { return false; }

inline bool AlikedCudaClampInPlace(ggml_backend_t, float *, size_t, float, float) {
  return false;
}

inline bool AlikedCudaSigmoidInPlace(ggml_backend_t, float *, size_t) { return false; }

inline bool AlikedCudaL2NormalizeChannels(ggml_backend_t, float *, int32_t, int32_t,
                                          int32_t) {
  return false;
}

inline bool AlikedCudaAvgPool2d(ggml_backend_t, const float *, int32_t, int32_t, int32_t,
                                int32_t, int32_t, int32_t, float *, int32_t, int32_t) {
  return false;
}

inline bool AlikedCudaUpsampleBilinear(ggml_backend_t, const float *, int32_t, int32_t,
                                       int32_t, int32_t, int32_t, float *) {
  return false;
}

inline bool AlikedCudaCropWhcn(ggml_backend_t, const float *, int32_t, int32_t, int32_t,
                               int32_t, int32_t, int32_t, int32_t, float *) {
  return false;
}

inline bool AlikedCudaRunDkd(ggml_backend_t, const float *, int32_t, int32_t, int32_t,
                             int32_t, float, int32_t, float *, float *, int32_t *,
                             AlikedDkdScratch *) {
  return false;
}

inline bool AlikedCudaRunSddh(ggml_backend_t, const float *, int32_t, int32_t, int32_t,
                              const float *, int32_t, int32_t, int32_t, const float *,
                              const float *, const float *, const float *,
                              const float *, const float *, float *, AlikedSddhScratch *) {
  return false;
}

#endif

} // namespace lightglue::aliked_internal
