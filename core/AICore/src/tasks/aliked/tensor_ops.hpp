#pragma once

// CPU reference implementations of the ALIKED tensor ops. Each function
// mirrors the corresponding PyTorch op exactly (torchvision deform_conv2d,
// nn.Conv2d, nn.BatchNorm2d, F.avg_pool2d, F.interpolate(..., 'bilinear'),
// nn.SELU) so the GPU paths (CUDA / Vulkan) have a bit-exact reference.
//
// Layout convention (row-major, contiguous):
//   images: [C, H, W]  (batch = 1; no N dim)
//   conv weight: [OC, IC, KH, KW]
//   conv offset (deform): [offset_groups*2*KH*KW, H, W]
// Functions that take `std::vector<float>* output` write into a pre-sized
// vector; helpers that take `std::vector<float>* tensor` operate in place.

#include <cmath>
#include <cstdint>
#include <vector>

namespace lightglue::aliked_internal {

constexpr float kSeluAlpha = 1.6732632423543772848170429916717f;
constexpr float kSeluScale = 1.0507009873554804934193349852946f;
constexpr float kBnEps = 1e-5f;

inline float Selu(float x) {
  return x > 0.0f ? kSeluScale * x
                  : kSeluScale * kSeluAlpha * (std::exp(x) - 1.0f);
}

void Conv2d(const std::vector<float> &input, int32_t ic, int32_t ih, int32_t iw,
            const std::vector<float> &weight, int32_t oc, int32_t kh, int32_t kw,
            const std::vector<float> *bias, int32_t pad, int32_t stride,
            std::vector<float> *output, int32_t *oh, int32_t *ow);

void BatchNorm2d(const std::vector<float> &input, int32_t c, int32_t h, int32_t w,
                 const std::vector<float> &gamma, const std::vector<float> &beta,
                 const std::vector<float> &mean, const std::vector<float> &var,
                 std::vector<float> *output);

void ApplySelu(std::vector<float> *tensor);

void AvgPool2d(const std::vector<float> &input, int32_t c, int32_t h, int32_t w,
               int32_t kh, int32_t kw, int32_t stride, std::vector<float> *output,
               int32_t *oh, int32_t *ow);

void UpsampleBilinear(const std::vector<float> &input, int32_t c, int32_t h,
                      int32_t w, int32_t out_h, int32_t out_w,
                      std::vector<float> *output);

void ConcatChannel(const std::vector<float> &a, int32_t ca,
                   const std::vector<float> &b, int32_t cb, int32_t h, int32_t w,
                   std::vector<float> *output);

void L2NormalizeChannels(std::vector<float> *tensor, int32_t c, int32_t h,
                         int32_t w);

void Sigmoid(std::vector<float> *tensor);

float BilinearSample(const std::vector<float> &tensor, int32_t c, int32_t h,
                     int32_t w, int32_t channel, float y, float x);

// Matches torchvision deform_conv2d boundary rules (zero outside the map).
float BilinearSampleDeform(const std::vector<float> &tensor, int32_t c, int32_t h,
                           int32_t w, float y, float x);

} // namespace lightglue::aliked_internal
