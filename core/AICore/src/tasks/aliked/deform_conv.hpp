#pragma once

#include "tasks/aliked/tensor_ops.hpp"


#include <vector>

namespace lightglue::aliked_internal {

/** Deformable convolution (CPU reference path of the ALIKED DCN block).
 *
 *  For each output pixel (oh, ow) the kernel samples the input at
 *  (oh*stride + kh + offset[...], ow*stride + kw + offset[...]) where the
 *  per-pixel offsets come from the ALIKED offset branch (2*kh*kw*offset_groups
 *  channels). Bilinear interpolation is applied at fractional sample
 *  positions; out-of-bounds samples are zero-filled (no padding clamping).
 *  The GPU path (aliked_cuda.cu / ggml_gpu_ops.cpp) mirrors this contract
 *  numerically.
 *
 *  Layouts (row-major): input [ic, ih, iw], offset [offset_groups*2*kh*kw,
 *  ih, iw], weight [oc, ic, kh, kw], output [oc, oh, ow]. stride is implied
 *  to be 1; dilation is not supported. */
void DeformConv2d(const std::vector<float> &input, int32_t ic, int32_t ih, int32_t iw,
                  const std::vector<float> &offset, int32_t offset_groups,
                  const std::vector<float> &weight, int32_t oc, int32_t kh, int32_t kw,
                  const std::vector<float> *bias, int32_t pad,
                  std::vector<float> *output, int32_t *oh, int32_t *ow);

} // namespace lightglue::aliked_internal
