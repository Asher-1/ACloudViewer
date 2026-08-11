#pragma once

#include "tensor_ops.hpp"

#include <vector>

namespace lightglue::aliked_internal {

void DeformConv2d(const std::vector<float> &input, int32_t ic, int32_t ih, int32_t iw,
                  const std::vector<float> &offset, int32_t offset_groups,
                  const std::vector<float> &weight, int32_t oc, int32_t kh, int32_t kw,
                  const std::vector<float> *bias, int32_t pad,
                  std::vector<float> *output, int32_t *oh, int32_t *ow);

} // namespace lightglue::aliked_internal
