// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "deform_conv.hpp"

#include <algorithm>

namespace lightglue::aliked_internal {
namespace {

inline int32_t IndexNchw(
        int32_t c, int32_t y, int32_t x, int32_t h, int32_t w) {
    return c * h * w + y * w + x;
}

}  // namespace

void DeformConv2d(const std::vector<float> &input,
                  int32_t ic,
                  int32_t ih,
                  int32_t iw,
                  const std::vector<float> &offset,
                  int32_t offset_groups,
                  const std::vector<float> &weight,
                  int32_t oc,
                  int32_t kh,
                  int32_t kw,
                  const std::vector<float> *bias,
                  int32_t pad,
                  std::vector<float> *output,
                  int32_t *oh,
                  int32_t *ow) {
    (void)offset_groups;
    *oh = ih;
    *ow = iw;
    output->assign(static_cast<size_t>(oc) * ih * iw, 0.0f);

    for (int32_t oy = 0; oy < ih; ++oy) {
        for (int32_t ox = 0; ox < iw; ++ox) {
            for (int32_t o = 0; o < oc; ++o) {
                float sum = bias ? (*bias)[o] : 0.0f;
                for (int32_t i = 0; i < ic; ++i) {
                    for (int32_t ky = 0; ky < kh; ++ky) {
                        for (int32_t kx = 0; kx < kw; ++kx) {
                            const int32_t k_idx = ky * kw + kx;
                            const int32_t off_c_y = k_idx * 2 + 0;
                            const int32_t off_c_x = k_idx * 2 + 1;
                            const float sample_y =
                                    static_cast<float>(oy - pad + ky) +
                                    offset[IndexNchw(off_c_y, oy, ox, ih, iw)];
                            const float sample_x =
                                    static_cast<float>(ox - pad + kx) +
                                    offset[IndexNchw(off_c_x, oy, ox, ih, iw)];
                            const size_t widx =
                                    static_cast<size_t>(o) * ic * kh * kw +
                                    static_cast<size_t>(i) * kh * kw + ky * kw +
                                    kx;
                            sum += BilinearSampleDeform(input, i, ih, iw,
                                                        sample_y, sample_x) *
                                   weight[widx];
                        }
                    }
                }
                (*output)[IndexNchw(o, oy, ox, ih, iw)] = sum;
            }
        }
    }
}

}  // namespace lightglue::aliked_internal
