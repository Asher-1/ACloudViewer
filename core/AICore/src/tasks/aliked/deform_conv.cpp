// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/aliked/deform_conv.hpp"

#include <algorithm>
#include <array>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace lightglue::aliked_internal {
namespace {

inline int32_t IndexNchw(
        int32_t c, int32_t y, int32_t x, int32_t h, int32_t w) {
    return c * h * w + y * w + x;
}

#if defined(_OPENMP)
int CpuDeformConvThreads() { return std::min(8, omp_get_max_threads()); }
#endif

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

    // Each output spatial position owns a disjoint set of output elements.
    // Parallelizing positions therefore preserves the per-element summation
    // order, which keeps ALIKED's CPU/Vulkan parity unchanged.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(CpuDeformConvThreads())
#endif
    for (int32_t oy = 0; oy < ih; ++oy) {
        for (int32_t ox = 0; ox < iw; ++ox) {
            // ALIKED's DCN layers have at most 128 output channels.  A fixed
            // local buffer avoids a heap allocation per output pixel while
            // letting every sampled input value be reused by all channels.
            if (oc <= 256) {
                std::array<float, 256> sums{};
                for (int32_t o = 0; o < oc; ++o) {
                    sums[static_cast<size_t>(o)] =
                            bias ? (*bias)[static_cast<size_t>(o)] : 0.0f;
                }
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
                            const float sample = BilinearSampleDeform(
                                    input, i, ih, iw, sample_y, sample_x);
                            const size_t kernel_offset =
                                    static_cast<size_t>(i) * kh * kw + ky * kw +
                                    kx;
                            const size_t output_stride =
                                    static_cast<size_t>(ic) * kh * kw;
                            for (int32_t o = 0; o < oc; ++o) {
                                sums[static_cast<size_t>(o)] +=
                                        sample * weight[static_cast<size_t>(o) *
                                                                output_stride +
                                                        kernel_offset];
                            }
                        }
                    }
                }
                for (int32_t o = 0; o < oc; ++o) {
                    (*output)[IndexNchw(o, oy, ox, ih, iw)] =
                            sums[static_cast<size_t>(o)];
                }
                continue;
            }

            // Preserve the general implementation for any future model with
            // an unusually wide deformable-convolution layer.
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
