// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tensor_ops.hpp"

#include <algorithm>
#include <cmath>

namespace lightglue::aliked_internal {
namespace {

inline int32_t IndexNchw(
        int32_t c, int32_t y, int32_t x, int32_t h, int32_t w) {
    return c * h * w + y * w + x;
}

}  // namespace

void Conv2d(const std::vector<float> &input,
            int32_t ic,
            int32_t ih,
            int32_t iw,
            const std::vector<float> &weight,
            int32_t oc,
            int32_t kh,
            int32_t kw,
            const std::vector<float> *bias,
            int32_t pad,
            int32_t stride,
            std::vector<float> *output,
            int32_t *oh,
            int32_t *ow) {
    const std::vector<float> src = input;
    *oh = (ih + 2 * pad - kh) / stride + 1;
    *ow = (iw + 2 * pad - kw) / stride + 1;
    output->assign(static_cast<size_t>(oc) * (*oh) * (*ow), 0.0f);

    for (int32_t o = 0; o < oc; ++o) {
        for (int32_t oy = 0; oy < *oh; ++oy) {
            for (int32_t ox = 0; ox < *ow; ++ox) {
                float sum = bias ? (*bias)[o] : 0.0f;
                for (int32_t i = 0; i < ic; ++i) {
                    for (int32_t ky = 0; ky < kh; ++ky) {
                        for (int32_t kx = 0; kx < kw; ++kx) {
                            const int32_t iy = oy * stride + ky - pad;
                            const int32_t ix = ox * stride + kx - pad;
                            if (iy < 0 || ix < 0 || iy >= ih || ix >= iw) {
                                continue;
                            }
                            const size_t widx =
                                    static_cast<size_t>(o) * ic * kh * kw +
                                    static_cast<size_t>(i) * kh * kw + ky * kw +
                                    kx;
                            sum += src[IndexNchw(i, iy, ix, ih, iw)] *
                                   weight[widx];
                        }
                    }
                }
                (*output)[IndexNchw(o, oy, ox, *oh, *ow)] = sum;
            }
        }
    }
}

void BatchNorm2d(const std::vector<float> &input,
                 int32_t c,
                 int32_t h,
                 int32_t w,
                 const std::vector<float> &gamma,
                 const std::vector<float> &beta,
                 const std::vector<float> &mean,
                 const std::vector<float> &var,
                 std::vector<float> *output) {
    output->resize(input.size());
    const int32_t spatial = h * w;
    for (int32_t ch = 0; ch < c; ++ch) {
        const float inv_std =
                1.0f / std::sqrt(var[static_cast<size_t>(ch)] + kBnEps);
        const float scale = gamma[static_cast<size_t>(ch)] * inv_std;
        const float shift = beta[static_cast<size_t>(ch)] -
                            mean[static_cast<size_t>(ch)] * scale;
        const int32_t base = ch * spatial;
        for (int32_t i = 0; i < spatial; ++i) {
            (*output)[static_cast<size_t>(base + i)] =
                    input[static_cast<size_t>(base + i)] * scale + shift;
        }
    }
}

void ApplySelu(std::vector<float> *tensor) {
    for (float &value : *tensor) {
        value = Selu(value);
    }
}

void AvgPool2d(const std::vector<float> &input,
               int32_t c,
               int32_t h,
               int32_t w,
               int32_t kh,
               int32_t kw,
               int32_t stride,
               std::vector<float> *output,
               int32_t *oh,
               int32_t *ow) {
    *oh = (h - kh) / stride + 1;
    *ow = (w - kw) / stride + 1;
    output->assign(static_cast<size_t>(c) * (*oh) * (*ow), 0.0f);
    const float norm = 1.0f / static_cast<float>(kh * kw);
    for (int32_t ch = 0; ch < c; ++ch) {
        for (int32_t oy = 0; oy < *oh; ++oy) {
            for (int32_t ox = 0; ox < *ow; ++ox) {
                float sum = 0.0f;
                for (int32_t ky = 0; ky < kh; ++ky) {
                    for (int32_t kx = 0; kx < kw; ++kx) {
                        sum += input[IndexNchw(ch, oy * stride + ky,
                                               ox * stride + kx, h, w)];
                    }
                }
                (*output)[IndexNchw(ch, oy, ox, *oh, *ow)] = sum * norm;
            }
        }
    }
}

float BilinearSample(const std::vector<float> &tensor,
                     int32_t c,
                     int32_t h,
                     int32_t w,
                     int32_t channel,
                     float y,
                     float x) {
    if (h <= 0 || w <= 0 || c <= 0) {
        return 0.0f;
    }
    const size_t need = static_cast<size_t>(c) * static_cast<size_t>(h) *
                        static_cast<size_t>(w);
    if (tensor.size() < need) {
        return 0.0f;
    }
    const float clamped_y =
            std::min(std::max(y, 0.0f), static_cast<float>(h - 1));
    const float clamped_x =
            std::min(std::max(x, 0.0f), static_cast<float>(w - 1));
    const int32_t y0 = static_cast<int32_t>(std::floor(clamped_y));
    const int32_t x0 = static_cast<int32_t>(std::floor(clamped_x));
    const int32_t y1 = std::min(y0 + 1, h - 1);
    const int32_t x1 = std::min(x0 + 1, w - 1);
    const float ly = clamped_y - static_cast<float>(y0);
    const float lx = clamped_x - static_cast<float>(x0);
    const float hy = 1.0f - ly;
    const float hx = 1.0f - lx;
    const float v00 = tensor[IndexNchw(channel, y0, x0, h, w)];
    const float v01 = tensor[IndexNchw(channel, y0, x1, h, w)];
    const float v10 = tensor[IndexNchw(channel, y1, x0, h, w)];
    const float v11 = tensor[IndexNchw(channel, y1, x1, h, w)];
    return hy * hx * v00 + hy * lx * v01 + ly * hx * v10 + ly * lx * v11;
}

float BilinearSampleDeform(const std::vector<float> &tensor,
                           int32_t c,
                           int32_t h,
                           int32_t w,
                           float y,
                           float x) {
    if (y <= -1.0f || y >= static_cast<float>(h) || x <= -1.0f ||
        x >= static_cast<float>(w)) {
        return 0.0f;
    }

    const int32_t y0 = static_cast<int32_t>(std::floor(y));
    const int32_t x0 = static_cast<int32_t>(std::floor(x));
    const int32_t y1 = y0 + 1;
    const int32_t x1 = x0 + 1;
    const float ly = y - static_cast<float>(y0);
    const float lx = x - static_cast<float>(x0);
    const float hy = 1.0f - ly;
    const float hx = 1.0f - lx;

    auto at = [&](int32_t yy, int32_t xx) -> float {
        if (yy < 0 || yy >= h || xx < 0 || xx >= w) {
            return 0.0f;
        }
        return tensor[IndexNchw(c, yy, xx, h, w)];
    };

    return hy * hx * at(y0, x0) + hy * lx * at(y0, x1) + ly * hx * at(y1, x0) +
           ly * lx * at(y1, x1);
}

void UpsampleBilinear(const std::vector<float> &input,
                      int32_t c,
                      int32_t h,
                      int32_t w,
                      int32_t out_h,
                      int32_t out_w,
                      std::vector<float> *output) {
    if (h == out_h && w == out_w) {
        *output = input;
        return;
    }

    const std::vector<float> src = input;
    output->assign(static_cast<size_t>(c) * out_h * out_w, 0.0f);
    const float scale_h = out_h > 1 ? static_cast<float>(h - 1) /
                                              static_cast<float>(out_h - 1)
                                    : 0.0f;
    const float scale_w = out_w > 1 ? static_cast<float>(w - 1) /
                                              static_cast<float>(out_w - 1)
                                    : 0.0f;
    for (int32_t ch = 0; ch < c; ++ch) {
        for (int32_t oy = 0; oy < out_h; ++oy) {
            for (int32_t ox = 0; ox < out_w; ++ox) {
                const float in_y = oy * scale_h;
                const float in_x = ox * scale_w;
                (*output)[IndexNchw(ch, oy, ox, out_h, out_w)] =
                        BilinearSample(src, c, h, w, ch, in_y, in_x);
            }
        }
    }
}

void ConcatChannel(const std::vector<float> &a,
                   int32_t ca,
                   const std::vector<float> &b,
                   int32_t cb,
                   int32_t h,
                   int32_t w,
                   std::vector<float> *output) {
    output->assign(static_cast<size_t>(ca + cb) * h * w, 0.0f);
    for (int32_t ch = 0; ch < ca; ++ch) {
        for (int32_t y = 0; y < h; ++y) {
            for (int32_t x = 0; x < w; ++x) {
                (*output)[IndexNchw(ch, y, x, h, w)] =
                        a[IndexNchw(ch, y, x, h, w)];
            }
        }
    }
    for (int32_t ch = 0; ch < cb; ++ch) {
        for (int32_t y = 0; y < h; ++y) {
            for (int32_t x = 0; x < w; ++x) {
                (*output)[IndexNchw(ca + ch, y, x, h, w)] =
                        b[IndexNchw(ch, y, x, h, w)];
            }
        }
    }
}

void L2NormalizeChannels(std::vector<float> *tensor,
                         int32_t c,
                         int32_t h,
                         int32_t w) {
    const int32_t spatial = h * w;
    for (int32_t i = 0; i < spatial; ++i) {
        float norm = 0.0f;
        for (int32_t ch = 0; ch < c; ++ch) {
            const float value = (*tensor)[IndexNchw(ch, i / w, i % w, h, w)];
            norm += value * value;
        }
        norm = std::sqrt(std::max(norm, 1e-12f));
        for (int32_t ch = 0; ch < c; ++ch) {
            (*tensor)[IndexNchw(ch, i / w, i % w, h, w)] /= norm;
        }
    }
}

void Sigmoid(std::vector<float> *tensor) {
    for (float &value : *tensor) {
        value = 1.0f / (1.0f + std::exp(-value));
    }
}

}  // namespace lightglue::aliked_internal
