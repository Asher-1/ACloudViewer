// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "rmbg_preprocess.hpp"

// Qt-based image decode/encode (matching the gaussian task pattern).
// QImage/QBuffer-based image decoding and encoding.
#include <QBuffer>
#include <QIODevice>
#include <QImage>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace rmbg {
namespace {

float cubic(float x) {
    x = std::fabs(x);
    if (x < 1.f) return ((1.5f * x - 2.5f) * x) * x + 1.f;
    if (x < 2.f) return ((-0.5f * x + 2.5f) * x - 4.f) * x + 2.f;
    return 0.f;
}

template <typename Sample>
float resize_bicubic(
        Sample sample, int sw, int sh, int dw, int dh, int x, int y) {
    const float sx = ((float)x + 0.5f) * sw / dw - 0.5f;
    const float sy = ((float)y + 0.5f) * sh / dh - 0.5f;
    const int ix = (int)std::floor(sx), iy = (int)std::floor(sy);
    float sum = 0.f, norm = 0.f;
    for (int ky = -1; ky <= 2; ++ky) {
        const float wy = cubic(sy - (iy + ky));
        const int py = std::clamp(iy + ky, 0, sh - 1);
        for (int kx = -1; kx <= 2; ++kx) {
            const float w = wy * cubic(sx - (ix + kx));
            const int px = std::clamp(ix + kx, 0, sw - 1);
            sum += sample(px, py) * w;
            norm += w;
        }
    }
    return norm != 0.f ? sum / norm : 0.f;
}

template <typename Sample>
float resize_bilinear(
        Sample sample, int sw, int sh, int dw, int dh, int x, int y) {
    const float sx = ((float)x + 0.5f) * sw / dw - 0.5f;
    const float sy = ((float)y + 0.5f) * sh / dh - 0.5f;
    const int x0 = std::clamp((int)std::floor(sx), 0, sw - 1);
    const int y0 = std::clamp((int)std::floor(sy), 0, sh - 1);
    const int x1 = std::min(x0 + 1, sw - 1);
    const int y1 = std::min(y0 + 1, sh - 1);
    const float dx = std::clamp(sx, 0.f, (float)sw - 1.f) - x0;
    const float dy = std::clamp(sy, 0.f, (float)sh - 1.f) - y0;
    return (1.f - dy) * ((1.f - dx) * sample(x0, y0) + dx * sample(x1, y0)) +
           dy * ((1.f - dx) * sample(x0, y1) + dx * sample(x1, y1));
}

}  // namespace

bool decode_preprocess(const void *bytes,
                       int length,
                       int size,
                       const float mean[3],
                       const float std[3],
                       std::vector<uint8_t> &original_rgba,
                       int &width,
                       int &height,
                       std::vector<float> &input_nchw,
                       std::string &err) {
    if (!bytes || length <= 0 || size <= 0) {
        err = "invalid image input";
        return false;
    }
    QImage img = QImage::fromData(static_cast<const uchar *>(bytes), length);
    if (img.isNull()) {
        err = "image decode failed";
        return false;
    }
    img = img.convertToFormat(QImage::Format_RGBA8888);
    width = img.width();
    height = img.height();
    original_rgba.resize((size_t)width * height * 4);
    for (int y = 0; y < height; ++y) {
        memcpy(original_rgba.data() + (size_t)y * width * 4,
               img.constScanLine(y), width * 4);
    }
    input_nchw.resize((size_t)3 * size * size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            for (int c = 0; c < 3; ++c) {
                const float interpolated = resize_bilinear(
                        [&](int px, int py) {
                            return original_rgba[((size_t)py * width + px) * 4 +
                                                 c] /
                                   255.f;
                        },
                        width, height, size, size, x, y);
                const float value =
                        std::lround(std::clamp(interpolated, 0.f, 1.f) *
                                    255.f) /
                        255.f;
                input_nchw[((size_t)c * size + y) * size + x] =
                        (value - mean[c]) / std[c];
            }
        }
    }
    return true;
}

bool decode_preprocess_rgb(const uint8_t *rgb,
                           int rgb_w,
                           int rgb_h,
                           int size,
                           const float mean[3],
                           const float std[3],
                           std::vector<uint8_t> &original_rgba,
                           int &width,
                           int &height,
                           std::vector<float> &input_nchw,
                           std::string &err) {
    if (!rgb || rgb_w <= 0 || rgb_h <= 0 || size <= 0) {
        err = "invalid rgb input";
        return false;
    }
    width = rgb_w;
    height = rgb_h;
    original_rgba.resize((size_t)rgb_w * rgb_h * 4);
    for (size_t i = 0; i < (size_t)rgb_w * rgb_h; ++i) {
        original_rgba[i * 4 + 0] = rgb[i * 3 + 0];
        original_rgba[i * 4 + 1] = rgb[i * 3 + 1];
        original_rgba[i * 4 + 2] = rgb[i * 3 + 2];
        original_rgba[i * 4 + 3] = 255;
    }
    input_nchw.resize((size_t)3 * size * size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            for (int c = 0; c < 3; ++c) {
                const float interpolated = resize_bilinear(
                        [&](int px, int py) {
                            return rgb[((size_t)py * rgb_w + px) * 3 + c] /
                                   255.f;
                        },
                        rgb_w, rgb_h, size, size, x, y);
                const float value =
                        std::lround(std::clamp(interpolated, 0.f, 1.f) *
                                    255.f) /
                        255.f;
                input_nchw[((size_t)c * size + y) * size + x] =
                        (value - mean[c]) / std[c];
            }
        }
    }
    return true;
}

bool upsample_alpha(const std::vector<float> &alpha,
                    int alpha_width,
                    int alpha_height,
                    int width,
                    int height,
                    std::vector<uint8_t> &out_alpha8,
                    std::string &err) {
    if (alpha_width <= 0 || alpha_height <= 0 || width <= 0 || height <= 0 ||
        alpha.size() != (size_t)alpha_width * alpha_height) {
        err = "invalid alpha matte shape";
        return false;
    }
    out_alpha8.resize((size_t)width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float value = resize_bicubic(
                    [&](int px, int py) {
                        return alpha[(size_t)py * alpha_width + px];
                    },
                    alpha_width, alpha_height, width, height, x, y);
            out_alpha8[(size_t)y * width + x] =
                    (uint8_t)std::lround(std::clamp(value, 0.f, 1.f) * 255.f);
        }
    }
    return true;
}

bool compose_alpha(std::vector<uint8_t> &rgba,
                   int width,
                   int height,
                   const std::vector<float> &alpha,
                   int alpha_width,
                   int alpha_height,
                   std::string &err) {
    if (width <= 0 || height <= 0 || alpha_width <= 0 || alpha_height <= 0 ||
        rgba.size() != (size_t)width * height * 4 ||
        alpha.size() != (size_t)alpha_width * alpha_height) {
        err = "invalid result image shape";
        return false;
    }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float value = resize_bicubic(
                    [&](int px, int py) {
                        return alpha[(size_t)py * alpha_width + px];
                    },
                    alpha_width, alpha_height, width, height, x, y);
            rgba[((size_t)y * width + x) * 4 + 3] =
                    (uint8_t)std::lround(std::clamp(value, 0.f, 1.f) * 255.f);
        }
    }
    return true;
}

bool encode_result_png(std::vector<uint8_t> &rgba,
                       int width,
                       int height,
                       const std::vector<float> &alpha,
                       int alpha_width,
                       int alpha_height,
                       std::vector<uint8_t> &png,
                       std::string &err) {
    /* Composite in-place: `rgba` is exclusively owned by the caller after the
     * request, so no defensive copy is needed (was a full-frame memcpy). */
    if (!compose_alpha(rgba, width, height, alpha, alpha_width, alpha_height,
                       err)) {
        return false;
    }
    png.clear();
    QImage qimg(rgba.data(), width, height, width * 4, QImage::Format_RGBA8888);
    QByteArray ba;
    QBuffer buf(&ba);
    if (!qimg.save(&buf, "PNG")) {
        err = "PNG encode failed";
        return false;
    }
    png.assign(reinterpret_cast<const uint8_t *>(ba.constData()),
               reinterpret_cast<const uint8_t *>(ba.constData()) + ba.size());
    return true;
}

}  // namespace rmbg
