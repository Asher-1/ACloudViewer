// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "common/image_utils.hpp"

#include <QImage>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <utility>

#include "common/capi_utils.hpp"

namespace aicore {
namespace img {

namespace {

bool rgbSize(int width, int height, size_t* size) {
    if (!size || width <= 0 || height <= 0) return false;
    const size_t w = static_cast<size_t>(width);
    const size_t h = static_cast<size_t>(height);
    if (w > std::numeric_limits<size_t>::max() / 3 / h) return false;
    *size = w * h * 3;
    return true;
}

}  // namespace

bool load_image_rgb(const std::string& path, Image& out) {
    QImage decoded(QString::fromUtf8(path.c_str()));
    if (decoded.isNull()) {
        return false;
    }

    aicore::capi::PackedRgb packed =
            aicore::capi::qimage_to_packed_rgb(decoded);
    if (packed.data == nullptr) {
        return false;
    }
    const size_t byteCount =
            static_cast<size_t>(packed.width) * packed.height * 3;

    Image result;
    result.width = packed.width;
    result.height = packed.height;
    result.rgb.assign(packed.data, packed.data + byteCount);
    result.row_stride_bytes = static_cast<size_t>(result.width) * 3;
    std::free(packed.data);
    out = std::move(result);
    return true;
}

bool image_from_rgb(const uint8_t* rgb, int width, int height, Image& out) {
    size_t byteCount = 0;
    if (!rgb || !rgbSize(width, height, &byteCount)) return false;

    Image result;
    result.width = width;
    result.height = height;
    result.rgb.assign(rgb, rgb + byteCount);
    result.row_stride_bytes = static_cast<size_t>(width) * 3;
    out = std::move(result);
    return true;
}

bool image_from_view(const aicore_image_view& view, Image& out) {
    if (view.data == nullptr || view.width <= 0 || view.height <= 0) {
        return false;
    }
    size_t channels = 0;
    bool bgr = false;
    switch (view.format) {
        case AICORE_IMAGE_RGB8:
            channels = 3;
            break;
        case AICORE_IMAGE_RGBA8:
            channels = 4;
            break;
        case AICORE_IMAGE_GRAY8:
            channels = 1;
            break;
        case AICORE_IMAGE_BGR8:
            channels = 3;
            bgr = true;
            break;
        case AICORE_IMAGE_BGRA8:
            channels = 4;
            bgr = true;
            break;
        default:
            return false;
    }
    const size_t row_bytes = static_cast<size_t>(view.width) * channels;
    if (view.row_stride_bytes < row_bytes ||
        static_cast<size_t>(view.height) >
                std::numeric_limits<size_t>::max() / row_bytes) {
        return false;
    }
    Image result;
    result.width = view.width;
    result.height = view.height;
    result.borrowed_data = view.data;
    result.row_stride_bytes = view.row_stride_bytes;
    result.channels = static_cast<int>(channels);
    result.bgr = bgr;
    out = std::move(result);
    return true;
}

// OpenCV cv2.resize INTER_LINEAR: the per-axis fixed-point coefficients match
// OpenCV's 11-bit alpha/beta layout, and the vertical multiply rounds through
// a 22-bit shift. The horizontal pass stays scalar (the SIMD path in OpenCV
// narrows the intermediate result to int16 first, so this version differs by
// at most 1 on a minority of pixels; parity gates feed the reference's exact
// resized pixels where that matters).
void cv_resize_linear_u8(
        const Image& src, uint8_t* dst, int dw, int dh, int cn) {
    const int sw = src.width;
    const int sh = src.height;
    const int BITS = 11;
    const float SCALE = (float)(1 << BITS);  // 2048
    auto sat_short = [](float v) -> int {
        int i = (int)std::lrint(v);
        return i < -32768 ? -32768 : (i > 32767 ? 32767 : i);
    };
    std::vector<int> xofs(dw);
    std::vector<int> ialpha(dw * 2);
    std::vector<int> yofs(dh);
    std::vector<int> ibeta(dh * 2);
    const double sxs = (double)sw / dw, sys = (double)sh / dh;
    for (int dx = 0; dx < dw; ++dx) {
        float fx = (float)((dx + 0.5) * sxs - 0.5);
        int sx = (int)std::floor(fx);
        fx -= sx;
        if (sx < 0) {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= sw - 1) {
            sx = sw - 1;
            fx = 0.f;
        }
        xofs[dx] = sx;
        ialpha[dx * 2] = sat_short((1.f - fx) * SCALE);
        ialpha[dx * 2 + 1] = sat_short(fx * SCALE);
    }
    for (int dy = 0; dy < dh; ++dy) {
        float fy = (float)((dy + 0.5) * sys - 0.5);
        int sy = (int)std::floor(fy);
        fy -= sy;
        if (sy < 0) {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= sh - 1) {
            sy = sh - 1;
            fy = 0.f;
        }
        yofs[dy] = sy;
        ibeta[dy * 2] = sat_short((1.f - fy) * SCALE);
        ibeta[dy * 2 + 1] = sat_short(fy * SCALE);
    }
    // Per-row fill: rows are independent (each reads a fixed two-row band
    // of the source). Large outputs (det pre-process resizes 736-row images
    // every frame) fill in parallel; small crops (rec, 48 rows) stay
    // serial — the per-call thread pool costs more than the fill itself.
    auto fillRow = [&](int dy) {
        const int sy0 = yofs[dy], sy1 = std::min(sy0 + 1, sh - 1);
        const int b0 = ibeta[dy * 2], b1 = ibeta[dy * 2 + 1];
        uint8_t* drow = dst + (size_t)dy * dw * cn;
        for (int dx = 0; dx < dw; ++dx) {
            const int sx = xofs[dx], sx1 = std::min(sx + 1, sw - 1);
            const int a0 = ialpha[dx * 2], a1 = ialpha[dx * 2 + 1];
            for (int c = 0; c < cn; ++c) {
                const int p0 = src.channel(sx, sy0, c) * a0 +
                               src.channel(sx1, sy0, c) * a1;
                const int p1 = src.channel(sx, sy1, c) * a0 +
                               src.channel(sx1, sy1, c) * a1;
                int v = ((int64_t)p0 * b0 + (int64_t)p1 * b1 + (1 << 21)) >> 22;
                v = v < 0 ? 0 : (v > 255 ? 255 : v);
                drow[dx * cn + c] = (uint8_t)v;
            }
        }
    };
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    if (hw > 1 && dh >= 256 && (unsigned)dh > hw) {
        std::atomic<int> nextRow{0};
        std::vector<std::thread> pool;
        pool.reserve(hw - 1);
        for (unsigned i = 1; i < hw; ++i)
            pool.emplace_back([&] {
                for (;;) {
                    const int dy = nextRow.fetch_add(1);
                    if (dy >= dh) return;
                    fillRow(dy);
                }
            });
        for (;;) {
            const int dy = nextRow.fetch_add(1);
            if (dy >= dh) break;
            fillRow(dy);
        }
        for (auto& t : pool) t.join();
    } else {
        for (int dy = 0; dy < dh; ++dy) fillRow(dy);
    }
}

}  // namespace img
}  // namespace aicore
