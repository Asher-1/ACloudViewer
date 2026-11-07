// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "common/image_utils.hpp"

#include <QBuffer>
#include <QImage>
#include <algorithm>
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

bool encode_png(const aicore_image_view& view, std::vector<uint8_t>& out) {
    Image src;
    if (!image_from_view(view, src) ||
        view.row_stride_bytes > size_t(std::numeric_limits<int>::max()) ||
        size_t(view.height) >
                std::numeric_limits<size_t>::max() / view.row_stride_bytes)
        return false;
    // O(width * height). One RGBA working image (16 MiB at 2048 square)
    // accepts unaligned/padded caller rows without exposing Qt across the ABI.
    QImage image(view.width, view.height, QImage::Format_RGBA8888);
    if (image.isNull()) return false;
    for (int y = 0; y < view.height; ++y) {
        const uint8_t* row = src.row(y);
        uint8_t* dst = image.scanLine(y);
        for (int x = 0; x < view.width; ++x) {
            const uint8_t* pixel = row + size_t(x) * src.channels;
            dst[4 * size_t(x)] = pixel[src.channel_index(0)];
            dst[4 * size_t(x) + 1] = pixel[src.channel_index(1)];
            dst[4 * size_t(x) + 2] = pixel[src.channel_index(2)];
            dst[4 * size_t(x) + 3] = src.channels == 4 ? pixel[3] : 255;
        }
    }
    QByteArray encoded;
    QBuffer buffer(&encoded);
    if (!buffer.open(QIODevice::WriteOnly) || !image.save(&buffer, "PNG"))
        return false;
    const auto* bytes = reinterpret_cast<const uint8_t*>(encoded.constData());
    std::vector<uint8_t> result(bytes, bytes + encoded.size());
    out.swap(result);
    return true;
}

// OpenCV cv2.resize INTER_LINEAR on 8U, bit-exact (imgproc/src/resize.cpp):
//  - 11-bit alpha/beta from float (1 - f, f), rounded by saturate_cast;
//  - x: sx < 0 or sx >= w - 1 clamps the index AND zeroes fx;
//  - y: fy is kept as computed and only the two source row indices clamp
//    (resizeGeneric_Invoker), so the first/last upscaled rows still split
//    the weight across two copies of the edge row;
//  - vertical pass: the uchar VResizeLinear specialization (scalar tail and
//    VResizeLinearVec_32s8u alike) evaluates
//      ((b0 * (S0 >> 4)) >> 16) + ((b1 * (S1 >> 4)) >> 16) + 2) >> 2,
//    not the generic (b0 * S0 + b1 * S1 + 2^21) >> 22 rounding.
// The generic rounding and the clamped fy differed by 1 LSB on 3-11% of the
// pixels of real table/OCR inputs, enough to flip a borderline SLANet token
// (table_recognition_v2.jpg: rowspan 9 -> 4) on identical decoded pixels.
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
    // scale = 1 / inv_scale with inv_scale = dsize / ssize, as OpenCV does
    // (not ssize / dsize: the two differ in the last ulp for some sizes).
    const double sxs = 1.0 / ((double)dw / sw), sys = 1.0 / ((double)dh / sh);
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
        const int sy = (int)std::floor(fy);
        fy -= sy;
        yofs[dy] = sy;
        ibeta[dy * 2] = sat_short((1.f - fy) * SCALE);
        ibeta[dy * 2 + 1] = sat_short(fy * SCALE);
    }
    // Per-row fill: rows are independent (each reads a fixed two-row band
    // of the source). Large outputs (det pre-process resizes 736-row images
    // every frame) fill in parallel; small crops (rec, 48 rows) stay
    // serial — the per-call thread pool costs more than the fill itself.
    // Source addressing is hoisted out of the pixel loop (see Image::row):
    // same bytes as channel(x, y, c), without the per-sample branches.
    const int scn = src.channels;
    int cmap[4] = {0, 0, 0, 0};
    for (int c = 0; c < cn && c < 4; ++c) cmap[c] = src.channel_index(c);
    auto fillRow = [&](int dy) {
        const int sy0 = std::clamp(yofs[dy], 0, sh - 1);
        const int sy1 = std::clamp(yofs[dy] + 1, 0, sh - 1);
        const int b0 = ibeta[dy * 2], b1 = ibeta[dy * 2 + 1];
        const uint8_t* srow0 = src.row(sy0);
        const uint8_t* srow1 = src.row(sy1);
        uint8_t* drow = dst + (size_t)dy * dw * cn;
        for (int dx = 0; dx < dw; ++dx) {
            const int sx = xofs[dx], sx1 = std::min(sx + 1, sw - 1);
            const int a0 = ialpha[dx * 2], a1 = ialpha[dx * 2 + 1];
            const uint8_t* s00 = srow0 + (size_t)sx * scn;
            const uint8_t* s01 = srow0 + (size_t)sx1 * scn;
            const uint8_t* s10 = srow1 + (size_t)sx * scn;
            const uint8_t* s11 = srow1 + (size_t)sx1 * scn;
            for (int c = 0; c < cn; ++c) {
                const int ci = cmap[c];
                const int p0 = s00[ci] * a0 + s01[ci] * a1;
                const int p1 = s10[ci] * a0 + s11[ci] * a1;
                int v = (((b0 * (p0 >> 4)) >> 16) + ((b1 * (p1 >> 4)) >> 16) +
                         2) >>
                        2;
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
