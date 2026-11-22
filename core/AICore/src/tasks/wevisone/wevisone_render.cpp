// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/wevisone/wevisone_render.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "common/image_utils.hpp"
#include "tasks/wevisone/wevisone_image_prep.h"
#include "tasks/wevisone/wevisone_pipeline_common.h"

namespace wwv {

namespace {

constexpr int kBoxHalfWidth = 1;  // width-3 rectangle: center +- 1

void blend_px(std::vector<uint8_t>& rgb,
              int32_t w,
              int32_t h,
              int32_t x,
              int32_t y,
              uint8_t r,
              uint8_t g,
              uint8_t b,
              float alpha) {
    if (x < 0 || x >= w || y < 0 || y >= h) return;
    uint8_t* p = rgb.data() + (static_cast<size_t>(y) * w + x) * 3;
    p[0] = static_cast<uint8_t>(std::round(p[0] * (1.0f - alpha) + r * alpha));
    p[1] = static_cast<uint8_t>(std::round(p[1] * (1.0f - alpha) + g * alpha));
    p[2] = static_cast<uint8_t>(std::round(p[2] * (1.0f - alpha) + b * alpha));
}

}  // namespace

bool render_result_png(const aicore_image_view& img,
                       const float* boxes,
                       int32_t n_boxes,
                       const float* matte,
                       int32_t matte_w,
                       int32_t matte_h,
                       const std::string& out_png_path) {
    RgbImage base;
    std::string err;
    if (!view_to_rgb(img, &base, &err)) return false;
    std::vector<uint8_t> rgb = std::move(base.pixels);
    const int32_t w = base.width;
    const int32_t h = base.height;

    if (matte != nullptr && matte_w > 0 && matte_h > 0) {
        for (int32_t y = 0; y < h; ++y) {
            const int32_t my = y * matte_h / h;
            for (int32_t x = 0; x < w; ++x) {
                const int32_t mx = x * matte_w / w;
                const float a = matte[static_cast<size_t>(my) * matte_w + mx];
                if (a > 0.01f) {
                    blend_px(rgb, w, h, x, y, 0, 220, 0, a * 0.55f);
                }
            }
        }
    }
    for (int32_t bi = 0; bi < n_boxes; ++bi) {
        const float* b = boxes + 4 * bi;
        const int x0 = std::max(0, static_cast<int>(b[0]));
        const int y0 = std::max(0, static_cast<int>(b[1]));
        const int x1 = std::min(w - 1, static_cast<int>(b[2]));
        const int y1 = std::min(h - 1, static_cast<int>(b[3]));
        for (int d = -kBoxHalfWidth; d <= kBoxHalfWidth; ++d) {
            for (int x = x0; x <= x1; ++x) {
                blend_px(rgb, w, h, x, y0 + d, 255, 0, 0, 1.0f);
                blend_px(rgb, w, h, x, y1 + d, 255, 0, 0, 1.0f);
            }
            for (int y = y0; y <= y1; ++y) {
                blend_px(rgb, w, h, x0 + d, y, 255, 0, 0, 1.0f);
                blend_px(rgb, w, h, x1 + d, y, 255, 0, 0, 1.0f);
            }
        }
    }
    // PNG encode through the shared AICore Qt helper (export boundary; no
    // stb), then an atomic-enough file write.
    aicore_image_view view;
    view.data = rgb.data();
    view.width = w;
    view.height = h;
    view.row_stride_bytes = static_cast<size_t>(w) * 3;
    view.format = AICORE_IMAGE_RGB8;
    std::vector<uint8_t> png;
    if (!aicore::img::encode_png(view, png)) return false;
    return write_file_bytes(out_png_path, png.data(), png.size());
}

}  // namespace wwv
