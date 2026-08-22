// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/yolo/yolo_image.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace yolo {

namespace {

std::vector<float> resize_bilinear_float(
        const float* src, int sw, int sh, int dw, int dh) {
    std::vector<float> dst((size_t)dw * dh);
    const float fx = (float)sw / dw;
    const float fy = (float)sh / dh;
    std::vector<int> x0(dw), x1(dw), y0(dh), y1(dh);
    std::vector<float> wx(dw), wy(dh);
    for (int x = 0; x < dw; x++) {
        const float sx = (x + 0.5f) * fx - 0.5f;
        const int ix = (int)std::floor(sx);
        x0[x] = std::clamp(ix, 0, sw - 1);
        x1[x] = std::clamp(ix + 1, 0, sw - 1);
        wx[x] = sx - ix;
    }
    for (int y = 0; y < dh; y++) {
        const float sy = (y + 0.5f) * fy - 0.5f;
        const int iy = (int)std::floor(sy);
        y0[y] = std::clamp(iy, 0, sh - 1);
        y1[y] = std::clamp(iy + 1, 0, sh - 1);
        wy[y] = sy - iy;
    }
    for (int y = 0; y < dh; y++) {
        const int yc0 = y0[y], yc1 = y1[y];
        const float wyv = wy[y];
        for (int x = 0; x < dw; x++) {
            const float wxv = wx[x];
            const float v0 = src[(size_t)yc0 * sw + x0[x]] * (1.0f - wxv) +
                             src[(size_t)yc0 * sw + x1[x]] * wxv;
            const float v1 = src[(size_t)yc1 * sw + x0[x]] * (1.0f - wxv) +
                             src[(size_t)yc1 * sw + x1[x]] * wxv;
            dst[(size_t)y * dw + x] = v0 * (1.0f - wyv) + v1 * wyv;
        }
    }
    return dst;
}

}  // namespace

void letterbox_image(const Image& img,
                     int imgsz,
                     LetterboxInfo& info,
                     std::vector<float>& out) {
    const float r = std::min((float)imgsz / img.w, (float)imgsz / img.h);
    // nearbyint = round-half-to-even, matching Python round().
    const int new_w = (int)std::nearbyint(img.w * r);
    const int new_h = (int)std::nearbyint(img.h * r);

    // Ultralytics LetterBox(auto=True, center=True): mod stride first, then
    // split padding.
    int dw = (imgsz - new_w) % 32, dh = (imgsz - new_h) % 32;
    const float hw = dw / 2.0f, hh = dh / 2.0f;
    const int left = (int)std::nearbyint(hw - 0.1f),
              right = (int)std::nearbyint(hw + 0.1f);
    const int top = (int)std::nearbyint(hh - 0.1f),
              bottom = (int)std::nearbyint(hh + 0.1f);
    const int canvas_w = new_w + left + right;
    const int canvas_h = new_h + top + bottom;

    info = LetterboxInfo{r, left, top, new_w, new_h, canvas_w, canvas_h};

    const size_t plane = (size_t)canvas_w * canvas_h;
    out.resize(3 * plane);
    if (left || right || top || bottom) {
        constexpr float pad = 114.0f / 255.0f;
        for (int c = 0; c < 3; c++) {
            float* channel = out.data() + (size_t)c * plane;
            std::fill(channel, channel + (size_t)top * canvas_w, pad);
            std::fill(channel + (size_t)(top + new_h) * canvas_w,
                      channel + plane, pad);
            for (int y = top; y < top + new_h; y++) {
                float* row = channel + (size_t)y * canvas_w;
                std::fill(row, row + left, pad);
                std::fill(row + left + new_w, row + canvas_w, pad);
            }
        }
    }

    const float fx = (float)img.w / new_w;
    const float fy = (float)img.h / new_h;
    std::vector<int> x0(new_w), x1(new_w);
    std::vector<float> wx(new_w);
    for (int x = 0; x < new_w; x++) {
        const float sx = (x + 0.5f) * fx - 0.5f;
        const int ix0 = (int)std::floor(sx);
        x0[x] = std::clamp(ix0, 0, img.w - 1);
        x1[x] = std::clamp(ix0 + 1, 0, img.w - 1);
        wx[x] = sx - ix0;
    }

    // Each output row writes disjoint positions, so the resize loop is
    // embarrassingly parallel; OpenMP mirrors the upstream yolo-cli path
    // (which measures ~0.4 ms preprocess vs ~2.7 ms single-threaded here).
    // Cap at 8 threads: the row workload is small, so oversubscribing the
    // thread pool costs more in wakeup/sync than it saves.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(8) if (new_h >= 64)
#endif
    for (int y = 0; y < new_h; y++) {
        const float sy = (y + 0.5f) * fy - 0.5f;
        const int iy0 = (int)std::floor(sy);
        const int yc0 = std::clamp(iy0, 0, img.h - 1);
        const int yc1 = std::clamp(iy0 + 1, 0, img.h - 1);
        const float wy = sy - iy0;
        for (int x = 0; x < new_w; x++) {
            const size_t p00 = ((size_t)yc0 * img.w + x0[x]) * 3;
            const size_t p01 = ((size_t)yc0 * img.w + x1[x]) * 3;
            const size_t p10 = ((size_t)yc1 * img.w + x0[x]) * 3;
            const size_t p11 = ((size_t)yc1 * img.w + x1[x]) * 3;
            const size_t dst = (size_t)(y + top) * canvas_w + x + left;
            for (int c = 0; c < 3; c++) {
                const float v0 = img.rgb[p00 + c] +
                                 (img.rgb[p01 + c] - img.rgb[p00 + c]) * wx[x];
                const float v1 = img.rgb[p10 + c] +
                                 (img.rgb[p11 + c] - img.rgb[p10 + c]) * wx[x];
                const uint8_t value = (uint8_t)(v0 + (v1 - v0) * wy + 0.5f);
                out[(size_t)c * plane + dst] = value / 255.0f;
            }
        }
    }
}

void unscale_boxes(std::vector<Detection>& dets, const LetterboxInfo& info) {
    for (auto& d : dets) {
        d.x1 = (d.x1 - info.pad_w) / info.scale;
        d.y1 = (d.y1 - info.pad_h) / info.scale;
        d.x2 = (d.x2 - info.pad_w) / info.scale;
        d.y2 = (d.y2 - info.pad_h) / info.scale;
    }
}

void unscale_masks(std::vector<SegMask>& masks,
                   const LetterboxInfo& info,
                   int image_w,
                   int image_h) {
    if (image_w <= 0 || image_h <= 0) return;
    // Source pixel (x, y) sits at canvas coordinate
    // floor((p + 0.5) * scale - 0.5) + pad (inverse of the letterbox resize
    // sampling in letterbox_image); subtracting the window origin gives the
    // position inside mask.bits. Rounding is the same convention as
    // unscale_boxes, so the tint stays aligned with the boxes. Window loop
    // bounds below are a linear bounding box only — pixels mapping outside
    // the window are skipped, never clamped (clamping would smear the
    // window's edge row over the whole padded band above it).
    for (auto& mask : masks) {
        if (mask.w <= 0 || mask.h <= 0 ||
            mask.bits.size() < (size_t)mask.w * mask.h) {
            continue;
        }
        const int ix0 =
                std::clamp((int)std::floor((mask.x - info.pad_w) / info.scale),
                           0, image_w - 1);
        const int iy0 =
                std::clamp((int)std::floor((mask.y - info.pad_h) / info.scale),
                           0, image_h - 1);
        const int ix1 = std::clamp(
                (int)std::ceil((mask.x + mask.w - info.pad_w) / info.scale), 0,
                image_w);
        const int iy1 = std::clamp(
                (int)std::ceil((mask.y + mask.h - info.pad_h) / info.scale), 0,
                image_h);
        std::vector<uint8_t> full((size_t)image_w * image_h, 0);
        for (int y = iy0; y < iy1; ++y) {
            const int cy = (int)std::floor((y + 0.5f) * info.scale - 0.5f) +
                           info.pad_h - mask.y;
            if (cy < 0 || cy >= mask.h) continue;
            const size_t srcRow = (size_t)cy * mask.w;
            const size_t dstRow = (size_t)y * image_w;
            for (int x = ix0; x < ix1; ++x) {
                const int cx = (int)std::floor((x + 0.5f) * info.scale - 0.5f) +
                               info.pad_w - mask.x;
                if (cx < 0 || cx >= mask.w) continue;
                if (mask.bits[srcRow + cx]) {
                    full[dstRow + x] = 1;
                }
            }
        }
        mask.bits = std::move(full);
        mask.x = 0;
        mask.y = 0;
        mask.w = image_w;
        mask.h = image_h;
    }
}

std::vector<float> restore_depth(const std::vector<float>& depth,
                                 int depth_w,
                                 int depth_h,
                                 const LetterboxInfo& info,
                                 int image_w,
                                 int image_h) {
    if ((int)depth.size() != depth_w * depth_h || depth_w <= 0 ||
        depth_h <= 0 || image_w <= 0 || image_h <= 0) {
        return {};
    }
    std::vector<float> canvas = resize_bilinear_float(
            depth.data(), depth_w, depth_h, info.imgsz_w, info.imgsz_h);
    std::vector<float> crop((size_t)info.new_w * info.new_h);
    for (int y = 0; y < info.new_h; y++) {
        memcpy(crop.data() + (size_t)y * info.new_w,
               canvas.data() + (size_t)(y + info.pad_h) * info.imgsz_w +
                       info.pad_w,
               (size_t)info.new_w * sizeof(float));
    }
    return resize_bilinear_float(crop.data(), info.new_w, info.new_h, image_w,
                                 image_h);
}

}  // namespace yolo
