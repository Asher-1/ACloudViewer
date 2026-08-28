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

void unscale_pose(std::vector<PoseDetection>& poses,
                  const LetterboxInfo& info) {
    for (auto& p : poses) {
        p.det.x1 = (p.det.x1 - info.pad_w) / info.scale;
        p.det.y1 = (p.det.y1 - info.pad_h) / info.scale;
        p.det.x2 = (p.det.x2 - info.pad_w) / info.scale;
        p.det.y2 = (p.det.y2 - info.pad_h) / info.scale;
        for (size_t k = 0; k + 1 < p.kpts.size(); k += 2) {
            p.kpts[k] = (p.kpts[k] - info.pad_w) / info.scale;
            p.kpts[k + 1] = (p.kpts[k + 1] - info.pad_h) / info.scale;
        }
    }
}

void unscale_obb(std::vector<OBBDetection>& obbs, const LetterboxInfo& info) {
    for (auto& o : obbs) {
        o.cx = (o.cx - info.pad_w) / info.scale;
        o.cy = (o.cy - info.pad_h) / info.scale;
        o.w /= info.scale;
        o.h /= info.scale;
    }
}

// torchvision antialias=True bilinear resize (two-pass, horizontal then
// vertical, float intermediate, uint8 round at the end). Downsample uses
// the ATen area-pixel weights w(i) = 1 - |i - src_idx| * (1/scale) over the
// support [src_idx - scale, src_idx + scale]; upsample (scale <= 1) falls
// back to the plain bilinear w = 1 - |i - src_idx| over 2 taps.
// src_idx = (dst + 0.5) * scale - 0.5 in both modes, matching
// F.interpolate(align_corners=False).
static void tv_resize_linear(
        const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
    std::vector<float> tmp((size_t)dw * sh * 3);
    const double scale_x = (double)sw / dw;
    const double inv_x = scale_x > 1.0 ? 1.0 / scale_x : 1.0;
    const double support_x = scale_x > 1.0 ? scale_x : 1.0;
    for (int yy = 0; yy < sh; yy++) {
        for (int xx = 0; xx < dw; xx++) {
            const double src_idx = scale_x * (xx + 0.5) - 0.5;
            const int i0 = std::max(0, (int)std::ceil(src_idx - support_x));
            const int i1 =
                    std::min(sw - 1, (int)std::floor(src_idx + support_x));
            float ww = 0.0f;
            float acc[3] = {0.0f, 0.0f, 0.0f};
            const uint8_t* row = src + (size_t)yy * sw * 3;
            for (int i = i0; i <= i1; i++) {
                const float w = (float)std::max(
                        0.0, 1.0 - std::abs((i - src_idx) * inv_x));
                if (w <= 0.0f) continue;
                for (int c = 0; c < 3; c++)
                    acc[c] += row[(size_t)i * 3 + c] * w;
                ww += w;
            }
            if (ww <= 0.0f) ww = 1.0f;
            float* out = &tmp[((size_t)yy * dw + xx) * 3];
            for (int c = 0; c < 3; c++) out[c] = acc[c] / ww;
        }
    }
    const double scale_y = (double)sh / dh;
    const double inv_y = scale_y > 1.0 ? 1.0 / scale_y : 1.0;
    const double support_y = scale_y > 1.0 ? scale_y : 1.0;
    for (int yy = 0; yy < dh; yy++) {
        const double src_idx = scale_y * (yy + 0.5) - 0.5;
        const int j0 = std::max(0, (int)std::ceil(src_idx - support_y));
        const int j1 = std::min(sh - 1, (int)std::floor(src_idx + support_y));
        for (int xx = 0; xx < dw; xx++) {
            float ww = 0.0f;
            float acc[3] = {0.0f, 0.0f, 0.0f};
            for (int j = j0; j <= j1; j++) {
                const float w = (float)std::max(
                        0.0, 1.0 - std::abs((j - src_idx) * inv_y));
                if (w <= 0.0f) continue;
                const float* px = &tmp[((size_t)j * dw + xx) * 3];
                for (int c = 0; c < 3; c++) acc[c] += px[c] * w;
                ww += w;
            }
            if (ww <= 0.0f) ww = 1.0f;
            uint8_t* out = &dst[((size_t)yy * dw + xx) * 3];
            for (int c = 0; c < 3; c++)
                out[c] = (uint8_t)std::clamp((int)(acc[c] / ww + 0.5f), 0, 255);
        }
    }
}

void classify_preprocess(const Image& img, int size, std::vector<float>& out) {
    // The released yolo26-cls checkpoint bakes its own transforms:
    // Resize(size, BILINEAR, antialias=True) on the shortest edge,
    // CenterCrop(size), then a plain /255 (ImageNet mean/std are NOT
    // applied).
    const int min_edge = std::min(img.w, img.h);
    // torchvision: new_long = int(size * long / short) — multiply before
    // divide (1080x810 -> int(224*1080/810) = 298, not
    // int(1080*0.2765...) = 223).
    const int new_w = std::max(1, (int)(size * (double)img.w / min_edge));
    const int new_h = std::max(1, (int)(size * (double)img.h / min_edge));

    std::vector<uint8_t> resized((size_t)new_w * new_h * 3);
    tv_resize_linear(img.rgb, img.w, img.h, resized.data(), new_w, new_h);

    const int left = (new_w - size) / 2, top = (new_h - size) / 2;
    const size_t plane = (size_t)size * size;
    out.resize(3 * plane);
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            const size_t src = ((size_t)(y + top) * new_w + x + left) * 3;
            for (int c = 0; c < 3; c++)
                out[(size_t)c * plane + (size_t)y * size + x] =
                        resized[src + c] / 255.0f;
        }
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
