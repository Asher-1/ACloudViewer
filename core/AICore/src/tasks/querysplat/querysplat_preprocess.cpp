// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Preprocessing — implementation. See querysplat_preprocess.hpp.
// resize_rgb_hwc / resize_planes are verbatim ports of upstream cpp_ggml
// image_io.cpp (the antialias semantics comments carry over).
#include "querysplat_preprocess.hpp"

#include <algorithm>
#include <cmath>

namespace qs {

// Upscaling on both axes: bit-exact replication of ATen's CPU
// upsample_bilinear2d (what F.interpolate(antialias=True) reduces to when
// nothing shrinks — verified 1.2e-7 on the DL3DV 270->512 geometry):
//   src = scale*(dst+0.5)-0.5, clamped to 0; i1 = i0+1 clamped;
//   val = h0*(w0*p00 + w1*p01) + h1*(w0*p10 + w1*p11) — the exact
// accumulation order of ATen's out_frame loop.
// Any axis shrinking (e.g. a 6205x4135 photo cropped to 4135x4135 -> 512, or
// the evaluator's 512 -> 256 GT): ATen's antialias path, two separable passes
// of a triangle filter of half-width support = max(scale, 1) with
// per-output-pixel normalization (matches F.interpolate(antialias=True) to
// ~2.4e-7 on the 4135 -> 512 geometry; impulse-response-derived). Plain
// bilinear there produces materially different model inputs (courtyard
// cpp-vs-py was 30 dB for exactly this reason).
void resize_rgb_hwc(
        const float* src, int sh, int sw, int H, int W, float* dst) {
    const float scale_x = (float)sw / (float)W;
    const float scale_y = (float)sh / (float)H;
    if (scale_x > 1.0f || scale_y > 1.0f) {
        const float sup_x = std::max(scale_x, 1.0f),
                    sup_y = std::max(scale_y, 1.0f);
        // pass 1: horizontal sw -> W (rows unchanged)
        std::vector<float> tmp((size_t)sh * W * 3);
        for (int y = 0; y < sh; y++) {
            for (int x = 0; x < W; x++) {
                const float c = scale_x * ((float)x + 0.5f) - 0.5f;
                const int lo = std::max(0, (int)std::floor(c - sup_x));
                const int hi = std::min(sw - 1, (int)std::ceil(c + sup_x));
                float wsum = 0.f;
                for (int j = lo; j <= hi; j++)
                    wsum += std::max(0.f,
                                     1.f - std::fabs((float)j - c) / sup_x);
                if (wsum <= 0.f) wsum = 1.f;
                float* d = &tmp[((size_t)y * W + x) * 3];
                d[0] = d[1] = d[2] = 0.f;
                for (int j = lo; j <= hi; j++) {
                    const float w =
                            std::max(0.f,
                                     1.f - std::fabs((float)j - c) / sup_x) /
                            wsum;
                    const float* p = &src[((size_t)y * sw + j) * 3];
                    d[0] += w * p[0];
                    d[1] += w * p[1];
                    d[2] += w * p[2];
                }
            }
        }
        // pass 2: vertical sh -> H
        for (int y = 0; y < H; y++) {
            const float c = scale_y * ((float)y + 0.5f) - 0.5f;
            const int lo = std::max(0, (int)std::floor(c - sup_y));
            const int hi = std::min(sh - 1, (int)std::ceil(c + sup_y));
            float wsum = 0.f;
            for (int j = lo; j <= hi; j++)
                wsum += std::max(0.f, 1.f - std::fabs((float)j - c) / sup_y);
            if (wsum <= 0.f) wsum = 1.f;
            for (int x = 0; x < W; x++) {
                float* d = &dst[((size_t)y * W + x) * 3];
                d[0] = d[1] = d[2] = 0.f;
                for (int j = lo; j <= hi; j++) {
                    const float w =
                            std::max(0.f,
                                     1.f - std::fabs((float)j - c) / sup_y) /
                            wsum;
                    const float* p = &tmp[((size_t)j * W + x) * 3];
                    d[0] += w * p[0];
                    d[1] += w * p[1];
                    d[2] += w * p[2];
                }
            }
        }
        return;
    }
    for (int y = 0; y < H; y++) {
        float sy = scale_y * ((float)y + 0.5f) - 0.5f;
        sy = sy < 0.0f ? 0.0f : sy;
        const int y0 = (int)sy;
        const int y1 = y0 < sh - 1 ? y0 + 1 : y0;
        const float ly = sy - (float)y0;
        const float h0 = 1.0f - ly, h1 = ly;
        for (int x = 0; x < W; x++) {
            float sx = scale_x * ((float)x + 0.5f) - 0.5f;
            sx = sx < 0.0f ? 0.0f : sx;
            const int x0 = (int)sx;
            const int x1 = x0 < sw - 1 ? x0 + 1 : x0;
            const float lx = sx - (float)x0;
            const float w0 = 1.0f - lx, w1 = lx;
            const float* p00 = &src[((size_t)y0 * sw + x0) * 3];
            const float* p01 = &src[((size_t)y0 * sw + x1) * 3];
            const float* p10 = &src[((size_t)y1 * sw + x0) * 3];
            const float* p11 = &src[((size_t)y1 * sw + x1) * 3];
            float* d = &dst[((size_t)y * W + x) * 3];
            for (int c = 0; c < 3; c++)
                d[c] = h0 * (w0 * p00[c] + w1 * p01[c]) +
                       h1 * (w0 * p10[c] + w1 * p11[c]);
        }
    }
}

std::vector<float> resize_planes(
        const float* src, int n, int h, int w, int H, int W) {
    std::vector<float> out((size_t)n * 3 * H * W);
    std::vector<float> a((size_t)h * w * 3), b((size_t)H * W * 3);
    for (int v = 0; v < n; v++) {
        const float* s = src + (size_t)v * 3 * h * w;
        for (int c = 0; c < 3; c++)
            for (size_t k = 0; k < (size_t)h * w; k++)
                a[k * 3 + c] = s[(size_t)c * h * w + k];
        resize_rgb_hwc(a.data(), h, w, H, W, b.data());
        float* d = out.data() + (size_t)v * 3 * H * W;
        for (int c = 0; c < 3; c++)
            for (size_t k = 0; k < (size_t)H * W; k++)
                d[(size_t)c * H * W + k] = b[k * 3 + c];
    }
    return out;
}

std::vector<float> preprocess_rgb8_view(
        const uint8_t* rgb8, int w, int h, int H, int W) {
    // Center square crop of side min(w, h) (torchvision ImageTransform
    // max_crop=True), normalized to [0,1] — the exact composition
    // load_images_as_frames applies after decoding.
    const int side = std::min(w, h);
    const int x0 = (w - side) / 2, y0 = (h - side) / 2;
    std::vector<float> sq((size_t)side * side * 3);
    for (int y = 0; y < side; y++) {
        for (int x = 0; x < side; x++) {
            const uint8_t* p = rgb8 + ((size_t)(y0 + y) * w + (x0 + x)) * 3;
            float* q = &sq[((size_t)y * side + x) * 3];
            q[0] = p[0] / 255.f;
            q[1] = p[1] / 255.f;
            q[2] = p[2] / 255.f;
        }
    }
    std::vector<float> rgb((size_t)W * H * 3);
    resize_rgb_hwc(sq.data(), side, side, H, W, rgb.data());
    // HWC interleaved -> (3,H,W) torch planes.
    std::vector<float> planes((size_t)3 * H * W);
    const size_t plane = (size_t)H * W;
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                planes[(size_t)c * plane + (size_t)y * W + x] =
                        rgb[((size_t)y * W + x) * 3 + c];
    return planes;
}

bool preprocess_view_bytes(const uint8_t* data,
                           int w,
                           int h,
                           int channels_per_pixel,
                           bool bgr_order,
                           int H,
                           int W,
                           std::vector<float>* out_frames) {
    if (!data || w <= 0 || h <= 0 || H <= 0 || W <= 0 || !out_frames) {
        return false;
    }
    switch (channels_per_pixel) {
        case 1:    // GRAY: replicate channels
        case 3:    // RGB / BGR
        case 4: {  // RGBA / BGRA: drop alpha
            break;
        }
        default:
            return false;
    }
    std::vector<uint8_t> rgb8;
    const uint8_t* src = data;
    if (channels_per_pixel != 3 || bgr_order) {
        rgb8.resize((size_t)w * h * 3);
        for (size_t k = 0; k < (size_t)w * h; k++) {
            const uint8_t* p = data + k * channels_per_pixel;
            if (channels_per_pixel == 1) {
                rgb8[k * 3 + 0] = rgb8[k * 3 + 1] = rgb8[k * 3 + 2] = p[0];
            } else if (bgr_order) {
                rgb8[k * 3 + 0] = p[2];
                rgb8[k * 3 + 1] = p[1];
                rgb8[k * 3 + 2] = p[0];
            } else {
                rgb8[k * 3 + 0] = p[0];
                rgb8[k * 3 + 1] = p[1];
                rgb8[k * 3 + 2] = p[2];
            }
        }
        src = rgb8.data();
    }
    *out_frames = preprocess_rgb8_view(src, w, h, H, W);
    return true;
}

}  // namespace qs
