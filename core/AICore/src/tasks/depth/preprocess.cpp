// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
namespace aicore {
namespace depth {

// ---- legacy bilinear path (unchanged) -------------------------------------
static void resize_bilinear(const Image& s,
                            int dw,
                            int dh,
                            std::vector<float>& dst_hwc) {
    dst_hwc.assign((size_t)dw * dh * 3, 0.f);
    const float sx = (float)s.w / dw, sy = (float)s.h / dh;
    for (int y = 0; y < dh; ++y) {
        float fy = (y + 0.5f) * sy - 0.5f;
        int y0 = (int)std::floor(fy);
        float wy = fy - y0;
        int y0c = std::clamp(y0, 0, s.h - 1),
            y1c = std::clamp(y0 + 1, 0, s.h - 1);
        for (int x = 0; x < dw; ++x) {
            float fx = (x + 0.5f) * sx - 0.5f;
            int x0 = (int)std::floor(fx);
            float wx = fx - x0;
            int x0c = std::clamp(x0, 0, s.w - 1),
                x1c = std::clamp(x0 + 1, 0, s.w - 1);
            for (int c = 0; c < 3; ++c) {
                auto P = [&](int yy, int xx) {
                    return (float)s.rgb[((size_t)yy * s.w + xx) * 3 + c];
                };
                float top = P(y0c, x0c) * (1 - wx) + P(y0c, x1c) * wx;
                float bot = P(y1c, x0c) * (1 - wx) + P(y1c, x1c) * wx;
                dst_hwc[((size_t)y * dw + x) * 3 + c] =
                        top * (1 - wy) + bot * wy;
            }
        }
    }
}
bool preprocess(const Image& img, const Config& cfg, Preprocessed& out) {
    if (img.w <= 0 || img.h <= 0 || cfg.img_mean.size() < 3 ||
        cfg.img_std.size() < 3)
        return false;
    const int patch = (int)cfg.patch_size;
    int dw = (img.w / patch) * patch, dh = (img.h / patch) * patch;
    if (dw == 0) dw = patch;
    if (dh == 0) dh = patch;
    std::vector<float> hwc;
    resize_bilinear(img, dw, dh, hwc);
    out.W = dw;
    out.H = dh;
    out.chw.assign((size_t)3 * dh * dw, 0.f);
    out.orig_w = img.w;
    out.orig_h = img.h;
    out.scale_w = (float)dw / img.w;
    out.scale_h = (float)dh / img.h;
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < dh; ++y)
            for (int x = 0; x < dw; ++x) {
                float v = hwc[((size_t)y * dw + x) * 3 + c] / 255.f;
                out.chw[((size_t)c * dh + y) * dw + x] =
                        (v - cfg.img_mean[c]) / cfg.img_std[c];
            }
    return true;
}

// ---- cv2-faithful uint8 resizers ------------------------------------------
// saturate_cast<uchar>(float): round half-to-even (cvRound) then clamp [0,255].
static inline uint8_t sat_u8(float v) {
    long r = std::lrintf(v);  // FE_TONEAREST -> ties to even (== cvRound)
    if (r < 0)
        r = 0;
    else if (r > 255)
        r = 255;
    return (uint8_t)r;
}

static inline float cubic_w(
        float x) {  // Catmull-Rom, a=-0.75 (cv2 INTER_CUBIC)
    const float a = -0.75f;
    x = std::fabs(x);
    if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.f;
}

// Separable bicubic, float intermediate, single saturate at the end.
// Coordinate mapping: src = (dst+0.5)*scale - 0.5, scale = src_size/dst_size.
// Source indices clamped to [0, src-1] (cv2 border replicate).
Image resize_cubic(const Image& src, int dw, int dh) {
    Image dst;
    dst.w = dw;
    dst.h = dh;
    dst.rgb.assign((size_t)dw * dh * 3, 0);
    if (src.w <= 0 || src.h <= 0 || dw <= 0 || dh <= 0) return dst;
    const int sw = src.w, sh = src.h;
    const double sx = (double)sw / dw, sy = (double)sh / dh;
    // Precompute x taps.
    std::vector<int> xidx((size_t)dw * 4);
    std::vector<float> xwt((size_t)dw * 4);
    for (int x = 0; x < dw; ++x) {
        double fx = (x + 0.5) * sx - 0.5;
        int ix = (int)std::floor(fx);
        float t = (float)(fx - ix);
        float w[4] = {cubic_w(t + 1), cubic_w(t), cubic_w(t - 1),
                      cubic_w(t - 2)};
        for (int k = 0; k < 4; ++k) {
            int s = std::clamp(ix - 1 + k, 0, sw - 1);
            xidx[(size_t)x * 4 + k] = s;
            xwt[(size_t)x * 4 + k] = w[k];
        }
    }
    // Horizontal pass: sh x dw x 3 float.
    std::vector<float> tmp((size_t)sh * dw * 3);
    for (int y = 0; y < sh; ++y)
        for (int x = 0; x < dw; ++x)
            for (int c = 0; c < 3; ++c) {
                float acc = 0;
                for (int k = 0; k < 4; ++k) {
                    int s = xidx[(size_t)x * 4 + k];
                    acc += xwt[(size_t)x * 4 + k] *
                           (float)src.rgb[((size_t)y * sw + s) * 3 + c];
                }
                tmp[((size_t)y * dw + x) * 3 + c] = acc;
            }
    // Vertical pass + saturate.
    for (int y = 0; y < dh; ++y) {
        double fy = (y + 0.5) * sy - 0.5;
        int iy = (int)std::floor(fy);
        float t = (float)(fy - iy);
        float w[4] = {cubic_w(t + 1), cubic_w(t), cubic_w(t - 1),
                      cubic_w(t - 2)};
        int yi[4];
        for (int k = 0; k < 4; ++k) yi[k] = std::clamp(iy - 1 + k, 0, sh - 1);
        for (int x = 0; x < dw; ++x)
            for (int c = 0; c < 3; ++c) {
                float acc = 0;
                for (int k = 0; k < 4; ++k)
                    acc += w[k] * tmp[((size_t)yi[k] * dw + x) * 3 + c];
                dst.rgb[((size_t)y * dw + x) * 3 + c] = sat_u8(acc);
            }
    }
    return dst;
}

// cv2 INTER_AREA decimation table (computeResizeAreaTab). One (di,si,alpha) per
// tap.
struct AreaTap {
    int di, si;
    float alpha;
};
static std::vector<AreaTap> area_tab(int ssize, int dsize) {
    std::vector<AreaTap> tab;
    const double scale = (double)ssize / dsize;
    for (int dx = 0; dx < dsize; ++dx) {
        double fsx1 = dx * scale, fsx2 = fsx1 + scale;
        double cellWidth = std::min(scale, (double)ssize - fsx1);
        int sx1 = (int)std::ceil(fsx1), sx2 = (int)std::floor(fsx2);
        sx2 = std::min(sx2, ssize - 1);
        sx1 = std::min(sx1, sx2);
        if (sx1 - fsx1 > 1e-3)
            tab.push_back({dx, sx1 - 1, (float)((sx1 - fsx1) / cellWidth)});
        for (int sx = sx1; sx < sx2; ++sx)
            tab.push_back({dx, sx, (float)(1.0 / cellWidth)});
        if (fsx2 - sx2 > 1e-3)
            tab.push_back(
                    {dx, sx2,
                     (float)(std::min(std::min(fsx2 - sx2, 1.0), cellWidth) /
                             cellWidth)});
    }
    return tab;
}

// Separable area resampling, float intermediate, single saturate at the end.
Image resize_area(const Image& src, int dw, int dh) {
    const int sw = src.w, sh = src.h;
    if (dw >= sw && dh >= sh) {
        // cv2 falls back to bilinear when INTER_AREA is asked to upscale; the
        // DA3 policy never does this, but stay safe.
        Image up;
        std::vector<float> hwc;
        resize_bilinear(src, dw, dh, hwc);
        up.w = dw;
        up.h = dh;
        up.rgb.resize((size_t)dw * dh * 3);
        for (size_t i = 0; i < up.rgb.size(); ++i) up.rgb[i] = sat_u8(hwc[i]);
        return up;
    }
    auto xtab = area_tab(sw, dw), ytab = area_tab(sh, dh);
    // Horizontal pass: sh x dw x 3 float.
    std::vector<float> tmp((size_t)sh * dw * 3, 0.f);
    for (int y = 0; y < sh; ++y)
        for (const auto& t : xtab)
            for (int c = 0; c < 3; ++c)
                tmp[((size_t)y * dw + t.di) * 3 + c] +=
                        t.alpha *
                        (float)src.rgb[((size_t)y * sw + t.si) * 3 + c];
    // Vertical pass: accumulate into dh x dw x 3 float, then saturate.
    std::vector<float> acc((size_t)dh * dw * 3, 0.f);
    for (const auto& t : ytab)
        for (int x = 0; x < dw; ++x)
            for (int c = 0; c < 3; ++c)
                acc[((size_t)t.di * dw + x) * 3 + c] +=
                        t.alpha * tmp[((size_t)t.si * dw + x) * 3 + c];
    Image dst;
    dst.w = dw;
    dst.h = dh;
    dst.rgb.resize((size_t)dw * dh * 3);
    for (size_t i = 0; i < acc.size(); ++i) dst.rgb[i] = sat_u8(acc[i]);
    return dst;
}

// ---- real DA3 upper/lower-bound resize policy -----------------------------
static inline int py_round(double x) {
    return (int)std::nearbyint(x);
}  // ties to even, == Python round()
static inline int nearest_multiple(int x, int p) {
    int down = (x / p) * p, up = down + p;
    return (std::abs(up - x) <= std::abs(x - down)) ? up : down;
}

bool preprocess_real(const Image& img,
                     const Config& cfg,
                     Preprocessed& out,
                     std::vector<uint8_t>* rgb_u8_out) {
    if (img.w <= 0 || img.h <= 0 || cfg.img_mean.size() < 3 ||
        cfg.img_std.size() < 3)
        return false;
    const int patch = (int)cfg.patch_size;
    const int target =
            cfg.img_resize_target > 0 ? (int)cfg.img_resize_target : 504;
    const bool upper =
            cfg.img_resize_mode.rfind("lower", 0) != 0;  // default: upper_bound

    const int ow = img.w, oh = img.h;
    Image cur = img;

    // Step 1: boundary resize (longest/shortest side -> target).
    {
        int bound = upper ? std::max(cur.w, cur.h) : std::min(cur.w, cur.h);
        if (bound != target) {
            double scale = (double)target / bound;
            int nw = std::max(1, py_round(cur.w * scale));
            int nh = std::max(1, py_round(cur.h * scale));
            cur = (scale > 1.0) ? resize_cubic(cur, nw, nh)
                                : resize_area(cur, nw, nh);
        }
    }
    // Step 2: make each dim divisible by patch via a small resize.
    {
        int nw = std::max(1, nearest_multiple(cur.w, patch));
        int nh = std::max(1, nearest_multiple(cur.h, patch));
        if (nw != cur.w || nh != cur.h) {
            bool upscale = (nw > cur.w) || (nh > cur.h);
            cur = upscale ? resize_cubic(cur, nw, nh)
                          : resize_area(cur, nw, nh);
        }
    }

    const int H = cur.h, W = cur.w;
    out.H = H;
    out.W = W;
    out.orig_w = ow;
    out.orig_h = oh;
    out.scale_w = (float)W / ow;
    out.scale_h = (float)H / oh;
    if (rgb_u8_out)
        *rgb_u8_out = cur.rgb;  // resized HWC RGB uint8 (pre-normalize)
    out.chw.assign((size_t)3 * H * W, 0.f);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float v = (float)cur.rgb[((size_t)y * W + x) * 3 + c] / 255.f;
                out.chw[((size_t)c * H + y) * W + x] =
                        (v - cfg.img_mean[c]) / cfg.img_std[c];
            }
    return true;
}
}  // namespace depth
}  // namespace aicore
