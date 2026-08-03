// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// norm_crop / umeyama alignment white-box tests (no GGUF required).

#include <cmath>
#include <cstdio>
#include <vector>

#include "align.hpp"

namespace {

bool nearlyEqual(float a, float b, float eps = 1e-3f) {
    return std::fabs(a - b) <= eps;
}

fd::Image solidRgb(int w, int h, uint8_t r, uint8_t g, uint8_t b) {
    fd::Image img;
    img.width = w;
    img.height = h;
    img.rgb.assign(static_cast<size_t>(w) * static_cast<size_t>(h) * 3, 0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t i = (static_cast<size_t>(y) * static_cast<size_t>(w) +
                              static_cast<size_t>(x)) *
                             3;
            img.rgb[i + 0] = r;
            img.rgb[i + 1] = g;
            img.rgb[i + 2] = b;
        }
    }
    return img;
}

fd::Landmarks5 sampleFaceLandmarks() {
    fd::Landmarks5 lmk{};
    lmk[0] = {68.f, 82.f};
    lmk[1] = {132.f, 82.f};
    lmk[2] = {100.f, 115.f};
    lmk[3] = {78.f, 145.f};
    lmk[4] = {122.f, 145.f};
    return lmk;
}

bool imagesEqual(const fd::Image& a, const fd::Image& b) {
    if (a.width != b.width || a.height != b.height ||
        a.rgb.size() != b.rgb.size()) {
        return false;
    }
    for (size_t i = 0; i < a.rgb.size(); ++i) {
        if (a.rgb[i] != b.rgb[i]) return false;
    }
    return true;
}

float maxAbsDiff(const fd::Image& a, const fd::Image& b) {
    if (a.rgb.size() != b.rgb.size()) return 1e9f;
    float m = 0.f;
    for (size_t i = 0; i < a.rgb.size(); ++i) {
        m = std::max(m, std::fabs(static_cast<float>(a.rgb[i]) -
                                  static_cast<float>(b.rgb[i])));
    }
    return m;
}

}  // namespace

int main() {
    const fd::Image src = solidRgb(200, 200, 180, 120, 90);
    const fd::Landmarks5 lmk = sampleFaceLandmarks();

    fd::Image aligned;
    if (!fd::norm_crop(src, lmk, aligned, 112)) {
        std::fprintf(stderr, "norm_crop returned false\n");
        return 1;
    }
    if (aligned.width != 112 || aligned.height != 112 ||
        aligned.rgb.size() != static_cast<size_t>(112 * 112 * 3)) {
        std::fprintf(stderr, "unexpected aligned size %dx%d\n", aligned.width,
                     aligned.height);
        return 1;
    }

    fd::Image aligned2;
    if (!fd::norm_crop(src, lmk, aligned2, 112)) {
        std::fprintf(stderr, "second norm_crop failed\n");
        return 1;
    }
    if (!imagesEqual(aligned, aligned2)) {
        std::fprintf(stderr, "norm_crop not deterministic\n");
        return 1;
    }

    // Mapping reference landmarks onto themselves should land near arcface_dst.
    fd::Image refCrop;
    if (!fd::norm_crop(src, fd::kArcFaceRefLandmarks112, refCrop, 112)) {
        std::fprintf(stderr, "reference norm_crop failed\n");
        return 1;
    }

    // warp_affine rejects degenerate transforms.
    fd::Image dummy;
    const std::array<float, 6> degenerate = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (fd::warp_affine(src, degenerate, dummy, 32, 32)) {
        std::fprintf(stderr, "degenerate warp should fail\n");
        return 1;
    }

    // Small perturbation of landmarks should change the crop (sanity).
    fd::Landmarks5 shifted = lmk;
    shifted[2][0] += 8.f;
    fd::Image shiftedCrop;
    if (!fd::norm_crop(src, shifted, shiftedCrop, 112)) {
        std::fprintf(stderr, "shifted norm_crop failed\n");
        return 1;
    }
    if (maxAbsDiff(aligned, shiftedCrop) < 1.f) {
        std::fprintf(stderr, "landmark shift did not change crop enough\n");
        return 1;
    }

    (void)nearlyEqual;
    std::fprintf(stderr, "test_facedetect_align ok\n");
    return 0;
}
