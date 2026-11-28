// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// norm_crop / umeyama alignment white-box tests (no GGUF required).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "tasks/facedetect/align.hpp"

namespace {

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

}  // namespace

class FaceDetectAlignment : public ::testing::Test {
protected:
    const fd::Image source = solidRgb(200, 200, 180, 120, 90);
    const fd::Landmarks5 landmarks = sampleFaceLandmarks();
};

TEST_F(FaceDetectAlignment, NormCropProducesArcFaceDimensions) {
    fd::Image aligned;
    ASSERT_TRUE(fd::norm_crop(source, landmarks, aligned, 112));
    EXPECT_EQ(aligned.width, 112);
    EXPECT_EQ(aligned.height, 112);
    EXPECT_EQ(aligned.rgb.size(), static_cast<size_t>(112 * 112 * 3));
}

TEST_F(FaceDetectAlignment, NormCropIsDeterministic) {
    fd::Image first, second;
    ASSERT_TRUE(fd::norm_crop(source, landmarks, first, 112));
    ASSERT_TRUE(fd::norm_crop(source, landmarks, second, 112));
    EXPECT_EQ(first.width, second.width);
    EXPECT_EQ(first.height, second.height);
    EXPECT_EQ(first.rgb, second.rgb);
}

TEST_F(FaceDetectAlignment, AcceptsReferenceLandmarks) {
    fd::Image crop;
    EXPECT_TRUE(fd::norm_crop(source, fd::kArcFaceRefLandmarks112, crop, 112));
}

TEST_F(FaceDetectAlignment, WarpAffineRejectsDegenerateTransform) {
    fd::Image crop;
    const std::array<float, 6> degenerate = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    EXPECT_FALSE(fd::warp_affine(source, degenerate, crop, 32, 32));
}

TEST_F(FaceDetectAlignment, LandmarkShiftChangesCrop) {
    fd::Image aligned, shifted_crop;
    ASSERT_TRUE(fd::norm_crop(source, landmarks, aligned, 112));
    fd::Landmarks5 shifted = landmarks;
    shifted[2][0] += 8.f;
    ASSERT_TRUE(fd::norm_crop(source, shifted, shifted_crop, 112));
    ASSERT_EQ(aligned.rgb.size(), shifted_crop.rgb.size());
    float max_difference = 0.f;
    for (size_t i = 0; i < aligned.rgb.size(); ++i) {
        max_difference =
                std::max(max_difference,
                         std::fabs(static_cast<float>(aligned.rgb[i]) -
                                   static_cast<float>(shifted_crop.rgb[i])));
    }
    EXPECT_GE(max_difference, 1.f);
}
