// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include "tasks/depth/reconstruct.hpp"

TEST(DepthReconstruct, IdentityBackProjectionPreservesColorAndProvenance) {
    const int H = 2, W = 2, N = 1;
    std::vector<float> depth(N * H * W, 2.0f);
    std::vector<float> conf(N * H * W, 2.0f);
    std::array<float, 9> K = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::array<float, 16> ext = {1, 0, 0, 0, 0, 1, 0, 0,
                                 0, 0, 1, 0, 0, 0, 0, 1};
    std::vector<uint8_t> img(H * W * 3);
    for (int i = 0; i < H * W; ++i) {
        img[3 * i + 0] = 10;
        img[3 * i + 1] = 20;
        img[3 * i + 2] = 30;
    }
    std::vector<const uint8_t*> images = {img.data()};
    std::vector<std::array<float, 9>> Ks = {K};
    std::vector<std::array<float, 16>> exts = {ext};
    const aicore::depth::WorldPoints points = aicore::depth::back_project(
            depth, conf, Ks, exts, images, H, W, N, 1.0f);
    ASSERT_EQ(points.xyz.size(), 4u * 3);
    ASSERT_EQ(points.rgb.size(), 4u * 3);
    ASSERT_EQ(points.frame.size(), 4u);
    ASSERT_EQ(points.u.size(), 4u);
    ASSERT_EQ(points.v.size(), 4u);
    // Pixel order: v outer, u inner; world coordinates are (2u, 2v, 2).
    int idx = 0;
    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            SCOPED_TRACE(::testing::Message() << "pixel " << u << "," << v);
            EXPECT_NEAR(points.xyz[3 * idx + 0], 2.0 * u, 1e-5);
            EXPECT_NEAR(points.xyz[3 * idx + 1], 2.0 * v, 1e-5);
            EXPECT_NEAR(points.xyz[3 * idx + 2], 2.0, 1e-5);
            EXPECT_EQ(points.rgb[3 * idx + 0], 10);
            EXPECT_EQ(points.rgb[3 * idx + 1], 20);
            EXPECT_EQ(points.rgb[3 * idx + 2], 30);
            EXPECT_EQ(points.frame[idx], 0);
            EXPECT_EQ(points.u[idx], u);
            EXPECT_EQ(points.v[idx], v);
            ++idx;
        }
    }
}

TEST(DepthReconstruct, PercentileInterpolatesWithoutMutatingInput) {
    const std::vector<float> expected = {1, 2, 3, 4};
    std::vector<float> values = expected;
    EXPECT_NEAR(aicore::depth::percentile_linear(values, 50.0), 2.5, 1e-9);
    EXPECT_NEAR(aicore::depth::percentile_linear(values, 40.0), 2.2, 1e-9);
    EXPECT_EQ(values, expected);
}

TEST(DepthReconstruct, RotationQuaternionRoundTripPreservesPositiveScalar) {
    // 90 degrees about Z, row-major.
    const std::array<float, 9> R = {0, -1, 0, 1, 0, 0, 0, 0, 1};
    const std::array<float, 4> q = aicore::depth::rotmat2qvec(R);
    EXPECT_GE(q[0], 0);
    const double qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    const std::array<double, 9> recovered = {
            1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz,
            2 * qz * qx + 2 * qw * qy,     2 * qx * qy + 2 * qw * qz,
            1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx,
            2 * qz * qx - 2 * qw * qy,     2 * qy * qz + 2 * qw * qx,
            1 - 2 * qx * qx - 2 * qy * qy};
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(recovered[i], R[i], 1e-6) << "element " << i;
    }
}
