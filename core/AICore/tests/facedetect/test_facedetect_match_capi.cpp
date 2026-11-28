// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>

#include "aicore/facedetect_capi.h"

TEST(FaceDetectMatching, ComputesCosineDistanceMatrix) {
    const float inv = 1.0f / std::sqrt(2.0f);
    const float queries[] = {1.f, 0.f, 0.f, inv, inv, 0.f};
    const float gallery[] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
    float distances[6] = {};
    ASSERT_EQ(aicore_facedetect_cosine_distance_matrix(queries, 2, gallery, 3,
                                                       3, distances),
              0);
    EXPECT_NEAR(distances[0], 0.f, 1e-6f);
    EXPECT_NEAR(distances[1], 1.f, 1e-6f);
    EXPECT_NEAR(distances[3], 1.f - inv, 1e-6f);
    EXPECT_NEAR(distances[4], 1.f - inv, 1e-6f);
}

TEST(FaceDetectMatching, RejectsNullQueries) {
    const float gallery[] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
    float distances[6] = {};
    EXPECT_NE(aicore_facedetect_cosine_distance_matrix(nullptr, 1, gallery, 3,
                                                       3, distances),
              0);
}
