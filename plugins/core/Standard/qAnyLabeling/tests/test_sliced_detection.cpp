// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "SlicedDetection.h"

TEST(SlicedDetection, ProducesSourceCompatibleSliceCoverage) {
    const QVector<QRect> slices =
            detectionSlices(QSize(1024, 768), QSize(512, 512), 0.2);
    ASSERT_EQ(slices.size(), 6);
    EXPECT_EQ(slices.front(), QRect(0, 0, 512, 512));
    EXPECT_EQ(slices.back().right(), 1023);
    EXPECT_EQ(slices.back().bottom(), 767);
}

TEST(SlicedDetection, MergesOverlappingBoxesWithinTheirClass) {
    const QVector<SlicedDetectionBox> input = {
            {QRectF(10, 10, 100, 100), 0.9f, 2, QStringLiteral("car")},
            {QRectF(20, 20, 100, 100), 0.7f, 2, QStringLiteral("car")},
            {QRectF(20, 20, 100, 100), 0.99f, 3, QStringLiteral("person")}};
    const auto output = mergeSlicedDetections(input, 0.5);
    ASSERT_EQ(output.size(), 2);
    int carIndex = -1, personIndex = -1;
    for (int i = 0; i < output.size(); ++i) {
        if (output[i].classId == 2) carIndex = i;
        if (output[i].classId == 3) personIndex = i;
    }
    ASSERT_GE(carIndex, 0);
    EXPECT_EQ(output[carIndex].bounds, QRectF(10, 10, 110, 110));
    EXPECT_GE(personIndex, 0);
    EXPECT_NE(personIndex, carIndex);
}
