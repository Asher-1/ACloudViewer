// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Functional gate for the app-layer biggest-empty-rectangle finder
// (ccInnerRect2DFinder): input validation, dimension mapping and the
// recursive subdivision contract on synthetic clouds. Pure geometry, no
// QApplication required.

#include <ecvBox.h>
#include <ecvPointCloud.h>
#include <gtest/gtest.h>

#include <memory>

#include "ecvInnerRect2DFinder.h"

namespace {

// Dense border of the unit square in the (X, Y) plane; every point lies on
// the perimeter, so the strict-interior test sees an empty rectangle.
void addUnitSquareBorder(ccPointCloud& cloud,
                         unsigned char xDim,
                         unsigned char yDim) {
    const int steps = 20;
    for (int i = 0; i <= steps; ++i) {
        const double t = static_cast<double>(i) / steps;
        for (double edge : {0.0, 1.0}) {
            CCVector3 p(0, 0, 0);
            p.u[xDim] = static_cast<PointCoordinateType>(t);
            p.u[yDim] = static_cast<PointCoordinateType>(edge);
            cloud.addPoint(p);
            p.u[xDim] = static_cast<PointCoordinateType>(edge);
            p.u[yDim] = static_cast<PointCoordinateType>(t);
            cloud.addPoint(p);
        }
    }
}

}  // namespace

TEST(InnerRect2DFinder, RejectsInvalidInput) {
    ccInnerRect2DFinder finder;
    EXPECT_EQ(finder.process(nullptr), nullptr);

    ccPointCloud empty;
    EXPECT_EQ(finder.process(&empty), nullptr);

    ccPointCloud cloud;
    addUnitSquareBorder(cloud, 0, 1);
    EXPECT_EQ(finder.process(&cloud, 3), nullptr);  // zDim must be <= 2
}

TEST(InnerRect2DFinder, BorderOnlyCloudYieldsFullBoundingRect) {
    ccPointCloud cloud;
    addUnitSquareBorder(cloud, 0, 1);

    ccInnerRect2DFinder finder;
    std::unique_ptr<ccBox> box(finder.process(&cloud));
    ASSERT_NE(box, nullptr);

    // No point is strictly inside the bounding rect, so the result must be
    // the whole unit square in X/Y, centered on it.
    const CCVector3& dims = box->getDimensions();
    const float* center = box->getTransformation().getTranslation();
    EXPECT_NEAR(dims.x, 1.0, 1e-4);
    EXPECT_NEAR(dims.y, 1.0, 1e-4);
    EXPECT_NEAR(center[0], 0.5, 1e-4);
    EXPECT_NEAR(center[1], 0.5, 1e-4);
}

TEST(InnerRect2DFinder, CenterPointSplitsIntoHalfAreaQuadrants) {
    ccPointCloud cloud;
    addUnitSquareBorder(cloud, 0, 1);
    cloud.addPoint(CCVector3(0.5, 0.5, 0));

    ccInnerRect2DFinder finder;
    std::unique_ptr<ccBox> box(finder.process(&cloud));
    ASSERT_NE(box, nullptr);

    // The center point lies on the boundary of all four 0.5 x 1 / 1 x 0.5
    // sub-rectangles, so the biggest empty area is exactly half the square.
    const CCVector3& dims = box->getDimensions();
    const float* center = box->getTransformation().getTranslation();
    EXPECT_NEAR(dims.x * dims.y, 0.5, 1e-4);
    EXPECT_GE(center[0] - dims.x / 2, -1e-4);
    EXPECT_GE(center[1] - dims.y / 2, -1e-4);
    EXPECT_LE(center[0] + dims.x / 2, 1.0 + 1e-4);
    EXPECT_LE(center[1] + dims.y / 2, 1.0 + 1e-4);
}

TEST(InnerRect2DFinder, ZDimZeroMapsToYZPlane) {
    // zDim = 0 -> X = 1, Y = 2: the finder must operate in the YZ plane.
    ccPointCloud cloud;
    addUnitSquareBorder(cloud, 1, 2);

    ccInnerRect2DFinder finder;
    std::unique_ptr<ccBox> box(finder.process(&cloud, 0));
    ASSERT_NE(box, nullptr);

    const CCVector3& dims = box->getDimensions();
    const float* center = box->getTransformation().getTranslation();
    EXPECT_NEAR(dims.x, 0.0, 1e-4);
    EXPECT_NEAR(dims.y, 1.0, 1e-4);
    EXPECT_NEAR(dims.z, 1.0, 1e-4);
    EXPECT_NEAR(center[1], 0.5, 1e-4);
    EXPECT_NEAR(center[2], 0.5, 1e-4);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}