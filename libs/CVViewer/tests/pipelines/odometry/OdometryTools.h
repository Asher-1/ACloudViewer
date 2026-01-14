// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "Image.h"
#include "tests/Tests.h"

namespace cloudViewer {
namespace tests {

namespace odometry_tools {
std::shared_ptr<cloudViewer::geometry::Image> GenerateImage(
        const int& width,
        const int& height,
        const int& num_of_channels,
        const int& bytes_per_channel,
        const float& vmin,
        const float& vmax,
        const int& seed);

// Shift the pixels left with a specified step.
void ShiftLeft(std::shared_ptr<cloudViewer::geometry::Image> image,
               const int& step);

// Shift the pixels up with a specified step.
void ShiftUp(std::shared_ptr<cloudViewer::geometry::Image> image,
             const int& step);

// Create dummy correspondence map object.
std::shared_ptr<cloudViewer::geometry::Image> CorrespondenceMap(
        const int& width,
        const int& height,
        const int& vmin,
        const int& vmax,
        const int& seed);

// Create dummy depth buffer object.
std::shared_ptr<cloudViewer::geometry::Image> DepthBuffer(const int& width,
                                                          const int& height,
                                                          const float& vmin,
                                                          const float& vmax,
                                                          const int& seed);
}  // namespace odometry_tools
}  // namespace tests
}  // namespace cloudViewer
