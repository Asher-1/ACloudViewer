// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>

#include "util/alignment.h"
#include "util/bitmap.h"

namespace colmap {

struct LineSegment {
    CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector2d start;
    Eigen::Vector2d end;
};

enum class LineSegmentOrientation {
    HORIZONTAL = 1,
    VERTICAL = -1,
    UNDEFINED = 0,
};

// Detect line segments in the given bitmap image.
std::vector<LineSegment> DetectLineSegments(const Bitmap& bitmap,
                                            const double min_length = 3);

// Classify line segments into horizontal/vertical.
std::vector<LineSegmentOrientation> ClassifyLineSegmentOrientations(
        const std::vector<LineSegment>& segments,
        const double tolerance = 0.25);

}  // namespace colmap

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(colmap::LineSegment)
