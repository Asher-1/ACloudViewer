// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ml/contrib/Cloud.h"

namespace cloudViewer {
namespace ml {
namespace contrib {

static_assert(std::is_standard_layout<PointXYZ>::value &&
                      std::is_trivial<PointXYZ>::value,
              "PointXYZ class must be a StandardLayout and TrivialType type.");

// Getters
// *******

PointXYZ max_point(std::vector<PointXYZ> points) {
    // Initialize limits
    PointXYZ maxP(points[0]);

    // Loop over all points
    for (auto p : points) {
        if (p.x > maxP.x) maxP.x = p.x;

        if (p.y > maxP.y) maxP.y = p.y;

        if (p.z > maxP.z) maxP.z = p.z;
    }

    return maxP;
}

PointXYZ min_point(std::vector<PointXYZ> points) {
    // Initialize limits
    PointXYZ minP(points[0]);

    // Loop over all points
    for (auto p : points) {
        if (p.x < minP.x) minP.x = p.x;

        if (p.y < minP.y) minP.y = p.y;

        if (p.z < minP.z) minP.z = p.z;
    }

    return minP;
}

}  // namespace contrib
}  // namespace ml
}  // namespace cloudViewer
