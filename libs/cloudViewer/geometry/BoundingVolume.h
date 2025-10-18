// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Legacy AABB API adapter mapped to core/include/BoundingBox.h

#include <ecvBBox.h>
#include <ecvOrientedBBox.h>

#include "cloudViewer/geometry/Geometry.h"

namespace cloudViewer {
namespace geometry {

// Legacy-style alias to project BoundingBox into expected name.
using AxisAlignedBoundingBox = ::ccBBox;
using OrientedBoundingBox = ::ecvOrientedBBox;

}  // namespace geometry
}  // namespace cloudViewer
