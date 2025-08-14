#pragma once

// Legacy AABB API adapter mapped to core/include/BoundingBox.h

#include <ecvBBox.h>

#include "ecvOrientedBBox.h"

namespace cloudViewer {
namespace geometry {

// Legacy-style alias to project BoundingBox into expected name.
using AxisAlignedBoundingBox = ::ccBBox;
using OrientedBoundingBox = ::ecvOrientedBBox;

}  // namespace geometry
}  // namespace cloudViewer
