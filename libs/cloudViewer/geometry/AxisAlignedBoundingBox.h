#pragma once

// Legacy AABB API adapter mapped to core/include/BoundingBox.h

#include <BoundingBox.h>

namespace cloudViewer {
namespace geometry {

// Legacy-style alias to project BoundingBox into expected name.
using AxisAlignedBoundingBox = cloudViewer::BoundingBox;

} // namespace geometry
} // namespace cloudViewer



