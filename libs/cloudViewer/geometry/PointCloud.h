#pragma once

// Shim header to match upstream include path.
// Maps cloudViewer/geometry/PointCloud.h -> repo's ecvPointCloud.h

#include <ecvPointCloud.h>

namespace cloudViewer {
namespace geometry {
// Provide legacy alias expected by t::geometry interop code
using PointCloud = ::ccPointCloud;
}  // namespace geometry
}  // namespace cloudViewer


