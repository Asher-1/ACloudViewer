// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Shim header to match upstream include path.
// Maps cloudViewer/geometry/PointCloud.h -> repo's ecvPointCloud.h

#include <ecvPointCloud.h>

#include "cloudViewer/geometry/Geometry.h"

namespace cloudViewer {
namespace geometry {
using PointCloud = ::ccPointCloud;
}  // namespace geometry
}  // namespace cloudViewer
