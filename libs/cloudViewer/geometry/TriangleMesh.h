// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Shim header to match upstream include path.
// Maps cloudViewer/geometry/TriangleMesh.h -> repo's ecvMesh.h

#include <ecvMesh.h>

#include "cloudViewer/geometry/Geometry.h"

namespace cloudViewer {
namespace geometry {
using TriangleMesh = ::ccMesh;
}  // namespace geometry
}  // namespace cloudViewer
