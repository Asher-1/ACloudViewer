// Shim header to match upstream include path.
// Maps cloudViewer/geometry/HalfEdgeTriangleMesh.h -> repo's ecvHalfEdgeMesh.h

#pragma once

#include <ecvHalfEdgeMesh.h>

#include "cloudViewer/geometry/Geometry.h"

namespace cloudViewer {
namespace geometry {
using HalfEdgeTriangleMesh = ::ecvHalfEdgeMesh;
}  // namespace geometry
}  // namespace cloudViewer
