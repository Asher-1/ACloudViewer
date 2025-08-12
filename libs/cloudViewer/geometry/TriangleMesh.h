#pragma once

// Shim header to match upstream include path.
// Maps cloudViewer/geometry/TriangleMesh.h -> repo's ecvMesh.h

#include <ecvMesh.h>

namespace cloudViewer {
namespace geometry {
using TriangleMesh = ::ccMesh;  // legacy alias for t::geometry interop
}  // namespace geometry
}  // namespace cloudViewer


