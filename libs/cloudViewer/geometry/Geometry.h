// Shim header to match upstream include path.
// Maps cloudViewer/geometry/Geometry.h -> repo's ecvHObject.h

#pragma once

#include <ecvHObject.h>

namespace cloudViewer {
namespace geometry {

using Geometry = ::ccHObject;

}
}  // namespace cloudViewer