// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Shim header to match upstream include path.
// Maps cloudViewer/geometry/Geometry.h -> repo's ecvHObject.h

#pragma once

#include <ecvHObject.h>

namespace cloudViewer {
namespace geometry {

using Geometry = ::ccHObject;

}
}  // namespace cloudViewer