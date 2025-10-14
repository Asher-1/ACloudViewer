// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/t/geometry/DrawableGeometry.h"

#include "pybind/t/geometry/geometry.h"

namespace cloudViewer {
namespace t {
namespace geometry {

void pybind_drawable_geometry(py::module& m) {
    py::class_<DrawableGeometry, std::shared_ptr<DrawableGeometry>>
            drawable_geometry(m, "DrawableGeometry",
                              "Base class for visualizable geometry.");
    drawable_geometry.def("has_valid_material", &DrawableGeometry::HasMaterial,
                          "Returns true if the geometry's material is valid.");
    drawable_geometry.def_property(
            "material", py::overload_cast<>(&DrawableGeometry::GetMaterial),
            &DrawableGeometry::SetMaterial);
}

}  // namespace geometry
}  // namespace t
}  // namespace cloudViewer
