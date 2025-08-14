// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2024 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "t/geometry/LineSet.h"

#include <string>

#include "pybind/t/geometry/geometry.h"
#include "t/geometry/TriangleMesh.h"

namespace cloudViewer {
namespace t {
namespace geometry {

void pybind_lineset(py::module& m) {
    py::class_<LineSet, PyGeometry<LineSet>, std::shared_ptr<LineSet>, Geometry,
               DrawableGeometry>
            line_set(m, "LineSet",
                     "A LineSet contains points and lines with attributes.");

    // Constructors.
    line_set.def(py::init<const core::Device&>(),
                 "device"_a = core::Device("CPU:0"))
            .def(py::init<const core::Tensor&, const core::Tensor&>(),
                 "point_positions"_a, "line_indices"_a);

    // Attributes
    line_set.def_property_readonly(
            "point", py::overload_cast<>(&LineSet::GetPointAttr, py::const_),
            "Dictionary containing point attributes. The primary key "
            "'positions'.");
    line_set.def_property_readonly(
            "line", py::overload_cast<>(&LineSet::GetLineAttr, py::const_),
            "Dictionary containing line attributes. The primary key "
            "'indices'.");

    // Repr
    line_set.def("__repr__", &LineSet::ToString);

    // Device transfers.
    line_set.def("to", &LineSet::To, "device"_a, "copy"_a = false);
    line_set.def("clone", &LineSet::Clone);
    line_set.def("cpu", [](const LineSet& self) {
        return self.To(core::Device("CPU:0"));
    });
    line_set.def(
            "cuda",
            [](const LineSet& self, int device_id) {
                return self.To(core::Device("CUDA", device_id));
            },
            "device_id"_a = 0);

    // Specific functions.
    line_set.def("get_min_bound", &LineSet::GetMinBound);
    line_set.def("get_max_bound", &LineSet::GetMaxBound);
    line_set.def("get_center", &LineSet::GetCenter);
    line_set.def("transform", &LineSet::Transform, "transformation"_a);
    line_set.def("translate", &LineSet::Translate, "translation"_a,
                 "relative"_a = true);
    line_set.def("scale", &LineSet::Scale, "scale"_a, "center"_a);
    line_set.def("rotate", &LineSet::Rotate, "R"_a, "center"_a);

    line_set.def_static("from_legacy", &LineSet::FromLegacy, "lineset_legacy"_a,
                        "float_dtype"_a = core::Float32,
                        "int_dtype"_a = core::Int64,
                        "device"_a = core::Device("CPU:0"));
    line_set.def("to_legacy", &LineSet::ToLegacy);

    line_set.def("get_axis_aligned_bounding_box",
                 &LineSet::GetAxisAlignedBoundingBox);
    line_set.def("get_oriented_bounding_box", &LineSet::GetOrientedBoundingBox);

    line_set.def("extrude_rotation", &LineSet::ExtrudeRotation, "angle"_a,
                 "axis"_a, "resolution"_a = 16, "translation"_a = 0.0,
                 "capping"_a = true);
    line_set.def("extrude_linear", &LineSet::ExtrudeLinear, "vector"_a,
                 "scale"_a = 1.0, "capping"_a = true);

    line_set.def("paint_uniform_color", &LineSet::PaintUniformColor, "color"_a);

    line_set.def_static("create_camera_visualization",
                        &LineSet::CreateCameraVisualization, "view_width_px"_a,
                        "view_height_px"_a, "intrinsic"_a, "extrinsic"_a,
                        "scale"_a = 1.0,
                        py::arg_v("color", core::Tensor({}, core::Float32),
                                  "cloudViewer.core.Tensor([], "
                                  "dtype=cloudViewer.core.Dtype.Float32)"));
}

}  // namespace geometry
}  // namespace t
}  // namespace cloudViewer
