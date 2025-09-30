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

#include "t/geometry/BoundingVolume.h"

#include <string>

#include "pybind/t/geometry/geometry.h"

namespace cloudViewer {
namespace t {
namespace geometry {

void pybind_boundingvolume(py::module& m) {
    // AxisAlignedBoundingBox
    py::class_<AxisAlignedBoundingBox, PyGeometry<AxisAlignedBoundingBox>,
               std::shared_ptr<AxisAlignedBoundingBox>, Geometry,
               DrawableGeometry>
            aabb(m, "AxisAlignedBoundingBox",
                 "A bounding box aligned with coordinate axes.");

    aabb.def(py::init<const core::Device&>(), "device"_a = core::Device("CPU:0"))
        .def(py::init<const core::Tensor&, const core::Tensor&>(),
             "min_bound"_a, "max_bound"_a)
        .def("to", &AxisAlignedBoundingBox::To, "device"_a,
             "copy"_a = false)
        .def("clone", &AxisAlignedBoundingBox::Clone)
        .def("cpu",
             [](const AxisAlignedBoundingBox& self) {
                 return self.To(core::Device("CPU:0"));
             })
        .def("cuda",
             [](const AxisAlignedBoundingBox& self, int device_id) {
                 return self.To(core::Device("CUDA", device_id));
             },
             "device_id"_a = 0)
        .def("set_min_bound", &AxisAlignedBoundingBox::SetMinBound, "min_bound"_a)
        .def("set_max_bound", &AxisAlignedBoundingBox::SetMaxBound, "max_bound"_a)
        .def("set_color", &AxisAlignedBoundingBox::SetColor, "color"_a)
        .def_property_readonly("min_bound", &AxisAlignedBoundingBox::GetMinBound)
        .def_property_readonly("max_bound", &AxisAlignedBoundingBox::GetMaxBound)
        .def_property_readonly("color", &AxisAlignedBoundingBox::GetColor)
        .def("get_center", &AxisAlignedBoundingBox::GetCenter)
        .def("translate", &AxisAlignedBoundingBox::Translate,
             "translation"_a, "relative"_a = true)
        .def("scale", &AxisAlignedBoundingBox::Scale, "scale"_a,
             "center"_a = utility::nullopt)
        .def("get_extent", &AxisAlignedBoundingBox::GetExtent)
        .def("get_half_extent", &AxisAlignedBoundingBox::GetHalfExtent)
        .def("get_max_extent", &AxisAlignedBoundingBox::GetMaxExtent)
        .def("volume", &AxisAlignedBoundingBox::Volume)
        .def("get_box_points", &AxisAlignedBoundingBox::GetBoxPoints)
        .def("get_point_indices_within_bounding_box",
             &AxisAlignedBoundingBox::GetPointIndicesWithinBoundingBox,
             "points"_a)
        .def("to_legacy", &AxisAlignedBoundingBox::ToLegacy)
        .def("get_oriented_bounding_box",
             &AxisAlignedBoundingBox::GetOrientedBoundingBox)
        .def_static("from_legacy", &AxisAlignedBoundingBox::FromLegacy,
                    "box"_a, "dtype"_a = core::Float32,
                    "device"_a = core::Device("CPU:0"))
        .def_static("create_from_points", &AxisAlignedBoundingBox::CreateFromPoints,
                    "points"_a);

    // OrientedBoundingBox
    py::class_<OrientedBoundingBox, PyGeometry<OrientedBoundingBox>,
               std::shared_ptr<OrientedBoundingBox>, Geometry,
               DrawableGeometry>
            obb(m, "OrientedBoundingBox",
                "A bounding box oriented along an arbitrary frame.");

    obb.def(py::init<const core::Device&>(), "device"_a = core::Device("CPU:0"))
        .def(py::init<const core::Tensor&, const core::Tensor&, const core::Tensor&>(),
             "center"_a, "rotation"_a, "extent"_a)
        .def("to", &OrientedBoundingBox::To, "device"_a, "copy"_a = false)
        .def("clone", &OrientedBoundingBox::Clone)
        .def("cpu",
             [](const OrientedBoundingBox& self) {
                 return self.To(core::Device("CPU:0"));
             })
        .def("cuda",
             [](const OrientedBoundingBox& self, int device_id) {
                 return self.To(core::Device("CUDA", device_id));
             },
             "device_id"_a = 0)
        .def("set_center", &OrientedBoundingBox::SetCenter, "center"_a)
        .def("set_rotation", &OrientedBoundingBox::SetRotation, "rotation"_a)
        .def("set_extent", &OrientedBoundingBox::SetExtent, "extent"_a)
        .def("set_color", &OrientedBoundingBox::SetColor, "color"_a)
        .def_property_readonly("center", &OrientedBoundingBox::GetCenter)
        .def_property_readonly("rotation", &OrientedBoundingBox::GetRotation)
        .def_property_readonly("extent", &OrientedBoundingBox::GetExtent)
        .def_property_readonly("color", &OrientedBoundingBox::GetColor)
        .def("get_min_bound", &OrientedBoundingBox::GetMinBound)
        .def("get_max_bound", &OrientedBoundingBox::GetMaxBound)
        .def("translate", &OrientedBoundingBox::Translate,
             "translation"_a, "relative"_a = true)
        .def("rotate", &OrientedBoundingBox::Rotate,
             "rotation"_a, "center"_a = utility::nullopt)
        .def("transform", &OrientedBoundingBox::Transform, "transformation"_a)
        .def("scale", &OrientedBoundingBox::Scale,
             "scale"_a, "center"_a = utility::nullopt)
        .def("volume", &OrientedBoundingBox::Volume)
        .def("get_box_points", &OrientedBoundingBox::GetBoxPoints)
        .def("get_point_indices_within_bounding_box",
             &OrientedBoundingBox::GetPointIndicesWithinBoundingBox,
             "points"_a)
        .def("get_axis_aligned_bounding_box",
             &OrientedBoundingBox::GetAxisAlignedBoundingBox)
        .def("to_legacy", &OrientedBoundingBox::ToLegacy)
        .def_static("from_legacy", &OrientedBoundingBox::FromLegacy,
                    "box"_a, "dtype"_a = core::Float32,
                    "device"_a = core::Device("CPU:0"))
        .def_static("create_from_axis_aligned_bounding_box",
                    &OrientedBoundingBox::CreateFromAxisAlignedBoundingBox,
                    "aabb"_a)
        .def_static("create_from_points", &OrientedBoundingBox::CreateFromPoints,
                    "points"_a, "robust"_a = false,
                    "method"_a = MethodOBBCreate::MINIMAL_APPROX);
}

}  // namespace geometry
}  // namespace t
}  // namespace cloudViewer


