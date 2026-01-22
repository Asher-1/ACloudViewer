// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <ecvBBox.h>
#include <ecvGLMatrix.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccBBox(py::module &m)
{
    py::class_<ccBBox, ccHObject, cloudViewer::BoundingBox>(m, "ccBBox")
        .def(py::init<>())
        .def(py::init<const CCVector3 &, const CCVector3 &, const std::string &>(),
             "bbMinCorner"_a,
             "bbMaxCorner"_a,
             "name"_a = "ccBBox")
        .def(
            "__mul__", [](ccBBox &self, ccGLMatrix &mat) { return self * mat; }, py::is_operator())
        .def(
            "__mul__", [](ccBBox &self, ccGLMatrixd &mat) { return self * mat; }, py::is_operator())
        // Expose both draw overloads: one with default color, one with custom color
        .def("draw",
             static_cast<void (ccBBox::*)(CC_DRAW_CONTEXT &)>(&ccBBox::draw),
             "Draw bounding box using default color from context",
             "context"_a)
        .def("draw",
             static_cast<void (ccBBox::*)(CC_DRAW_CONTEXT &, const ecvColor::Rgb &)>(&ccBBox::draw),
             "Draw bounding box with custom color",
             "context"_a,
             "col"_a);
}
