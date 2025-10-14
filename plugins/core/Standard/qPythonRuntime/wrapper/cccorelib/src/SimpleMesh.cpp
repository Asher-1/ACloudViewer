// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include <GenericIndexedCloud.h>
#include <GenericIndexedMesh.h>
#include <SimpleMesh.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_SimpleMesh(py::module &cccorelib)
{
    py::class_<cloudViewer::SimpleMesh, cloudViewer::GenericIndexedMesh>(cccorelib, "SimpleMesh")
        .def(py::init<cloudViewer::GenericIndexedCloud *, bool>(),
             "theVertices"_a,
             "linkVerticesWithMesh"_a = false)
        .def("capacity", &cloudViewer::SimpleMesh::capacity)
        .def("vertices", &cloudViewer::SimpleMesh::vertices, py::return_value_policy::reference)
        .def("clear", &cloudViewer::SimpleMesh::clear)
        .def("addTriangle", &cloudViewer::SimpleMesh::addTriangle, "i1"_a, "i2"_a, "i3"_a)
        .def("reserve", &cloudViewer::SimpleMesh::reserve, "n"_a)
        .def("resize", &cloudViewer::SimpleMesh::resize, "n"_a);
}
