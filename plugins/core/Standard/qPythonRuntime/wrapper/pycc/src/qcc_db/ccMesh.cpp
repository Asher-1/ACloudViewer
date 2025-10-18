// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <ecvGenericMesh.h>
#include <ecvGenericPointCloud.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccMesh(py::module &m)
{
    py::class_<ccMesh, ccGenericMesh>(m, "ccMesh")
        .def(py::init<ccGenericPointCloud *>(),
             "vertices"_a,
             // Keep the cloud (2) alive while the mesh (1) is alive            py::keep_alive<1,
             // 2>())
             // one could have expected <0, 1> as the mesh is the return value in C++
             // but in python __init__ there is a self/this as arg 1
             py::keep_alive<1, 2>())
        .def("setAssociatedCloud",
             &ccMesh::setAssociatedCloud,
             "cloud"_a,
             py::keep_alive<1, 2>() /* Keep the cloud (2) alive while the mesh (1) is alive */)
        .def("addTriangle",
             py::overload_cast<unsigned, unsigned, unsigned>(&ccMesh::addTriangle),
             "i1"_a,
             "i2"_a,
             "i3"_a,
             R"pbdoc(
                 Adds a triangle to the mesh
                 Bounding-box validity is broken after a call to this method.
                 However, for the sake of performance, no call to notifyGeometryUpdate
                 is made automatically. Make sure to do so when all modifications are done!

                 Parameters
                 ----------
                 i1 first vertex index (relatively to the vertex cloud)
                 i2 second vertex index (relatively to the vertex cloud)
                 i3 third vertex index (relatively to the vertex cloud)
             )pbdoc")
        .def("addTriangle",
             py::overload_cast<const cloudViewer::VerticesIndexes &>(&ccMesh::addTriangle),
             "triangle"_a)
        .def("addTriangles", &ccMesh::addTriangles, "triangles"_a);
}
