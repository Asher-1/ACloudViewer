// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>

#include <GenericMesh.h>
#include <GenericTriangle.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_GenericMesh(py::module &cccorelib)
{
    py::class_<cloudViewer::GenericMesh>(cccorelib, "GenericMesh", R"pbdoc(
	A generic mesh interface for data communication between library and client applications
)pbdoc")
        .def("size", &cloudViewer::GenericMesh::size, R"pbdoc(
	Returns the number of triangles
)pbdoc")
        .def("forEach", &cloudViewer::GenericMesh::forEach, "action"_a)
        .def("getBoundingBox", &cloudViewer::GenericMesh::getBoundingBox, "bbMin"_a, "bbMax"_a, R"pbdoc(
	Returns the mesh bounding-box

	Parameters
	----------
	bbMin: out parameter, lower bounding-box limits (Xmin,Ymin,Zmin)
	bbMax: out parameter, higher bounding-box limits (Xmax,Ymax,Zmax)
)pbdoc")
        .def("placeIteratorAtBeginning", &cloudViewer::GenericMesh::placeIteratorAtBeginning)
        .def("_getNextTriangle",
             &cloudViewer::GenericMesh::_getNextTriangle,
             py::return_value_policy::reference);
}
