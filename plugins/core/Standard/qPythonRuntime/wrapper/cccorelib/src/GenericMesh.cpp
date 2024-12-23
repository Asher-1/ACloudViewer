// ##########################################################################
// #                                                                        #
// #                ACLOUDVIEWER PLUGIN: PythonRuntime                       #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 of the License.               #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #                   COPYRIGHT: Thomas Montaigu                           #
// #                                                                        #
// ##########################################################################

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
