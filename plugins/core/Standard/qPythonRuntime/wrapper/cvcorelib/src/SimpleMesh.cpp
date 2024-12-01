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
#include <pybind11/stl_bind.h>

#include <GenericIndexedCloud.h>
#include <GenericIndexedMesh.h>
#include <SimpleMesh.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_SimpleMesh(py::module &cvcorelib)
{
    py::class_<cloudViewer::SimpleMesh, cloudViewer::GenericIndexedMesh>(cvcorelib, "SimpleMesh")
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
