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
        .def("draw", &ccBBox::draw, "context"_a, "col"_a);
}
