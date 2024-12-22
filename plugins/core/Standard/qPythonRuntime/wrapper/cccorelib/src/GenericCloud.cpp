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

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <GenericCloud.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_GenericCloud(py::module &cccorelib)
{
    py::class_<cloudViewer::GenericCloud>(cccorelib, "GenericCloud")
        .def("size", &cloudViewer::GenericCloud::size)
        .def("forEach", &cloudViewer::GenericCloud::forEach, "action"_a)
        .def("getBoundingBox", &cloudViewer::GenericCloud::getBoundingBox, "bbMin"_a, "bbMax"_a)
        .def("testVisibility", &cloudViewer::GenericCloud::testVisibility, "P"_a)
        .def("placeIteratorAtBeginning", &cloudViewer::GenericCloud::placeIteratorAtBeginning)
        .def("getNextPoint", &cloudViewer::GenericCloud::getNextPoint, py::return_value_policy::reference)
        .def("enableScalarField", &cloudViewer::GenericCloud::enableScalarField)
        .def("isScalarFieldEnabled", &cloudViewer::GenericCloud::enableScalarField)
        .def("setPointScalarValue", &cloudViewer::GenericCloud::setPointScalarValue, "pointIndex"_a, "value"_a)
        .def("getPointScalarValue", &cloudViewer::GenericCloud::getPointScalarValue, "pointIndex"_a);
}
