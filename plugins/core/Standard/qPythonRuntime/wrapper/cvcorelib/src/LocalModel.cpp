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

#include <LocalModel.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_LocalModel(py::module &cvcorelib)
{
    py::class_<cloudViewer::LocalModel>(cvcorelib, "LocalModel")
        // TODO
        //			.def_static("New", &cloudViewer::LocalModel::New, "type"_a, "subset"_a,
        //"center"_a, "squaredRadius"_a)
        .def("getType", &cloudViewer::LocalModel::getType)
        .def("getCenter", &cloudViewer::LocalModel::getCenter)
        .def("getSquareSize", &cloudViewer::LocalModel::getSquareSize)
        .def("computeDistanceFromModelToPoint",
             &cloudViewer::LocalModel::computeDistanceFromModelToPoint,
             "P"_a,
             "nearestPoint"_a = nullptr);
}
