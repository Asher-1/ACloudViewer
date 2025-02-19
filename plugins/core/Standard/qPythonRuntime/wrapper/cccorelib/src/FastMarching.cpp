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

#include <FastMarching.h>
#include <FastMarchingForPropagation.h>
#include <GenericCloud.h>
#include <ReferenceCloud.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_FastMarching(py::module &cccorelib)
{
    cccorelib.attr("c_FastMarchingNeighbourPosShift") = cloudViewer::c_FastMarchingNeighbourPosShift;
    py::class_<cloudViewer::FastMarching>(cccorelib, "FastMarching")
        .def("setSeedCell", &cloudViewer::FastMarching::setSeedCell, "pos"_a)
        .def("propagate", &cloudViewer::FastMarching::setSeedCell)
        .def("cleanLastPropagation", &cloudViewer::FastMarching::cleanLastPropagation)
        .def("getTime", &cloudViewer::FastMarching::getTime, "pos"_a, "absoluteCoordinates"_a = false)
        .def("setExtendedConnectivity", &cloudViewer::FastMarching::setExtendedConnectivity, "state"_a);
}
