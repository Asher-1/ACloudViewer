// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

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
