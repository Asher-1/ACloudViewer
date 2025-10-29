// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

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
        .def(
            "setPointScalarValue", &cloudViewer::GenericCloud::setPointScalarValue, "pointIndex"_a, "value"_a)
        .def("getPointScalarValue", &cloudViewer::GenericCloud::getPointScalarValue, "pointIndex"_a);
}
