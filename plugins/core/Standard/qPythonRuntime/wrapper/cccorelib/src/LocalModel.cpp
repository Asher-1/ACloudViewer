// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>

#include <LocalModel.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_LocalModel(py::module &cccorelib)
{
    py::class_<cloudViewer::LocalModel>(cccorelib, "LocalModel")
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
