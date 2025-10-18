// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <GenericTriangle.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_GenericTriangle(py::module &cccorelib)
{
    py::class_<cloudViewer::GenericTriangle>(cccorelib, "GenericTriangle")
        .def("_getA", &cloudViewer::GenericTriangle::_getA, py::return_value_policy::reference)
        .def("_getB", &cloudViewer::GenericTriangle::_getB, py::return_value_policy::reference)
        .def("_getC", &cloudViewer::GenericTriangle::_getC, py::return_value_policy::reference);
}
