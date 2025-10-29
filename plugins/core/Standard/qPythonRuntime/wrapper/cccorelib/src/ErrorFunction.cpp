// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>

#include <ErrorFunction.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_ErrorFunction(py::module &cccorelib)
{
    cccorelib.attr("c_erfRelativeError") = cloudViewer::c_erfRelativeError;
    py::class_<cloudViewer::ErrorFunction>(cccorelib, "ErrorFunction")
        .def_static("erfc", &cloudViewer::ErrorFunction::erfc, "x"_a)
        .def_static("erf", &cloudViewer::ErrorFunction::erf, "x"_a);
}
