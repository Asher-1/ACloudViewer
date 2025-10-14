// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>

#include <CVMath.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_CCMath(py::module &cccorelib)
{
    cccorelib.def("LessThanEpsilon", static_cast<bool (*)(double)>(&cloudViewer::LessThanEpsilon));
    cccorelib.def("GreaterThanEpsilon", static_cast<bool (*)(double)>(&cloudViewer::GreaterThanEpsilon));
    cccorelib.def("RadiansToDegrees", static_cast<double (*)(double)>(&cloudViewer::RadiansToDegrees));
    cccorelib.def("DegreesToRadians", static_cast<double (*)(double)>(&cloudViewer::DegreesToRadians));
}
