// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <ecv2DLabel.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccInteractor(py::module &m)
{
    py::class_<ccInteractor>(m, "ccInteractor")
        .def("acceptClick", &ccInteractor::acceptClick, "x"_a, "y"_a, "button"_a)
        .def("move2D",
             &ccInteractor::move2D,
             "x"_a,
             "y"_a,
             "dx"_a,
             "dy"_a,
             "screenWidth"_a,
             "screenHeight"_a)
        .def("move3D", &ccInteractor::move3D, "u"_a);
}
