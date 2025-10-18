// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <ecvGLMatrix.h>
#include <ecvGenericPrimitive.h>
#include <ecvMesh.h>

#include "../casters.h"

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccGenericPrimitive(py::module &m)
{
    py::class_<ccGenericPrimitive, ccMesh>(m, "ccGenericPrimitive")
        .def("getTypeName", &ccGenericPrimitive::getTypeName)
        .def("clone", &ccGenericPrimitive::getTypeName)
        .def("setColor", &ccGenericPrimitive::setColor, "col"_a)
        .def("hasDrawingPrecision", &ccGenericPrimitive::hasDrawingPrecision)
        .def("setDrawingPrecision", &ccGenericPrimitive::setDrawingPrecision, "steps"_a)
        .def("getDrawingPrecision", &ccGenericPrimitive::getDrawingPrecision)
        .def("getTransformation",
             (const ccGLMatrix &(ccGenericPrimitive::*)()
                  const)(&ccGenericPrimitive::getTransformation));
}
