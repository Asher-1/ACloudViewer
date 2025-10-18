// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <ecvDish.h>

#include "../casters.h"

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccDish(py::module &m)
{
    py::class_<ccDish, ccGenericPrimitive>(m, "ccDish", R"doc(
    ccDish

    Parameters
    ----------
    radius : PointCoordinateType
        base radius
    height : PointCoordinateType
        maximum height of dished surface above base
    radius2 : PointCoordinateType, default = 0
        If radius2 is zero, dish is drawn as a section of sphere.
         If radius2 is >0, dish is defined as half of an ellipsoid.
    transMat : , optional
        optional 3D transformation (can be set afterwards with ccDrawableObject::setGLTransformation)
    name : str, default: Sphere
        name of the sphere object

    Example
    -------

    .. code:: Python

    dish = pycc.ccDish(1.0, 4.0)
    )doc")
        .def(
            py::init<PointCoordinateType,
                     PointCoordinateType,
                     PointCoordinateType,
                     const ccGLMatrix *,
                     QString,
                     unsigned>(),
            "radius"_a,
            "height"_a,
            "radius2"_a = 0,
            "transMat"_a = nullptr,
            "name"_a = QString("Dish"),
            "precision"_a = []() { return ccDish::DEFAULT_DRAWING_PRECISION; }());
}
