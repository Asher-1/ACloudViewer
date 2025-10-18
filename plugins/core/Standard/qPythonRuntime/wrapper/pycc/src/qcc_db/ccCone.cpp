// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <ecvCone.h>

#include "../casters.h"

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccCone(py::module &m)
{
    py::class_<ccCone, ccGenericPrimitive>(m, "ccCone", R"doc(
    ccCone

    Parameters
    ----------
    bottomRadius : PointCoordinateType
    topRadius : PointCoordinateType
    height : PointCoordinateType
        cone height (transformation should point to the axis center)
    xOff : PointCoordinateType, default = 0
        displacement of axes along X-axis (Snout mode)
    yOff : PointCoordinateType, default = 0
        displacement of axes along Y-axis (Snout mode)
    transMat : , optional
         optional 3D transformation (can be set afterwards with ccDrawableObject::setGLTransformation)
    name : str, default: Sphere
        name of the sphere object
    precision : int, default: 24
        drawing precision (angular step = 360/precision)
    uniqueID : int, optional
        unique ID (handle with care)

    Example
    -------

    .. code:: Python

        cone = pycc.ccCone(10.0, 5.0, 20.0)
)doc")
        .def(
            py::init<PointCoordinateType,
                     PointCoordinateType,
                     PointCoordinateType,
                     PointCoordinateType,
                     PointCoordinateType,
                     const ccGLMatrix *,
                     QString,
                     unsigned>(),
            "bottomRadius"_a,
            "topRadius"_a,
            "height"_a,
            "xOff"_a = 0,
            "yOff"_a = 0,
            "transMat"_a = nullptr,
            "name"_a = QString("Cone"),
            "precision"_a = []() { return ccCone::DEFAULT_DRAWING_PRECISION; }())
        .def("getHeight", &ccCone::getHeight)
        .def("setHeight", &ccCone::setHeight, "height"_a)
        .def("getBottomRadius", &ccCone::getBottomRadius)
        .def("setBottomRadius", &ccCone::setBottomRadius, "radius"_a)
        .def("getTopRadius", &ccCone::getTopRadius)
        .def("setTopRadius", &ccCone::setTopRadius, "radius"_a)
        .def("getBottomCenter", &ccCone::getBottomCenter)
        .def("getTopCenter", &ccCone::getTopCenter)
        .def("getSmallCenter", &ccCone::getSmallCenter)
        .def("getLargeCenter", &ccCone::getLargeCenter)
        .def("getSmallRadius", &ccCone::getSmallRadius)
        .def("getLargeRadius", &ccCone::getLargeRadius)
        .def("isSnoutMode", &ccCone::isSnoutMode);
}
