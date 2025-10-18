// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <ecvSphere.h>

#include "../casters.h"

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccSphere(py::module &m)
{
    py::class_<ccSphere, ccGenericPrimitive>(m, "ccSphere", R"doc(
    ccSphere

    Parameters
    ----------
    radius : PointCoordinateType
        radius of the sphere
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

        sphere = pycc.ccSphere(3.0)
        sphere2 = pycc.ccSphere(5.0, name="Sphere2")
)doc")
        .def(py::init<PointCoordinateType, const ccGLMatrix *, QString, unsigned>(),
             "radius"_a,
             "transMat"_a = nullptr,
             "name"_a = QString("Sphere"),
             "precision"_a = 24)
        .def("getRadius", &ccSphere::getRadius)
        .def("setRadius", &ccSphere::setRadius, "radius"_a);
}
