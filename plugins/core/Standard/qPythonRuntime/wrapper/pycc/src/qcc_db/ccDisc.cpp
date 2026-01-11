// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <ecvDisc.h>
#include <ecvGLMatrix.h>
#include <ecvGenericPrimitive.h>

#include "../casters.h"

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccDisc(py::module &m)
{
    py::class_<ccDisc, ccGenericPrimitive>(m, "ccDisc", R"doc(
    ccDisc

    A 3D disc primitive.

    Parameters
    ----------
    radius : PointCoordinateType
        disc radius
    transMat : ccGLMatrix, optional
        optional 3D transformation (can be set afterwards with ccDrawableObject::setGLTransformation)
    name : str, default: "Disc"
        name of the disc object
    precision : int, default: 72
        drawing precision (angular step = 360/precision)

    Example
    -------

    .. code:: Python

        disc = pycc.ccDisc(5.0)
        disc2 = pycc.ccDisc(radius=10.0, precision=64, name="MyDisc")
)doc")
        .def(
            py::init<PointCoordinateType, const ccGLMatrix *, QString, unsigned>(),
            "radius"_a,
            "transMat"_a = nullptr,
            "name"_a = QString("Disc"),
            "precision"_a = []() { return ccDisc::DEFAULT_DRAWING_PRECISION; }())
        .def(py::init<QString>(), "name"_a = QString("Disc"))
        .def("getRadius", &ccDisc::getRadius, R"doc(
        Returns the radius of the disc.

        Returns
        -------
        PointCoordinateType
            The radius of the disc.
)doc")
        .def("setRadius", &ccDisc::setRadius, "radius"_a, R"doc(
        Sets the radius of the disc.

        Parameters
        ----------
        radius : PointCoordinateType
            The desired radius.

        Note
        ----
        This changes the primitive content (calls ccGenericPrimitive::updateRepresentation).
)doc")
        .def("clone",
             &ccDisc::clone,
             R"doc(
        Clones this disc.

        Returns
        -------
        ccDisc
            A new disc object that is a copy of this disc.
)doc",
             py::return_value_policy::take_ownership);
}
