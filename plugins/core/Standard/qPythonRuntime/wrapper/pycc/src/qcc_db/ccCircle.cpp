// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <ecvCircle.h>
#include <ecvObject.h>
#include <ecvPolyline.h>

#include "../casters.h"

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccCircle(py::module &m)
{
    py::class_<ccCircle, ccPolyline>(m, "ccCircle", R"doc(
    ccCircle

    A 3D circle represented as a polyline.

    Parameters
    ----------
    radius : double, default: 0.0
        radius of the circle
    resolution : int, default: 48
        circle displayed resolution (number of segments)
    uniqueID : int, optional
        unique ID (handle with care)

    Example
    -------

    .. code:: Python

        circle = pycc.ccCircle(5.0, resolution=64)
        circle2 = pycc.ccCircle(radius=10.0, resolution=32)
)doc")
        .def(
            py::init<double, unsigned, unsigned>(),
            "radius"_a = 0.0,
            "resolution"_a = 48,
            "uniqueID"_a = []() { return ccUniqueIDGenerator::InvalidUniqueID; }())
        .def("getRadius", &ccCircle::getRadius, R"doc(
        Returns the radius of the circle.

        Returns
        -------
        double
            The radius of the circle.
)doc")
        .def("setRadius", &ccCircle::setRadius, "radius"_a, R"doc(
        Sets the radius of the circle.

        Parameters
        ----------
        radius : double
            The desired radius.
)doc")
        .def("getResolution", &ccCircle::getResolution, R"doc(
        Returns the resolution of the displayed circle.

        Returns
        -------
        int
            The resolution (number of segments) of the circle.
)doc")
        .def("setResolution", &ccCircle::setResolution, "resolution"_a, R"doc(
        Sets the resolution of the displayed circle.

        Parameters
        ----------
        resolution : int
            The displayed resolution (>= 4).
)doc")
        .def("clone",
             &ccCircle::clone,
             R"doc(
        Clones this circle.

        Returns
        -------
        ccCircle
            A new circle object that is a copy of this circle.
)doc",
             py::return_value_policy::take_ownership);
}
