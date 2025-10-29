// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <GenericIndexedCloudPersist.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>

#include "../casters.h"

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccPolyline(py::module &m)
{
    py::class_<ccPolyline, cloudViewer::Polyline, ccShiftedObject>(m, "ccPolyline")
        .def(py::init([](cloudViewer::GenericIndexedCloudPersist *cloud)
                      { return new ccPolyline(cloud); }),
             "associatedCloud"_a,
             py::keep_alive<1, 2>() // Keep cloud alive while polyline is
             )

        .def("set2DMode", &ccPolyline::set2DMode, "state"_a)
        .def("is2DMode", &ccPolyline::is2DMode)
        .def("setForeground", &ccPolyline::setForeground, "state"_a)
        .def("setColor", &ccPolyline::setColor, "col"_a)
        .def("setWidth", &ccPolyline::setWidth, "width"_a)
        .def("getWidth", &ccPolyline::getWidth)
        .def("getColor", &ccPolyline::getColor)
        .def(
            "split",
            [](ccPolyline &self, PointCoordinateType maxEdgeLength)
            {
                std::vector<ccPolyline *> parts;
                self.split(maxEdgeLength, parts);
                return parts;
            },
            "maxEdgeLength"_a)
        .def("computeLength", &ccPolyline::computeLength)
        .def("showVertices", &ccPolyline::showVertices, "state"_a)
        .def("verticesShown", &ccPolyline::verticesShown)
        .def("setVertexMarkerWidth", &ccPolyline::setVertexMarkerWidth, "width"_a)
        .def("getVertexMarkerWidth", &ccPolyline::getVertexMarkerWidth)
        // TODO .def("initWith", &ccPolyline::initWith, "vertices"_a, "poly"_a)
        .def("importParametersFrom", &ccPolyline::importParametersFrom, "poly"_a)
        .def("showArrow", &ccPolyline::showArrow, "state"_a, "vertIndex"_a, "length"_a)
        .def("segmentCount", &ccPolyline::segmentCount)
        .def("samplePoints",
             &ccPolyline::samplePoints,
             "densityBased"_a,
             "samplingParameter"_a,
             "withRGB"_a)
        .def("smoothChaikin", &ccPolyline::smoothChaikin, "ratio"_a, "iterationCount"_a);
}
