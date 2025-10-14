// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>

#include <ecvGenericDisplayTools.h>
#include <ecvSensor.h>

#include "../casters.h"

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccSensor(py::module &m)
{
    py::native_enum<CC_SENSOR_TYPE>(m, "CC_SENSOR_TYPE", "enum.Enum", "CC_SENSOR_TYPE.")
        .value("UNKNOWN_SENSOR", CC_SENSOR_TYPE::UNKNOWN_SENSOR)
        .value("GROUND_BASED_LIDAR", CC_SENSOR_TYPE::GROUND_BASED_LIDAR)
        .export_values()
        .finalize();

    py::class_<ccSensor, ccHObject>(m, "ccSensor")
        .def(py::init<const QString &>())
        .def("checkVisibility", &ccSensor::checkVisibility, "P"_a)
        //    TODO.def("getPositions", &ccSensor::getPositions)
        .def("addPosition", &ccSensor::addPosition, "trans"_a, "index"_a)
        .def(
            "getAbsoluteTransformation", &ccSensor::getAbsoluteTransformation, "trans"_a, "index"_a)
        .def("getActiveAbsoluteTransformation",
             &ccSensor::getActiveAbsoluteTransformation,
             "trans"_a)
        .def("getActiveAbsoluteCenter", &ccSensor::getActiveAbsoluteCenter, "vec"_a)
        .def("getActiveAbsoluteRotation", &ccSensor::getActiveAbsoluteRotation, "rotation"_a)
        .def("setRigidTransformation", &ccSensor::setRigidTransformation, "mat"_a)
        .def("getRigidTransformation",
             static_cast<ccGLMatrix &(ccSensor::*)()>(&ccSensor::getRigidTransformation))
        .def("getIndexBounds",
             [](const ccSensor &self)
             {
                 double min, max;
                 self.getIndexBounds(min, max);

                 return py::make_tuple(min, max);
             })
        .def("getActiveIndex", &ccSensor::getActiveIndex)
        .def("setActiveIndex", &ccSensor::setActiveIndex, "index"_a)
        .def("setGraphicScale", &ccSensor::setGraphicScale, "scale"_a)
        .def("getGraphicScale", &ccSensor::getGraphicScale)
        .def("applyViewport", &ccSensor::applyViewport);
}
