// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include <GenericIndexedCloudPersist.h>
#include <ReferenceCloud.h>

namespace py = pybind11;
using namespace pybind11::literals;

using cloudViewer::ReferenceCloud;

void define_ReferenceCloud(py::module &cccorelib)
{
    py::class_<ReferenceCloud, cloudViewer::GenericIndexedCloudPersist>(cccorelib, "ReferenceCloud")
        .def(py::init<cloudViewer::GenericIndexedCloudPersist *>(),
             "associatedCloud"_a,
             py::keep_alive<1, 2>())
        .def("getPointGlobalIndex", &ReferenceCloud::getPointGlobalIndex)
        .def("getCurrentPointCoordinates", &ReferenceCloud::getCurrentPointCoordinates)
        .def("getCurrentPointGlobalIndex", &ReferenceCloud::getCurrentPointGlobalIndex)
        .def("getCurrentPointScalarValue", &ReferenceCloud::getCurrentPointScalarValue)
        .def("setCurrentPointScalarValue", &ReferenceCloud::setCurrentPointScalarValue)
        .def("forwardIterator", &ReferenceCloud::forwardIterator)
        .def("clear", &ReferenceCloud::clear, "releaseMemory"_a = false)
        .def("addPointIndex",
             static_cast<bool (ReferenceCloud::*)(unsigned)>(&ReferenceCloud::addPointIndex),
             "unsigned"_a)
        .def("addPointIndex",
             static_cast<bool (ReferenceCloud::*)(unsigned, unsigned)>(&ReferenceCloud::addPointIndex),
             "firstIndex"_a,
             "lastIndex"_a)
        .def("setPointIndex", &ReferenceCloud::setPointIndex, "firstIndex"_a, "lastIndex"_a)
        .def("reserve", &ReferenceCloud::reserve, "n"_a)
        .def("resize", &ReferenceCloud::resize, "_n")
        .def("capacity", &ReferenceCloud::capacity)
        .def("swap", &ReferenceCloud::swap, "i"_a, "j"_a)
        .def("removeCurrentPointGlobalIndex", &ReferenceCloud::removeCurrentPointGlobalIndex)
        .def("removePointGlobalIndex", &ReferenceCloud::removePointGlobalIndex, "localIndex"_a)
        .def("getAssociatedCloud",
             static_cast<cloudViewer::GenericIndexedCloudPersist *(ReferenceCloud::*)()>(
                 &ReferenceCloud::getAssociatedCloud),
             py::return_value_policy::reference)
        .def("setAssociatedCloud", &ReferenceCloud::setAssociatedCloud, "cloud"_a)
        .def("add", &ReferenceCloud::add, "cloud"_a)
        .def("invalidateBoundingBox", &ReferenceCloud::invalidateBoundingBox);
}
