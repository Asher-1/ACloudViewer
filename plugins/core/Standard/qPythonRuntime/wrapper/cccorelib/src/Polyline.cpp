// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>

#include <Polyline.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_Polyline(py::module &cccorelib)
{
    py::class_<cloudViewer::Polyline, cloudViewer::ReferenceCloud>(cccorelib, "Polyline")
        .def(py::init<cloudViewer::GenericIndexedCloudPersist *>(), "associatedCloud"_a)
        .def("isClosed", &cloudViewer::Polyline::isClosed)
        .def("setClosed", &cloudViewer::Polyline::setClosed, "state"_a)
        .def("clear", &cloudViewer::Polyline::clear, "unusedParam"_a = true);
}
