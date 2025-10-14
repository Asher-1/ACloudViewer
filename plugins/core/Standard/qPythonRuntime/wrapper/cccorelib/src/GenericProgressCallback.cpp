// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>

#include <GenericProgressCallback.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_GenericProgressCallback(py::module &cccorelib)
{
    py::class_<cloudViewer::GenericProgressCallback>(cccorelib, "GenericProgressCallback")
        .def("update", &cloudViewer::GenericProgressCallback::update)
        .def("setMethodTitle", &cloudViewer::GenericProgressCallback::setMethodTitle)
        .def("setInfo", &cloudViewer::GenericProgressCallback::setInfo)
        .def("start", &cloudViewer::GenericProgressCallback::start)
        .def("stop", &cloudViewer::GenericProgressCallback::stop)
        .def("isCancelRequested", &cloudViewer::GenericProgressCallback::isCancelRequested)
        .def("textCanBeEdited", &cloudViewer::GenericProgressCallback::textCanBeEdited);

    py::class_<cloudViewer::NormalizedProgress>(cccorelib, "NormalizedProgress")
        .def(py::init<cloudViewer::GenericProgressCallback *, unsigned, unsigned>(),
             "callback"_a,
             "totalSteps"_a,
             "totalPercentage"_a = 100)
        .def("scale",
             &cloudViewer::NormalizedProgress::scale,
             "totalSteps"_a,
             "totalPercentage"_a = 100,
             "updateCurrentProgress"_a = false)
        .def("reset", &cloudViewer::NormalizedProgress::reset)
        .def("oneStep", &cloudViewer::NormalizedProgress::oneStep)
        .def("steps", &cloudViewer::NormalizedProgress::steps);
}
