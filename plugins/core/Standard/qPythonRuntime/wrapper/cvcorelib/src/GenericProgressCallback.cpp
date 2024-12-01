// ##########################################################################
// #                                                                        #
// #                ACLOUDVIEWER PLUGIN: PythonRuntime                       #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 of the License.               #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #                   COPYRIGHT: Thomas Montaigu                           #
// #                                                                        #
// ##########################################################################

#include <pybind11/pybind11.h>

#include <GenericProgressCallback.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_GenericProgressCallback(py::module &cvcorelib)
{
    py::class_<cloudViewer::GenericProgressCallback>(cvcorelib, "GenericProgressCallback")
        .def("update", &cloudViewer::GenericProgressCallback::update)
        .def("setMethodTitle", &cloudViewer::GenericProgressCallback::setMethodTitle)
        .def("setInfo", &cloudViewer::GenericProgressCallback::setInfo)
        .def("start", &cloudViewer::GenericProgressCallback::start)
        .def("stop", &cloudViewer::GenericProgressCallback::stop)
        .def("isCancelRequested", &cloudViewer::GenericProgressCallback::isCancelRequested)
        .def("textCanBeEdited", &cloudViewer::GenericProgressCallback::textCanBeEdited);

    py::class_<cloudViewer::NormalizedProgress>(cvcorelib, "NormalizedProgress")
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
