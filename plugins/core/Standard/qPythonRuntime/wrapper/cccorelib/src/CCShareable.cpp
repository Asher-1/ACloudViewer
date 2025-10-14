// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>

#include <CVShareable.h>

#include "wrappers.h"

namespace py = pybind11;
using namespace pybind11::literals;

void define_CCShareable(py::module &cccorelib)
{
    py::class_<CCShareable, CCShareableHolder<CCShareable>>(cccorelib, "CCShareable")
        .def(py::init<>())
        .def("link", &CCShareable::link)
        .def("release", &CCShareable::release)
        .def("getLinkCount", &CCShareable::getLinkCount)
#ifdef CC_TRACK_ALIVE_SHARED_OBJECTS
        .def_static("GetAliveCount", &CCShareable::GetAliveCount);
#else
        ;
#endif
}
