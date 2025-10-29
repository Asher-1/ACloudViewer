// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include <GenericIndexedCloudPersist.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_GenericIndexedCloudPersist(py::module &cccorelib)
{
    py::class_<cloudViewer::GenericIndexedCloudPersist, cloudViewer::GenericIndexedCloud>(
        cccorelib, "GenericIndexedCloudPersist")
        .def("getPointPersistentPtr",
             &cloudViewer::GenericIndexedCloudPersist::getPointPersistentPtr,
             "index"_a,
             py::return_value_policy::reference);
}
