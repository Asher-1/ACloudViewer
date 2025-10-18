// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <ecvGenericPointCloud.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccGenericPointCloud(py::module &m)
{
    py::class_<ccGenericPointCloud, ccShiftedObject, cloudViewer::GenericIndexedCloudPersist>(
        m, "ccGenericPointCloud")
        .def("clone",
             &ccGenericPointCloud::clone,
             "desttClous"_a = nullptr,
             "ignoreChildren"_a = false)
        .def("clear", &ccGenericPointCloud::clear);
}
