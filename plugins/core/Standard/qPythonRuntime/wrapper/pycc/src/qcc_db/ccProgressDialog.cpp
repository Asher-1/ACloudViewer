// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <GenericProgressCallback.h>
#include <ecvProgressDialog.h>

#include "../casters.h"

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccProgressDialog(py::module &m)
{
    py::class_<ecvProgressDialog, QProgressDialog, cloudViewer::GenericProgressCallback>(
        m, "ccProgressDialog")
        .def(py::init<bool>(), "cancelButton"_a = false)
        .def("setMethodTitle",
             (void(ecvProgressDialog::*)(QString)) & ecvProgressDialog::setMethodTitle,
             "methodTitle"_a)
        .def("setInfo",
             (void(ecvProgressDialog::*)(QString)) & ecvProgressDialog::setInfo,
             "infoStr"_a);
}
