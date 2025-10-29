// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include <GenericOctree.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_GenericOctree(py::module &cccorelib)
{
    py::class_<cloudViewer::GenericOctree>(cccorelib, "GenericOctree");
}
