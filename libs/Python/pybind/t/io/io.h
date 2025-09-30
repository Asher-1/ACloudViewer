// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/cloudViewer_pybind.h"

namespace cloudViewer {
namespace t {
namespace io {

void pybind_io(py::module& m);
void pybind_class_io(py::module& m);
void pybind_sensor(py::module& m);

}  // namespace io
}  // namespace t
}  // namespace cloudViewer
