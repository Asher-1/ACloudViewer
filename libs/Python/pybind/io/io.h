// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/cloudViewer_pybind.h"

namespace cloudViewer {
namespace io {

void pybind_io(py::module& m);

void pybind_class_io(py::module& m);

void pybind_rpc(py::module& m);

#ifdef BUILD_AZURE_KINECT
void pybind_sensor(py::module& m);
#endif

}  // namespace io
}  // namespace cloudViewer