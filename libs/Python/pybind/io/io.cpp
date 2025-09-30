// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/io/io.h"
#include "pybind/cloudViewer_pybind.h"

namespace cloudViewer {
namespace io {

void pybind_io(py::module &m) {
    py::module m_io = m.def_submodule("io");
    pybind_class_io(m_io);
    pybind_rpc(m_io);
#ifdef BUILD_AZURE_KINECT
    pybind_sensor(m_io);
#endif
}

}  // namespace io
}  // namespace cloudViewer