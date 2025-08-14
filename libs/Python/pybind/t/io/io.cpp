// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/t/io/io.h"

#include "pybind/cloudViewer_pybind.h"

namespace cloudViewer {
namespace t {
namespace io {

void pybind_io(py::module& m) {
    py::module m_io =
            m.def_submodule("io", "Tensor-based input-output handling module.");
    pybind_class_io(m_io);
    pybind_sensor(m_io);
}

}  // namespace io
}  // namespace t
}  // namespace cloudViewer
