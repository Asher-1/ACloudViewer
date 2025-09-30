// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/kernel/Kernel.h"

#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/cloudViewer_pybind.h"

namespace cloudViewer {
namespace core {

void pybind_core_kernel(py::module &m) {
    py::module m_kernel = m.def_submodule("kernel");
    m_kernel.def("test_linalg_integration",
                 &core::kernel::TestLinalgIntegration);
}

}  // namespace core
}  // namespace cloudViewer
