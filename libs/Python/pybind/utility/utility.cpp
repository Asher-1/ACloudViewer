// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/utility/utility.h"

#include "pybind/docstring.h"
#include "pybind/cloudViewer_pybind.h"

namespace cloudViewer {
namespace utility {

void pybind_utility(py::module &m) {
    py::module m_submodule = m.def_submodule("utility");
    pybind_scalarfield(m_submodule);
    pybind_matrix(m_submodule);
    pybind_logging(m_submodule);
    pybind_eigen(m_submodule);
}

}  // namespace utility
}  // namespace cloudViewer
