// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/ml/ml.h"

#include "pybind/cloudViewer_pybind.h"
#include "pybind/ml/contrib/contrib.h"

namespace cloudViewer {
namespace ml {

void pybind_ml(py::module &m) {
    py::module m_ml = m.def_submodule("ml");
    contrib::pybind_contrib(m_ml);
}

}  // namespace ml
}  // namespace cloudViewer
