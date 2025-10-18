// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/ml/contrib/contrib.h"

#include "pybind/cloudViewer_pybind.h"
#include "pybind/docstring.h"
#include "pybind/pybind_utils.h"

namespace cloudViewer {
namespace ml {
namespace contrib {

void pybind_contrib(py::module& m) {
    py::module m_contrib = m.def_submodule("contrib");

    pybind_contrib_subsample(m_contrib);
    pybind_contrib_nns(m_contrib);
    pybind_contrib_iou(m_contrib);
}

}  // namespace contrib
}  // namespace ml
}  // namespace cloudViewer
