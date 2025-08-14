// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/cloudViewer_pybind.h"
#include "pybind/pybind_utils.h"

namespace cloudViewer {
namespace ml {
namespace contrib {

void pybind_contrib(py::module &m);
void pybind_contrib_subsample(py::module &m_contrib);
void pybind_contrib_nns(py::module &m_contrib);
void pybind_contrib_iou(py::module &m_contrib);

}  // namespace contrib
}  // namespace ml
}  // namespace cloudViewer
