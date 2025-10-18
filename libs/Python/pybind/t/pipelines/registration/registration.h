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
namespace pipelines {
namespace registration {

void pybind_registration(py::module &m);
void pybind_feature(py::module &m);
void pybind_robust_kernels(py::module &m);

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace cloudViewer
