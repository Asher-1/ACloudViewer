// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/cloudViewer_pybind.h"

namespace cloudViewer {
namespace reconstruction {
namespace estimators {

/// Upstream pycolmap parity (src/pycolmap/estimators/cost_functions.cc):
/// binds the engine ceres cost-function factories into a ``cost_functions``
/// submodule of the reconstruction module.
void pybind_cost_functions(py::module& m);

}  // namespace estimators
}  // namespace reconstruction
}  // namespace cloudViewer
