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

/// Upstream pycolmap parity (src/pycolmap/estimators/ceres_bindings.cc):
/// binds the vendored ceres-solver surface (option/result enums, solver
/// options/summaries, and the cost-function base type) into a ``pyceres``
/// submodule of the reconstruction module, so bundle adjustment and the
/// cost-function factories are usable without an external pyceres install.
void pybind_ceres(py::module& m);

}  // namespace estimators
}  // namespace reconstruction
}  // namespace cloudViewer
