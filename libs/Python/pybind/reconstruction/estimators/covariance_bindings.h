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

// Upstream pycolmap parity (src/pycolmap/estimators/covariance.cc): the
// bundle-adjustment covariance surface. Registered into the existing
// `estimators` submodule.
void pybind_covariance(py::module& m);

}  // namespace estimators
}  // namespace reconstruction
}  // namespace cloudViewer
