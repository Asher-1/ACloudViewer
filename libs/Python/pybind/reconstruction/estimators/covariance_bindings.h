// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/cloudViewer_pybind.h"

namespace ceres {
class Problem;
}  // namespace ceres

namespace cloudViewer {
namespace reconstruction {
namespace estimators {

// Opaque handle for an engine ceres::Problem (the fork does not bind the
// ceres Problem API; the handle round-trips a problem from the bundle
// adjuster into estimate_ba_covariance_from_problem and exposes block counts).
// Bound as "CeresProblem" by pybind_covariance(); the referenced problem is
// owned by the producing adjuster/averager.
struct PyCeresProblem {
    ceres::Problem* problem = nullptr;
};

// Upstream pycolmap parity (src/pycolmap/estimators/covariance.cc): the
// bundle-adjustment covariance surface. Registered into the existing
// `estimators` submodule.
void pybind_covariance(py::module& m);

}  // namespace estimators
}  // namespace reconstruction
}  // namespace cloudViewer
