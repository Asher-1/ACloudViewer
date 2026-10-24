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
namespace pipeline {

/// Upstream pycolmap parity bindings for the pipeline options family
/// (global/hierarchical pipelines, undistortion, scene clustering, view
/// graph calibration, dataset synthesis).
void pybind_pipeline(py::module& m, py::module& m_root);

}  // namespace pipeline
}  // namespace reconstruction
}  // namespace cloudViewer
