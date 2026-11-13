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

// Upstream pycolmap parity (src/pycolmap/pipeline/match_features.cc): the
// pairing-driven feature matching surface - PairingOptions dataclasses, the
// PairGenerator family, the Device selector, and the match_*/verify entry
// points. Registered into the `pipeline` submodule of `reconstruction`.
void pybind_match_features(py::module& m, py::module& m_root);

}  // namespace pipeline
}  // namespace reconstruction
}  // namespace cloudViewer
