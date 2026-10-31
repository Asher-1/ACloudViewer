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
namespace sfm {

// Upstream pycolmap parity (src/pycolmap/sfm): the incremental mapping class
// bindings. Registered into the existing `sfm` submodule (no second
// reconstruction entry is added - the function-level pipelines stay canonical
// per the W16 dedup table).
void pybind_mappers(py::module& m);

}  // namespace sfm
}  // namespace reconstruction
}  // namespace cloudViewer
