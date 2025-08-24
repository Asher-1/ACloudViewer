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

void pybind_structure_from_motion(py::module &m);

}  // namespace sfm
}  // namespace reconstruction
}  // namespace cloudViewer