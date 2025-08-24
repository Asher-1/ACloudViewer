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
namespace mvs {

void pybind_multi_views_stereo(py::module &m);

}  // namespace mvs
}  // namespace reconstruction
}  // namespace cloudViewer