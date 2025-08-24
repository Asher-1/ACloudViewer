// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/cloudViewer_pybind.h"

namespace cloudViewer {
namespace pipelines {
namespace integration {

void pybind_integration(py::module &m);

}  // namespace integration
}  // namespace pipelines
}  // namespace cloudViewer