// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/cloudViewer_pybind.h"

namespace cloudViewer {
namespace t {
namespace pipelines {
namespace slam {

void pybind_slam(py::module &m);

}  // namespace slam
}  // namespace pipelines
}  // namespace t
}  // namespace cloudViewer
