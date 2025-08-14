// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/cloudViewer_pybind.h"
#include "pybind/pybind_utils.h"

namespace cloudViewer {
namespace core {
namespace nns {

void pybind_core_nns(py::module &m);

}  // namespace nns
}  // namespace core
}  // namespace cloudViewer
