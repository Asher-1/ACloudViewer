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
namespace util {

// Upstream pycolmap parity (src/pycolmap/util/{timer,logging,cancellation}.cc):
// the utility surface - Timer, glog control, and cancellation tokens.
// Registered into the `util` submodule of `reconstruction`.
void pybind_util(py::module& m);

}  // namespace util
}  // namespace reconstruction
}  // namespace cloudViewer
