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
namespace options {

// Registers the whole options surface on the functional submodules that
// own it (upstream pycolmap has no "options" namespace). MUST be called
// FIRST in the module assembly: several binders reference these classes as
// default-argument values of their entry points, which pybind converts at
// registration time.
void pybind_reconstruction_options_early(py::module &m);

}  // namespace options
}  // namespace reconstruction
}  // namespace cloudViewer