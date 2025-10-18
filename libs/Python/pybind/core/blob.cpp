// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/Blob.h"

#include "pybind/cloudViewer_pybind.h"
#include "pybind/core/core.h"
#include "pybind/docstring.h"

namespace cloudViewer {
namespace core {

void pybind_core_blob(py::module &m) { py::class_<Blob> blob(m, "Blob"); }

}  // namespace core
}  // namespace cloudViewer
