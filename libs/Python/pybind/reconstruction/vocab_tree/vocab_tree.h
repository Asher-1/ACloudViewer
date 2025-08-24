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
namespace vocab_tree {

void pybind_vocab_tree(py::module &m);

}  // namespace vocab_tree
}  // namespace reconstruction
}  // namespace cloudViewer