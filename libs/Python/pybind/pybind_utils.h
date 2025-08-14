// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "core/Dtype.h"
#include "core/Tensor.h"
#include "pybind/cloudViewer_pybind.h"

namespace cloudViewer {
namespace pybind_utils {

core::Dtype ArrayFormatToDtype(const std::string& format, size_t byte_size);

std::string DtypeToArrayFormat(const core::Dtype& dtype);

}  // namespace pybind_utils

}  // namespace cloudViewer
