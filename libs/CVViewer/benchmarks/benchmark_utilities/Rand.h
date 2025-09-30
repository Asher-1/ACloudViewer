// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cloudViewer/core/Dtype.h"
#include "cloudViewer/core/Scalar.h"
#include "cloudViewer/core/SizeVector.h"
#include "cloudViewer/core/Tensor.h"

namespace cloudViewer {
namespace benchmarks {

/// Returns a Tensor with random values within the range \p range .
core::Tensor Rand(const core::SizeVector& shape,
                  size_t seed,
                  const std::pair<core::Scalar, core::Scalar>& range,
                  core::Dtype dtype,
                  const core::Device& device = core::Device("CPU:0"));

}  // namespace benchmarks
}  // namespace cloudViewer
