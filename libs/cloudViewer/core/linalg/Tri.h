// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cloudViewer/core/Tensor.h"

namespace cloudViewer {
namespace core {

// See documentation for `core::Tensor::Triu`.
void Triu(const Tensor& A, Tensor& output, const int diagonal = 0);

// See documentation for `core::Tensor::Tril`.
void Tril(const Tensor& A, Tensor& output, const int diagonal = 0);

// See documentation for `core::Tensor::Triul`.
void Triul(const Tensor& A,
           Tensor& upper,
           Tensor& lower,
           const int diagonal = 0);

}  // namespace core
}  // namespace cloudViewer
