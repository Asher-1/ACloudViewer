// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cloudViewer/core/Tensor.h"

namespace cloudViewer {
namespace core {

// See documentation for `core::Tensor::LUIpiv`.
void LUIpiv(const Tensor& A, Tensor& ipiv, Tensor& output);

// See documentation for `core::Tensor::LU`.
void LU(const Tensor& A,
        Tensor& permutation,
        Tensor& lower,
        Tensor& upper,
        const bool permute_l = false);

}  // namespace core
}  // namespace cloudViewer
