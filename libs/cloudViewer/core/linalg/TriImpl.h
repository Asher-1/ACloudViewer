// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cloudViewer/core/Tensor.h"
#include "cloudViewer/core/linalg/Tri.h"

namespace cloudViewer {
namespace core {

void TriuCPU(const Tensor& A, Tensor& output, const int diagonal = 0);

void TrilCPU(const Tensor& A, Tensor& output, const int diagonal = 0);

void TriulCPU(const Tensor& A,
              Tensor& upper,
              Tensor& lower,
              const int diagonal = 0);

#ifdef BUILD_SYCL_MODULE
void TriuSYCL(const Tensor& A, Tensor& output, const int diagonal = 0);

void TrilSYCL(const Tensor& A, Tensor& output, const int diagonal = 0);

void TriulSYCL(const Tensor& A,
               Tensor& upper,
               Tensor& lower,
               const int diagonal = 0);
#endif

#ifdef BUILD_CUDA_MODULE
void TriuCUDA(const Tensor& A, Tensor& output, const int diagonal = 0);

void TrilCUDA(const Tensor& A, Tensor& output, const int diagonal = 0);

void TriulCUDA(const Tensor& A,
               Tensor& upper,
               Tensor& lower,
               const int diagonal = 0);
#endif
}  // namespace core
}  // namespace cloudViewer
