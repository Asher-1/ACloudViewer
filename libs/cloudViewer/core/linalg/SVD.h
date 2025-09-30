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

/// Computes SVD decomposition A = U S VT, where A is an m x n, U is an m x m, S
/// is a min(m, n), VT is an n x n tensor.
void SVD(const Tensor& A, Tensor& U, Tensor& S, Tensor& VT);

#ifdef BUILD_SYCL_MODULE
void SVDSYCL(const void* A_data,
             void* U_data,
             void* S_data,
             void* VT_data,
             int64_t m,
             int64_t n,
             Dtype dtype,
             const Device& device);
#endif

#ifdef BUILD_CUDA_MODULE
void SVDCUDA(const void* A_data,
             void* U_data,
             void* S_data,
             void* VT_data,
             void* superb_data,
             int64_t m,
             int64_t n,
             Dtype dtype,
             const Device& device);
#endif

void SVDCPU(const void* A_data,
            void* U_data,
            void* S_data,
            void* VT_data,
            void* superb_data,
            int64_t m,
            int64_t n,
            Dtype dtype,
            const Device& device);

}  // namespace core
}  // namespace cloudViewer
