// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/Dispatch.h"
#include "core/ParallelFor.h"
#include "core/Tensor.h"
#include "core/linalg/TriImpl.h"

namespace cloudViewer {
namespace core {

void TriuCUDA(const Tensor &A, Tensor &output, const int diagonal) {
    DISPATCH_DTYPE_TO_TEMPLATE(A.GetDtype(), [&]() {
        const scalar_t *A_ptr = static_cast<const scalar_t *>(A.GetDataPtr());
        scalar_t *output_ptr = static_cast<scalar_t *>(output.GetDataPtr());
        int cols = A.GetShape()[1];
        int n = A.GetShape()[0] * cols;

        core::ParallelFor(
                A.GetDevice(), n, [=] CLOUDVIEWER_DEVICE(int64_t workload_idx) {
                    const int64_t idx = workload_idx / cols;
                    const int64_t idy = workload_idx % cols;
                    if (idy - idx >= diagonal) {
                        output_ptr[workload_idx] = A_ptr[idx * cols + idy];
                    }
                });
    });
}

void TrilCUDA(const Tensor &A, Tensor &output, const int diagonal) {
    DISPATCH_DTYPE_TO_TEMPLATE(A.GetDtype(), [&]() {
        const scalar_t *A_ptr = static_cast<const scalar_t *>(A.GetDataPtr());
        scalar_t *output_ptr = static_cast<scalar_t *>(output.GetDataPtr());
        int cols = A.GetShape()[1];
        int n = A.GetShape()[0] * cols;

        core::ParallelFor(
                A.GetDevice(), n, [=] CLOUDVIEWER_DEVICE(int64_t workload_idx) {
                    const int64_t idx = workload_idx / cols;
                    const int64_t idy = workload_idx % cols;
                    if (idy - idx <= diagonal) {
                        output_ptr[workload_idx] = A_ptr[idx * cols + idy];
                    }
                });
    });
}

void TriulCUDA(const Tensor &A,
               Tensor &upper,
               Tensor &lower,
               const int diagonal) {
    DISPATCH_DTYPE_TO_TEMPLATE(A.GetDtype(), [&]() {
        const scalar_t *A_ptr = static_cast<const scalar_t *>(A.GetDataPtr());
        scalar_t *lower_ptr = static_cast<scalar_t *>(lower.GetDataPtr());
        scalar_t *upper_ptr = static_cast<scalar_t *>(upper.GetDataPtr());
        int cols = A.GetShape()[1];
        int n = A.GetShape()[0] * cols;

        core::ParallelFor(
                A.GetDevice(), n, [=] CLOUDVIEWER_DEVICE(int64_t workload_idx) {
                    const int64_t idx = workload_idx / cols;
                    const int64_t idy = workload_idx % cols;
                    if (idy - idx < diagonal) {
                        lower_ptr[workload_idx] = A_ptr[idx * cols + idy];
                    } else if (idy - idx > diagonal) {
                        upper_ptr[workload_idx] = A_ptr[idx * cols + idy];
                    } else {
                        lower_ptr[workload_idx] = 1;
                        upper_ptr[workload_idx] = A_ptr[idx * cols + idy];
                    }
                });
    });
}

}  // namespace core
}  // namespace cloudViewer
