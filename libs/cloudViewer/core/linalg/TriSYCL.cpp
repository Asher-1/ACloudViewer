// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/core/Dispatch.h"
#include "cloudViewer/core/SYCLContext.h"
#include "cloudViewer/core/Tensor.h"
#include "cloudViewer/core/linalg/TriImpl.h"

namespace cloudViewer {
namespace core {

void TriuSYCL(const Tensor &A, Tensor &output, const int diagonal) {
    sycl::queue queue =
            sy::SYCLContext::GetInstance().GetDefaultQueue(A.GetDevice());
    DISPATCH_DTYPE_TO_TEMPLATE(A.GetDtype(), [&]() {
        const scalar_t *A_ptr = static_cast<const scalar_t *>(A.GetDataPtr());
        scalar_t *output_ptr = static_cast<scalar_t *>(output.GetDataPtr());
        auto rows = static_cast<size_t>(A.GetShape()[0]),
             cols = static_cast<size_t>(A.GetShape()[1]);
        queue.parallel_for({cols, rows}, [=](auto wid) {
                 const auto wid_1d = wid[1] * cols + wid[0];
                 if (static_cast<int>(wid[0]) - static_cast<int>(wid[1]) >=
                     diagonal) {
                     output_ptr[wid_1d] = A_ptr[wid_1d];
                 }
             }).wait_and_throw();
    });
}

void TrilSYCL(const Tensor &A, Tensor &output, const int diagonal) {
    sycl::queue queue =
            sy::SYCLContext::GetInstance().GetDefaultQueue(A.GetDevice());
    DISPATCH_DTYPE_TO_TEMPLATE(A.GetDtype(), [&]() {
        const scalar_t *A_ptr = static_cast<const scalar_t *>(A.GetDataPtr());
        scalar_t *output_ptr = static_cast<scalar_t *>(output.GetDataPtr());
        auto rows = static_cast<size_t>(A.GetShape()[0]),
             cols = static_cast<size_t>(A.GetShape()[1]);
        queue.parallel_for({cols, rows}, [=](auto wid) {
                 const auto wid_1d = wid[1] * cols + wid[0];
                 if (static_cast<int>(wid[0]) - static_cast<int>(wid[1]) <=
                     diagonal) {
                     output_ptr[wid_1d] = A_ptr[wid_1d];
                 }
             }).wait_and_throw();
    });
}

void TriulSYCL(const Tensor &A,
               Tensor &upper,
               Tensor &lower,
               const int diagonal) {
    sycl::queue queue =
            sy::SYCLContext::GetInstance().GetDefaultQueue(A.GetDevice());
    DISPATCH_DTYPE_TO_TEMPLATE(A.GetDtype(), [&]() {
        const scalar_t *A_ptr = static_cast<const scalar_t *>(A.GetDataPtr());
        scalar_t *upper_ptr = static_cast<scalar_t *>(upper.GetDataPtr());
        scalar_t *lower_ptr = static_cast<scalar_t *>(lower.GetDataPtr());
        auto rows = static_cast<size_t>(A.GetShape()[0]),
             cols = static_cast<size_t>(A.GetShape()[1]);
        queue.parallel_for({cols, rows}, [=](auto wid) {
                 const auto wid_1d = wid[1] * cols + wid[0];
                 if (static_cast<int>(wid[0]) - static_cast<int>(wid[1]) <
                     diagonal) {
                     lower_ptr[wid_1d] = A_ptr[wid_1d];
                 } else if (static_cast<int>(wid[0]) -
                                    static_cast<int>(wid[1]) >
                            diagonal) {
                     upper_ptr[wid_1d] = A_ptr[wid_1d];
                 } else {
                     lower_ptr[wid_1d] = 1;
                     upper_ptr[wid_1d] = A_ptr[wid_1d];
                 }
             }).wait_and_throw();
    });
}

}  // namespace core
}  // namespace cloudViewer
