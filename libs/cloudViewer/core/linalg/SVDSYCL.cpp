// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <sycl/sycl.hpp>

#include "oneapi/mkl.hpp"
#include "cloudViewer/core/Blob.h"
#include "cloudViewer/core/SYCLContext.h"
#include "cloudViewer/core/linalg/LinalgUtils.h"
#include "cloudViewer/core/linalg/SVD.h"

namespace cloudViewer {
namespace core {

void SVDSYCL(const void* A_data,
             void* U_data,
             void* S_data,
             void* VT_data,
             int64_t m,
             int64_t n,
             Dtype dtype,
             const Device& device) {
    using namespace oneapi::mkl;
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        int64_t lda = m, ldvt = n, ldu = m;
        int64_t scratchpad_size = lapack::gesvd_scratchpad_size<scalar_t>(
                queue, jobsvd::vectors, jobsvd::vectors, m, n, lda, ldu, ldvt);
        // Use blob to ensure cleanup of scratchpad memory.
        Blob scratchpad(scratchpad_size * sizeof(scalar_t), device);
        lapack::gesvd(
                queue, jobsvd::vectors, jobsvd::vectors, m, n,
                const_cast<scalar_t*>(static_cast<const scalar_t*>(A_data)),
                lda, static_cast<scalar_t*>(S_data),
                static_cast<scalar_t*>(U_data), ldu,
                static_cast<scalar_t*>(VT_data), ldvt,
                static_cast<scalar_t*>(scratchpad.GetDataPtr()),
                scratchpad_size)
                .wait_and_throw();
    });
}

}  // namespace core
}  // namespace cloudViewer
