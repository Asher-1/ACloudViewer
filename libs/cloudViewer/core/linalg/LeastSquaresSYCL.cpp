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
#include "cloudViewer/core/linalg/LeastSquares.h"
#include "cloudViewer/core/linalg/LinalgUtils.h"

namespace cloudViewer {
namespace core {

void LeastSquaresSYCL(void* A_data,
                      void* B_data,
                      int64_t m,
                      int64_t n,
                      int64_t k,
                      Dtype dtype,
                      const Device& device) {
    using namespace oneapi::mkl;
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    int nrhs = k, lda = m, stride_a = lda * n, ldb = std::max(m, n),
        stride_b = ldb * nrhs, batch_size = 1;
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        // Use blob to ensure cleanup of scratchpad memory.
        int64_t scratchpad_size = lapack::gels_batch_scratchpad_size<scalar_t>(
                queue, transpose::N, m, n, nrhs, lda, stride_a, ldb, stride_b,
                batch_size);
        core::Blob scratchpad(scratchpad_size * sizeof(scalar_t), device);
        lapack::gels_batch(
                queue, transpose::N, m, n, nrhs, static_cast<scalar_t*>(A_data),
                lda, stride_a, static_cast<scalar_t*>(B_data), ldb, stride_b,
                batch_size, static_cast<scalar_t*>(scratchpad.GetDataPtr()),
                scratchpad_size)
                .wait_and_throw();
    });
}

}  // namespace core
}  // namespace cloudViewer
