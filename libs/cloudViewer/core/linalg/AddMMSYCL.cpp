// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <Logging.h>

#include <sycl/sycl.hpp>

#include "cloudViewer/core/SYCLContext.h"
#include "cloudViewer/core/linalg/AddMM.h"
#include "cloudViewer/core/linalg/LinalgUtils.h"
#include "oneapi/mkl.hpp"

namespace cloudViewer {
namespace core {

void AddMMSYCL(void* A_data,
               void* B_data,
               void* C_data,
               int64_t m,
               int64_t k,
               int64_t n,
               double alpha,
               double beta,
               bool gemmTrA,
               bool gemmTrB,
               int lda,
               int ldb,
               int ldc,
               Dtype dtype,
               const Device& device) {
    using namespace oneapi::mkl;
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        blas::column_major::gemm(queue, gemmTrA ? transpose::T : transpose::N,
                                 gemmTrB ? transpose::T : transpose::N, m, n, k,
                                 static_cast<scalar_t>(alpha),
                                 static_cast<const scalar_t*>(A_data), lda,
                                 static_cast<const scalar_t*>(B_data), ldb,
                                 static_cast<scalar_t>(beta),
                                 static_cast<scalar_t*>(C_data), ldc)
                .wait_and_throw();
    });
}

}  // namespace core
}  // namespace cloudViewer
