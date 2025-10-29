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
void MatmulSYCL(void* A_data,
                void* B_data,
                void* C_data,
                int64_t m,
                int64_t k,
                int64_t n,
                Dtype dtype,
                const Device& device) {
    using namespace oneapi::mkl;
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t alpha = 1, beta = 0;
        blas::column_major::gemm(queue, transpose::N, transpose::N, m, n, k,
                                 alpha, static_cast<const scalar_t*>(A_data), m,
                                 static_cast<const scalar_t*>(B_data), k, beta,
                                 static_cast<scalar_t*>(C_data), m)
                .wait_and_throw();
    });
}

}  // namespace core
}  // namespace cloudViewer
