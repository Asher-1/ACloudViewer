// ----------------------------------------------------------------------------
// -                        cloudViewer: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/linalg/AddMM.h"
#include "core/linalg/BlasWrapper.h"
#include "core/linalg/LinalgUtils.h"
#include <Logging.h>

namespace cloudViewer {
namespace core {

void AddMMCUDA(void* A_data,
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
    cublasHandle_t handle = CuBLASContext::GetInstance().GetHandle(device);
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t alpha_ = scalar_t(alpha);
        scalar_t beta_ = scalar_t(beta);
        CLOUDVIEWER_CUBLAS_CHECK(
                gemm_cuda(handle, gemmTrA ? CUBLAS_OP_T : CUBLAS_OP_N,
                          gemmTrB ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha_,
                          static_cast<const scalar_t*>(A_data), lda,
                          static_cast<const scalar_t*>(B_data), ldb, &beta_,
                          static_cast<scalar_t*>(C_data), ldc),
                "cuda gemm failed");
    });
}

}  // namespace core
}  // namespace cloudViewer
