// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/core/linalg/AddMM.h"
#include "cloudViewer/core/linalg/BlasWrapper.h"
#include "cloudViewer/core/linalg/LinalgUtils.h"
#include <Logging.h>

namespace cloudViewer {
namespace core {

void AddMMCPU(void* A_data,
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
              Dtype dtype) {
    DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
        gemm_cpu(CblasColMajor, gemmTrA ? CblasTrans : CblasNoTrans,
                 gemmTrB ? CblasTrans : CblasNoTrans, m, n, k,
                 static_cast<scalar_t>(alpha),
                 static_cast<const scalar_t*>(A_data), lda,
                 static_cast<const scalar_t*>(B_data), ldb,
                 static_cast<scalar_t>(beta), static_cast<scalar_t*>(C_data),
                 ldc);
    });
}

}  // namespace core
}  // namespace cloudViewer
