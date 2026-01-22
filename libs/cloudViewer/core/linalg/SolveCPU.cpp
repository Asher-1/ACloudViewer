// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/core/linalg/LapackWrapper.h"
#include "cloudViewer/core/linalg/LinalgUtils.h"
#include "cloudViewer/core/linalg/Solve.h"

namespace cloudViewer {
namespace core {

void SolveCPU(void* A_data,
              void* B_data,
              void* ipiv_data,
              int64_t n,
              int64_t k,
              Dtype dtype,
              const Device& device) {
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        OPEN3D_LAPACK_CHECK(
                gesv_cpu<scalar_t>(
                        LAPACK_COL_MAJOR, n, k, static_cast<scalar_t*>(A_data),
                        n, static_cast<CLOUDVIEWER_CPU_LINALG_INT*>(ipiv_data),
                        static_cast<scalar_t*>(B_data), n),
                "gels failed in SolveCPU");
    });
}

}  // namespace core
}  // namespace cloudViewer
