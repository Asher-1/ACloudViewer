// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/core/Blob.h"
#include "cloudViewer/core/CUDAUtils.h"
#include "cloudViewer/core/linalg/BlasWrapper.h"
#include "cloudViewer/core/linalg/LapackWrapper.h"
#include "cloudViewer/core/linalg/LinalgUtils.h"
#include "cloudViewer/core/linalg/Solve.h"

namespace cloudViewer {
namespace core {

// cuSolver's gesv will crash when A is a singular matrix.
// We implement LU decomposition-based solver (similar to Inverse) instead.
void SolveCUDA(void* A_data,
               void* B_data,
               void* ipiv_data,
               int64_t n,
               int64_t k,
               Dtype dtype,
               const Device& device) {
    cusolverDnHandle_t handle =
            CuSolverContext::GetInstance().GetHandle(device);

    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        int len;
        Blob dinfo(sizeof(int), device);

        OPEN3D_CUSOLVER_CHECK(
                getrf_cuda_buffersize<scalar_t>(handle, n, n, n, &len),
                "getrf_buffersize failed in SolveCUDA");
        Blob workspace(len * sizeof(scalar_t), device);

        OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                getrf_cuda<scalar_t>(
                        handle, n, n, static_cast<scalar_t*>(A_data), n,
                        static_cast<scalar_t*>(workspace.GetDataPtr()),
                        static_cast<int*>(ipiv_data),
                        static_cast<int*>(dinfo.GetDataPtr())),
                "getrf failed in SolveCUDA",
                static_cast<int*>(dinfo.GetDataPtr()), device);

        OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                getrs_cuda<scalar_t>(handle, CUBLAS_OP_N, n, k,
                                     static_cast<scalar_t*>(A_data), n,
                                     static_cast<int*>(ipiv_data),
                                     static_cast<scalar_t*>(B_data), n,
                                     static_cast<int*>(dinfo.GetDataPtr())),
                "getrs failed in SolveCUDA",
                static_cast<int*>(dinfo.GetDataPtr()), device);
    });
}

}  // namespace core
}  // namespace cloudViewer
