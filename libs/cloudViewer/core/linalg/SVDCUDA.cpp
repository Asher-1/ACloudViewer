// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/core/Blob.h"
#include "cloudViewer/core/linalg/LapackWrapper.h"
#include "cloudViewer/core/linalg/LinalgUtils.h"
#include "cloudViewer/core/linalg/SVD.h"

namespace cloudViewer {
namespace core {

void SVDCUDA(const void* A_data,
             void* U_data,
             void* S_data,
             void* VT_data,
             void* superb_data,
             int64_t m,
             int64_t n,
             Dtype dtype,
             const Device& device) {
    cusolverDnHandle_t handle =
            CuSolverContext::GetInstance().GetHandle(device);

    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        int len;
        Blob dinfo(sizeof(int), device);

        OPEN3D_CUSOLVER_CHECK(
                gesvd_cuda_buffersize<scalar_t>(handle, m, n, &len),
                "gesvd_buffersize failed in SVDCUDA");

        Blob workspace(len * sizeof(scalar_t), device);

        OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                gesvd_cuda<scalar_t>(
                        handle, 'A', 'A', m, n,
                        const_cast<scalar_t*>(
                                static_cast<const scalar_t*>(A_data)),
                        m, static_cast<scalar_t*>(S_data),
                        static_cast<scalar_t*>(U_data), m,
                        static_cast<scalar_t*>(VT_data), n,
                        static_cast<scalar_t*>(workspace.GetDataPtr()), len,
                        static_cast<scalar_t*>(superb_data),
                        static_cast<int*>(dinfo.GetDataPtr())),
                "gesvd failed in SVDCUDA",
                static_cast<int*>(dinfo.GetDataPtr()), device);
    });
}
}  // namespace core
}  // namespace cloudViewer
