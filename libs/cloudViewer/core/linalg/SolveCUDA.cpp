// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "core/Blob.h"
#include "core/CUDAUtils.h"
#include "core/linalg/BlasWrapper.h"
#include "core/linalg/LapackWrapper.h"
#include "core/linalg/LinalgUtils.h"
#include "core/linalg/Solve.h"

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

        CLOUDVIEWER_CUSOLVER_CHECK(
                getrf_cuda_buffersize<scalar_t>(handle, n, n, n, &len),
                "getrf_buffersize failed in SolveCUDA");
        Blob workspace(len * sizeof(scalar_t), device);

        CLOUDVIEWER_CUSOLVER_CHECK_WITH_DINFO(
                getrf_cuda<scalar_t>(
                        handle, n, n, static_cast<scalar_t*>(A_data), n,
                        static_cast<scalar_t*>(workspace.GetDataPtr()),
                        static_cast<int*>(ipiv_data),
                        static_cast<int*>(dinfo.GetDataPtr())),
                "getrf failed in SolveCUDA",
                static_cast<int*>(dinfo.GetDataPtr()), device);

        CLOUDVIEWER_CUSOLVER_CHECK_WITH_DINFO(
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
