// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ATen/cuda/CUDAContext.h"
#include "ml/contrib/BallQuery.cuh"
#include "ml/contrib/cuda_utils.h"
#include "ml/pytorch/pointnet/BallQueryKernel.h"

using namespace cloudViewer::ml::contrib;

void ball_query_launcher(int b,
                         int n,
                         int m,
                         float radius,
                         int nsample,
                         const float *new_xyz,
                         const float *xyz,
                         int *idx) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)

    cudaError_t err;

    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK),
                b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    ball_query_kernel<<<blocks, threads, 0, stream>>>(b, n, m, radius, nsample,
                                                      new_xyz, xyz, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
