// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#define EIGEN_USE_GPU
#include "BallQueryOpKernel.h"
#include "ml/Helper.h"
#include "ml/contrib/BallQuery.cuh"
#include "ml/contrib/cuda_utils.h"

using namespace cloudViewer;
using namespace cloudViewer::ml;
using namespace cloudViewer::ml::contrib;
using namespace tensorflow;

class BallQueryOpKernelCUDA : public BallQueryOpKernel {
public:
    explicit BallQueryOpKernelCUDA(OpKernelConstruction *construction)
        : BallQueryOpKernel(construction) {}

    void Kernel(tensorflow::OpKernelContext *context,
                int b,
                int n,
                int m,
                float radius,
                int nsample,
                const float *new_xyz,
                const float *xyz,
                int *idx) {
        // dataset: (B, N, 3)
        // tmp: (B, N)
        // output:
        //      idx: (B, M)

        auto stream = context->eigen_gpu_device().stream();

        cudaError_t err;

        dim3 blocks(DIVUP(m, THREADS_PER_BLOCK),
                    b);  // blockIdx.x(col), blockIdx.y(row)
        dim3 threads(THREADS_PER_BLOCK);

        ball_query_kernel<<<blocks, threads, 0, stream>>>(
                b, n, m, radius, nsample, new_xyz, xyz, idx);
        // cudaDeviceSynchronize();  // for using printf in kernel function
        err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "CUDA kernel failed : %s\n",
                    cudaGetErrorString(err));
            exit(-1);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("CloudviewerBallQuery").Device(DEVICE_GPU),
                        BallQueryOpKernelCUDA);
