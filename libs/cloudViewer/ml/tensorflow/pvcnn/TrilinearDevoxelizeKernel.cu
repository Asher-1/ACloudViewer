// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#define EIGEN_USE_GPU
#include "TrilinearDevoxelizeKernel.h"
#include "ml/Helper.h"
#include "ml/contrib/TrilinearDevoxelize.cuh"
#include "ml/contrib/cuda_utils.h"

using namespace cloudViewer;
using namespace cloudViewer::ml;
using namespace cloudViewer::ml::contrib;
using namespace tensorflow;

class TrilinearDevoxelizeOpKernelCUDA : public TrilinearDevoxelizeOpKernel {
public:
    explicit TrilinearDevoxelizeOpKernelCUDA(OpKernelConstruction* context)
        : TrilinearDevoxelizeOpKernel(context) {}

    void Kernel(tensorflow::OpKernelContext* context,
                int b,
                int c,
                int n,
                int r,
                int r2,
                int r3,
                bool training,
                const float* coords,
                const float* feat,
                int* inds,
                float* wgts,
                float* outs) {
        auto stream = context->eigen_gpu_device().stream();

        cudaError_t err;

        TrilinearDevoxelizeKernel<<<b, OptNumThreads(n), 0, stream>>>(
                b, c, n, r, r2, r3, training, coords, feat, inds, wgts, outs);

        err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "CUDA kernel failed : %s\n",
                    cudaGetErrorString(err));
            exit(-1);
        }
    }
};

REGISTER_KERNEL_BUILDER(
        Name("CloudViewerTrilinearDevoxelize").Device(DEVICE_GPU),
        TrilinearDevoxelizeOpKernelCUDA)

class TrilinearDevoxelizeGradOpKernelCUDA
    : public TrilinearDevoxelizeGradOpKernel {
public:
    explicit TrilinearDevoxelizeGradOpKernelCUDA(OpKernelConstruction* context)
        : TrilinearDevoxelizeGradOpKernel(context) {}

    void Kernel(tensorflow::OpKernelContext* context,
                int b,
                int c,
                int n,
                int r3,
                const int* inds,
                const float* wgts,
                const float* grad_y,
                float* grad_x) {
        auto stream = context->eigen_gpu_device().stream();

        cudaError_t err;

        TrilinearDevoxelizeGradKernel<<<b, OptNumThreads(n), 0, stream>>>(
                b, c, n, r3, inds, wgts, grad_y, grad_x);

        err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "CUDA kernel failed : %s\n",
                    cudaGetErrorString(err));
            exit(-1);
        }
    }
};

REGISTER_KERNEL_BUILDER(
        Name("CloudViewerTrilinearDevoxelizeGrad").Device(DEVICE_GPU),
        TrilinearDevoxelizeGradOpKernelCUDA)
