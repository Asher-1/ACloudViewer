#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ATen/cuda/CUDAContext.h"
#include "ml/contrib/TrilinearDevoxelize.cuh"
#include "ml/contrib/cuda_utils.h"
#include "ml/pytorch/pvcnn/TrilinearDevoxelizeKernel.h"

using namespace cloudViewer::ml::contrib;

void TrilinearDevoxelize(int b,
                         int c,
                         int n,
                         int r,
                         int r2,
                         int r3,
                         bool training,
                         const float *coords,
                         const float *feat,
                         int *inds,
                         float *wgts,
                         float *outs) {
    cudaError_t err;

    auto stream = at::cuda::getCurrentCUDAStream();

    TrilinearDevoxelizeKernel<<<b, OptNumThreads(n), 0, stream>>>(
            b, c, n, r, r2, r3, training, coords, feat, inds, wgts, outs);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void TrilinearDevoxelizeGrad(int b,
                             int c,
                             int n,
                             int r3,
                             const int *inds,
                             const float *wgts,
                             const float *grad_y,
                             float *grad_x) {
    cudaError_t err;

    auto stream = at::cuda::getCurrentCUDAStream();

    TrilinearDevoxelizeGradKernel<<<b, OptNumThreads(n), 0, stream>>>(
            b, c, n, r3, inds, wgts, grad_y, grad_x);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
