// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>

#include <vector>

#include "ATen/cuda/CUDAContext.h"
#include "ml/contrib/PointSampling.cuh"
#include "ml/contrib/cuda_utils.h"
#include "ml/pytorch/pointnet/SamplingKernel.h"

using namespace cloudViewer::ml::contrib;

void furthest_point_sampling_launcher(
        int b, int n, int m, const float *dataset, float *temp, int *idxs) {
    // dataset: (B, N, 3)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    cudaError_t err;

    auto stream = at::cuda::getCurrentCUDAStream();

    unsigned int n_threads = OptNumThreads(n);

    switch (n_threads) {
        case 1024:
            furthest_point_sampling_kernel<1024>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 512:
            furthest_point_sampling_kernel<512>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 256:
            furthest_point_sampling_kernel<256>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 128:
            furthest_point_sampling_kernel<128>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 64:
            furthest_point_sampling_kernel<64>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 32:
            furthest_point_sampling_kernel<32>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 16:
            furthest_point_sampling_kernel<16>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 8:
            furthest_point_sampling_kernel<8>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 4:
            furthest_point_sampling_kernel<4>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 2:
            furthest_point_sampling_kernel<2>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 1:
            furthest_point_sampling_kernel<1>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        default:
            furthest_point_sampling_kernel<512>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    }

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
