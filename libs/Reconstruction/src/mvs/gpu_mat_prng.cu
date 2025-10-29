// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "mvs/gpu_mat_prng.h"

namespace colmap {
namespace mvs {
namespace {

__global__ void InitRandomStateKernel(GpuMat<curandState> output) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    const size_t uniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;
    const size_t id = uniqueBlockIndex * blockDim.y * blockDim.x +
                      threadIdx.y * blockDim.x + threadIdx.x;

    // Each thread gets same seed, a different sequence number, no offset.
    if (col < output.GetWidth() && row < output.GetHeight()) {
        curand_init(id, 0, 0, &output.GetRef(row, col));
    }
}

}  // namespace

GpuMatPRNG::GpuMatPRNG(const int width, const int height)
    : GpuMat(width, height) {
    InitRandomStateKernel<<<gridSize_, blockSize_>>>(*this);
}

}  // namespace mvs
}  // namespace colmap
