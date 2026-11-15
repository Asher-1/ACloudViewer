// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#include "pytorch_cuda_rng.hpp"

namespace sam3d {
namespace {

constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint32_t kUnrollFactor = 4;
constexpr uint64_t kOffsetPerCurandCall = 4;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

__global__ void pytorch_normal_f32_kernel(float* output,
                                          int64_t count,
                                          uint64_t seed,
                                          uint64_t offset) {
    const int64_t thread_index =
            static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, thread_index, offset, &state);

    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t iteration_stride = stride * kUnrollFactor;
    const int64_t rounded_count =
            ((count + iteration_stride - 1) / iteration_stride) *
            iteration_stride;
    for (int64_t index = thread_index; index < rounded_count;
         index += iteration_stride) {
        const float4 samples = curand_normal4(&state);
#pragma unroll
        for (int lane = 0; lane < kUnrollFactor; ++lane) {
            const int64_t output_index = index + stride * lane;
            if (output_index < count) output[output_index] = (&samples.x)[lane];
        }
        __syncthreads();
    }
}

uint32_t distribution_blocks_for(const cudaDeviceProp& properties) {
    const uint64_t blocks_per_sm =
            static_cast<uint64_t>(properties.maxThreadsPerMultiProcessor) /
            kThreadsPerBlock;
    return static_cast<uint32_t>(
            static_cast<uint64_t>(properties.multiProcessorCount) *
            blocks_per_sm);
}

uint32_t grid_size_for(int64_t count, uint32_t distribution_blocks) {
    const uint64_t requested =
            (static_cast<uint64_t>(count) + kThreadsPerBlock - 1) /
            kThreadsPerBlock;
    return static_cast<uint32_t>(
            std::min<uint64_t>(requested, distribution_blocks));
}

uint64_t offset_increment_for(int64_t count, uint32_t grid_size) {
    const uint64_t per_iteration =
            static_cast<uint64_t>(kThreadsPerBlock) * grid_size * kUnrollFactor;
    return ((static_cast<uint64_t>(count) + per_iteration - 1) /
            per_iteration) *
           kOffsetPerCurandCall;
}

}  // namespace

PytorchCudaNormalRng::PytorchCudaNormalRng(uint64_t seed,
                                           uint32_t distribution_blocks)
    : seed_(seed), distribution_blocks_(distribution_blocks) {}

bool PytorchCudaNormalRng::distribution_blocks(uint32_t& blocks,
                                               std::string& error) {
    try {
        int device = 0;
        cudaDeviceProp properties{};
        check_cuda(cudaGetDevice(&device),
                   "cudaGetDevice for PyTorch-compatible RNG");
        check_cuda(cudaGetDeviceProperties(&properties, device),
                   "cudaGetDeviceProperties for PyTorch-compatible RNG");
        blocks = distribution_blocks_for(properties);
        if (blocks == 0) {
            error = "CUDA device reported zero available Philox distribution "
                    "blocks";
            return false;
        }
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

bool PytorchCudaNormalRng::fill(std::vector<float>& values,
                                std::string& error) {
    if (values.empty()) return true;
    if (values.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        error = "PyTorch-compatible CUDA RNG only supports up to INT_MAX "
                "elements per draw";
        return false;
    }

    try {
        int device = 0;
        cudaDeviceProp properties{};
        check_cuda(cudaGetDevice(&device),
                   "cudaGetDevice for PyTorch-compatible RNG");
        check_cuda(cudaGetDeviceProperties(&properties, device),
                   "cudaGetDeviceProperties for PyTorch-compatible RNG");
        const uint32_t device_distribution_blocks =
                distribution_blocks_for(properties);
        if (device_distribution_blocks == 0) {
            error = "CUDA device reported zero available Philox distribution "
                    "blocks";
            return false;
        }
        if (distribution_blocks_ != 0 &&
            distribution_blocks_ > device_distribution_blocks) {
            error = "requested PyTorch Philox distribution blocks exceed CUDA "
                    "device capacity";
            return false;
        }
        const int64_t count = static_cast<int64_t>(values.size());
        const uint32_t grid_size = grid_size_for(
                count, distribution_blocks_ == 0 ? device_distribution_blocks
                                                 : distribution_blocks_);
        if (grid_size == 0) {
            error = "CUDA device reported zero available Philox distribution "
                    "blocks";
            return false;
        }

        float* device_values = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_values),
                              values.size() * sizeof(float)),
                   "cudaMalloc PyTorch-compatible RNG output");
        try {
            pytorch_normal_f32_kernel<<<grid_size, kThreadsPerBlock>>>(
                    device_values, count, seed_, offset_);
            check_cuda(cudaGetLastError(),
                       "PyTorch-compatible RNG kernel launch");
            check_cuda(cudaMemcpy(values.data(), device_values,
                                  values.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy PyTorch-compatible RNG output");
        } catch (...) {
            cudaFree(device_values);
            throw;
        }
        check_cuda(cudaFree(device_values),
                   "cudaFree PyTorch-compatible RNG output");
        offset_ += offset_increment_for(count, grid_size);
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

}  // namespace sam3d
