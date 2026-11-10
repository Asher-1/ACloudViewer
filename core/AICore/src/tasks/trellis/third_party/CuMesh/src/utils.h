// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace cumesh {
namespace detail {

// Raised for every failed CUDA call. Derives from std::exception so the C
// glue's catch blocks keep converting failures into a 0 return value.
[[noreturn]] inline void cuda_check_fail(const char* file,
                                         int line,
                                         cudaError_t error_code) {
    std::ostringstream message;
    message << "[CuMesh] CUDA error:\n"
            << "    File:       " << file << "\n"
            << "    Line:       " << line << "\n"
            << "    Error code: " << static_cast<int>(error_code) << "\n"
            << "    Error text: " << cudaGetErrorString(error_code) << "\n";
    throw std::runtime_error(message.str());
}

}  // namespace detail
}  // namespace cumesh

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        const cudaError_t error_code = call;                                  \
        if (error_code != cudaSuccess) {                                      \
            ::cumesh::detail::cuda_check_fail(__FILE__, __LINE__, error_code); \
        }                                                                     \
    } while (0)

namespace cumesh {

/**
 * Stream every CuMesh kernel and copy is ordered on: the legacy default
 * stream. The glue previously pinned PyTorch's current stream to its default
 * stream (handle 0) for the whole call, so this is the identical stream.
 */
inline cudaStream_t current_stream() { return cudaStream_t(0); }

/**
 * A GPU buffer class that manages device memory.
 */
template <typename T>
struct Buffer {
    T* ptr;
    size_t size;
    size_t capacity;

    Buffer() : ptr(nullptr), size(0), capacity(0) {}

    bool is_empty() const { return size == 0; }

    void init(size_t capacity) {
        this->capacity = capacity;
        CUDA_CHECK(cudaMalloc(&ptr, capacity * sizeof(T)));
    }

    void free() {
        if (ptr != nullptr) {
            cudaStream_t stream = current_stream();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaFree(ptr));
        }
        ptr = nullptr;
        size = 0;
        capacity = 0;
    }

    void resize(size_t size) {
        if (size > capacity) {
            free();
            init(size);
        }
        this->size = size;
    }

    void extend(size_t size) {
        size_t new_size = size + this->size;
        if (new_size > capacity) {
            T* new_ptr;
            CUDA_CHECK(cudaMalloc(&new_ptr, new_size * sizeof(T)));
            cudaStream_t stream = current_stream();
            CUDA_CHECK(cudaMemcpyAsync(new_ptr, ptr, this->size * sizeof(T),
                                       cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaFree(ptr));
            ptr = new_ptr;
            this->capacity = new_size;
        }
        this->size = new_size;
    }

    void zero() {
        cudaStream_t stream = current_stream();
        CUDA_CHECK(cudaMemsetAsync(ptr, 0, size * sizeof(T), stream));
    }

    void fill(T val) {
        cudaStream_t stream = current_stream();
        std::vector<T> tmp(size, val);
        CUDA_CHECK(cudaMemcpyAsync(ptr, tmp.data(), size * sizeof(T),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    /** Synchronous device-to-host copy of the first `size` elements. */
    std::vector<T> to_host() const {
        std::vector<T> host(size);
        if (size > 0) {
            CUDA_CHECK(cudaMemcpy(host.data(), ptr, size * sizeof(T),
                                  cudaMemcpyDeviceToHost));
        }
        return host;
    }
};

/**
 * Swap the contents of two buffers.
 */
template <typename T1, typename T2>
void swap_buffers(Buffer<T1>& b1, Buffer<T2>& b2) {
    void* b1_ptr = reinterpret_cast<void*>(b1.ptr);
    void* b2_ptr = reinterpret_cast<void*>(b2.ptr);
    size_t b1_capacity_bytes = b1.capacity * sizeof(T1);
    size_t b2_capacity_bytes = b2.capacity * sizeof(T2);
    size_t b1_size_bytes = b1.size * sizeof(T1);
    size_t b2_size_bytes = b2.size * sizeof(T2);

    b1.ptr = reinterpret_cast<T1*>(b2_ptr);
    b2.ptr = reinterpret_cast<T2*>(b1_ptr);
    b1.capacity = b2_capacity_bytes / sizeof(T1);
    b2.capacity = b1_capacity_bytes / sizeof(T2);
    b1.size = b2_size_bytes / sizeof(T1);
    b2.size = b1_size_bytes / sizeof(T2);
}

}  // namespace cumesh
