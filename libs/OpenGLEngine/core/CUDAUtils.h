// ----------------------------------------------------------------------------
// -                        CloudViewer: www.erow.cn                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
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

/// \file CUDAUtils.h
/// \brief Common CUDA utilities
///
/// CUDAUtils.h may be included from CPU-only code.
/// Use \#ifdef __CUDACC__ to mark conitional compilation

#pragma once

#include <Console.h>

#ifdef BUILD_CUDA_MODULE

#include <cuda.h>
#include <cuda_runtime.h>

#define CLOUDVIEWER_HOST_DEVICE __host__ __device__
#define CLOUDVIEWER_DEVICE __device__
#define CLOUDVIEWER_ASSERT_HOST_DEVICE_LAMBDA(type)                       \
    static_assert(__nv_is_extended_host_device_lambda_closure_type(type), \
                  #type " must be a __host__ __device__ lambda")
#define CLOUDVIEWER_CUDA_CHECK(err) \
    cloudViewer::core::__CLOUDVIEWER_CUDA_CHECK(err, __FILE__, __LINE__)
#define CLOUDVIEWER_GET_LAST_CUDA_ERROR(message) \
    __CLOUDVIEWER_GET_LAST_CUDA_ERROR(message, __FILE__, __LINE__)

#define CUDA_CALL(cuda_function, ...) cuda_function(__VA_ARGS__);

#else  // #ifdef BUILD_CUDA_MODULE

#define CLOUDVIEWER_HOST_DEVICE
#define CLOUDVIEWER_DEVICE
#define CLOUDVIEWER_ASSERT_HOST_DEVICE_LAMBDA(type)
#define CLOUDVIEWER_CUDA_CHECK(err)
#define CLOUDVIEWER_GET_LAST_CUDA_ERROR(message)
#define CUDA_CALL(cuda_function, ...) \
    utility::LogError("Not built with CUDA, cannot call " #cuda_function);

#endif  // #ifdef BUILD_CUDA_MODULE

namespace cloudViewer {
namespace core {
#ifdef BUILD_CUDA_MODULE
inline void __CLOUDVIEWER_CUDA_CHECK(cudaError_t err,
                                const char* file,
                                const int line) {
    if (err != cudaSuccess) {
        cloudViewer::utility::LogError("{}:{} CUDA runtime error: {}", file, line,
                          cudaGetErrorString(err));
    }
}

inline void __CLOUDVIEWER_GET_LAST_CUDA_ERROR(const char* message,
                                         const char* file,
                                         const int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cloudViewer::utility::LogError("{}:{} {}: CLOUDVIEWER_GET_LAST_CUDA_ERROR(): {}", file,
                          line, message, cudaGetErrorString(err));
    }
}

/// Returns the texture alignment in bytes for the current device.
inline int GetCUDACurrentDeviceTextureAlignment() {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        cloudViewer::utility::LogError(
                "GetCUDACurrentDeviceTextureAlignment(): cudaGetDevice failed "
                "with {}",
                cudaGetErrorString(err));
    }

    int value = 0;
    err = cudaDeviceGetAttribute(&value, cudaDevAttrTextureAlignment, device);
    if (err != cudaSuccess) {
        cloudViewer::utility::LogError(
                "GetCUDACurrentDeviceTextureAlignment(): "
                "cudaDeviceGetAttribute failed with {}",
                cudaGetErrorString(err));
    }
    return value;
}
#endif

namespace cuda {

int DeviceCount();
bool IsAvailable();
void ReleaseCache();

}  // namespace cuda
}  // namespace core
}  // namespace cloudViewer
