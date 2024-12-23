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

#pragma once

#include <cstdint>
#include <vector>

#include "core/Device.h"
#include "utility/Overload.h"
#include "utility/Preprocessor.h"
#include <Logging.h>
#include <Parallel.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>

#include "core/CUDAUtils.h"
#endif

namespace cloudViewer {
namespace core {

#ifdef __CUDACC__

static constexpr int64_t CLOUDVIEWER_PARFOR_BLOCK = 128;
static constexpr int64_t CLOUDVIEWER_PARFOR_THREAD = 4;

/// Calls f(n) with the "grid-stride loops" pattern.
template <int64_t block_size, int64_t thread_size, typename func_t>
__global__ void ElementWiseKernel_(int64_t n, func_t f) {
    int64_t items_per_block = block_size * thread_size;
    int64_t idx = blockIdx.x * items_per_block + threadIdx.x;
#pragma unroll
    for (int64_t i = 0; i < thread_size; ++i) {
        if (idx < n) {
            f(idx);
            idx += block_size;
        }
    }
}

/// Run a function in parallel on CUDA.
template <typename func_t>
void ParallelForCUDA_(const Device& device, int64_t n, const func_t& func) {
    if (device.GetType() != Device::DeviceType::CUDA) {
        utility::LogError("ParallelFor for CUDA cannot run on device {}.",
                          device.ToString());
    }
    if (n == 0) {
        return;
    }

    CUDAScopedDevice scoped_device(device);
    int64_t items_per_block = CLOUDVIEWER_PARFOR_BLOCK * CLOUDVIEWER_PARFOR_THREAD;
    int64_t grid_size = (n + items_per_block - 1) / items_per_block;

    ElementWiseKernel_<CLOUDVIEWER_PARFOR_BLOCK, CLOUDVIEWER_PARFOR_THREAD>
            <<<grid_size, CLOUDVIEWER_PARFOR_BLOCK, 0, core::cuda::GetStream()>>>(
                    n, func);
    CLOUDVIEWER_GET_LAST_CUDA_ERROR("ParallelFor failed.");
}

#else

/// Run a function in parallel on CPU.
template <typename func_t>
void ParallelForCPU_(const Device& device, int64_t n, const func_t& func) {
    if (!device.IsCPU()) {
        utility::LogError("ParallelFor for CPU cannot run on device {}.",
                          device.ToString());
    }
    if (n == 0) {
        return;
    }

#pragma omp parallel for num_threads(utility::EstimateMaxThreads())
    for (int64_t i = 0; i < n; ++i) {
        func(i);
    }
}

#endif

/// Run a function in parallel on CPU or CUDA.
///
/// \param device The device for the parallel for loop to run on.
/// \param n The number of workloads.
/// \param func The function to be executed in parallel. The function should
/// take an int64_t workload index and returns void, i.e., `void func(int64_t)`.
///
/// \note This is optimized for uniform work items, i.e. where each call to \p
/// func takes the same time.
/// \note If you use a lambda function, capture only the required variables
/// instead of all to prevent accidental race conditions. If you want the
/// kernel to be used on both CPU and CUDA, capture the variables by value.
template <typename func_t>
void ParallelFor(const Device& device, int64_t n, const func_t& func) {
#ifdef __CUDACC__
    ParallelForCUDA_(device, n, func);
#else
    ParallelForCPU_(device, n, func);
#endif
}

/// Run a potentially vectorized function in parallel on CPU or CUDA.
///
/// \param device The device for the parallel for loop to run on.
/// \param n The number of workloads.
/// \param func The function to be executed in parallel. The function should
/// take an int64_t workload index and returns void, i.e., `void func(int64_t)`.
/// \param vec_func The vectorized function to be executed in parallel. The
/// function should be provided using the CLOUDVIEWER_VECTORIZED macro, e.g.,
/// `CLOUDVIEWER_VECTORIZED(MyISPCKernel, some_used_variable)`.
///
/// \note This is optimized for uniform work items, i.e. where each call to \p
/// func takes the same time.
/// \note If you use a lambda function, capture only the required variables
/// instead of all to prevent accidental race conditions. If you want the
/// kernel to be used on both CPU and CUDA, capture the variables by value.
///
/// Example:
///
/// \code
/// /* MyFile.cpp */
///
/// std::vector<float> v(1000);
/// float fill_value = 42.0f;
/// core::ParallelFor(
///         core::Device("CPU:0"),
///         v.size(),
///         [&](int64_t idx) { v[idx] = fill_value; },
///         CLOUDVIEWER_VECTORIZED(MyFillKernel, v.data(), fill_value));
///
/// static inline void MyFillFunction(int64_t idx,
///                                   float* uniform v,
///                                   uniform float fill_value) {
///     v[idx] = fill_value;
/// }
///
/// CLOUDVIEWER_EXPORT_VECTORIZED(MyFillKernel,
///                          MyFillFunction,
///                          float* uniform,
///                          uniform float)
/// \endcode
template <typename vec_func_t, typename func_t>
void ParallelFor(const Device& device,
                 int64_t n,
                 const func_t& func,
                 const vec_func_t& vec_func) {
#ifdef __CUDACC__
    ParallelForCUDA_(device, n, func);
#else
    ParallelForCPU_(device, n, func);
#endif
}

#ifdef BUILD_ISPC_MODULE

// Internal helper macro.
#define CLOUDVIEWER_CALL_ISPC_KERNEL_(ISPCKernel, start, end, ...)  \
    using namespace ispc;                                           \
    ISPCKernel(start, end, __VA_ARGS__);

#else

// Internal helper macro.
#define CLOUDVIEWER_CALL_ISPC_KERNEL_(ISPCKernel, start, end, ...)       \
    utility::LogError(                                                   \
            "ISPC module disabled. Unable to call vectorized kernel {}", \
            CLOUDVIEWER_STRINGIFY(ISPCKernel));

#endif

/// Internal helper macro.
#define CLOUDVIEWER_OVERLOADED_LAMBDA_(T, ISPCKernel, ...)                              \
    [&](T, int64_t start, int64_t end) {                                                \
        CLOUDVIEWER_CALL_ISPC_KERNEL_(                                                  \
                CLOUDVIEWER_CONCAT(ISPCKernel, CLOUDVIEWER_CONCAT(_, T)), start, end,   \
                __VA_ARGS__);                                                           \
    }

/// CLOUDVIEWER_VECTORIZED(ISPCKernel, ...)
///
/// Defines a lambda function to call the provided kernel.
///
/// Use the CLOUDVIEWER_EXPORT_TEMPLATE_VECTORIZED macro to define the
/// kernel in the ISPC source file.
///
/// Note: The arguments to the kernel only have to exist if ISPC support is
/// enabled via BUILD_ISPC_MODULE=ON.
#define CLOUDVIEWER_VECTORIZED(ISPCKernel, ...)                             \
    [&](int64_t start, int64_t end) {                                       \
        CLOUDVIEWER_CALL_ISPC_KERNEL_(ISPCKernel, start, end, __VA_ARGS__); \
    }

/// CLOUDVIEWER_TEMPLATE_VECTORIZED(T, ISPCKernel, ...)
///
/// Defines a lambda function to call the provided template-like kernel.
/// Supported types:
/// - bool
/// - unsigned + signed {8,16,32,64} bit integers,
/// - float, double
///
/// Use the CLOUDVIEWER_EXPORT_TEMPLATE_VECTORIZED macro to define the
/// kernel in the ISPC source file.
///
/// Note: The arguments to the kernel only have to exist if ISPC support is
/// enabled via BUILD_ISPC_MODULE=ON.
#define CLOUDVIEWER_TEMPLATE_VECTORIZED(T, ISPCKernel, ...)                        \
    [&](int64_t start, int64_t end) {                                         \
        static_assert(std::is_arithmetic<T>::value,                           \
                      "Data type is not an arithmetic type");                 \
        utility::Overload(                                                    \
                CLOUDVIEWER_OVERLOADED_LAMBDA_(bool, ISPCKernel, __VA_ARGS__),     \
                CLOUDVIEWER_OVERLOADED_LAMBDA_(uint8_t, ISPCKernel, __VA_ARGS__),  \
                CLOUDVIEWER_OVERLOADED_LAMBDA_(int8_t, ISPCKernel, __VA_ARGS__),   \
                CLOUDVIEWER_OVERLOADED_LAMBDA_(uint16_t, ISPCKernel, __VA_ARGS__), \
                CLOUDVIEWER_OVERLOADED_LAMBDA_(int16_t, ISPCKernel, __VA_ARGS__),  \
                CLOUDVIEWER_OVERLOADED_LAMBDA_(uint32_t, ISPCKernel, __VA_ARGS__), \
                CLOUDVIEWER_OVERLOADED_LAMBDA_(int32_t, ISPCKernel, __VA_ARGS__),  \
                CLOUDVIEWER_OVERLOADED_LAMBDA_(uint64_t, ISPCKernel, __VA_ARGS__), \
                CLOUDVIEWER_OVERLOADED_LAMBDA_(int64_t, ISPCKernel, __VA_ARGS__),  \
                CLOUDVIEWER_OVERLOADED_LAMBDA_(float, ISPCKernel, __VA_ARGS__),    \
                CLOUDVIEWER_OVERLOADED_LAMBDA_(double, ISPCKernel, __VA_ARGS__),   \
                [&](auto&& generic, int64_t start, int64_t end) {             \
                    utility::LogError(                                        \
                            "Unsupported data type {} for calling "           \
                            "vectorized kernel {}",                           \
                            typeid(generic).name(),                           \
                            CLOUDVIEWER_STRINGIFY(ISPCKernel));                    \
                })(T{}, start, end);                                          \
    }


}  // namespace core
}  // namespace cloudViewer
