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

#include <memory>
#include <string>

#include "core/Device.h"
#include "core/Dtype.h"
#include "core/MemoryManager.h"
#include "core/linalg/LinalgHeadersCPU.h"
#include "core/linalg/LinalgHeadersCUDA.h"
#include <Logging.h>

namespace cloudViewer {
namespace core {


#define DISPATCH_LINALG_DTYPE_TO_TEMPLATE(DTYPE, ...)    \
    [&] {                                                \
        if (DTYPE == cloudViewer::core::Float32) {            \
            using scalar_t = float;                      \
            return __VA_ARGS__();                        \
        } else if (DTYPE == cloudViewer::core::Float64) {     \
            using scalar_t = double;                     \
            return __VA_ARGS__();                        \
        } else {                                         \
            utility::LogError("Unsupported data type."); \
        }                                                \
    }()

inline void CLOUDVIEWER_LAPACK_CHECK(CLOUDVIEWER_CPU_LINALG_INT info,
                                const std::string& msg) {
    if (info < 0) {
        utility::LogError("{}: {}-th parameter is invalid.", msg, -info);
    } else if (info > 0) {
        utility::LogError("{}: singular condition detected.", msg);
    }
}

#ifdef BUILD_CUDA_MODULE
inline void CLOUDVIEWER_CUBLAS_CHECK(cublasStatus_t status, const std::string& msg) {
    if (CUBLAS_STATUS_SUCCESS != status) {
        utility::LogError("{}", msg);
    }
}

inline void CLOUDVIEWER_CUSOLVER_CHECK(cusolverStatus_t status,
                                  const std::string& msg) {
    if (CUSOLVER_STATUS_SUCCESS != status) {
        utility::LogError("{}", msg);
    }
}

inline void CLOUDVIEWER_CUSOLVER_CHECK_WITH_DINFO(cusolverStatus_t status,
                                             const std::string& msg,
                                             int* dinfo,
                                             const Device& device) {
    int hinfo;
    MemoryManager::MemcpyToHost(&hinfo, dinfo, device, sizeof(int));
    if (status != CUSOLVER_STATUS_SUCCESS || hinfo != 0) {
        if (hinfo < 0) {
            utility::LogError("{}: {}-th parameter is invalid.", msg, -hinfo);
        } else if (hinfo > 0) {
            utility::LogError("{}: singular condition detected.", msg);
        } else {
            utility::LogError("{}: status error code = {}.", msg, status);
        }
    }
}

class CuSolverContext {
public:
    static CuSolverContext& GetInstance();

    CuSolverContext(const CuSolverContext&) = delete;
    CuSolverContext& operator=(const CuSolverContext&) = delete;
    ~CuSolverContext();

    cusolverDnHandle_t& GetHandle(const Device& device);

private:
    CuSolverContext();
    std::unordered_map<Device, cusolverDnHandle_t> map_device_to_handle_;
};

class CuBLASContext {
public:
    static CuBLASContext& GetInstance();

    CuBLASContext(const CuBLASContext&) = delete;
    CuBLASContext& operator=(const CuBLASContext&) = delete;
    ~CuBLASContext();

    cublasHandle_t& GetHandle(const Device& device);

private:
    CuBLASContext();
    std::unordered_map<Device, cublasHandle_t> map_device_to_handle_;
};
#endif
}  // namespace core
}  // namespace cloudViewer
