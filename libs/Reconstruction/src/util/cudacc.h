// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLMAP_SRC_UTIL_CUDACC_H_
#define COLMAP_SRC_UTIL_CUDACC_H_

#include <cuda_runtime.h>

#include <string>

#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define CUDA_CHECK() CudaCheck(__FILE__, __LINE__)
#define CUDA_SYNC_AND_CHECK() CudaSyncAndCheck(__FILE__, __LINE__)

namespace colmap {

class CudaTimer {
public:
    CudaTimer();
    ~CudaTimer();

    void Print(const std::string& message);

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
    float elapsed_time_;
};

void CudaSafeCall(const cudaError_t error,
                  const std::string& file,
                  const int line);

void CudaCheck(const char* file, const int line);
void CudaSyncAndCheck(const char* file, const int line);

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_CUDACC_H_
