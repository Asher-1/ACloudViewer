// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "BevRemapCudaProbe.h"

#ifdef MCALIB_BEV_CUDA_ENABLED

#include <cuda_runtime.h>
#ifdef _MSC_VER
#include <excpt.h>
#endif

namespace mcalib {
namespace bev_cuda {

bool probeAvailable() {
#ifdef _MSC_VER
    // Windows: the CUDA runtime DLL is delay-loaded (/DELAYLOAD), so calling
    // into it without a CUDA toolkit raises a SEH exception; __try/__except
    // keeps this probe in a C-only function (no C++ objects that need
    // unwinding), matching qSIBR's sibrCudaRuntimeAvailable().
    __try {
        int deviceCount = 0;
        return cudaGetDeviceCount(&deviceCount) == cudaSuccess &&
               deviceCount > 0;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
#else
    // Linux/macOS: the CUDA runtime is statically linked into the plugin, so
    // the call simply returns an error code when no NVIDIA driver / GPU is
    // present (libcuda.so.1 is dlopen'd by cudart itself only on first use).
    int deviceCount = 0;
    return cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0;
#endif
}

}  // namespace bev_cuda
}  // namespace mcalib

#endif  // MCALIB_BEV_CUDA_ENABLED
