// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/// \file Helper.h
/// \brief Helper functions for the ml ops

#pragma once

#ifdef BUILD_CUDA_MODULE

#include <cuda.h>
#include <cuda_runtime.h>

#include "cloudViewer/core/CUDAUtils.h"
#include <Logging.h>

#endif  // #ifdef BUILD_CUDA_MODULE

namespace cloudViewer {
namespace ml {

#ifdef BUILD_CUDA_MODULE

#define CLOUDVIEWER_ML_CUDA_DRIVER_CHECK(err) \
    __CLOUDVIEWER_ML_CUDA_DRIVER_CHECK(err, __FILE__, __LINE__)

inline void __CLOUDVIEWER_ML_CUDA_DRIVER_CHECK(CUresult err,
                                          const char *file,
                                          const int line,
                                          bool abort = true) {
    if (err != CUDA_SUCCESS) {
        const char *error_string;
        CUresult err_get_string = cuGetErrorString(err, &error_string);

        if (err_get_string == CUDA_SUCCESS) {
            utility::LogError("{}:{} CUDA driver error: {}", file, line,
                              error_string);
        } else {
            utility::LogError("{}:{} CUDA driver error: UNKNOWN", file, line);
        }
    }
}

inline cudaStream_t GetDefaultStream() { return (cudaStream_t)0; }

inline int GetDevice(cudaStream_t stream) {
    if (stream == GetDefaultStream()) {
        // Default device.
        return 0;
    }

    // Remember current context.
    CUcontext current_context;
    CLOUDVIEWER_ML_CUDA_DRIVER_CHECK(cuCtxGetCurrent(&current_context));

    // Switch to context of provided stream.
    CUcontext context;
    CLOUDVIEWER_ML_CUDA_DRIVER_CHECK(cuStreamGetCtx(stream, &context));
    CLOUDVIEWER_ML_CUDA_DRIVER_CHECK(cuCtxSetCurrent(context));

    // Query device of current context.
    // This is the device of the provided stream.
    CUdevice device;
    CLOUDVIEWER_ML_CUDA_DRIVER_CHECK(cuCtxGetDevice(&device));

    // Restore previous context.
    CLOUDVIEWER_ML_CUDA_DRIVER_CHECK(cuCtxSetCurrent(current_context));

    // CUdevice is a typedef to int.
    return device;
}

class CUDAScopedDeviceStream {
public:
    explicit CUDAScopedDeviceStream(cudaStream_t stream)
        : scoped_device_(GetDevice(stream)), scoped_stream_(stream) {}

    CUDAScopedDeviceStream(CUDAScopedDeviceStream const &) = delete;
    void operator=(CUDAScopedDeviceStream const &) = delete;

private:
    core::CUDAScopedDevice scoped_device_;
    core::CUDAScopedStream scoped_stream_;
};
#endif

}  // namespace ml
}  // namespace cloudViewer
