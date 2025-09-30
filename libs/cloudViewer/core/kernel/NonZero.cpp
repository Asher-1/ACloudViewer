// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/core/kernel/NonZero.h"

#include "cloudViewer/core/Device.h"
#include "cloudViewer/core/Tensor.h"
#include <Logging.h>

namespace cloudViewer {
namespace core {
namespace kernel {

Tensor NonZero(const Tensor& src) {
    if (src.IsCPU()) {
        return NonZeroCPU(src);
    } else if (src.IsSYCL()) {
#ifdef BUILD_SYCL_MODULE
        return NonZeroSYCL(src);
#else
        utility::LogError("Not compiled with SYCL, but SYCL device is used.");
#endif
    } else if (src.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        return NonZeroCUDA(src);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("NonZero: Unimplemented device");
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace cloudViewer
