// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <Optional.h>
#include "core/CUDAUtils.h"
#include "pybind/core/core.h"

namespace cloudViewer {
namespace core {

void pybind_cuda_utils(py::module& m) {
    py::module m_cuda = m.def_submodule("cuda");

    m_cuda.def("device_count", cuda::DeviceCount,
               "Returns the number of available CUDA devices. Returns 0 if "
               "CloudViewer is not compiled with CUDA support.");
    m_cuda.def("is_available", cuda::IsAvailable,
               "Returns true if CloudViewer is compiled with CUDA support and at "
               "least one compatible CUDA device is detected.");
    m_cuda.def("release_cache", cuda::ReleaseCache,
               "Releases CUDA memory manager cache. This is typically used for "
               "debugging.");
    m_cuda.def(
            "synchronize",
            [](const utility::optional<Device>& device) {
                if (device.has_value()) {
                    cuda::Synchronize(device.value());
                } else {
                    cuda::Synchronize();
                }
            },
            "Synchronizes CUDA devices. If no device is specified, all CUDA "
            "devices will be synchronized. No effect if the specified device "
            "is not a CUDA device. No effect if CloudViewer is not compiled with "
            "CUDA support."
            "device"_a = py::none());
}

}  // namespace core
}  // namespace cloudViewer
