// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
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
