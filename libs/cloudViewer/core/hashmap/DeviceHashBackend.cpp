// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/core/hashmap/DeviceHashBackend.h"

#include "cloudViewer/core/hashmap/HashMap.h"
#include <Helper.h>
#include <Logging.h>

namespace cloudViewer {
namespace core {

std::shared_ptr<DeviceHashBackend> CreateDeviceHashBackend(
        int64_t init_capacity,
        const Dtype& key_dtype,
        const SizeVector& key_element_shape,
        const std::vector<Dtype>& value_dtypes,
        const std::vector<SizeVector>& value_element_shapes,
        const Device& device,
        const HashBackendType& backend) {
    if (device.IsCPU()) {
        return CreateCPUHashBackend(init_capacity, key_dtype, key_element_shape,
                                    value_dtypes, value_element_shapes, device,
                                    backend);
    }
#if defined(BUILD_CUDA_MODULE)
    else if (device.IsCUDA()) {
        return CreateCUDAHashBackend(init_capacity, key_dtype,
                                     key_element_shape, value_dtypes,
                                     value_element_shapes, device, backend);
    }
#endif
    else {
        utility::LogError("Unimplemented device");
    }
}

}  // namespace core
}  // namespace cloudViewer
