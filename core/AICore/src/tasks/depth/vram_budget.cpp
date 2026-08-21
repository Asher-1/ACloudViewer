// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/depth/vram_budget.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "ggml-backend.h"
#include "tasks/depth/backend.hpp"
#include "tasks/depth/common.hpp"

namespace aicore {
namespace depth {

GpuMemoryInfo query_gpu_memory(const Backend& be) {
    GpuMemoryInfo info;
    if (!be.is_offloading()) {
        return info;
    }
    ggml_backend_t backend = be.handle();
    if (!backend) {
        return info;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev) {
        return info;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
    if (total_bytes == 0) {
        return info;
    }
    info.free_bytes = free_bytes;
    info.total_bytes = total_bytes;
    info.valid = true;
    return info;
}

int cap_resize_target_for_vram(int requested,
                               bool nested_metric,
                               const GpuMemoryInfo& mem,
                               const std::string& device_name) {
    (void)nested_metric;
    (void)mem;
    (void)device_name;
    return requested;
}

}  // namespace depth
}  // namespace aicore
