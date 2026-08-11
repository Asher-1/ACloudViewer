// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstddef>
#include <string>

namespace aicore {
namespace depth {

class Backend;

struct GpuMemoryInfo {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    bool valid = false;
};

GpuMemoryInfo query_gpu_memory(const Backend& be);

// Preserve the caller-selected preprocessing long edge. Backend-specific
// implicit caps change model inputs and therefore invalidate CPU/GPU parity.
// A device that cannot allocate the requested graph must report an allocation
// failure; callers can then explicitly select a smaller task resolution.
int cap_resize_target_for_vram(int requested,
                               bool nested_metric,
                               const GpuMemoryInfo& mem,
                               const std::string& device_name = "");

} // namespace depth
} // namespace aicore
