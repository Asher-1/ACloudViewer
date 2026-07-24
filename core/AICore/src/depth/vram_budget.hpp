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

// Cap preprocess long-edge for a *single-view* GIANT nested graph (backbone +
// DPT head). Uses free VRAM, not view count. Returns `requested` unchanged on
// CPU or when VRAM query is unavailable.
// `device_name` is used to apply backend-specific limits (e.g. Metal's slow
// conv_transpose_2d caps more aggressively than CUDA).
int cap_resize_target_for_vram(int requested,
                               bool nested_metric,
                               const GpuMemoryInfo& mem,
                               const std::string& device_name = "");

} // namespace depth
} // namespace aicore
