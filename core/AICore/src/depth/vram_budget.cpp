// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "vram_budget.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "backend.hpp"
#include "common.hpp"
#include "ggml-backend.h"

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

namespace {

bool is_metal_device(const std::string& name) {
    if (name.empty()) return false;
    for (size_t i = 0; i + 2 < name.size(); ++i) {
        char c0 = (char)std::tolower((unsigned char)name[i]);
        char c1 = (char)std::tolower((unsigned char)name[i + 1]);
        char c2 = (char)std::tolower((unsigned char)name[i + 2]);
        if (c0 == 'm' && c1 == 't' && c2 == 'l') return true;
    }
    if (name.find("etal") != std::string::npos) return true;
    if (name.find("Apple") != std::string::npos) return true;
    return false;
}

}  // namespace

int cap_resize_target_for_vram(int requested,
                               bool nested_metric,
                               const GpuMemoryInfo& mem,
                               const std::string& device_name) {
    constexpr int kPatchSize = 14;
    constexpr int kMinTarget = 504;

    if (requested <= 0) {
        return requested;
    }

    // Metal's conv_transpose_2d kernel is orders of magnitude slower than CUDA;
    // large DPT head activations can hang the GPU (macOS command buffer
    // timeout). Cap to 1008 (72 ViT patches per side) which is empirically safe
    // on M-series.
    constexpr int kMetalMaxTarget = 1008;
    if (is_metal_device(device_name) && requested > kMetalMaxTarget) {
        DA_DEBUG_LOG(
                "Metal perf cap: img_resize_target %d -> %d "
                "(conv_transpose_2d slow on Metal; reduce to avoid GPU "
                "timeout)",
                requested, kMetalMaxTarget);
        requested = kMetalMaxTarget;
    }

    if (!mem.valid) {
        return requested;
    }

    // Reserve model weights + driver/display headroom (single-view sequential).
    const size_t weight_reserve =
            nested_metric ? (size_t(3) << 30)   // ~3 GiB nested anyview+metric
                          : (size_t(2) << 30);  // ~2 GiB single GIANT
    const size_t safety = size_t(768) << 20;    // 768 MiB
    if (mem.free_bytes <= weight_reserve + safety) {
        return std::min(requested, kMinTarget);
    }

    const double usable =
            static_cast<double>(mem.free_bytes - weight_reserve - safety);
    // Empirical: GIANT q8 single-view activation peak ~9e8 bytes at target=504
    // on RTX 3060-class GPUs (scales ~ (target/504)^2 ).
    constexpr double kPeakAt504 = 9.0e8;
    const double ratio = std::sqrt(std::max(0.0, usable / kPeakAt504));
    int cap = static_cast<int>(504.0 * ratio);
    cap = std::max(kMinTarget, (cap / kPatchSize) * kPatchSize);
    cap = std::min(cap, requested);

    if (cap < requested) {
        DA_DEBUG_LOG(
                "VRAM cap: img_resize_target %d -> %d (GPU free %.1f GiB / "
                "total %.1f GiB, single-view peak)",
                requested, cap, mem.free_bytes / (1024.0 * 1024.0 * 1024.0),
                mem.total_bytes / (1024.0 * 1024.0 * 1024.0));
    }
    return cap;
}

}  // namespace depth
}  // namespace aicore
