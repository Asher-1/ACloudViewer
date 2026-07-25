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

    const bool is_metal = is_metal_device(device_name);

    // Metal's conv_transpose_2d is significantly slower than CUDA/Vulkan.
    // When VRAM info is unavailable, fall back to a conservative hard cap.
    // When VRAM info IS available, use dynamic calculation with a Metal-
    // specific overhead factor instead.
    constexpr int kMetalFallbackMax = 1008;

    if (!mem.valid) {
        if (is_metal && requested > kMetalFallbackMax) {
            DA_DEBUG_LOG(
                    "Metal fallback cap (no VRAM info): img_resize_target "
                    "%d -> %d",
                    requested, kMetalFallbackMax);
            return kMetalFallbackMax;
        }
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
    // Metal conv_transpose_2d uses ~1.8x more intermediate memory than CUDA.
    constexpr double kPeakAt504_default = 9.0e8;
    constexpr double kMetalOverhead = 1.8;
    const double peak_at_504 =
            is_metal ? kPeakAt504_default * kMetalOverhead : kPeakAt504_default;
    const double ratio = std::sqrt(std::max(0.0, usable / peak_at_504));
    int cap = static_cast<int>(504.0 * ratio);
    cap = std::max(kMinTarget, (cap / kPatchSize) * kPatchSize);
    cap = std::min(cap, requested);

    if (cap < requested) {
        DA_DEBUG_LOG(
                "VRAM cap%s: img_resize_target %d -> %d (GPU free %.1f GiB / "
                "total %.1f GiB, single-view peak)",
                is_metal ? " (Metal)" : "", requested, cap,
                mem.free_bytes / (1024.0 * 1024.0 * 1024.0),
                mem.total_bytes / (1024.0 * 1024.0 * 1024.0));
    }
    return cap;
}

}  // namespace depth
}  // namespace aicore
