// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "vram_budget.hpp"

#include <algorithm>
#include <cmath>

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

int cap_resize_target_for_vram(int requested,
                               bool nested_metric,
                               const GpuMemoryInfo& mem) {
    constexpr int kPatchSize = 14;
    constexpr int kMinTarget = 504;

    if (requested <= 0) {
        return requested;
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
