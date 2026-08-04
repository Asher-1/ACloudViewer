// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "gpu_sync.hpp"

#include <cstdlib>
#include <string>

#if defined(AICORE_VULKAN_ALIKED)
#include "vulkan/vulkan_aliked_dispatch.hpp"
#endif

#if defined(AICORE_CUDA_ALIKED)
#include <cuda_runtime.h>
#endif

#include <ggml-alloc.h>
#include <ggml-backend.h>

namespace lightglue::aliked_internal {
namespace {

bool VulkanDeferGpuSync(internal::Backend *backend) {
#if defined(AICORE_VULKAN_ALIKED)
    if (backend == nullptr || !backend->IsVulkan()) {
        return false;
    }
    const char *env = std::getenv("LIGHTGLUE_ALIKED_VULKAN_DEFER_SYNC");
    if (env != nullptr) {
        return env[0] != '0';
    }
    return true;
#else
    (void)backend;
    return false;
#endif
}

void SyncBackendRaw(internal::Backend *backend) {
    if (backend == nullptr || backend->handle == nullptr) {
        return;
    }
    ggml_backend_synchronize(backend->handle);
#if defined(AICORE_CUDA_ALIKED)
    if (backend->IsCuda()) {
        cudaDeviceSynchronize();
    }
#endif
}

}  // namespace

void ApplyVulkanAlikedPerfDefaults() {
#if defined(AICORE_VULKAN_ALIKED)
    // Parity-gated full GPU path — no manual env required for GUI / C API
    // users. Opt-out any flag with =0 (e.g. LIGHTGLUE_ALIKED_VULKAN_COMPUTE=0).
    if (std::getenv("LIGHTGLUE_ALIKED_VULKAN_COMPUTE") == nullptr) {
        setenv("LIGHTGLUE_ALIKED_VULKAN_COMPUTE", "1", 0);
    }
    if (std::getenv("LIGHTGLUE_ALIKED_VULKAN_SDDH") == nullptr) {
        // The scalar Vulkan SDDH shader is slower than the parity-preserving
        // OpenMP fallback and can stall repeated large dispatches.
        setenv("LIGHTGLUE_ALIKED_VULKAN_SDDH", "0", 0);
    }
    if (std::getenv("LIGHTGLUE_ALIKED_VULKAN_GPU_UPSAMPLE") == nullptr) {
        // The custom path remains opt-in until its layout parity gate passes.
        setenv("LIGHTGLUE_ALIKED_VULKAN_GPU_UPSAMPLE", "0", 0);
    }
    if (std::getenv("LIGHTGLUE_ALIKED_VULKAN_SCHED") == nullptr) {
        // Legacy sched path paused; 0015 fence/buffer pin is the parity
        // baseline.
        setenv("LIGHTGLUE_ALIKED_VULKAN_SCHED", "0", 0);
    }
    if (std::getenv("LIGHTGLUE_ALIKED_VULKAN_DEFER_SYNC") == nullptr) {
        // Correctness first: defer-sync caused score/descriptor readback races
        // on same-ctx multi-extract. Opt-in DEFER_SYNC=1 for experiments.
        setenv("LIGHTGLUE_ALIKED_VULKAN_DEFER_SYNC", "0", 0);
    }
    if (std::getenv("LIGHTGLUE_ALIKED_VULKAN_SDDH_SINGLE") == nullptr &&
        std::getenv("LIGHTGLUE_ALIKED_VULKAN_SDDH_CHUNK") == nullptr) {
        // Settings for explicit LIGHTGLUE_ALIKED_VULKAN_SDDH=1 experiments.
        setenv("LIGHTGLUE_ALIKED_VULKAN_SDDH_SINGLE", "1", 0);
        setenv("LIGHTGLUE_ALIKED_VULKAN_SDDH_CHUNK", "16", 0);
    }
    if (std::getenv("LIGHTGLUE_ALIKED_VULKAN_POST") == nullptr) {
        // The custom Vulkan DKD path does not yet satisfy the strict CPU/CUDA
        // parity gate. The CPU postprocess keeps the Vulkan CNN output exact
        // while remaining below the end-to-end latency target.
        setenv("LIGHTGLUE_ALIKED_VULKAN_POST", "0", 0);
    }
#endif
}

void SyncGpuPipeline(internal::Backend *backend) {
    if (VulkanDeferGpuSync(backend)) {
        return;
    }
    SyncBackendRaw(backend);
}

void FlushGpuPipeline(internal::Backend *backend) { SyncBackendRaw(backend); }

void BarrierGpuPipeline(internal::Backend *backend) {
#if defined(AICORE_VULKAN_ALIKED)
    if (backend != nullptr && backend->IsVulkan()) {
        VkAlikedQueueIdle(backend->handle);
        FlushGpuPipeline(backend);
        return;
    }
#endif
    SyncGpuPipeline(backend);
}

bool GallocrComputeGraph(internal::Backend *backend,
                         ggml_cgraph *graph,
                         std::string *error,
                         ggml_gallocr_t graph_gallocr) {
    if (backend == nullptr || backend->handle == nullptr || graph == nullptr) {
        if (error) {
            *error = "invalid backend or graph for gallocr compute";
        }
        return false;
    }
    ggml_gallocr_t gallocr =
            graph_gallocr != nullptr ? graph_gallocr : backend->allocator;
    if (gallocr == nullptr) {
        if (error) {
            *error = "ggml graph allocator is null";
        }
        return false;
    }
    if (graph_gallocr == nullptr) {
        if (!ggml_gallocr_reserve(gallocr, graph)) {
            if (error) {
                *error = "failed to reserve ggml graph allocator";
            }
            return false;
        }
    }
    if (!ggml_gallocr_alloc_graph(gallocr, graph)) {
        if (error) {
            *error = "failed to bind ggml graph allocator";
        }
        return false;
    }
    if (ggml_backend_graph_compute(backend->handle, graph) !=
        GGML_STATUS_SUCCESS) {
        if (error) {
            *error = "ggml graph compute failed";
        }
        return false;
    }
    return true;
}

bool RunCachedGraphCompute(internal::Backend *backend,
                           ggml_cgraph *graph,
                           std::string *error,
                           ggml_gallocr_t graph_gallocr) {
    if (backend == nullptr || backend->handle == nullptr || graph == nullptr) {
        if (error) {
            *error = "invalid backend or graph for cached compute";
        }
        return false;
    }
    ggml_gallocr_t gallocr =
            graph_gallocr != nullptr ? graph_gallocr : backend->allocator;
    if (gallocr != nullptr && !ggml_gallocr_alloc_graph(gallocr, graph)) {
        if (error) {
            *error = "failed to bind cached ggml graph allocator";
        }
        return false;
    }
    if (ggml_backend_graph_compute(backend->handle, graph) !=
        GGML_STATUS_SUCCESS) {
        if (error) {
            *error = "cached ggml graph compute failed";
        }
        return false;
    }
    SyncGpuPipeline(backend);
    return true;
}

}  // namespace lightglue::aliked_internal
