// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "gpu_sync.hpp"

#include <cstdio>
#include <exception>
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
    return backend->vulkan_config.defer_sync;
#else
    (void)backend;
    return false;
#endif
}

void SyncBackendRaw(internal::Backend *backend) {
    if (backend == nullptr || backend->handle == nullptr) {
        return;
    }
    try {
        ggml_backend_synchronize(backend->handle);
#if defined(AICORE_CUDA_ALIKED)
        if (backend->IsCuda()) {
            cudaDeviceSynchronize();
        }
#endif
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[vk-aliked] synchronize: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr,
                     "[vk-aliked] synchronize failed with unknown exception\n");
    }
}

}  // namespace

void ApplyVulkanAlikedPerfDefaults() {
#if defined(AICORE_VULKAN_ALIKED)
    // Defaults are encoded at each use site. This compatibility entry point
    // intentionally has no process-environment side effect: contexts created
    // concurrently must not change one another's execution policy.
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
