// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/aliked/backend.h"

#include <ggml-backend.h>

#include "common/ggml_backend_utils.hpp"
#include "tasks/aliked/aliked_common.hpp"

#if defined(AICORE_VULKAN_ALIKED)
#include "tasks/aliked/gpu_sync.hpp"
#include "tasks/aliked/vulkan/vulkan_aliked_dispatch.hpp"

#endif

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

namespace lightglue::internal {
namespace {

std::string Lower(std::string value) {
    for (char &c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

VulkanAlikedConfig SnapshotVulkanConfig() {
    VulkanAlikedConfig config;
    config.initialized = true;
    // The custom Vulkan ALIKED kernels are not parity-qualified yet.  The
    // regular ggml graph still executes on Vulkan; this only selects the
    // exact CPU bridge for ALIKED's custom operators. No process-environment
    // variable may re-enable an unqualified path in a GUI session; the
    // historical LIGHTGLUE_ALIKED_VULKAN_* escape hatches are removed (the
    // future batch API will opt in through Session config).
    config.compute = false;
    // The current custom command-buffer implementation for these two stages
    // has not completed its lifecycle/parity qualification.
    config.gpu_upsample = false;
    // Per-stage DCN values alone are insufficient: they still destabilize
    // cross-image LightGlue matching. Keep the exact CPU bridge until the
    // full extractor-and-matcher parity suite qualifies this path.
    config.dcn = false;
    config.postprocess = false;
    config.sddh = false;
    config.defer_sync = false;
    config.scheduler = false;
    config.scheduler_tail_only = false;
    // Rebuilding the Vulkan backend on every extract tears down and recreates
    // the vk device, which (a) re-triggers NVIDIA TDR/device-lost on concurrent
    // submits and (b) re-runs the full GPU warmup, dominating end-to-end
    // latency. The backend is session-persistent.
    config.fresh_extract = false;
    // The ggml-vulkan conv2d path on NVIDIA CoopMat2 devices is NOT safe for
    // ALIKED's precision-sensitive convs: the COOPMAT2 path hard-codes an fp16
    // accumulator (corrupts F32 conv score maps), and routing F32 convs through
    // the scalar _unroll pipeline is non-deterministic (spurious correct/wrong
    // outputs across runs) because the fp32 shmem layout collides with the
    // fp16-tuned spec constants. CPU bridge is the only parity-qualified path
    // (stable 0.001px / cos=1.0 for f32 & f16).
    config.force_cpu_conv = true;
    return config;
}

}  // namespace

bool Backend::Init(const std::string &request, int num_threads) {
    Release();
    const int threads =
            num_threads > 0
                    ? num_threads
                    : static_cast<int>(ggml_common::default_cpu_threads());
    lease = aicore::runtime::acquire_backend_lease(
            request.empty() ? "auto" : request, threads, &error);
    if (!lease) return false;
    handle = lease.handle();
    device = lease.device();
    // Rewarming a Vulkan backend belongs to the same Session and preserves
    // its original policy even when another component changes its env.
    if (IsVulkan() && !vulkan_config.initialized) {
        vulkan_config = SnapshotVulkanConfig();
    }

    {
        const auto backend_lock = Lock();
        allocator =
                ggml_gallocr_new(ggml_backend_get_default_buffer_type(handle));
    }
    if (allocator == nullptr) {
        error = "failed to create the ggml graph allocator";
        Release();
        return false;
    }

    if (IsVulkan() && vulkan_config.scheduler) {
        cpu_lease =
                aicore::runtime::acquire_backend_lease("cpu", threads, &error);
        cpu_backend = cpu_lease.handle();
        if (cpu_backend == nullptr) {
            error = "failed to initialize CPU backend for Vulkan scheduler";
            Release();
            return false;
        }
        const auto scheduler_lock = Lock();
        std::vector<ggml_backend_t> backs = {handle, cpu_backend};
        std::vector<ggml_backend_buffer_type_t> bufts = {
                ggml_backend_get_default_buffer_type(handle),
                ggml_backend_get_default_buffer_type(cpu_backend),
        };
        sched = ggml_backend_sched_new(backs.data(), bufts.data(),
                                       static_cast<int>(backs.size()),
                                       /*graph_size=*/512, /*parallel=*/false,
                                       /*op_offload=*/false);
        if (sched == nullptr) {
            error = "failed to create ggml backend scheduler for Vulkan";
            Release();
            return false;
        }
        use_sched = true;
    }
    return true;
}

bool Backend::SchedRunGraph(ggml_cgraph *graph,
                            const std::function<void()> &set_inputs,
                            std::string *error,
                            const std::function<void()> &before_alloc) {
    if (!HasSched() || graph == nullptr) {
        if (error) {
            *error = "Vulkan scheduler is not initialized";
        }
        return false;
    }
    const auto backend_lock = Lock();
    ggml_backend_sched_reset(sched);
    if (before_alloc) {
        before_alloc();
    }
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        if (error) {
            *error = "ggml_backend_sched_alloc_graph failed";
        }
        return false;
    }
    if (set_inputs) {
        set_inputs();
    }
    if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
        if (error) {
            *error = "ggml_backend_sched_graph_compute failed";
        }
        return false;
    }
    ggml_backend_sched_synchronize(sched);
#if defined(AICORE_VULKAN_ALIKED)
    if (IsVulkan()) {
        lightglue::aliked_internal::VkAlikedQueueIdle(handle);
    }
#endif
    return true;
}

bool Backend::IsCpu() const {
    return handle != nullptr &&
           ggml_backend_dev_type(ggml_backend_get_device(handle)) ==
                   GGML_BACKEND_DEVICE_TYPE_CPU;
}

bool Backend::IsGpu() const {
    return handle != nullptr &&
           ggml_backend_dev_type(ggml_backend_get_device(handle)) ==
                   GGML_BACKEND_DEVICE_TYPE_GPU;
}

bool Backend::IsCuda() const {
    if (!IsGpu() || handle == nullptr) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(handle);
    const char *registry =
            ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
    return registry != nullptr && Lower(registry) == "cuda";
}

bool Backend::IsVulkan() const {
    if (!IsGpu() || handle == nullptr) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(handle);
    const char *registry =
            ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
    return registry != nullptr && Lower(registry) == "vulkan";
}

aicore::runtime::BackendLeaseLock Backend::Lock() const {
    std::vector<aicore::runtime::BackendLease> leases;
    if (lease) leases.push_back(lease);
    if (cpu_lease) leases.push_back(cpu_lease);
    return aicore::runtime::lock_backend_leases(leases);
}

void Backend::Release() {
    const auto backend_lock = Lock();
#if defined(AICORE_VULKAN_ALIKED)
    if (IsVulkan() && handle != nullptr) {
        try {
            lightglue::aliked_internal::VkAlikedQueueIdle(handle);
            ggml_backend_synchronize(handle);
        } catch (const std::exception &e) {
            // A lost device cannot be recovered in-place. Destruction must
            // still release host resources without terminating the GUI.
            ALIKED_LOG_ERR("backend release: %s", e.what());
        } catch (...) {
            ALIKED_LOG_ERR("backend release failed with unknown exception");
        }
    }
#endif
    if (sched != nullptr) {
        ggml_backend_sched_free(sched);
        sched = nullptr;
    }
    if (allocator != nullptr) {
        ggml_gallocr_free(allocator);
        allocator = nullptr;
    }
    cpu_backend = nullptr;
    handle = nullptr;
    cpu_lease.reset();
    lease.reset();
    use_sched = false;
    device.clear();
    error.clear();
}

}  // namespace lightglue::internal
