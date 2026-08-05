// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ggml-backend.h>

#include <functional>
#include <string>

#include "ggml_backend_registry.hpp"

struct ggml_cgraph;

namespace lightglue::internal {

// Vulkan execution policy is sampled when an ALIKED session is created.  It
// must never be read from or written to the process environment during graph
// execution: multiple contexts can then coexist with deterministic policies.
struct VulkanAlikedConfig {
    bool initialized = false;
    bool compute = true;
    bool gpu_upsample = false;
    bool dcn = true;
    bool postprocess = false;
    bool sddh = false;
    bool defer_sync = false;
    bool scheduler = false;
    bool scheduler_tail_only = false;
    bool fresh_extract = true;
    bool force_cpu_conv = false;
};

struct Backend {
    aicore::runtime::BackendLease lease;
    aicore::runtime::BackendLease cpu_lease;
    // Non-owning aliases retained for the existing ALIKED ggml/Vulkan code.
    ggml_backend_t handle = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_gallocr_t allocator = nullptr;
    bool use_sched = false;
    VulkanAlikedConfig vulkan_config;
    std::string device;
    std::string error;

    bool Init(const std::string &request, int num_threads);
    bool IsCpu() const;
    bool IsGpu() const;
    bool IsCuda() const;
    bool IsVulkan() const;
    aicore::runtime::BackendLeaseLock Lock() const;
    bool HasSched() const { return use_sched && sched != nullptr; }
    // Vulkan [gpu, cpu] scheduler: reset → optional pin → alloc → inputs →
    // compute → sync.
    bool SchedRunGraph(ggml_cgraph *graph,
                       const std::function<void()> &set_inputs,
                       std::string *error,
                       const std::function<void()> &before_alloc = {});
    void Release();
    ~Backend() { Release(); }
};

}  // namespace lightglue::internal
