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

struct ggml_cgraph;

namespace lightglue::internal {

struct Backend {
    ggml_backend_t handle = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_gallocr_t allocator = nullptr;
    bool use_sched = false;
    std::string device;
    std::string error;

    bool Init(const std::string &request, int num_threads);
    bool IsCpu() const;
    bool IsGpu() const;
    bool IsCuda() const;
    bool IsVulkan() const;
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
