// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// MapAnything ggml runtime - backend facade over the shared AICore lease
// registry (template: tasks/gkd/gkd_backend.cpp).
//
// The upstream cpp_ggml graph computes the WHOLE graph on one backend
// (verified gate matrix: f32/f16/q8_0/q5_K x CPU/CUDA/Vulkan), so unlike the
// gkd scheduler bundle this facade keeps the upstream single-backend shape:
// the lease supplies the physical handle, the graph allocates weights and
// computes on that handle, and the session holds lease.lock() around every
// graph pass (the lease registry serializes access to shared physical
// devices process-wide).
#pragma once

#include <string>

#include "common/ggml_backend_registry.hpp"

struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;

namespace map_anything {

/* Compute backend for one session. `handle` points into process-shared
 * BackendLease state and stays valid while the lease is held. `name` is
 * exactly "CPU" for the CPU backend (the graph builders test
 * `be.name == "CPU"` for F16-weight staging decisions) and the resolved
 * device display name otherwise. */
struct Backend {
    aicore::runtime::BackendLease lease;
    ggml_backend_t handle = nullptr;
    std::string name = "CPU";
    int n_threads = 1;

    bool valid() const { return handle != nullptr; }
};

/* Initialize the compute backend for device_request ("auto" | "cpu" |
 * "cuda" | "vulkan" | ...). "auto" follows the platform order (CUDA ->
 * Vulkan -> CPU on Linux/Windows, Metal -> CPU on macOS). On failure
 * returns an invalid Backend (handle == nullptr) and fills *error. */
Backend init_backend(const std::string& device_request,
                     int n_threads,
                     std::string* error);

/* Release a Backend. Safe to call on a default-constructed struct. */
void free_backend(Backend& be);

}  // namespace map_anything
