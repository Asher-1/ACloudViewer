// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// AICore adaptation of the GKDT runtime backend bundle (upstream
// cpp_ggml/src/backend.* used compile-time GKD_USE_CUDA/VULKAN/METAL macros
// and owned its backends). Here the device is resolved through the AICore
// runtime: "auto" follows the platform order (CUDA -> Vulkan -> CPU on
// Linux/Windows, Metal -> CPU on macOS) and every physical backend handle is
// shared process-wide through the lease registry, matching every other task.
#pragma once

#include <string>

#include "common/ggml_backend_registry.hpp"

#include "tasks/gkd/gkd_common.hpp"

struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;
struct ggml_tensor;
struct ggml_cgraph;
struct ggml_gallocr;
typedef struct ggml_gallocr* ggml_gallocr_t;
struct ggml_backend_sched;
typedef struct ggml_backend_sched* ggml_backend_sched_t;
struct ggml_backend_buffer_type;
typedef struct ggml_backend_buffer_type* ggml_backend_buffer_type_t;

namespace gkd {

/* Compute-side backend bundle, owned by the GKD session. Carries the CPU
 * backend plus, when a GPU is available, a scheduler spanning [gpu, cpu]
 * with op offload so ops the GPU backend cannot run fall back to CPU. All
 * ggml_backend_t handles point into process-shared BackendLease state. */
struct BackendCtx {
    aicore::runtime::BackendLease cpu_lease;
    aicore::runtime::BackendLease gpu_lease; /* optional */

    ggml_backend_t cpu = nullptr; /* lease handle; never null after init */
    ggml_backend_t gpu = nullptr; /* lease handle; null on CPU-only */
    int n_threads = 1;

    /* Resolved device display name ("Vulkan0 (NVIDIA ...)" / "CUDA0" / "cpu"). */
    std::string device_name;

    /* Persistent graph allocator for the CPU-only path. */
    ggml_gallocr_t galloc = nullptr;

    /* Scheduler spanning [gpu, cpu] when gpu != nullptr. */
    ggml_backend_sched_t sched = nullptr;
};

/* Initialize the compute backend bundle for device_request
 * ("auto" | "cpu" | "cuda" | "vulkan" | ...). On failure returns an empty
 * BackendCtx (cpu == nullptr). */
BackendCtx init_backend_ctx(int n_threads, const std::string& device_request);

/* Release a BackendCtx. Safe to call on a zero-initialized struct. */
void free_backend_ctx(BackendCtx& ctx);

/* Buffer type that model weights should be realized on (GPU buffer when a
 * GPU is active so weights live in VRAM, else the host buffer). */
ggml_backend_buffer_type_t backend_weight_buft(const BackendCtx& ctx);

/* Allocate buffers for the graph: sched path on GPU builds (with the
 * external leaves pinned to the GPU), persistent gallocr on CPU-only.
 * The sched is reset internally, so pinning must happen here. Returns false
 * on allocation failure. */
bool backend_graph_alloc(BackendCtx& ctx,
                         ::ggml_cgraph* graph,
                         ::ggml_tensor* pin_input,
                         ::ggml_tensor* pin_output);

/* Run the graph on the bundle. Honors the AICore cancel token (returns
 * GGML_STATUS_ABORTED without launching compute when cancellation was
 * requested). Returns the ggml_status of the compute otherwise. */
int /* ggml_status */ backend_graph_compute(BackendCtx& ctx,
                                            ::ggml_cgraph* graph);

/* Resolved backend display name. */
const char* backend_name(const BackendCtx& ctx);

}  // namespace gkd
