#pragma once


// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: AGPL-3.0
// ----------------------------------------------------------------------------
//
// AICore adaptation of ultralytics-ggml's backend bundle. The upstream
// YOLO_USE_CUDA / YOLO_USE_VULKAN compile-time dispatch is replaced by the
// AICore runtime device resolution (aicore::runtime::BackendLease +
// ggml_common): "auto" follows the platform order (CUDA -> Vulkan -> CPU on
// Linux/Windows, Metal -> CPU on macOS) and every physical backend handle is
// shared process-wide through the lease registry. The op graph is built once
// from the generic ggml op vocabulary; the scheduler spanning [gpu, cpu]
// routes each op to whatever the active GPU backend supports (im2col /
// mul_mat / interpolate / conv_transpose are standard ops), with CPU
// fallback for the rest.

#include <cstdint>
#include <string>

#include "common/ggml_backend_registry.hpp"

#include "tasks/yolo/yolo_common.hpp"


struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;
struct ggml_threadpool;
typedef struct ggml_threadpool* ggml_threadpool_t;
struct ggml_cgraph;
struct ggml_gallocr;
typedef struct ggml_gallocr* ggml_gallocr_t;
struct ggml_backend_sched;
typedef struct ggml_backend_sched* ggml_backend_sched_t;
struct ggml_backend_buffer_type;
typedef struct ggml_backend_buffer_type* ggml_backend_buffer_type_t;

namespace yolo {

/* Compute-side backend bundle, owned by the yolo session. Carries the CPU
 * backend (tensor I/O, weight realization, graph compute) plus, when a GPU
 * is available, a scheduler spanning [gpu, cpu] with op offload so ops the
 * GPU backend cannot run fall back to CPU automatically.
 *
 * All ggml_backend_t handles point into process-shared BackendLease state;
 * free_backend_ctx() releases the leases in the correct order. */
struct BackendCtx {
    aicore::runtime::BackendLease cpu_lease;
    aicore::runtime::BackendLease gpu_lease; /* optional (GPU builds only) */

    ggml_backend_t cpu = nullptr; /* lease handle; never null after init */
    ggml_backend_t gpu = nullptr; /* lease handle; null on CPU-only builds */
    int n_threads = 1;

    /* Resolved backend FAMILY of the active GPU handle, detected at runtime
     * from the lease's device name ("CUDA0" / "Vulkan0 (…)" / "Metal").
     * This replaces the upstream YOLO_USE_CUDA / YOLO_USE_VULKAN
     * compile-time dispatch: in a dynamic-backend build the actual device
     * is only known after the lease resolves, so per-backend data-flow
     * choices (f16 casts, q8 direct conv) must key off these flags, never
     * off compile-time macros. Both stay false for CPU / Metal sessions. */
    bool is_cuda = false;
    bool is_vulkan = false;

    /* Persistent graph allocator for the CPU-only path (keeps the compute
     * scratch buffer alive across inferences; the sched owns allocation on
     * GPU builds). Lazily created on first use, freed in free_backend_ctx. */
    ggml_gallocr_t galloc = nullptr;

    /* Scheduler spanning [gpu, cpu] when gpu != nullptr. */
    ggml_backend_sched_t sched = nullptr;

    /* Resolved device display name (e.g. "Vulkan0 (NVIDIA GeForce ...)" or
     * "CPU"). */
    std::string device_name;
};

/* Initialize the compute backend bundle. Creates (or leases) a CPU backend
 * and optionally a GPU backend + scheduler for p device_request
 * ("auto" | "cpu" | "cuda" | "vulkan" | "metal" | ...).
 *
 * On failure returns an empty BackendCtx (all members nullptr). */
BackendCtx init_backend_ctx(int n_threads, const std::string& device_request);

/* Install the per-op profiling callback on the scheduler (GPU builds).
 * Every node is then dispatched and synced individually: the printed table
 * is useful for relative shares, not absolute latency. */
void backend_enable_op_profile(BackendCtx& ctx);

/* Dump the collected per-op profile (total_ms / calls / avg_us) to stderr.
 * Call from free_session() when profiling was enabled. */
void backend_print_op_profile();

/* Release a BackendCtx. Safe to call on a zero-initialized struct. */
void free_backend_ctx(BackendCtx& ctx);

/* Buffer type that model weights should be realized on. Returns the GPU
 * backend's default buffer type when a GPU is active (so weights live in
 * VRAM), otherwise the CPU host buffer type. Never returns null on a
 * successfully-initialized BackendCtx. */
ggml_backend_buffer_type_t backend_ctx_weight_buft(const BackendCtx& ctx);

/* Allocate buffers for the (single) compute graph. Uses the sched when
 * active (GPU), else the persistent gallocr. Returns false on allocation
 * failure. Call this, then ggml_backend_tensor_set() the graph inputs,
 * then backend_ctx_graph_compute(). */
/* Allocate buffers for the (single) compute graph. Uses the sched when
 * active (GPU), else the persistent gallocr. Returns false on allocation
 * failure. Call this, then ggml_backend_tensor_set() the graph inputs,
 * then backend_ctx_graph_compute().
 *
 * When the scheduler is active, pin_input/pin_output (if non-null) are
 * forced to live on the GPU backend so upload and readback don't bounce
 * through CPU host memory. These MUST be set AFTER reset (i.e. right
 * before alloc_graph) or the scheduler clears them. */
bool backend_ctx_graph_alloc(BackendCtx& ctx, ::ggml_cgraph* graph,
                             ::ggml_tensor* pin_input = nullptr,
                             ::ggml_tensor* pin_output = nullptr);

/* Run the graph on the bundle. Honors the AICore cancel token: returns
 * GGML_STATUS_ABORTED without launching compute when cancellation was
 * requested. Returns the ggml_status of the compute otherwise. */
int /* ggml_status */ backend_ctx_graph_compute(BackendCtx& ctx,
                                                ::ggml_cgraph* graph);

/* Resolved backend display name ("Vulkan0 (NVIDIA...)" or "cpu"). */
const char* backend_name(const BackendCtx& ctx);

}  // namespace yolo
