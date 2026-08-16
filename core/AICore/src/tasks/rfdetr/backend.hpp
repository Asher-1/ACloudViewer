#ifndef RFDETR_BACKEND_HPP
#define RFDETR_BACKEND_HPP

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// AICore adaptation of rf-detr.cpp's backend bundle. The upstream
// RFDETR_USE_CUDA/_METAL/_VULKAN compile-time dispatch is replaced by the
// AICore runtime device resolution (aicore::runtime::BackendLease +
// ggml_common): "auto" follows the platform order (CUDA -> Vulkan -> CPU on
// Linux/Windows, Metal -> CPU on macOS) and every physical backend handle is
// shared process-wide through the lease registry.

#include "rfdetr.h"

#include <cstdint>
#include <string>

#include "ggml_backend_registry.hpp"

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

namespace rfdetr {

/* Compute-side backend bundle, owned by rfdetr_context. Carries the CPU
 * backend (used for tensor I/O, weight realization, and graph compute) plus
 * the per-graph gallocrs and, when a GPU is available, a scheduler spanning
 * [gpu, cpu] so ops the GPU cannot run (the deformable-attention
 * ggml_custom_4d sampler) fall back to CPU automatically.
 *
 * All ggml_backend_t handles point into process-shared BackendLease state;
 * free_backend_ctx() releases the leases in the correct order. */
struct BackendCtx {
    aicore::runtime::BackendLease cpu_lease;
    aicore::runtime::BackendLease gpu_lease; /* optional (GPU builds only) */

    ggml_backend_t cpu = nullptr; /* lease handle; never null after init */
    ggml_backend_t gpu = nullptr; /* lease handle; null on CPU-only builds */
    int n_threads = 1;

    /* Persistent graph allocators for the two compute graphs in
     * rfdetr_model_forward (graph A = backbone+projector+two_stage,
     * graph B = decoder+heads).
     *
     * Without this, every inference allocates ~1.9 GB of scratch space for
     * every intermediate tensor and then frees it — the free alone costs
     * ~55 ms/inference (glibc munmaps the large mmap). gallocr packs
     * intermediate tensors compactly (lifetime-aware reuse) AND keeps the
     * underlying buffer alive across calls, so the kernel doesn't see
     * mmap/munmap traffic on the steady-state loop.
     *
     * Lazily created on first use, freed in free_backend_ctx. */
    ggml_gallocr_t galloc_a = nullptr;
    ggml_gallocr_t galloc_b = nullptr;

    /* Scheduler spanning [gpu, cpu] when gpu != nullptr. Routes ops to the
     * GPU and falls back to CPU for ops the GPU backend can't run. When
     * gpu == nullptr this stays null and we use the plain CPU compute path. */
    ggml_backend_sched_t sched = nullptr;

    /* Resolved device display name (e.g. "Vulkan0 (NVIDIA GeForce ...)" or
     * "CPU"). */
    std::string device_name;
};

/* Initialize the compute backend bundle. Creates (or leases) a CPU backend
 * and optionally a GPU backend + scheduler for \p device_request
 * ("auto" | "cpu" | "cuda" | "vulkan" | "metal" | ...).
 *
 * On failure returns an empty BackendCtx (all members nullptr) and writes
 * the error to *out_status. */
BackendCtx init_backend_ctx(int n_threads,
                            const std::string& device_request,
                            rfdetr_status* out_status);

/* Release a BackendCtx. Safe to call on a zero-initialized struct. */
void free_backend_ctx(BackendCtx& ctx);

/* Buffer type that model weights should be realized on. Returns the GPU
 * backend's default buffer type when a GPU is active (so weights live in
 * VRAM), otherwise the CPU host buffer type. Never returns null on a
 * successfully-initialized BackendCtx. */
ggml_backend_buffer_type_t backend_ctx_weight_buft(const BackendCtx& ctx);

/* Allocate buffers for a graph. Uses the sched when active (GPU), else the
 * persistent per-graph gallocr (which_graph: 0 = A, 1 = B). Returns false on
 * allocation failure. Call this, then ggml_backend_tensor_set() the graph
 * inputs, then backend_ctx_graph_compute(). */
bool backend_ctx_graph_alloc(BackendCtx& ctx, ::ggml_cgraph* graph,
                             int which_graph);

/* Allocate + run a graph on the bundle. When a GPU + sched are present the
 * graph is allocated and computed via ggml_backend_sched (which places ops
 * across GPU/CPU and inserts cross-device copies as needed). On CPU-only
 * bundles it falls back to the persistent-gallocr + cpu-backend path.
 *
 * `which_graph` selects the persistent allocator slot on CPU-only builds
 * (0 = graph A, 1 = graph B). Ignored when the sched is active (the sched
 * owns allocation). */
int /* ggml_status */ backend_ctx_graph_compute(BackendCtx& ctx,
                                                ::ggml_cgraph* graph,
                                                int which_graph);

}  // namespace rfdetr

#endif
