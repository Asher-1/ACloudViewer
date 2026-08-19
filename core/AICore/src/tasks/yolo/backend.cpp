// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "backend.hpp"

#include <cstring>
#include <string>
#include <vector>

#include "aicore/runtime_capi.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ggml_backend_utils.hpp"

namespace yolo {

static constexpr size_t kGraphSize = 16384;

ggml_backend_buffer_type_t backend_ctx_weight_buft(const BackendCtx& ctx) {
    if (ctx.gpu) {
        return ggml_backend_get_default_buffer_type(ctx.gpu);
    }
    return ggml_backend_get_default_buffer_type(ctx.cpu);
}

BackendCtx init_backend_ctx(int n_threads, const std::string& device_request) {
    BackendCtx ctx{};
    ctx.n_threads = (n_threads > 0) ? n_threads : 1;

    ggml_common::load_backends_once();

    // Resolve the requested device through the AICore runtime (shared
    // process-wide backends). "auto" follows the platform order
    // (CUDA -> Vulkan -> CPU on Linux/Windows, Metal -> CPU on macOS).
    const std::string want = ggml_common::to_lower(
            device_request.empty() ? "auto" : device_request);
    const bool force_cpu = want == "cpu";

    if (!force_cpu) {
        ggml_common::GpuBackendGroup group =
                ggml_common::resolve_gpu_group(want);
        if (group.primary()) {
            ctx.gpu_lease = aicore::runtime::adopt_backend_lease(
                    group.gpus[0], group.names[0], ctx.n_threads);
            group.gpus[0] = nullptr;
            group.release();
            if (ctx.gpu_lease) {
                ctx.gpu = ctx.gpu_lease.handle();
                ctx.device_name = ctx.gpu_lease.device();
                YOLO_LOG_INFO("GPU backend: %s", ctx.device_name.c_str());
            } else {
                YOLO_LOG_WARN("failed to acquire GPU backend lease; using CPU");
            }
        } else if (want != "auto") {
            YOLO_LOG_WARN("requested device %s not found; falling back to CPU",
                          want.c_str());
        }
    }

    // CPU backend: needed both as the fallback and as the sched's CPU half.
    std::string cpu_error;
    ctx.cpu_lease = aicore::runtime::acquire_backend_lease("cpu", ctx.n_threads,
                                                           &cpu_error);
    ctx.cpu = ctx.cpu_lease.handle();
    if (!ctx.cpu) {
        YOLO_LOG_ERROR("CPU backend init failed: %s", cpu_error.c_str());
        free_backend_ctx(ctx);
        return ctx;
    }
    if (ctx.device_name.empty()) {
        ctx.device_name = ctx.cpu_lease.device();
    }

    /* When a GPU is present, build a scheduler spanning [gpu, cpu] so ops the
     * GPU can't run fall back to CPU automatically (op offload). All standard
     * yolo ops (im2col / mul_mat / interpolate / conv_transpose / …) are
     * covered by the Vulkan/Metal/CUDA backends; only exotic combos degrade
     * to CPU. */
    if (ctx.gpu) {
        std::vector<ggml_backend_t> backends = {ctx.gpu, ctx.cpu};
        std::vector<ggml_backend_buffer_type_t> bufts = {
                ggml_backend_get_default_buffer_type(ctx.gpu),
                ggml_backend_get_default_buffer_type(ctx.cpu),
        };
        ctx.sched = ggml_backend_sched_new(
                backends.data(), bufts.data(), (int)backends.size(),
                /*graph_size*/ kGraphSize, /*parallel*/ false,
                /*op_offload*/ true);
        if (!ctx.sched) {
            YOLO_LOG_WARN(
                    "ggml_backend_sched_new failed; falling back to CPU-only");
            ctx.gpu_lease.reset();
            ctx.gpu = nullptr;
        }
    }

    return ctx;
}

void free_backend_ctx(BackendCtx& ctx) {
    /* Free the gallocr BEFORE the backends. The gallocr owns the compute
     * scratch buffer (allocated via the backend's buffer_type); freeing it
     * first matches the construction order. */
    if (ctx.galloc) {
        ggml_gallocr_free(ctx.galloc);
        ctx.galloc = nullptr;
    }
    if (ctx.sched) {
        ggml_backend_sched_free(ctx.sched);
        ctx.sched = nullptr;
    }
    ctx.gpu = nullptr;
    ctx.cpu = nullptr;
    ctx.gpu_lease.reset();
    ctx.cpu_lease.reset();
    ctx.device_name.clear();
}

bool backend_ctx_graph_alloc(BackendCtx& ctx,
                             ::ggml_cgraph* graph,
                             ::ggml_tensor* pin_input,
                             ::ggml_tensor* pin_output) {
    if (ctx.sched) {
        /* Pre-flight: every node must be claimed by the GPU or the CPU half
         * of the scheduler. A node neither backend supports would leave the
         * gallocr without any buffer candidate and trip GGML_ASSERT
         * (buffer_id >= 0) in ggml-alloc.c instead of returning an error,
         * aborting the process from inside ggml_backend_sched_alloc_graph.
         * When that happens, drop the GPU lease once and fall back to the
         * CPU-only gallocr path below. */
        bool gpu_covers_all = true;
        const int n_nodes = ggml_graph_n_nodes(graph);
        for (int i = 0; i < n_nodes; i++) {
            struct ggml_tensor* node = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(ctx.gpu, node) &&
                !ggml_backend_supports_op(ctx.cpu, node)) {
                gpu_covers_all = false;
                break;
            }
        }
        if (!gpu_covers_all) {
            YOLO_LOG_WARN(
                    "graph contains ops neither GPU nor CPU support; "
                    "falling back to CPU-only inference");
            ggml_backend_sched_free(ctx.sched);
            ctx.sched = nullptr;
            ctx.gpu_lease.reset();
            ctx.gpu = nullptr;
        } else {
            ggml_backend_sched_reset(ctx.sched);
            // Pin input/output AFTER reset, before alloc_graph.
            // ggml_backend_sched_reset clears all hv_tensor_backend_ids to
            // -1, so any ggml_backend_sched_set_tensor_backend calls made
            // before reset are lost.  The upstream llama.cpp pattern is:
            //   reset → set_tensor_backend → alloc_graph
            if (ctx.gpu && pin_input) {
                ggml_backend_sched_set_tensor_backend(ctx.sched, pin_input,
                                                      ctx.gpu);
            }
            if (ctx.gpu && pin_output) {
                ggml_backend_sched_set_tensor_backend(ctx.sched, pin_output,
                                                      ctx.gpu);
            }
            if (!ggml_backend_sched_alloc_graph(ctx.sched, graph)) {
                YOLO_LOG_ERROR("backend_ctx_graph_alloc: sched alloc failed");
                return false;
            }
            return true;
        }
    }
    /* CPU path: persistent gallocr, reused across inferences of the same
     * graph shape. The gallocr reserves enough address space for the largest
     * graph it has seen, so rebuilds for a larger canvas may re-allocate. */
    if (!ctx.galloc) {
        ctx.galloc =
                ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx.cpu));
        if (!ctx.galloc) {
            YOLO_LOG_ERROR("backend_ctx_graph_alloc: gallocr_new failed");
            return false;
        }
    }
    if (!ggml_gallocr_alloc_graph(ctx.galloc, graph)) {
        YOLO_LOG_ERROR("backend_ctx_graph_alloc: gallocr_alloc_graph failed");
        return false;
    }
    return true;
}

int backend_ctx_graph_compute(BackendCtx& ctx, ::ggml_cgraph* graph) {
    if (aicore_cancel_requested()) {
        return (int)GGML_STATUS_ABORTED;
    }
    if (ctx.sched) {
        ggml_status st = ggml_backend_sched_graph_compute(ctx.sched, graph);
        ggml_backend_sched_synchronize(ctx.sched);
        return (int)st;
    }
    ggml_status st = ggml_backend_graph_compute(ctx.cpu, graph);
    ggml_backend_synchronize(ctx.cpu);
    return (int)st;
}

}  // namespace yolo
