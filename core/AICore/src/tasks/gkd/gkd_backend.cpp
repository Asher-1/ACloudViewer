// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// GKDT ggml runtime - backend management over the shared AICore lease
// registry (template: tasks/yolo/backend.cpp).

#include "tasks/gkd/gkd_backend.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <thread>
#include <vector>

#include "aicore/runtime_capi.h"
#include "common/ggml_backend_utils.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

namespace gkd {

static constexpr size_t kGraphSize = 4096;

static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

BackendCtx init_backend_ctx(int n_threads, const std::string& device_request) {
    BackendCtx ctx{};
    // threads <= 0 means the backend default: the upstream GKDT runtime
    // sizes the CPU threadpool from hardware_concurrency() (the graphs are
    // large ViT transformer blocks, so single-thread CPU inference is ~Nx
    // slower than the upstream reference). GPU leases ignore the count.
    ctx.n_threads = (n_threads > 0)
                            ? n_threads
                            : std::max(1u, std::thread::hardware_concurrency());

    ggml_common::load_backends_once();

    // Resolve the requested device through the AICore runtime (shared
    // process-wide backends). "auto" follows the platform order
    // (CUDA -> Vulkan -> CPU on Linux/Windows, Metal -> CPU on macOS).
    const std::string want =
            to_lower(device_request.empty() ? "auto" : device_request);
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
                GKD_LOG_INFO("GPU backend: %s", ctx.device_name.c_str());
            } else {
                GKD_LOG_WARN("failed to acquire GPU backend lease; using CPU");
            }
        } else if (want != "auto") {
            GKD_LOG_WARN("requested device %s not found; falling back to CPU",
                         want.c_str());
        }
    }

    // CPU backend: needed both as the fallback and as the sched's CPU half.
    std::string cpu_error;
    ctx.cpu_lease = aicore::runtime::acquire_backend_lease("cpu", ctx.n_threads,
                                                           &cpu_error);
    ctx.cpu = ctx.cpu_lease.handle();
    if (!ctx.cpu) {
        GKD_LOG_ERROR("CPU backend init failed: %s", cpu_error.c_str());
        free_backend_ctx(ctx);
        return ctx;
    }
    if (ctx.device_name.empty()) {
        ctx.device_name = ctx.cpu_lease.device();
    }

    // Scheduler spanning [gpu, cpu] so ops the GPU can't run fall back to
    // CPU automatically (op offload). Every op the GKD graphs use (mul_mat /
    // flash_attn_ext / interpolate / get_rows / concat / ...) is covered by
    // the GPU backends; only exotic combos degrade to CPU.
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
            GKD_LOG_WARN(
                    "ggml_backend_sched_new failed; falling back to CPU-only");
            ctx.gpu_lease.reset();
            ctx.gpu = nullptr;
        }
    }

    return ctx;
}

void free_backend_ctx(BackendCtx& ctx) {
    // Free the gallocr BEFORE the backends (it owns compute scratch allocated
    // through the backend's buffer type; matches the construction order).
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

ggml_backend_buffer_type_t backend_weight_buft(const BackendCtx& ctx) {
    if (ctx.gpu) {
        return ggml_backend_get_default_buffer_type(ctx.gpu);
    }
    return ggml_backend_get_default_buffer_type(ctx.cpu);
}

bool backend_graph_alloc(BackendCtx& ctx,
                         ::ggml_cgraph* graph,
                         ::ggml_tensor* pin_input,
                         ::ggml_tensor* pin_output) {
    if (ctx.sched) {
        // Pre-flight: every node must be claimed by the GPU or the CPU half
        // of the scheduler. A node neither backend supports would trip a
        // GGML_ASSERT inside ggml_backend_sched_alloc_graph instead of
        // returning an error; drop the GPU lease once and fall back to the
        // CPU-only gallocr path in that case.
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
            GKD_LOG_WARN(
                    "graph contains ops neither GPU nor CPU support; falling "
                    "back to CPU-only inference");
            ggml_backend_sched_free(ctx.sched);
            ctx.sched = nullptr;
            ctx.gpu_lease.reset();
            ctx.gpu = nullptr;
        } else {
            ggml_backend_sched_reset(ctx.sched);
            // Pin the external leaves AFTER reset, before alloc_graph (the
            // reset clears every tensor->backend assignment).
            if (ctx.gpu && pin_input) {
                ggml_backend_sched_set_tensor_backend(ctx.sched, pin_input,
                                                      ctx.gpu);
            }
            if (ctx.gpu && pin_output) {
                ggml_backend_sched_set_tensor_backend(ctx.sched, pin_output,
                                                      ctx.gpu);
            }
            if (!ggml_backend_sched_alloc_graph(ctx.sched, graph)) {
                GKD_LOG_ERROR("backend_graph_alloc: sched alloc failed");
                return false;
            }
            return true;
        }
    }
    // CPU path: persistent gallocr reused across inferences of the same
    // graph shape.
    if (!ctx.galloc) {
        ctx.galloc =
                ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx.cpu));
        if (!ctx.galloc) {
            GKD_LOG_ERROR("backend_graph_alloc: gallocr_new failed");
            return false;
        }
    }
    if (!ggml_gallocr_alloc_graph(ctx.galloc, graph)) {
        GKD_LOG_ERROR("backend_graph_alloc: gallocr_alloc_graph failed");
        return false;
    }
    return true;
}

int backend_graph_compute(BackendCtx& ctx, ::ggml_cgraph* graph) {
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

const char* backend_name(const BackendCtx& ctx) {
    return ctx.device_name.empty() ? "cpu" : ctx.device_name.c_str();
}

}  // namespace gkd
