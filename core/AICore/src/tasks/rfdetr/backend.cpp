// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/rfdetr/backend.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/runtime_capi.h"
#include "common/ggml_backend_utils.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "tasks/rfdetr/common.hpp"

namespace rfdetr {

static constexpr size_t kGraphSize = 16384;

ggml_backend_buffer_type_t backend_ctx_weight_buft(const BackendCtx& ctx) {
    if (ctx.gpu) {
        return ggml_backend_get_default_buffer_type(ctx.gpu);
    }
    return ggml_backend_get_default_buffer_type(ctx.cpu);
}

BackendCtx init_backend_ctx(int n_threads,
                            const std::string& device_request,
                            rfdetr_status* out_status) {
    auto set = [&](rfdetr_status s) {
        if (out_status) *out_status = s;
    };

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
                rfdetr_logf(RFDETR_LOG_INFO, "GPU backend: %s",
                            ctx.device_name.c_str());
            } else {
                rfdetr_logf(RFDETR_LOG_WARN,
                            "failed to acquire GPU backend lease; using CPU");
            }
        } else if (want != "auto") {
            rfdetr_logf(RFDETR_LOG_WARN,
                        "requested device %s not found; falling back to CPU",
                        want.c_str());
        }
    }

    // CPU backend: needed both as the fallback and as the sched's CPU half.
    std::string cpu_error;
    ctx.cpu_lease = aicore::runtime::acquire_backend_lease("cpu", ctx.n_threads,
                                                           &cpu_error);
    ctx.cpu = ctx.cpu_lease.handle();
    if (!ctx.cpu) {
        rfdetr_logf(RFDETR_LOG_ERROR, "CPU backend init failed: %s",
                    cpu_error.c_str());
        set(RFDETR_ERR_INFERENCE);
        free_backend_ctx(ctx);
        return ctx;
    }
    if (ctx.device_name.empty()) {
        ctx.device_name = ctx.cpu_lease.device();
    }

    /* When a GPU is present, build a scheduler spanning [gpu, cpu] so ops the
     * GPU can't run (the deformable ggml_custom_4d) fall back to CPU
     * automatically. */
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
            rfdetr_logf(RFDETR_LOG_WARN,
                        "ggml_backend_sched_new failed; falling back to "
                        "CPU-only");
            ctx.gpu_lease.reset();
            ctx.gpu = nullptr;
        }
    }

    set(RFDETR_OK);
    return ctx;
}

void free_backend_ctx(BackendCtx& ctx) {
    /* Free the gallocrs BEFORE the backends. The gallocr owns the compute
     * scratch buffers (allocated via the backend's buffer_type); freeing
     * it after the backend would still be safe (the buffer keeps a ref to
     * its buffer_type), but doing it first matches the construction order
     * (gallocrs created lazily during forward, on top of the backend). */
    if (ctx.galloc_a) {
        ggml_gallocr_free(ctx.galloc_a);
        ctx.galloc_a = nullptr;
    }
    if (ctx.galloc_b) {
        ggml_gallocr_free(ctx.galloc_b);
        ctx.galloc_b = nullptr;
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
                             int which_graph) {
    if (ctx.sched) {
        ggml_backend_sched_reset(ctx.sched);
        if (!ggml_backend_sched_alloc_graph(ctx.sched, graph)) {
            rfdetr_logf(RFDETR_LOG_ERROR,
                        "backend_ctx_graph_alloc: sched alloc failed");
            return false;
        }
        return true;
    }
    /* CPU path: persistent gallocr per graph. */
    ggml_gallocr_t* slot = (which_graph == 0) ? &ctx.galloc_a : &ctx.galloc_b;
    if (!*slot) {
        *slot = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx.cpu));
        if (!*slot) {
            rfdetr_logf(RFDETR_LOG_ERROR,
                        "backend_ctx_graph_alloc: gallocr_new failed");
            return false;
        }
    }
    if (!ggml_gallocr_alloc_graph(*slot, graph)) {
        rfdetr_logf(RFDETR_LOG_ERROR,
                    "backend_ctx_graph_alloc: gallocr_alloc_graph failed");
        return false;
    }
    return true;
}

int backend_ctx_graph_compute(BackendCtx& ctx,
                              ::ggml_cgraph* graph,
                              int which_graph) {
    (void)which_graph;
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

}  // namespace rfdetr
