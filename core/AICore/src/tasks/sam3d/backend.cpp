// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "backend.hpp"

#include <cstdint>
#include <cstring>

#include "common.hpp"
#include "common/ggml_backend_registry.hpp"
#include "common/ggml_backend_utils.hpp"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

namespace sam3d {

std::unique_ptr<Backend> Backend::create(const std::string& backend_name,
                                         int n_threads) {
    auto b = std::unique_ptr<Backend>(new Backend());
    if (!b->init(backend_name, n_threads)) return nullptr;
    return b;
}

bool Backend::init(const std::string& backend_name, int n_threads) {
    n_threads_ = n_threads;
    backend_name_ = "cpu";

    // Register the dynamic ggml backends (libggml-cpu-<isa>.so / CUDA /
    // Vulkan) from the library directory before querying the registry.
    ggml_common::load_backends_once();

    // Physical handles are process-shared leases. The CPU fallback is a
    // private instance (acquire_parallel_backend_lease) so two sam3d sessions
    // can at least own independent host graphs; GPU command queues stay in
    // the shared registry and are serialized by the caller-held lease lock.
    std::string error;
    aicore::runtime::BackendLease cpu_lease =
            aicore::runtime::acquire_parallel_backend_lease("cpu", n_threads,
                                                            &error);
    if (!cpu_lease || !cpu_lease.handle()) {
        LOGE("sam3d: failed to acquire CPU backend lease: %s", error.c_str());
        return false;
    }
    cpu_backend_ = cpu_lease.handle();

    if (backend_name != "cpu") {
        const std::string want =
                backend_name == "auto" ? std::string("gpu") : backend_name;
        std::string family = want;
        int want_index = 0;
        ggml_common::parse_device(want, family, want_index);
        if (family.empty() || family == "auto") family = "gpu";
        std::string resolved;
        ggml_backend_t gpu =
                ggml_common::find_gpu_backend(family, want_index, resolved);
        if (gpu) {
            aicore::runtime::BackendLease gpu_lease =
                    aicore::runtime::adopt_backend_lease(gpu, resolved, 0);
            if (gpu_lease && gpu_lease.handle()) {
                gpu_backend_ = gpu_lease.handle();
                backend_name_ = resolved;
                leases_.push_back(std::move(gpu_lease));
            } else {
                LOGE("sam3d: failed to adopt GPU backend lease for '%s'",
                     resolved.c_str());
            }
        } else if (backend_name != "auto") {
            LOGE("sam3d: requested backend '%s' not available",
                 backend_name.c_str());
            return false;
        } else {
            LOGI("sam3d: no GPU backend registered in this build, using CPU");
        }
    }
    leases_.push_back(std::move(cpu_lease));

    // Primary path: a single compute backend (GPU when present) with a
    // gallocr, since every weight lives in that same buffer type. The source
    // repo's optional backend-scheduler experiment (SAM3D_USE_SCHED) was an
    // environment-variable opt-in and is not carried over: the current ggml
    // release fails its auto-realloc path on the very large conv3d graphs of
    // the SS decoder.
    compute_backend_ = gpu_backend_ ? gpu_backend_ : cpu_backend_;
    gallocr_ = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(compute_backend_));
    if (!gallocr_) {
        LOGE("sam3d: failed to create graph allocator");
        return false;
    }
    LOGI("sam3d backend: %s (%s)", backend_name_.c_str(), device_name());
    return true;
}

Backend::~Backend() {
    if (gallocr_) ggml_gallocr_free(gallocr_);
    // The physical handles belong to the runtime registry leases; reset the
    // lease objects and never ggml_backend_free them here.
    for (auto& lease : leases_) {
        lease.reset();
    }
    gpu_backend_ = nullptr;
    cpu_backend_ = nullptr;
    compute_backend_ = nullptr;
}

void Backend::set_profile_label(std::string label) {
    // The source repo used this label for an environment-controlled JSONL
    // trace; AICore reports truthful stage timings through the common
    // aicore_pipeline_timings contract instead. Kept as a no-op so the
    // ported session code stays byte-identical at its call sites.
    (void)label;
}

ggml_backend_buffer_type_t Backend::weights_buffer_type() const {
    return ggml_backend_get_default_buffer_type(compute_backend_);
}

const char* Backend::device_name() const {
    if (gpu_backend_) {
        const std::string& resolved =
                leases_.empty() ? backend_name_ : leases_.front().device();
        return resolved.c_str();
    }
    return "CPU";
}

bool Backend::alloc(ggml_cgraph* graph, std::vector<ggml_tensor*> inputs) {
    (void)inputs;
    // The AICore ggml patch gates the SAM 3D fast paths (cutlass dense FMHA,
    // cublasLt F16 linear fusions, PyTorch-Welford NORM) on the "sam3d_"
    // node-name prefix so no other task's validated numerics change. The
    // ported graph builders leave nodes with the automatic "node_N" names,
    // so stamp the whole graph here — every sam3d graph passes through this
    // allocator exactly once, before any compute scans the node names.
    const int n_nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_format_name(ggml_graph_node(graph, i), "sam3d_%d", i);
    }
    if (ggml_gallocr_reserve(gallocr_, graph) == false) {
        LOGE("sam3d: graph reserve failed (graph too large for memory?)");
        return false;
    }
    if (!ggml_gallocr_alloc_graph(gallocr_, graph)) {
        LOGE("sam3d: graph alloc failed");
        return false;
    }
    return true;
}

bool Backend::run(ggml_cgraph* graph) {
    ggml_backend_t be = compute_backend_;
    const ggml_status status = ggml_backend_graph_compute_async(be, graph);
    if (status != GGML_STATUS_SUCCESS) {
        LOGE("sam3d: graph compute failed");
        return false;
    }
    ggml_backend_synchronize(be);
    return true;
}

bool Backend::compute(ggml_cgraph* graph, std::vector<ggml_tensor*> inputs) {
    if (!alloc(graph, inputs)) return false;
    return run(graph);
}

bool Backend::set_input_f32(ggml_tensor* t,
                            const float* data,
                            size_t n_floats) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    if ((size_t)ggml_nelements(t) != n_floats) {
        LOGE("set_input_f32: expected %lld floats, got %zu",
             (long long)ggml_nelements(t), n_floats);
        return false;
    }
    ggml_backend_tensor_set(t, data, 0, n_floats * sizeof(float));
    return true;
}

bool Backend::set_input_i32(ggml_tensor* t,
                            const int32_t* data,
                            size_t n_ints) {
    GGML_ASSERT(t->type == GGML_TYPE_I32);
    if ((size_t)ggml_nelements(t) != n_ints) {
        LOGE("set_input_i32: expected %lld ints, got %zu",
             (long long)ggml_nelements(t), n_ints);
        return false;
    }
    ggml_backend_tensor_set(t, data, 0, n_ints * sizeof(int32_t));
    return true;
}

bool Backend::get_tensor_f32(ggml_tensor* t, std::vector<float>& out) {
    const size_t n = ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
        return true;
    }
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> f16(n);
        ggml_backend_tensor_get(t, f16.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(f16[i]);
        return true;
    }
    LOGE("get_tensor_f32: unsupported tensor type %s", ggml_type_name(t->type));
    out.clear();
    return false;
}

bool Backend::get_tensor_i32(ggml_tensor* t, std::vector<int32_t>& out) {
    GGML_ASSERT(t->type == GGML_TYPE_I32);
    const size_t n = ggml_nelements(t);
    out.resize(n);
    ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(int32_t));
    return true;
}

}  // namespace sam3d
